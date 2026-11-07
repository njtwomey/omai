/**
 * Exponential integrators for stiff semilinear problems y' = L y + N(t, y), which integrate the linear part exactly
 * through matrix or diagonal exponentials. The phi functions evaluated without cancellation (scalar and matrix forms),
 * exponential Euler, and ETDRK4 (Cox and Matthews).
 */

import { MdmlError } from '../../base/index.ts'
import { expm } from '../../numerics/linalg/index.ts'
import { add, matmul, mul, primalOf, reshape, type Tensor, type TensorLike, type Value } from '../../tensor/index.ts'
import {
  atBound,
  attempt,
  type Counter,
  checked,
  counter,
  initialState,
  snapshot,
  statusAfter,
  tally,
  towards,
} from './_step.ts'
import { allFinite, combine, floatState, fromNumbers, numbersOf } from './_vec.ts'
import { hermiteSegment, type OdeVectorField, type Stepper, type StepperState } from './problem.ts'

/**
 * A semilinear problem y' = L y + N(t, y): a stiff linear part (a diagonal, given with the state's shape, as for
 * Fourier spectral methods, or a matrix `[n, n]` over the flattened state) and a nonstiff nonlinear part.
 */
export interface SemilinearProblem {
  readonly linear: TensorLike
  readonly nonlinear: OdeVectorField
}

/**
 * The full vector field L y + N(t, y) of a semilinear problem (to pass to other solvers, or for dense output).
 *
 * @param problem - The problem.
 * @returns The field.
 * @example
 * ```ts
 * import { semilinearField } from 'mdml-engine/dynamics/ode'
 * import { tensor, toArray, zerosLike } from 'mdml-engine/tensor'
 * toArray(semilinearField({ linear: [-2], nonlinear: (_t, y) => zerosLike(y) })(0, tensor([3]))) // => [-6]
 * ```
 */
export function semilinearField(problem: SemilinearProblem): OdeVectorField {
  const l = floatState(problem.linear)
  return (t, y) => add(applyLinear(l, y), problem.nonlinear(t, y))
}

function applyLinear(l: Value, y: Value): Value {
  if (l.shape.length === y.shape.length && l.shape.every((n, i) => n === y.shape[i])) return mul(l, y)
  const n = numbersOf(y).length
  if (l.shape.length === 2 && l.shape[0] === n && l.shape[1] === n) return reshape(matmul(l, reshape(y, [n])), y.shape)
  throw new MdmlError('shape', `the linear part [${l.shape}] fits neither the state [${y.shape}] nor [n, n]`)
}

/**
 * The phi functions phi_0(z) = e^z, phi_k(z) = (phi_{k-1}(z) - 1/(k-1)!) / z, evaluated without cancellation (a
 * Taylor series for |z| < 1).
 *
 * @param z - The argument.
 * @param k - The highest index.
 * @returns `[phi_0(z), ..., phi_k(z)]`.
 * @concept exponential-integrator
 * @example
 * ```ts
 * import { phi } from 'mdml-engine/dynamics/ode'
 * phi(0, 3) // => [1, 1, 0.5, 0.16666666666666666]
 * ```
 */
export function phi(z: number, k: number): number[] {
  const out = [Math.exp(z)]
  if (Math.abs(z) < 1) {
    for (let j = 1; j <= k; j++) {
      // sum_{m >= 0} z^m / (m + j)!
      let term = 1
      for (let i = 2; i <= j; i++) term /= i
      let s = term
      for (let m = 1; m < 30; m++) {
        term *= z / (m + j)
        s += term
        if (Math.abs(term) < 1e-17 * Math.abs(s)) break
      }
      out.push(s)
    }
    return out
  }
  let fact = 1
  for (let j = 1; j <= k; j++) {
    out.push(((out[j - 1] ?? 0) - 1 / fact) / z)
    fact *= j
  }
  return out
}

/**
 * The matrix phi functions [e^A, phi_1(A), ..., phi_k(A)] from one exponential of an augmented block matrix (Sidje's
 * construction), as used by exponential integrators with a matrix linear part.
 *
 * @param a - Square matrix `[n, n]`.
 * @param k - The highest index.
 * @returns The k + 1 matrices.
 * @concept exponential-integrator
 * @example
 * ```ts
 * import { phiMatrix } from 'mdml-engine/dynamics/ode'
 * import { toFlat } from 'mdml-engine/tensor'
 * phiMatrix([[0]], 2).map((m) => toFlat(m)) // => [[1], [1], [0.5]]
 * ```
 */
export function phiMatrix(a: TensorLike, k: number): Tensor[] {
  const av = numbersOf(floatState(a))
  const n = Math.round(Math.sqrt(av.length))
  if (n * n !== av.length) throw new MdmlError('shape', 'phiMatrix: a square matrix is needed')
  const m = n * (k + 1)
  const big = new Float64Array(m * m)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) big[i * m + j] = av[i * n + j] ?? 0
  for (let b = 0; b < k; b++) for (let i = 0; i < n; i++) big[(b * n + i) * m + (b + 1) * n + i] = 1
  const e = numbersOf(primalOf(expm(fromNumbers(big, [m, m]))))
  return Array.from({ length: k + 1 }, (_, b) => {
    const out = new Float64Array(n * n)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i * n + j] = e[i * m + b * n + j] ?? 0
    return fromNumbers(out, [n, n])
  })
}

/** Options of the exponential integrators. */
export interface ExponentialOptions {
  readonly dt: number
}

/** The state of an exponential integrator. */
export interface ExponentialState extends StepperState {
  readonly f: OdeVectorField
  readonly fy: Value
  readonly fOld: Value
}

type Operators = (h: number) => readonly ((v: Value) => Value)[]

/** phi_k(hL) applied to a value, for a diagonal or matrix L, cached per step size. */
function operators(linear: TensorLike, kmax: number, half: boolean): Operators {
  const l = floatState(linear)
  const cache = new Map<number, ((v: Value) => Value)[]>()
  return (h) => {
    const key = half ? h / 2 : h
    const hit = cache.get(key)
    if (hit !== undefined) return hit
    const ln = numbersOf(l)
    let ops: ((v: Value) => Value)[]
    if (l.shape.length === 2 && l.shape[0] === l.shape[1] && ln.length > 1) {
      const n = l.shape[0] ?? 0
      const mats = phiMatrix(
        fromNumbers(
          Float64Array.from(ln, (v) => v * key),
          [n, n],
        ),
        kmax,
      )
      ops = mats.map((m) => (v: Value) => reshape(matmul(m, reshape(v, [n])), v.shape))
    } else {
      const table = Array.from(ln, (v) => phi(v * key, kmax))
      ops = Array.from({ length: kmax + 1 }, (_, j) => {
        const d = fromNumbers(
          table.map((row) => row[j] ?? 0),
          l.shape,
        )
        return (v: Value) => mul(d, v)
      })
    }
    cache.set(key, ops)
    return ops
  }
}

function exponentialStepper(
  name: string,
  order: number,
  problem: SemilinearProblem,
  options: ExponentialOptions,
  advance: (t: number, y: Value, h: number, n: (t: number, y: Value) => Value) => Value,
): Stepper<ExponentialState> {
  const dt = options.dt
  if (!(dt > 0) || !Number.isFinite(dt)) throw new MdmlError('value', `${name}: dt must be positive`)
  const full = semilinearField(problem)
  return {
    name,
    order,
    stiff: true,
    init: (_field, t0, y0, tBound) => {
      const c = counter()
      const f = checked(full, y0.shape)
      const fy = tally(f, c)(t0, y0)
      return { ...initialState(t0, y0, tBound, dt, c, order), f, fy, fOld: fy, counts: snapshot(c) }
    },
    step: (s) => {
      if (s.status !== 'running') throw new MdmlError('value', `${name}: the integration has ended`)
      if (atBound(s)) return { ...s, tOld: s.t, t: s.tBound, status: 'finished', attempts: [] }
      const c: Counter = { ...s.counts }
      const n = tally(problem.nonlinear, c)
      const { tNew, h } = towards(s.t, dt, s.direction, s.tBound)
      const yNew = advance(s.t, s.y, h, n)
      const ok = allFinite(numbersOf(yNew))
      const fNew = ok ? tally(s.f, c)(tNew, yNew) : yNew
      return {
        ...s,
        tOld: s.t,
        yOld: s.y,
        fOld: s.fy,
        t: tNew,
        y: yNew,
        fy: fNew,
        attempts: [attempt(h, 0, ok)],
        status: ok ? statusAfter(tNew, s.tBound, s.direction) : 'failed',
        message: ok ? '' : `${name}: the solution became non-finite at t = ${tNew}`,
        counts: snapshot(c),
      }
    },
    dense: (s) => hermiteSegment(s.tOld, s.yOld, s.fOld, s.t, s.y, s.fy),
  }
}

/**
 * Exponential Euler for y' = L y + N(t, y): y_{n+1} = e^{hL} y_n + h phi_1(hL) N(t_n, y_n). Exact for linear problems
 * whatever the stiffness of L; first order in the nonlinear part.
 *
 * @param problem - Linear part and nonlinear field.
 * @param options - The step.
 * @returns The stepper (the field passed to `init` is ignored; the problem defines it).
 * @concept exponential-integrator
 * @example
 * ```ts
 * import { exponentialEuler, semilinearField, solveIvp } from 'mdml-engine/dynamics/ode'
 * import { toFlat, zerosLike } from 'mdml-engine/tensor'
 * const problem = { linear: [-2], nonlinear: (_t: number, y: never) => zerosLike(y) }
 * const sol = solveIvp(semilinearField(problem), [0, 1], [1], { method: exponentialEuler(problem, { dt: 0.5 }) })
 * Math.abs((toFlat(sol.y)[2] ?? 0) - Math.exp(-2)) < 1e-15 // => true
 * ```
 */
export function exponentialEuler(problem: SemilinearProblem, options: ExponentialOptions): Stepper<ExponentialState> {
  const ops = operators(problem.linear, 1, false)
  return exponentialStepper('exponential-euler', 1, problem, options, (t, y, h, n) => {
    const [e, p1] = ops(h) as [(v: Value) => Value, (v: Value) => Value]
    return combine(e(y), [[h, p1(n(t, y))]])
  })
}

/**
 * ETDRK4 (Cox and Matthews 2002, with the phi-function coefficients that make it stable to evaluate): a fourth-order
 * exponential time-differencing Runge-Kutta method for stiff semilinear problems such as spectral discretisations of
 * KdV, Kuramoto-Sivashinsky and Allen-Cahn.
 *
 * @param problem - Linear part and nonlinear field.
 * @param options - The step.
 * @returns The stepper (the field passed to `init` is ignored; the problem defines it).
 * @concept exponential-integrator
 * @reference Kassam and Trefethen (2005), Fourth-order time-stepping for stiff PDEs.
 * @example
 * ```ts
 * import { etdrk4 } from 'mdml-engine/dynamics/ode'
 * import { zerosLike } from 'mdml-engine/tensor'
 * etdrk4({ linear: [-1], nonlinear: (_t, y) => zerosLike(y) }, { dt: 0.1 }).order // => 4
 * ```
 */
export function etdrk4(problem: SemilinearProblem, options: ExponentialOptions): Stepper<ExponentialState> {
  const full = operators(problem.linear, 3, false)
  const halfOps = operators(problem.linear, 1, true)
  return exponentialStepper('etdrk4', 4, problem, options, (t, y, h, n) => {
    const [e, p1, p2, p3] = full(h) as [
      (v: Value) => Value,
      (v: Value) => Value,
      (v: Value) => Value,
      (v: Value) => Value,
    ]
    const [e2, q1] = halfOps(h) as [(v: Value) => Value, (v: Value) => Value]
    const nu = n(t, y)
    const e2y = e2(y)
    const a = combine(e2y, [[h / 2, q1(nu)]])
    const na = n(t + h / 2, a)
    const b = combine(e2y, [[h / 2, q1(na)]])
    const nb = n(t + h / 2, b)
    const c = combine(e2(a), [
      [
        h / 2,
        q1(
          combine(null, [
            [2, nb],
            [-1, nu],
          ]),
        ),
      ],
    ])
    const nc = n(t + h, c)
    // f1 = phi1 - 3 phi2 + 4 phi3, f2 = phi2 - 2 phi3, f3 = -phi2 + 4 phi3 (all at hL).
    const f1 = (v: Value): Value =>
      combine(p1(v), [
        [-3, p2(v)],
        [4, p3(v)],
      ])
    const f2 = (v: Value): Value => combine(p2(v), [[-2, p3(v)]])
    const f3 = (v: Value): Value =>
      combine(null, [
        [-1, p2(v)],
        [4, p3(v)],
      ])
    return combine(e(y), [
      [h, f1(nu)],
      [2 * h, f2(combine(na, [[1, nb]]))],
      [h, f3(nc)],
    ])
  })
}
