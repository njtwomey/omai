/**
 * Boundary value problems for ODEs, where conditions are given at both ends of an interval rather than at the start.
 * `solveBvp` ports scipy's `solve_bvp` (fourth-order Lobatto IIIA collocation with damped Newton and mesh refinement),
 * alongside single and multiple shooting with Jacobians taken by differentiating through the integrator.
 */

import { MdmlError, maxOf } from '../../base/index.ts'
import { pinv } from '../../numerics/linalg/index.ts'
import { fsolve } from '../../numerics/roots/index.ts'
import { concat, reshape, sliceAxis, sub, type Tensor, type TensorLike, type Value } from '../../tensor/index.ts'
import {
  collocation,
  column,
  createSpline,
  modifyMesh,
  type NumBc,
  type NumField,
  rmsResiduals,
  solveNewton,
} from './_bvp.ts'
import { floatState, fromNumbers, numbersOf } from './_vec.ts'
import type { Stepper, VectorField } from './problem.ts'
import { type OdeMethodName, solveIvp } from './solve-ivp.ts'

/** A vectorised boundary-value field: x `[m]`, y `[n, m]`, parameters `[k]` to dy/dx `[n, m]` (scipy's `fun`). */
export type BvpField = (x: Tensor, y: Tensor, p: Tensor) => Value

/** Boundary conditions: ya `[n]`, yb `[n]`, parameters `[k]` to residuals `[n + k]` (scipy's `bc`). */
export type BvpConditions = (ya: Tensor, yb: Tensor, p: Tensor) => Value

/** Options of {@link solveBvp}. */
export interface SolveBvpOptions {
  /** Initial guesses of unknown parameters. */
  readonly p?: TensorLike
  /** The singular term S of y' = S y / (x - a) + f(x, y, p). */
  readonly S?: TensorLike
  /** Relative residual tolerance (default 1e-3). */
  readonly tol?: number
  /** The mesh-node budget (default 1000). */
  readonly maxNodes?: number
  /** Boundary-condition tolerance (default tol). */
  readonly bcTol?: number
}

/** The result of {@link solveBvp} (scipy's `BVPResult`). */
export interface BvpSolution {
  /** The final mesh. */
  readonly x: readonly number[]
  /** The solution at the mesh `[n, m]`. */
  readonly y: Tensor
  /** Its derivative at the mesh `[n, m]`. */
  readonly yp: Tensor
  /** The parameters found, when there were any. */
  readonly p: Tensor | undefined
  /** The C1 cubic spline solution: values `[n]`, or derivatives with `nu = 1`. */
  readonly sol: (x: number, nu?: 0 | 1) => Tensor
  readonly rmsResiduals: readonly number[]
  readonly niter: number
  /** 0 converged, 1 too many nodes, 2 singular Jacobian, 3 boundary conditions not met. */
  readonly status: 0 | 1 | 2 | 3
  readonly message: string
  readonly success: boolean
}

const MESSAGES = [
  'The algorithm converged to the desired accuracy.',
  'The maximum number of mesh nodes is exceeded.',
  'A singular Jacobian encountered when solving the collocation system.',
  'The solver was unable to satisfy boundary conditions tolerance on iteration 10.',
] as const

/**
 * Solve a boundary value problem y' = f(x, y, p), bc(y(a), y(b), p) = 0 by collocation: a port of scipy's
 * `solve_bvp` (Kierzenka and Shampine's fourth-order Lobatto IIIA collocation with a damped Newton method on the
 * global sparse system, residuals estimated by Lobatto quadrature and the mesh refined where they exceed `tol`).
 * Unknown parameters and a singular term S/(x - a) are supported. Never throws on hard problems: the status says why
 * it stopped.
 *
 * @param fun - The vectorised field (x `[m]`, y `[n, m]`, p `[k]`).
 * @param bc - The boundary residuals (`[n + k]`).
 * @param x - The initial mesh (increasing).
 * @param y - The initial guess `[n, m]`.
 * @param options - Parameters, singular term, tolerances and node budget.
 * @returns The solution with its mesh, spline, residuals and status.
 * @throws MdmlError with code `shape` or `value` for bad inputs.
 * @concept boundary-value-problem
 * @example
 * ```ts
 * import { solveBvp } from 'mdml-engine/dynamics/ode'
 * import { concat, neg, sliceAxis, sub, tensor } from 'mdml-engine/tensor'
 * // y'' = -y on [0, pi/2], y(0) = 0, y(pi/2) = 1: y = sin x.
 * const fun = (_x, y) => concat([sliceAxis(y, 0, 1, 2), neg(sliceAxis(y, 0, 0, 1))], 0)
 * const bc = (ya, yb) => concat([sliceAxis(ya, 0, 0, 1), sub(sliceAxis(yb, 0, 0, 1), 1)], 0)
 * const r = solveBvp(fun, bc, [0, 0.5, 1, Math.PI / 2], tensor([[0, 0, 0, 0], [1, 1, 1, 1]]), { tol: 1e-6 })
 * Math.abs((r.sol(Math.PI / 6).data[0] ?? 0) - 0.5) < 1e-5 // => true
 * ```
 */
export function solveBvp(
  fun: BvpField,
  bc: BvpConditions,
  x: TensorLike,
  y: TensorLike,
  options: SolveBvpOptions = {},
): BvpSolution {
  let xs: Float64Array = numbersOf(floatState(x))
  const yv = floatState(y)
  if (yv.shape.length !== 2) throw new MdmlError('shape', 'solveBvp: y must be [n, m]')
  const n = yv.shape[0] ?? 0
  const m0 = yv.shape[1] ?? 0
  if (m0 !== xs.length) throw new MdmlError('shape', `solveBvp: y needs ${xs.length} columns`)
  let h: Float64Array = Float64Array.from({ length: xs.length - 1 }, (_, j) => (xs[j + 1] ?? 0) - (xs[j] ?? 0))
  if (h.some((v) => !(v > 0))) throw new MdmlError('value', 'solveBvp: x must be strictly increasing')
  let yN: Float64Array = Float64Array.from(numbersOf(yv))
  let p: Float64Array =
    options.p === undefined ? new Float64Array(0) : Float64Array.from(numbersOf(floatState(options.p)))
  const k = p.length
  const tol = Math.max(options.tol ?? 1e-3, 100 * Number.EPSILON)
  const bcTol = options.bcTol ?? tol
  const maxNodes = options.maxNodes ?? 1000
  const a = xs[0] ?? 0
  let B: Float64Array | undefined
  let D: Float64Array | undefined
  let S: Float64Array | undefined
  if (options.S !== undefined) {
    const sv = floatState(options.S)
    S = Float64Array.from(numbersOf(sv))
    if (S.length !== n * n) throw new MdmlError('shape', `solveBvp: S must be ${n} x ${n}`)
    const ps = numbersOf(pinv(sv))
    B = new Float64Array(n * n)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0
        for (let c = 0; c < n; c++) s += (ps[i * n + c] ?? 0) * (S[c * n + j] ?? 0)
        B[i * n + j] = (i === j ? 1 : 0) - s
      }
    }
    const ims = Float64Array.from(S, (v, idx) => (Math.floor(idx / n) === idx % n ? 1 : 0) - v)
    D = Float64Array.from(numbersOf(pinv(fromNumbers(ims, [n, n]))))
    const y0 = column(yN, n, m0, 0)
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let c = 0; c < n; c++) s += (B[i * n + c] ?? 0) * (y0[c] ?? 0)
      yN[i * m0] = s
    }
  }
  const raw: NumField = (xv, yy, pp) => {
    const out = numbersOf(
      fun(fromNumbers(xv, [xv.length]), fromNumbers(yy, [n, xv.length]), fromNumbers(pp, [pp.length])),
    )
    if (out.length !== n * xv.length) throw new MdmlError('shape', `solveBvp: fun must return [${n}, ${xv.length}]`)
    return Float64Array.from(out)
  }
  const field: NumField =
    S === undefined
      ? raw
      : (xv, yy, pp) => {
          const mm = xv.length
          const f = raw(xv, yy, pp)
          const Sv = S as Float64Array
          for (let j = 0; j < mm; j++) {
            const xj = xv[j] ?? 0
            if (j === 0 && xj === a) {
              const f0 = column(f, n, mm, 0)
              for (let i = 0; i < n; i++) {
                let s = 0
                for (let c = 0; c < n; c++) s += (D?.[i * n + c] ?? 0) * (f0[c] ?? 0)
                f[i * mm] = s
              }
              continue
            }
            for (let i = 0; i < n; i++) {
              let s = 0
              for (let c = 0; c < n; c++) s += (Sv[i * n + c] ?? 0) * (yy[c * mm + j] ?? 0)
              f[i * mm + j] = (f[i * mm + j] ?? 0) + s / (xj - a)
            }
          }
          return f
        }
  const bcN: NumBc = (ya, yb, pp) => {
    const out = numbersOf(bc(fromNumbers(ya, [n]), fromNumbers(yb, [n]), fromNumbers(pp, [pp.length])))
    if (out.length !== n + k) throw new MdmlError('shape', `solveBvp: bc must return ${n + k} residuals`)
    return Float64Array.from(out)
  }
  let status: 0 | 1 | 2 | 3 = 0
  let iteration = 0
  let sol = createSpline(yN, field(xs, yN, p), xs, h, n)
  let rms: Float64Array = new Float64Array(0)
  let f: Float64Array = new Float64Array(0)
  for (;;) {
    const m = xs.length
    const r = solveNewton(field, bcN, xs, h, yN, p, B, tol, bcTol, n)
    yN = r.y
    p = r.p
    iteration += 1
    const col = collocation(field, yN, p, xs, h, n)
    f = col.f
    const bcRes = bcN(column(yN, n, m, 0), column(yN, n, m, m - 1), p)
    const maxBc = maxOf(Array.from(bcRes, Math.abs))
    const rMid = Float64Array.from(col.colRes, (v, idx) => (1.5 * v) / (h[idx % (m - 1)] ?? 1))
    sol = createSpline(yN, f, xs, h, n)
    rms = rmsResiduals(field, sol, xs, h, p, rMid, col.fMid, n)
    if (r.singular) {
      status = 2
      break
    }
    const one: number[] = []
    const two: number[] = []
    rms.forEach((v, j) => {
      if (v > tol && v < 100 * tol) one.push(j)
      else if (v >= 100 * tol) two.push(j)
    })
    const added = one.length + 2 * two.length
    if (m + added > maxNodes) {
      status = 1
      break
    }
    if (added > 0) {
      xs = modifyMesh(xs, one, two)
      h = Float64Array.from({ length: xs.length - 1 }, (_, j) => (xs[j + 1] ?? 0) - (xs[j] ?? 0))
      const next = new Float64Array(n * xs.length)
      xs.forEach((xv, j) => {
        const v = sol.at(xv)
        for (let i = 0; i < n; i++) next[i * xs.length + j] = v[i] ?? 0
      })
      yN = next
    } else if (maxBc <= bcTol) {
      status = 0
      break
    } else if (iteration >= 10) {
      status = 3
      break
    }
  }
  const m = xs.length
  const finalSol = sol
  return {
    x: Array.from(xs),
    y: fromNumbers(yN, [n, m]),
    yp: fromNumbers(f, [n, m]),
    p: k === 0 ? undefined : fromNumbers(p, [k]),
    sol: (xv, nu = 0) => fromNumbers(finalSol.at(xv, nu), [n]),
    rmsResiduals: Array.from(rms),
    niter: iteration,
    status,
    message: MESSAGES[status],
    success: status === 0,
  }
}

/** Options of the shooting methods. */
export interface ShootingOptions {
  /** The integrator (default DOP853). */
  readonly method?: OdeMethodName | Stepper
  readonly rtol?: number
  readonly atol?: number
  /** Newton iterations (default 50). */
  readonly maxIterations?: number
  /** Residual tolerance (default 1e-10). */
  readonly ftol?: number
}

/** The result of a shooting method. */
export interface ShootingResult {
  /** The initial state found. */
  readonly y0: Tensor
  /** The states at the shooting nodes `[nodes, n]` (the initial state alone for single shooting). */
  readonly nodes: Tensor
  readonly residual: number
  readonly iterations: number
  readonly converged: boolean
  readonly message: string
}

function integrate(f: VectorField, t0: number, t1: number, y0: Value, options: ShootingOptions): Value {
  const sol = solveIvp(f, [t0, t1], y0, {
    method: options.method ?? 'DOP853',
    rtol: options.rtol ?? 1e-10,
    atol: options.atol ?? 1e-12,
    tEval: [t1],
  })
  if (!sol.success) throw new MdmlError('convergence', `shooting: the integration failed: ${sol.message}`)
  return sol.ys[sol.ys.length - 1] as Value
}

/**
 * Single shooting: find the initial state s with bc(s, y(b; s)) = 0 by Newton's method, the Jacobian of the end
 * state with respect to s taken by differentiating through the integrator.
 *
 * @param f - The vector field dy/dt = f(t, y).
 * @param bc - Boundary residuals bc(ya, yb) of length n.
 * @param span - [a, b].
 * @param guess - The initial-state guess `[n]`.
 * @param options - Integrator, tolerances and iteration cap.
 * @returns The initial state and convergence information.
 * @concept shooting-method
 * @example
 * ```ts
 * import { shooting } from 'mdml-engine/dynamics/ode'
 * import { concat, neg, sliceAxis, sub } from 'mdml-engine/tensor'
 * const f = (_t, y) => concat([sliceAxis(y, 0, 1, 2), neg(sliceAxis(y, 0, 0, 1))], 0)
 * const bc = (ya, yb) => concat([sliceAxis(ya, 0, 0, 1), sub(sliceAxis(yb, 0, 0, 1), 1)], 0)
 * const r = shooting(f, bc, [0, Math.PI / 2], [0, 0.5])
 * Math.abs((r.y0.data[1] ?? 0) - 1) < 1e-8 // => true
 * ```
 */
export function shooting(
  f: VectorField,
  bc: (ya: Value, yb: Value) => Value,
  span: readonly [number, number],
  guess: TensorLike,
  options: ShootingOptions = {},
): ShootingResult {
  const s0 = floatState(guess)
  const residual = (s: Value): Value => bc(s, integrate(f, span[0], span[1], s, options))
  const r = fsolve(residual, s0, { maxIterations: options.maxIterations ?? 50, ftol: options.ftol ?? 1e-10 })
  return {
    y0: r.x,
    nodes: reshape(r.x, [1, -1]) as Tensor,
    residual: r.history[r.history.length - 1] ?? Number.NaN,
    iterations: r.iterations,
    converged: r.converged,
    message: r.message,
  }
}

/**
 * Multiple shooting: unknown states at every node t_0 < ... < t_M-1 (with t_0 = a and the last interval ending at b),
 * matched by continuity y(t_{j+1}; s_j) = s_{j+1} and the boundary conditions, all solved together by Newton's
 * method. Better conditioned than single shooting for unstable or long intervals.
 *
 * @param f - The vector field.
 * @param bc - Boundary residuals bc(ya, yb) of length n.
 * @param nodes - The shooting nodes, starting at a, ending at b.
 * @param guesses - Guesses `[nodes - 1, n]` for the state at each node but the last.
 * @param options - Integrator, tolerances and iteration cap.
 * @returns The states at the nodes and convergence information.
 * @throws MdmlError with code `value` for fewer than two nodes or mismatched guesses.
 * @concept shooting-method
 * @example
 * ```ts
 * import { multipleShooting } from 'mdml-engine/dynamics/ode'
 * import { concat, neg, sliceAxis, sub } from 'mdml-engine/tensor'
 * const f = (_t, y) => concat([sliceAxis(y, 0, 1, 2), neg(sliceAxis(y, 0, 0, 1))], 0)
 * const bc = (ya, yb) => concat([sliceAxis(ya, 0, 0, 1), sub(sliceAxis(yb, 0, 0, 1), 1)], 0)
 * const r = multipleShooting(f, bc, [0, 0.8, Math.PI / 2], [[0, 0.5], [0.5, 0.5]])
 * Math.abs((r.y0.data[1] ?? 0) - 1) < 1e-8 // => true
 * ```
 */
export function multipleShooting(
  f: VectorField,
  bc: (ya: Value, yb: Value) => Value,
  nodes: readonly number[],
  guesses: TensorLike,
  options: ShootingOptions = {},
): ShootingResult {
  if (nodes.length < 2) throw new MdmlError('value', 'multipleShooting: give at least two nodes')
  const g = floatState(guesses)
  const segments = nodes.length - 1
  if (g.shape.length !== 2 || g.shape[0] !== segments) {
    throw new MdmlError('value', `multipleShooting: give ${segments} guesses as [${segments}, n]`)
  }
  const n = g.shape[1] ?? 0
  const residual = (z: Value): Value => {
    const s = (j: number): Value => sliceAxis(z, 0, j * n, (j + 1) * n)
    const parts: Value[] = []
    for (let j = 0; j < segments - 1; j++) {
      parts.push(sub(integrate(f, nodes[j] as number, nodes[j + 1] as number, s(j), options), s(j + 1)))
    }
    const end = integrate(f, nodes[segments - 1] as number, nodes[segments] as number, s(segments - 1), options)
    parts.push(bc(s(0), end))
    return concat(parts, 0)
  }
  const r = fsolve(residual, reshape(g, [-1]), {
    maxIterations: options.maxIterations ?? 50,
    ftol: options.ftol ?? 1e-10,
  })
  return {
    y0: fromNumbers(numbersOf(r.x).slice(0, n), [n]),
    nodes: fromNumbers(numbersOf(r.x), [segments, n]),
    residual: r.history[r.history.length - 1] ?? Number.NaN,
    iterations: r.iterations,
    converged: r.converged,
    message: r.message,
  }
}
