/**
 * Implicit solvers for stiff ODEs, where explicit methods would need tiny steps for stability: ports of scipy's Radau
 * IIA (order 5) and variable-order BDF, fixed-step diagonally implicit Runge-Kutta methods (backward Euler, trapezoid,
 * implicit midpoint, SDIRK), the Rosenbrock method of MATLAB's ode23s, and a stiffness report from the Jacobian's
 * eigenvalues.
 */

import { jacfwd } from '../../autodiff/index.ts'
import { maxOf, minOf, OmaiError } from '../../base/index.ts'
import { eigvals, type LuFactorisation, solve } from '../../numerics/linalg/index.ts'
import { mul, reshape, sub, type TensorLike, type Value } from '../../tensor/index.ts'
import { BDF_MAX_ORDER, bdfDense, bdfStep } from './_bdf.ts'
import {
  checkMass,
  identityNumbers,
  implicitCorrection,
  isTracedState,
  type JacobianOption,
  jacobianAt,
  luApply,
  massNumbers,
  massTimes,
  newtonTolerance,
  realLu,
} from './_newton.ts'
import { radauDense, radauStep } from './_radau.ts'
import {
  atBound,
  attempt,
  type Counter,
  checked,
  counter,
  initialState,
  selectInitialStep,
  snapshot,
  statusAfter,
  tally,
  towards,
  validFirstStep,
  validMaxStep,
  validTolerances,
} from './_step.ts'
import { allFinite, combine, errorScale, floatState, fromNumbers, minStep, numbersOf, rms, scaledRms } from './_vec.ts'
import {
  hermiteSegment,
  type OdeVectorField,
  type StepAttempt,
  type Stepper,
  type StepperOptions,
  type StepperState,
} from './problem.ts'

const EPS = Number.EPSILON

/** Options of the implicit steppers. */
export interface ImplicitOptions extends StepperOptions {
  /** How df/dy is obtained (default: forward-mode autodiff). */
  readonly jac?: JacobianOption
  /** A constant mass matrix M for M y' = f (`[n, n]`, possibly singular for index-1 DAEs). */
  readonly mass?: TensorLike
}

/** The state of {@link radau}. */
export interface RadauState extends StepperState {
  readonly f: OdeVectorField
  readonly fy: Value
  readonly jacobian: Float64Array
  readonly luReal: LuFactorisation | undefined
  readonly luComplex: LuFactorisation | undefined
  readonly currentJac: boolean
  readonly hAbsOld: number | undefined
  readonly errorNormOld: number | undefined
  /** The stage increments of the last step (y(t_old + c_i h) - y_old). */
  readonly z: readonly Value[]
  /** The signed size of the last step. */
  readonly h: number
}

/**
 * The three-stage Radau IIA method of order 5 for stiff problems, a port of scipy's `Radau` (Hairer and Wanner's
 * RADAU5 with simplified Newton iterations on the transformed system, real and complex LU factorisations, the
 * embedded error estimate filtered through (M/h - J)^-1, and Gustafsson's predictive step control). It accepts a
 * constant mass matrix, so it also solves index-1 DAEs. Untraced, the Newton iterations run on numbers; when the
 * state or field is traced, each accepted step is made differentiable by an implicit-function correction with the
 * exact stage Jacobians.
 *
 * @param options - Tolerances (default rtol 1e-3, atol 1e-6), step limits, Jacobian and mass matrix.
 * @returns The stepper.
 * @concept stiff-ode
 * @example
 * ```ts
 * import { radau } from 'omai-engine/dynamics/ode'
 * radau().order // => 5
 * ```
 */
export function radau(options: ImplicitOptions = {}): Stepper<RadauState> {
  const { rtol, atol } = validTolerances(options.rtol ?? 1e-3, options.atol ?? 1e-6)
  const maxStep = validMaxStep(options.maxStep)
  const newtonTol = newtonTolerance(rtol)
  const jacOpt = options.jac ?? 'autodiff'
  const mass = massNumbers(options.mass)
  return {
    name: 'radau',
    order: 5,
    stiff: true,
    init: (field, t0, y0, tBound) => {
      const c = counter()
      const f = checked(field, y0.shape)
      const fy = tally(f, c)(t0, y0)
      const base = initialState(t0, y0, tBound, 0, c, 5)
      const hAbs =
        options.firstStep === undefined
          ? selectInitialStep(tally(f, c), t0, y0, tBound, maxStep, fy, base.direction, 3, rtol, atol)
          : validFirstStep(options.firstStep, t0, tBound)
      const jacobian = jacobianAt(f, t0, y0, jacOpt)
      c.njev += 1
      return {
        ...base,
        hAbs,
        f,
        fy,
        jacobian,
        luReal: undefined,
        luComplex: undefined,
        currentJac: true,
        hAbsOld: undefined,
        errorNormOld: undefined,
        z: [],
        h: 0,
        counts: snapshot(c),
      }
    },
    step: (s) => radauStep(s, { rtol, atol, maxStep, newtonTol, jac: jacOpt, mass }),
    dense: (s) => radauDense(s),
  }
}

/** The state of {@link bdf}: the modified divided differences D of the solution and the Newton matrices. */
export interface BdfState extends StepperState {
  readonly f: OdeVectorField
  readonly d: readonly Value[]
  readonly nEqualSteps: number
  readonly jacobian: Float64Array
  readonly lu: LuFactorisation | undefined
}

/**
 * The variable-order (1 to 5), variable-step backward differentiation formulas in quasi-constant step form with the
 * Klopfenstein-Shampine NDF modification: a port of scipy's `BDF` (simplified Newton iterations, step changes by
 * rescaling the difference array, the order raised or lowered after order + 1 equal steps). Accepts a constant mass
 * matrix. When traced, each step is made differentiable by an implicit-function correction.
 *
 * @param options - Tolerances (default rtol 1e-3, atol 1e-6), step limits, Jacobian and mass matrix.
 * @returns The stepper.
 * @concept stiff-ode
 * @example
 * ```ts
 * import { bdf } from 'omai-engine/dynamics/ode'
 * bdf().stiff // => true
 * ```
 */
export function bdf(options: ImplicitOptions = {}): Stepper<BdfState> {
  const { rtol, atol } = validTolerances(options.rtol ?? 1e-3, options.atol ?? 1e-6)
  const maxStep = validMaxStep(options.maxStep)
  const newtonTol = newtonTolerance(rtol)
  const jacOpt = options.jac ?? 'autodiff'
  const mass = massNumbers(options.mass)
  return {
    name: 'bdf',
    order: BDF_MAX_ORDER,
    stiff: true,
    init: (field, t0, y0, tBound) => {
      const c = counter()
      const f = checked(field, y0.shape)
      const fy = tally(f, c)(t0, y0)
      const base = initialState(t0, y0, tBound, 0, c, 1)
      const hAbs =
        options.firstStep === undefined
          ? selectInitialStep(tally(f, c), t0, y0, tBound, maxStep, fy, base.direction, 1, rtol, atol)
          : validFirstStep(options.firstStep, t0, tBound)
      const jacobian = jacobianAt(f, t0, y0, jacOpt)
      c.njev += 1
      const zero = combine(null, [[0, y0]])
      const d = [
        y0,
        combine(null, [[hAbs * base.direction, fy]]),
        ...Array.from({ length: BDF_MAX_ORDER + 1 }, () => zero),
      ]
      return { ...base, hAbs, f, d, nEqualSteps: 0, jacobian, lu: undefined, counts: snapshot(c) }
    },
    step: (s) => bdfStep(s, { rtol, atol, maxStep, newtonTol, jac: jacOpt, mass }),
    dense: (s) => bdfDense(s),
  }
}

/** A diagonally implicit Runge-Kutta method: `a` lower triangular including the diagonal. */
export interface DirkTableau {
  readonly name: string
  readonly a: readonly (readonly number[])[]
  readonly b: readonly number[]
  readonly c: readonly number[]
  readonly order: number
}

const SDIRK2_GAMMA = 1 - Math.SQRT2 / 2
const SDIRK3_GAMMA = 0.5 + Math.sqrt(3) / 6

/** Backward Euler (order 1, L-stable). */
export const BACKWARD_EULER: DirkTableau = { name: 'backward-euler', a: [[1]], b: [1], c: [1], order: 1 }
/** The trapezoid rule, Crank-Nicolson (order 2, A-stable), as an explicit-first-stage DIRK. */
export const TRAPEZOID: DirkTableau = {
  name: 'trapezoid',
  a: [
    [0, 0],
    [0.5, 0.5],
  ],
  b: [0.5, 0.5],
  c: [0, 1],
  order: 2,
}
/** The implicit midpoint rule (order 2, symplectic, A-stable). */
export const IMPLICIT_MIDPOINT: DirkTableau = { name: 'implicit-midpoint', a: [[0.5]], b: [1], c: [0.5], order: 2 }
/** Alexander's two-stage L-stable SDIRK of order 2 (gamma = 1 - 1/sqrt 2). */
export const SDIRK2: DirkTableau = {
  name: 'sdirk2',
  a: [
    [SDIRK2_GAMMA, 0],
    [1 - SDIRK2_GAMMA, SDIRK2_GAMMA],
  ],
  b: [1 - SDIRK2_GAMMA, SDIRK2_GAMMA],
  c: [SDIRK2_GAMMA, 1],
  order: 2,
}
/** Crouzeix's two-stage A-stable SDIRK of order 3 (gamma = 1/2 + sqrt 3 / 6). */
export const SDIRK3: DirkTableau = {
  name: 'sdirk3',
  a: [
    [SDIRK3_GAMMA, 0],
    [1 - 2 * SDIRK3_GAMMA, SDIRK3_GAMMA],
  ],
  b: [0.5, 0.5],
  c: [SDIRK3_GAMMA, 1 - SDIRK3_GAMMA],
  order: 3,
}

/** Options of the fixed-step implicit steppers. */
export interface DirkOptions {
  readonly dt: number
  readonly jac?: JacobianOption
  /** Newton tolerance on the scaled increment (default 1e-10). */
  readonly newtonTol?: number
  /** Newton iterations per stage before refreshing the Jacobian and finally failing (default 10). */
  readonly maxNewton?: number
}

/** The state of a fixed-step implicit stepper. */
export interface DirkState extends StepperState {
  readonly f: OdeVectorField
  readonly fy: Value
  readonly fOld: Value
}

/**
 * A fixed-step diagonally implicit Runge-Kutta stepper: each implicit stage is solved by Newton's method with the
 * Jacobian at the stage (refreshed when convergence stalls); the step fails with a message when Newton diverges or the
 * state stops being finite. Traced steps are differentiable through implicit-function corrections.
 *
 * @param tableau - The method.
 * @param options - Step size, Jacobian and Newton settings.
 * @returns The stepper.
 * @throws OmaiError with code `value` when dt is not positive.
 * @concept stiff-ode
 * @example
 * ```ts
 * import { dirk, BACKWARD_EULER } from 'omai-engine/dynamics/ode'
 * import { mul, tensor, toArray } from 'omai-engine/tensor'
 * const s = dirk(BACKWARD_EULER, { dt: 0.5 })
 * // y' = -y: one backward Euler step gives y / (1 + h).
 * toArray(s.step(s.init((_t, y) => mul(y, -1), 0, tensor([3]), 1)).y) // => [2]
 * ```
 */
export function dirk(tableau: DirkTableau, options: DirkOptions): Stepper<DirkState> {
  const dt = options.dt
  if (!(dt > 0) || !Number.isFinite(dt)) throw new OmaiError('value', `${tableau.name}: dt must be positive`)
  const jac = options.jac ?? 'autodiff'
  const tol = options.newtonTol ?? 1e-10
  const maxNewton = options.maxNewton ?? 10
  return {
    name: tableau.name,
    order: tableau.order,
    stiff: true,
    init: (field, t0, y0, tBound) => {
      const c = counter()
      const f = checked(field, y0.shape)
      const fy = tally(f, c)(t0, y0)
      return { ...initialState(t0, y0, tBound, dt, c, tableau.order), f, fy, fOld: fy, counts: snapshot(c) }
    },
    step: (s) => {
      if (s.status !== 'running') throw new OmaiError('value', `${tableau.name}: the integration has ended`)
      if (atBound(s)) return { ...s, tOld: s.t, t: s.tBound, status: 'finished', attempts: [] }
      const c: Counter = { ...s.counts }
      const f = tally(s.f, c)
      const { tNew, h } = towards(s.t, dt, s.direction, s.tBound)
      const shape = s.y.shape
      const ks: Value[] = []
      for (let i = 0; i < tableau.b.length; i++) {
        const row = tableau.a[i] ?? []
        const ti = s.t + (tableau.c[i] ?? 0) * h
        const base = combine(
          s.y,
          ks.map((k, j) => [h * (row[j] ?? 0), k] as const),
        )
        const aii = row[i] ?? 0
        if (aii === 0) {
          ks.push(i === 0 && tableau.c[0] === 0 ? s.fy : f(ti, base))
          continue
        }
        const z = newtonStage(f, s.f, ti, base, h * aii, ks[i - 1] ?? s.fy, jac, tol, maxNewton, c)
        if (z === undefined) {
          return {
            ...s,
            status: 'failed',
            message: `${tableau.name}: Newton's method did not converge at t = ${ti}`,
            attempts: [attempt(h, Number.POSITIVE_INFINITY, false)],
            counts: snapshot(c),
          }
        }
        const zv = isTracedState(base, s.fy)
          ? implicitCorrection(
              z,
              shape,
              (v) =>
                combine(v, [
                  [-1, base],
                  [-h * aii, f(ti, v)],
                ]),
              () => {
                const jz = jacobianAt(s.f, ti, fromNumbers(z, shape), jac)
                const n = z.length
                return Float64Array.from(jz, (v, k) => (k % (n + 1) === 0 ? 1 : 0) - h * aii * v)
              },
            )
          : fromNumbers(z, shape)
        ks.push(
          combine(null, [
            [1 / (h * aii), zv],
            [-1 / (h * aii), base],
          ]),
        )
      }
      const yNew = combine(
        s.y,
        tableau.b.map((bi, i) => [h * bi, ks[i] as Value] as const),
      )
      const ok = allFinite(numbersOf(yNew))
      const fNew = ok ? f(tNew, yNew) : yNew
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
        message: ok ? '' : `${tableau.name}: the solution became non-finite at t = ${tNew}`,
        counts: snapshot(c),
      }
    },
    dense: (s) => hermiteSegment(s.tOld, s.yOld, s.fOld, s.t, s.y, s.fy),
  }
}

/** Solve z = base + ha f(t, z) by Newton's method on numbers; undefined when it does not converge. */
function newtonStage(
  f: OdeVectorField,
  raw: OdeVectorField,
  t: number,
  base: Value,
  ha: number,
  guessSlope: Value,
  jac: JacobianOption,
  tol: number,
  maxNewton: number,
  c: Counter,
): Float64Array | undefined {
  const b = numbersOf(base)
  const shape = base.shape
  const n = b.length
  const g = numbersOf(guessSlope)
  const z = Float64Array.from(b, (v, i) => v + ha * (g[i] ?? 0))
  for (let refresh = 0; refresh < 2; refresh++) {
    const J = jacobianAt(raw, t, fromNumbers(z, shape), jac)
    c.njev += 1
    const m = Float64Array.from(J, (v, k) => (k % (n + 1) === 0 ? 1 : 0) - ha * v)
    const lu = realLu(m, n)
    c.nlu += 1
    for (let it = 0; it < maxNewton; it++) {
      const fz = numbersOf(f(t, fromNumbers(z, shape)))
      if (!allFinite(fz)) return undefined
      const r = Float64Array.from(z, (v, i) => -(v - (b[i] ?? 0) - ha * (fz[i] ?? 0)))
      const dz = luApply(lu, r)
      for (let i = 0; i < n; i++) z[i] = (z[i] ?? 0) + (dz[i] ?? 0)
      const size = rms(dz) / (1 + rms(z))
      if (size <= tol) return z
    }
  }
  return undefined
}

/**
 * The backward Euler stepper.
 *
 * @param options - Step size, Jacobian and Newton settings.
 * @returns The stepper.
 * @example
 * ```ts
 * import { backwardEuler } from 'omai-engine/dynamics/ode'
 * backwardEuler({ dt: 0.1 }).order // => 1
 * ```
 */
export function backwardEuler(options: DirkOptions): Stepper<DirkState> {
  return dirk(BACKWARD_EULER, options)
}

/**
 * The trapezoid (Crank-Nicolson) stepper.
 *
 * @param options - Step size, Jacobian and Newton settings.
 * @returns The stepper.
 * @example
 * ```ts
 * import { trapezoid } from 'omai-engine/dynamics/ode'
 * trapezoid({ dt: 0.1 }).order // => 2
 * ```
 */
export function trapezoid(options: DirkOptions): Stepper<DirkState> {
  return dirk(TRAPEZOID, options)
}

/**
 * The implicit midpoint stepper.
 *
 * @param options - Step size, Jacobian and Newton settings.
 * @returns The stepper.
 * @example
 * ```ts
 * import { implicitMidpoint } from 'omai-engine/dynamics/ode'
 * implicitMidpoint({ dt: 0.1 }).name // => 'implicit-midpoint'
 * ```
 */
export function implicitMidpoint(options: DirkOptions): Stepper<DirkState> {
  return dirk(IMPLICIT_MIDPOINT, options)
}

/**
 * An SDIRK stepper: Alexander's L-stable order-2 method (default) or Crouzeix's order-3 method.
 *
 * @param options - Step size, Jacobian, Newton settings and the order.
 * @returns The stepper.
 * @example
 * ```ts
 * import { sdirk } from 'omai-engine/dynamics/ode'
 * sdirk({ dt: 0.1, order: 3 }).order // => 3
 * ```
 */
export function sdirk(options: DirkOptions & { readonly order?: 2 | 3 }): Stepper<DirkState> {
  return dirk(options.order === 3 ? SDIRK3 : SDIRK2, options)
}

/** The state of {@link rosenbrock23}. */
export interface RosenbrockState extends StepperState {
  readonly f: OdeVectorField
  readonly fy: Value
  readonly k1: Value
  readonly k2: Value
  readonly h: number
}

const ROS_D = 1 / (2 + Math.SQRT2)
const ROS_E32 = 6 + Math.SQRT2

/**
 * The linearly implicit Rosenbrock method of Shampine and Reichelt (MATLAB's ode23s): order 2 with an order-3 error
 * estimate, one Jacobian and three linear solves per step and no Newton iteration, L-stable, with a continuous
 * extension. Built from tensor operations (the Jacobian traced when the problem is), so it is differentiable through
 * its steps. Accepts a constant mass matrix.
 *
 * @param options - Tolerances (default rtol 1e-3, atol 1e-6), step limits, Jacobian and mass matrix.
 * @returns The stepper.
 * @concept stiff-ode
 * @reference Shampine and Reichelt (1997), The MATLAB ODE suite.
 * @example
 * ```ts
 * import { rosenbrock23 } from 'omai-engine/dynamics/ode'
 * rosenbrock23().order // => 2
 * ```
 */
export function rosenbrock23(options: ImplicitOptions = {}): Stepper<RosenbrockState> {
  const { rtol, atol } = validTolerances(options.rtol ?? 1e-3, options.atol ?? 1e-6)
  const maxStep = validMaxStep(options.maxStep)
  const jac = options.jac ?? 'autodiff'
  const mass = massNumbers(options.mass)
  return {
    name: 'rosenbrock23',
    order: 2,
    stiff: true,
    init: (field, t0, y0, tBound) => {
      const c = counter()
      const f = checked(field, y0.shape)
      const fy = tally(f, c)(t0, y0)
      const base = initialState(t0, y0, tBound, 0, c, 2)
      const hAbs =
        options.firstStep === undefined
          ? selectInitialStep(tally(f, c), t0, y0, tBound, maxStep, fy, base.direction, 2, rtol, atol)
          : validFirstStep(options.firstStep, t0, tBound)
      return { ...base, hAbs, f, fy, k1: fy, k2: fy, h: 0, counts: snapshot(c) }
    },
    step: (s) => {
      if (s.status !== 'running') throw new OmaiError('value', 'rosenbrock23: the integration has ended')
      if (atBound(s)) return { ...s, tOld: s.t, t: s.tBound, status: 'finished', attempts: [] }
      const c: Counter = { ...s.counts }
      const f = tally(s.f, c)
      const shape = s.y.shape
      const n = numbersOf(s.y).length
      const m = checkMass(mass, n)
      const flat = (v: Value): Value => reshape(v, [n])
      const back = (v: Value): Value => reshape(v, shape)
      const J = jacobianValue(s.f, s.t, s.y, jac)
      c.njev += 1
      const delta = Math.sqrt(EPS) * Math.max(Math.abs(s.t), 1)
      const T = combine(null, [
        [1 / delta, f(s.t + delta, s.y)],
        [-1 / delta, s.fy],
      ])
      const hmin = minStep(s.t, s.direction)
      let hAbs = Math.min(Math.max(s.hAbs, hmin), maxStep)
      const attempts: StepAttempt[] = []
      let rejected = false
      for (let tries = 0; tries < 50; tries++) {
        if (hAbs < hmin) {
          return { ...s, status: 'failed', message: 'rosenbrock23: step size underflow', attempts, counts: snapshot(c) }
        }
        const { tNew, h } = towards(s.t, hAbs, s.direction, s.tBound)
        hAbs = Math.abs(h)
        const eye = fromNumbers(m ?? identityNumbers(n), [n, n])
        const W = sub(eye, mul(J, h * ROS_D))
        c.nlu += 1
        const M = (v: Value): Value => massTimes(m, v, n)
        const k1 = back(solve(W, flat(combine(s.fy, [[h * ROS_D, T]]))))
        const f1 = f(s.t + h / 2, combine(s.y, [[h / 2, k1]]))
        const k2 = combine(back(solve(W, flat(combine(f1, [[-1, M(k1)]])))), [[1, k1]])
        const yNew = combine(s.y, [[h, k2]])
        const f2 = f(tNew, yNew)
        const k3 = back(
          solve(
            W,
            flat(
              combine(f2, [
                [-ROS_E32, M(k2)],
                [ROS_E32, f1],
                [-2, M(k1)],
                [2, s.fy],
                [h * ROS_D, T],
              ]),
            ),
          ),
        )
        const errV = numbersOf(
          combine(null, [
            [h / 6, k1],
            [-h / 3, k2],
            [h / 6, k3],
          ]),
        )
        const err = scaledRms(errV, errorScale(numbersOf(s.y), numbersOf(yNew), rtol, atol))
        if (err <= 1) {
          let factor = err === 0 ? 5 : Math.min(5, 0.9 * err ** (-1 / 3))
          if (rejected) factor = Math.min(1, factor)
          attempts.push(attempt(h, err, true))
          return {
            ...s,
            tOld: s.t,
            yOld: s.y,
            t: tNew,
            y: yNew,
            fy: f2,
            k1,
            k2,
            h,
            hAbs: hAbs * factor,
            attempts,
            status: statusAfter(tNew, s.tBound, s.direction),
            counts: snapshot(c),
          }
        }
        attempts.push(attempt(h, err, false))
        hAbs *= Number.isFinite(err) ? Math.max(0.2, 0.9 * err ** (-1 / 3)) : 0.2
        rejected = true
      }
      return {
        ...s,
        status: 'failed',
        message: 'rosenbrock23: too many step rejections',
        attempts,
        counts: snapshot(c),
      }
    },
    dense: (s) => ({
      t0: s.tOld,
      t1: s.t,
      at: (t) => {
        const x = s.h === 0 ? 0 : (t - s.tOld) / s.h
        const den = 1 - 2 * ROS_D
        return combine(s.yOld, [
          [(s.h * x * (1 - x)) / den, s.k1],
          [(s.h * x * (x - 2 * ROS_D)) / den, s.k2],
        ])
      },
    }),
  }
}

/** df/dy as a value: traced (by forward mode) when the state or the field is traced, else concrete numbers. */
function jacobianValue(f: OdeVectorField, t: number, y: Value, jac: JacobianOption): Value {
  const n = numbersOf(y).length
  if (typeof jac === 'function') return reshape(jac(t, y), [n, n])
  const probe = f(t, y)
  if (isTracedState(y, probe) && jac === 'autodiff') {
    const g = (v: Value): Value => reshape(f(t, reshape(v, y.shape)), [n])
    return reshape(jacfwd(g)(reshape(y, [n])), [n, n])
  }
  return fromNumbers(jacobianAt(f, t, y, jac), [n, n])
}

/** The eigenvalue picture of df/dy at a point: stiffness ratio and spectral radius. */
export interface StiffnessReport {
  /** max |Re lambda| / min |Re lambda| over eigenvalues with negative real part (1 when fewer than two). */
  readonly ratio: number
  /** The largest |lambda|. */
  readonly spectralRadius: number
  /** h times the spectral radius, for a step h. */
  readonly hRho: number
  /** Whether h rho exceeds the explicit method's stability boundary (stiff for that method at that step). */
  readonly stiff: boolean
  readonly eigenvalues: { readonly re: readonly number[]; readonly im: readonly number[] }
}

/**
 * Stiffness detection from the Jacobian's eigenvalues: the stiffness ratio, the spectral radius rho and whether a
 * step h is limited by stability rather than accuracy for an explicit method (h rho beyond its stability boundary,
 * about 3.3 for Dormand-Prince on the negative real axis). This is the test LSODA-style switching uses.
 *
 * @param f - The vector field.
 * @param t - Time.
 * @param y - State.
 * @param options - Step h (default 1), stability boundary (default 3.3) and Jacobian option.
 * @returns The report.
 * @concept stiffness
 * @example
 * ```ts
 * import { stiffness } from 'omai-engine/dynamics/ode'
 * import { matmul, tensor } from 'omai-engine/tensor'
 * const a = tensor([[-1, 0], [0, -1000]])
 * const r = stiffness((_t, y) => matmul(a, y), 0, tensor([1, 1]), { h: 0.01 })
 * [r.ratio, r.stiff] // => [1000, true]
 * ```
 */
export function stiffness(
  f: OdeVectorField,
  t: number,
  y: TensorLike,
  options: { readonly h?: number; readonly boundary?: number; readonly jac?: JacobianOption } = {},
): StiffnessReport {
  const yv = floatState(y)
  const n = numbersOf(yv).length
  const J = jacobianAt(f, t, yv, options.jac ?? 'autodiff')
  const ev = eigvals(fromNumbers(J, [n, n]))
  const re = Array.from(ev.re.data)
  const im = Array.from(ev.im.data)
  const neg = re.filter((v) => v < 0).map((v) => Math.abs(v))
  const ratio = neg.length < 2 ? 1 : maxOf(neg) / minOf(neg)
  const rho = maxOf(re.map((v, i) => Math.hypot(v, im[i] ?? 0)))
  const hRho = (options.h ?? 1) * rho
  return { ratio, spectralRadius: rho, hRho, stiff: hRho > (options.boundary ?? 3.3), eigenvalues: { re, im } }
}
