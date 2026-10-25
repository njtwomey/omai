/**
 * The shared vocabulary of the ODE solvers: vector fields, initial value problems with explicit parameters and an
 * optional mass matrix, event specifications, dense (continuous) solutions built from step interpolants, solve results
 * in scipy's `OdeResult` shape, and the `Stepper` contract every integrator implements.
 */

import { MdmlError } from '../../base/index.ts'
import { stack, type TensorLike, type Value } from '../../tensor/index.ts'
import { combine, floatState } from './_vec.ts'

/** A vector field dy/dt = f(t, y). The state may have any shape; the output has the same shape. */
export type VectorField = (t: number, y: Value) => Value

/** A vector field with explicit parameters, dy/dt = f(t, y, p) (the form the adjoint method differentiates). */
export type ParametricField = (t: number, y: Value, params: Value) => Value

/**
 * An initial value problem M dy/dt = f(t, y, p), y(t0) = y0 on [t0, t1]: the vector field, initial state, time span,
 * parameters (explicit, so the adjoint method can differentiate them), an optional constant mass matrix and an
 * optional Jacobian df/dy.
 */
export interface OdeProblem {
  readonly f: ParametricField
  readonly y0: Value
  readonly tspan: readonly [number, number]
  readonly params: Value | undefined
  /** A constant mass matrix M (`[n, n]` over the flattened state); singular M makes the problem a DAE. */
  readonly mass: Value | undefined
  /** The Jacobian df/dy as `[n, n]` over the flattened state (by automatic differentiation when absent). */
  readonly jac: ((t: number, y: Value, params: Value) => Value) | undefined
}

/** Options of {@link odeProblem}. */
export interface OdeProblemOptions {
  readonly params?: TensorLike
  readonly mass?: TensorLike
  readonly jac?: (t: number, y: Value, params: Value) => Value
}

/**
 * Build an initial value problem. Parameters are passed to the vector field as its third argument, so solutions are
 * differentiable in them by the adjoint method as well as by differentiating through the steps.
 *
 * @param f - The vector field f(t, y, p).
 * @param y0 - The initial state.
 * @param tspan - The interval [t0, t1] (t1 < t0 integrates backwards).
 * @param options - Parameters, mass matrix and Jacobian.
 * @returns The problem.
 * @throws MdmlError with code `value` when the time span is not finite.
 * @concept initial-value-problem
 * @example
 * ```ts
 * import { odeProblem } from 'mdml-engine/dynamics/ode'
 * import { mul, neg } from 'mdml-engine/tensor'
 * const p = odeProblem((_t, y, k) => neg(mul(k, y)), [1], [0, 1], { params: 2 })
 * p.tspan // => [0, 1]
 * ```
 */
export function odeProblem(
  f: ParametricField,
  y0: TensorLike,
  tspan: readonly [number, number],
  options: OdeProblemOptions = {},
): OdeProblem {
  if (!Number.isFinite(tspan[0]) || !Number.isFinite(tspan[1])) {
    throw new MdmlError('value', 'odeProblem: the time span must be finite')
  }
  return {
    f,
    y0: floatState(y0),
    tspan: [tspan[0], tspan[1]],
    params: options.params === undefined ? undefined : floatState(options.params),
    mass: options.mass === undefined ? undefined : floatState(options.mass),
    jac: options.jac,
  }
}

/**
 * The vector field of a problem with its parameters bound: (t, y) => f(t, y, p).
 *
 * @param problem - The problem.
 * @param params - Parameters to use instead of the problem's (a traced value, say).
 * @returns The field.
 * @example
 * ```ts
 * import { fieldOf, odeProblem } from 'mdml-engine/dynamics/ode'
 * import { mul, neg, tensor, toArray } from 'mdml-engine/tensor'
 * const p = odeProblem((_t, y, k) => neg(mul(k, y)), [1], [0, 1], { params: 2 })
 * toArray(fieldOf(p)(0, tensor([3]))) // => [-6]
 * ```
 */
export function fieldOf(problem: OdeProblem, params?: Value): VectorField {
  const p = params ?? problem.params ?? floatState(0)
  return (t, y) => problem.f(t, y, p)
}

/**
 * An event: a scalar function g(t, y) whose zero crossings are located (scipy's events). `direction` restricts the
 * crossings counted (+1 from negative to positive, -1 the reverse, 0 both); a terminal event stops the integration at
 * its first (or `terminal`-th) occurrence.
 */
export interface EventSpec {
  readonly fn: (t: number, y: Value) => number
  readonly terminal?: boolean | number
  readonly direction?: -1 | 0 | 1
  readonly name?: string
}

/**
 * An event specification.
 *
 * @param fn - The event function g(t, y); events are its zeros.
 * @param options - Whether it is terminal (or after how many occurrences), the crossing direction and a name.
 * @returns The event.
 * @example
 * ```ts
 * import { event } from 'mdml-engine/dynamics/ode'
 * import { item, sliceAxis } from 'mdml-engine/tensor'
 * event((_t, y) => item(sliceAxis(y, 0, 0, 1)), { terminal: true, direction: -1 }).direction // => -1
 * ```
 */
export function event(
  fn: (t: number, y: Value) => number,
  options: { readonly terminal?: boolean | number; readonly direction?: -1 | 0 | 1; readonly name?: string } = {},
): EventSpec {
  return { fn, ...options }
}

/** A continuous interpolant of one step, valid on [t0, t1] (in either order). */
export interface DenseSegment {
  readonly t0: number
  readonly t1: number
  at(t: number): Value
}

/** A continuous solution made of the interpolants of every step (scipy's `OdeSolution`). */
export interface DenseSolution {
  readonly tMin: number
  readonly tMax: number
  readonly segments: readonly DenseSegment[]
  /** The solution at one time (extrapolating the nearest segment outside the range). */
  at(t: number): Value
  /** The solution at several times, stacked on a new first axis. */
  many(ts: readonly number[]): Value
}

/**
 * Join step interpolants into one continuous solution.
 *
 * @param segments - The interpolants, in integration order.
 * @returns The continuous solution.
 * @throws MdmlError with code `value` when there are no segments.
 * @example
 * ```ts
 * import { denseSolution, hermiteSegment } from 'mdml-engine/dynamics/ode'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const s = denseSolution([hermiteSegment(0, tensor([0]), tensor([0]), 1, tensor([1]), tensor([2]))])
 * toArray(s.at(1)) // => [1]
 * ```
 */
export function denseSolution(segments: readonly DenseSegment[]): DenseSolution {
  const first = segments[0]
  const last = segments[segments.length - 1]
  if (first === undefined || last === undefined) throw new MdmlError('value', 'denseSolution: no segments')
  const ascending = first.t1 >= first.t0
  const lo = ascending ? first.t0 : last.t1
  const hi = ascending ? last.t1 : first.t0
  const find = (t: number): DenseSegment => {
    // Binary search over segment ends in integration order.
    let a = 0
    let b = segments.length - 1
    while (a < b) {
      const m = (a + b) >> 1
      const s = segments[m] as DenseSegment
      const beyond = ascending ? t > s.t1 : t < s.t1
      if (beyond) a = m + 1
      else b = m
    }
    return segments[a] as DenseSegment
  }
  return {
    tMin: lo,
    tMax: hi,
    segments,
    at: (t) => find(t).at(t),
    many: (ts) => stack(ts.map((t) => find(t).at(t))),
  }
}

/**
 * The cubic Hermite interpolant of a step from its end values and slopes (third-order dense output for methods
 * without a native interpolant).
 *
 * @param t0 - Start time.
 * @param y0 - State at t0.
 * @param f0 - Derivative at t0.
 * @param t1 - End time.
 * @param y1 - State at t1.
 * @param f1 - Derivative at t1.
 * @returns The interpolant.
 * @example
 * ```ts
 * import { hermiteSegment } from 'mdml-engine/dynamics/ode'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * // y = t^2 on [0, 1] is reproduced exactly.
 * toArray(hermiteSegment(0, tensor([0]), tensor([0]), 1, tensor([1]), tensor([2])).at(0.5)) // => [0.25]
 * ```
 */
export function hermiteSegment(t0: number, y0: Value, f0: Value, t1: number, y1: Value, f1: Value): DenseSegment {
  const h = t1 - t0
  return {
    t0,
    t1,
    at: (t) => {
      const s = h === 0 ? 0 : (t - t0) / h
      const s2 = s * s
      const s3 = s2 * s
      return combine(null, [
        [2 * s3 - 3 * s2 + 1, y0],
        [h * (s3 - 2 * s2 + s), f0],
        [-2 * s3 + 3 * s2, y1],
        [h * (s3 - s2), f1],
      ])
    },
  }
}

/** Evaluation counts of a solve. */
export interface SolveCounts {
  /** Vector field evaluations. */
  readonly nfev: number
  /** Jacobian evaluations. */
  readonly njev: number
  /** LU decompositions. */
  readonly nlu: number
}

/** How a solve ended. */
export type OdeStatus = 'success' | 'event' | 'failed'

/** The solution of an initial value problem (scipy's `OdeResult`, time-major). */
export interface OdeSolution extends SolveCounts {
  /** Output times. */
  readonly t: readonly number[]
  /** States stacked on a new first axis: `[nt, ...stateShape]`. */
  readonly y: Value
  /** The states at each output time. */
  readonly ys: readonly Value[]
  /** The continuous solution, when dense output was requested. */
  readonly sol: DenseSolution | undefined
  /** Event times, one list per event. */
  readonly tEvents: readonly (readonly number[])[]
  /** States at the events, one list per event. */
  readonly yEvents: readonly (readonly Value[])[]
  readonly status: OdeStatus
  readonly success: boolean
  readonly message: string
  /** Accepted steps. */
  readonly nSteps: number
  /** Rejected step attempts. */
  readonly nRejected: number
}

/** One attempt at a step: its size, scaled error norm (1 is the tolerance) and whether it was accepted. */
export interface StepAttempt {
  readonly h: number
  readonly error: number
  readonly accepted: boolean
}

/** Whether a stepper can continue. */
export type StepperStatus = 'running' | 'finished' | 'failed'

/** The state every stepper carries: where it is, where it was, its next step size and its counts. */
export interface StepperState {
  readonly t: number
  readonly y: Value
  readonly tOld: number
  readonly yOld: Value
  readonly tBound: number
  readonly direction: 1 | -1
  /** The magnitude of the next step to try. */
  readonly hAbs: number
  readonly status: StepperStatus
  readonly message: string
  /** The attempts made by the last step (rejections first, then the accepted one). */
  readonly attempts: readonly StepAttempt[]
  readonly counts: SolveCounts
  /** The method's order in use (variable-order methods change it). */
  readonly order: number
}

/** Tolerances and step limits shared by the adaptive steppers. */
export interface StepperOptions {
  readonly rtol?: number | readonly number[]
  readonly atol?: number | readonly number[]
  readonly maxStep?: number
  readonly firstStep?: number
}

/**
 * A one-step-at-a-time integrator (scipy's `OdeSolver`): `init` sets up at (t0, y0) towards `tBound`, `step` advances
 * by one accepted step (or fails with a message), and `dense` interpolates the last step.
 */
export interface Stepper<S extends StepperState = StepperState> {
  readonly name: string
  readonly order: number
  /** Whether the method is meant for stiff problems. */
  readonly stiff: boolean
  init(f: VectorField, t0: number, y0: Value, tBound: number): S
  step(state: S): S
  dense(state: S): DenseSegment
}
