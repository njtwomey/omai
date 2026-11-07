/**
 * A flow as a stepped algorithm whose state serialises: a vector field integrated at a fixed step by an explicit
 * Runge-Kutta method, one step per algorithm step, with a hook for the quantities a system is watched by (an energy, a
 * Jacobi constant, an order parameter), logged in the same step. Unlike the
 * steppers of `solveIvp` (which keep the field in their state), the state here is plain numbers, so a run checkpoints,
 * seeks and resumes, and a session can fork it at any step.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import type { TensorLike } from '../../tensor/index.ts'
import { type Algorithm, type RunStatus, type StepContext, stateFromJson, stateToJson } from '../../trace/index.ts'
import { floatState, fromNumbers, numbersOf } from './_vec.ts'
import { type ButcherTableau, RK4, rkStep } from './explicit.ts'
import type { OdeVectorField } from './problem.ts'

/** The state of {@link flowAlgorithm}. */
export interface FlowState {
  readonly t: number
  /** Steps taken. */
  readonly step: number
  /** The state, flat. */
  readonly y: Float64Array
}

/** Options of {@link flowAlgorithm}. */
export interface FlowOptions<K extends string> {
  /** The step. */
  readonly dt: number
  /** The steps to take. */
  readonly steps: number
  /** The Runge-Kutta method (default RK4). */
  readonly method?: ButcherTableau
  /** The start time (default 0). */
  readonly t0?: number
  /** The shape of the state (default a vector). */
  readonly shape?: readonly number[]
  /** The name of the algorithm (default `flow`). */
  readonly name?: string
  /** What the algorithm's configuration says, beside the step and method. */
  readonly config?: { readonly [key: string]: Json }
  /** Called after each step with the new time and state, to log the system's own paths. */
  readonly observe?: (t: number, y: Float64Array, ctx: StepContext<K>) => void
  /** A status to stop with (a capture, an escape), or undefined to continue. */
  readonly stop?: (t: number, y: Float64Array) => RunStatus | undefined
}

/**
 * Integrate a vector field at a fixed step as a stepped algorithm: the state is flat numbers (so it serialises), each
 * step is one step of the Runge-Kutta method, and whatever `observe` logs goes on the tape (the paths are the caller's,
 * typed by `K`).
 *
 * @param field - The vector field dy/dt = f(t, y).
 * @param y0 - The initial state.
 * @param options - Step, step count, method, the observer and a stopping rule.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a non-positive step.
 * @concept runge-kutta
 * @example
 * ```ts
 * import { flowAlgorithm } from 'mdml-engine/dynamics/ode'
 * import { neg } from 'mdml-engine/tensor'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = flowAlgorithm((_t, y) => neg(y), [1], { dt: 0.1, steps: 10 })
 * Math.abs((runAlgorithm(alg, { maxSteps: 100 }).state.y[0] ?? 0) - Math.exp(-1)) < 1e-6 // => true
 * ```
 */
export function flowAlgorithm<K extends string = string>(
  field: OdeVectorField,
  y0: TensorLike,
  options: FlowOptions<K>,
): Algorithm<FlowState, K> {
  const { dt, steps } = options
  if (!(dt > 0)) throw new MdmlError('value', 'flowAlgorithm: dt must be positive')
  const method = options.method ?? RK4
  const start = floatState(y0)
  const shape = options.shape ?? start.shape
  const t0 = options.t0 ?? 0
  return {
    name: options.name ?? 'flow',
    config: { ...(options.config ?? {}), dt, steps, method: method.name, t0 },
    init: () => ({ t: t0, step: 0, y: numbersOf(start).slice() }),
    step: (s, ctx) => {
      const next = numbersOf(rkStep(field, s.t, fromNumbers(s.y, shape), dt, method)).slice()
      const step = s.step + 1
      const t = t0 + step * dt
      options.observe?.(t, next, ctx)
      return { t, step, y: next }
    },
    done: (s) => (s.step >= steps ? 'completed' : options.stop?.(s.t, s.y)),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as FlowState,
  }
}
