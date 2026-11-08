/**
 * Gradient descent driven by an optax-style transformation, over an objective that may be summed over a batch of
 * examples: the loop of every stochastic learner. Each step applies the optimiser's update at the scheduled learning
 * rate, then draws the next batch from the step's key (without replacement, so the same run draws the same batches
 * however it was paused or sought) and evaluates the objective and gradient there. The state is the iterate, its
 * objective and gradient, the optimiser's own state and the step count, all of which serialise. The gradient tolerance
 * is tested on the gradient of the whole objective, which with a batch is evaluated once per epoch (never on the
 * batch's gradient, which is exactly zero whenever the examples it drew have no loss).
 */

import type { Json } from '../base/index.ts'
import { OmaiError } from '../base/index.ts'
import { daxpy, dnrm2 } from '../numerics/linalg/index.ts'
import { type Key, subsample } from '../random/index.ts'
import { asTensor, primalOf, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { sgd } from './first-order.ts'
import { observeVector, type VectorState, vectorDone, vectorSchema } from './minimize/_shared.ts'
import { flatObjective, infNorm, type Parameters } from './minimize/objective.ts'
import type { OptimiserShape } from './optimiser-table.ts'
import type { ScheduleShape } from './schedule-table.ts'
import { constantSchedule, type LearningRateSchedule } from './schedules.ts'
import type { ParamTree } from './transform.ts'

/**
 * An objective that can be summed over a batch: a scalar function of the parameters and, optionally, the indices of
 * the examples to use (the mean over those examples, so batches of different sizes are comparable). Called without
 * a batch it is the objective over the whole set; an objective of the parameters alone is one.
 */
export type BatchObjective = { bivariant(params: ParamTree, batch?: readonly number[]): Value }['bivariant']

/** The examples an objective is summed over, when it can be summed over a part of them. */
export interface Batches {
  /** The number of examples. */
  readonly n: number
}

/** Options of {@link gradientAlgorithm}. */
export interface GradientOptions {
  /** The optimiser, as a shape that takes the learning rate (default SGD without momentum). */
  readonly optimiser?: OptimiserShape
  /** The learning rate, the peak of the schedule (default 0.01). */
  readonly learningRate?: number
  /** The schedule over steps, as a shape that takes the peak (default constant). */
  readonly schedule?: ScheduleShape
  /** Examples per step, or null for the whole set (default). Needs `batches`. */
  readonly batch?: number | null
  /** How many examples the objective sums over; required with a `batch`. */
  readonly batches?: Batches
  /** Stop with status `converged` once the infinity norm of the gradient falls to this (default 1e-5). */
  readonly gradTol?: number
}

/** The state of a {@link gradientAlgorithm} run: the common vector state, the optimiser's state and the rate used. */
export interface GradientState extends VectorState {
  /** The steps taken (the optimiser's count). */
  readonly t: number
  /** The learning rate of the last step (the schedule at the step before it). */
  readonly lr: number
  /** The optimiser's own state. */
  readonly opt: unknown
}

/**
 * Minimise an objective by an optimiser stepping at a scheduled rate, over the whole set or over a batch drawn afresh
 * at every step. The state is coherent at every step: `value` and `grad` are the objective and gradient at `x` over the
 * batch the step drew (the whole set without a batch), and `step` is the length of the last update.
 *
 * @param objective - The objective, with an optional batch of example indices as its second argument.
 * @param x0 - The start (a vector, or any parameter tree).
 * @param options - The optimiser, learning rate, schedule, batch size and the gradient tolerance.
 * @returns The algorithm; its steps are keyed, so the batches are the same on every run of a key.
 * @throws OmaiError with code `value` for a batch without `batches`, or a batch that is not a positive integer.
 * @concept optimisers
 * @example
 * ```ts
 * import { gradientAlgorithm } from 'omai-engine/optim'
 * import { runAlgorithm } from 'omai-engine/trace'
 * import { square, sub, sum, type Value } from 'omai-engine/tensor'
 * const alg = gradientAlgorithm((x: Value) => sum(square(sub(x, 2))), [0, 0], { learningRate: 0.25 })
 * Array.from(runAlgorithm(alg, { seed: 0, maxSteps: 60 }).state.x, Math.round) // => [2, 2]
 * ```
 */
export function gradientAlgorithm(
  objective: BatchObjective,
  x0: Parameters,
  options: GradientOptions = {},
): Algorithm<GradientState> {
  const peak = options.learningRate ?? 1e-2
  const schedule: LearningRateSchedule = (options.schedule ?? constantSchedule)(peak)
  const make = options.optimiser ?? ((learningRate) => sgd({ learningRate }))
  const optimiser = make(schedule)
  const gtol = options.gradTol ?? 1e-5
  const batch = options.batch ?? null
  const n = options.batches?.n
  if (batch !== null && (!Number.isInteger(batch) || batch < 1))
    throw new OmaiError('value', `gradientAlgorithm: the batch must be a positive integer or null, got ${batch}`)
  if (batch !== null && n === undefined)
    throw new OmaiError('value', 'gradientAlgorithm: a batch needs the problem to say how many examples it sums over')
  const stochastic = batch !== null && n !== undefined && batch < n
  const full = flatObjective(objective, x0)
  const evaluate = (x: Float64Array, k: Key): { value: number; grad: Float64Array } => {
    if (!stochastic) return full.valueAndGrad(x)
    const indices = Array.from(subsample(k, n as number, batch as number).data)
    return flatObjective((p) => objective(p, indices), x0).valueAndGrad(x)
  }
  // The tolerance is a statement about the gradient of the whole objective: a batch whose examples all have zero loss
  // (a satisfied margin, a dead unit) has an exactly zero gradient at a point that is far from a minimum. Without a
  // batch the gradient in hand is the whole one; with one the whole gradient is evaluated once per epoch.
  const epoch = stochastic ? Math.ceil((n as number) / (batch as number)) : 1
  const stopped = (
    x: Float64Array,
    grad: Float64Array,
    t: number,
  ): Pick<GradientState, 'stop'> & { readonly evals: number } => {
    if (stochastic && t % epoch !== 0) return { evals: 0 }
    const whole = stochastic ? full.valueAndGrad(x).grad : grad
    const evals = stochastic ? 1 : 0
    return infNorm(whole) <= gtol
      ? { stop: { status: 'converged', message: 'gradient norm below tolerance' }, evals }
      : { evals }
  }
  return {
    name: 'gradient',
    config: {
      optimiser: optimiser.name,
      learningRate: peak,
      batch,
      gradTol: gtol,
    } as Json,
    schema: vectorSchema({ metrics: { lr: { description: 'learning rate of the last step' } } }),
    init: (k) => {
      const { value, grad } = evaluate(full.x0, k)
      const { evals, ...stop } = stopped(full.x0, grad, 0)
      return {
        x: full.x0,
        value,
        grad,
        nfev: 1 + evals,
        ngev: 1 + evals,
        nhev: 0,
        step: 0,
        t: 0,
        lr: schedule(0),
        opt: optimiser.init(asTensor(full.x0)),
        ...stop,
      }
    },
    step: (s, ctx) => {
      const { updates, state } = optimiser.update(asTensor(s.grad), s.opt, asTensor(s.x))
      const update = toFlat(primalOf(updates as Value))
      const x = daxpy(1, update, s.x)
      const { value, grad } = evaluate(x, ctx.key)
      const { evals, ...stop } = stopped(x, grad, s.t + 1)
      return {
        x,
        value,
        grad,
        nfev: s.nfev + 1 + evals,
        ngev: s.ngev + 1 + evals,
        nhev: 0,
        step: dnrm2(update),
        t: s.t + 1,
        lr: schedule(s.t),
        opt: state,
        ...stop,
      }
    },
    done: vectorDone,
    observe: (s) => observeVector(s, { metrics: { lr: s.lr } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as GradientState,
  }
}
