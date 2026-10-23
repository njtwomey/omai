import {
  addDecayedWeights,
  chain,
  type GradientTransformation,
  type LearningRate,
  scaleByAdagrad,
  scaleByLearningRate,
  scaleByRms,
  traceMomentum,
} from './transform.ts'

/** Options of {@link sgd}. */
export interface SgdOptions {
  readonly learningRate: LearningRate
  readonly momentum?: number
  readonly nesterov?: boolean
  /** L2 penalty added to the gradient (torch's weight_decay). */
  readonly weightDecay?: number
}

/**
 * Stochastic gradient descent with optional momentum, Nesterov momentum and L2 weight decay (torch.optim.SGD with no
 * dampening).
 *
 * @param options - Learning rate, momentum, Nesterov and weight decay.
 * @returns The optimiser.
 * @concept optimisers
 * @example
 * ```ts
 * import { sgd, applyUpdates } from 'whitebox-engine/optim'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * const opt = sgd({ learningRate: 0.1 })
 * const { updates } = opt.update(tensor([1, -2]), opt.init(tensor([0, 0])), tensor([0, 0]))
 * toArray(updates as never) // => [-0.1, 0.2]
 * ```
 */
export function sgd(options: SgdOptions): GradientTransformation<unknown[]> {
  return chain(
    addDecayedWeights(options.weightDecay ?? 0),
    ...((options.momentum ?? 0) > 0 ? [traceMomentum(options.momentum ?? 0, options.nesterov ?? false)] : []),
    scaleByLearningRate(options.learningRate),
  )
}

/** Options of {@link rmsprop}. */
export interface RmspropOptions {
  readonly learningRate: LearningRate
  readonly decay?: number
  readonly eps?: number
  readonly momentum?: number
  readonly weightDecay?: number
}

/**
 * RMSprop (torch.optim.RMSprop, uncentred): g / (sqrt(v) + eps), with optional momentum and L2 weight decay.
 *
 * @param options - Learning rate, decay (alpha), eps, momentum and weight decay.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { rmsprop } from 'whitebox-engine/optim'
 * rmsprop({ learningRate: 0.01 }).name // => 'chain(addDecayedWeights, scaleByRms, scaleByLearningRate)'
 * ```
 */
export function rmsprop(options: RmspropOptions): GradientTransformation<unknown[]> {
  return chain(
    addDecayedWeights(options.weightDecay ?? 0),
    scaleByRms(options.decay ?? 0.99, options.eps ?? 1e-8),
    ...((options.momentum ?? 0) > 0 ? [traceMomentum(options.momentum ?? 0)] : []),
    scaleByLearningRate(options.learningRate),
  )
}

/** Options of {@link adagrad}. */
export interface AdagradOptions {
  readonly learningRate: LearningRate
  readonly eps?: number
  readonly initialAccumulator?: number
  readonly weightDecay?: number
}

/**
 * Adagrad (torch.optim.Adagrad with no learning-rate decay).
 *
 * @param options - Learning rate, eps, initial accumulator and weight decay.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { adagrad } from 'whitebox-engine/optim'
 * adagrad({ learningRate: 0.1 }).name // => 'chain(addDecayedWeights, scaleByAdagrad, scaleByLearningRate)'
 * ```
 */
export function adagrad(options: AdagradOptions): GradientTransformation<unknown[]> {
  return chain(
    addDecayedWeights(options.weightDecay ?? 0),
    scaleByAdagrad(options.eps ?? 1e-10, options.initialAccumulator ?? 0),
    scaleByLearningRate(options.learningRate),
  )
}
