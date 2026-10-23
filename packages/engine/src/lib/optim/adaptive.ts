import {
  addDecayedWeights,
  chain,
  type GradientTransformation,
  type LearningRate,
  scaleByAdam,
  scaleByLearningRate,
} from './transform.ts'

/** Options of {@link adam} and {@link adamw}. */
export interface AdamOptions {
  readonly learningRate: LearningRate
  readonly b1?: number
  readonly b2?: number
  readonly eps?: number
  readonly amsgrad?: boolean
  readonly weightDecay?: number
}

/**
 * Adam (torch.optim.Adam): bias-corrected moment estimates; `weightDecay` is an L2 penalty added to the gradient.
 *
 * @param options - Learning rate, betas, eps, AMSGrad and weight decay.
 * @returns The optimiser.
 * @concept optimisers
 * @example
 * ```ts
 * import { adam } from 'whitebox-engine/optim'
 * adam({ learningRate: 1e-3 }).name // => 'chain(addDecayedWeights, scaleByAdam, scaleByLearningRate)'
 * ```
 */
export function adam(options: AdamOptions): GradientTransformation<unknown[]> {
  return chain(
    addDecayedWeights(options.weightDecay ?? 0),
    scaleByAdam(options.b1 ?? 0.9, options.b2 ?? 0.999, options.eps ?? 1e-8, options.amsgrad ?? false),
    scaleByLearningRate(options.learningRate),
  )
}

/**
 * AdamW (torch.optim.AdamW, Loshchilov and Hutter): Adam with decoupled weight decay (default 0.01), applied to the
 * parameters rather than added to the gradient.
 *
 * @param options - Learning rate, betas, eps, AMSGrad and weight decay.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { adamw } from 'whitebox-engine/optim'
 * adamw({ learningRate: 1e-3 }).name // => 'chain(scaleByAdam, addDecayedWeights, scaleByLearningRate)'
 * ```
 */
export function adamw(options: AdamOptions): GradientTransformation<unknown[]> {
  return chain(
    scaleByAdam(options.b1 ?? 0.9, options.b2 ?? 0.999, options.eps ?? 1e-8, options.amsgrad ?? false),
    addDecayedWeights(options.weightDecay ?? 0.01),
    scaleByLearningRate(options.learningRate),
  )
}
