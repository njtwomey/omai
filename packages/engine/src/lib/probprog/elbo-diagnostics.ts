/**
 * Variance tooling for ELBO gradient estimators: the mean and the per-coordinate variance of an estimator's gradient over
 * repeated draws, the total variance and the signal-to-noise ratio that follow, and the ELBO's own mean and sd. It is how
 * an estimator is compared with another (the score-function estimator against its Rao-Blackwellised form, a baseline
 * against none) and how a method puts `svi/gradVariance` on its tape.
 */

import { valueAndGrad } from '../autodiff/index.ts'
import { OmaiError } from '../base/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import {
  asTensor,
  concat,
  div,
  item,
  primalOf,
  reshape,
  square,
  stack,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import type { ElboEstimator } from './elbo.ts'
import type { Model } from './primitives.ts'

/** What {@link gradientMoments} measured. */
export interface GradientMoments {
  /** The mean gradient of the loss (minus the ELBO's gradient) per parameter, flattened. */
  readonly mean: Readonly<Record<string, Float64Array>>
  /** The unbiased sample variance of each coordinate of the gradient. */
  readonly variance: Readonly<Record<string, Float64Array>>
  /** The sum of the coordinates' variances: the expected squared distance of one gradient from the mean. */
  readonly totalVariance: number
  /** The squared norm of the mean gradient over the total variance (large: the gradient is mostly signal). */
  readonly signalToNoise: number
  readonly elboMean: number
  readonly elboSd: number
  readonly repeats: number
}

/** Options of {@link gradientMoments}. */
export interface GradientMomentsOptions<D> {
  readonly estimator: ElboEstimator
  readonly model: Model<D>
  readonly guide: Model<D>
  readonly data: D
  /** The point at which the gradient is taken (the values the guide and model read as parameters). */
  readonly params: Readonly<Record<string, TensorLike>>
  /** The key of repeat `i` is `foldIn(key, i)`. */
  readonly key: Key
  /** Number of estimates (at least 2). */
  readonly repeats: number
  /** The estimator's state, the same at every repeat (default: the state after one estimate from `init()`). */
  readonly state?: unknown
}

/**
 * The first two moments of an ELBO estimator's gradient at fixed parameters: draw `repeats` estimates under keys derived
 * from one key and difference each with respect to the parameters. The state is the same at every repeat (a baseline does
 * not move while it is measured), so the result is the variance of the estimator as it would behave at that state.
 *
 * @param options - Estimator, programs, parameters, key and repeats.
 * @returns The mean, the variance, the total variance, the signal-to-noise ratio and the ELBO's mean and sd.
 * @throws OmaiError with code `value` for fewer than two repeats.
 * @concept variational-inference
 * @example
 * ```ts
 * import { autoNormal, gradientMoments, observe, sample, traceElbo } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * const r = gradientMoments({ estimator: traceElbo(), model, guide: autoNormal(model), data: undefined, params: { m_auto_loc: 0, m_auto_scale: 1 }, key: key(0), repeats: 20 })
 * r.totalVariance > 0 // => true
 * ```
 */
export function gradientMoments<D = void>(options: GradientMomentsOptions<D>): GradientMoments {
  const { estimator, model, guide, data, params, key, repeats } = options
  if (!(Number.isInteger(repeats) && repeats >= 2))
    throw new OmaiError('value', 'gradientMoments: repeats must be at least 2')
  const names = Object.keys(params)
  const start = Object.fromEntries(names.map((k) => [k, primalOf(asTensor(params[k] as TensorLike))]))
  // Default: the state after one estimate, which sets up what an estimator caches (a dependency graph) and starts its baselines.
  const state =
    options.state ?? estimator.estimate(foldIn(key, 0xfffffff0), model, guide, data, start, estimator.init()).state
  const grads: Record<string, Value[]> = Object.fromEntries(names.map((k) => [k, []]))
  const elbos: number[] = []
  for (let i = 0; i < repeats; i++) {
    let elbo = Number.NaN
    const { grad } = valueAndGrad((p: Record<string, Value>) => {
      const r = estimator.estimate(foldIn(key, i), model, guide, data, p, state)
      elbo = r.elbo
      return r.loss
    })(start as Record<string, Value>)
    for (const k of names) grads[k]?.push(reshape(primalOf((grad as Record<string, Value>)[k] as Value), [-1]))
    elbos.push(elbo)
  }
  const mean: Record<string, Float64Array> = {}
  const variance: Record<string, Float64Array> = {}
  const variances: Value[] = []
  const means: Value[] = []
  for (const k of names) {
    // One row per repeat, one column per coordinate of the parameter.
    const g = stack(grads[k] as Value[])
    const m = div(sum(g, 0), repeats)
    const v = div(sum(square(sub(g, m)), 0), repeats - 1)
    means.push(m)
    variances.push(v)
    mean[k] = Float64Array.from(toFlat(m))
    variance[k] = Float64Array.from(toFlat(v))
  }
  // The totals run over every coordinate of every parameter, in order.
  const total = names.length === 0 ? 0 : item(sum(concat(variances)))
  const norm2 = names.length === 0 ? 0 : item(sum(square(concat(means))))
  const e = asTensor(elbos)
  const eMean = item(div(sum(e), repeats))
  const eVar = item(div(sum(square(sub(e, eMean))), repeats - 1))
  return {
    mean,
    variance,
    totalVariance: total,
    signalToNoise: total === 0 ? Number.POSITIVE_INFINITY : norm2 / total,
    elboMean: eMean,
    elboSd: Math.sqrt(eVar),
    repeats,
  }
}
