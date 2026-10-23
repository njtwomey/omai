import type { Direction, MetricSpec } from '../base/index.ts'
import { type Metric, type MetricOptions, perOutputMetric, weightedMean, weightedPercentile } from './metric.ts'

const map = (n: number, f: (i: number) => number): Float64Array => Float64Array.from({ length: n }, (_, i) => f(i))

const spec = (name: string, direction: Direction, range: readonly [number, number]): MetricSpec => ({
  name,
  direction,
  range,
  inputs: 'values',
  category: 'regression',
  concept: 'regression-metrics',
})

/** Options of {@link meanTweedieDeviance}. */
export interface TweedieOptions extends MetricOptions {
  /** The Tweedie power: 0 normal, 1 Poisson, 2 gamma, 3 inverse Gaussian; (1, 2) compound Poisson-gamma. */
  readonly power?: number
}

/** The unit Tweedie deviance of one observation (sklearn's _mean_tweedie_deviance terms). */
export function tweedieUnitDeviance(y: number, mu: number, power: number): number {
  if (power === 0) return (y - mu) ** 2
  if (power === 1) return 2 * ((y > 0 ? y * Math.log(y / mu) : 0) - y + mu)
  if (power === 2) return 2 * (Math.log(mu / y) + y / mu - 1)
  if (power < 0) {
    const yp = Math.max(y, 0)
    return (
      2 *
      (yp ** (2 - power) / ((1 - power) * (2 - power)) -
        (y * mu ** (1 - power)) / (1 - power) +
        mu ** (2 - power) / (2 - power))
    )
  }
  return (
    2 *
    (y ** (2 - power) / ((1 - power) * (2 - power)) -
      (y * mu ** (1 - power)) / (1 - power) +
      mu ** (2 - power) / (2 - power))
  )
}

/**
 * Mean Tweedie deviance (sklearn.metrics.mean_tweedie_deviance).
 *
 * @param yTrue - Targets.
 * @param yPred - Predicted means (positive).
 * @param options - The power (default 0) and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { meanTweedieDeviance } from 'whitebox-engine/metrics'
 * meanTweedieDeviance([1, 2], [1, 2], { power: 1.5 }).estimate.value // => 0
 * ```
 */
export const meanTweedieDeviance: Metric<TweedieOptions> = perOutputMetric<TweedieOptions>(
  spec('meanTweedieDeviance', 'lower', [0, Infinity]),
  (t, p, w, o) =>
    weightedMean(
      map(t.length, (i) => tweedieUnitDeviance(t[i] ?? 0, p[i] ?? 0, o.power ?? 0)),
      w,
    ),
)

/**
 * Mean Poisson deviance (sklearn.metrics.mean_poisson_deviance), Tweedie power 1.
 *
 * @param yTrue - Counts.
 * @param yPred - Predicted means.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { meanPoissonDeviance } from 'whitebox-engine/metrics'
 * meanPoissonDeviance([0, 2], [1, 2]).estimate.value // => 1
 * ```
 */
export const meanPoissonDeviance: Metric = perOutputMetric(
  spec('meanPoissonDeviance', 'lower', [0, Infinity]),
  (t, p, w) =>
    weightedMean(
      map(t.length, (i) => tweedieUnitDeviance(t[i] ?? 0, p[i] ?? 0, 1)),
      w,
    ),
)

/**
 * Mean gamma deviance (sklearn.metrics.mean_gamma_deviance), Tweedie power 2.
 *
 * @param yTrue - Positive targets.
 * @param yPred - Predicted means.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { meanGammaDeviance } from 'whitebox-engine/metrics'
 * meanGammaDeviance([2], [2]).estimate.value // => 0
 * ```
 */
export const meanGammaDeviance: Metric = perOutputMetric(spec('meanGammaDeviance', 'lower', [0, Infinity]), (t, p, w) =>
  weightedMean(
    map(t.length, (i) => tweedieUnitDeviance(t[i] ?? 0, p[i] ?? 0, 2)),
    w,
  ),
)

/** Options of {@link meanPinballLoss}. */
export interface PinballOptions extends MetricOptions {
  /** The quantile level (default 0.5). */
  readonly alpha?: number
}

/**
 * Mean pinball loss (sklearn.metrics.mean_pinball_loss).
 *
 * @param yTrue - Targets.
 * @param yPred - Predicted quantiles.
 * @param options - The level alpha and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { meanPinballLoss } from 'whitebox-engine/metrics'
 * meanPinballLoss([1], [0], { alpha: 0.9 }).estimate.value // => 0.9
 * ```
 */
export const meanPinballLoss: Metric<PinballOptions> = perOutputMetric<PinballOptions>(
  spec('meanPinballLoss', 'lower', [0, Infinity]),
  (t, p, w, o) => {
    const a = o.alpha ?? 0.5
    return weightedMean(
      map(t.length, (i) => {
        const d = (t[i] ?? 0) - (p[i] ?? 0)
        return d >= 0 ? a * d : (a - 1) * d
      }),
      w,
    )
  },
)

/**
 * D^2 score for the absolute error (sklearn.metrics.d2_absolute_error_score): the fraction of the absolute error of the
 * (weighted) median that the predictions explain.
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { d2AbsoluteErrorScore } from 'whitebox-engine/metrics'
 * d2AbsoluteErrorScore([1, 2, 3], [1, 2, 3]).estimate.value // => 1
 * ```
 */
export const d2AbsoluteErrorScore: Metric = perOutputMetric(
  spec('d2AbsoluteErrorScore', 'higher', [-Infinity, 1]),
  (t, p, w) => d2Pinball(t, p, w, 0.5),
)

const pinballTerm = (d: number, a: number): number => (d >= 0 ? a * d : (a - 1) * d)

/** 1 - pinball(y, yhat) / pinball(y, the weighted alpha-quantile of y), finite as sklearn's force_finite. */
function d2Pinball(t: Float64Array, p: Float64Array, w: Float64Array, alpha: number): number {
  const q = weightedPercentile(t, w, alpha * 100, true)
  const num = weightedMean(
    map(t.length, (i) => pinballTerm((t[i] ?? 0) - (p[i] ?? 0), alpha)),
    w,
  )
  const den = weightedMean(
    map(t.length, (i) => pinballTerm((t[i] ?? 0) - q, alpha)),
    w,
  )
  if (den === 0) return num === 0 ? 1 : 0
  return 1 - num / den
}

/**
 * D^2 score for the pinball loss (sklearn.metrics.d2_pinball_score): the fraction of the pinball loss of the
 * constant alpha-quantile that the predicted quantiles explain.
 *
 * @param yTrue - Targets.
 * @param yPred - Predicted alpha-quantiles.
 * @param options - alpha (default 0.5) and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { d2PinballScore } from 'whitebox-engine/metrics'
 * d2PinballScore([1, 2, 3], [1, 2, 3], { alpha: 0.9 }).estimate.value // => 1
 * ```
 */
export const d2PinballScore: Metric<PinballOptions> = perOutputMetric<PinballOptions>(
  spec('d2PinballScore', 'higher', [-Infinity, 1]),
  (t, p, w, o) => d2Pinball(t, p, w, o.alpha ?? 0.5),
)

/**
 * D^2 score for the Tweedie deviance (sklearn.metrics.d2_tweedie_score): 1 - deviance(y, yhat) / deviance(y, mean y),
 * a generalisation of R^2 (power 0) to the Tweedie family; single output.
 *
 * @param yTrue - Targets.
 * @param yPred - Predicted means.
 * @param options - power (default 0) and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { d2TweedieScore } from 'whitebox-engine/metrics'
 * d2TweedieScore([1, 2, 3], [1, 2, 3], { power: 1 }).estimate.value // => 1
 * ```
 */
export const d2TweedieScore: Metric<TweedieOptions> = perOutputMetric<TweedieOptions>(
  spec('d2TweedieScore', 'higher', [-Infinity, 1]),
  (t, p, w, o) => {
    const power = o.power ?? 0
    const m = weightedMean(t, w)
    const num = weightedMean(
      map(t.length, (i) => tweedieUnitDeviance(t[i] ?? 0, p[i] ?? 0, power)),
      w,
    )
    const den = weightedMean(
      map(t.length, (i) => tweedieUnitDeviance(t[i] ?? 0, m, power)),
      w,
    )
    return 1 - num / den
  },
)
