import { type Direction, type MetricSpec, maxOf, WhiteboxError } from '../base/index.ts'
import { type Metric, type MetricOptions, perOutputMetric, weightedMean } from './metric.ts'

const map = (n: number, f: (i: number) => number): Float64Array => Float64Array.from({ length: n }, (_, i) => f(i))
const residuals = (t: Float64Array, p: Float64Array): Float64Array => map(t.length, (i) => (t[i] ?? 0) - (p[i] ?? 0))

/** Median, or with weights sklearn's weighted percentile at 50% (the lowest value reaching half the total weight). */
function median(values: Float64Array, weight: Float64Array, weighted: boolean): number {
  const order = Array.from(values.keys()).sort((a, b) => (values[a] ?? 0) - (values[b] ?? 0))
  if (!weighted) {
    const m = order.length
    const lo = values[order[Math.floor((m - 1) / 2)] ?? 0] ?? 0
    const hi = values[order[Math.ceil((m - 1) / 2)] ?? 0] ?? 0
    return (lo + hi) / 2
  }
  const total = weight.reduce((a, b) => a + b, 0)
  let acc = 0
  for (const i of order) {
    acc += weight[i] ?? 0
    if (acc >= 0.5 * total) return values[i] ?? 0
  }
  return values[order[order.length - 1] ?? 0] ?? 0
}

const spec = (name: string, direction: Direction, range: readonly [number, number]): MetricSpec => ({
  name,
  direction,
  range,
  inputs: 'values',
  category: 'regression',
  concept: 'regression-metrics',
})

/**
 * Mean squared error (sklearn.metrics.mean_squared_error).
 *
 * @param yTrue - Targets `[n]` or `[n, k]`.
 * @param yPred - Predictions of the same shape.
 * @param options - Sample weights.
 * @returns The metric result (multi-output scores are averaged, with a per-output breakdown).
 * @concept regression-metrics
 * @example
 * ```ts
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * meanSquaredError([1, 2, 3], [1, 2, 5]).estimate.value // => 1.3333333333333333
 * ```
 */
export const meanSquaredError: Metric = perOutputMetric(spec('meanSquaredError', 'lower', [0, Infinity]), (t, p, w) =>
  weightedMean(
    map(t.length, (i) => ((t[i] ?? 0) - (p[i] ?? 0)) ** 2),
    w,
  ),
)

/**
 * Root mean squared error (sklearn.metrics.root_mean_squared_error).
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { rootMeanSquaredError } from 'whitebox-engine/metrics'
 * rootMeanSquaredError([0, 0], [3, 4]).estimate.value === Math.sqrt(12.5) // => true
 * ```
 */
export const rootMeanSquaredError: Metric = perOutputMetric(
  spec('rootMeanSquaredError', 'lower', [0, Infinity]),
  (t, p, w) =>
    Math.sqrt(
      weightedMean(
        map(t.length, (i) => ((t[i] ?? 0) - (p[i] ?? 0)) ** 2),
        w,
      ),
    ),
)

/**
 * Mean absolute error (sklearn.metrics.mean_absolute_error).
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { meanAbsoluteError } from 'whitebox-engine/metrics'
 * meanAbsoluteError([1, 2], [2, 4]).estimate.value // => 1.5
 * ```
 */
export const meanAbsoluteError: Metric = perOutputMetric(spec('meanAbsoluteError', 'lower', [0, Infinity]), (t, p, w) =>
  weightedMean(
    map(t.length, (i) => Math.abs((t[i] ?? 0) - (p[i] ?? 0))),
    w,
  ),
)

/**
 * Median absolute error (sklearn.metrics.median_absolute_error).
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @param options - Sample weights (a weighted median).
 * @returns The metric result.
 * @example
 * ```ts
 * import { medianAbsoluteError } from 'whitebox-engine/metrics'
 * medianAbsoluteError([0, 0, 0], [1, 5, 2]).estimate.value // => 2
 * ```
 */
export const medianAbsoluteError: Metric = perOutputMetric(
  spec('medianAbsoluteError', 'lower', [0, Infinity]),
  (t, p, w, o) =>
    median(
      map(t.length, (i) => Math.abs((t[i] ?? 0) - (p[i] ?? 0))),
      w,
      o.sampleWeight !== undefined,
    ),
)

/**
 * The largest absolute error (sklearn.metrics.max_error; single output).
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @returns The metric result.
 * @example
 * ```ts
 * import { maxError } from 'whitebox-engine/metrics'
 * maxError([1, 2, 3], [1, 5, 2]).estimate.value // => 3
 * ```
 */
export const maxError: Metric = perOutputMetric(spec('maxError', 'lower', [0, Infinity]), (t, p) =>
  maxOf(residuals(t, p).map(Math.abs)),
)

/**
 * The coefficient of determination R^2 (sklearn.metrics.r2_score): 1 - SS_res / SS_tot. For constant targets it is 1
 * for a perfect prediction and 0 otherwise, as sklearn's `force_finite`.
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { r2Score } from 'whitebox-engine/metrics'
 * r2Score([1, 2, 3], [1, 2, 3]).estimate.value // => 1
 * ```
 */
export const r2Score: Metric = perOutputMetric(spec('r2Score', 'higher', [-Infinity, 1]), (t, p, w) => {
  const m = weightedMean(t, w)
  const ssRes = weightedMean(
    map(t.length, (i) => ((t[i] ?? 0) - (p[i] ?? 0)) ** 2),
    w,
  )
  const ssTot = weightedMean(
    map(t.length, (i) => ((t[i] ?? 0) - m) ** 2),
    w,
  )
  if (ssTot === 0) return ssRes === 0 ? 1 : 0
  return 1 - ssRes / ssTot
})

/**
 * Explained variance (sklearn.metrics.explained_variance_score): 1 - Var(residuals) / Var(targets).
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { explainedVarianceScore } from 'whitebox-engine/metrics'
 * explainedVarianceScore([1, 2, 3], [2, 3, 4]).estimate.value // => 1
 * ```
 */
export const explainedVarianceScore: Metric = perOutputMetric(
  spec('explainedVarianceScore', 'higher', [-Infinity, 1]),
  (t, p, w) => {
    const r = residuals(t, p)
    const mr = weightedMean(r, w)
    const mt = weightedMean(t, w)
    const vr = weightedMean(
      map(r.length, (i) => ((r[i] ?? 0) - mr) ** 2),
      w,
    )
    const vt = weightedMean(
      map(t.length, (i) => ((t[i] ?? 0) - mt) ** 2),
      w,
    )
    if (vt === 0) return vr === 0 ? 1 : 0
    return 1 - vr / vt
  },
)

function requireNonNegative(name: string, t: Float64Array, p: Float64Array): void {
  if (t.some((v) => v < 0) || p.some((v) => v < 0))
    throw new WhiteboxError('value', `${name}: values must be non-negative`)
}

/**
 * Mean squared logarithmic error (sklearn.metrics.mean_squared_log_error), for non-negative values.
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { meanSquaredLogError } from 'whitebox-engine/metrics'
 * meanSquaredLogError([0], [Math.E - 1]).estimate.value // => 1
 * ```
 */
export const meanSquaredLogError: Metric = perOutputMetric(
  spec('meanSquaredLogError', 'lower', [0, Infinity]),
  (t, p, w) => {
    requireNonNegative('meanSquaredLogError', t, p)
    return weightedMean(
      map(t.length, (i) => (Math.log1p(t[i] ?? 0) - Math.log1p(p[i] ?? 0)) ** 2),
      w,
    )
  },
)

/**
 * Mean absolute percentage error as a fraction (sklearn.metrics.mean_absolute_percentage_error, which divides by
 * max(|y|, eps)).
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { meanAbsolutePercentageError } from 'whitebox-engine/metrics'
 * meanAbsolutePercentageError([2, 4], [1, 5]).estimate.value // => 0.375
 * ```
 */
export const meanAbsolutePercentageError: Metric = perOutputMetric(
  spec('meanAbsolutePercentageError', 'lower', [0, Infinity]),
  (t, p, w) =>
    weightedMean(
      map(t.length, (i) => Math.abs((t[i] ?? 0) - (p[i] ?? 0)) / Math.max(Math.abs(t[i] ?? 0), 2.220446049250313e-16)),
      w,
    ),
)

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
  (t, p, w, o) => {
    const m = median(t, w, o.sampleWeight !== undefined)
    const num = weightedMean(
      map(t.length, (i) => Math.abs((t[i] ?? 0) - (p[i] ?? 0))),
      w,
    )
    const den = weightedMean(
      map(t.length, (i) => Math.abs((t[i] ?? 0) - m)),
      w,
    )
    if (den === 0) return num === 0 ? 1 : 0
    return 1 - num / den
  },
)
