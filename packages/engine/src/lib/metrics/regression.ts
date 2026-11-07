/**
 * The standard regression metrics, scikit-learn style: mean squared, root mean squared, mean absolute, median absolute
 * and maximum errors, R^2 and adjusted R^2, explained variance, mean squared log error, mean absolute percentage error,
 * and the log-cosh and Huber errors that mirror the corresponding losses.
 */

import { type Direction, MdmlError, type MetricSpec, maxOf } from '../base/index.ts'
import { type Metric, type MetricOptions, perOutputMetric, weightedMean, weightedPercentile } from './metric.ts'

const map = (n: number, f: (i: number) => number): Float64Array => Float64Array.from({ length: n }, (_, i) => f(i))
const residuals = (t: Float64Array, p: Float64Array): Float64Array => map(t.length, (i) => (t[i] ?? 0) - (p[i] ?? 0))

/** The median, or with weights sklearn's weighted percentile at 50% (the lowest value reaching half the weight). */
const median = (values: Float64Array, weight: Float64Array, weighted: boolean): number =>
  weighted ? weightedPercentile(values, weight, 50, false) : weightedPercentile(values, undefined, 50, true)

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
 * @index function Mean squared error; R2, MAE and the rest sit beside it.
 * @example
 * ```ts
 * import { meanSquaredError } from 'mdml-engine/metrics'
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
 * import { rootMeanSquaredError } from 'mdml-engine/metrics'
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
 * import { meanAbsoluteError } from 'mdml-engine/metrics'
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
 * import { medianAbsoluteError } from 'mdml-engine/metrics'
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
 * @param options - Sample weights (a sample of weight 0 is ignored).
 * @returns The metric result.
 * @example
 * ```ts
 * import { maxError } from 'mdml-engine/metrics'
 * maxError([1, 2, 3], [1, 5, 2]).estimate.value // => 3
 * ```
 */
export const maxError: Metric = perOutputMetric(spec('maxError', 'lower', [0, Infinity]), (t, p, w) => {
  // A sample of weight 0 is absent (weights count repetitions), so it cannot be the maximum.
  const kept = residuals(t, p)
    .map(Math.abs)
    .filter((_, i) => (w[i] ?? 0) > 0)
  return kept.length === 0 ? Number.NaN : maxOf(kept)
})

/**
 * The coefficient of determination R^2 (sklearn.metrics.r2_score): 1 - SS_res / SS_tot. For constant targets it is 1
 * for a perfect prediction and 0 otherwise, as sklearn's `force_finite`.
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @param options - Sample weights.
 * @returns The metric result.
 * @index function Coefficient of determination.
 * @example
 * ```ts
 * import { r2Score } from 'mdml-engine/metrics'
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

/** Options of {@link adjustedR2Score}. */
export interface AdjustedR2Options extends MetricOptions {
  /** The number of predictors p in the model (required). */
  readonly nFeatures: number
}

/**
 * Adjusted R^2: 1 - (1 - R^2)(n - 1) / (n - p - 1), which penalises R^2 for the number of predictors p (n is the
 * number of samples, or the sum of the weights when weighted).
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @param options - nFeatures and sample weights.
 * @returns The metric result.
 * @throws MdmlError with code `value` when n - p - 1 <= 0.
 * @example
 * ```ts
 * import { adjustedR2Score } from 'mdml-engine/metrics'
 * adjustedR2Score([1, 2, 3, 4], [1, 2, 3, 4], { nFeatures: 1 }).estimate.value // => 1
 * ```
 */
export const adjustedR2Score: Metric<AdjustedR2Options> = perOutputMetric<AdjustedR2Options>(
  spec('adjustedR2Score', 'higher', [-Infinity, 1]),
  (t, p, w, o) => {
    const n = o.sampleWeight === undefined ? t.length : w.reduce((a, b) => a + b, 0)
    const dof = n - o.nFeatures - 1
    if (!(dof > 0)) throw new MdmlError('value', `adjustedR2Score: needs more samples than nFeatures + 1`)
    const r2 = r2Score(t, p, o.sampleWeight === undefined ? {} : { sampleWeight: w }).estimate.value
    return 1 - ((1 - r2) * (n - 1)) / dof
  },
)

/**
 * Mean log-cosh error: the mean of log(cosh(y - yhat)), quadratic for small errors and linear for large ones (the
 * metric counterpart of `losses.logCosh`; Keras's LogCoshError).
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { meanLogCoshError } from 'mdml-engine/metrics'
 * meanLogCoshError([0, 1], [0, 1]).estimate.value // => 0
 * ```
 */
export const meanLogCoshError: Metric = perOutputMetric(spec('meanLogCoshError', 'lower', [0, Infinity]), (t, p, w) =>
  weightedMean(
    map(t.length, (i) => {
      const r = Math.abs((t[i] ?? 0) - (p[i] ?? 0))
      return r + Math.log1p(Math.exp(-2 * r)) - Math.LN2
    }),
    w,
  ),
)

/** Options of {@link meanHuberLoss}. */
export interface HuberOptions extends MetricOptions {
  /** The threshold between the quadratic and linear regions (default 1). */
  readonly delta?: number
}

/**
 * Mean Huber loss: 0.5 r^2 for |r| <= delta and delta (|r| - 0.5 delta) beyond (the metric counterpart of
 * `losses.huber`; Keras's Huber).
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @param options - delta and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { meanHuberLoss } from 'mdml-engine/metrics'
 * meanHuberLoss([0], [3], { delta: 1 }).estimate.value // => 2.5
 * ```
 */
export const meanHuberLoss: Metric<HuberOptions> = perOutputMetric<HuberOptions>(
  spec('meanHuberLoss', 'lower', [0, Infinity]),
  (t, p, w, o) => {
    const delta = o.delta ?? 1
    return weightedMean(
      map(t.length, (i) => {
        const r = Math.abs((t[i] ?? 0) - (p[i] ?? 0))
        return r <= delta ? 0.5 * r * r : delta * (r - 0.5 * delta)
      }),
      w,
    )
  },
)

/**
 * Explained variance (sklearn.metrics.explained_variance_score): 1 - Var(residuals) / Var(targets).
 *
 * @param yTrue - Targets.
 * @param yPred - Predictions.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { explainedVarianceScore } from 'mdml-engine/metrics'
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
  if (t.some((v) => v < 0) || p.some((v) => v < 0)) throw new MdmlError('value', `${name}: values must be non-negative`)
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
 * import { meanSquaredLogError } from 'mdml-engine/metrics'
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
 * import { meanAbsolutePercentageError } from 'mdml-engine/metrics'
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
