/**
 * Forecast accuracy metrics, sktime and M-competition style: percentage errors (MAPE, sMAPE, WMAPE), the scaled errors
 * MASE and RMSSE (relative to a naive forecast), interval scores, MSIS and coverage for prediction intervals,
 * multi-quantile loss, the continuous ranked probability score (CRPS) for ensembles and Gaussians, and Theil's U2.
 */

import { type MetricResult, type MetricSpec, WhiteboxError } from '../base/index.ts'
import { normal } from '../dists/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import {
  defineMetric,
  flat,
  type Metric,
  type MetricFunction,
  type MetricOptions,
  matrix,
  metricResult,
  perOutputMetric,
  sampleWeights,
  weightedMean,
} from './metric.ts'
import { meanAbsolutePercentageError } from './regression.ts'
import { crps } from './scoring-rules.ts'

const spec = (
  name: string,
  direction: 'higher' | 'lower' = 'lower',
  range: readonly [number, number] = [0, Infinity],
): MetricSpec => ({
  name,
  direction,
  range,
  inputs: 'values',
  category: 'forecasting',
  concept: 'forecast-evaluation',
})

/**
 * Mean absolute percentage error, as a fraction: the forecasting name of {@link meanAbsolutePercentageError}
 * (sklearn's definition, dividing by max(|y|, eps)).
 *
 * @param yTrue - Actual values.
 * @param yPred - Forecasts.
 * @param options - Sample weights.
 * @returns The metric result.
 * @concept forecast-evaluation
 * @example
 * ```ts
 * import { mape } from 'whitebox-engine/metrics'
 * mape([2, 4], [1, 5]).estimate.value // => 0.375
 * ```
 */
export const mape: Metric = meanAbsolutePercentageError

/**
 * Symmetric mean absolute percentage error, as a fraction in [0, 2]: mean 2|y - yhat| / (|y| + |yhat|)
 * (torchmetrics' SymmetricMeanAbsolutePercentageError; the M4 sMAPE divided by 100).
 *
 * @param yTrue - Actual values.
 * @param yPred - Forecasts.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { symmetricMeanAbsolutePercentageError } from 'whitebox-engine/metrics'
 * symmetricMeanAbsolutePercentageError([1, 2], [3, 2]).estimate.value // => 0.5
 * ```
 */
export const symmetricMeanAbsolutePercentageError: Metric = perOutputMetric(
  spec('symmetricMeanAbsolutePercentageError', 'lower', [0, 2]),
  (t, p, w) =>
    weightedMean(
      Array.from(t, (y, i) => (2 * Math.abs(y - (p[i] ?? 0))) / Math.max(Math.abs(y) + Math.abs(p[i] ?? 0), 1.17e-6)),
      w,
    ),
)

/**
 * Weighted mean absolute percentage error sum |y - yhat| / sum |y| (torchmetrics'
 * WeightedMeanAbsolutePercentageError), which stays finite when some actuals are zero.
 *
 * @param yTrue - Actual values.
 * @param yPred - Forecasts.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { weightedMeanAbsolutePercentageError } from 'whitebox-engine/metrics'
 * weightedMeanAbsolutePercentageError([0, 4], [1, 5]).estimate.value // => 0.5
 * ```
 */
export const weightedMeanAbsolutePercentageError: Metric = perOutputMetric(
  spec('weightedMeanAbsolutePercentageError'),
  (t, p, w) => {
    let num = 0
    let den = 0
    for (const [i, y] of t.entries()) {
      num += (w[i] ?? 0) * Math.abs(y - (p[i] ?? 0))
      den += (w[i] ?? 0) * Math.abs(y)
    }
    return num / Math.max(den, 1.17e-6)
  },
)

/** Options of the scaled errors. */
export interface ScaledErrorOptions extends MetricOptions {
  /** The in-sample (training) series the naive forecast is computed on. */
  readonly yTrain: TensorLike
  /** The seasonal period m of the naive forecast y_t = y_{t-m} (default 1). */
  readonly seasonality?: number
}

/** The in-sample error of the seasonal naive forecast, mean |y_t - y_{t-m}|^p. */
function naiveScale(yTrain: TensorLike, m: number, p: number): number {
  const y = flat(yTrain)
  if (y.length <= m)
    throw new WhiteboxError('value', 'scaled errors need a training series longer than the seasonality')
  let s = 0
  for (let t = m; t < y.length; t++) s += Math.abs((y[t] ?? 0) - (y[t - m] ?? 0)) ** p
  return s / (y.length - m)
}

/**
 * Mean absolute scaled error (Hyndman and Koehler, 2006; sktime's mean_absolute_scaled_error): the forecast MAE
 * divided by the in-sample MAE of the seasonal naive forecast on the training series.
 *
 * @param yTrue - Actual values over the horizon.
 * @param yPred - Forecasts.
 * @param options - yTrain, seasonality (default 1) and sample weights.
 * @returns The metric result (below 1 beats the naive forecast).
 * @example
 * ```ts
 * import { meanAbsoluteScaledError } from 'whitebox-engine/metrics'
 * meanAbsoluteScaledError([5, 6], [5, 8], { yTrain: [1, 2, 3, 4] }).estimate.value // => 1
 * ```
 */
export const meanAbsoluteScaledError: Metric<ScaledErrorOptions> = Object.assign(
  (yTrue: TensorLike, yPred: TensorLike, options: ScaledErrorOptions = { yTrain: [] }): MetricResult => {
    const t = flat(yTrue)
    const p = flat(yPred)
    const w = sampleWeights('meanAbsoluteScaledError', t.length, options.sampleWeight)
    const mae = weightedMean(
      Array.from(t, (y, i) => Math.abs(y - (p[i] ?? 0))),
      w,
    )
    return metricResult(
      'meanAbsoluteScaledError',
      mae / naiveScale(options.yTrain, options.seasonality ?? 1, 1),
      t.length,
    )
  },
  { spec: spec('meanAbsoluteScaledError') },
)

/**
 * Root mean squared scaled error (the M5 competition's RMSSE; sktime's mean_squared_scaled_error with square_root):
 * sqrt(forecast MSE / in-sample MSE of the seasonal naive forecast).
 *
 * @param yTrue - Actual values over the horizon.
 * @param yPred - Forecasts.
 * @param options - yTrain, seasonality and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { rootMeanSquaredScaledError } from 'whitebox-engine/metrics'
 * rootMeanSquaredScaledError([5, 6], [6, 7], { yTrain: [1, 2, 3, 4] }).estimate.value // => 1
 * ```
 */
export const rootMeanSquaredScaledError: Metric<ScaledErrorOptions> = Object.assign(
  (yTrue: TensorLike, yPred: TensorLike, options: ScaledErrorOptions = { yTrain: [] }): MetricResult => {
    const t = flat(yTrue)
    const p = flat(yPred)
    const w = sampleWeights('rootMeanSquaredScaledError', t.length, options.sampleWeight)
    const mse = weightedMean(
      Array.from(t, (y, i) => (y - (p[i] ?? 0)) ** 2),
      w,
    )
    return metricResult(
      'rootMeanSquaredScaledError',
      Math.sqrt(mse / naiveScale(options.yTrain, options.seasonality ?? 1, 2)),
      t.length,
    )
  },
  { spec: spec('rootMeanSquaredScaledError') },
)

/** Options of the interval scores. */
export interface IntervalScoreOptions {
  /** The interval's nominal miscoverage alpha (a central 1 - alpha interval; default 0.05). */
  readonly alpha?: number
}

/** The interval scores (u - l) + (2 / alpha)(l - y)+ + (2 / alpha)(y - u)+ of each observation. */
function intervalScores(lower: TensorLike, upper: TensorLike, y: TensorLike, alpha: number): number[] {
  const l = flat(lower)
  const u = flat(upper)
  const t = flat(y)
  return Array.from(t, (v, i) => {
    const lo = l[i] ?? 0
    const hi = u[i] ?? 0
    return hi - lo + (2 / alpha) * Math.max(lo - v, 0) + (2 / alpha) * Math.max(v - hi, 0)
  })
}

/**
 * The interval score of central prediction intervals (Gneiting and Raftery, 2007; the Winkler score):
 * (u - l) + (2 / alpha)(l - y)+ + (2 / alpha)(y - u)+, averaged; a proper score for the interval's quantiles.
 *
 * @param lower - Lower bounds `[N]`.
 * @param upper - Upper bounds `[N]`.
 * @param y - Observations `[N]`.
 * @param options - alpha (default 0.05).
 * @returns The metric result.
 * @example
 * ```ts
 * import { intervalScore } from 'whitebox-engine/metrics'
 * intervalScore([0], [2], [3], { alpha: 0.5 }).estimate.value // => 6
 * ```
 */
export const intervalScore: MetricFunction<[TensorLike, TensorLike, TensorLike, IntervalScoreOptions?]> = defineMetric(
  spec('intervalScore'),
  (lower: TensorLike, upper: TensorLike, y: TensorLike, options: IntervalScoreOptions = {}): MetricResult => {
    const s = intervalScores(lower, upper, y, options.alpha ?? 0.05)
    return metricResult('intervalScore', s.reduce((a, b) => a + b, 0) / s.length, s.length)
  },
)

/**
 * Mean scaled interval score (the M4 competition's MSIS): the mean interval score of the forecast intervals divided by
 * the in-sample MAE of the seasonal naive forecast.
 *
 * @param lower - Lower bounds over the horizon.
 * @param upper - Upper bounds.
 * @param y - Observations.
 * @param options - yTrain, seasonality and alpha (default 0.05).
 * @returns The metric result.
 * @example
 * ```ts
 * import { meanScaledIntervalScore } from 'whitebox-engine/metrics'
 * meanScaledIntervalScore([4], [6], [5], { yTrain: [1, 2, 3] }).estimate.value // => 2
 * ```
 */
export const meanScaledIntervalScore: MetricFunction<
  [
    TensorLike,
    TensorLike,
    TensorLike,
    IntervalScoreOptions & { readonly yTrain: TensorLike; readonly seasonality?: number },
  ]
> = defineMetric(
  spec('meanScaledIntervalScore'),
  (
    lower: TensorLike,
    upper: TensorLike,
    y: TensorLike,
    options: IntervalScoreOptions & { readonly yTrain: TensorLike; readonly seasonality?: number },
  ): MetricResult => {
    const s = intervalScores(lower, upper, y, options.alpha ?? 0.05)
    const scale = naiveScale(options.yTrain, options.seasonality ?? 1, 1)
    return metricResult('meanScaledIntervalScore', s.reduce((a, b) => a + b, 0) / s.length / scale, s.length)
  },
)

/**
 * Empirical coverage of prediction intervals: the fraction of observations inside [lower, upper], to compare with
 * the nominal level.
 *
 * @param lower - Lower bounds `[N]`.
 * @param upper - Upper bounds `[N]`.
 * @param y - Observations `[N]`.
 * @returns The metric result.
 * @example
 * ```ts
 * import { intervalCoverage } from 'whitebox-engine/metrics'
 * intervalCoverage([0, 0], [1, 1], [0.5, 2]).estimate.value // => 0.5
 * ```
 */
export const intervalCoverage: MetricFunction<[TensorLike, TensorLike, TensorLike]> = defineMetric(
  { ...spec('intervalCoverage', 'higher', [0, 1]), concept: 'prediction-intervals' },
  (lower: TensorLike, upper: TensorLike, y: TensorLike): MetricResult => {
    const l = flat(lower)
    const u = flat(upper)
    const t = flat(y)
    const inside = Array.from(t).filter((v, i) => v >= (l[i] ?? 0) && v <= (u[i] ?? 0)).length
    return metricResult('intervalCoverage', inside / t.length, t.length, { successes: inside, trials: t.length })
  },
)

/** Options of {@link quantileLoss}. */
export interface QuantileLossOptions {
  /** The quantile levels of the forecast columns (required). */
  readonly quantiles: readonly number[]
  /** 'mean' pinball over levels and observations (default), or 'weighted': 2 sum pinball / sum |y| per level, averaged (GluonTS's weighted quantile loss). */
  readonly normalize?: 'mean' | 'weighted'
}

/**
 * Quantile loss of multi-quantile forecasts: the pinball loss of each forecast column against its level, averaged
 * over levels and observations (pytorch-forecasting's QuantileLoss), or GluonTS's weighted quantile loss.
 *
 * @param yTrue - Observations `[N]`.
 * @param yPred - Quantile forecasts `[N, Q]`.
 * @param options - quantiles and normalize.
 * @returns The metric result, with the per-level losses in the breakdown.
 * @example
 * ```ts
 * import { quantileLoss } from 'whitebox-engine/metrics'
 * quantileLoss([1], [[0, 2]], { quantiles: [0.25, 0.75] }).estimate.value // => 0.25
 * ```
 */
export const quantileLoss: MetricFunction<[TensorLike, TensorLike, QuantileLossOptions]> = defineMetric(
  spec('quantileLoss'),
  (yTrue: TensorLike, yPred: TensorLike, options: QuantileLossOptions): MetricResult => {
    const t = flat(yTrue)
    const p = matrix(yPred)
    const q = options.quantiles.length
    if ((p.shape[1] ?? 1) !== q) throw new WhiteboxError('shape', 'quantileLoss: one forecast column per quantile')
    const absTotal = t.reduce((a, v) => a + Math.abs(v), 0)
    const per = options.quantiles.map((tau, j) => {
      let s = 0
      for (const [i, y] of t.entries()) {
        const r = y - (p.data[i * q + j] ?? 0)
        s += Math.max(tau * r, (tau - 1) * r)
      }
      return options.normalize === 'weighted' ? (2 * s) / absTotal : s / t.length
    })
    return metricResult(
      'quantileLoss',
      per.reduce((a, b) => a + b, 0) / q,
      t.length,
      Object.fromEntries(options.quantiles.map((tau, j) => [String(tau), per[j] ?? 0])),
    )
  },
)

/**
 * CRPS of ensemble (sample) forecasts (properscoring's crps_ensemble): E|X - y| - E|X - X'| / 2 over the members, or
 * the fair (unbiased) version that divides the second term by M (M - 1) instead of M^2.
 *
 * @param y - Observations `[N]`.
 * @param samples - Ensemble members `[N, M]`.
 * @param options - fair (default false) and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { crpsEnsemble } from 'whitebox-engine/metrics'
 * crpsEnsemble([0], [[0, 0]]).estimate.value // => 0
 * ```
 */
export const crpsEnsemble: MetricFunction<[TensorLike, TensorLike, (MetricOptions & { readonly fair?: boolean })?]> =
  defineMetric(
    spec('crpsEnsemble'),
    (y: TensorLike, samples: TensorLike, options: MetricOptions & { readonly fair?: boolean } = {}): MetricResult => {
      const t = flat(y)
      const s = matrix(samples)
      const m = s.shape[1] ?? 1
      const w = sampleWeights('crpsEnsemble', t.length, options.sampleWeight)
      const values = Array.from(t, (v, i) => {
        const xs = s.data.subarray(i * m, (i + 1) * m)
        let first = 0
        let second = 0
        for (let a = 0; a < m; a++) {
          first += Math.abs((xs[a] ?? 0) - v)
          for (let b = 0; b < m; b++) second += Math.abs((xs[a] ?? 0) - (xs[b] ?? 0))
        }
        const denominator = options.fair === true ? m * (m - 1) : m * m
        return first / m - second / (2 * denominator)
      })
      return metricResult('crpsEnsemble', weightedMean(values, w), t.length)
    },
  )

/**
 * CRPS of Gaussian forecasts N(mu, sigma^2) in closed form (properscoring's crps_gaussian; the normal case of
 * {@link crps}).
 *
 * @param y - Observations `[N]`.
 * @param mu - Forecast means.
 * @param sigma - Forecast standard deviations.
 * @returns The metric result.
 * @example
 * ```ts
 * import { crpsNormal } from 'whitebox-engine/metrics'
 * Math.abs(crpsNormal([0], [0], [1]).estimate.value - (Math.SQRT2 - 1) / Math.sqrt(Math.PI)) < 1e-15 // => true
 * ```
 */
export const crpsNormal: MetricFunction<[TensorLike, TensorLike, TensorLike]> = defineMetric(
  spec('crpsNormal'),
  (y: TensorLike, mu: TensorLike, sigma: TensorLike): MetricResult => ({
    ...crps(normal(mu, sigma), y),
    metric: 'crpsNormal',
  }),
)

/**
 * Theil's U2 (Theil, 1966): the RMS of relative forecast errors over the RMS of the relative changes of the naive
 * no-change forecast, sqrt(sum ((yhat_{t+1} - y_{t+1}) / y_t)^2 / sum ((y_{t+1} - y_t) / y_t)^2); below 1 beats the
 * naive forecast.
 *
 * @param yTrue - Actual series `[T]`.
 * @param yPred - One-step forecasts `[T]` (the first is unused).
 * @returns The metric result.
 * @example
 * ```ts
 * import { theilsU2 } from 'whitebox-engine/metrics'
 * theilsU2([1, 2, 4], [1, 2, 4]).estimate.value // => 0
 * ```
 */
export const theilsU2: Metric = Object.assign(
  (yTrue: TensorLike, yPred: TensorLike): MetricResult => {
    const t = flat(yTrue)
    const p = flat(yPred)
    let num = 0
    let den = 0
    for (let i = 0; i + 1 < t.length; i++) {
      const base = t[i] ?? 1
      num += (((p[i + 1] ?? 0) - (t[i + 1] ?? 0)) / base) ** 2
      den += (((t[i + 1] ?? 0) - base) / base) ** 2
    }
    return metricResult('theilsU2', Math.sqrt(num / den), t.length)
  },
  { spec: spec('theilsU2') },
)
