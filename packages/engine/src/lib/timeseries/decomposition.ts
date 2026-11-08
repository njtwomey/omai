/**
 * Classical seasonal decomposition by moving averages (statsmodels' `seasonal_decompose`): the trend is a centred moving
 * average over one period, the seasonal component the mean of the detrended values at each phase, and the residual what
 * is left, for additive and multiplicative models. The loess-based STL and MSTL are in `stl.ts`; both return the
 * {@link DecompositionResult} defined here.
 */

import { OmaiError } from '../base/index.ts'
import { leastSquares } from './_ml.ts'
import { asSeries } from './_series.ts'

/** A series split into trend, seasonal and residual components. */
export interface DecompositionResult {
  readonly observed: readonly number[]
  /** The trend (NaN where a moving average cannot reach, unless extrapolated). */
  readonly trend: readonly number[]
  readonly seasonal: readonly number[]
  readonly resid: readonly number[]
  /** Robustness weights (STL with robust fitting). */
  readonly weights?: readonly number[]
}

/** Options of {@link seasonalDecompose}. */
export interface SeasonalDecomposeOptions {
  /** `additive` (default) or `multiplicative` (y = trend * seasonal * resid). */
  readonly model?: 'additive' | 'multiplicative'
  /** Use a moving average that ends at the current point instead of a centred one (default false). */
  readonly oneSided?: boolean
  /**
   * Extrapolate the trend over the NaN ends from a line through this many end points, or `'period'` for one fewer than
   * the period (default 0: none).
   */
  readonly extrapolateTrend?: number | 'period'
}

/** The weights of the moving average over one period: `[0.5, 1, ..., 1, 0.5] / period` for an even period. */
function filterWeights(period: number): number[] {
  return period % 2 === 0
    ? [0.5, ...new Array<number>(period - 1).fill(1), 0.5].map((v) => v / period)
    : new Array<number>(period).fill(1 / period)
}

/**
 * Decompose a series into trend, seasonal and residual by moving averages (statsmodels' `seasonal_decompose`).
 *
 * @param y - The series (positive for the multiplicative model).
 * @param period - The seasonal period.
 * @param options - The model, the filter side and the trend extrapolation.
 * @returns The components; the trend is NaN at the ends where the centred filter does not fit.
 * @throws OmaiError with code `value` for a series shorter than two periods or a non-positive multiplicative series.
 * @example
 * ```ts
 * import { seasonalDecompose } from 'omai-engine/timeseries'
 * const y = Array.from({ length: 12 }, (_, t) => t + [2, 0, -2, 0][t % 4])
 * const d = seasonalDecompose(y, 4)
 * [d.seasonal.slice(0, 4), d.trend.slice(2, 5)] // => [[2, 0, -2, 0], [2, 3, 4]]
 * ```
 */
export function seasonalDecompose(
  y: ArrayLike<number>,
  period: number,
  options: SeasonalDecomposeOptions = {},
): DecompositionResult {
  const x = asSeries(Float64Array.from(y), 'seasonalDecompose')
  const n = x.length
  if (!(Number.isInteger(period) && period >= 2) || n < 2 * period)
    throw new OmaiError('value', 'seasonalDecompose: need an integer period of at least 2 and two periods of data')
  const multiplicative = options.model === 'multiplicative'
  if (multiplicative && x.some((v) => !(v > 0)))
    throw new OmaiError('value', 'seasonalDecompose: a multiplicative model needs a positive series')
  const weights = filterWeights(period)
  const half = Math.floor(weights.length / 2)
  const trend = new Float64Array(n).fill(Number.NaN)
  const oneSided = options.oneSided === true
  for (let t = 0; t < n; t++) {
    // Centred: window t - half .. t + half; one-sided: t - len + 1 .. t.
    const lo = oneSided ? t - weights.length + 1 : t - half
    if (lo < 0 || lo + weights.length > n) continue
    let s = 0
    // The convolution flips the filter, which is symmetric for the centred case.
    for (let j = 0; j < weights.length; j++) s += (weights[j] ?? 0) * (x[lo + weights.length - 1 - j] ?? 0)
    trend[t] = s
  }
  const points = options.extrapolateTrend === 'period' ? period - 1 : (options.extrapolateTrend ?? 0)
  if (points > 0) extrapolate(trend, points + 1)
  const detrended = Float64Array.from(x, (v, t) =>
    multiplicative ? v / (trend[t] ?? Number.NaN) : v - (trend[t] ?? Number.NaN),
  )
  const phase = new Float64Array(period)
  for (let k = 0; k < period; k++) {
    let s = 0
    let c = 0
    for (let t = k; t < n; t += period) {
      const v = detrended[t] ?? Number.NaN
      if (!Number.isNaN(v)) {
        s += v
        c += 1
      }
    }
    phase[k] = s / c
  }
  const level = phase.reduce((a, b) => a + b, 0) / period
  for (let k = 0; k < period; k++) phase[k] = multiplicative ? (phase[k] ?? 0) / level : (phase[k] ?? 0) - level
  const seasonal = Float64Array.from({ length: n }, (_, t) => phase[t % period] ?? 0)
  const resid = Float64Array.from(detrended, (v, t) =>
    multiplicative ? v / (seasonal[t] ?? 1) : v - (seasonal[t] ?? 0),
  )
  return { observed: Array.from(x), trend: Array.from(trend), seasonal: Array.from(seasonal), resid: Array.from(resid) }
}

/** Fill the NaN ends of a trend with the lines fitted to its first and last `npoints` finite values (statsmodels'). */
function extrapolate(trend: Float64Array, npoints: number): void {
  const n = trend.length
  const front = trend.findIndex((v) => !Number.isNaN(v))
  let back = n - 1
  while (back > 0 && Number.isNaN(trend[back])) back -= 1
  if (front < 0) return
  const line = (from: number, to: number): readonly [number, number] => {
    const idx = Array.from({ length: to - from }, (_, i) => from + i)
    const beta = leastSquares(
      [idx, idx.map(() => 1)],
      idx.map((i) => trend[i] ?? 0),
    )
    return [beta[0] ?? 0, beta[1] ?? 0]
  }
  const frontLast = Math.min(front + npoints, back)
  const backFirst = Math.max(front, back - npoints)
  const [k1, c1] = line(front, frontLast)
  for (let i = 0; i < front; i++) trend[i] = k1 * i + c1
  const [k2, c2] = line(backFirst, back)
  for (let i = back + 1; i < n; i++) trend[i] = k2 * i + c2
}
