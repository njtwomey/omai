/**
 * A feature set for a time series, returned as a labelled vector: the summary statistics, autocorrelation structure,
 * entropy measures (sample, approximate, permutation, spectral), spectral shape, trend and stability measures,
 * crossing counts and strikes, the Hurst exponent and the detrended-fluctuation exponent, and the efficient
 * tsfresh-style measures (`c3`, time-reversal asymmetry, `cid_ce`, absolute energy, mean absolute change). The
 * features are meant to be computed in one pass per series for classification and clustering of series; the names are
 * stable and each feature is defined in its own comment. The set is in the spirit of catch22 (a short list of features
 * that carry most of the signal) and tsfresh (cheap closed-form measures), with the definitions written out here
 * rather than the catch22 C code, so the values are not the pycatch22 numbers.
 */

import { MdmlError, maxOf, meanOf, varianceOf } from '../base/index.ts'
import { approximateEntropy, sampleEntropy } from '../information/index.ts'
import { periodogram } from '../signal/index.ts'
import { detrendedFluctuation, hurstRs } from '../stochastic/index.ts'
import { toFlat } from '../tensor/index.ts'
import { asSeries } from './_series.ts'
import { acf } from './acf.ts'

/** A labelled feature vector. */
export interface FeatureVector {
  readonly names: readonly string[]
  readonly values: readonly number[]
  /** The value of a feature by name. */
  readonly byName: Readonly<Record<string, number>>
}

/** Options of {@link timeSeriesFeatures}. */
export interface FeatureOptions {
  /** Which features: the `'efficient'` set (default) skips the slower fractal estimators; `'all'` includes them. */
  readonly set?: 'efficient' | 'all'
  /** The window of the stability and lumpiness measures (default 10, or the period when given). */
  readonly window?: number
}

const quantileOf = (sorted: readonly number[], q: number): number => {
  const pos = q * (sorted.length - 1)
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  return (sorted[lo] ?? 0) + ((sorted[hi] ?? 0) - (sorted[lo] ?? 0)) * (pos - lo)
}

/**
 * The permutation entropy (Bandt and Pompe): the Shannon entropy of the distribution of ordinal patterns of `order`
 * consecutive values `delay` apart, normalised by log(order!) to lie in [0, 1].
 *
 * @param x - The series.
 * @param order - The pattern length (default 3).
 * @param delay - The spacing of the values in a pattern (default 1).
 * @param normalise - Divide by log(order!) (default true).
 * @returns The entropy: 0 for a monotone series, near 1 for white noise.
 * @example
 * ```ts
 * import { permutationEntropy } from 'mdml-engine/timeseries'
 * permutationEntropy([1, 2, 3, 4, 5, 6, 7], 3) // => 0
 * ```
 */
export function permutationEntropy(x: ArrayLike<number>, order = 3, delay = 1, normalise = true): number {
  if (!(Number.isInteger(order) && order >= 2))
    throw new MdmlError('value', 'permutationEntropy: order must be at least 2')
  const n = x.length - (order - 1) * delay
  if (n <= 0) throw new MdmlError('value', 'permutationEntropy: the series is too short')
  const counts = new Map<string, number>()
  for (let i = 0; i < n; i++) {
    const idx = Array.from({ length: order }, (_, k) => k)
    // Ordinal pattern: the ranks of the window, ties broken by position (stable argsort).
    idx.sort((a, b) => (x[i + a * delay] ?? 0) - (x[i + b * delay] ?? 0) || a - b)
    const key = idx.join(',')
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  let h = 0
  for (const c of counts.values()) h -= (c / n) * Math.log(c / n)
  if (!normalise) return h
  let fact = 1
  for (let k = 2; k <= order; k++) fact *= k
  return h / Math.log(fact)
}

function longestStrike(flags: readonly boolean[], want: boolean): number {
  let best = 0
  let run = 0
  for (const f of flags) {
    run = f === want ? run + 1 : 0
    best = Math.max(best, run)
  }
  return best
}

/**
 * The features of one series as a labelled vector. Features are computed on the series as given, except those that
 * name a standardised series (entropies and spectral measures use the z-scored series so that they do not depend on
 * the level or scale).
 *
 * @param y - The series (at least 16 observations).
 * @param options - The set and the window.
 * @returns The names, values and a lookup by name.
 * @example
 * ```ts
 * import { timeSeriesFeatures } from 'mdml-engine/timeseries'
 * const x = Array.from({ length: 64 }, (_, t) => Math.sin(t / 3) + 0.1 * Math.cos(7 * t))
 * (timeSeriesFeatures(x).byName['mean_crossings'] ?? 0) > 0 // => true
 * ```
 */
export function timeSeriesFeatures(y: ArrayLike<number>, options: FeatureOptions = {}): FeatureVector {
  const x = Array.from(asSeries(Float64Array.from(y), 'timeSeriesFeatures'))
  const n = x.length
  if (n < 16) throw new MdmlError('value', 'timeSeriesFeatures: need at least 16 observations')
  const mean = meanOf(x)
  const variance = varianceOf(x, 0)
  const sd = Math.sqrt(variance)
  const z = sd > 0 ? x.map((v) => (v - mean) / sd) : x.map(() => 0)
  const sorted = [...x].sort((a, b) => a - b)
  const f: Record<string, number> = {}

  // Location, spread and shape.
  f['mean'] = mean
  f['std'] = sd
  f['min'] = sorted[0] ?? 0
  f['max'] = sorted[n - 1] ?? 0
  f['median'] = quantileOf(sorted, 0.5)
  f['iqr'] = quantileOf(sorted, 0.75) - quantileOf(sorted, 0.25)
  f['skewness'] = sd > 0 ? z.reduce((a, v) => a + v ** 3, 0) / n : 0
  f['kurtosis'] = sd > 0 ? z.reduce((a, v) => a + v ** 4, 0) / n - 3 : 0
  f['ratio_beyond_1_sigma'] = x.filter((v) => Math.abs(v - mean) > sd).length / n

  // Autocorrelation: lag 1 and 2, where it first drops below 1/e and first turns up, the sum of squares of the first ten.
  const lags = Math.min(n - 1, Math.max(10, Math.floor(n / 4)))
  const r = Array.from(acf(x, lags).acf)
  f['acf_lag1'] = r[1] ?? 0
  f['acf_lag2'] = r[2] ?? 0
  let eCrossing = lags
  for (let k = 1; k <= lags; k++)
    if ((r[k] ?? 0) < Math.exp(-1)) {
      eCrossing = k
      break
    }
  f['acf_first_e_crossing'] = eCrossing
  let firstMin = lags
  for (let k = 1; k < lags; k++)
    if ((r[k] ?? 0) < (r[k - 1] ?? 0) && (r[k] ?? 0) <= (r[k + 1] ?? 0)) {
      firstMin = k
      break
    }
  f['acf_first_min'] = firstMin
  f['acf_sumsq_10'] = r.slice(1, 11).reduce((a, v) => a + v * v, 0)

  // Entropies of the standardised series.
  f['sample_entropy'] = sd > 0 ? sampleEntropy(z) : 0
  f['approximate_entropy'] = sd > 0 ? approximateEntropy(z) : 0
  f['permutation_entropy'] = permutationEntropy(x, 3)

  // Spectral shape of the standardised series: dominant frequency, centroid, power below a fifth of Nyquist, entropy.
  const p = periodogram(z, { detrend: 'constant', scaling: 'spectrum' })
  const freq = Array.from(toFlat(p.frequencies))
  const power = Array.from(toFlat(p.values)).slice(1)
  const fr = freq.slice(1)
  const total = power.reduce((a, v) => a + v, 0) || 1
  f['dominant_frequency'] = fr[power.indexOf(maxOf(power))] ?? 0
  f['spectral_centroid'] = fr.reduce((a, v, k) => a + v * (power[k] ?? 0), 0) / total
  f['spectral_low_power_ratio'] =
    power.slice(0, Math.max(1, Math.floor(power.length / 5))).reduce((a, v) => a + v, 0) / total
  f['spectral_entropy'] =
    power.reduce((a, v) => (v > 0 ? a - (v / total) * Math.log(v / total) : a), 0) / Math.log(power.length)

  // Trend and stability.
  const t = Array.from({ length: n }, (_, i) => i - (n - 1) / 2)
  const stt = t.reduce((a, v) => a + v * v, 0)
  const slope = t.reduce((a, v, i) => a + v * ((x[i] ?? 0) - mean), 0) / stt
  f['trend_slope'] = slope
  f['trend_r2'] = variance > 0 ? (slope * slope * stt) / (n * variance) : 0
  const w = Math.max(2, Math.min(options.window ?? 10, Math.floor(n / 2)))
  const means: number[] = []
  const vars: number[] = []
  for (let s = 0; s + w <= n; s += w) {
    const blk = x.slice(s, s + w)
    means.push(meanOf(blk))
    vars.push(varianceOf(blk, 0))
  }
  f['stability'] = varianceOf(means, 0)
  f['lumpiness'] = varianceOf(vars, 0)
  f['max_level_shift'] = Math.max(0, maxOf(means.slice(1).map((m, k) => Math.abs(m - (means[k] ?? 0)))))
  f['max_variance_shift'] = Math.max(0, maxOf(vars.slice(1).map((m, k) => Math.abs(m - (vars[k] ?? 0)))))

  // Crossings, peaks and strikes.
  const above = x.map((v) => v > mean)
  f['mean_crossings'] = above.slice(1).filter((a, k) => a !== above[k]).length / (n - 1)
  f['longest_strike_above_mean'] = longestStrike(above, true)
  f['longest_strike_below_mean'] = longestStrike(above, false)
  f['number_peaks'] = x.slice(1, -1).filter((v, k) => v > (x[k] ?? 0) && v > (x[k + 2] ?? 0)).length
  f['count_above_mean'] = above.filter(Boolean).length / n
  f['first_location_of_max'] = x.indexOf(f['max'] as number) / n
  f['last_location_of_min'] = (x.lastIndexOf(f['min'] as number) + 1) / n

  // Cheap closed-form measures in the style of tsfresh.
  const d = x.slice(1).map((v, k) => v - (x[k] ?? 0))
  f['mean_change'] = ((x[n - 1] ?? 0) - (x[0] ?? 0)) / (n - 1)
  f['mean_abs_change'] = d.reduce((a, v) => a + Math.abs(v), 0) / (n - 1)
  f['absolute_energy'] = x.reduce((a, v) => a + v * v, 0)
  f['cid_ce'] = Math.sqrt(d.reduce((a, v) => a + v * v, 0))
  f['c3_lag1'] = x.slice(0, n - 2).reduce((a, v, k) => a + v * (x[k + 1] ?? 0) * (x[k + 2] ?? 0), 0) / (n - 2)
  f['time_reversal_asymmetry_lag1'] =
    x.slice(0, n - 2).reduce((a, v, k) => a + (x[k + 2] ?? 0) ** 2 * (x[k + 1] ?? 0) - (x[k + 1] ?? 0) * v * v, 0) /
    (n - 2)
  f['fraction_unique'] = new Set(x).size / n

  if (options.set === 'all') {
    f['hurst_rs'] = hurstRs(x)
    f['dfa_exponent'] = detrendedFluctuation(x)
  }
  const names = Object.keys(f)
  return { names, values: names.map((k) => f[k] ?? Number.NaN), byName: f }
}

/**
 * The feature matrix of several series, one row per series, with the same columns for all.
 *
 * @param series - The series (each at least 16 observations).
 * @param options - The set and window.
 * @returns The names and the rows.
 * @example
 * ```ts
 * import { featureMatrix } from 'mdml-engine/timeseries'
 * const a = Array.from({ length: 32 }, (_, t) => Math.sin(t))
 * const b = Array.from({ length: 32 }, (_, t) => t)
 * featureMatrix([a, b]).rows.length // => 2
 * ```
 */
export function featureMatrix(
  series: readonly ArrayLike<number>[],
  options: FeatureOptions = {},
): { readonly names: readonly string[]; readonly rows: readonly (readonly number[])[] } {
  const all = series.map((s) => timeSeriesFeatures(s, options))
  return { names: all[0]?.names ?? [], rows: all.map((v) => v.values) }
}
