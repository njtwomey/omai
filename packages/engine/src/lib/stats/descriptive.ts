/**
 * Descriptive statistics of a sample, numpy and scipy.stats style: mean, variance and standard deviation (optionally
 * weighted), central moments, skewness and kurtosis with or without bias correction, the `describe` summary, z-scores,
 * the standard error of the mean, the mode, and the geometric and harmonic means.
 */

import { MdmlError, meanOf, pairwiseSum, varianceOf } from '../base/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { sample, vec } from './_util.ts'

/** Sample weights: frequency weights (counts, the default) or reliability weights (inverse variances). */
export interface WeightOptions {
  /** One non-negative weight per observation. */
  readonly weights?: TensorLike
  /**
   * How `ddof` uses the weights: `'frequency'` divides by sum(w) - ddof (statsmodels DescrStatsW), `'reliability'` by
   * sum(w) - ddof sum(w^2) / sum(w) (numpy.cov aweights).
   */
  readonly weightKind?: 'frequency' | 'reliability'
}

/** Options of the variance family. */
export interface VarianceOptions extends WeightOptions {
  /** Delta degrees of freedom (default 0, numpy's convention; 1 for the unbiased sample variance). */
  readonly ddof?: number
}

function weightsOf(name: string, w: TensorLike | undefined, n: number): Float64Array | undefined {
  if (w === undefined) return undefined
  const out = sample(name, w)
  if (out.length !== n) throw new MdmlError('shape', `${name}: weights must match the data`)
  for (const v of out) if (!(v >= 0)) throw new MdmlError('domain', `${name}: weights must be non-negative`)
  return out
}

function weightedMeanOf(x: Float64Array, w: Float64Array): number {
  const p = new Float64Array(x.length)
  for (let i = 0; i < x.length; i++) p[i] = (w[i] ?? 0) * (x[i] ?? 0)
  return pairwiseSum(p) / pairwiseSum(w)
}

/** Central moment of order k about c, weighted or not. */
function momentAbout(x: Float64Array, k: number, c: number, w?: Float64Array): number {
  const p = new Float64Array(x.length)
  for (let i = 0; i < x.length; i++) p[i] = (w === undefined ? 1 : (w[i] ?? 0)) * ((x[i] ?? 0) - c) ** k
  return pairwiseSum(p) / (w === undefined ? x.length : pairwiseSum(w))
}

/**
 * The arithmetic mean (numpy.mean, numpy.average with weights), by pairwise summation.
 *
 * @param x - The sample (flattened).
 * @param options - Optional weights.
 * @returns The mean.
 * @example
 * ```ts
 * import { mean } from 'mdml-engine/stats'
 * mean([1, 2, 3, 6]) // => 3
 * mean([1, 3], { weights: [3, 1] }) // => 1.5
 * ```
 */
export function mean(x: TensorLike, options: WeightOptions = {}): number {
  const a = sample('mean', x)
  const w = weightsOf('mean', options.weights, a.length)
  return w === undefined ? meanOf(a) : weightedMeanOf(a, w)
}

/**
 * The variance by the two-pass algorithm (numpy.var), dividing by n - ddof; with weights, statsmodels DescrStatsW's
 * frequency-weighted variance or the reliability-weighted one.
 *
 * @param x - The sample.
 * @param options - ddof (default 0) and weights.
 * @returns The variance (`NaN` when the divisor is not positive).
 * @example
 * ```ts
 * import { variance } from 'mdml-engine/stats'
 * variance([1, 2, 3, 4]) // => 1.25
 * variance([1, 2, 3, 4], { ddof: 1 }) // => 1.6666666666666667
 * variance([1, 2, 3], { weights: [1, 2, 1], ddof: 1 }) // => 0.6666666666666666
 * ```
 */
export function variance(x: TensorLike, options: VarianceOptions = {}): number {
  const a = sample('variance', x)
  const ddof = options.ddof ?? 0
  const w = weightsOf('variance', options.weights, a.length)
  if (w === undefined) return varianceOf(a, ddof)
  const m = weightedMeanOf(a, w)
  const sw = pairwiseSum(w)
  const ss = momentAbout(a, 2, m, w) * sw
  let denominator = sw - ddof
  if (options.weightKind === 'reliability') {
    let sw2 = 0
    for (const v of w) sw2 += v * v
    denominator = sw - (ddof * sw2) / sw
  }
  return denominator > 0 ? ss / denominator : Number.NaN
}

/**
 * The standard deviation, the square root of {@link variance}.
 *
 * @param x - The sample.
 * @param options - ddof (default 0) and weights.
 * @returns The standard deviation.
 * @example
 * ```ts
 * import { std } from 'mdml-engine/stats'
 * std([2, 4, 4, 4, 5, 5, 7, 9]) // => 2
 * ```
 */
export function std(x: TensorLike, options: VarianceOptions = {}): number {
  return Math.sqrt(variance(x, options))
}

/** Options of {@link moment}. */
export interface MomentOptions extends WeightOptions {
  /** The centre (default the mean). */
  readonly center?: number
}

/**
 * The k-th moment about a centre (scipy.stats.moment; default the mean), sum w (x - c)^k / sum w.
 *
 * @param x - The sample.
 * @param order - The order k (a non-negative integer).
 * @param options - Centre and weights.
 * @returns The moment; order 0 is 1 and order 1 about the mean is 0.
 * @example
 * ```ts
 * import { moment } from 'mdml-engine/stats'
 * moment([1, 2, 3, 4, 5], 2) // => 2
 * moment([1, 2, 3], 1, { center: 0 }) // => 2
 * ```
 */
export function moment(x: TensorLike, order: number, options: MomentOptions = {}): number {
  if (!(Number.isInteger(order) && order >= 0)) throw new MdmlError('value', 'moment: order must be an integer >= 0')
  const a = sample('moment', x)
  const w = weightsOf('moment', options.weights, a.length)
  if (order === 0) return 1
  const c = options.center ?? (w === undefined ? meanOf(a) : weightedMeanOf(a, w))
  if (order === 1 && options.center === undefined) return 0
  return momentAbout(a, order, c, w)
}

/** Options of {@link skewness} and {@link kurtosis}. */
export interface ShapeOptions extends WeightOptions {
  /** Use the biased moment estimator (default true, scipy's default); false applies the small-sample correction. */
  readonly bias?: boolean
}

/**
 * The sample skewness g1 = m3 / m2^(3/2) (scipy.stats.skew); with `bias: false` the adjusted Fisher-Pearson G1 =
 * g1 sqrt(n (n - 1)) / (n - 2).
 *
 * @param x - The sample.
 * @param options - bias (default true) and weights (moment ratios of the weighted moments).
 * @returns The skewness (`NaN` for a constant sample).
 * @example
 * ```ts
 * import { skewness } from 'mdml-engine/stats'
 * skewness([1, 2, 3]) // => 0
 * skewness([1, 1, 1, 4]) // => 1.1547005383792515
 * ```
 */
export function skewness(x: TensorLike, options: ShapeOptions = {}): number {
  const a = sample('skewness', x)
  const w = weightsOf('skewness', options.weights, a.length)
  const m = w === undefined ? meanOf(a) : weightedMeanOf(a, w)
  const m2 = momentAbout(a, 2, m, w)
  const m3 = momentAbout(a, 3, m, w)
  if (m2 <= (1e-15 * m) ** 2) return Number.NaN
  const g1 = m3 / m2 ** 1.5
  const n = a.length
  if (options.bias !== false) return g1
  if (n < 3) return Number.NaN
  return (g1 * Math.sqrt(n * (n - 1))) / (n - 2)
}

/** Options of {@link kurtosis}. */
export interface KurtosisOptions extends ShapeOptions {
  /** Fisher's definition, excess kurtosis (default true); false gives Pearson's m4 / m2^2. */
  readonly fisher?: boolean
}

/**
 * The sample kurtosis (scipy.stats.kurtosis): Fisher's excess m4 / m2^2 - 3 by default, Pearson's without the 3; with
 * `bias: false` the unbiased estimator under normality, ((n^2 - 1) m4 / m2^2 - 3 (n - 1)^2) / ((n - 2)(n - 3)).
 *
 * @param x - The sample.
 * @param options - fisher (default true), bias (default true) and weights.
 * @returns The kurtosis (`NaN` for a constant sample).
 * @example
 * ```ts
 * import { kurtosis } from 'mdml-engine/stats'
 * kurtosis([1, 2, 3, 4]) // => -1.36
 * kurtosis([1, 2, 3, 4], { fisher: false }) // => 1.64
 * ```
 */
export function kurtosis(x: TensorLike, options: KurtosisOptions = {}): number {
  const a = sample('kurtosis', x)
  const w = weightsOf('kurtosis', options.weights, a.length)
  const m = w === undefined ? meanOf(a) : weightedMeanOf(a, w)
  const m2 = momentAbout(a, 2, m, w)
  const m4 = momentAbout(a, 4, m, w)
  if (m2 <= (1e-15 * m) ** 2) return Number.NaN
  const n = a.length
  let g = m4 / (m2 * m2)
  if (options.bias === false) {
    if (n < 4) return Number.NaN
    g = ((n * n - 1) * g - 3 * (n - 1) ** 2) / ((n - 2) * (n - 3)) + 3
  }
  return options.fisher === false ? g : g - 3
}

/** The summary of {@link describe}. */
export interface Description {
  readonly n: number
  readonly min: number
  readonly max: number
  readonly mean: number
  /** The sample variance (ddof 1 by default). */
  readonly variance: number
  readonly skewness: number
  /** Excess kurtosis. */
  readonly kurtosis: number
}

/**
 * Summary statistics (scipy.stats.describe): size, range, mean, variance (ddof 1), skewness and excess kurtosis
 * (biased by default).
 *
 * @param x - The sample.
 * @param options - ddof of the variance (default 1) and bias of the shape statistics (default true).
 * @returns The description.
 * @example
 * ```ts
 * import { describe } from 'mdml-engine/stats'
 * const d = describe([1, 2, 3, 4, 5])
 * [d.n, d.min, d.max, d.mean, d.variance, d.skewness] // => [5, 1, 5, 3, 2.5, 0]
 * ```
 */
export function describe(
  x: TensorLike,
  options: { readonly ddof?: number; readonly bias?: boolean } = {},
): Description {
  const a = sample('describe', x)
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of a) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  const bias = options.bias ?? true
  return {
    n: a.length,
    min: lo,
    max: hi,
    mean: meanOf(a),
    variance: varianceOf(a, options.ddof ?? 1),
    skewness: skewness(a, { bias }),
    kurtosis: kurtosis(a, { bias }),
  }
}

/**
 * Standard scores (x - mean) / std (scipy.stats.zscore).
 *
 * @param x - The sample.
 * @param options - ddof of the standard deviation (default 0).
 * @returns The z-scores, one per observation.
 * @example
 * ```ts
 * import { zscore } from 'mdml-engine/stats'
 * zscore([1, 2, 3]) // => [-1.224745, 0, 1.224745]
 * ```
 */
export function zscore(x: TensorLike, options: { readonly ddof?: number } = {}): Tensor {
  const a = sample('zscore', x)
  const m = meanOf(a)
  const s = Math.sqrt(varianceOf(a, options.ddof ?? 0))
  return vec(Array.from(a, (v) => (v - m) / s))
}

/**
 * The standard error of the mean, std(ddof) / sqrt(n) (scipy.stats.sem).
 *
 * @param x - The sample.
 * @param options - ddof (default 1).
 * @returns The standard error.
 * @example
 * ```ts
 * import { sem } from 'mdml-engine/stats'
 * sem([1, 2, 3, 4]) // => 0.6454972243679028
 * ```
 */
export function sem(x: TensorLike, options: { readonly ddof?: number } = {}): number {
  const a = sample('sem', x)
  return Math.sqrt(varianceOf(a, options.ddof ?? 1) / a.length)
}

/** The result of {@link mode}. */
export interface ModeResult {
  readonly mode: number
  readonly count: number
}

/**
 * The most frequent value and its count (scipy.stats.mode); ties go to the smallest value.
 *
 * @param x - The sample.
 * @returns The mode and its count.
 * @example
 * ```ts
 * import { mode } from 'mdml-engine/stats'
 * mode([3, 1, 3, 2, 1]) // => { mode: 1, count: 2 }
 * ```
 */
export function mode(x: TensorLike): ModeResult {
  const a = Float64Array.from(sample('mode', x)).sort()
  let best = a[0] ?? Number.NaN
  let bestCount = 0
  let i = 0
  while (i < a.length) {
    let j = i
    while (j + 1 < a.length && a[j + 1] === a[i]) j++
    if (j - i + 1 > bestCount) {
      bestCount = j - i + 1
      best = a[i] ?? Number.NaN
    }
    i = j + 1
  }
  return { mode: best, count: bestCount }
}

function positiveSample(name: string, x: TensorLike): Float64Array {
  const a = sample(name, x)
  for (const v of a) if (v < 0) throw new MdmlError('domain', `${name}: values must be non-negative`)
  return a
}

/**
 * The geometric mean exp(mean(log x)) (scipy.stats.gmean), optionally weighted.
 *
 * @param x - Non-negative values (a zero gives 0).
 * @param options - Optional weights.
 * @returns The geometric mean.
 * @throws MdmlError with code `domain` for negative values.
 * @example
 * ```ts
 * import { geometricMean } from 'mdml-engine/stats'
 * geometricMean([1, 4]) // => 2
 * geometricMean([1, 2, 4]) // => 2
 * ```
 */
export function geometricMean(x: TensorLike, options: { readonly weights?: TensorLike } = {}): number {
  const a = positiveSample('geometricMean', x)
  const logs = Float64Array.from(a, Math.log)
  const w = weightsOf('geometricMean', options.weights, a.length)
  return Math.exp(w === undefined ? meanOf(logs) : weightedMeanOf(logs, w))
}

/**
 * The harmonic mean n / sum(1 / x) (scipy.stats.hmean), optionally weighted: sum(w) / sum(w / x).
 *
 * @param x - Non-negative values (a zero gives 0).
 * @param options - Optional weights.
 * @returns The harmonic mean.
 * @throws MdmlError with code `domain` for negative values.
 * @example
 * ```ts
 * import { harmonicMean } from 'mdml-engine/stats'
 * harmonicMean([1, 4, 4]) // => 2
 * ```
 */
export function harmonicMean(x: TensorLike, options: { readonly weights?: TensorLike } = {}): number {
  const a = positiveSample('harmonicMean', x)
  const w = weightsOf('harmonicMean', options.weights, a.length)
  for (const v of a) if (v === 0) return 0
  const inv = Float64Array.from(a, (v, i) => (w === undefined ? 1 : (w[i] ?? 0)) / v)
  return (w === undefined ? a.length : pairwiseSum(w)) / pairwiseSum(inv)
}

/**
 * The Gini coefficient of a sample of non-negative values: half the mean absolute difference between two draws,
 * relative to the mean, `sum_i sum_j |x_i - x_j| / (2 n^2 mean)`. 0 is perfect equality; a single holder of everything
 * gives `(n - 1) / n`.
 *
 * @param x - Non-negative values (wealth, say).
 * @returns The coefficient, in [0, (n - 1) / n]; 0 when every value is 0.
 * @throws MdmlError with code `domain` for negative values.
 * @concept inequality
 * @example
 * ```ts
 * import { gini } from 'mdml-engine/stats'
 * gini([1, 1, 1, 1]) // => 0
 * gini([0, 0, 0, 4]) // => 0.75
 * ```
 */
export function gini(x: TensorLike): number {
  const a = positiveSample('gini', x)
  const n = a.length
  a.sort()
  let total = 0
  let weighted = 0
  for (let i = 0; i < n; i++) {
    total += a[i] ?? 0
    weighted += (i + 1) * (a[i] ?? 0)
  }
  return total === 0 ? 0 : (2 * weighted) / (n * total) - (n + 1) / n
}
