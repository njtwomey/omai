/**
 * Order statistics of a sample: quantiles by any of the nine Hyndman-Fan definitions (numpy.quantile), the median, the
 * interquartile range, the percentile rank of a score, the empirical CDF with Greenwood intervals and the median
 * absolute deviation, as numpy and scipy.stats.
 */

import { WhiteboxError } from '../base/index.ts'
import { asTensor, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { normPpf, sample, sortedCopy, vec } from './_util.ts'

/**
 * The nine sample-quantile definitions of Hyndman and Fan (1996), by numpy's names or by their numbers 1 to 9:
 * discontinuous `inverted_cdf` (1), `averaged_inverted_cdf` (2), `closest_observation` (3); continuous
 * `interpolated_inverted_cdf` (4), `hazen` (5), `weibull` (6), `linear` (7, the default of numpy and R), `median_unbiased`
 * (8, recommended by Hyndman and Fan) and `normal_unbiased` (9).
 */
export type QuantileMethod =
  | 'inverted_cdf'
  | 'averaged_inverted_cdf'
  | 'closest_observation'
  | 'interpolated_inverted_cdf'
  | 'hazen'
  | 'weibull'
  | 'linear'
  | 'median_unbiased'
  | 'normal_unbiased'
  | 1
  | 2
  | 3
  | 4
  | 5
  | 6
  | 7
  | 8
  | 9

const NAMES: readonly QuantileMethod[] = [
  'inverted_cdf',
  'averaged_inverted_cdf',
  'closest_observation',
  'interpolated_inverted_cdf',
  'hazen',
  'weibull',
  'linear',
  'median_unbiased',
  'normal_unbiased',
]

// (alpha, beta) plotting positions of the continuous methods (numpy's _compute_virtual_index).
const CONTINUOUS: Readonly<Record<string, readonly [number, number]>> = {
  interpolated_inverted_cdf: [0, 1],
  hazen: [0.5, 0.5],
  weibull: [0, 0],
  linear: [1, 1],
  median_unbiased: [1 / 3, 1 / 3],
  normal_unbiased: [3 / 8, 3 / 8],
}

function methodName(method: QuantileMethod): string {
  const name = typeof method === 'number' ? NAMES[method - 1] : method
  if (name === undefined || !NAMES.includes(name))
    throw new WhiteboxError('value', `quantile: unknown method ${method}`)
  return String(name)
}

// numpy's _lerp: exact at both ends.
function lerp(a: number, b: number, t: number): number {
  const d = b - a
  return t >= 0.5 ? b - d * (1 - t) : a + d * t
}

/** The quantile of an ascending, NaN-free array (numpy.quantile's algorithm for one level). */
function quantileSorted(s: Float64Array, q: number, name: string): number {
  const n = s.length
  if (n === 0) return Number.NaN
  const at = (i: number): number => s[Math.min(n - 1, Math.max(0, i))] ?? Number.NaN
  if (name === 'inverted_cdf' || name === 'closest_observation') {
    const index = name === 'inverted_cdf' ? n * q - 1 : n * q - 1.5
    const prev = Math.floor(index)
    const g = index - prev
    const take = name === 'inverted_cdf' ? g === 0 : g === 0 && Math.abs(prev % 2) === 1
    return at(take ? prev : prev + 1)
  }
  const index = name === 'averaged_inverted_cdf' ? n * q - 1 : virtual(n, q, CONTINUOUS[name] ?? [1, 1])
  if (index >= n - 1) return at(n - 1)
  if (index < 0) return at(0)
  const prev = Math.floor(index)
  let g = index - prev
  if (name === 'averaged_inverted_cdf') g = g === 0 ? 0.5 : 1
  return lerp(at(prev), at(prev + 1), g)
}

function virtual(n: number, q: number, [alpha, beta]: readonly [number, number]): number {
  return n * q + (alpha + q * (1 - alpha - beta)) - 1
}

function levels(q: number | TensorLike): { values: number[]; scalar: boolean } {
  const scalar = typeof q === 'number'
  const values = scalar ? [q] : toFlat(asTensor(q))
  for (const v of values) if (!(v >= 0 && v <= 1)) throw new WhiteboxError('value', 'quantile: q must lie in [0, 1]')
  return { values, scalar }
}

/** Options of {@link quantile}. */
export interface QuantileOptions {
  /** The Hyndman-Fan definition (default `'linear'`). */
  readonly method?: QuantileMethod
}

/**
 * Sample quantiles by any of the nine Hyndman-Fan definitions (numpy.quantile); a NaN in the data gives NaN.
 *
 * @param x - The sample (flattened).
 * @param q - A level in [0, 1], or several.
 * @param options - method (default `'linear'`).
 * @returns The quantile (a number for a single level, a tensor for several).
 * @example
 * ```ts
 * import { quantile } from 'whitebox-engine/stats'
 * quantile([1, 2, 3, 4], 0.5) // => 2.5
 * quantile([1, 2, 3, 4], 0.5, { method: 'inverted_cdf' }) // => 2
 * quantile([1, 2, 3, 4], [0.25, 0.75], { method: 'hazen' }) // => [1.5, 3.5]
 * ```
 */
export function quantile(x: TensorLike, q: number, options?: QuantileOptions): number
export function quantile(x: TensorLike, q: TensorLike, options?: QuantileOptions): Tensor
export function quantile(x: TensorLike, q: number | TensorLike, options: QuantileOptions = {}): number | Tensor {
  const name = methodName(options.method ?? 'linear')
  const data = sample('quantile', x, 0)
  const { values, scalar } = levels(q)
  const hasNaN = data.some(Number.isNaN)
  const s = sortedCopy(data)
  const out = values.map((v) => (hasNaN ? Number.NaN : quantileSorted(s, v, name)))
  return scalar ? (out[0] ?? Number.NaN) : vec(out)
}

/**
 * The median (numpy.median).
 *
 * @param x - The sample.
 * @returns The median.
 * @example
 * ```ts
 * import { median } from 'whitebox-engine/stats'
 * median([3, 1, 2]) // => 2
 * median([4, 1, 3, 2]) // => 2.5
 * ```
 */
export function median(x: TensorLike): number {
  return quantile(x, 0.5)
}

/** Options of {@link iqr}. */
export interface IqrOptions extends QuantileOptions {
  /** The percentiles spanned (default [25, 75]). */
  readonly rng?: readonly [number, number]
  /** Divide by this, or by 2 sqrt(2) erfinv(1/2) ~ 1.349 with `'normal'`, to estimate a normal standard deviation. */
  readonly scale?: number | 'normal'
}

/**
 * The interquartile range (scipy.stats.iqr): the spread between two percentiles, optionally scaled to estimate the
 * standard deviation of a normal sample.
 *
 * @param x - The sample.
 * @param options - Percentiles (default [25, 75]), scale and quantile method.
 * @returns The range.
 * @example
 * ```ts
 * import { iqr } from 'whitebox-engine/stats'
 * iqr([1, 2, 3, 4, 5, 6, 7, 8, 9]) // => 4
 * ```
 */
export function iqr(x: TensorLike, options: IqrOptions = {}): number {
  const [lo, hi] = options.rng ?? [25, 75]
  if (!(lo >= 0 && hi <= 100 && lo <= hi))
    throw new WhiteboxError('value', 'iqr: rng must satisfy 0 <= lo <= hi <= 100')
  const method = options.method ?? 'linear'
  const r = quantile(x, [lo / 100, hi / 100], { method })
  const [a, b] = toFlat(r)
  const scale = options.scale === 'normal' ? 2 * normPpf(0.75) : (options.scale ?? 1)
  return ((b ?? Number.NaN) - (a ?? Number.NaN)) / scale
}

/** How {@link percentileOfScore} counts ties with the score. */
export type PercentileKind = 'rank' | 'weak' | 'strict' | 'mean'

/**
 * The percentile rank of a score relative to a sample (scipy.stats.percentileofscore): `'weak'` counts values <= the
 * score, `'strict'` values < it, `'mean'` averages the two and `'rank'` (default) averages the ranks a tied score
 * would take.
 *
 * @param x - The sample.
 * @param score - The score.
 * @param options - kind (default `'rank'`).
 * @returns A percentage in [0, 100].
 * @example
 * ```ts
 * import { percentileOfScore } from 'whitebox-engine/stats'
 * percentileOfScore([1, 2, 3, 4], 3) // => 75
 * percentileOfScore([1, 2, 3, 3, 4], 3, { kind: 'strict' }) // => 40
 * ```
 */
export function percentileOfScore(
  x: TensorLike,
  score: number,
  options: { readonly kind?: PercentileKind } = {},
): number {
  const a = sample('percentileOfScore', x)
  let left = 0
  let right = 0
  for (const v of a) {
    if (v < score) left++
    if (v <= score) right++
  }
  const n = a.length
  switch (options.kind ?? 'rank') {
    case 'strict':
      return (left / n) * 100
    case 'weak':
      return (right / n) * 100
    case 'mean':
      return ((left + right) * 50) / n
    default:
      return ((left + right + (right > left ? 1 : 0)) * 50) / n
  }
}

/** An empirical distribution function. */
export interface Ecdf {
  /** The distinct observed values, ascending. */
  readonly quantiles: Tensor
  /** The ECDF at each distinct value. */
  readonly probabilities: Tensor
  /** Evaluate the step function at points. */
  evaluate(t: TensorLike): Tensor
  /**
   * Pointwise confidence limits at the distinct values (scipy's `confidence_interval`): Greenwood's variance on the
   * survival scale, `'linear'` (default) or `'log-log'`.
   */
  confidenceInterval(level?: number, method?: 'linear' | 'log-log'): { readonly low: Tensor; readonly high: Tensor }
}

/**
 * The empirical cumulative distribution function (scipy.stats.ecdf for uncensored data), with Greenwood pointwise
 * intervals.
 *
 * @param x - The sample.
 * @returns The ECDF.
 * @example
 * ```ts
 * import { ecdf } from 'whitebox-engine/stats'
 * const F = ecdf([1, 3, 3, 4])
 * F.probabilities // => [0.25, 0.75, 1]
 * F.evaluate([0, 3, 3.5]) // => [0, 0.75, 0.75]
 * ```
 */
export function ecdf(x: TensorLike): Ecdf {
  const s = sortedCopy(sample('ecdf', x))
  const n = s.length
  const values: number[] = []
  const events: number[] = []
  for (let i = 0; i < n; i++) {
    if (i > 0 && s[i] === s[i - 1]) events[events.length - 1] = (events[events.length - 1] ?? 0) + 1
    else {
      values.push(s[i] ?? 0)
      events.push(1)
    }
  }
  const cumulative: number[] = []
  let c = 0
  for (const d of events) {
    c += d
    cumulative.push(c / n)
  }
  const evaluate = (t: TensorLike): Tensor =>
    vec(
      toFlat(asTensor(t)).map((v) => {
        let lo = 0
        let hi = n
        while (lo < hi) {
          const mid = (lo + hi) >> 1
          if ((s[mid] ?? 0) <= v) lo = mid + 1
          else hi = mid
        }
        return lo / n
      }),
    )
  const confidenceInterval = (
    level = 0.95,
    method: 'linear' | 'log-log' = 'linear',
  ): { readonly low: Tensor; readonly high: Tensor } => {
    const z = normPpf(0.5 + level / 2)
    const low: number[] = []
    const high: number[] = []
    let atRisk = n
    let greenwood = 0
    for (let i = 0; i < values.length; i++) {
      const d = events[i] ?? 0
      greenwood += d / (atRisk * (atRisk - d))
      atRisk -= d
      const sf = 1 - (cumulative[i] ?? 0)
      let lo: number
      let hi: number
      if (method === 'linear') {
        const half = z * Math.sqrt(sf * sf * greenwood)
        lo = Math.min(1, Math.max(0, sf - half))
        hi = Math.min(1, Math.max(0, sf + half))
      } else {
        const half = z * Math.sqrt(greenwood / Math.log(sf) ** 2)
        const lnl = Math.log(-Math.log(sf))
        lo = Math.exp(-Math.exp(lnl + half))
        hi = Math.exp(-Math.exp(lnl - half))
      }
      low.push(1 - hi)
      high.push(1 - lo)
    }
    return { low: vec(low), high: vec(high) }
  }
  return { quantiles: vec(values), probabilities: vec(cumulative), evaluate, confidenceInterval }
}

/** Options of {@link medianAbsoluteDeviation}. */
export interface MadOptions {
  /** The centre (default the median). */
  readonly center?: number
  /** Divide by this, or by Phi^-1(3/4) ~ 0.6745 with `'normal'` to estimate a normal standard deviation. */
  readonly scale?: number | 'normal'
}

/**
 * The median absolute deviation median(|x - c|) / scale (scipy.stats.median_abs_deviation).
 *
 * @param x - The sample.
 * @param options - Centre (default the median) and scale (default 1).
 * @returns The MAD.
 * @example
 * ```ts
 * import { medianAbsoluteDeviation } from 'whitebox-engine/stats'
 * medianAbsoluteDeviation([1, 2, 3, 4, 100]) // => 1
 * ```
 */
export function medianAbsoluteDeviation(x: TensorLike, options: MadOptions = {}): number {
  const a = sample('medianAbsoluteDeviation', x)
  const c = options.center ?? median(a)
  const scale = options.scale === 'normal' ? normPpf(0.75) : (options.scale ?? 1)
  return median(Float64Array.from(a, (v) => Math.abs(v - c))) / scale
}
