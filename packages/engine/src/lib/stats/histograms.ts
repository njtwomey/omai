/**
 * Histograms and binning, numpy and scipy style: bin edges by count, explicit edges or the automatic rules
 * (Freedman-Diaconis, Scott, Sturges, Knuth and others), one- and two-dimensional histograms as counts or densities,
 * and `binnedStatistic`, any statistic of the values falling in each bin.
 */

import { meanOf, OmaiError, varianceOf } from '../base/index.ts'
import { lgamma } from '../numerics/special/index.ts'
import { asTensor, fromStorage, item, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { mat, sample, sortedCopy, vec } from './_util.ts'
import { median, quantile } from './quantiles.ts'

/**
 * A rule for the number of equal-width bins (numpy.histogram_bin_edges, plus Knuth's): `'auto'` (the smaller width of
 * Freedman-Diaconis and Sturges), `'fd'` (Freedman-Diaconis, 2 IQR n^(-1/3)), `'scott'` ((24 sqrt(pi) / n)^(1/3) sd),
 * `'sturges'` (range / (log2 n + 1)), `'doane'` (Sturges corrected for skewness), `'rice'`, `'sqrt'` and `'knuth'` (the
 * number of bins maximising Knuth's Bayesian posterior).
 */
export type BinRule = 'auto' | 'fd' | 'scott' | 'sturges' | 'doane' | 'rice' | 'sqrt' | 'knuth'

/** Bins: a count of equal-width bins, a rule, or explicit monotone edges. */
export type Bins = number | BinRule | TensorLike

/** Options of {@link histogramBinEdges} and {@link histogram}. */
export interface HistogramOptions {
  /** Bins (default 10). */
  readonly bins?: Bins
  /** The lower and upper range of the bins (default the data's minimum and maximum); values outside are ignored. */
  readonly range?: readonly [number, number]
  /** One weight per observation (not with automatic rules). */
  readonly weights?: TensorLike
}

function ptp(a: Float64Array): number {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of a) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  return hi - lo
}

function outerEdges(a: Float64Array, range?: readonly [number, number]): [number, number] {
  let lo: number
  let hi: number
  if (range !== undefined) {
    ;[lo, hi] = range
    if (lo > hi) throw new OmaiError('value', 'histogram: range must be increasing')
    if (!(Number.isFinite(lo) && Number.isFinite(hi))) throw new OmaiError('value', 'histogram: range must be finite')
  } else if (a.length === 0) {
    ;[lo, hi] = [0, 1]
  } else {
    lo = Number.POSITIVE_INFINITY
    hi = Number.NEGATIVE_INFINITY
    for (const v of a) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
    if (!(Number.isFinite(lo) && Number.isFinite(hi))) throw new OmaiError('value', 'histogram: data must be finite')
  }
  if (lo === hi) return [lo - 0.5, hi + 0.5]
  return [lo, hi]
}

// numpy.linspace: start + i step, the last edge exactly stop.
function linspace(lo: number, hi: number, bins: number): number[] {
  const step = (hi - lo) / bins
  return Array.from({ length: bins + 1 }, (_, i) => (i === bins ? hi : i * step + lo))
}

// Knuth's log posterior of M equal bins (Knuth 2006, eq. 32, up to a constant).
function knuthBins(a: Float64Array, lo: number, hi: number): number {
  const n = a.length
  const maxBins = Math.max(1, Math.min(n, 1000))
  let best = 1
  let bestValue = Number.NEGATIVE_INFINITY
  const lgHalf = item(lgamma(0.5))
  const table = toFlat(lgamma(vec(Array.from({ length: n + 1 }, (_, c) => c + 0.5))))
  for (let m = 1; m <= maxBins; m++) {
    const counts = countUniform(a, lo, hi, m)
    let s = 0
    for (const c of counts) s += table[c] ?? 0
    const value = n * Math.log(m) + item(lgamma(m / 2)) - m * lgHalf - item(lgamma(n + m / 2)) + s
    if (value > bestValue) {
      bestValue = value
      best = m
    }
  }
  return best
}

function binWidth(rule: BinRule, a: Float64Array): number {
  const n = a.length
  const sturges = ptp(a) / (Math.log2(n) + 1)
  switch (rule) {
    case 'sqrt':
      return ptp(a) / Math.sqrt(n)
    case 'sturges':
      return sturges
    case 'rice':
      return ptp(a) / (2 * n ** (1 / 3))
    case 'scott':
      return ((24 * Math.sqrt(Math.PI)) / n) ** (1 / 3) * Math.sqrt(varianceOf(a))
    case 'doane': {
      if (n <= 2) return 0
      const sg1 = Math.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3)))
      const sigma = Math.sqrt(varianceOf(a))
      if (!(sigma > 0)) return 0
      const m = meanOf(a)
      let g1 = 0
      for (const v of a) g1 += ((v - m) / sigma) ** 3
      g1 /= n
      return ptp(a) / (1 + Math.log2(n) + Math.log2(1 + Math.abs(g1) / sg1))
    }
    case 'fd':
    case 'auto': {
      const s = sortedCopy(a)
      const fd = 2 * (quantile(s, 0.75) - quantile(s, 0.25)) * n ** (-1 / 3)
      if (rule === 'fd') return fd
      return fd ? Math.min(fd, sturges) : sturges
    }
    default:
      throw new OmaiError('value', `histogramBinEdges: unknown rule ${String(rule)}`)
  }
}

function edgesOf(a: Float64Array, options: HistogramOptions): { edges: number[]; uniform: boolean } {
  const bins = options.bins ?? 10
  if (typeof bins === 'string') {
    if (options.weights !== undefined)
      throw new OmaiError('value', 'histogram: automatic bin rules do not support weights')
    const [lo, hi] = outerEdges(a, options.range)
    const kept = options.range === undefined ? a : a.filter((v) => v >= lo && v <= hi)
    let count = 1
    if (kept.length > 0) {
      if (bins === 'knuth') count = knuthBins(kept, lo, hi)
      else {
        const width = binWidth(bins, kept)
        count = width ? Math.ceil((hi - lo) / width) : 1
      }
    }
    return { edges: linspace(lo, hi, count), uniform: true }
  }
  if (typeof bins === 'number') {
    if (!(Number.isInteger(bins) && bins >= 1))
      throw new OmaiError('value', 'histogram: bins must be a positive integer')
    const [lo, hi] = outerEdges(a, options.range)
    return { edges: linspace(lo, hi, bins), uniform: true }
  }
  const edges = toFlat(asTensor(bins))
  for (let i = 1; i < edges.length; i++)
    if (!((edges[i] ?? 0) >= (edges[i - 1] ?? 0)))
      throw new OmaiError('value', 'histogram: bin edges must increase monotonically')
  if (edges.length < 2) throw new OmaiError('value', 'histogram: need at least two edges')
  return { edges, uniform: false }
}

/**
 * The bin edges a histogram would use (numpy.histogram_bin_edges), including the automatic rules and Knuth's rule.
 *
 * @param x - The data (flattened).
 * @param options - bins (count, rule or edges, default 10) and range.
 * @returns The edges, one more than the number of bins.
 * @example
 * ```ts
 * import { histogramBinEdges } from 'omai-engine/stats'
 * histogramBinEdges([0, 1, 2, 3, 4], { bins: 4 }) // => [0, 1, 2, 3, 4]
 * histogramBinEdges([1, 2, 3, 4, 5, 6, 7, 8], { bins: 'sturges' }) // => [1, 2.75, 4.5, 6.25, 8]
 * ```
 */
export function histogramBinEdges(x: TensorLike, options: HistogramOptions = {}): Tensor {
  return vec(edgesOf(sample('histogramBinEdges', x, 0), options).edges)
}

function countUniform(a: Float64Array, lo: number, hi: number, bins: number): Float64Array {
  return countInto(a, linspace(lo, hi, bins), true)
}

function countInto(a: Float64Array, edges: readonly number[], uniform: boolean, w?: Float64Array): Float64Array {
  const bins = edges.length - 1
  const counts = new Float64Array(bins)
  const lo = edges[0] ?? 0
  const hi = edges[bins] ?? 0
  for (let i = 0; i < a.length; i++) {
    const v = a[i] ?? 0
    if (!(v >= lo && v <= hi)) continue
    let k: number
    if (uniform) {
      k = Math.trunc(((v - lo) / (hi - lo)) * bins)
      if (k === bins) k--
      if (v < (edges[k] ?? 0)) k--
      if (v >= (edges[k + 1] ?? 0) && k !== bins - 1) k++
    } else {
      // The last bin is closed on the right; the others are half-open.
      let l = 0
      let r = bins
      while (l < r) {
        const mid = (l + r) >> 1
        if ((edges[mid + 1] ?? 0) <= v) l = mid + 1
        else r = mid
      }
      k = Math.min(l, bins - 1)
    }
    counts[k] = (counts[k] ?? 0) + (w === undefined ? 1 : (w[i] ?? 0))
  }
  return counts
}

/** The result of {@link histogram}. */
export interface Histogram {
  /** Counts (or summed weights, or densities) per bin. */
  readonly counts: Tensor
  readonly edges: Tensor
}

/**
 * A histogram (numpy.histogram): counts in half-open bins [e_i, e_i+1), the last closed; with `density` the counts are
 * divided by n times the bin width so the histogram integrates to one.
 *
 * @param x - The data (flattened).
 * @param options - bins, range, weights and density.
 * @returns The counts and the edges.
 * @example
 * ```ts
 * import { histogram } from 'omai-engine/stats'
 * histogram([0, 1, 1, 2, 3], { bins: 3 }).counts // => [1, 2, 2]
 * histogram([0, 1, 1, 2, 3], { bins: 3, density: true }).counts // => [0.2, 0.4, 0.4]
 * ```
 */
export function histogram(x: TensorLike, options: HistogramOptions & { readonly density?: boolean } = {}): Histogram {
  const a = sample('histogram', x, 0, true)
  const w = options.weights === undefined ? undefined : sample('histogram', options.weights)
  if (w !== undefined && w.length !== a.length) throw new OmaiError('shape', 'histogram: weights must match the data')
  const { edges, uniform } = edgesOf(a, options)
  const counts = countInto(a, edges, uniform, w)
  if (options.density === true) {
    let s = 0
    for (const c of counts) s += c
    for (let i = 0; i < counts.length; i++) counts[i] = (counts[i] ?? 0) / ((edges[i + 1] ?? 0) - (edges[i] ?? 0)) / s
  }
  return { counts: vec(counts), edges: vec(edges) }
}

/** The result of {@link histogram2d}. */
export interface Histogram2d {
  /** Counts `[nx, ny]`: rows index x bins, columns y bins. */
  readonly counts: Tensor
  readonly xEdges: Tensor
  readonly yEdges: Tensor
}

// numpy.histogramdd's bin index of each value along one axis: searchsorted right, values on the last edge moved
// into the last bin, -1 for values outside.
function binIndices(v: Float64Array, edges: readonly number[]): Int32Array {
  const bins = edges.length - 1
  const out = new Int32Array(v.length)
  const last = edges[bins] ?? 0
  for (let i = 0; i < v.length; i++) {
    const x = v[i] ?? 0
    let l = 0
    let r = edges.length
    while (l < r) {
      const mid = (l + r) >> 1
      if ((edges[mid] ?? 0) <= x) l = mid + 1
      else r = mid
    }
    if (x === last) l -= 1
    out[i] = l >= 1 && l <= bins ? l - 1 : -1
  }
  return out
}

function axisEdges(a: Float64Array, bins: number | TensorLike, range?: readonly [number, number]): number[] {
  if (typeof bins === 'number') {
    const [lo, hi] = outerEdges(a, range)
    return linspace(lo, hi, bins)
  }
  return toFlat(asTensor(bins))
}

/**
 * A two-dimensional histogram (numpy.histogram2d).
 *
 * @param x - First coordinates `[N]`.
 * @param y - Second coordinates `[N]`.
 * @param options - bins (a count, a pair of counts, or a pair of edge arrays; default 10), range per axis, weights and
 *   density.
 * @returns The counts and both edge arrays.
 * @throws OmaiError with code `value` when a coordinate is NaN.
 * @example
 * ```ts
 * import { histogram2d } from 'omai-engine/stats'
 * histogram2d([0, 0, 1], [0, 1, 1], { bins: 2 }).counts // => [[1, 1], [0, 1]]
 * ```
 */
export function histogram2d(
  x: TensorLike,
  y: TensorLike,
  options: {
    readonly bins?: number | readonly [number | TensorLike, number | TensorLike]
    readonly range?: readonly [readonly [number, number], readonly [number, number]]
    readonly weights?: TensorLike
    readonly density?: boolean
  } = {},
): Histogram2d {
  const a = sample('histogram2d', x, 0, true)
  const b = sample('histogram2d', y, 0, true)
  if (a.length !== b.length) throw new OmaiError('shape', 'histogram2d: x and y differ in length')
  const w = options.weights === undefined ? undefined : sample('histogram2d', options.weights)
  const bins = options.bins ?? 10
  const [bx, by] = typeof bins === 'number' ? [bins, bins] : bins
  const ex = axisEdges(a, bx, options.range?.[0])
  const ey = axisEdges(b, by, options.range?.[1])
  const ix = binIndices(a, ex)
  const iy = binIndices(b, ey)
  const nx = ex.length - 1
  const ny = ey.length - 1
  const counts = new Float64Array(nx * ny)
  for (let i = 0; i < a.length; i++) {
    const r = ix[i] ?? -1
    const c = iy[i] ?? -1
    if (r < 0 || c < 0) continue
    counts[r * ny + c] = (counts[r * ny + c] ?? 0) + (w === undefined ? 1 : (w[i] ?? 0))
  }
  if (options.density === true) {
    let s = 0
    for (const v of counts) s += v
    for (let r = 0; r < nx; r++)
      for (let c = 0; c < ny; c++) {
        const area = ((ex[r + 1] ?? 0) - (ex[r] ?? 0)) * ((ey[c + 1] ?? 0) - (ey[c] ?? 0))
        counts[r * ny + c] = (counts[r * ny + c] ?? 0) / area / s
      }
  }
  return { counts: mat(counts, nx, ny), xEdges: vec(ex), yEdges: vec(ey) }
}

/** A statistic of the values in one bin. */
export type BinStatistic =
  | 'mean'
  | 'median'
  | 'count'
  | 'sum'
  | 'std'
  | 'min'
  | 'max'
  | ((values: Float64Array) => number)

/** The result of {@link binnedStatistic}. */
export interface BinnedStatistic {
  /** The statistic per bin (NaN for empty bins, except 0 for count and sum). */
  readonly statistic: Tensor
  readonly edges: Tensor
  /** The bin of each observation, 1-based; 0 below the range and bins + 1 above, as scipy. */
  readonly binNumber: Tensor
}

/**
 * A statistic of values within bins of x (scipy.stats.binned_statistic): the mean, median, count, sum, standard
 * deviation (ddof 0), minimum or maximum, or any function of a bin's values.
 *
 * @param x - The binned coordinate `[N]`.
 * @param values - The values `[N]`.
 * @param options - statistic (default `'mean'`), bins (count or edges, default 10) and range.
 * @returns The statistic per bin, the edges and each observation's bin number.
 * @example
 * ```ts
 * import { binnedStatistic } from 'omai-engine/stats'
 * binnedStatistic([0, 1, 2, 3], [1, 2, 3, 4], { bins: 2 }).statistic // => [1.5, 3.5]
 * binnedStatistic([0, 1, 2, 3], [1, 2, 3, 4], { bins: 2, statistic: 'sum' }).statistic // => [3, 7]
 * ```
 */
export function binnedStatistic(
  x: TensorLike,
  values: TensorLike,
  options: {
    readonly statistic?: BinStatistic
    readonly bins?: number | TensorLike
    readonly range?: readonly [number, number]
  } = {},
): BinnedStatistic {
  const a = sample('binnedStatistic', x)
  const v = sample('binnedStatistic', values)
  if (a.length !== v.length) throw new OmaiError('shape', 'binnedStatistic: x and values differ in length')
  const edges = axisEdges(a, options.bins ?? 10, options.range)
  const bins = edges.length - 1
  const groups: number[][] = Array.from({ length: bins }, () => [])
  const numbers = new Int32Array(a.length)
  const last = edges[bins] ?? 0
  for (let i = 0; i < a.length; i++) {
    const xi = a[i] ?? 0
    let l = 0
    let r = edges.length
    while (l < r) {
      const mid = (l + r) >> 1
      if ((edges[mid] ?? 0) <= xi) l = mid + 1
      else r = mid
    }
    if (xi === last) l -= 1
    numbers[i] = l
    if (l >= 1 && l <= bins) groups[l - 1]?.push(v[i] ?? 0)
  }
  const stat = options.statistic ?? 'mean'
  const out = groups.map((g) => {
    const arr = Float64Array.from(g)
    if (typeof stat === 'function') return stat(arr)
    if (stat === 'count') return g.length
    if (stat === 'sum') return g.reduce((s, t) => s + t, 0)
    if (g.length === 0) return Number.NaN
    switch (stat) {
      case 'mean':
        return meanOf(arr)
      case 'median':
        return median(arr)
      case 'std':
        return Math.sqrt(varianceOf(arr))
      case 'min':
        return g.reduce((m, t) => Math.min(m, t), Number.POSITIVE_INFINITY)
      default:
        return g.reduce((m, t) => Math.max(m, t), Number.NEGATIVE_INFINITY)
    }
  })
  return { statistic: vec(out), edges: vec(edges), binNumber: fromStorage(numbers, [a.length], 'int32') }
}
