/**
 * Distances between points, scipy.spatial.distance style: `distanceBetween` evaluates any of scipy's metrics (plus
 * haversine) over the coordinate axis of broadcast operands, `cdist`, `pdist` and `squareform` are the pairwise forms
 * built on it, and scalar metric functions on flat number arrays serve the trees and indexes of `numerics/spatial`,
 * with the list of true metrics they may use.
 */

import { MdmlError } from '../../base/index.ts'
import {
  abs,
  add,
  arcsin,
  cast,
  concat,
  cos,
  div,
  fromStorage,
  logicalOr,
  logicalXor,
  max,
  mean,
  minimum,
  mul,
  notEqual,
  pow,
  primalOf,
  promoteFloat,
  reshape,
  sin,
  sliceAxis,
  sqrt,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  take,
  tensordot,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { inv } from '../linalg/index.ts'

/** The metrics of scipy.spatial.distance (and haversine, as sklearn). */
export type Metric =
  | 'euclidean'
  | 'sqeuclidean'
  | 'cityblock'
  | 'chebyshev'
  | 'minkowski'
  | 'seuclidean'
  | 'cosine'
  | 'correlation'
  | 'hamming'
  | 'jaccard'
  | 'mahalanobis'
  | 'canberra'
  | 'braycurtis'
  | 'haversine'

/** Metric parameters. */
export interface MetricOptions {
  /** Minkowski order (default 2). */
  readonly p?: number
  /** Per-feature weights (minkowski, euclidean, sqeuclidean, cityblock, chebyshev). */
  readonly w?: TensorLike
  /** Per-feature variances for seuclidean (default the sample variances of the inputs, ddof 1, as scipy). */
  readonly v?: TensorLike
  /** The inverse covariance for mahalanobis (default the inverse sample covariance, as scipy). */
  readonly vi?: TensorLike
}

function zeroMask(x: Value, isZero: (v: number) => boolean): Tensor {
  return fromStorage(
    Uint8Array.from(toFlat(primalOf(x)), (v) => (isZero(v) ? 1 : 0)),
    x.shape,
    'bool',
  )
}

function safeSqrt(x: Value): Value {
  // sqrt with a zero gradient at zero distance (where the true derivative does not exist).
  const zero = zeroMask(x, (v) => v <= 0)
  return where(zero, 0, sqrt(where(zero, 1, x)))
}

function safeDivide(n: Value, d: Value): Value {
  const zero = zeroMask(d, (v) => v === 0)
  return where(zero, 0, div(n, where(zero, 1, d)))
}

function sampleVariance(x: Value): Value {
  const n = x.shape[0] ?? 1
  const centred = sub(x, mean(x, 0, true))
  return div(sum(mul(centred, centred), 0), n - 1)
}

function inverseCovariance(x: Value): Tensor {
  const xs = toFlat(primalOf(x))
  const [n = 0, d = 0] = x.shape
  const mu = Array.from(
    { length: d },
    (_, j) => Array.from({ length: n }, (_, i) => xs[i * d + j] ?? 0).reduce((s, v) => s + v, 0) / n,
  )
  const cov = new Float64Array(d * d)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++)
      for (let k = 0; k < d; k++)
        cov[j * d + k] =
          (cov[j * d + k] ?? 0) +
          (((xs[i * d + j] ?? 0) - (mu[j] ?? 0)) * ((xs[i * d + k] ?? 0) - (mu[k] ?? 0))) / (n - 1)
  return inv(fromStorage(cov, [d, d], 'float64')) as Tensor
}

/** Coordinate k of every point: the last axis sliced and dropped. */
function coordinate(x: Value, k: number): Value {
  return reshape(sliceAxis(x, x.shape.length - 1, k, k + 1), x.shape.slice(0, -1))
}

/** Whether a metric still needs scipy's data-dependent defaults. */
function needsStatistics(metric: Metric, options: MetricOptions): boolean {
  return (metric === 'seuclidean' && options.v === undefined) || (metric === 'mahalanobis' && options.vi === undefined)
}

/**
 * scipy's data-dependent defaults, filled from the points when not given: seuclidean's `v` is their sample variance
 * per coordinate (ddof 1) and mahalanobis's `vi` the inverse of their sample covariance (`cdist` stacks both inputs,
 * `pdist` uses its one input). Other metrics, and given values, pass through unchanged.
 *
 * @param metric - The metric.
 * @param points - Points `[n, d]` the statistics are taken over.
 * @param options - The options as given.
 * @returns The options with `v` or `vi` filled in where the metric needs them.
 * @example
 * ```ts
 * import { resolveMetricOptions } from 'mdml-engine/numerics/geometry'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(resolveMetricOptions('seuclidean', [[0, 0], [2, 4]]).v) // => [2, 8]
 * resolveMetricOptions('euclidean', [[0, 0], [2, 4]], { p: 2 }) // => { p: 2 }
 * ```
 */
export function resolveMetricOptions(metric: Metric, points: TensorLike, options: MetricOptions = {}): MetricOptions {
  if (!needsStatistics(metric, options)) return options
  const [x] = promoteFloat(points) as [Value]
  if (x.shape.length !== 2) throw new MdmlError('shape', 'resolveMetricOptions: points must be [n, d]')
  return metric === 'seuclidean' ? { ...options, v: sampleVariance(x) } : { ...options, vi: inverseCovariance(x) }
}

/**
 * A metric between corresponding points of `a` and `b`, which are `[..., d]` broadcast together with the last axis as
 * the coordinates: rows of equal shape give one distance per row `[n]`, two points `[d]` give a scalar, and `[m, 1, d]`
 * against `[1, n, d]` gives the matrix of {@link cdist}. Continuous metrics are built from operations, so they are
 * differentiable in the points (with zero gradient where a distance is exactly zero); hamming and jaccard compare
 * values and have no derivative. Haversine takes `[latitude, longitude]` in radians on the unit sphere. seuclidean
 * and mahalanobis need `v` and `vi` here ({@link resolveMetricOptions} fills scipy's defaults from the points).
 *
 * @param a - Points `[..., d]`.
 * @param b - Points `[..., d]`, broadcastable against `a`.
 * @param metric - The metric (default euclidean).
 * @param options - Minkowski order, weights, variances or inverse covariance.
 * @returns The distances, the broadcast shape without the last axis.
 * @throws MdmlError `shape` when the coordinate axes differ, `value` when seuclidean or mahalanobis lack their
 *   statistics.
 * @concept distances
 * @example
 * ```ts
 * import { distanceBetween } from 'mdml-engine/numerics/geometry'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(distanceBetween([[0, 0], [1, 1]], [[3, 4], [1, 1]])) // => [5, 0]
 * toArray(distanceBetween([0, 0], [3, 4], 'cityblock')) // => 7
 * ```
 */
export function distanceBetween(
  a: TensorLike,
  b: TensorLike,
  metric: Metric = 'euclidean',
  options: MetricOptions = {},
): Value {
  const [x, y] = promoteFloat(a, b) as [Value, Value]
  const d = x.shape[x.shape.length - 1]
  if (x.shape.length === 0 || y.shape.length === 0 || d !== y.shape[y.shape.length - 1])
    throw new MdmlError(
      'shape',
      `distanceBetween: the coordinate axes differ, [${x.shape.join(', ')}] against [${y.shape.join(', ')}]`,
    )
  const weights = options.w === undefined ? undefined : (promoteFloat(options.w)[0] as Value)
  const weigh = (v: Value): Value => (weights === undefined ? v : mul(v, weights))
  switch (metric) {
    case 'euclidean':
      return safeSqrt(sum(weigh(square(sub(x, y))), -1))
    case 'sqeuclidean':
      return sum(weigh(square(sub(x, y))), -1)
    case 'cityblock':
      return sum(weigh(abs(sub(x, y))), -1)
    case 'chebyshev':
      return max(weigh(abs(sub(x, y))), -1)
    case 'minkowski': {
      const p = options.p ?? 2
      const s = sum(weigh(pow(abs(sub(x, y)), p)), -1)
      const zero = zeroMask(s, (v) => v === 0)
      return where(zero, 0, pow(where(zero, 1, s), 1 / p))
    }
    case 'seuclidean': {
      if (options.v === undefined)
        throw new MdmlError('value', 'distanceBetween: seuclidean needs the variances v (resolveMetricOptions)')
      const v = promoteFloat(options.v)[0] as Value
      return safeSqrt(sum(div(square(sub(x, y)), v), -1))
    }
    case 'cosine': {
      const dotXy = sum(mul(x, y), -1)
      const nx = safeSqrt(sum(square(x), -1))
      const ny = safeSqrt(sum(square(y), -1))
      return sub(1, safeDivide(dotXy, mul(nx, ny)))
    }
    case 'correlation':
      return distanceBetween(sub(x, mean(x, -1, true)), sub(y, mean(y, -1, true)), 'cosine')
    case 'hamming':
      // Fraction of differing coordinates (on the values, not differentiable).
      return mean(cast(notEqual(x, y), 'float64'), -1)
    case 'jaccard': {
      // Among coordinates where either is non-zero, the fraction that differ (scipy's boolean Jaccard).
      const nx = notEqual(x, 0)
      const ny = notEqual(y, 0)
      const union = cast(sum(logicalOr(nx, ny), -1), 'float64')
      const differ = cast(sum(logicalXor(nx, ny), -1), 'float64')
      return safeDivide(differ, union)
    }
    case 'mahalanobis': {
      if (options.vi === undefined)
        throw new MdmlError(
          'value',
          'distanceBetween: mahalanobis needs the inverse covariance vi (resolveMetricOptions)',
        )
      const vi = promoteFloat(options.vi)[0] as Value
      const diff = sub(x, y)
      return safeSqrt(sum(mul(tensordot(diff, vi, [[-1], [0]]), diff), -1))
    }
    case 'canberra':
      return sum(safeDivide(abs(sub(x, y)), add(abs(x), abs(y))), -1)
    case 'braycurtis':
      return safeDivide(sum(abs(sub(x, y)), -1), sum(abs(add(x, y)), -1))
    case 'haversine': {
      if (d !== 2) throw new MdmlError('shape', 'distanceBetween: haversine needs [latitude, longitude] coordinates')
      const lat1 = coordinate(x, 0)
      const lat2 = coordinate(y, 0)
      const dlat = sub(lat2, lat1)
      const dlon = sub(coordinate(y, 1), coordinate(x, 1))
      const h = add(square(sin(mul(dlat, 0.5))), mul(mul(cos(lat1), cos(lat2)), square(sin(mul(dlon, 0.5)))))
      return mul(arcsin(safeSqrt(minimum(h, 1))), 2)
    }
  }
}

/**
 * Distances between each row of XA and each row of XB (scipy.spatial.distance.cdist), for the metrics of scipy plus
 * haversine (rows of [latitude, longitude] in radians, unit sphere): {@link distanceBetween} on `[m, 1, d]` against
 * `[1, n, d]`, with seuclidean's variances and mahalanobis's inverse covariance defaulting to the statistics of both
 * inputs stacked, as scipy. Continuous metrics are differentiable in the points.
 *
 * @param xa - Points `[m, d]`.
 * @param xb - Points `[n, d]`.
 * @param metric - The metric (default euclidean).
 * @param options - Minkowski order, weights, variances or inverse covariance.
 * @returns Distances `[m, n]`.
 * @concept distances
 * @index function Pairwise distances between two sets of points, scipy metrics.
 * @example
 * ```ts
 * import { cdist } from 'mdml-engine/numerics/geometry'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(cdist([[0, 0]], [[3, 4], [1, 0]])) // => [[5, 1]]
 * ```
 */
export function cdist(
  xa: TensorLike,
  xb: TensorLike,
  metric: Metric = 'euclidean',
  options: MetricOptions = {},
): Value {
  const [a, b] = promoteFloat(xa, xb) as [Value, Value]
  if (a.shape.length !== 2 || b.shape.length !== 2 || a.shape[1] !== b.shape[1])
    throw new MdmlError('shape', 'cdist: inputs must be [m, d] and [n, d]')
  const d = a.shape[1] ?? 0
  const resolved = needsStatistics(metric, options) ? resolveMetricOptions(metric, concat([a, b], 0), options) : options
  return distanceBetween(reshape(a, [-1, 1, d]), reshape(b, [1, -1, d]), metric, resolved)
}

/**
 * Distances between all pairs of rows, condensed (scipy.spatial.distance.pdist): the upper triangle row by row,
 * n (n - 1) / 2 entries.
 *
 * @param x - Points `[n, d]`.
 * @param metric - The metric (default euclidean).
 * @param options - Metric parameters (seuclidean and mahalanobis default to the statistics of x, as scipy).
 * @returns The condensed distances.
 * @example
 * ```ts
 * import { pdist } from 'mdml-engine/numerics/geometry'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(pdist([[0, 0], [3, 4], [6, 8]])) // => [5, 10, 5]
 * ```
 */
export function pdist(x: TensorLike, metric: Metric = 'euclidean', options: MetricOptions = {}): Value {
  const [v] = promoteFloat(x) as [Value]
  const n = v.shape[0] ?? 0
  const full = reshape(cdist(v, v, metric, resolveMetricOptions(metric, v, options)), [-1])
  const idx: number[] = []
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) idx.push(i * n + j)
  return take(full, fromStorage(Int32Array.from(idx), [idx.length], 'int32'), 0)
}

/**
 * Convert between condensed and square distance matrices (scipy.spatial.distance.squareform).
 *
 * @param d - A condensed vector or a symmetric square matrix with zero diagonal.
 * @returns The other form.
 * @example
 * ```ts
 * import { squareform } from 'mdml-engine/numerics/geometry'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(squareform([5, 10, 5])) // => [[0, 5, 10], [5, 0, 5], [10, 5, 0]]
 * ```
 */
export function squareform(d: TensorLike): Value {
  const [v] = promoteFloat(d) as [Value]
  if (v.shape.length === 1) {
    const len = v.shape[0] ?? 0
    const n = (1 + Math.sqrt(1 + 8 * len)) / 2
    if (!Number.isInteger(n)) throw new MdmlError('shape', 'squareform: the length is not n (n - 1) / 2')
    // Gather from [0, d_0, d_1, ...] with index 0 on the diagonal.
    const idx = new Int32Array(n * n)
    let k = 1
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        idx[i * n + j] = k
        idx[j * n + i] = k
        k++
      }
    }
    const padded = concat([mul(sliceAxis(v, 0, 0, Math.min(1, len)), 0), v], 0)
    const source = len === 0 ? fromStorage(new Float64Array(1), [1], 'float64') : padded
    return reshape(take(source, fromStorage(idx, [n * n], 'int32'), 0), [n, n])
  }
  const n = v.shape[0] ?? 0
  if (v.shape.length !== 2 || v.shape[1] !== n)
    throw new MdmlError('shape', 'squareform: expected a vector or a square matrix')
  const idx: number[] = []
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) idx.push(i * n + j)
  return take(reshape(v, [-1]), fromStorage(Int32Array.from(idx), [idx.length], 'int32'), 0)
}

/** A distance between two points stored at offsets of flat arrays (for trees and indexes iterating on numbers). */
export type PointDistance = (a: ArrayLike<number>, ia: number, b: ArrayLike<number>, ib: number) => number

/** Metrics satisfying the triangle inequality (usable by ball and vantage-point trees). */
export const TRUE_METRICS: ReadonlySet<Metric> = new Set<Metric>([
  'euclidean',
  'cityblock',
  'chebyshev',
  'minkowski',
  'seuclidean',
  'mahalanobis',
  'canberra',
  'hamming',
  'jaccard',
  'haversine',
])

/**
 * The scalar form of a metric on number arrays: d(a[ia .. ia + dim], b[ib .. ib + dim]), the same values as
 * {@link cdist} (seuclidean and mahalanobis need explicit v and vi here).
 *
 * @param metric - The metric.
 * @param dim - The dimension.
 * @param options - Minkowski order, weights, variances or inverse covariance.
 * @returns The distance function.
 * @example
 * ```ts
 * import { metricFunction } from 'mdml-engine/numerics/geometry'
 * metricFunction('cityblock', 2)([0, 0, 1, 2], 0, [0, 0, 1, 2], 2) // => 3
 * ```
 */
export function metricFunction(metric: Metric, dim: number, options: MetricOptions = {}): PointDistance {
  const w = options.w === undefined ? undefined : toFlat(primalOf(promoteFloat(options.w)[0] as Value))
  const weight = (k: number): number => (w === undefined ? 1 : (w[k] ?? 1))
  switch (metric) {
    case 'euclidean':
    case 'sqeuclidean': {
      const squared = metric === 'sqeuclidean'
      return (a, ia, b, ib) => {
        let s = 0
        for (let k = 0; k < dim; k++) s += weight(k) * ((a[ia + k] ?? 0) - (b[ib + k] ?? 0)) ** 2
        return squared ? s : Math.sqrt(s)
      }
    }
    case 'cityblock':
      return (a, ia, b, ib) => {
        let s = 0
        for (let k = 0; k < dim; k++) s += weight(k) * Math.abs((a[ia + k] ?? 0) - (b[ib + k] ?? 0))
        return s
      }
    case 'chebyshev':
      return (a, ia, b, ib) => {
        let s = 0
        for (let k = 0; k < dim; k++) s = Math.max(s, weight(k) * Math.abs((a[ia + k] ?? 0) - (b[ib + k] ?? 0)))
        return s
      }
    case 'minkowski': {
      const p = options.p ?? 2
      return (a, ia, b, ib) => {
        let s = 0
        for (let k = 0; k < dim; k++) s += weight(k) * Math.abs((a[ia + k] ?? 0) - (b[ib + k] ?? 0)) ** p
        return s ** (1 / p)
      }
    }
    case 'seuclidean': {
      if (options.v === undefined) throw new MdmlError('value', 'metricFunction: seuclidean needs the variances v')
      const v = toFlat(primalOf(promoteFloat(options.v)[0] as Value))
      return (a, ia, b, ib) => {
        let s = 0
        for (let k = 0; k < dim; k++) s += ((a[ia + k] ?? 0) - (b[ib + k] ?? 0)) ** 2 / (v[k] ?? 1)
        return Math.sqrt(s)
      }
    }
    case 'mahalanobis': {
      if (options.vi === undefined)
        throw new MdmlError('value', 'metricFunction: mahalanobis needs the inverse covariance vi')
      const vi = toFlat(primalOf(promoteFloat(options.vi)[0] as Value))
      return (a, ia, b, ib) => {
        let s = 0
        for (let j = 0; j < dim; j++) {
          const dj = (a[ia + j] ?? 0) - (b[ib + j] ?? 0)
          for (let k = 0; k < dim; k++) s += dj * (vi[j * dim + k] ?? 0) * ((a[ia + k] ?? 0) - (b[ib + k] ?? 0))
        }
        return Math.sqrt(Math.max(0, s))
      }
    }
    case 'cosine':
    case 'correlation':
      return (a, ia, b, ib) => {
        let ma = 0
        let mb = 0
        if (metric === 'correlation') {
          for (let k = 0; k < dim; k++) {
            ma += (a[ia + k] ?? 0) / dim
            mb += (b[ib + k] ?? 0) / dim
          }
        }
        let dotAb = 0
        let na = 0
        let nb = 0
        for (let k = 0; k < dim; k++) {
          const x = (a[ia + k] ?? 0) - ma
          const y = (b[ib + k] ?? 0) - mb
          dotAb += x * y
          na += x * x
          nb += y * y
        }
        return na === 0 || nb === 0 ? 1 : 1 - dotAb / Math.sqrt(na * nb)
      }
    case 'hamming':
      return (a, ia, b, ib) => {
        let s = 0
        for (let k = 0; k < dim; k++) if ((a[ia + k] ?? 0) !== (b[ib + k] ?? 0)) s++
        return s / dim
      }
    case 'jaccard':
      return (a, ia, b, ib) => {
        let union = 0
        let differ = 0
        for (let k = 0; k < dim; k++) {
          const x = (a[ia + k] ?? 0) !== 0
          const y = (b[ib + k] ?? 0) !== 0
          if (x || y) union++
          if (x !== y) differ++
        }
        return union === 0 ? 0 : differ / union
      }
    case 'canberra':
      return (a, ia, b, ib) => {
        let s = 0
        for (let k = 0; k < dim; k++) {
          const x = a[ia + k] ?? 0
          const y = b[ib + k] ?? 0
          const den = Math.abs(x) + Math.abs(y)
          if (den > 0) s += Math.abs(x - y) / den
        }
        return s
      }
    case 'braycurtis':
      return (a, ia, b, ib) => {
        let num = 0
        let den = 0
        for (let k = 0; k < dim; k++) {
          num += Math.abs((a[ia + k] ?? 0) - (b[ib + k] ?? 0))
          den += Math.abs((a[ia + k] ?? 0) + (b[ib + k] ?? 0))
        }
        return den === 0 ? 0 : num / den
      }
    case 'haversine':
      return (a, ia, b, ib) => {
        const lat1 = a[ia] ?? 0
        const lat2 = b[ib] ?? 0
        const h =
          Math.sin((lat2 - lat1) / 2) ** 2 +
          Math.cos(lat1) * Math.cos(lat2) * Math.sin(((b[ib + 1] ?? 0) - (a[ia + 1] ?? 0)) / 2) ** 2
        return 2 * Math.asin(Math.sqrt(Math.min(1, h)))
      }
  }
}
