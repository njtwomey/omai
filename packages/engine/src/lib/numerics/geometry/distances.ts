import { WhiteboxError } from '../../base/index.ts'
import {
  abs,
  add,
  arcsin,
  concat,
  cos,
  div,
  fromStorage,
  matmul,
  max,
  mean,
  mul,
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
  swapaxes,
  type Tensor,
  type TensorLike,
  take,
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

/** Pairwise differences `[m, n, d]` of rows. */
function differences(a: Value, b: Value): Value {
  const d = a.shape[1] ?? 0
  return sub(reshape(a, [-1, 1, d]), reshape(b, [1, -1, d]))
}

function safeSqrt(x: Value): Value {
  // sqrt with a zero gradient at zero distance (where the true derivative does not exist).
  const zero = fromStorage(
    Uint8Array.from(toFlat(primalOf(x)), (v) => (v <= 0 ? 1 : 0)),
    x.shape,
    'bool',
  )
  return where(zero, 0, sqrt(where(zero, 1, x)))
}

function safeDivide(n: Value, d: Value): Value {
  const zero = fromStorage(
    Uint8Array.from(toFlat(primalOf(d)), (v) => (v === 0 ? 1 : 0)),
    d.shape,
    'bool',
  )
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

/**
 * Distances between each row of XA and each row of XB (scipy.spatial.distance.cdist), for the metrics of scipy plus
 * haversine (rows of [latitude, longitude] in radians, unit sphere). Continuous metrics are built from operations, so
 * differentiable in the points (with zero gradient where a distance is exactly zero).
 *
 * @param xa - Points `[m, d]`.
 * @param xb - Points `[n, d]`.
 * @param metric - The metric (default euclidean).
 * @param options - Minkowski order, weights, variances or inverse covariance.
 * @returns Distances `[m, n]`.
 * @concept distances
 * @example
 * ```ts
 * import { cdist } from 'whitebox-engine/numerics/geometry'
 * import { toArray } from 'whitebox-engine/tensor'
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
    throw new WhiteboxError('shape', 'cdist: inputs must be [m, d] and [n, d]')
  const d = a.shape[1] ?? 0
  const weights = options.w === undefined ? undefined : reshape(promoteFloat(options.w)[0] as Value, [1, 1, d])
  const weigh = (x: Value): Value => (weights === undefined ? x : mul(x, weights))
  switch (metric) {
    case 'euclidean':
      return safeSqrt(sum(weigh(square(differences(a, b))), 2))
    case 'sqeuclidean':
      return sum(weigh(square(differences(a, b))), 2)
    case 'cityblock':
      return sum(weigh(abs(differences(a, b))), 2)
    case 'chebyshev':
      return max(weigh(abs(differences(a, b))), 2)
    case 'minkowski': {
      const p = options.p ?? 2
      const s = sum(weigh(pow(abs(differences(a, b)), p)), 2)
      const zero = fromStorage(
        Uint8Array.from(toFlat(primalOf(s)), (v) => (v === 0 ? 1 : 0)),
        s.shape,
        'bool',
      )
      return where(zero, 0, pow(where(zero, 1, s), 1 / p))
    }
    case 'seuclidean': {
      const v = options.v === undefined ? sampleVariance(concat([a, b], 0)) : (promoteFloat(options.v)[0] as Value)
      return safeSqrt(sum(div(square(differences(a, b)), reshape(v, [1, 1, d])), 2))
    }
    case 'cosine': {
      const dotAb = matmul(a, swapaxes(b, 0, 1))
      const na = reshape(safeSqrt(sum(square(a), 1)), [-1, 1])
      const nb = reshape(safeSqrt(sum(square(b), 1)), [1, -1])
      return sub(1, safeDivide(dotAb, mul(na, nb)))
    }
    case 'correlation':
      return cdist(sub(a, mean(a, 1, true)), sub(b, mean(b, 1, true)), 'cosine')
    case 'hamming': {
      // Fraction of differing coordinates (on the numbers, not differentiable).
      const diff = toFlat(primalOf(differences(a, b))).map((v) => (v !== 0 ? 1 : 0))
      return mean(fromStorage(Float64Array.from(diff), [a.shape[0] ?? 0, b.shape[0] ?? 0, d], 'float64'), 2)
    }
    case 'jaccard': {
      // Among coordinates where either is non-zero, the fraction that differ (scipy's boolean Jaccard).
      const av = toFlat(primalOf(a))
      const bv = toFlat(primalOf(b))
      const m = a.shape[0] ?? 0
      const n = b.shape[0] ?? 0
      const out = new Float64Array(m * n)
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let union = 0
          let differ = 0
          for (let k = 0; k < d; k++) {
            const x = (av[i * d + k] ?? 0) !== 0
            const y = (bv[j * d + k] ?? 0) !== 0
            if (x || y) union++
            if (x !== y) differ++
          }
          out[i * n + j] = union === 0 ? 0 : differ / union
        }
      }
      return fromStorage(out, [m, n], 'float64')
    }
    case 'mahalanobis': {
      const vi =
        options.vi === undefined ? inverseCovariance(concat([a, b], 0)) : (promoteFloat(options.vi)[0] as Value)
      const diff = differences(a, b)
      const projected = matmul(reshape(diff, [-1, d]), vi)
      return safeSqrt(reshape(sum(mul(projected, reshape(diff, [-1, d])), 1), [a.shape[0] ?? 0, b.shape[0] ?? 0]))
    }
    case 'canberra': {
      const diff = abs(differences(a, b))
      const denom = add(abs(reshape(a, [-1, 1, d])), abs(reshape(b, [1, -1, d])))
      return sum(safeDivide(diff, denom), 2)
    }
    case 'braycurtis': {
      const num = sum(abs(differences(a, b)), 2)
      const den = sum(abs(add(reshape(a, [-1, 1, d]), reshape(b, [1, -1, d]))), 2)
      return safeDivide(num, den)
    }
    case 'haversine': {
      if (d !== 2) throw new WhiteboxError('shape', 'cdist: haversine needs [latitude, longitude] rows')
      const lat1 = reshape(sliceAxis(a, 1, 0, 1), [-1, 1])
      const lat2 = reshape(sliceAxis(b, 1, 0, 1), [1, -1])
      const dlat = sub(lat2, lat1)
      const dlon = sub(reshape(sliceAxis(b, 1, 1, 2), [1, -1]), reshape(sliceAxis(a, 1, 1, 2), [-1, 1]))
      const h = add(square(sin(mul(dlat, 0.5))), mul(mul(cos(lat1), cos(lat2)), square(sin(mul(dlon, 0.5)))))
      return mul(arcsin(safeSqrt(h)), 2)
    }
  }
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
 * import { pdist } from 'whitebox-engine/numerics/geometry'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(pdist([[0, 0], [3, 4], [6, 8]])) // => [5, 10, 5]
 * ```
 */
export function pdist(x: TensorLike, metric: Metric = 'euclidean', options: MetricOptions = {}): Value {
  const [v] = promoteFloat(x) as [Value]
  const n = v.shape[0] ?? 0
  const opts =
    metric === 'seuclidean' && options.v === undefined
      ? { ...options, v: sampleVariance(v) }
      : metric === 'mahalanobis' && options.vi === undefined
        ? { ...options, vi: inverseCovariance(v) }
        : options
  const full = reshape(cdist(v, v, metric, opts), [-1])
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
 * import { squareform } from 'whitebox-engine/numerics/geometry'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(squareform([5, 10, 5])) // => [[0, 5, 10], [5, 0, 5], [10, 5, 0]]
 * ```
 */
export function squareform(d: TensorLike): Value {
  const [v] = promoteFloat(d) as [Value]
  if (v.shape.length === 1) {
    const len = v.shape[0] ?? 0
    const n = (1 + Math.sqrt(1 + 8 * len)) / 2
    if (!Number.isInteger(n)) throw new WhiteboxError('shape', 'squareform: the length is not n (n - 1) / 2')
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
    throw new WhiteboxError('shape', 'squareform: expected a vector or a square matrix')
  const idx: number[] = []
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) idx.push(i * n + j)
  return take(reshape(v, [-1]), fromStorage(Int32Array.from(idx), [idx.length], 'int32'), 0)
}
