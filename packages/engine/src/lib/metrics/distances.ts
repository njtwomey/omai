import { type MetricResult, type MetricSpec, WhiteboxError } from '../base/index.ts'
import { cdist, metricFunction, procrustes } from '../numerics/geometry/index.ts'
import { item, type TensorLike } from '../tensor/index.ts'
import { defineMetric, flat, type MetricFunction, matrix, metricResult } from './metric.ts'

const spec = (
  name: string,
  direction: 'higher' | 'lower' = 'lower',
  range: readonly [number, number] = [0, Infinity],
): MetricSpec => ({
  name,
  direction,
  range,
  inputs: 'values',
  category: 'distance',
  concept: 'distances',
})

const vectors = (name: string, u: TensorLike, v: TensorLike): [Float64Array, Float64Array] => {
  const a = flat(u)
  const b = flat(v)
  if (a.length !== b.length) throw new WhiteboxError('shape', `${name}: vectors differ in length`)
  return [a, b]
}

/**
 * The Minkowski distance (sum w |u - v|^p)^(1/p) between two vectors (scipy.spatial.distance.minkowski; an alias of
 * numerics/geometry's metric): p = 1 city block, p = 2 Euclidean, p = Infinity Chebyshev.
 *
 * @param u - First vector.
 * @param v - Second vector.
 * @param options - p (default 2) and per-coordinate weights w.
 * @returns The metric result.
 * @concept distances
 * @example
 * ```ts
 * import { minkowskiDistance } from 'whitebox-engine/metrics'
 * minkowskiDistance([0, 0], [3, 4]).estimate.value // => 5
 * ```
 */
export const minkowskiDistance: MetricFunction<
  [TensorLike, TensorLike, { readonly p?: number; readonly w?: TensorLike }?]
> = defineMetric(
  spec('minkowskiDistance'),
  (u: TensorLike, v: TensorLike, options: { readonly p?: number; readonly w?: TensorLike } = {}): MetricResult => {
    const [a, b] = vectors('minkowskiDistance', u, v)
    const p = options.p ?? 2
    const metric = p === Number.POSITIVE_INFINITY ? 'chebyshev' : 'minkowski'
    const d = metricFunction(metric, a.length, { p, ...(options.w === undefined ? {} : { w: options.w }) })(a, 0, b, 0)
    return metricResult('minkowskiDistance', d, 1)
  },
)

/**
 * Cosine similarity u . v / (|u| |v|) of two vectors, or the mean over rows of two matrices (torchmetrics'
 * cosine_similarity with mean reduction); 1 - scipy's cosine distance.
 *
 * @param u - First vector `[D]` or rows `[N, D]`.
 * @param v - Second vector or rows, the same shape.
 * @returns The metric result, in [-1, 1].
 * @example
 * ```ts
 * import { cosineSimilarity } from 'whitebox-engine/metrics'
 * Math.abs(cosineSimilarity([1, 0], [1, 1]).estimate.value - Math.SQRT1_2) < 1e-15 // => true
 * ```
 */
export const cosineSimilarity: MetricFunction<[TensorLike, TensorLike]> = defineMetric(
  spec('cosineSimilarity', 'higher', [-1, 1]),
  (u: TensorLike, v: TensorLike): MetricResult => {
    const a = matrix(u)
    const b = matrix(v)
    const d = a.shape[a.shape.length - 1] ?? 1
    const rows = a.data.length / d
    if (b.data.length !== a.data.length) throw new WhiteboxError('shape', 'cosineSimilarity: inputs differ in shape')
    let total = 0
    for (let r = 0; r < rows; r++) {
      let dot = 0
      let na = 0
      let nb = 0
      for (let j = 0; j < d; j++) {
        const x = a.data[r * d + j] ?? 0
        const y = b.data[r * d + j] ?? 0
        dot += x * y
        na += x * x
        nb += y * y
      }
      total += dot / Math.sqrt(na * nb)
    }
    return metricResult('cosineSimilarity', total / rows, rows)
  },
)

/**
 * The Mahalanobis distance sqrt((u - v)^T VI (u - v)) for an inverse covariance VI (scipy.spatial.distance.mahalanobis;
 * numerics/geometry's metric).
 *
 * @param u - First vector `[D]`.
 * @param v - Second vector `[D]`.
 * @param vi - The inverse covariance `[D, D]`.
 * @returns The metric result.
 * @example
 * ```ts
 * import { mahalanobisDistance } from 'whitebox-engine/metrics'
 * mahalanobisDistance([0, 0], [2, 0], [[0.25, 0], [0, 1]]).estimate.value // => 1
 * ```
 */
export const mahalanobisDistance: MetricFunction<[TensorLike, TensorLike, TensorLike]> = defineMetric(
  spec('mahalanobisDistance'),
  (u: TensorLike, v: TensorLike, vi: TensorLike): MetricResult => {
    const [a, b] = vectors('mahalanobisDistance', u, v)
    return metricResult('mahalanobisDistance', metricFunction('mahalanobis', a.length, { vi })(a, 0, b, 0), 1)
  },
)

/**
 * The Procrustes disparity of two configurations (scipy.spatial.procrustes, whose canonical home is
 * numerics/geometry's `procrustes`): the sum of squared differences after centring, unit scaling and the best rotation.
 *
 * @param data1 - Reference points `[N, D]`.
 * @param data2 - Points to fit `[N, D]`.
 * @returns The metric result, in [0, 1].
 * @example
 * ```ts
 * import { procrustesDisparity } from 'whitebox-engine/metrics'
 * procrustesDisparity([[0, 0], [1, 0], [0, 1]], [[0, 0], [2, 0], [0, 2]]).estimate.value < 1e-24 // => true
 * ```
 */
export const procrustesDisparity: MetricFunction<[TensorLike, TensorLike]> = defineMetric(
  spec('procrustesDisparity', 'lower', [0, 1]),
  (data1: TensorLike, data2: TensorLike): MetricResult =>
    metricResult('procrustesDisparity', item(procrustes(data1, data2).disparity), matrix(data1).shape[0] ?? 0),
)

/** For each point of a, the distance to its nearest point of b (Euclidean). */
function nearest(a: TensorLike, b: TensorLike): { ab: number[]; ba: number[] } {
  const d = matrix(cdist(a, b))
  const n = d.shape[0] ?? 0
  const m = d.shape[1] ?? 0
  const ab = Array.from({ length: n }, (_, i) => {
    let best = Number.POSITIVE_INFINITY
    for (let j = 0; j < m; j++) best = Math.min(best, d.data[i * m + j] ?? 0)
    return best
  })
  const ba = Array.from({ length: m }, (_, j) => {
    let best = Number.POSITIVE_INFINITY
    for (let i = 0; i < n; i++) best = Math.min(best, d.data[i * m + j] ?? 0)
    return best
  })
  return { ab, ba }
}

/**
 * The Hausdorff distance between two point sets: the largest distance from a point of one set to the nearest point of
 * the other (scipy.spatial.distance.directed_hausdorff, symmetrised by default).
 *
 * @param a - Points `[N, D]`.
 * @param b - Points `[M, D]`.
 * @param options - directed (only from a to b; default false).
 * @returns The metric result.
 * @example
 * ```ts
 * import { hausdorffDistance } from 'whitebox-engine/metrics'
 * hausdorffDistance([[0, 0], [1, 0]], [[0, 0], [4, 0]]).estimate.value // => 3
 * ```
 */
export const hausdorffDistance: MetricFunction<[TensorLike, TensorLike, { readonly directed?: boolean }?]> =
  defineMetric(
    spec('hausdorffDistance'),
    (a: TensorLike, b: TensorLike, options: { readonly directed?: boolean } = {}): MetricResult => {
      const { ab, ba } = nearest(a, b)
      const forward = ab.reduce((x, y) => Math.max(x, y), 0)
      const value =
        options.directed === true
          ? forward
          : Math.max(
              forward,
              ba.reduce((x, y) => Math.max(x, y), 0),
            )
      return metricResult('hausdorffDistance', value, ab.length + ba.length)
    },
  )

/**
 * The Chamfer distance between two point sets (as PyTorch3D's chamfer_distance): the mean over each set of the
 * squared (default) or plain distance to the nearest point of the other set, summed over the two directions.
 *
 * @param a - Points `[N, D]`.
 * @param b - Points `[M, D]`.
 * @param options - squared (default true).
 * @returns The metric result.
 * @example
 * ```ts
 * import { chamferDistance } from 'whitebox-engine/metrics'
 * chamferDistance([[0, 0]], [[0, 2], [0, 0]]).estimate.value // => 2
 * ```
 */
export const chamferDistance: MetricFunction<[TensorLike, TensorLike, { readonly squared?: boolean }?]> = defineMetric(
  spec('chamferDistance'),
  (a: TensorLike, b: TensorLike, options: { readonly squared?: boolean } = {}): MetricResult => {
    const { ab, ba } = nearest(a, b)
    const f = (d: number): number => ((options.squared ?? true) ? d * d : d)
    const mean = (xs: readonly number[]): number => xs.reduce((s, d) => s + f(d), 0) / xs.length
    return metricResult('chamferDistance', mean(ab) + mean(ba), ab.length + ba.length)
  },
)
