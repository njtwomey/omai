/**
 * Internal cluster validity indices, which judge a clustering from the data alone without reference labels: the
 * silhouette coefficient per sample and on average, the Calinski-Harabasz variance ratio, the Davies-Bouldin index and
 * the Dunn index, as in scikit-learn.
 */

import { MdmlError, type MetricResult, type MetricSpec } from '../base/index.ts'
import { cdist, type Metric as DistanceName } from '../numerics/geometry/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { labelValues, sortLabels } from './_labels.ts'
import { defineMetric, type MetricFunction, type MetricInput, matrix, metricResult } from './metric.ts'

const spec = (name: string, direction: 'higher' | 'lower', range: readonly [number, number]): MetricSpec => ({
  name,
  direction,
  range,
  inputs: 'values',
  category: 'clustering',
  concept: 'cluster-validity',
})

/** Options of the internal validity indices. */
export interface ValidityOptions {
  /** The distance between samples (any scipy metric; default 'euclidean'), or 'precomputed' when x is a distance matrix. */
  readonly metric?: DistanceName | 'precomputed'
}

/** Points, their cluster indices and the cluster count. */
function clusters(
  name: string,
  x: TensorLike,
  labels: MetricInput,
): { points: { shape: readonly number[]; data: Float64Array }; idx: Int32Array; k: number; n: number } {
  const points = matrix(x)
  const l = labelValues(labels)
  const n = points.shape[0] ?? 0
  if (l.length !== n) throw new MdmlError('shape', `${name}: one label per sample`)
  const classes = sortLabels(l)
  const pos = new Map(classes.map((c, i) => [c, i]))
  return { points, idx: Int32Array.from(l, (v) => pos.get(v) ?? 0), k: classes.length, n }
}

/** The full distance matrix of the samples. */
function distances(x: TensorLike, n: number, metric: DistanceName | 'precomputed' | undefined): Float64Array {
  if (metric === 'precomputed') return matrix(x).data
  const d = matrix(cdist(x, x, metric ?? 'euclidean')).data
  if (d.length !== n * n) throw new MdmlError('shape', 'distance matrix has the wrong size')
  return d
}

/**
 * Silhouette coefficient of each sample (sklearn.metrics.silhouette_samples): (b - a) / max(a, b) with a the mean
 * distance to the other members of its cluster and b the smallest mean distance to another cluster; 0 for samples
 * alone in their cluster.
 *
 * @param x - Samples `[N, D]` (or a distance matrix with metric 'precomputed').
 * @param labels - Cluster labels `[N]` (between 2 and N - 1 clusters).
 * @param options - The distance metric.
 * @returns One coefficient per sample.
 * @throws MdmlError with code `value` for fewer than 2 or more than N - 1 clusters.
 * @example
 * ```ts
 * import { silhouetteSamples } from 'mdml-engine/metrics'
 * silhouetteSamples([[0], [1], [10], [11]], [0, 0, 1, 1])[0] // => 0.9047619047619048
 * ```
 */
export function silhouetteSamples(x: TensorLike, labels: MetricInput, options: ValidityOptions = {}): number[] {
  const c = clusters('silhouetteSamples', x, labels)
  if (c.k < 2 || c.k > c.n - 1) throw new MdmlError('value', `silhouette: ${c.k} clusters; needs 2 to n_samples - 1`)
  const d = distances(x, c.n, options.metric)
  const sizes = new Float64Array(c.k)
  for (const i of c.idx) sizes[i] = (sizes[i] ?? 0) + 1
  return Array.from({ length: c.n }, (_, i) => {
    const sums = new Float64Array(c.k)
    for (let j = 0; j < c.n; j++) sums[c.idx[j] ?? 0] = (sums[c.idx[j] ?? 0] ?? 0) + (d[i * c.n + j] ?? 0)
    const own = c.idx[i] ?? 0
    const size = sizes[own] ?? 1
    if (size <= 1) return 0
    const a = (sums[own] ?? 0) / (size - 1)
    let b = Number.POSITIVE_INFINITY
    for (let q = 0; q < c.k; q++) if (q !== own) b = Math.min(b, (sums[q] ?? 0) / (sizes[q] ?? 1))
    const m = Math.max(a, b)
    return m === 0 ? 0 : (b - a) / m
  })
}

/**
 * The mean silhouette coefficient (Rousseeuw, 1987; sklearn.metrics.silhouette_score), in [-1, 1].
 *
 * @param x - Samples `[N, D]` (or a distance matrix with metric 'precomputed').
 * @param labels - Cluster labels `[N]`.
 * @param options - The distance metric.
 * @returns The metric result.
 * @concept cluster-validity
 * @example
 * ```ts
 * import { silhouetteScore } from 'mdml-engine/metrics'
 * Number(silhouetteScore([[0], [1], [10], [11]], [0, 0, 1, 1]).estimate.value.toFixed(4)) // => 0.8997
 * ```
 */
export const silhouetteScore: MetricFunction<[TensorLike, MetricInput, ValidityOptions?]> = defineMetric(
  spec('silhouetteScore', 'higher', [-1, 1]),
  (x: TensorLike, labels: MetricInput, options: ValidityOptions = {}): MetricResult => {
    const s = silhouetteSamples(x, labels, options)
    return metricResult('silhouetteScore', s.reduce((a, b) => a + b, 0) / s.length, s.length)
  },
)

/** Cluster centroids `[K, D]` and the overall mean. */
function centroids(
  points: { shape: readonly number[]; data: Float64Array },
  idx: Int32Array,
  k: number,
): { c: Float64Array; mean: Float64Array; sizes: Float64Array; d: number } {
  const n = points.shape[0] ?? 0
  const d = points.shape[1] ?? 1
  const c = new Float64Array(k * d)
  const mean = new Float64Array(d)
  const sizes = new Float64Array(k)
  for (let i = 0; i < n; i++) {
    const q = idx[i] ?? 0
    sizes[q] = (sizes[q] ?? 0) + 1
    for (let j = 0; j < d; j++) {
      c[q * d + j] = (c[q * d + j] ?? 0) + (points.data[i * d + j] ?? 0)
      mean[j] = (mean[j] ?? 0) + (points.data[i * d + j] ?? 0) / n
    }
  }
  for (let q = 0; q < k; q++) for (let j = 0; j < d; j++) c[q * d + j] = (c[q * d + j] ?? 0) / (sizes[q] ?? 1)
  return { c, mean, sizes, d }
}

/**
 * The Calinski-Harabasz index (variance ratio criterion; sklearn.metrics.calinski_harabasz_score):
 * [tr(B) / (k - 1)] / [tr(W) / (n - k)] for between- and within-cluster dispersion; 1 when W is zero.
 *
 * @param x - Samples `[N, D]`.
 * @param labels - Cluster labels `[N]`.
 * @returns The metric result (higher is better).
 * @example
 * ```ts
 * import { calinskiHarabaszScore } from 'mdml-engine/metrics'
 * calinskiHarabaszScore([[0], [2], [10], [12]], [0, 0, 1, 1]).estimate.value // => 50
 * ```
 */
export const calinskiHarabaszScore: MetricFunction<[TensorLike, MetricInput]> = defineMetric(
  spec('calinskiHarabaszScore', 'higher', [0, Infinity]),
  (x: TensorLike, labels: MetricInput): MetricResult => {
    const c = clusters('calinskiHarabaszScore', x, labels)
    if (c.k < 2 || c.k > c.n - 1) throw new MdmlError('value', 'calinskiHarabaszScore: needs 2 to n - 1 clusters')
    const { c: mu, mean, sizes, d } = centroids(c.points, c.idx, c.k)
    let between = 0
    for (let q = 0; q < c.k; q++) {
      let s = 0
      for (let j = 0; j < d; j++) s += ((mu[q * d + j] ?? 0) - (mean[j] ?? 0)) ** 2
      between += (sizes[q] ?? 0) * s
    }
    let within = 0
    for (let i = 0; i < c.n; i++) {
      const q = c.idx[i] ?? 0
      for (let j = 0; j < d; j++) within += ((c.points.data[i * d + j] ?? 0) - (mu[q * d + j] ?? 0)) ** 2
    }
    const value = within === 0 ? 1 : (between * (c.n - c.k)) / (within * (c.k - 1))
    return metricResult('calinskiHarabaszScore', value, c.n)
  },
)

/**
 * The Davies-Bouldin index (sklearn.metrics.davies_bouldin_score): the mean over clusters of the worst ratio
 * (s_i + s_j) / d(c_i, c_j) of within-cluster scatter to centroid separation (lower is better).
 *
 * @param x - Samples `[N, D]`.
 * @param labels - Cluster labels `[N]`.
 * @returns The metric result.
 * @example
 * ```ts
 * import { daviesBouldinScore } from 'mdml-engine/metrics'
 * daviesBouldinScore([[0], [2], [10], [12]], [0, 0, 1, 1]).estimate.value // => 0.2
 * ```
 */
export const daviesBouldinScore: MetricFunction<[TensorLike, MetricInput]> = defineMetric(
  spec('daviesBouldinScore', 'lower', [0, Infinity]),
  (x: TensorLike, labels: MetricInput): MetricResult => {
    const c = clusters('daviesBouldinScore', x, labels)
    if (c.k < 2 || c.k > c.n - 1) throw new MdmlError('value', 'daviesBouldinScore: needs 2 to n - 1 clusters')
    const { c: mu, sizes, d } = centroids(c.points, c.idx, c.k)
    const scatter = new Float64Array(c.k)
    for (let i = 0; i < c.n; i++) {
      const q = c.idx[i] ?? 0
      let s = 0
      for (let j = 0; j < d; j++) s += ((c.points.data[i * d + j] ?? 0) - (mu[q * d + j] ?? 0)) ** 2
      scatter[q] = (scatter[q] ?? 0) + Math.sqrt(s) / (sizes[q] ?? 1)
    }
    const sep = (a: number, b: number): number => {
      let s = 0
      for (let j = 0; j < d; j++) s += ((mu[a * d + j] ?? 0) - (mu[b * d + j] ?? 0)) ** 2
      return Math.sqrt(s)
    }
    let allScatterZero = true
    let allSepZero = true
    for (const v of scatter) if (Math.abs(v) > 1e-8) allScatterZero = false
    let total = 0
    for (let a = 0; a < c.k; a++) {
      let worst = 0
      for (let b = 0; b < c.k; b++) {
        if (a === b) continue
        const s = sep(a, b)
        if (Math.abs(s) > 1e-8) allSepZero = false
        // sklearn replaces zero separations with infinity, so the pair contributes 0.
        const r = s === 0 ? 0 : ((scatter[a] ?? 0) + (scatter[b] ?? 0)) / s
        worst = Math.max(worst, r)
      }
      total += worst
    }
    const value = allScatterZero || allSepZero ? 0 : total / c.k
    return metricResult('daviesBouldinScore', value, c.n)
  },
)

/**
 * The Dunn index (Dunn, 1974): the smallest distance between points of different clusters divided by the largest
 * cluster diameter (higher is better: compact, well-separated clusters).
 *
 * @param x - Samples `[N, D]` (or a distance matrix with metric 'precomputed').
 * @param labels - Cluster labels `[N]`.
 * @param options - The distance metric.
 * @returns The metric result.
 * @example
 * ```ts
 * import { dunnIndex } from 'mdml-engine/metrics'
 * dunnIndex([[0], [2], [10], [12]], [0, 0, 1, 1]).estimate.value // => 4
 * ```
 */
export const dunnIndex: MetricFunction<[TensorLike, MetricInput, ValidityOptions?]> = defineMetric(
  spec('dunnIndex', 'higher', [0, Infinity]),
  (x: TensorLike, labels: MetricInput, options: ValidityOptions = {}): MetricResult => {
    const c = clusters('dunnIndex', x, labels)
    if (c.k < 2) throw new MdmlError('value', 'dunnIndex: needs at least 2 clusters')
    const d = distances(x, c.n, options.metric)
    let separation = Number.POSITIVE_INFINITY
    let diameter = 0
    for (let i = 0; i < c.n; i++) {
      for (let j = i + 1; j < c.n; j++) {
        const v = d[i * c.n + j] ?? 0
        if (c.idx[i] === c.idx[j]) diameter = Math.max(diameter, v)
        else separation = Math.min(separation, v)
      }
    }
    return metricResult('dunnIndex', diameter === 0 ? Number.POSITIVE_INFINITY : separation / diameter, c.n)
  },
)
