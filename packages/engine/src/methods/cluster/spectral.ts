/**
 * Spectral clustering (Ng, Jordan and Weiss, 2001; as in scikit-learn): build an affinity graph from nearest neighbours
 * or an RBF kernel, embed the rows with the leading eigenvectors of its normalised Laplacian, and cluster the embedding
 * with k-means, which separates clusters that are connected but not convex.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { clusteringSteps, recordsOf } from '../../lib/exposition/index.ts'
import { spectralEmbedding } from '../../lib/graph/index.ts'
import { pairwiseKernels } from '../../lib/kernels/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  matrixView,
} from '../../lib/learn/index.ts'
import { kNearest } from '../../lib/numerics/spatial/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { type TensorLike, toArray } from '../../lib/tensor/index.ts'
import { clusterMeans, kmeans } from './kmeans.ts'

/** How the affinity graph is built: symmetrised k-nearest-neighbour connectivity, or an RBF kernel. */
export type Affinity = 'nearest_neighbors' | 'rbf'

/**
 * The affinity matrix of sklearn's spectral methods: 0.5 (A + A^T) for k-nearest-neighbour connectivity A (each row
 * connected to itself and its k - 1 nearest others), or exp(-gamma |x - y|^2).
 *
 * @param x - Points `[n, d]`.
 * @param affinity - The construction.
 * @param options - Neighbours (default max(n / 10, 1)) and gamma (default 1 / d).
 * @returns The symmetric affinity `[n][n]`.
 */
export function affinityMatrix(
  x: TensorLike,
  affinity: Affinity = 'nearest_neighbors',
  options: { readonly nNeighbors?: number; readonly gamma?: number } = {},
): number[][] {
  const m = matrixView(x, 'affinityMatrix', { minSamples: 2 })
  if (affinity === 'rbf')
    return toArray(pairwiseKernels(x, undefined, 'rbf', { gamma: options.gamma ?? 1 / m.d })) as number[][]
  const k = options.nNeighbors ?? Math.max(Math.floor(m.n / 10), 1)
  if (!(Number.isInteger(k) && k >= 1 && k <= m.n))
    throw new MdmlError('value', `affinityMatrix: nNeighbors must be in 1..${m.n}`)
  const a = Array.from({ length: m.n }, (_, i) => Array.from({ length: m.n }, (_, j) => (i === j ? 1 : 0)))
  if (k > 1) {
    const idx = Array.from(kNearest(x, undefined, k - 1, { excludeSelf: true }).indices.data)
    for (let i = 0; i < m.n; i++) for (let t = 0; t < k - 1; t++) (a[i] as number[])[idx[i * (k - 1) + t] ?? 0] = 1
  }
  return a.map((r, i) => r.map((v, j) => 0.5 * (v + (a[j]?.[i] ?? 0))))
}

/** Hyperparameters of spectral clustering. */
export type SpectralClusteringHyper = {
  readonly nClusters?: number
  readonly affinity?: Affinity
  readonly nNeighbors?: number
  readonly gamma?: number
}

/** The parameters of a fitted spectral clustering. */
export type SpectralClusteringParams = {
  readonly labels: number[]
  readonly embedding: number[][]
  readonly nFeaturesIn: number
}

/**
 * Spectral clustering (Ng, Jordan and Weiss 2001; sklearn.cluster.SpectralClustering): the leading eigenvectors of the
 * normalised Laplacian of the affinity graph (`graph`'s spectral embedding, rows normalised), clustered by k-means.
 * Its exposition is the k-means `clusteringSteps` record, with each step's centres the means of its clusters in the
 * data's own coordinates (k-means' centres live in the spectral embedding). TODO: spectral co-clustering.
 *
 * @param hyper - Clusters and the affinity.
 * @returns The estimator.
 */
export function spectralClustering(
  hyper: SpectralClusteringHyper = {},
): Estimator<SpectralClusteringHyper, ExplainedFitted<SpectralClusteringParams, { readonly nFeaturesIn: number }>> {
  return defineExplainedEstimator<SpectralClusteringHyper, SpectralClusteringParams, { readonly nFeaturesIn: number }>(
    {
      name: 'spectral-clustering',
      kind: 'clusterer',
      hyperSpace: (h) =>
        dict({
          nClusters: int(2, 20, { default: h.nClusters ?? 2, label: 'clusters' }),
          affinity: choice(['rbf', 'nearest_neighbors'], {
            default: h.affinity ?? 'rbf',
            label: 'affinity',
            description: 'A Gaussian kernel on distances, or a symmetric k-nearest-neighbour graph.',
          }),
          nNeighbors: int(2, 50, { default: h.nNeighbors ?? 10, label: 'neighbours' }),
          gamma: real(1e-3, 100, {
            log: true,
            default: h.gamma ?? 1,
            label: 'kernel γ',
            description: 'The Gaussian kernel exp(-γ |x - y|²) of the rbf affinity.',
          }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'spectralClustering', { minSamples: 3 })
        const k = h.nClusters ?? 2
        const a = affinityMatrix(data.x, h.affinity ?? 'rbf', {
          ...(h.nNeighbors === undefined ? {} : { nNeighbors: h.nNeighbors }),
          gamma: h.gamma ?? 1,
        })
        const off = a.map((r, i) => r.map((v, j) => (i === j ? 0 : v)))
        const embedding = toArray(spectralEmbedding(off, k, { rowNormalise: true })) as number[][]
        const km = kmeans({ nClusters: k, nInit: 4 }).fit({ x: embedding }, options)
        const [steps] = recordsOf(km.exposition, 'clusteringSteps')
        const assignments = steps?.assignments ?? [km.params.labels]
        const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        return {
          params: { labels: km.params.labels, embedding, nFeaturesIn: m.d },
          exposition: [
            clusteringSteps({
              method: 'spectralClustering',
              assignments,
              ...(steps?.objective === undefined ? {} : { objective: steps.objective }),
              centres: assignments.map((labels) => clusterMeans(rows, labels, k)),
            }),
          ],
        }
      },
      methods: (p) => ({ nFeaturesIn: p.nFeaturesIn }),
    },
    hyper,
  )
}

/** Spectral clustering as a class. */
export class SpectralClustering extends EstimatorClass<
  SpectralClusteringHyper,
  ExplainedFitted<SpectralClusteringParams, { readonly nFeaturesIn: number }>
> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link spectralClustering}.
   */
  constructor(hyper: SpectralClusteringHyper = {}) {
    super(spectralClustering(hyper))
  }
}
