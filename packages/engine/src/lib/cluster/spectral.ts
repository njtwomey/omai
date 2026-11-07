/**
 * Spectral clustering (Ng, Jordan and Weiss 2001; as scikit-learn assembles it): an affinity graph from nearest
 * neighbours or an RBF kernel, the rows embedded by the leading eigenvectors of its normalised Laplacian
 * (`graph/spectral.ts`), and k-means on the embedding, which separates clusters that are connected but not convex.
 */

import { MdmlError, toRows } from '../base/index.ts'
import { spectralEmbedding } from '../graph/index.ts'
import { pairwiseKernels } from '../kernels/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { kNearest } from '../numerics/spatial/index.ts'
import type { Key } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import type { Algorithm } from '../trace/index.ts'
import { labelledPredict } from './_support.ts'
import { initialCentres, type LloydState, lloydAlgorithm } from './kmeans.ts'

/** How the affinity graph is built: symmetrised k-nearest-neighbour connectivity, an RBF kernel, or given. */
export type Affinity = 'nearest_neighbors' | 'rbf' | 'precomputed'

/**
 * The affinity matrix of scikit-learn's spectral methods: 0.5 (A + A^T) for k-nearest-neighbour connectivity A (each
 * row connected to itself and its k - 1 nearest others), or exp(-gamma |x - y|^2).
 *
 * @param x - Points `[n, d]` (the affinity itself with `'precomputed'`).
 * @param affinity - The construction (default nearest_neighbors).
 * @param options - Neighbours (default max(n / 10, 1)) and gamma (default 1 / d).
 * @returns The symmetric affinity `[n, n]`, flat.
 * @example
 * ```ts
 * import { affinityMatrix } from 'mdml-engine/cluster'
 * affinityMatrix([[0], [1], [5]], 'nearest_neighbors', { nNeighbors: 2 }) // => [1, 1, 0, 1, 1, 0.5, 0, 0.5, 1]
 * ```
 */
export function affinityMatrix(
  x: TensorLike,
  affinity: Affinity = 'nearest_neighbors',
  options: { readonly nNeighbors?: number; readonly gamma?: number } = {},
): number[] {
  const m = matrixView(x, 'affinityMatrix', { minSamples: 2 })
  if (affinity === 'precomputed') return Array.from(m.data)
  if (affinity === 'rbf') return toFlat(pairwiseKernels(x, undefined, 'rbf', { gamma: options.gamma ?? 1 / m.d }))
  const k = options.nNeighbors ?? Math.max(Math.floor(m.n / 10), 1)
  if (!(Number.isInteger(k) && k >= 1 && k <= m.n))
    throw new MdmlError('value', `affinityMatrix: nNeighbors must be in 1..${m.n}`)
  const a = new Array<number>(m.n * m.n).fill(0)
  for (let i = 0; i < m.n; i++) a[i * m.n + i] = 1
  if (k > 1) {
    const idx = kNearest(x, undefined, k - 1, { excludeSelf: true }).indices.data
    for (let i = 0; i < m.n; i++) for (let t = 0; t < k - 1; t++) a[i * m.n + (idx[i * (k - 1) + t] ?? 0)] = 1
  }
  return a.map((_, p) => 0.5 * ((a[p] ?? 0) + (a[(p % m.n) * m.n + Math.floor(p / m.n)] ?? 0)))
}

/** Options of spectral clustering. */
export interface SpectralOptions {
  readonly nClusters: number
  readonly affinity?: Affinity
  readonly nNeighbors?: number
  readonly gamma?: number
}

/**
 * The spectral embedding of the rows for clustering: the leading eigenvectors of the normalised Laplacian of the
 * affinity graph (self-affinities removed), rows normalised to unit length.
 *
 * @param x - Points `[n, d]`.
 * @param options - Clusters (the embedding dimension) and the affinity.
 * @returns The embedding `[n, k]`, flat, and `n`.
 * @example
 * ```ts
 * import { spectralClusteringEmbedding } from 'mdml-engine/cluster'
 * spectralClusteringEmbedding([[0], [0.1], [5], [5.1]], { nClusters: 2, affinity: 'rbf', gamma: 1 }).n // => 4
 * ```
 */
export function spectralClusteringEmbedding(
  x: TensorLike,
  options: SpectralOptions,
): { readonly n: number; readonly data: number[] } {
  const m = matrixView(x, 'spectralClustering', { minSamples: 3 })
  const a = affinityMatrix(x, options.affinity ?? 'rbf', {
    ...(options.nNeighbors === undefined ? {} : { nNeighbors: options.nNeighbors }),
    ...(options.gamma === undefined ? {} : { gamma: options.gamma }),
  })
  for (let i = 0; i < m.n; i++) a[i * m.n + i] = 0
  const y = spectralEmbedding(fromStorage(Float64Array.from(a), [m.n, m.n], 'float64'), options.nClusters, {
    rowNormalise: true,
  })
  return { n: m.n, data: toFlat(y) }
}

/**
 * Spectral clustering as a traced {@link Algorithm}: the spectral embedding of the rows, then Lloyd's algorithm on it,
 * one iteration per step. The centres and labels it observes live in the embedding; the labels are the clusters.
 *
 * @param x - Points `[n, d]`.
 * @param options - Clusters and the affinity.
 * @param key - Randomness for the k-means start.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { spectralClusteringAlgorithm } from 'mdml-engine/cluster'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0], [0.1], [0.2], [5], [5.1], [5.2]]
 * const alg = spectralClusteringAlgorithm(x, { nClusters: 2, affinity: 'rbf', gamma: 1 }, key(0))
 * runAlgorithm(alg, { maxSteps: 20 }).state.labels // => [0, 0, 0, 1, 1, 1]
 * ```
 */
export function spectralClusteringAlgorithm(x: TensorLike, options: SpectralOptions, key: Key): Algorithm<LloydState> {
  const { n, data } = spectralClusteringEmbedding(x, options)
  const embedding = toRows(data, n, options.nClusters)
  return {
    ...lloydAlgorithm(embedding, initialCentres(embedding, options.nClusters, 'k-means++', key)),
    name: 'spectral-clustering',
  }
}

/** Hyperparameters of {@link spectralClustering}. */
export interface SpectralClusteringHyper {
  readonly nClusters: number
  readonly affinity: 'nearest_neighbors' | 'rbf'
  readonly nNeighbors: number
  readonly gamma: number
  /** Starts of the k-means on the embedding; the lowest inertia wins. */
  readonly nInit: number
}

/** The fitted parameters of {@link spectralClustering}: the labels, and the training rows (new rows take their nearest row's label). */
export type SpectralClusteringParams = {
  readonly labels: number[]
  readonly data: number[][]
}

/** The verbs of fitted spectral clustering. */
export interface SpectralVerbs {
  readonly predict: (x: TensorLike) => Tensor
}

/**
 * Spectral clustering (sklearn.cluster.SpectralClustering) on the iterative base: the embedding is computed once and
 * the k-means on it runs a step at a time. Spectral clustering has no out-of-sample extension: `predict` gives a new
 * row the label of its nearest training row.
 *
 * @index estimator Spectral clustering on an affinity graph.
 * @example
 * ```ts
 * import { spectralClustering } from 'mdml-engine/cluster'
 * spectralClustering.with({ nClusters: 2, gamma: 1 }).fit({ x: [[0], [0.1], [0.2], [5], [5.1], [5.2]] }).params.labels.length // => 6
 * ```
 */
export const spectralClustering: IterativeEstimator<
  SpectralClusteringHyper,
  LloydState,
  SpectralClusteringParams,
  SpectralVerbs
> = iterative<SpectralClusteringHyper, LloydState, SpectralClusteringParams, SpectralVerbs>({
  name: 'cluster/spectral',
  kind: 'clusterer',
  defaults: { nClusters: 2, affinity: 'rbf', nNeighbors: 10, gamma: 1, nInit: 4 },
  hyperSpace: (h) =>
    dict({
      nClusters: int(2, 20, { default: h.nClusters, label: 'clusters' }),
      affinity: choice(['rbf', 'nearest_neighbors'] as const, { default: h.affinity }),
      nNeighbors: int(2, 50, { default: h.nNeighbors, label: 'neighbours' }),
      gamma: real(1e-3, 100, { log: true, default: h.gamma, label: 'kernel gamma' }),
    }),
  algorithm: (data, h, key) =>
    spectralClusteringAlgorithm(
      data.x,
      { nClusters: h.nClusters, affinity: h.affinity, nNeighbors: h.nNeighbors, gamma: h.gamma },
      key,
    ),
  maxSteps: () => 300,
  restarts: { count: (h) => h.nInit, better: (a, b) => a.inertia < b.inertia },
  params: (s, data) => {
    const m = matrixView(data.x, 'spectralClustering')
    return { labels: s.labels, data: toRows(m.data, m.n, m.d) }
  },
  verbs: (p) => ({
    predict: labelledPredict(p.data, p.labels, Number.POSITIVE_INFINITY, 'euclidean', p.data[0]?.length ?? 0),
  }),
})
