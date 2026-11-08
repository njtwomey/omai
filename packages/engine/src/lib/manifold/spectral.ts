/**
 * Graph-spectral embeddings: Laplacian eigenmaps (Belkin and Niyogi 2003; scikit-learn's `SpectralEmbedding`), the bottom
 * eigenvectors of the normalised Laplacian of a neighbour or kernel graph, and diffusion maps (Coifman and Lafon 2006),
 * which embed by the leading eigenvectors of a random walk on the data so distances reflect diffusion over `t` steps.
 */

import { OmaiError } from '../base/index.ts'
import { type Affinity, affinityMatrix } from '../cluster/index.ts'
import { distanceMatrix } from '../distances/index.ts'
import { spectralEmbedding as graphSpectralEmbedding } from '../graph/index.ts'
import { defineEstimator, type Estimator, type Fitted, matrixView } from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { fromStorage, toArray, toFlat } from '../tensor/index.ts'
import { type EmbeddingParams, type EmbeddingVerbs, embeddingVerbs, trainingRows } from './_support.ts'

/** Hyperparameters of {@link spectralEmbedding}. */
export interface SpectralEmbeddingHyper {
  readonly nComponents: number
  readonly affinity: Affinity
  readonly nNeighbors: number
  readonly gamma: number | null
}

/**
 * Laplacian eigenmaps (sklearn.manifold.SpectralEmbedding): the bottom non-trivial eigenvectors of the normalised graph
 * Laplacian (`graph`'s spectral embedding), divided by the square roots of the degrees as sklearn does.
 *
 * @param hyper - Dimension and the affinity.
 * @returns The estimator.
 * @example
 * ```ts
 * import { spectralEmbedding } from 'omai-engine/manifold'
 * const x = Array.from({ length: 10 }, (_, i) => [i, (i % 3) * 0.1])
 * spectralEmbedding({ nNeighbors: 3 }).fit({ x }).params.embedding.length // => 10
 * ```
 */
export function spectralEmbedding(
  hyper: Partial<SpectralEmbeddingHyper> = {},
): Estimator<SpectralEmbeddingHyper, Fitted<EmbeddingParams> & EmbeddingVerbs> {
  return defineEstimator<SpectralEmbeddingHyper, EmbeddingParams, EmbeddingVerbs>(
    {
      name: 'manifold/spectral-embedding',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 3, { default: h.nComponents, label: 'dimensions' }),
          affinity: choice(['nearest_neighbors', 'rbf'] as const, {
            default: h.affinity === 'rbf' ? 'rbf' : 'nearest_neighbors',
          }),
          nNeighbors: int(2, 50, { default: h.nNeighbors, label: 'neighbours' }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'spectralEmbedding', { minSamples: 3 })
        const a = affinityMatrix(data.x, h.affinity, {
          nNeighbors: h.nNeighbors,
          ...(h.gamma === null ? {} : { gamma: h.gamma }),
        })
        // sklearn's laplacian ignores self-loops.
        for (let i = 0; i < m.n; i++) a[i * m.n + i] = 0
        const degree = Array.from({ length: m.n }, (_, i) => {
          let s = 0
          for (let j = 0; j < m.n; j++) s += a[i * m.n + j] ?? 0
          return Math.sqrt(s)
        })
        if (degree.some((v) => v === 0)) throw new OmaiError('value', 'spectralEmbedding: a row has no neighbours')
        const u = toArray(
          graphSpectralEmbedding(fromStorage(Float64Array.from(a), [m.n, m.n], 'float64'), h.nComponents, {
            dropFirst: true,
          }),
        ) as number[][]
        return {
          embedding: u.map((r, i) => r.map((v) => v / (degree[i] ?? 1))),
          data: trainingRows(data.x, 'spectralEmbedding'),
          objective: 0,
        }
      },
      methods: (p) => embeddingVerbs(p, 'spectral'),
    },
    { nComponents: 2, affinity: 'nearest_neighbors', nNeighbors: 10, gamma: null, ...hyper },
  )
}

/** Hyperparameters of {@link diffusionMap}. */
export interface DiffusionMapHyper {
  readonly nComponents: number
  /** The kernel width k(x, y) = exp(-|x - y|^2 / epsilon) (`null`: `bandwidth` times the median squared distance). */
  readonly epsilon: number | null
  /** The kernel width as a multiple of the median squared distance, when `epsilon` is `null`. */
  readonly bandwidth: number
  /** Diffusion time t. */
  readonly time: number
  /** Density normalisation exponent (0 graph Laplacian, 1 Laplace-Beltrami). */
  readonly alpha: number
}

/**
 * Diffusion maps (Coifman and Lafon 2006): a Gaussian kernel, alpha-normalised for density, made a Markov matrix; the
 * embedding is its leading non-trivial right eigenvectors scaled by eigenvalue^t.
 *
 * @param hyper - Dimension, kernel width, diffusion time and density normalisation.
 * @returns The estimator.
 * @example
 * ```ts
 * import { diffusionMap } from 'omai-engine/manifold'
 * const x = Array.from({ length: 10 }, (_, i) => [i, (i % 3) * 0.1])
 * diffusionMap({ nComponents: 2 }).fit({ x }).params.embedding.length // => 10
 * ```
 */
export function diffusionMap(
  hyper: Partial<DiffusionMapHyper> = {},
): Estimator<DiffusionMapHyper, Fitted<EmbeddingParams> & EmbeddingVerbs> {
  return defineEstimator<DiffusionMapHyper, EmbeddingParams, EmbeddingVerbs>(
    {
      name: 'manifold/diffusion-map',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 3, { default: h.nComponents, label: 'dimensions' }),
          time: int(1, 20, { default: h.time, label: 'diffusion time' }),
          alpha: real(0, 1, { default: h.alpha, label: 'density normalisation' }),
          bandwidth: real(0.01, 10, { log: true, default: h.bandwidth, label: 'kernel width' }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'diffusionMap', { minSamples: 3 })
        const k = h.nComponents
        const { n, data: d } = distanceMatrix(data.x, 'squaredEuclidean')
        const upper: number[] = []
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) upper.push(d[i * n + j] ?? 0)
        upper.sort((a, b) => a - b)
        const eps = h.epsilon ?? h.bandwidth * (upper[Math.floor(upper.length / 2)] ?? 1)
        const kern = Array.from({ length: n }, (_, i) =>
          Array.from({ length: n }, (_, j) => Math.exp(-(d[i * n + j] ?? 0) / eps)),
        )
        const q = kern.map((r) => r.reduce((s, v) => s + v, 0) ** h.alpha)
        const ka = kern.map((r, i) => r.map((v, j) => v / ((q[i] ?? 1) * (q[j] ?? 1))))
        const deg = ka.map((r) => r.reduce((s, v) => s + v, 0))
        const sym = ka.map((r, i) => r.map((v, j) => v / Math.sqrt((deg[i] ?? 1) * (deg[j] ?? 1))))
        const { eigenvalues, eigenvectors } = linalg.eigh(sym)
        const values = toFlat(eigenvalues)
        const vectors = toArray(eigenvectors) as number[][]
        const top = Array.from({ length: k }, (_, c) => m.n - 2 - c)
        return {
          embedding: vectors.map((r, i) =>
            top.map((c) => ((r[c] ?? 0) / Math.sqrt(deg[i] ?? 1)) * (values[c] ?? 0) ** h.time),
          ),
          data: trainingRows(data.x, 'diffusionMap'),
          objective: 0,
        }
      },
      methods: (p) => embeddingVerbs(p, 'diffusion'),
    },
    { nComponents: 2, epsilon: null, bandwidth: 1, time: 1, alpha: 1, ...hyper },
  )
}
