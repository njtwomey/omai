import { WhiteboxError } from '../../lib/base/index.ts'
import { embeddingSteps } from '../../lib/exposition/index.ts'
import { spectralEmbedding as graphSpectralEmbedding } from '../../lib/graph/index.ts'
import { pairwiseKernels } from '../../lib/kernels/index.ts'
import { defineExplainedEstimator, type Estimator, EstimatorClass, matrixView } from '../../lib/learn/index.ts'
import { linalg } from '../../lib/numerics/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { type TensorLike, toArray, toFlat } from '../../lib/tensor/index.ts'
import { neighbourEdges } from './isomap.ts'
import type { EmbeddingParams, FittedEmbedding } from './mds.ts'

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
  const a = Array.from({ length: m.n }, (_, i) => Array.from({ length: m.n }, (_, j) => (i === j ? 1 : 0)))
  if (k > 1) for (const [i, j] of neighbourEdges(x, k - 1).edges) (a[i] as number[])[j] = 1
  return a.map((r, i) => r.map((v, j) => 0.5 * (v + (a[j]?.[i] ?? 0))))
}

/** Hyperparameters of the spectral embedding. */
export type SpectralEmbeddingHyper = {
  readonly nComponents?: number
  readonly affinity?: Affinity
  readonly nNeighbors?: number
  readonly gamma?: number
}

/**
 * Laplacian eigenmaps (Belkin and Niyogi 2003; sklearn.manifold.SpectralEmbedding): the bottom non-trivial
 * eigenvectors of the normalised graph Laplacian (`graph`'s spectral embedding), divided by the square roots of the
 * degrees as sklearn does. Its exposition is an `embeddingSteps` record with the affinity graph.
 *
 * @param hyper - Dimension and the affinity.
 * @returns The estimator.
 */
export function spectralEmbedding(
  hyper: SpectralEmbeddingHyper = {},
): Estimator<SpectralEmbeddingHyper, FittedEmbedding> {
  return defineExplainedEstimator<SpectralEmbeddingHyper, EmbeddingParams, { readonly nFeaturesIn: number }>(
    {
      name: 'spectral-embedding',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 3, { default: h.nComponents ?? 2, label: 'dimensions' }),
          affinity: choice(['nearest_neighbors', 'rbf'], { default: h.affinity ?? 'nearest_neighbors' }),
          nNeighbors: int(2, 50, { default: h.nNeighbors ?? 10, label: 'neighbours' }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'spectralEmbedding', { minSamples: 3 })
        const k = h.nComponents ?? 2
        const a = affinityMatrix(data.x, h.affinity ?? 'nearest_neighbors', {
          ...(h.nNeighbors === undefined ? {} : { nNeighbors: h.nNeighbors }),
          ...(h.gamma === undefined ? {} : { gamma: h.gamma }),
        })
        // sklearn's laplacian ignores self-loops.
        const off = a.map((r, i) => r.map((v, j) => (i === j ? 0 : v)))
        const degree = off.map((r) => Math.sqrt(r.reduce((s, v) => s + v, 0)))
        if (degree.some((v) => v === 0)) throw new WhiteboxError('value', 'spectralEmbedding: a row has no neighbours')
        const u = toArray(graphSpectralEmbedding(off, k, { dropFirst: true })) as number[][]
        const embedding = u.map((r, i) => r.map((v) => v / (degree[i] ?? 1)))
        const graph = off.flatMap((r, i) =>
          r.flatMap((v, j): [number, number, number][] => (j > i && v > 0 ? [[i, j, v]] : [])),
        )
        return {
          params: { embedding, objective: 0, nIter: 0, nFeaturesIn: m.d },
          exposition: [embeddingSteps({ method: 'spectralEmbedding', positions: [embedding], iterations: [0], graph })],
        }
      },
      methods: (p) => ({ nFeaturesIn: p.nFeaturesIn }),
    },
    hyper,
  )
}

/** The spectral embedding as a class. */
export class SpectralEmbedding extends EstimatorClass<SpectralEmbeddingHyper, FittedEmbedding> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link spectralEmbedding}.
   */
  constructor(hyper: SpectralEmbeddingHyper = {}) {
    super(spectralEmbedding(hyper))
  }
}

/** Hyperparameters of diffusion maps. */
export type DiffusionMapHyper = {
  readonly nComponents?: number
  /** The kernel width: k(x, y) = exp(-|x - y|^2 / epsilon) (default the median squared distance). */
  readonly epsilon?: number
  /** Diffusion time t (default 1). */
  readonly time?: number
  /** Density normalisation exponent (0 graph Laplacian, 1 Laplace-Beltrami; default 1). */
  readonly alpha?: number
}

/**
 * Diffusion maps (Coifman and Lafon 2006): a Gaussian kernel, alpha-normalised for density, made a Markov matrix;
 * the embedding is its leading non-trivial right eigenvectors scaled by eigenvalue^t. Its exposition is an
 * `embeddingSteps` record.
 *
 * @param hyper - Dimension, kernel width, diffusion time and density normalisation.
 * @returns The estimator.
 */
export function diffusionMap(hyper: DiffusionMapHyper = {}): Estimator<DiffusionMapHyper, FittedEmbedding> {
  return defineExplainedEstimator<DiffusionMapHyper, EmbeddingParams, { readonly nFeaturesIn: number }>(
    {
      name: 'diffusion-map',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 3, { default: h.nComponents ?? 2, label: 'dimensions' }),
          time: int(1, 20, { default: h.time ?? 1, label: 'diffusion time' }),
          alpha: real(0, 1, { default: h.alpha ?? 1 }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'diffusionMap', { minSamples: 3 })
        const k = h.nComponents ?? 2
        const sq = toArray(pairwiseKernels(data.x, undefined, 'linear')) as number[][]
        const norms = sq.map((r, i) => r[i] ?? 0)
        const d2 = sq.map((r, i) => r.map((v, j) => Math.max(0, (norms[i] ?? 0) + (norms[j] ?? 0) - 2 * v)))
        const sorted = d2.flatMap((r, i) => r.filter((_, j) => j > i)).sort((a, b) => a - b)
        const eps = h.epsilon ?? sorted[Math.floor(sorted.length / 2)] ?? 1
        const kern = d2.map((r) => r.map((v) => Math.exp(-v / eps)))
        const alpha = h.alpha ?? 1
        const q = kern.map((r) => r.reduce((s, v) => s + v, 0) ** alpha)
        const ka = kern.map((r, i) => r.map((v, j) => v / ((q[i] ?? 1) * (q[j] ?? 1))))
        const deg = ka.map((r) => r.reduce((s, v) => s + v, 0))
        const sym = ka.map((r, i) => r.map((v, j) => v / Math.sqrt((deg[i] ?? 1) * (deg[j] ?? 1))))
        const { eigenvalues, eigenvectors } = linalg.eigh(sym)
        const values = toFlat(eigenvalues)
        const vectors = toArray(eigenvectors) as number[][]
        const top = Array.from({ length: k }, (_, c) => m.n - 2 - c)
        const t = h.time ?? 1
        const embedding = vectors.map((r, i) =>
          top.map((c) => ((r[c] ?? 0) / Math.sqrt(deg[i] ?? 1)) * (values[c] ?? 0) ** t),
        )
        return {
          params: { embedding, objective: 0, nIter: 0, nFeaturesIn: m.d },
          exposition: [embeddingSteps({ method: 'diffusionMap', positions: [embedding], iterations: [0] })],
        }
      },
      methods: (p) => ({ nFeaturesIn: p.nFeaturesIn }),
    },
    hyper,
  )
}
