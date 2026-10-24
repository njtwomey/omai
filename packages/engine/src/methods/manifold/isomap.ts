import { WhiteboxError } from '../../lib/base/index.ts'
import { embeddingSteps } from '../../lib/exposition/index.ts'
import { floydWarshall, graph as makeGraph } from '../../lib/graph/index.ts'
import { defineExplainedEstimator, type Estimator, EstimatorClass, matrixView } from '../../lib/learn/index.ts'
import { kNearest } from '../../lib/numerics/spatial/index.ts'
import { dict, int } from '../../lib/space/index.ts'
import { type TensorLike, toArray, toFlat } from '../../lib/tensor/index.ts'
import { classicalMds, type EmbeddingParams, type FittedEmbedding } from './mds.ts'

/** A k-nearest-neighbour graph as weighted edges [i, j, distance] (each row to its k nearest, itself excluded). */
export type NeighbourEdges = [number, number, number][]

/**
 * The k-nearest-neighbour edges of points (brute force, `numerics/spatial`).
 *
 * @param x - Points `[n, d]`.
 * @param k - Neighbours per row.
 * @returns The directed edges i -> j with their distances, and the neighbour indices per row.
 */
export function neighbourEdges(
  x: TensorLike,
  k: number,
): { readonly edges: NeighbourEdges; readonly indices: number[][] } {
  const n = matrixView(x, 'neighbourEdges').n
  if (!(Number.isInteger(k) && k >= 1 && k < n))
    throw new WhiteboxError('value', `neighbours: k must be in 1..${n - 1}`)
  const nb = kNearest(x, undefined, k, { excludeSelf: true })
  const idx = Array.from(nb.indices.data)
  const dist = toFlat(nb.distances)
  const indices = Array.from({ length: n }, (_, i) => idx.slice(i * k, (i + 1) * k))
  const edges: NeighbourEdges = []
  for (let i = 0; i < n; i++) for (let a = 0; a < k; a++) edges.push([i, idx[i * k + a] ?? 0, dist[i * k + a] ?? 0])
  return { edges, indices }
}

/**
 * Geodesic distances along the symmetrised k-nearest-neighbour graph (Floyd-Warshall from `graph`).
 *
 * @param x - Points `[n, d]`.
 * @param k - Neighbours per row.
 * @returns The distances `[n][n]` and the graph's edges.
 * @throws WhiteboxError `value` when the graph is disconnected.
 */
export function geodesicDistances(
  x: TensorLike,
  k: number,
): { readonly distances: number[][]; readonly edges: NeighbourEdges } {
  const n = matrixView(x, 'geodesicDistances').n
  const { edges } = neighbourEdges(x, k)
  const g = makeGraph(edges, { nodes: n, duplicates: 'min' })
  const distances = toArray(floydWarshall(g).distance) as number[][]
  if (distances.some((r) => r.some((v) => !Number.isFinite(v))))
    throw new WhiteboxError('value', 'isomap: the neighbour graph is disconnected; raise nNeighbors')
  return { distances, edges }
}

/** Hyperparameters of Isomap. */
export type IsomapHyper = { readonly nNeighbors?: number; readonly nComponents?: number }

/**
 * Isomap (Tenenbaum et al. 2000; sklearn.manifold.Isomap): classical MDS of the geodesic distances along the
 * k-nearest-neighbour graph. Its exposition is an `embeddingSteps` record with the neighbour graph.
 *
 * @param hyper - Neighbours and dimension.
 * @returns The estimator.
 */
export function isomap(hyper: IsomapHyper = {}): Estimator<IsomapHyper, FittedEmbedding> {
  return defineExplainedEstimator<IsomapHyper, EmbeddingParams, { readonly nFeaturesIn: number }>(
    {
      name: 'isomap',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nNeighbors: int(2, 50, { default: h.nNeighbors ?? 5, label: 'neighbours' }),
          nComponents: int(1, 3, { default: h.nComponents ?? 2, label: 'dimensions' }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'isomap', { minSamples: 3 })
        const { distances, edges } = geodesicDistances(data.x, h.nNeighbors ?? 5)
        const { embedding, eigenvalues } = classicalMds(distances, h.nComponents ?? 2)
        return {
          params: { embedding, objective: eigenvalues.reduce((a, v) => a + v, 0), nIter: 0, nFeaturesIn: m.d },
          exposition: [embeddingSteps({ method: 'isomap', positions: [embedding], iterations: [0], graph: edges })],
        }
      },
      methods: (p) => ({ nFeaturesIn: p.nFeaturesIn }),
    },
    hyper,
  )
}

/** Isomap as a class. */
export class Isomap extends EstimatorClass<IsomapHyper, FittedEmbedding> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link isomap}.
   */
  constructor(hyper: IsomapHyper = {}) {
    super(isomap(hyper))
  }
}
