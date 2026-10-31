/**
 * Isomap (Tenenbaum, de Silva and Langford 2000; scikit-learn's `Isomap`): a nonlinear embedding that preserves geodesic
 * distances, measured as shortest paths through the k-nearest-neighbour graph and then embedded by classical MDS, so a
 * curled-up manifold such as the Swiss roll unrolls. Also the neighbour-graph and geodesic-distance helpers other
 * embeddings reuse.
 */

import { MdmlError } from '../base/index.ts'
import { floydWarshall, graph as makeGraph } from '../graph/index.ts'
import { defineEstimator, type Estimator, type Fitted, matrixView } from '../learn/index.ts'
import { kNearest } from '../numerics/spatial/index.ts'
import { dict, int } from '../space/index.ts'
import { type TensorLike, toArray, toFlat } from '../tensor/index.ts'
import { type EmbeddingParams, type EmbeddingVerbs, embeddingVerbs, trainingRows } from './_support.ts'
import { classicalMds } from './mds.ts'

/** A k-nearest-neighbour graph as weighted edges [i, j, distance] (each row to its k nearest, itself excluded). */
export type NeighbourEdges = [number, number, number][]

/**
 * The k-nearest-neighbour edges of points (brute force, `numerics/spatial`).
 *
 * @param x - Points `[n, d]`.
 * @param k - Neighbours per row.
 * @returns The directed edges i -> j with their distances, and the neighbour indices per row.
 * @example
 * ```ts
 * import { neighbourEdges } from 'mdml-engine/manifold'
 * neighbourEdges([[0], [1], [5]], 1).indices // => [[1], [0], [1]]
 * ```
 */
export function neighbourEdges(
  x: TensorLike,
  k: number,
): { readonly edges: NeighbourEdges; readonly indices: number[][] } {
  const n = matrixView(x, 'neighbourEdges').n
  if (!(Number.isInteger(k) && k >= 1 && k < n)) throw new MdmlError('value', `neighbours: k must be in 1..${n - 1}`)
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
 * @returns The distances `[n, n]` and the graph's edges.
 * @throws MdmlError `value` when the graph is disconnected.
 * @example
 * ```ts
 * import { geodesicDistances } from 'mdml-engine/manifold'
 * geodesicDistances([[0], [1], [2], [3]], 1).distances[0] // => [0, 1, 2, 3]
 * ```
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
    throw new MdmlError('value', 'isomap: the neighbour graph is disconnected; raise nNeighbors')
  return { distances, edges }
}

/** Hyperparameters of {@link isomap}. */
export interface IsomapHyper {
  readonly nNeighbors: number
  readonly nComponents: number
}

/**
 * Isomap (sklearn.manifold.Isomap): classical MDS of the geodesic distances along the k-nearest-neighbour graph.
 *
 * @param hyper - Neighbours and dimension.
 * @returns The estimator.
 * @example
 * ```ts
 * import { isomap } from 'mdml-engine/manifold'
 * isomap({ nNeighbors: 2 }).fit({ x: [[0, 0], [1, 0], [2, 0], [3, 0]] }).params.embedding.length // => 4
 * ```
 */
export function isomap(
  hyper: Partial<IsomapHyper> = {},
): Estimator<IsomapHyper, Fitted<EmbeddingParams> & EmbeddingVerbs> {
  return defineEstimator<IsomapHyper, EmbeddingParams, EmbeddingVerbs>(
    {
      name: 'manifold/isomap',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nNeighbors: int(2, 50, { default: h.nNeighbors, label: 'neighbours' }),
          nComponents: int(1, 3, { default: h.nComponents, label: 'dimensions' }),
        }),
      fit: (data, h) => {
        matrixView(data.x, 'isomap', { minSamples: 3 })
        const { distances } = geodesicDistances(data.x, h.nNeighbors)
        const { embedding, eigenvalues } = classicalMds(distances, h.nComponents)
        return {
          embedding,
          data: trainingRows(data.x, 'isomap'),
          objective: eigenvalues.reduce((a, v) => a + v, 0),
        }
      },
      methods: (p) => embeddingVerbs(p, 'isomap'),
    },
    { nNeighbors: 5, nComponents: 2, ...hyper },
  )
}
