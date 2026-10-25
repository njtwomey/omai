/**
 * Exact nearest-neighbour search by computing every distance (scikit-learn's brute algorithm): k-nearest and radius
 * neighbours, and the k-nearest-neighbour and radius graphs as sparse matrices. It is the reference the trees and
 * approximate indexes are checked against.
 */

import { WhiteboxError } from '../../base/index.ts'
import {
  fromStorage,
  primalOf,
  reshape,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { cdist, type Metric, type MetricOptions } from '../geometry/index.ts'
import { type CsrMatrix, csr } from '../sparse/index.ts'

/** Neighbours of each query point: indices into the data and distances, nearest first. */
export interface Neighbours {
  /** `[q, k]` int32 indices. */
  readonly indices: Tensor
  /** `[q, k]` distances (differentiable in data and queries for continuous metrics). */
  readonly distances: Value
}

/** Options of brute-force neighbour searches. */
export interface BruteOptions extends MetricOptions {
  readonly metric?: Metric
  /** Leave each point out of its own neighbours when the queries are the data (default false). */
  readonly excludeSelf?: boolean
}

/**
 * The k nearest neighbours by computing every distance (sklearn's brute algorithm): exact, O(n q d), the reference for
 * the trees and approximate indexes. Ties are broken by index.
 *
 * @param data - Points `[n, d]`.
 * @param queries - Query points `[q, d]` (default the data).
 * @param k - Number of neighbours.
 * @param options - Metric, its parameters, and whether to exclude self-matches.
 * @returns Indices and distances.
 * @concept nearest-neighbours
 * @example
 * ```ts
 * import { kNearest } from 'whitebox-engine/numerics/spatial'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(kNearest([[0, 0], [1, 0], [5, 5]], [[0.9, 0.1]], 2).indices) // => [[1, 0]]
 * ```
 */
export function kNearest(
  data: TensorLike,
  queries: TensorLike | undefined,
  k: number,
  options: BruteOptions = {},
): Neighbours {
  const { metric = 'euclidean', excludeSelf = false, ...metricOptions } = options
  const d = cdist(queries ?? data, data, metric, metricOptions)
  const [q = 0, n = 0] = d.shape
  const values = toFlat(primalOf(d))
  if (k > n - (excludeSelf ? 1 : 0))
    throw new WhiteboxError('value', `kNearest: k = ${k} exceeds the number of candidates`)
  const indices = new Int32Array(q * k)
  const flat = new Int32Array(q * k)
  for (let r = 0; r < q; r++) {
    const order = Array.from({ length: n }, (_, j) => j)
      .filter((j) => !(excludeSelf && j === r))
      .sort((a, b) => (values[r * n + a] ?? 0) - (values[r * n + b] ?? 0) || a - b)
    for (let j = 0; j < k; j++) {
      indices[r * k + j] = order[j] ?? 0
      flat[r * k + j] = r * n + (order[j] ?? 0)
    }
  }
  const distances = reshape(take(reshape(d, [-1]), fromStorage(flat, [q * k], 'int32'), 0), [q, k])
  return { indices: fromStorage(indices, [q, k], 'int32'), distances }
}

/**
 * All data points within distance r of each query (sklearn's radius_neighbors), sorted by distance.
 *
 * @param data - Points `[n, d]`.
 * @param queries - Query points `[q, d]` (default the data).
 * @param r - The radius (inclusive).
 * @param options - Metric, parameters and self-exclusion.
 * @returns Per query: indices and distances.
 * @example
 * ```ts
 * import { radiusNeighbours } from 'whitebox-engine/numerics/spatial'
 * radiusNeighbours([[0, 0], [1, 0], [5, 5]], [[0, 0]], 1.5)[0].indices // => [0, 1]
 * ```
 */
export function radiusNeighbours(
  data: TensorLike,
  queries: TensorLike | undefined,
  r: number,
  options: BruteOptions = {},
): { readonly indices: number[]; readonly distances: number[] }[] {
  const { metric = 'euclidean', excludeSelf = false, ...metricOptions } = options
  const d = cdist(queries ?? data, data, metric, metricOptions)
  const [q = 0, n = 0] = d.shape
  const values = toFlat(primalOf(d))
  return Array.from({ length: q }, (_, row) => {
    const hits = Array.from({ length: n }, (_, j) => j)
      .filter((j) => (values[row * n + j] ?? Number.POSITIVE_INFINITY) <= r && !(excludeSelf && j === row))
      .sort((a, b) => (values[row * n + a] ?? 0) - (values[row * n + b] ?? 0) || a - b)
    return { indices: hits, distances: hits.map((j) => values[row * n + j] ?? 0) }
  })
}

/**
 * The k-nearest-neighbour graph of a point set (sklearn.neighbors.kneighbors_graph): row i has k entries, 1 for
 * connectivity or the distance, at its neighbours (excluding i itself unless includeSelf).
 *
 * @param data - Points `[n, d]`.
 * @param k - Neighbours per point.
 * @param options - Mode (connectivity or distance), includeSelf (default false) and metric.
 * @returns A sparse `[n, n]` matrix.
 * @example
 * ```ts
 * import { kNeighboursGraph } from 'whitebox-engine/numerics/spatial'
 * import { toDense } from 'whitebox-engine/numerics/sparse'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(toDense(kNeighboursGraph([[0], [1], [3]], 1))) // => [[0, 1, 0], [1, 0, 0], [0, 1, 0]]
 * ```
 */
export function kNeighboursGraph(
  data: TensorLike,
  k: number,
  options: { readonly mode?: 'connectivity' | 'distance'; readonly includeSelf?: boolean } & BruteOptions = {},
): CsrMatrix {
  const { mode = 'connectivity', includeSelf = false, ...rest } = options
  const nb = kNearest(data, undefined, k, { ...rest, excludeSelf: !includeSelf })
  const n = nb.indices.shape[0] ?? 0
  const idx = Array.from(nb.indices.data)
  const rows = idx.map((_, i) => Math.floor(i / k))
  // Columns within each row sorted, as scipy's CSR.
  const order = idx.map((_, i) => i).sort((a, b) => (rows[a] ?? 0) - (rows[b] ?? 0) || (idx[a] ?? 0) - (idx[b] ?? 0))
  const dist = toFlat(primalOf(nb.distances))
  const indptr = Array.from({ length: n + 1 }, (_, i) => i * k)
  return csr({
    indptr,
    indices: order.map((i) => idx[i] ?? 0),
    values: fromStorage(
      Float64Array.from(order, (i) => (mode === 'distance' ? (dist[i] ?? 0) : 1)),
      [order.length],
      'float64',
    ),
    shape: [n, n],
  })
}

/**
 * The radius-neighbour graph (sklearn.neighbors.radius_neighbors_graph).
 *
 * @param data - Points `[n, d]`.
 * @param r - The radius.
 * @param options - Mode, includeSelf and metric.
 * @returns A sparse `[n, n]` matrix.
 * @example
 * ```ts
 * import { radiusNeighboursGraph } from 'whitebox-engine/numerics/spatial'
 * import { nnz } from 'whitebox-engine/numerics/sparse'
 * nnz(radiusNeighboursGraph([[0], [1], [3]], 1.5)) // => 2
 * ```
 */
export function radiusNeighboursGraph(
  data: TensorLike,
  r: number,
  options: { readonly mode?: 'connectivity' | 'distance'; readonly includeSelf?: boolean } & BruteOptions = {},
): CsrMatrix {
  const { mode = 'connectivity', includeSelf = false, ...rest } = options
  const hits = radiusNeighbours(data, undefined, r, { ...rest, excludeSelf: !includeSelf })
  const indptr = [0]
  const indices: number[] = []
  const values: number[] = []
  for (const h of hits) {
    const order = h.indices.map((_, i) => i).sort((a, b) => (h.indices[a] ?? 0) - (h.indices[b] ?? 0))
    for (const i of order) {
      indices.push(h.indices[i] ?? 0)
      values.push(mode === 'distance' ? (h.distances[i] ?? 0) : 1)
    }
    indptr.push(indices.length)
  }
  return csr({
    indptr,
    indices,
    values: fromStorage(Float64Array.from(values), [values.length], 'float64'),
    shape: [hits.length, hits.length],
  })
}
