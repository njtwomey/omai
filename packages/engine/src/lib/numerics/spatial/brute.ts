/**
 * Exact nearest-neighbour search by computing every distance (scikit-learn's brute algorithm): k-nearest and radius
 * neighbours, and the k-nearest-neighbour and radius graphs as sparse matrices. It is the reference the trees and
 * approximate indexes are checked against.
 */

import { OmaiError } from '../../base/index.ts'
import {
  fromStorage,
  primalOf,
  promoteFloat,
  reshape,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { assertFinite } from '../_vector.ts'
import { cdist, type DistanceMetric, type MetricOptions } from '../geometry/index.ts'
import { type CsrMatrix, csr } from '../sparse/index.ts'

/** The NaN rule at the boundary of a query: a point with a non-finite coordinate has no neighbours worth ordering. */
function requireFinite(name: string, data: TensorLike, queries: TensorLike | undefined): void {
  assertFinite(name, toFlat(primalOf(promoteFloat(data)[0] as Value)), 'data')
  if (queries !== undefined) assertFinite(name, toFlat(primalOf(promoteFloat(queries)[0] as Value)), 'queries')
}

/** Neighbours of each query point: indices into the data and distances, nearest first. */
export interface Neighbours {
  /** `[q, k]` int32 indices. */
  readonly indices: Tensor
  /** `[q, k]` distances (differentiable in data and queries for continuous metrics). */
  readonly distances: Value
}

/** Options of brute-force neighbour searches. */
export interface BruteOptions extends MetricOptions {
  readonly metric?: DistanceMetric
  /** Leave each point out of its own neighbours when the queries are the data (default false). */
  readonly excludeSelf?: boolean
}

/**
 * The `k` smallest of `values[offset .. offset + n)` (skipping the index `skip`), by a bounded max-heap: O(n log k)
 * instead of sorting every candidate. They are left in `heapValue` and `heapIndex` sorted by value and then by index
 * (ties resolved as a stable sort of the whole row would), with NaN last. Returns how many were selected.
 */
function selectSmallest(
  values: ArrayLike<number>,
  offset: number,
  n: number,
  k: number,
  skip: number,
  heapValue: Float64Array,
  heapIndex: Int32Array,
): number {
  // The heap root is the worst kept candidate: the largest value, the largest index among equal values. Candidates come
  // in ascending index, so one that ties the root is never better, and a strict comparison is enough.
  let size = 0
  const worse = (va: number, ia: number, vb: number, ib: number): boolean => va > vb || (va === vb && ia > ib)
  const siftDown = (start: number): void => {
    let i = start
    for (;;) {
      const left = 2 * i + 1
      if (left >= size) return
      const right = left + 1
      const child =
        right < size && worse(heapValue[right] ?? 0, heapIndex[right] ?? 0, heapValue[left] ?? 0, heapIndex[left] ?? 0)
          ? right
          : left
      if (!worse(heapValue[child] ?? 0, heapIndex[child] ?? 0, heapValue[i] ?? 0, heapIndex[i] ?? 0)) return
      swap(i, child)
      i = child
    }
  }
  const swap = (a: number, b: number): void => {
    const v = heapValue[a] ?? 0
    heapValue[a] = heapValue[b] ?? 0
    heapValue[b] = v
    const x = heapIndex[a] ?? 0
    heapIndex[a] = heapIndex[b] ?? 0
    heapIndex[b] = x
  }
  for (let j = 0; j < n && k > 0; j++) {
    if (j === skip) continue
    const raw = values[offset + j] ?? 0
    const v = Number.isNaN(raw) ? Number.POSITIVE_INFINITY : raw
    if (size < k) {
      // Sift the new entry up.
      let i = size++
      heapValue[i] = v
      heapIndex[i] = j
      while (i > 0) {
        const parent = (i - 1) >> 1
        if (!worse(heapValue[i] ?? 0, heapIndex[i] ?? 0, heapValue[parent] ?? 0, heapIndex[parent] ?? 0)) break
        swap(i, parent)
        i = parent
      }
    } else if (v < (heapValue[0] ?? 0)) {
      heapValue[0] = v
      heapIndex[0] = j
      siftDown(0)
    }
  }
  // Heap sort in place: repeatedly move the worst to the end, leaving ascending (value, index) order.
  const count = size
  while (size > 1) {
    swap(0, size - 1)
    size -= 1
    siftDown(0)
  }
  return count
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
 * @throws OmaiError with code `value` when the data or the queries have a NaN or infinite coordinate (a NaN row used to
 *   come back as the nearest point, with distance NaN), or when k exceeds the number of candidates.
 * @concept nearest-neighbours
 * @index function Exact k nearest neighbours by brute force.
 * @example
 * ```ts
 * import { kNearest } from 'omai-engine/numerics/spatial'
 * import { toArray } from 'omai-engine/tensor'
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
  requireFinite('kNearest', data, queries)
  const d = cdist(queries ?? data, data, metric, metricOptions)
  const [q = 0, n = 0] = d.shape
  const values = toFlat(primalOf(d))
  if (k > n - (excludeSelf ? 1 : 0)) throw new OmaiError('value', `kNearest: k = ${k} exceeds the number of candidates`)
  const indices = new Int32Array(q * k)
  const flat = new Int32Array(q * k)
  const heapValue = new Float64Array(Math.max(k, 0))
  const heapIndex = new Int32Array(Math.max(k, 0))
  for (let r = 0; r < q; r++) {
    const size = selectSmallest(values, r * n, n, k, excludeSelf ? r : -1, heapValue, heapIndex)
    for (let j = 0; j < size; j++) {
      indices[r * k + j] = heapIndex[j] ?? 0
      flat[r * k + j] = r * n + (heapIndex[j] ?? 0)
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
 * @throws OmaiError with code `value` when the data or the queries have a NaN or infinite coordinate.
 * @example
 * ```ts
 * import { radiusNeighbours } from 'omai-engine/numerics/spatial'
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
  requireFinite('radiusNeighbours', data, queries)
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
 * @throws OmaiError with code `value` when the data have a NaN or infinite coordinate.
 * @example
 * ```ts
 * import { kNeighboursGraph } from 'omai-engine/numerics/spatial'
 * import { toDense } from 'omai-engine/numerics/sparse'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(toDense(kNeighboursGraph([[0], [1], [3]], 1))) // => [[0, 1, 0], [1, 0, 0], [0, 1, 0]]
 * ```
 */
export function kNeighboursGraph(
  data: TensorLike,
  k: number,
  options: { readonly mode?: 'connectivity' | 'distance'; readonly includeSelf?: boolean } & BruteOptions = {},
): CsrMatrix {
  const { mode = 'connectivity', includeSelf = false, ...rest } = options
  requireFinite('kNeighboursGraph', data, undefined)
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
 * @throws OmaiError with code `value` when the data have a NaN or infinite coordinate.
 * @example
 * ```ts
 * import { radiusNeighboursGraph } from 'omai-engine/numerics/spatial'
 * import { nnz } from 'omai-engine/numerics/sparse'
 * nnz(radiusNeighboursGraph([[0], [1], [3]], 1.5)) // => 2
 * ```
 */
export function radiusNeighboursGraph(
  data: TensorLike,
  r: number,
  options: { readonly mode?: 'connectivity' | 'distance'; readonly includeSelf?: boolean } & BruteOptions = {},
): CsrMatrix {
  const { mode = 'connectivity', includeSelf = false, ...rest } = options
  requireFinite('radiusNeighboursGraph', data, undefined)
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
