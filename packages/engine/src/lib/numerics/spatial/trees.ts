/**
 * Spatial trees for exact neighbour queries, as scikit-learn's: KD trees with axis-aligned median splits, ball trees
 * for any true metric (haversine included), and vantage-point trees that split by distance alone. Trees expose their
 * nodes and queries log what they pruned, for figures and comparisons.
 */

import { MdmlError } from '../../base/index.ts'
import { concreteOrThrow, fromStorage, promoteFloat, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { type Metric, type MetricOptions, metricFunction, type PointDistance, TRUE_METRICS } from '../geometry/index.ts'

/** What a query did: for figures of pruning and for comparing trees. */
export interface QueryLog {
  /** Node ids in the order they were entered. */
  readonly visited: readonly number[]
  /** Node ids skipped because their bound could not beat the current answer. */
  readonly pruned: readonly number[]
  readonly distanceEvaluations: number
}

/** A node of a spatial tree, for figures. */
export interface TreeNode {
  readonly id: number
  readonly depth: number
  /** Indices (into the data) of the points below this node. */
  readonly points: readonly number[]
  readonly children: readonly number[]
  /** KD trees: the bounding box; ball trees: centre and radius; VP trees: vantage point and median radius. */
  readonly bounds: {
    readonly lo?: readonly number[]
    readonly hi?: readonly number[]
    readonly centre?: readonly number[]
    readonly radius: number
    readonly vantage?: number
  }
}

/** A spatial index answering exact neighbour queries. */
export interface SpatialTree {
  readonly kind: 'kd' | 'ball' | 'vp'
  readonly size: number
  readonly dim: number
  readonly nodes: readonly TreeNode[]
  /** The k nearest data points of each query (nearest first), optionally with query logs. */
  readonly query: (
    queries: TensorLike,
    k: number,
    options?: { readonly log?: boolean },
  ) => { readonly indices: Tensor; readonly distances: Tensor; readonly logs?: readonly QueryLog[] }
  /** All data points within r of each query (sorted by distance). */
  readonly queryRadius: (
    queries: TensorLike,
    r: number,
    options?: { readonly log?: boolean },
  ) => { readonly indices: number[][]; readonly distances: number[][]; readonly logs?: readonly QueryLog[] }
  /** The number of data points within r of each query. */
  readonly countNeighbours: (queries: TensorLike, r: number) => number[]
}

/** Options of the tree builders. */
export interface TreeOptions extends MetricOptions {
  readonly metric?: Metric
  /** Maximum points in a leaf (default 40, as sklearn). */
  readonly leafSize?: number
}

interface Internal {
  start: number
  end: number
  left: number
  right: number
  depth: number
  lo: Float64Array
  hi: Float64Array
  centre: Float64Array
  radius: number
  vantage: number
}

function flatPoints(x: TensorLike, name: string): { data: number[]; n: number; d: number } {
  const t = concreteOrThrow(promoteFloat(x)[0] as Tensor, name)
  if (t.shape.length !== 2) throw new MdmlError('shape', `${name}: points must be [n, d]`)
  return { data: toFlat(t), n: t.shape[0] ?? 0, d: t.shape[1] ?? 0 }
}

/** Partition perm[start..end) around the median along dimension dim (quickselect). */
function medianSplit(perm: Int32Array, start: number, end: number, key: (i: number) => number): number {
  const mid = (start + end) >> 1
  let lo = start
  let hi = end - 1
  while (lo < hi) {
    const pivot = key(perm[(lo + hi) >> 1] ?? 0)
    let i = lo
    let j = hi
    while (i <= j) {
      while (key(perm[i] ?? 0) < pivot) i++
      while (key(perm[j] ?? 0) > pivot) j--
      if (i <= j) {
        const t = perm[i] ?? 0
        perm[i] = perm[j] ?? 0
        perm[j] = t
        i++
        j--
      }
    }
    if (mid <= j) hi = j
    else if (mid >= i) lo = i
    else break
  }
  return mid
}

function buildTree(kind: 'kd' | 'ball' | 'vp', x: TensorLike, options: TreeOptions): SpatialTree {
  const { data, n, d } = flatPoints(x, `${kind}Tree`)
  const metric = options.metric ?? 'euclidean'
  if (kind === 'kd' && !['euclidean', 'cityblock', 'chebyshev', 'minkowski'].includes(metric)) {
    throw new MdmlError('value', 'kdTree: only Minkowski-family metrics (euclidean, cityblock, chebyshev, minkowski)')
  }
  if (kind !== 'kd' && !TRUE_METRICS.has(metric))
    throw new MdmlError('value', `${kind}Tree: ${metric} is not a metric (no triangle inequality)`)
  const p =
    metric === 'cityblock'
      ? 1
      : metric === 'chebyshev'
        ? Number.POSITIVE_INFINITY
        : metric === 'euclidean'
          ? 2
          : (options.p ?? 2)
  const dist: PointDistance = metricFunction(metric, d, options)
  const leafSize = Math.max(1, options.leafSize ?? 40)
  const perm = Int32Array.from({ length: n }, (_, i) => i)
  const nodes: Internal[] = []
  const coord = (i: number, k: number): number => data[i * d + k] ?? 0
  const build = (start: number, end: number, depth: number): number => {
    const id = nodes.length
    const node: Internal = {
      start,
      end,
      left: -1,
      right: -1,
      depth,
      lo: new Float64Array(d),
      hi: new Float64Array(d),
      centre: new Float64Array(d),
      radius: 0,
      vantage: -1,
    }
    nodes.push(node)
    node.lo.fill(Number.POSITIVE_INFINITY)
    node.hi.fill(Number.NEGATIVE_INFINITY)
    for (let s = start; s < end; s++) {
      const i = perm[s] ?? 0
      for (let k = 0; k < d; k++) {
        const v = coord(i, k)
        if (v < (node.lo[k] ?? 0)) node.lo[k] = v
        if (v > (node.hi[k] ?? 0)) node.hi[k] = v
        node.centre[k] = (node.centre[k] ?? 0) + v / (end - start)
      }
    }
    if (kind === 'ball') {
      for (let s = start; s < end; s++)
        node.radius = Math.max(node.radius, dist(node.centre, 0, data, (perm[s] ?? 0) * d))
    }
    if (end - start <= leafSize) return id
    let mid: number
    if (kind === 'vp') {
      // Vantage point: the point farthest from the first one (a cheap spread heuristic); split at the median distance.
      const first = perm[start] ?? 0
      let far = start
      let best = -1
      for (let s = start; s < end; s++) {
        const dv = dist(data, first * d, data, (perm[s] ?? 0) * d)
        if (dv > best) {
          best = dv
          far = s
        }
      }
      const t = perm[start] ?? 0
      perm[start] = perm[far] ?? 0
      perm[far] = t
      const vp = perm[start] ?? 0
      node.vantage = vp
      const key = (i: number): number => dist(data, vp * d, data, i * d)
      mid = medianSplit(perm, start + 1, end, key)
      node.radius = key(perm[mid] ?? 0)
      // Points at the median radius go inside: [start + 1, mid] inside, (mid, end) outside.
      node.left = build(start + 1, mid + 1, depth + 1)
      node.right = mid + 1 < end ? build(mid + 1, end, depth + 1) : -1
      return id
    }
    // KD and ball trees split on the dimension of greatest spread at the median.
    let split = 0
    for (let k = 1; k < d; k++)
      if ((node.hi[k] ?? 0) - (node.lo[k] ?? 0) > (node.hi[split] ?? 0) - (node.lo[split] ?? 0)) split = k
    mid = medianSplit(perm, start, end, (i) => coord(i, split))
    node.left = build(start, mid, depth + 1)
    node.right = build(mid, end, depth + 1)
    return id
  }
  if (n > 0) build(0, n, 0)

  /** A lower bound on the distance from q (at offset iq of qs) to any point under the node. */
  const bound = (node: Internal, qs: ArrayLike<number>, iq: number, dq?: number): number => {
    if (kind === 'kd') {
      let s = 0
      for (let k = 0; k < d; k++) {
        const v = qs[iq + k] ?? 0
        const gap = v < (node.lo[k] ?? 0) ? (node.lo[k] ?? 0) - v : v > (node.hi[k] ?? 0) ? v - (node.hi[k] ?? 0) : 0
        s = p === Number.POSITIVE_INFINITY ? Math.max(s, gap) : s + gap ** p
      }
      return p === Number.POSITIVE_INFINITY ? s : s ** (1 / p)
    }
    if (kind === 'ball') return Math.max(0, dist(qs, iq, node.centre, 0) - node.radius)
    return dq ?? 0
  }

  const search = (
    shouldPrune: (lower: number) => boolean,
    onPoint: (index: number, distance: number) => void,
    log: boolean,
  ): ((qs: ArrayLike<number>, iq: number) => QueryLog | undefined) => {
    return (qs, iq) => {
      const visited: number[] = []
      const pruned: number[] = []
      let evaluations = 0
      const recurse = (id: number, lower: number): void => {
        const node = nodes[id] as Internal
        if (shouldPrune(lower)) {
          if (log) pruned.push(id)
          return
        }
        if (log) visited.push(id)
        if (node.left < 0 && node.right < 0) {
          for (let s = node.start; s < node.end; s++) {
            const i = perm[s] ?? 0
            evaluations++
            onPoint(i, dist(qs, iq, data, i * d))
          }
          return
        }
        if (kind === 'vp') {
          // The vantage point itself, then the two shells, nearer first.
          const v = node.vantage
          evaluations++
          const dv = dist(qs, iq, data, v * d)
          onPoint(v, dv)
          const inside = Math.max(0, dv - node.radius)
          const outside = Math.max(0, node.radius - dv)
          const order: [number, number][] =
            dv <= node.radius
              ? [
                  [node.left, inside],
                  [node.right, outside],
                ]
              : [
                  [node.right, outside],
                  [node.left, inside],
                ]
          for (const [child, lb] of order) if (child >= 0) recurse(child, Math.max(lower, lb))
          return
        }
        const children = [node.left, node.right]
          .filter((c) => c >= 0)
          .map((c) => [c, bound(nodes[c] as Internal, qs, iq)] as const)
        children.sort((a, b) => a[1] - b[1])
        for (const [child, lb] of children) recurse(child, lb)
      }
      if (nodes.length > 0) recurse(0, kind === 'vp' ? 0 : bound(nodes[0] as Internal, qs, iq))
      return log ? { visited, pruned, distanceEvaluations: evaluations } : undefined
    }
  }

  const query: SpatialTree['query'] = (queries, k, options = {}) => {
    const qs = flatPoints(queries, 'query')
    if (qs.d !== d) throw new MdmlError('shape', 'query: dimension mismatch')
    if (k < 1 || k > n) throw new MdmlError('value', `query: k must be in [1, ${n}]`)
    const indices = new Int32Array(qs.n * k)
    const distances = new Float64Array(qs.n * k)
    const logs: QueryLog[] = []
    for (let r = 0; r < qs.n; r++) {
      // The current best k, sorted by (distance, index).
      const best: [number, number][] = []
      const consider = (index: number, distance: number): void => {
        if (
          best.length === k &&
          (distance > (best[k - 1]?.[0] ?? 0) || (distance === best[k - 1]?.[0] && index > (best[k - 1]?.[1] ?? 0)))
        )
          return
        let pos = best.length
        while (
          pos > 0 &&
          ((best[pos - 1]?.[0] ?? 0) > distance ||
            ((best[pos - 1]?.[0] ?? 0) === distance && (best[pos - 1]?.[1] ?? 0) > index))
        )
          pos--
        best.splice(pos, 0, [distance, index])
        if (best.length > k) best.pop()
      }
      const visit = search(
        (lower) => best.length === k && lower > (best[k - 1]?.[0] ?? 0),
        consider,
        options.log === true,
      )
      const l = visit(qs.data, r * d)
      if (l !== undefined) logs.push(l)
      best.forEach(([dv, i], j) => {
        indices[r * k + j] = i
        distances[r * k + j] = dv
      })
    }
    return {
      indices: fromStorage(indices, [qs.n, k], 'int32'),
      distances: fromStorage(distances, [qs.n, k], 'float64'),
      ...(options.log === true ? { logs } : {}),
    }
  }

  const queryRadius: SpatialTree['queryRadius'] = (queries, r, options = {}) => {
    const qs = flatPoints(queries, 'queryRadius')
    const indices: number[][] = []
    const distances: number[][] = []
    const logs: QueryLog[] = []
    for (let row = 0; row < qs.n; row++) {
      const hits: [number, number][] = []
      const visit = search(
        (lower) => lower > r,
        (i, dv) => {
          if (dv <= r) hits.push([dv, i])
        },
        options.log === true,
      )
      const l = visit(qs.data, row * d)
      if (l !== undefined) logs.push(l)
      hits.sort((a, b) => a[0] - b[0] || a[1] - b[1])
      indices.push(hits.map((h) => h[1]))
      distances.push(hits.map((h) => h[0]))
    }
    return { indices, distances, ...(options.log === true ? { logs } : {}) }
  }

  const exported: TreeNode[] = nodes.map((node, id) => ({
    id,
    depth: node.depth,
    points: Array.from(perm.subarray(node.start, node.end)),
    children: [node.left, node.right].filter((c) => c >= 0),
    bounds:
      kind === 'kd'
        ? { lo: Array.from(node.lo), hi: Array.from(node.hi), radius: 0 }
        : kind === 'ball'
          ? { centre: Array.from(node.centre), radius: node.radius }
          : { vantage: node.vantage, radius: node.radius },
  }))
  return {
    kind,
    size: n,
    dim: d,
    nodes: exported,
    query,
    queryRadius,
    countNeighbours: (queries, r) => queryRadius(queries, r).indices.map((h) => h.length),
  }
}

/**
 * A KD tree (sklearn.neighbors.KDTree): axis-aligned splits at the median of the widest dimension; nodes keep bounding
 * boxes, whose distance to a query bounds the points below. Exact queries; Minkowski metrics.
 *
 * @param data - Points `[n, d]`.
 * @param options - Metric (euclidean, cityblock, chebyshev or minkowski with p) and leaf size.
 * @returns The tree.
 * @concept spatial-trees
 * @example
 * ```ts
 * import { kdTree } from 'mdml-engine/numerics/spatial'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(kdTree([[0, 0], [1, 0], [5, 5]], { leafSize: 1 }).query([[0.9, 0.1]], 1).indices) // => [[1]]
 * ```
 */
export function kdTree(data: TensorLike, options: TreeOptions = {}): SpatialTree {
  return buildTree('kd', data, options)
}

/**
 * A ball tree (sklearn.neighbors.BallTree): nodes are balls around the centroid of their points; works for any metric
 * with the triangle inequality (including haversine on [latitude, longitude] in radians).
 *
 * @param data - Points `[n, d]`.
 * @param options - Metric, its parameters and leaf size.
 * @returns The tree.
 * @example
 * ```ts
 * import { ballTree } from 'mdml-engine/numerics/spatial'
 * ballTree([[0, 0], [1, 0], [5, 5]], { metric: 'cityblock', leafSize: 1 }).countNeighbours([[0, 0]], 1) // => [2]
 * ```
 */
export function ballTree(data: TensorLike, options: TreeOptions = {}): SpatialTree {
  return buildTree('ball', data, options)
}

/**
 * A vantage-point tree (Yianilos): each node splits its points by their distance to a vantage point at the median, so
 * only distances are used (any metric with the triangle inequality).
 *
 * @param data - Points `[n, d]`.
 * @param options - Metric, its parameters and leaf size.
 * @returns The tree.
 * @example
 * ```ts
 * import { vpTree } from 'mdml-engine/numerics/spatial'
 * vpTree([[0, 0], [1, 0], [5, 5], [6, 5]], { leafSize: 1 }).queryRadius([[5.5, 5]], 1).indices // => [[2, 3]]
 * ```
 */
export function vpTree(data: TensorLike, options: TreeOptions = {}): SpatialTree {
  return buildTree('vp', data, options)
}
