import { WhiteboxError } from '../base/index.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'
import { type Graph, type NodeId, nodeIndex } from './graph.ts'
import { IndexedQueue } from './structures.ts'

/** Single-source shortest paths: distances (Infinity when unreachable) and the predecessor of each node on a shortest
 * path (-1 for the source and unreachable nodes), as scipy.sparse.csgraph returns them. */
export interface ShortestPaths {
  readonly source: number
  readonly distance: Float64Array
  readonly predecessor: Int32Array
}

/** All-pairs shortest paths: `distance[i, j]` and `predecessor[i, j]` (the node before j on a path from i, -1 when
 * none), as scipy's floyd_warshall and johnson return them. */
export interface AllPairsShortestPaths {
  readonly distance: Tensor<'float64'>
  readonly predecessor: Tensor<'int32'>
}

/** Options shared by the shortest-path functions. */
export interface PathOptions {
  /** Use edge weights (default true); false counts hops (scipy's `unweighted`). */
  readonly weighted?: boolean
}

function weightOf(g: Graph, options: PathOptions): (k: number) => number {
  const w = g.out.weights
  return options.weighted === false ? () => 1 : (k) => w[k] ?? 0
}

/** Dijkstra from one source, skipping blocked edges and nodes; stops once `target` is settled. */
function dijkstraCore(
  g: Graph,
  s: number,
  weight: (k: number) => number,
  target = -1,
  blockedEdges?: ReadonlySet<number>,
  blockedNodes?: ReadonlySet<number>,
): ShortestPaths {
  const a = g.out
  const distance = new Float64Array(g.n).fill(Number.POSITIVE_INFINITY)
  const predecessor = new Int32Array(g.n).fill(-1)
  const done = new Uint8Array(g.n)
  const queue = new IndexedQueue(g.n)
  distance[s] = 0
  queue.push(s, 0)
  for (let u = queue.pop(); u !== undefined; u = queue.pop()) {
    done[u] = 1
    if (u === target) break
    const du = distance[u] ?? 0
    for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
      const v = a.indices[k] ?? 0
      if (done[v] === 1 || blockedEdges?.has(a.edges[k] ?? -1) === true || blockedNodes?.has(v) === true) continue
      const w = weight(k)
      if (w < 0) throw new WhiteboxError('domain', 'dijkstra: negative edge weights need bellmanFord or johnson')
      const alt = du + w
      if (alt < (distance[v] ?? Number.POSITIVE_INFINITY)) {
        distance[v] = alt
        predecessor[v] = u
        queue.push(v, alt)
      }
    }
  }
  return { source: s, distance, predecessor }
}

/**
 * Single-source shortest paths with non-negative weights by Dijkstra's algorithm (scipy.sparse.csgraph.dijkstra,
 * NetworkX single_source_dijkstra), with an indexed priority queue; ties settle in node order.
 *
 * @param g - The graph.
 * @param source - The source.
 * @param options - `weighted` (default true), and a `target` to stop at once it is settled.
 * @returns Distances and predecessors.
 * @throws WhiteboxError with code `domain` when a negative weight is reached.
 * @concept shortest-paths
 * @example
 * ```ts
 * import { dijkstra, graph } from 'whitebox-engine/graph'
 * const r = dijkstra(graph([[0, 1, 4], [0, 2, 1], [2, 1, 2], [1, 3, 5]]), 0)
 * Array.from(r.distance) // => [0, 3, 1, 8]
 * Array.from(r.predecessor) // => [-1, 2, 0, 1]
 * ```
 */
export function dijkstra(
  g: Graph,
  source: NodeId,
  options: PathOptions & { readonly target?: NodeId } = {},
): ShortestPaths {
  const target = options.target === undefined ? -1 : nodeIndex(g, options.target)
  return dijkstraCore(g, nodeIndex(g, source), weightOf(g, options), target)
}

/**
 * Single-source shortest paths allowing negative weights by the Bellman-Ford algorithm (scipy's bellman_ford): up to
 * n - 1 rounds relaxing every edge, stopping early when a round changes nothing. An undirected edge of negative
 * weight is a negative cycle.
 *
 * @param g - The graph.
 * @param source - The source.
 * @param options - `weighted` (default true).
 * @returns Distances and predecessors.
 * @throws WhiteboxError with code `value` when a negative cycle is reachable from the source.
 * @concept shortest-paths
 * @example
 * ```ts
 * import { bellmanFord, digraph } from 'whitebox-engine/graph'
 * Array.from(bellmanFord(digraph([[0, 1, 4], [0, 2, 5], [2, 1, -3]]), 0).distance) // => [0, 2, 5]
 * ```
 */
export function bellmanFord(g: Graph, source: NodeId, options: PathOptions = {}): ShortestPaths {
  const s = nodeIndex(g, source)
  const weight = weightOf(g, options)
  const a = g.out
  const distance = new Float64Array(g.n).fill(Number.POSITIVE_INFINITY)
  const predecessor = new Int32Array(g.n).fill(-1)
  distance[s] = 0
  const relax = (): boolean => {
    let changed = false
    for (let u = 0; u < g.n; u++) {
      const du = distance[u] ?? Number.POSITIVE_INFINITY
      if (du === Number.POSITIVE_INFINITY) continue
      for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
        const v = a.indices[k] ?? 0
        const alt = du + weight(k)
        if (alt < (distance[v] ?? Number.POSITIVE_INFINITY)) {
          distance[v] = alt
          predecessor[v] = u
          changed = true
        }
      }
    }
    return changed
  }
  for (let round = 0; round < g.n - 1; round++) if (!relax()) return { source: s, distance, predecessor }
  if (relax()) throw new WhiteboxError('value', 'bellmanFord: the graph has a negative cycle')
  return { source: s, distance, predecessor }
}

/** The result of {@link aStar}. */
export interface AStarResult {
  /** The path from source to target, or null when the target is unreachable. */
  readonly path: number[] | null
  readonly distance: number
  /** Nodes settled in order (the search's footprint). */
  readonly expanded: readonly number[]
}

/**
 * The shortest path between two nodes by A* search (NetworkX astar_path): best-first on g + h with an admissible,
 * consistent heuristic h (a lower bound on the remaining distance); h = 0 gives Dijkstra.
 *
 * @param g - The graph (non-negative weights).
 * @param source - The source.
 * @param target - The target.
 * @param options - The heuristic (default 0) and `weighted` (default true).
 * @returns The path, its length and the nodes expanded.
 * @throws WhiteboxError with code `domain` for a negative weight.
 * @concept shortest-paths
 * @example
 * ```ts
 * import { aStar, gridGraph } from 'whitebox-engine/graph'
 * const g = gridGraph(3, 3)
 * const manhattan = (v: number) => Math.abs(Math.floor(v / 3) - 2) + Math.abs((v % 3) - 2)
 * const r = aStar(g, 0, 8, { heuristic: manhattan })
 * [r.distance, r.path?.length] // => [4, 5]
 * ```
 */
export function aStar(
  g: Graph,
  source: NodeId,
  target: NodeId,
  options: PathOptions & { readonly heuristic?: (v: number) => number } = {},
): AStarResult {
  const s = nodeIndex(g, source)
  const t = nodeIndex(g, target)
  const h = options.heuristic ?? (() => 0)
  const weight = weightOf(g, options)
  const a = g.out
  const cost = new Float64Array(g.n).fill(Number.POSITIVE_INFINITY)
  const predecessor = new Int32Array(g.n).fill(-1)
  const closed = new Uint8Array(g.n)
  const open = new IndexedQueue(g.n)
  const expanded: number[] = []
  cost[s] = 0
  open.push(s, h(s))
  for (let u = open.pop(); u !== undefined; u = open.pop()) {
    closed[u] = 1
    expanded.push(u)
    if (u === t) break
    for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
      const v = a.indices[k] ?? 0
      if (closed[v] === 1) continue
      const w = weight(k)
      if (w < 0) throw new WhiteboxError('domain', 'aStar: negative edge weights are not allowed')
      const alt = (cost[u] ?? 0) + w
      if (alt < (cost[v] ?? Number.POSITIVE_INFINITY)) {
        cost[v] = alt
        predecessor[v] = u
        open.push(v, alt + h(v))
      }
    }
  }
  const distance = cost[t] ?? Number.POSITIVE_INFINITY
  return { path: reconstructPath(predecessor, s, t), distance, expanded }
}

/**
 * All-pairs shortest paths by the Floyd-Warshall algorithm (scipy.sparse.csgraph.floyd_warshall), O(n^3), with
 * negative weights allowed.
 *
 * @param g - The graph.
 * @param options - `weighted` (default true).
 * @returns The distance matrix and the predecessor matrix.
 * @throws WhiteboxError with code `value` when there is a negative cycle.
 * @concept shortest-paths
 * @example
 * ```ts
 * import { floydWarshall, pathGraph } from 'whitebox-engine/graph'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(floydWarshall(pathGraph(3)).distance) // => [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
 * ```
 */
export function floydWarshall(g: Graph, options: PathOptions = {}): AllPairsShortestPaths {
  const n = g.n
  const weight = weightOf(g, options)
  const d = new Float64Array(n * n).fill(Number.POSITIVE_INFINITY)
  const p = new Int32Array(n * n).fill(-1)
  for (let i = 0; i < n; i++) d[i * n + i] = 0
  const a = g.out
  for (let u = 0; u < n; u++) {
    for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
      const v = a.indices[k] ?? 0
      const w = weight(k)
      if (u === v) {
        if (w < 0) throw new WhiteboxError('value', 'floydWarshall: the graph has a negative cycle')
        continue
      }
      if (w < (d[u * n + v] ?? Number.POSITIVE_INFINITY)) {
        d[u * n + v] = w
        p[u * n + v] = u
      }
    }
  }
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      const dik = d[i * n + k] ?? Number.POSITIVE_INFINITY
      if (dik === Number.POSITIVE_INFINITY) continue
      for (let j = 0; j < n; j++) {
        const alt = dik + (d[k * n + j] ?? Number.POSITIVE_INFINITY)
        if (alt < (d[i * n + j] ?? Number.POSITIVE_INFINITY)) {
          d[i * n + j] = alt
          p[i * n + j] = p[k * n + j] ?? -1
        }
      }
    }
  }
  for (let i = 0; i < n; i++) {
    if ((d[i * n + i] ?? 0) < 0) throw new WhiteboxError('value', 'floydWarshall: the graph has a negative cycle')
  }
  return { distance: fromStorage(d, [n, n], 'float64'), predecessor: fromStorage(p, [n, n], 'int32') }
}

/**
 * All-pairs shortest paths by Johnson's algorithm (scipy.sparse.csgraph.johnson): Bellman-Ford from a virtual node
 * gives potentials h that make every weight w(u, v) + h(u) - h(v) non-negative, then Dijkstra runs from every node.
 * O(n m log n), so better than Floyd-Warshall on sparse graphs.
 *
 * @param g - The graph.
 * @param options - `weighted` (default true).
 * @returns The distance matrix and the predecessor matrix.
 * @throws WhiteboxError with code `value` when there is a negative cycle.
 * @concept shortest-paths
 * @example
 * ```ts
 * import { digraph, johnson } from 'whitebox-engine/graph'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(johnson(digraph([[0, 1, 2], [1, 2, -1]])).distance)[0] // => [0, 2, 1]
 * ```
 */
export function johnson(g: Graph, options: PathOptions = {}): AllPairsShortestPaths {
  const n = g.n
  const weight = weightOf(g, options)
  const a = g.out
  // Bellman-Ford from a virtual node joined to every node by a zero edge: start every potential at 0.
  const h = new Float64Array(n)
  let changed = true
  for (let round = 0; round <= n && changed; round++) {
    changed = false
    for (let u = 0; u < n; u++) {
      for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
        const v = a.indices[k] ?? 0
        const alt = (h[u] ?? 0) + weight(k)
        if (alt < (h[v] ?? 0)) {
          h[v] = alt
          changed = true
        }
      }
    }
    if (changed && round === n) throw new WhiteboxError('value', 'johnson: the graph has a negative cycle')
  }
  const reweighted = (k: number): number => {
    const u = g.out.indices[k] ?? 0
    return Math.max(0, weight(k) + (h[sourceOf(g, k)] ?? 0) - (h[u] ?? 0))
  }
  const d = new Float64Array(n * n)
  const p = new Int32Array(n * n)
  for (let s = 0; s < n; s++) {
    const r = dijkstraCore(g, s, reweighted)
    for (let t = 0; t < n; t++) {
      const dt = r.distance[t] ?? Number.POSITIVE_INFINITY
      d[s * n + t] = dt === Number.POSITIVE_INFINITY ? dt : dt - (h[s] ?? 0) + (h[t] ?? 0)
      p[s * n + t] = r.predecessor[t] ?? -1
    }
  }
  return { distance: fromStorage(d, [n, n], 'float64'), predecessor: fromStorage(p, [n, n], 'int32') }
}

/** The row (source node) of adjacency entry k, by binary search in indptr. */
function sourceOf(g: Graph, k: number): number {
  const indptr = g.out.indptr
  let lo = 0
  let hi = g.n - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if ((indptr[mid] ?? 0) <= k) lo = mid
    else hi = mid - 1
  }
  return lo
}

/**
 * The path from a source to a target read off predecessors: a single-source predecessor array (from
 * {@link dijkstra} or {@link bellmanFord}), or an all-pairs predecessor matrix (from {@link floydWarshall} or
 * {@link johnson}).
 *
 * @param predecessor - Predecessors from `source`, or the `[n, n]` predecessor matrix.
 * @param source - The source.
 * @param target - The target.
 * @returns The nodes from source to target, or null when the target is unreachable.
 * @example
 * ```ts
 * import { dijkstra, pathGraph, reconstructPath } from 'whitebox-engine/graph'
 * reconstructPath(dijkstra(pathGraph(4), 0).predecessor, 0, 3) // => [0, 1, 2, 3]
 * ```
 */
export function reconstructPath(
  predecessor: ArrayLike<number> | Tensor<'int32'>,
  source: number,
  target: number,
): number[] | null {
  const matrix = 'shape' in predecessor && 'data' in predecessor
  const n = matrix ? ((predecessor as Tensor<'int32'>).shape[0] ?? 0) : (predecessor as ArrayLike<number>).length
  const row = matrix
    ? (predecessor as Tensor<'int32'>).data.subarray(source * n, source * n + n)
    : (predecessor as ArrayLike<number>)
  if (source === target) return [source]
  const path = [target]
  for (let v = target, steps = 0; v !== source; steps++) {
    v = row[v] ?? -1
    if (v < 0 || steps > n) return null
    path.push(v)
  }
  return path.reverse()
}

/** A path with its length, as listed by {@link kShortestPaths}. */
export interface WeightedPath {
  readonly path: readonly number[]
  readonly cost: number
}

/**
 * The k shortest loopless paths from a source to a target by Yen's algorithm (NetworkX shortest_simple_paths, which
 * lists them lazily in the same order): each new path deviates from a previous one at a spur node, with the edges
 * that previous paths take from the same root and the root's nodes removed. Ties are broken by the node sequence.
 *
 * @param g - The graph (non-negative weights).
 * @param source - The source.
 * @param target - The target.
 * @param k - How many paths.
 * @param options - `weighted` (default true).
 * @returns Up to k paths with their costs, in non-decreasing cost.
 * @concept shortest-paths
 * @reference J. Y. Yen, Finding the k shortest loopless paths in a network, Management Science 17 (1971).
 * @example
 * ```ts
 * import { cycleGraph, kShortestPaths } from 'whitebox-engine/graph'
 * kShortestPaths(cycleGraph(5), 0, 2, 2).map((p) => p.path) // => [[0, 1, 2], [0, 4, 3, 2]]
 * ```
 */
export function kShortestPaths(
  g: Graph,
  source: NodeId,
  target: NodeId,
  k: number,
  options: PathOptions = {},
): WeightedPath[] {
  const s = nodeIndex(g, source)
  const t = nodeIndex(g, target)
  const weight = weightOf(g, options)
  const a = g.out
  const edgeBetween = (u: number, v: number): number => {
    for (let q = a.indptr[u] ?? 0; q < (a.indptr[u + 1] ?? 0); q++) if (a.indices[q] === v) return q
    return -1
  }
  const costOf = (path: readonly number[]): number => {
    let c = 0
    for (let i = 0; i + 1 < path.length; i++) c += weight(edgeBetween(path[i] ?? 0, path[i + 1] ?? 0))
    return c
  }
  const first = dijkstraCore(g, s, weight, t)
  const p0 = reconstructPath(first.predecessor, s, t)
  if (p0 === null || k <= 0) return []
  const found: WeightedPath[] = [{ path: p0, cost: first.distance[t] ?? 0 }]
  const candidates = new Map<string, WeightedPath>()
  const order = (x: WeightedPath, y: WeightedPath): number => {
    if (x.cost !== y.cost) return x.cost - y.cost
    for (let i = 0; i < Math.min(x.path.length, y.path.length); i++) {
      if (x.path[i] !== y.path[i]) return (x.path[i] ?? 0) - (y.path[i] ?? 0)
    }
    return x.path.length - y.path.length
  }
  while (found.length < k) {
    const last = (found[found.length - 1] as WeightedPath).path
    for (let i = 0; i + 1 < last.length; i++) {
      const spur = last[i] ?? 0
      const root = last.slice(0, i + 1)
      const blockedEdges = new Set<number>()
      for (const { path } of found) {
        if (path.length > i + 1 && root.every((v, j) => path[j] === v)) {
          const q = edgeBetween(path[i] ?? 0, path[i + 1] ?? 0)
          if (q >= 0) blockedEdges.add(a.edges[q] ?? -1)
        }
      }
      const blockedNodes = new Set(root.slice(0, -1))
      const r = dijkstraCore(g, spur, weight, t, blockedEdges, blockedNodes)
      const tail = reconstructPath(r.predecessor, spur, t)
      if (tail === null) continue
      const path = [...root.slice(0, -1), ...tail]
      const key = path.join(',')
      if (!candidates.has(key) && !found.some((f) => f.path.join(',') === key)) {
        candidates.set(key, { path, cost: costOf(path) })
      }
    }
    if (candidates.size === 0) break
    const best = [...candidates.values()].sort(order)[0] as WeightedPath
    candidates.delete(best.path.join(','))
    found.push(best)
  }
  return found
}

/**
 * The shortest path between two nodes (NetworkX shortest_path): Dijkstra with weights, or breadth-first hop counts
 * with `weighted: false`; negative weights fall back to Bellman-Ford.
 *
 * @param g - The graph.
 * @param source - The source.
 * @param target - The target.
 * @param options - `weighted` (default true).
 * @returns The path and its length, or null when the target is unreachable.
 * @example
 * ```ts
 * import { graph, shortestPath } from 'whitebox-engine/graph'
 * shortestPath(graph([['a', 'b', 1], ['b', 'c', 1], ['a', 'c', 5]]), 'a', 'c') // => { path: [0, 1, 2], cost: 2 }
 * ```
 */
export function shortestPath(g: Graph, source: NodeId, target: NodeId, options: PathOptions = {}): WeightedPath | null {
  const s = nodeIndex(g, source)
  const t = nodeIndex(g, target)
  const negative = options.weighted !== false && g.edges.some((e) => e.weight < 0)
  const r = negative ? bellmanFord(g, s, options) : dijkstraCore(g, s, weightOf(g, options), t)
  const path = reconstructPath(r.predecessor, s, t)
  return path === null ? null : { path, cost: r.distance[t] ?? 0 }
}
