/**
 * Network flows: maximum s-t flow by Dinic's or Edmonds-Karp's algorithm, the minimum cut it certifies (max-flow
 * min-cut theorem), minimum-cost flows meeting node supplies and demands, maximum flow of minimum cost, and maximum
 * bipartite matching by Hopcroft-Karp, as in NetworkX and scipy.
 */

import { kahanSum, MdmlError } from '../base/index.ts'
import { type Graph, type NodeId, nodeIndex } from './graph.ts'
import { IndexedQueue } from './structures.ts'
import { bipartiteColouring } from './traversal.ts'

/**
 * The residual network: arc 2e runs along edge e (source to target), arc 2e + 1 against it. A directed edge's reverse
 * arc starts with no capacity; an undirected edge's two arcs both carry its capacity, so the net flow along the edge is
 * `capacity - residual[2e]` either way.
 */
interface Residual {
  readonly n: number
  readonly to: Int32Array
  readonly residual: Float64Array
  readonly capacity: Float64Array
  /** Arcs leaving each node, ordered by their head. */
  readonly arcs: readonly Int32Array[]
}

function residualNetwork(g: Graph, capacity: ArrayLike<number> | undefined): Residual {
  const m = g.edges.length
  const to = new Int32Array(2 * m)
  const residual = new Float64Array(2 * m)
  const cap = new Float64Array(m)
  const lists: number[][] = Array.from({ length: g.n }, () => [])
  for (const [e, edge] of g.edges.entries()) {
    const c = capacity === undefined ? edge.weight : (capacity[e] ?? 0)
    if (!(c >= 0)) throw new MdmlError('value', `flow: edge ${e} has a negative or missing capacity`)
    cap[e] = c
    to[2 * e] = edge.target
    to[2 * e + 1] = edge.source
    residual[2 * e] = c
    residual[2 * e + 1] = g.directed ? 0 : c
    lists[edge.source]?.push(2 * e)
    if (edge.source !== edge.target) lists[edge.target]?.push(2 * e + 1)
  }
  const arcs = lists.map((l) => Int32Array.from(l.sort((a, b) => (to[a] ?? 0) - (to[b] ?? 0) || a - b)))
  return { n: g.n, to, residual, capacity: cap, arcs }
}

/** Net flow per edge (source to target; negative when an undirected edge carries flow the other way). */
function edgeFlows(r: Residual): Float64Array {
  return Float64Array.from(r.capacity, (c, e) => c - (r.residual[2 * e] ?? 0))
}

function augment(r: Residual, path: readonly number[]): number {
  let bottleneck = Number.POSITIVE_INFINITY
  for (const a of path) bottleneck = Math.min(bottleneck, r.residual[a] ?? 0)
  if (bottleneck === Number.POSITIVE_INFINITY) throw new MdmlError('value', 'maxFlow: the flow is unbounded')
  for (const a of path) {
    r.residual[a] = (r.residual[a] ?? 0) - bottleneck
    r.residual[a ^ 1] = (r.residual[a ^ 1] ?? 0) + bottleneck
  }
  return bottleneck
}

const EPS = 1e-12

function edmondsKarp(r: Residual, s: number, t: number): { value: number; augmentations: number } {
  let value = 0
  let augmentations = 0
  for (;;) {
    const via = new Int32Array(r.n).fill(-1)
    const seen = new Uint8Array(r.n)
    seen[s] = 1
    const queue = [s]
    for (let head = 0; head < queue.length && seen[t] === 0; head++) {
      const u = queue[head] ?? 0
      for (const a of r.arcs[u] ?? []) {
        const v = r.to[a] ?? 0
        if (seen[v] === 1 || (r.residual[a] ?? 0) <= EPS) continue
        seen[v] = 1
        via[v] = a
        queue.push(v)
      }
    }
    if (seen[t] === 0) return { value, augmentations }
    const path: number[] = []
    for (let v = t; v !== s; v = r.to[(via[v] ?? 0) ^ 1] ?? s) path.push(via[v] ?? 0)
    value += augment(r, path)
    augmentations++
  }
}

function dinic(r: Residual, s: number, t: number): { value: number; augmentations: number } {
  let value = 0
  let augmentations = 0
  for (;;) {
    const level = new Int32Array(r.n).fill(-1)
    level[s] = 0
    const queue = [s]
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head] ?? 0
      for (const a of r.arcs[u] ?? []) {
        const v = r.to[a] ?? 0
        if (level[v] === -1 && (r.residual[a] ?? 0) > EPS) {
          level[v] = (level[u] ?? 0) + 1
          queue.push(v)
        }
      }
    }
    if (level[t] === -1) return { value, augmentations }
    // Blocking flow: depth-first along level-increasing arcs, with a current-arc pointer per node.
    const next = new Int32Array(r.n)
    for (;;) {
      const path: number[] = []
      let u = s
      while (u !== t) {
        const list = r.arcs[u] ?? new Int32Array(0)
        let advanced = false
        while ((next[u] ?? 0) < list.length) {
          const a = list[next[u] ?? 0] ?? 0
          const v = r.to[a] ?? 0
          if ((r.residual[a] ?? 0) > EPS && level[v] === (level[u] ?? 0) + 1) {
            path.push(a)
            u = v
            advanced = true
            break
          }
          next[u] = (next[u] ?? 0) + 1
        }
        if (advanced) continue
        // Dead end: retreat and skip the arc that led here.
        level[u] = -1
        const back = path.pop()
        if (back === undefined) break
        u = r.to[back ^ 1] ?? s
        next[u] = (next[u] ?? 0) + 1
      }
      if (u !== t) break
      value += augment(r, path)
      augmentations++
    }
  }
}

/** The result of {@link maxFlow}. */
export interface MaxFlow {
  readonly value: number
  /** Net flow along each edge, source to target. */
  readonly flow: Float64Array
  /** Augmenting paths used. */
  readonly augmentations: number
}

/** Options of the flow functions. */
export interface FlowOptions {
  /** Capacity per edge, in edge order (default: the edge weights); may be Infinity. */
  readonly capacity?: ArrayLike<number>
}

function solveMaxFlow(
  g: Graph,
  source: NodeId,
  sink: NodeId,
  options: FlowOptions & { readonly method?: 'edmonds-karp' | 'dinic' },
): { r: Residual; s: number; value: number; augmentations: number } {
  const s = nodeIndex(g, source)
  const t = nodeIndex(g, sink)
  if (s === t) throw new MdmlError('value', 'maxFlow: source and sink are the same node')
  const r = residualNetwork(g, options.capacity)
  const out = options.method === 'edmonds-karp' ? edmondsKarp(r, s, t) : dinic(r, s, t)
  return { r, s, ...out }
}

/**
 * The maximum flow from a source to a sink (NetworkX maximum_flow, scipy.sparse.csgraph.maximum_flow) by Dinic's
 * blocking flows (default) or Edmonds-Karp's shortest augmenting paths. An undirected edge carries flow either way up
 * to its capacity.
 *
 * @param g - The graph.
 * @param source - The source.
 * @param sink - The sink.
 * @param options - Capacities (default the weights) and `method`.
 * @returns The flow value, the flow on each edge and the number of augmenting paths.
 * @throws MdmlError with code `value` for a negative capacity, an unbounded flow or source equal to sink.
 * @concept network-flows
 * @example
 * ```ts
 * import { digraph, maxFlow } from 'mdml-engine/graph'
 * const g = digraph([[0, 1, 3], [0, 2, 2], [1, 2, 1], [1, 3, 2], [2, 3, 3]])
 * maxFlow(g, 0, 3).value // => 5
 * maxFlow(g, 0, 3, { method: 'edmonds-karp' }).value // => 5
 * ```
 */
export function maxFlow(
  g: Graph,
  source: NodeId,
  sink: NodeId,
  options: FlowOptions & { readonly method?: 'edmonds-karp' | 'dinic' } = {},
): MaxFlow {
  const { r, value, augmentations } = solveMaxFlow(g, source, sink, options)
  return { value, flow: edgeFlows(r), augmentations }
}

/** The result of {@link minCut}. */
export interface MinCut {
  readonly value: number
  /** The source side and the sink side, each sorted. */
  readonly partition: readonly [readonly number[], readonly number[]]
  /** The edges crossing from the source side to the sink side. */
  readonly cutEdges: readonly (readonly [number, number])[]
}

/**
 * A minimum s-t cut (NetworkX minimum_cut): after a maximum flow, the source side is everything reachable from the
 * source in the residual network; by the max-flow min-cut theorem its capacity equals the flow value.
 *
 * @param g - The graph.
 * @param source - The source.
 * @param sink - The sink.
 * @param options - Capacities (default the weights).
 * @returns The cut value, the two sides and the cut edges.
 * @concept network-flows
 * @example
 * ```ts
 * import { digraph, minCut } from 'mdml-engine/graph'
 * const c = minCut(digraph([[0, 1, 3], [0, 2, 2], [1, 2, 1], [1, 3, 2], [2, 3, 3]]), 0, 3)
 * [c.value, c.partition[0]] // => [5, [0]]
 * ```
 */
export function minCut(g: Graph, source: NodeId, sink: NodeId, options: FlowOptions = {}): MinCut {
  const { r, s, value } = solveMaxFlow(g, source, sink, options)
  const side = new Uint8Array(g.n)
  side[s] = 1
  const stack = [s]
  while (stack.length > 0) {
    const u = stack.pop() ?? 0
    for (const a of r.arcs[u] ?? []) {
      const v = r.to[a] ?? 0
      if (side[v] === 0 && (r.residual[a] ?? 0) > EPS) {
        side[v] = 1
        stack.push(v)
      }
    }
  }
  const left: number[] = []
  const right: number[] = []
  for (let v = 0; v < g.n; v++) (side[v] === 1 ? left : right).push(v)
  const cutEdges: [number, number][] = []
  for (const e of g.edges) {
    if (side[e.source] === 1 && side[e.target] === 0) cutEdges.push([e.source, e.target])
    else if (!g.directed && side[e.target] === 1 && side[e.source] === 0) cutEdges.push([e.target, e.source])
  }
  return { value, partition: [left, right], cutEdges }
}

/** The result of {@link minCostFlow}. */
export interface MinCostFlow {
  readonly cost: number
  /** Flow on each edge, in edge order. */
  readonly flow: Float64Array
}

/**
 * A minimum-cost flow meeting node demands (NetworkX min_cost_flow): `demand[v] < 0` is a supply of `-demand[v]`,
 * `demand[v] > 0` a demand, and the demands sum to zero. Negative-cost edges are saturated first, so every residual cost
 * is non-negative; then successive shortest paths with Dijkstra on reduced costs (Johnson potentials) route the
 * remaining supply from a virtual source to a virtual sink.
 *
 * @param g - A directed graph.
 * @param demand - The demand of each node.
 * @param options - Capacities (default the weights; Infinity allowed when the cost is non-negative) and the `cost` per
 *   edge (default 1).
 * @returns The total cost and the flow on each edge.
 * @throws MdmlError with code `value` for an undirected graph, unbalanced demands, an unbounded negative-cost edge
 *   or infeasible demands.
 * @concept network-flows
 * @example
 * ```ts
 * import { digraph, minCostFlow } from 'mdml-engine/graph'
 * const g = digraph([[0, 1], [0, 2], [1, 3], [2, 3]])
 * const r = minCostFlow(g, [-4, 0, 0, 4], { capacity: [4, 2, 3, 5], cost: [2, 1, 1, 3] })
 * [r.cost, Array.from(r.flow)] // => [13, [3, 1, 3, 1]]
 * ```
 */
export function minCostFlow(
  g: Graph,
  demand: ArrayLike<number>,
  options: FlowOptions & { readonly cost?: ArrayLike<number> } = {},
): MinCostFlow {
  if (!g.directed) throw new MdmlError('value', 'minCostFlow: the graph must be directed')
  if (demand.length !== g.n) throw new MdmlError('value', 'minCostFlow: give one demand per node')
  const total = kahanSum(demand)
  if (Math.abs(total) > 1e-9) throw new MdmlError('value', 'minCostFlow: the demands do not sum to zero')
  const m = g.edges.length
  const cost = Float64Array.from({ length: m }, (_, e) => options.cost?.[e] ?? 1)
  // Arcs 0 .. 2m - 1 as in the residual network, then 2m + 2v (virtual source to v) and 2m + 2v + 1 back, and
  // 2m + 2n + 2v (v to virtual sink) and its reverse.
  const S = g.n
  const T = g.n + 1
  const arcCount = 2 * m + 4 * g.n
  const to = new Int32Array(arcCount)
  const residual = new Float64Array(arcCount)
  const arcCost = new Float64Array(arcCount)
  const lists: number[][] = Array.from({ length: g.n + 2 }, () => [])
  const imbalance = Float64Array.from({ length: g.n }, (_, v) => -(demand[v] ?? 0))
  let base = 0
  const flow = new Float64Array(m)
  const addArc = (a: number, from: number, head: number, cap: number, c: number): void => {
    to[a] = head
    to[a ^ 1] = from
    residual[a] = cap
    residual[a ^ 1] = 0
    arcCost[a] = c
    arcCost[a ^ 1] = -c
    lists[from]?.push(a)
    lists[head]?.push(a ^ 1)
  }
  for (const [e, edge] of g.edges.entries()) {
    const cap = options.capacity === undefined ? edge.weight : (options.capacity[e] ?? 0)
    if (!(cap >= 0)) throw new MdmlError('value', `minCostFlow: edge ${e} has a negative or missing capacity`)
    const c = cost[e] ?? 0
    if (c < 0) {
      // Saturate: the reverse arc then has positive cost, and the endpoints' balances change.
      if (cap === Number.POSITIVE_INFINITY) throw new MdmlError('value', 'minCostFlow: unbounded negative cost')
      addArc(2 * e + 1, edge.target, edge.source, cap, -c)
      flow[e] = cap
      base += c * cap
      imbalance[edge.source] = (imbalance[edge.source] ?? 0) - cap
      imbalance[edge.target] = (imbalance[edge.target] ?? 0) + cap
    } else addArc(2 * e, edge.source, edge.target, cap, c)
  }
  let required = 0
  for (let v = 0; v < g.n; v++) {
    const b = imbalance[v] ?? 0
    if (b > 0) {
      addArc(2 * m + 2 * v, S, v, b, 0)
      required += b
    } else if (b < 0) addArc(2 * m + 2 * g.n + 2 * v, v, T, -b, 0)
  }
  const nodes = g.n + 2
  const potential = new Float64Array(nodes)
  let routed = 0
  let pathCost = 0
  while (routed < required - 1e-12) {
    const dist = new Float64Array(nodes).fill(Number.POSITIVE_INFINITY)
    const via = new Int32Array(nodes).fill(-1)
    const queue = new IndexedQueue(nodes)
    dist[S] = 0
    queue.push(S, 0)
    for (let u = queue.pop(); u !== undefined; u = queue.pop()) {
      for (const a of lists[u] ?? []) {
        if ((residual[a] ?? 0) <= EPS) continue
        const v = to[a] ?? 0
        const reduced = (arcCost[a] ?? 0) + (potential[u] ?? 0) - (potential[v] ?? 0)
        const alt = (dist[u] ?? 0) + Math.max(0, reduced)
        if (alt < (dist[v] ?? Number.POSITIVE_INFINITY) - 1e-15) {
          dist[v] = alt
          via[v] = a
          queue.push(v, alt)
        }
      }
    }
    if (dist[T] === Number.POSITIVE_INFINITY) throw new MdmlError('value', 'minCostFlow: the demands are infeasible')
    for (let v = 0; v < nodes; v++)
      if (dist[v] !== Number.POSITIVE_INFINITY) potential[v] = (potential[v] ?? 0) + (dist[v] ?? 0)
    let push = required - routed
    for (let v = T; v !== S; v = to[(via[v] ?? 0) ^ 1] ?? S) push = Math.min(push, residual[via[v] ?? 0] ?? 0)
    for (let v = T; v !== S; v = to[(via[v] ?? 0) ^ 1] ?? S) {
      const a = via[v] ?? 0
      residual[a] = (residual[a] ?? 0) - push
      residual[a ^ 1] = (residual[a ^ 1] ?? 0) + push
      pathCost += push * (arcCost[a] ?? 0)
    }
    routed += push
  }
  for (const [e] of g.edges.entries()) {
    // Net flow: arcs along the edge carry it forwards; a saturated negative edge starts full and loses what its
    // (reversed) arc carries.
    flow[e] = (cost[e] ?? 0) < 0 ? (flow[e] ?? 0) - (residual[2 * e] ?? 0) : (residual[2 * e + 1] ?? 0)
  }
  return { cost: base + pathCost, flow }
}

/**
 * A maximum flow of minimum cost from a source to a sink (NetworkX max_flow_min_cost).
 *
 * @param g - A directed graph.
 * @param source - The source.
 * @param sink - The sink.
 * @param options - Capacities (default the weights) and costs (default 1).
 * @returns The flow value, its cost and the flow on each edge.
 * @concept network-flows
 * @example
 * ```ts
 * import { digraph, maxFlowMinCost } from 'mdml-engine/graph'
 * const g = digraph([[0, 1, 1], [0, 2, 1], [1, 3, 1], [2, 3, 1], [1, 2, 1]])
 * const r = maxFlowMinCost(g, 0, 3, { cost: [1, 5, 1, 1, 1] })
 * [r.value, r.cost] // => [2, 8]
 * ```
 */
export function maxFlowMinCost(
  g: Graph,
  source: NodeId,
  sink: NodeId,
  options: FlowOptions & { readonly cost?: ArrayLike<number> } = {},
): MaxFlow & { readonly cost: number } {
  const s = nodeIndex(g, source)
  const t = nodeIndex(g, sink)
  const { value, augmentations } = maxFlow(g, s, t, options)
  const demand = new Float64Array(g.n)
  demand[s] = -value
  demand[t] = value
  const r = minCostFlow(g, demand, options)
  return { value, flow: r.flow, augmentations, cost: r.cost }
}

/** The result of {@link bipartiteMatching}. */
export interface Matching {
  readonly size: number
  /** Each node's partner (-1 when unmatched). */
  readonly mate: Int32Array
  /** The matched pairs `[left, right]`, sorted by the left node. */
  readonly pairs: readonly (readonly [number, number])[]
}

/**
 * A maximum matching of a bipartite graph by the Hopcroft-Karp algorithm (NetworkX bipartite.hopcroft_karp_matching,
 * scipy's maximum_bipartite_matching): phases of breadth-first layering from the free left nodes, then vertex-disjoint
 * shortest augmenting paths by depth-first search, in O(m sqrt(n)).
 *
 * @param g - A bipartite graph (edge directions are ignored).
 * @param options - The left nodes (default: colour 0 of {@link bipartiteColouring}).
 * @returns The matching size, the partners and the pairs.
 * @throws MdmlError with code `value` when the graph is not bipartite or an edge joins two left nodes.
 * @concept matching
 * @reference J. Hopcroft and R. Karp, An n^{5/2} algorithm for maximum matchings in bipartite graphs, SIAM J. Computing
 *   2 (1973).
 * @example
 * ```ts
 * import { bipartiteMatching, graph } from 'mdml-engine/graph'
 * bipartiteMatching(graph([[0, 3], [0, 4], [1, 3], [2, 5]])).size // => 3
 * ```
 */
export function bipartiteMatching(g: Graph, options: { readonly left?: readonly NodeId[] } = {}): Matching {
  const isLeft = new Uint8Array(g.n)
  if (options.left === undefined) {
    const colour = bipartiteColouring(g)
    if (colour === null) throw new MdmlError('value', 'bipartiteMatching: the graph is not bipartite')
    for (let v = 0; v < g.n; v++) isLeft[v] = colour[v] === 0 ? 1 : 0
  } else for (const v of options.left) isLeft[nodeIndex(g, v)] = 1
  const adj: number[][] = Array.from({ length: g.n }, () => [])
  for (const e of g.edges) {
    if (isLeft[e.source] === isLeft[e.target]) {
      throw new MdmlError('value', `bipartiteMatching: edge (${e.source}, ${e.target}) lies within one side`)
    }
    const [l, r] = isLeft[e.source] === 1 ? [e.source, e.target] : [e.target, e.source]
    adj[l]?.push(r)
  }
  for (const list of adj) list.sort((a, b) => a - b)
  const mate = new Int32Array(g.n).fill(-1)
  const left = Array.from({ length: g.n }, (_, v) => v).filter((v) => isLeft[v] === 1)
  const INF = Number.POSITIVE_INFINITY
  const dist = new Float64Array(g.n)
  let size = 0
  for (;;) {
    // Layer the left nodes by alternating-path distance from the free ones.
    const queue: number[] = []
    for (const u of left) {
      if (mate[u] === -1) {
        dist[u] = 0
        queue.push(u)
      } else dist[u] = INF
    }
    let found = INF
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head] ?? 0
      if ((dist[u] ?? INF) >= found) continue
      for (const v of adj[u] ?? []) {
        const w = mate[v] ?? -1
        if (w === -1) found = Math.min(found, (dist[u] ?? 0) + 1)
        else if (dist[w] === INF) {
          dist[w] = (dist[u] ?? 0) + 1
          queue.push(w)
        }
      }
    }
    if (found === INF) break
    // Vertex-disjoint shortest augmenting paths by iterative depth-first search.
    const next = new Int32Array(g.n)
    for (const root of left) {
      if (mate[root] !== -1) continue
      const stack = [root]
      const taken: number[] = []
      while (stack.length > 0) {
        const u = stack[stack.length - 1] ?? 0
        const list = adj[u] ?? []
        let moved = false
        while ((next[u] ?? 0) < list.length) {
          const v = list[next[u] ?? 0] ?? 0
          next[u] = (next[u] ?? 0) + 1
          const w = mate[v] ?? -1
          if (w === -1 ? (dist[u] ?? 0) + 1 === found : dist[w] === (dist[u] ?? 0) + 1) {
            taken.push(v)
            if (w === -1) {
              // Augment along the stack.
              for (let i = 0; i < stack.length; i++) {
                const a = stack[i] ?? 0
                const b = taken[i] ?? 0
                mate[a] = b
                mate[b] = a
              }
              size++
              stack.length = 0
            } else stack.push(w)
            moved = true
            break
          }
        }
        if (moved) continue
        dist[u] = INF
        stack.pop()
        taken.pop()
      }
    }
  }
  const pairs: [number, number][] = left.filter((u) => mate[u] !== -1).map((u) => [u, mate[u] ?? -1])
  return { size, mate, pairs }
}
