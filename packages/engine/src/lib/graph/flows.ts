/**
 * Network flows: maximum s-t flow by Dinic's or Edmonds-Karp's algorithm, the minimum cut it certifies (max-flow
 * min-cut theorem), minimum-cost flows meeting node supplies and demands, maximum flow of minimum cost, and maximum
 * bipartite matching by Hopcroft-Karp, as in NetworkX and scipy.
 */

import { runAlgorithm } from '../trace/index.ts'
import { edgeFlows, type ResidualNetwork, residualNetwork, sourceSide } from './_residual.ts'
import { type FlowOptions, maxFlowAlgorithm, minCostFlowAlgorithm } from './flow-algorithms.ts'
import { type Graph, type NodeId, nodeIndex } from './graph.ts'
import { hopcroftKarpAlgorithm } from './hopcroft-karp.ts'

/** The result of {@link maxFlow}. */
export interface MaxFlow {
  readonly value: number
  /** Net flow along each edge, source to target. */
  readonly flow: Float64Array
  /** Augmenting paths used. */
  readonly augmentations: number
}

/** Run the maximum-flow algorithm to the end and rebuild the residual network the flow leaves. */
function solveMaxFlow(
  g: Graph,
  source: NodeId,
  sink: NodeId,
  options: FlowOptions & { readonly method?: 'edmonds-karp' | 'dinic' },
): { r: ResidualNetwork; s: number; value: number; augmentations: number } {
  const alg = maxFlowAlgorithm(g, source, sink, options)
  const run = runAlgorithm(alg, { maxSteps: Number.MAX_SAFE_INTEGER })
  const r = { ...residualNetwork(g, options.capacity), residual: run.state.residual }
  return { r, s: nodeIndex(g, source), value: run.state.value, augmentations: run.state.augmentations }
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
 * @index function Maximum flow and minimum cut.
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
  const side = sourceSide(r, s)
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
  const run = runAlgorithm(minCostFlowAlgorithm(g, demand, options), { maxSteps: Number.MAX_SAFE_INTEGER })
  return { cost: run.state.cost, flow: run.state.flow }
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
  const { left, mate, size } = runAlgorithm(hopcroftKarpAlgorithm(g, options), { maxSteps: g.n + 1 }).state
  const pairs: [number, number][] = left.filter((u) => mate[u] !== -1).map((u) => [u, mate[u] ?? -1])
  return { size, mate, pairs }
}
