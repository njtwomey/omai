// The residual network of the flow algorithms and the pieces they share: building it from a graph, the net flow per
// edge, augmenting along a path and the source side of a cut. Not re-exported.
import { MdmlError } from '../base/index.ts'
import type { Graph } from './graph.ts'

/** Tolerance below which a residual capacity counts as empty. */
export const EPS = 1e-12

/**
 * The residual network: arc 2e runs along edge e (source to target), arc 2e + 1 against it. A directed edge's reverse
 * arc starts with no capacity; an undirected edge's two arcs both carry its capacity, so the net flow along the edge is
 * `capacity - residual[2e]` either way.
 */
export interface ResidualNetwork {
  readonly n: number
  readonly to: Int32Array
  readonly residual: Float64Array
  readonly capacity: Float64Array
  /** Arcs leaving each node, ordered by their head. */
  readonly arcs: readonly Int32Array[]
}

/** Build the residual network of a graph with the given capacities (default the edge weights). */
export function residualNetwork(g: Graph, capacity: ArrayLike<number> | undefined): ResidualNetwork {
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
export function edgeFlows(r: ResidualNetwork): Float64Array {
  return Float64Array.from(r.capacity, (c, e) => c - (r.residual[2 * e] ?? 0))
}

/** Push the bottleneck of a path of arcs through the residual capacities (in place); returns the amount pushed. */
export function augment(r: ResidualNetwork, path: readonly number[]): number {
  let bottleneck = Number.POSITIVE_INFINITY
  for (const a of path) bottleneck = Math.min(bottleneck, r.residual[a] ?? 0)
  if (bottleneck === Number.POSITIVE_INFINITY) throw new MdmlError('value', 'maxFlow: the flow is unbounded')
  for (const a of path) {
    r.residual[a] = (r.residual[a] ?? 0) - bottleneck
    r.residual[a ^ 1] = (r.residual[a ^ 1] ?? 0) + bottleneck
  }
  return bottleneck
}

/** 1 for the nodes reachable from `s` along arcs with residual capacity: the source side of the cut. */
export function sourceSide(r: ResidualNetwork, s: number): Uint8Array {
  const side = new Uint8Array(r.n)
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
  return side
}
