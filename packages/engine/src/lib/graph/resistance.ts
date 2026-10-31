/**
 * Electrical flow on a graph: the effective resistance between two nodes when every edge is a resistor (or, weighted,
 * a conductance), computed from the pseudoinverse of the graph Laplacian and independently by grounding one node and
 * solving for the potentials; the commute time of a random walk, which is the resistance times the volume of the
 * graph; the Kirchhoff index; the resistance of each edge, which is its probability of lying in a uniformly random
 * spanning tree and sums to n minus the number of components (Foster's theorem); and the number of spanning trees.
 * The resistance is the measure of how well two nodes are connected that a bottleneck (over-squashing) shows up in.
 */

import { MdmlError } from '../base/index.ts'
import { det, inv, solve } from '../numerics/linalg/index.ts'
import { fromStorage, type Tensor, toFlat } from '../tensor/index.ts'
import { connectedComponents } from './components.ts'
import { type Graph, type NodeId, nodeIndex } from './graph.ts'

/** Options of the resistance functions. */
export interface ResistanceOptions {
  /** Treat edge weights as conductances (default true); false counts every edge as one unit resistor. */
  readonly weighted?: boolean
}

function checkUndirected(g: Graph, name: string): void {
  if (g.directed) throw new MdmlError('value', `${name}: undirected graphs only`)
}

function conductance(g: Graph, e: number, weighted: boolean): number {
  return weighted ? (g.edges[e]?.weight ?? 0) : 1
}

/** The Laplacian of the subgraph on `nodes` (one connected component), as a dense row-major array. */
function componentLaplacian(g: Graph, nodes: readonly number[], weighted: boolean): Float64Array {
  const k = nodes.length
  const at = new Map(nodes.map((v, i) => [v, i]))
  const l = new Float64Array(k * k)
  for (let e = 0; e < g.edges.length; e++) {
    const edge = g.edges[e]
    const a = at.get(edge?.source ?? -1)
    const b = at.get(edge?.target ?? -1)
    if (a === undefined || b === undefined || a === b) continue
    const c = conductance(g, e, weighted)
    l[a * k + a] = (l[a * k + a] ?? 0) + c
    l[b * k + b] = (l[b * k + b] ?? 0) + c
    l[a * k + b] = (l[a * k + b] ?? 0) - c
    l[b * k + a] = (l[b * k + a] ?? 0) - c
  }
  return l
}

/**
 * The matrix of effective resistances R_ij = L+_ii + L+_jj - 2 L+_ij, with L+ the pseudoinverse of the Laplacian,
 * found per connected component as (L + 11^T / k)^-1 - 11^T / k. Nodes in different components are infinitely far
 * apart. For a tree it is the hop distance; for a complete graph on n nodes, 2 / n.
 *
 * @param g - An undirected graph.
 * @param options - `weighted` (default true): edge weights are conductances.
 * @returns The symmetric `n x n` matrix, zero on the diagonal.
 * @throws MdmlError with code `value` for a directed graph or a non-positive weight when weighted.
 * @concept effective-resistance
 * @example
 * ```ts
 * import { effectiveResistance, pathGraph } from 'mdml-engine/graph'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(effectiveResistance(pathGraph(3))) // => [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
 * ```
 */
export function effectiveResistance(g: Graph, options: ResistanceOptions = {}): Tensor<'float64'> {
  checkUndirected(g, 'effectiveResistance')
  const weighted = options.weighted ?? true
  if (weighted && g.edges.some((e) => !(e.weight > 0)))
    throw new MdmlError('value', 'effectiveResistance: weights (conductances) must be positive')
  const n = g.n
  const out = new Float64Array(n * n).fill(Number.POSITIVE_INFINITY)
  for (const nodes of connectedComponents(g).components) {
    const k = nodes.length
    const l = componentLaplacian(g, nodes, weighted)
    for (let i = 0; i < k * k; i++) l[i] = (l[i] ?? 0) + 1 / k
    const plus = toFlat(inv(fromStorage(l, [k, k], 'float64')))
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        const r = (plus[i * k + i] ?? 0) + (plus[j * k + j] ?? 0) - 2 * (plus[i * k + j] ?? 0)
        out[(nodes[i] as number) * n + (nodes[j] as number)] = i === j ? 0 : r
      }
    }
  }
  return fromStorage(out, [n, n], 'float64')
}

/**
 * The effective resistance between two nodes by grounding one of them: inject a unit current at `u`, hold `v` at zero
 * potential, solve the Laplacian system for the potentials of the rest; the potential at `u` is the resistance. A
 * different computation from {@link effectiveResistance}'s, used to cross-check it.
 *
 * @param g - An undirected graph.
 * @param u - The source node.
 * @param v - The sink node.
 * @param options - `weighted` (default true).
 * @returns The resistance (infinite when the nodes are in different components).
 * @throws MdmlError with code `value` for a directed graph.
 * @concept effective-resistance
 * @example
 * ```ts
 * import { cycleGraph, effectiveResistanceBetween } from 'mdml-engine/graph'
 * effectiveResistanceBetween(cycleGraph(6), 0, 3) // => 1.5
 * ```
 */
export function effectiveResistanceBetween(g: Graph, u: NodeId, v: NodeId, options: ResistanceOptions = {}): number {
  checkUndirected(g, 'effectiveResistanceBetween')
  const a = nodeIndex(g, u)
  const b = nodeIndex(g, v)
  if (a === b) return 0
  const comps = connectedComponents(g)
  if (comps.labels[a] !== comps.labels[b]) return Number.POSITIVE_INFINITY
  const nodes = comps.components[comps.labels[a] ?? 0] as readonly number[]
  const k = nodes.length
  const full = componentLaplacian(g, nodes, options.weighted ?? true)
  const keep = nodes.map((_, i) => i).filter((i) => nodes[i] !== b)
  const reduced = new Float64Array(keep.length * keep.length)
  for (let r = 0; r < keep.length; r++) {
    for (let c = 0; c < keep.length; c++)
      reduced[r * keep.length + c] = full[(keep[r] as number) * k + (keep[c] as number)] ?? 0
  }
  const rhs = new Float64Array(keep.length)
  rhs[keep.findIndex((i) => nodes[i] === a)] = 1
  const x = toFlat(
    solve(fromStorage(reduced, [keep.length, keep.length], 'float64'), fromStorage(rhs, [keep.length], 'float64')),
  )
  return x[keep.findIndex((i) => nodes[i] === a)] ?? 0
}

/**
 * The commute time between two nodes of a random walk on the graph: the expected steps to go from `u` to `v` and back,
 * H(u, v) + H(v, u) = vol(G) R_uv, with vol(G) the total (weighted) degree, 2m for an unweighted graph. Far nodes and
 * bottlenecks have long commute times because the resistance between them is high.
 *
 * @param g - A connected undirected graph.
 * @param u - One node.
 * @param v - The other.
 * @param options - `weighted` (default true): the walk follows edges in proportion to their weight.
 * @returns The commute time (infinite across components).
 * @throws MdmlError with code `value` for a directed graph.
 * @concept effective-resistance
 * @example
 * ```ts
 * import { commuteTime, pathGraph } from 'mdml-engine/graph'
 * commuteTime(pathGraph(3), 0, 2) // => 8
 * ```
 */
export function commuteTime(g: Graph, u: NodeId, v: NodeId, options: ResistanceOptions = {}): number {
  const weighted = options.weighted ?? true
  const volume = g.edges.reduce((acc, e) => acc + 2 * (weighted ? e.weight : 1), 0)
  return volume * effectiveResistanceBetween(g, u, v, options)
}

/**
 * The Kirchhoff index (NetworkX effective_graph_resistance): the sum of the effective resistances over all unordered
 * pairs of nodes. Equal to n times the sum of 1 / lambda_i over the nonzero Laplacian eigenvalues.
 *
 * @param g - A connected undirected graph.
 * @param options - `weighted` (default true).
 * @returns The index (infinite when the graph is disconnected).
 * @throws MdmlError with code `value` for a directed graph.
 * @concept effective-resistance
 * @example
 * ```ts
 * import { kirchhoffIndex, pathGraph } from 'mdml-engine/graph'
 * kirchhoffIndex(pathGraph(4)) // => 10
 * ```
 */
export function kirchhoffIndex(g: Graph, options: ResistanceOptions = {}): number {
  const r = toFlat(effectiveResistance(g, options))
  let total = 0
  for (let i = 0; i < g.n; i++) for (let j = i + 1; j < g.n; j++) total += r[i * g.n + j] ?? 0
  return total
}

/**
 * The effective resistance of each edge's endpoints, in edge order. Unweighted, it is the probability that the edge lies
 * in a uniformly random spanning tree; weighted, the weight times it is that probability for a tree drawn with
 * probability proportional to the product of its weights. They sum (weighted by conductance) to n minus the number of
 * components (Foster's theorem).
 *
 * @param g - An undirected graph.
 * @param options - `weighted` (default true).
 * @returns One resistance per edge.
 * @throws MdmlError with code `value` for a directed graph.
 * @concept effective-resistance
 * @example
 * ```ts
 * import { cycleGraph, edgeEffectiveResistances } from 'mdml-engine/graph'
 * edgeEffectiveResistances(cycleGraph(4)) // => [0.75, 0.75, 0.75, 0.75]
 * ```
 */
export function edgeEffectiveResistances(g: Graph, options: ResistanceOptions = {}): number[] {
  const r = toFlat(effectiveResistance(g, options))
  return g.edges.map((e) => r[e.source * g.n + e.target] ?? 0)
}

/**
 * The number of spanning trees of a connected undirected graph (Kirchhoff's matrix-tree theorem): the determinant of the
 * Laplacian with one row and column removed. Weighted, the sum over trees of the product of their edge weights. A
 * disconnected graph has none.
 *
 * @param g - An undirected graph.
 * @param options - `weighted` (default false: count the trees).
 * @returns The (weighted) number of spanning trees.
 * @throws MdmlError with code `value` for a directed graph.
 * @concept effective-resistance
 * @example
 * ```ts
 * import { completeGraph, spanningTreeCount } from 'mdml-engine/graph'
 * Math.round(spanningTreeCount(completeGraph(5))) // => 125
 * ```
 */
export function spanningTreeCount(g: Graph, options: ResistanceOptions = {}): number {
  checkUndirected(g, 'spanningTreeCount')
  if (g.n <= 1) return 1
  if (connectedComponents(g).count > 1) return 0
  const nodes = Array.from({ length: g.n }, (_, i) => i)
  const l = componentLaplacian(g, nodes, options.weighted ?? false)
  const k = g.n - 1
  const reduced = new Float64Array(k * k)
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) reduced[i * k + j] = l[i * g.n + j] ?? 0
  return toFlat(det(fromStorage(reduced, [k, k], 'float64')))[0] ?? 0
}
