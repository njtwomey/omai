/**
 * Triangulation and junction trees, the graph machinery behind exact inference: greedy elimination orderings
 * (min-degree, min-weight, min-fill), triangulation into a chordal graph, maximum cardinality search and the chordality
 * test, maximal cliques, induced width and exact treewidth, and the construction of a junction tree with the running
 * intersection property.
 */

import { MdmlError } from '../base/index.ts'
import { maximalCliques as cliques } from '../graph/index.ts'
import { copyGraph, indexedGraph, link } from './_graph.ts'
import { connectedComponents, type UndirectedGraph, undirectedGraph } from './graphs.ts'

/** The greedy criteria of {@link eliminationOrder}. */
export type EliminationHeuristic = 'minDegree' | 'minFill' | 'minWeight' | 'weightedMinFill'

/**
 * The cost of eliminating a vertex from the current graph, by a greedy criterion (Koller and Friedman, section 9.4.3):
 * `minDegree` counts neighbours, `minWeight` multiplies their cardinalities, `minFill` counts the fill-in edges that
 * elimination adds, and `weightedMinFill` sums the products of the cardinalities at the ends of each fill-in edge.
 *
 * @param graph - The current graph.
 * @param v - The vertex.
 * @param heuristic - The criterion.
 * @param cardinality - Numbers of states (default 2), for the weighted criteria.
 * @returns The cost.
 * @concept elimination-orderings
 * @example
 * ```ts
 * import { undirectedGraph, eliminationCost } from 'mdml-engine/pgm'
 * eliminationCost(undirectedGraph([['a', 'b'], ['a', 'c']]), 'a', 'minFill') // => 1
 * ```
 */
export function eliminationCost(
  graph: UndirectedGraph,
  v: string,
  heuristic: EliminationHeuristic,
  cardinality: Readonly<Record<string, number>> = {},
): number {
  const ns = [...(graph.get(v) ?? [])]
  const card = (w: string): number => cardinality[w] ?? 2
  if (heuristic === 'minDegree') return ns.length
  if (heuristic === 'minWeight') return ns.reduce((p, w) => p * card(w), 1)
  let cost = 0
  for (let i = 0; i < ns.length; i++) {
    for (let j = i + 1; j < ns.length; j++) {
      const a = ns[i] as string
      const b = ns[j] as string
      if (!graph.get(a)?.has(b)) cost += heuristic === 'minFill' ? 1 : card(a) * card(b)
    }
  }
  return cost
}

/** Connect the neighbours of v and remove it; returns the clique {v} plus its neighbours and the fill-in edges. */
function eliminateVertex(g: Map<string, Set<string>>, v: string): { clique: string[]; fill: [string, string][] } {
  const ns = [...(g.get(v) ?? [])]
  const fill: [string, string][] = []
  for (let i = 0; i < ns.length; i++) {
    for (let j = i + 1; j < ns.length; j++) {
      const a = ns[i] as string
      const b = ns[j] as string
      if (!g.get(a)?.has(b)) {
        fill.push(a < b ? [a, b] : [b, a])
        link(g, a, b)
      }
    }
  }
  for (const w of ns) g.get(w)?.delete(v)
  g.delete(v)
  return { clique: [v, ...ns], fill }
}

/**
 * A greedy elimination ordering: repeatedly eliminate the vertex of least cost (see {@link eliminationCost}), adding
 * its fill-in edges, ties broken by name.
 *
 * @param graph - Neighbours of each variable (for example a moral or interaction graph).
 * @param variables - The variables to order (others stay in the graph).
 * @param heuristic - The greedy criterion (default `minFill`).
 * @param cardinality - Numbers of states for the weighted criteria (default 2 each).
 * @returns The order.
 * @concept elimination-orderings
 * @example
 * ```ts
 * import { eliminationOrder, undirectedGraph } from 'mdml-engine/pgm'
 * eliminationOrder(undirectedGraph([['a', 'b'], ['b', 'c']]), ['a', 'b', 'c'], 'minDegree') // => ['a', 'b', 'c']
 * ```
 */
export function eliminationOrder(
  graph: UndirectedGraph,
  variables: readonly string[],
  heuristic: EliminationHeuristic = 'minFill',
  cardinality: Readonly<Record<string, number>> = {},
): string[] {
  const g = copyGraph(graph)
  for (const v of variables) if (!g.has(v)) g.set(v, new Set())
  const remaining = new Set(variables)
  const order: string[] = []
  while (remaining.size > 0) {
    let best = ''
    let bestCost = Number.POSITIVE_INFINITY
    for (const v of [...remaining].sort()) {
      const c = eliminationCost(g, v, heuristic, cardinality)
      if (c < bestCost) {
        best = v
        bestCost = c
      }
    }
    eliminateVertex(g, best)
    remaining.delete(best)
    order.push(best)
  }
  return order
}

/** The result of {@link triangulate}. */
export interface Triangulation {
  /** The chordal graph: the input plus the fill-in edges. */
  readonly graph: Map<string, Set<string>>
  /** The fill-in edges added. */
  readonly fillIns: readonly [string, string][]
  /** The maximal cliques created during elimination (the clusters of a junction tree), in elimination order. */
  readonly cliques: readonly string[][]
  /** The elimination order used. */
  readonly order: readonly string[]
}

/**
 * Triangulate a graph by eliminating its vertices in an order: each elimination connects the remaining neighbours of
 * the eliminated vertex; the result is chordal, and the elimination cliques that are maximal are its maximal cliques.
 *
 * @param graph - The graph.
 * @param order - An order of all vertices (default: greedy min-fill).
 * @returns The chordal graph, the fill-in edges, the maximal cliques and the order.
 * @throws MdmlError with code `value` when the order misses or repeats a vertex.
 * @concept triangulation
 * @example
 * ```ts
 * import { triangulate, undirectedGraph } from 'mdml-engine/pgm'
 * const square = undirectedGraph([['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'a']])
 * triangulate(square, ['a', 'b', 'c', 'd']).fillIns // => [['b', 'd']]
 * ```
 */
export function triangulate(graph: UndirectedGraph, order?: readonly string[]): Triangulation {
  const vertices = [...graph.keys()]
  const ord = order ?? eliminationOrder(graph, vertices, 'minFill')
  if (ord.length !== vertices.length || new Set(ord).size !== ord.length || ord.some((v) => !graph.has(v))) {
    throw new MdmlError('value', 'triangulate: the order must list every vertex once')
  }
  const g = copyGraph(graph)
  const chordal = copyGraph(graph)
  const fillIns: [string, string][] = []
  const all: string[][] = []
  for (const v of ord) {
    const { clique, fill } = eliminateVertex(g, v)
    for (const [a, b] of fill) link(chordal, a, b)
    fillIns.push(...fill)
    all.push(clique)
  }
  const cliques = all.filter(
    (c, i) =>
      !all.some(
        (d, j) => j !== i && d.length >= c.length && c.every((v) => d.includes(v)) && (d.length > c.length || j < i),
      ),
  )
  return { graph: chordal, fillIns, cliques: cliques.map((c) => [...c].sort()), order: [...ord] }
}

/**
 * Maximum cardinality search (Tarjan and Yannakakis): visit next the vertex with the most visited neighbours, ties by
 * name. The reverse of the visit order is a perfect elimination order exactly when the graph is chordal.
 *
 * @param graph - The graph.
 * @returns The visit order.
 * @example
 * ```ts
 * import { maximumCardinalitySearch, undirectedGraph } from 'mdml-engine/pgm'
 * maximumCardinalitySearch(undirectedGraph([['a', 'b'], ['b', 'c']])) // => ['a', 'b', 'c']
 * ```
 */
export function maximumCardinalitySearch(graph: UndirectedGraph): string[] {
  const weight = new Map([...graph.keys()].map((v) => [v, 0]))
  const order: string[] = []
  while (weight.size > 0) {
    let best = ''
    let bw = -1
    for (const v of [...weight.keys()].sort()) {
      const w = weight.get(v) ?? 0
      if (w > bw) {
        best = v
        bw = w
      }
    }
    order.push(best)
    weight.delete(best)
    for (const w of graph.get(best) ?? []) if (weight.has(w)) weight.set(w, (weight.get(w) ?? 0) + 1)
  }
  return order
}

/**
 * Whether a graph is chordal (every cycle of four or more vertices has a chord): maximum cardinality search gives a
 * perfect elimination order exactly then.
 *
 * @param graph - The graph.
 * @returns `true` when chordal.
 * @concept triangulation
 * @example
 * ```ts
 * import { isChordal, undirectedGraph } from 'mdml-engine/pgm'
 * isChordal(undirectedGraph([['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'a']])) // => false
 * ```
 */
export function isChordal(graph: UndirectedGraph): boolean {
  return triangulate(graph, maximumCardinalitySearch(graph).reverse()).fillIns.length === 0
}

/**
 * The maximal cliques of an undirected graph (Bron-Kerbosch with pivoting, as networkx.find_cliques).
 *
 * @param graph - The graph.
 * @returns Each clique sorted, the list sorted.
 * @example
 * ```ts
 * import { maximalCliques, undirectedGraph } from 'mdml-engine/pgm'
 * maximalCliques(undirectedGraph([['a', 'b'], ['b', 'c'], ['a', 'c'], ['c', 'd']])) // => [['a', 'b', 'c'], ['c', 'd']]
 * ```
 */
export function maximalCliques(graph: UndirectedGraph): string[][] {
  const { g, names } = indexedGraph(graph)
  return cliques(g)
    .map((c) => c.map((i) => names[i] ?? ''))
    .sort((a, b) => a.join('\u0000').localeCompare(b.join('\u0000')))
}

/**
 * The induced width of an elimination order: the size of the largest elimination clique minus one (an upper bound on
 * the treewidth, attained by an optimal order).
 *
 * @param graph - The graph.
 * @param order - An order of all vertices.
 * @returns The width.
 * @concept treewidth
 * @example
 * ```ts
 * import { inducedWidth, undirectedGraph } from 'mdml-engine/pgm'
 * inducedWidth(undirectedGraph([['a', 'b'], ['b', 'c']]), ['b', 'a', 'c']) // => 2
 * ```
 */
export function inducedWidth(graph: UndirectedGraph, order: readonly string[]): number {
  return triangulate(graph, order).cliques.reduce((w, c) => Math.max(w, c.length - 1), 0)
}

/**
 * The treewidth of a graph: exactly, by dynamic programming over vertex subsets (Bodlaender et al. 2006, O(2^n n), for
 * graphs of up to about 20 vertices), or as the best induced width of the greedy min-degree and min-fill orders (an
 * upper bound, networkx's `treewidth_min_degree` and `treewidth_min_fill_in`).
 *
 * @param graph - The graph.
 * @param method - `exact` (default for up to 20 vertices) or `heuristic`.
 * @returns The treewidth (or its upper bound).
 * @throws MdmlError with code `value` for an exact request on more than 24 vertices.
 * @concept treewidth
 * @example
 * ```ts
 * import { treewidth, undirectedGraph } from 'mdml-engine/pgm'
 * treewidth(undirectedGraph([['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'a']])) // => 2
 * ```
 */
export function treewidth(graph: UndirectedGraph, method?: 'exact' | 'heuristic'): number {
  const vertices = [...graph.keys()].sort()
  const n = vertices.length
  const how = method ?? (n <= 20 ? 'exact' : 'heuristic')
  if (how === 'heuristic') {
    return Math.min(
      inducedWidth(graph, eliminationOrder(graph, vertices, 'minDegree')),
      inducedWidth(graph, eliminationOrder(graph, vertices, 'minFill')),
    )
  }
  if (n > 24) throw new MdmlError('value', `treewidth: exact treewidth of ${n} vertices is too expensive`)
  if (n === 0) return -1
  const index = new Map(vertices.map((v, i) => [v, i]))
  const adj = vertices.map((v) => {
    let m = 0
    for (const w of graph.get(v) ?? []) m |= 1 << (index.get(w) ?? 0)
    return m
  })
  // q(S, v): the number of vertices outside S + {v} reachable from v through S.
  const q = (s: number, v: number): number => {
    let seen = 1 << v
    const stack = [v]
    let count = 0
    while (stack.length > 0) {
      const x = stack.pop() as number
      let nb = (adj[x] ?? 0) & ~seen
      while (nb !== 0) {
        const bit = nb & -nb
        nb &= ~bit
        seen |= bit
        const w = 31 - Math.clz32(bit)
        if (s & bit) stack.push(w)
        else count++
      }
    }
    return count
  }
  const full = (1 << n) - 1
  const tw = new Int32Array(1 << n).fill(0x7fffffff)
  tw[0] = -1
  for (let s = 1; s <= full; s++) {
    let best = 0x7fffffff
    for (let v = 0; v < n; v++) {
      if (!(s & (1 << v))) continue
      const rest = s & ~(1 << v)
      const cand = Math.max(tw[rest] ?? 0, q(rest, v))
      if (cand < best) best = cand
    }
    tw[s] = best
  }
  return tw[full] ?? 0
}

/** A junction tree: clusters of variables joined by separators, with the running intersection property. */
export interface JunctionTree {
  readonly cliques: readonly (readonly string[])[]
  /** Tree edges between clique indices, with their separators. */
  readonly edges: readonly { readonly from: number; readonly to: number; readonly separator: readonly string[] }[]
}

/**
 * A junction tree of a graph: triangulate it, take the maximal cliques, and join them by a maximum-weight spanning tree
 * on separator sizes (Kruskal), which has the running intersection property. Disconnected graphs give a forest joined
 * by empty separators, so the result is always one tree.
 *
 * @param graph - The graph (a moral or interaction graph).
 * @param order - The elimination order (default: greedy min-fill).
 * @returns The cliques and the tree edges.
 * @concept junction-trees
 * @example
 * ```ts
 * import { junctionTree, undirectedGraph } from 'mdml-engine/pgm'
 * const jt = junctionTree(undirectedGraph([['a', 'b'], ['b', 'c']]))
 * jt.cliques // => [['a', 'b'], ['b', 'c']]
 * jt.edges // => [{ from: 0, to: 1, separator: ['b'] }]
 * ```
 */
export function junctionTree(graph: UndirectedGraph, order?: readonly string[]): JunctionTree {
  const cliques = [...triangulate(graph, order).cliques].sort((a, b) =>
    a.join('\u0000').localeCompare(b.join('\u0000')),
  )
  const candidates: { from: number; to: number; separator: string[] }[] = []
  for (let i = 0; i < cliques.length; i++) {
    for (let j = i + 1; j < cliques.length; j++) {
      const ci = cliques[i] ?? []
      const cj = cliques[j] ?? []
      candidates.push({ from: i, to: j, separator: ci.filter((v) => cj.includes(v)) })
    }
  }
  candidates.sort((a, b) => b.separator.length - a.separator.length || a.from - b.from || a.to - b.to)
  const root = cliques.map((_, i) => i)
  const find = (i: number): number => {
    let r = i
    while (root[r] !== r) r = root[r] ?? r
    return r
  }
  const edges: { from: number; to: number; separator: string[] }[] = []
  for (const e of candidates) {
    const a = find(e.from)
    const b = find(e.to)
    if (a === b) continue
    root[a] = b
    edges.push(e)
  }
  return { cliques, edges }
}

/**
 * Whether a cluster tree has the running intersection property: for every variable, the clusters containing it form a
 * connected subtree.
 *
 * @param tree - The cluster tree.
 * @returns `true` when it holds.
 * @concept junction-trees
 * @example
 * ```ts
 * import { hasRunningIntersection } from 'mdml-engine/pgm'
 * hasRunningIntersection({ cliques: [['a', 'b'], ['c'], ['b', 'c']], edges: [{ from: 0, to: 1, separator: [] }, { from: 1, to: 2, separator: ['c'] }] }) // => false
 * ```
 */
export function hasRunningIntersection(tree: JunctionTree): boolean {
  const variables = new Set(tree.cliques.flat())
  for (const v of variables) {
    const holding = tree.cliques.map((c, i) => (c.includes(v) ? i : -1)).filter((i) => i >= 0)
    const g = undirectedGraph(
      tree.edges
        .filter((e) => holding.includes(e.from) && holding.includes(e.to))
        .map((e): [string, string] => [String(e.from), String(e.to)]),
      holding.map(String),
    )
    if (connectedComponents(g).length !== 1) return false
  }
  return true
}
