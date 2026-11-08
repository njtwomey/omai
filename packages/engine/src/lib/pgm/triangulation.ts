/**
 * Triangulation and junction trees, the graph machinery behind exact inference: greedy elimination orderings
 * (min-degree, min-weight, min-fill), triangulation into a chordal graph, maximum cardinality search and the chordality
 * test, maximal cliques, induced width and exact treewidth, and the construction of a junction tree with the running
 * intersection property.
 */

import { OmaiError } from '../base/index.ts'
import { maximalCliques as cliques } from '../graph/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  message,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
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
 * import { undirectedGraph, eliminationCost } from 'omai-engine/pgm'
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

/** A greedy criterion as a function: the cost of eliminating a vertex from the current graph (smaller goes first). */
export type EliminationCostFn = (
  graph: UndirectedGraph,
  v: string,
  cardinality: Readonly<Record<string, number>>,
) => number

/** The four named criteria as cost functions, by name. */
const costOf = (heuristic: EliminationHeuristic): EliminationCostFn => {
  return (graph, v, cardinality) => eliminationCost(graph, v, heuristic, cardinality)
}

/** Options of {@link eliminationOrderAlgorithm}. */
export interface EliminationOrderOptions {
  /**
   * The greedy criterion: a name (`minFill` by default, `minDegree`, `minWeight`, `weightedMinFill`) or a cost
   * function, as the `eliminationHeuristics` table of `omai-engine/pgm` makes. Ignored when an order is given.
   */
  readonly heuristic?: EliminationHeuristic | EliminationCostFn
  /** Numbers of states for the weighted criteria (default 2 each). */
  readonly cardinality?: Readonly<Record<string, number>>
  /** A fixed order to follow instead of choosing greedily (it must list the variables, each once). */
  readonly order?: readonly string[]
}

/**
 * The state of {@link eliminationOrderAlgorithm}: the graph as it stands (every elimination connected the neighbours of
 * the vertex it removed), what is left to eliminate, and what the eliminations so far produced. Plain data, so it
 * serialises and a run resumes.
 */
export interface EliminationState {
  /** Neighbours of each remaining vertex (those not yet eliminated), in insertion order. */
  readonly adjacency: Readonly<Record<string, readonly string[]>>
  /** The variables still to eliminate, sorted. */
  readonly remaining: readonly string[]
  /** The variables eliminated so far, in order. */
  readonly order: readonly string[]
  /** The fill-in edges added so far, each as a sorted pair. */
  readonly fillIns: readonly (readonly [string, string])[]
  /** The elimination clique of each step: the vertex and its neighbours at the time. */
  readonly cliques: readonly (readonly string[])[]
  /** The induced width so far: the largest elimination clique minus one. */
  readonly width: number
}

/** The paths {@link eliminationOrderAlgorithm} logs, as a type. */
export type EliminationPath = 'order/vertex' | 'order/cost' | 'order/fill' | 'order/width'

/**
 * What {@link eliminationOrderAlgorithm} logs, by path: the vertex chosen (a message with its cost, its elimination
 * clique and the fill-in edges it adds), the cost of the choice, the number of fill-in edges the step adds, and the
 * induced width so far.
 */
export const eliminationOrderContract: Contract<EliminationState, EliminationPath> = defineContract({
  'order/vertex': message({
    description: 'the vertex eliminated, with its cost, its elimination clique and the fill-in edges it adds',
    payload: {
      vertex: 'the variable',
      cost: 'its cost by the criterion',
      clique: 'the vertex and its neighbours',
      fill: 'new edges',
    },
  }),
  'order/cost': scalar({ description: 'the cost of the vertex chosen, by the criterion' }),
  'order/fill': scalar({ description: 'fill-in edges added by this elimination' }),
  'order/width': scalar({ description: 'the induced width so far (largest elimination clique minus one)' }),
})

/**
 * Greedy elimination ordering as an {@link Algorithm}: each step picks the vertex of least cost (see
 * {@link eliminationCost}; ties go to the first by name), connects its neighbours (the fill-in edges) and removes it,
 * recording its elimination clique and the induced width so far. With `order` the algorithm follows a given order
 * instead and records the same things, which is how {@link triangulate} reads the fill-ins and cliques of an order.
 *
 * @param graph - Neighbours of each variable (for example a moral or interaction graph).
 * @param variables - The variables to order (others stay in the graph).
 * @param options - The criterion (name or cost function), the cardinalities for the weighted criteria, or a fixed order.
 * @returns The algorithm; it is done when every variable is eliminated.
 * @throws OmaiError with code `value` for an order that is not a permutation of the variables.
 * @concept elimination-orderings
 * @example
 * ```ts
 * import { eliminationOrderAlgorithm, undirectedGraph } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const square = undirectedGraph([['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'a']])
 * const alg = eliminationOrderAlgorithm(square, ['a', 'b', 'c', 'd'], { order: ['a', 'b', 'c', 'd'] })
 * const end = runAlgorithm(alg, { maxSteps: 4 }).state
 * [end.fillIns, end.width] // => [[['b', 'd']], 2]
 * ```
 */
export function eliminationOrderAlgorithm(
  graph: UndirectedGraph,
  variables: readonly string[],
  options: EliminationOrderOptions = {},
): Algorithm<EliminationState, EliminationPath> {
  const cardinality = options.cardinality ?? {}
  const heuristic = options.heuristic ?? 'minFill'
  const cost = typeof heuristic === 'function' ? heuristic : costOf(heuristic)
  const todo = [...new Set(variables)].sort()
  const given = options.order
  if (
    given !== undefined &&
    (given.length !== todo.length || new Set(given).size !== todo.length || given.some((v) => !todo.includes(v)))
  )
    throw new OmaiError('value', 'eliminationOrder: the order must list the variables, each once')
  return {
    name: 'elimination-order',
    config: {
      heuristic: given !== undefined ? 'given' : typeof heuristic === 'function' ? 'custom' : heuristic,
      variables: todo.length,
    },
    init: () => {
      const g = copyGraph(graph)
      for (const v of todo) if (!g.has(v)) g.set(v, new Set())
      return { adjacency: adjacencyOf(g), remaining: todo, order: [], fillIns: [], cliques: [], width: 0 }
    },
    step: (s, ctx) => {
      const g = new Map(Object.entries(s.adjacency).map(([k, ns]) => [k, new Set(ns)]))
      let best = given === undefined ? '' : (given[s.order.length] as string)
      let bestCost = best === '' ? Number.POSITIVE_INFINITY : cost(g, best, cardinality)
      if (given === undefined) {
        for (const v of s.remaining) {
          const c = cost(g, v, cardinality)
          if (c < bestCost) {
            best = v
            bestCost = c
          }
        }
      }
      const { clique, fill } = eliminateVertex(g, best)
      const width = Math.max(s.width, clique.length - 1)
      ctx.emit('order/vertex', { vertex: best, cost: bestCost, clique, fill })
      ctx.log('order/cost', bestCost)
      ctx.log('order/fill', fill.length)
      ctx.log('order/width', width)
      return {
        adjacency: adjacencyOf(g),
        remaining: s.remaining.filter((v) => v !== best),
        order: [...s.order, best],
        fillIns: [...s.fillIns, ...fill],
        cliques: [...s.cliques, clique],
        width,
      }
    },
    done: (s) => (s.remaining.length === 0 ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as EliminationState,
  }
}

function adjacencyOf(g: ReadonlyMap<string, ReadonlySet<string>>): Record<string, string[]> {
  return Object.fromEntries([...g].map(([k, ns]) => [k, [...ns]]))
}

/**
 * A greedy elimination ordering: repeatedly eliminate the vertex of least cost (see {@link eliminationCost}), adding
 * its fill-in edges, ties broken by name. This is {@link eliminationOrderAlgorithm} run to the end.
 *
 * @param graph - Neighbours of each variable (for example a moral or interaction graph).
 * @param variables - The variables to order (others stay in the graph).
 * @param heuristic - The greedy criterion (default `minFill`).
 * @param cardinality - Numbers of states for the weighted criteria (default 2 each).
 * @returns The order.
 * @concept elimination-orderings
 * @example
 * ```ts
 * import { eliminationOrder, undirectedGraph } from 'omai-engine/pgm'
 * eliminationOrder(undirectedGraph([['a', 'b'], ['b', 'c']]), ['a', 'b', 'c'], 'minDegree') // => ['a', 'b', 'c']
 * ```
 */
export function eliminationOrder(
  graph: UndirectedGraph,
  variables: readonly string[],
  heuristic: EliminationHeuristic = 'minFill',
  cardinality: Readonly<Record<string, number>> = {},
): string[] {
  const alg = eliminationOrderAlgorithm(graph, variables, { heuristic, cardinality })
  return [...runAlgorithm(alg, { maxSteps: new Set(variables).size }).state.order]
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
 * @throws OmaiError with code `value` when the order misses or repeats a vertex.
 * @concept triangulation
 * @example
 * ```ts
 * import { triangulate, undirectedGraph } from 'omai-engine/pgm'
 * const square = undirectedGraph([['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'a']])
 * triangulate(square, ['a', 'b', 'c', 'd']).fillIns // => [['b', 'd']]
 * ```
 */
export function triangulate(graph: UndirectedGraph, order?: readonly string[]): Triangulation {
  const vertices = [...graph.keys()]
  const ord = order ?? eliminationOrder(graph, vertices, 'minFill')
  if (ord.length !== vertices.length || new Set(ord).size !== ord.length || ord.some((v) => !graph.has(v))) {
    throw new OmaiError('value', 'triangulate: the order must list every vertex once')
  }
  const run = runAlgorithm(eliminationOrderAlgorithm(graph, vertices, { order: ord }), { maxSteps: vertices.length })
  const chordal = copyGraph(graph)
  for (const [a, b] of run.state.fillIns) link(chordal, a, b)
  const fillIns = run.state.fillIns.map((e): [string, string] => [e[0], e[1]])
  const all = run.state.cliques.map((c) => [...c])
  const cliques = all.filter(
    (c, i) =>
      !all.some(
        (d, j) => j !== i && d.length >= c.length && c.every((v) => d.includes(v)) && (d.length > c.length || j < i),
      ),
  )
  return { graph: chordal, fillIns, cliques: cliques.map((c) => [...c].sort()), order: [...ord] }
}

/** The state of {@link maximumCardinalitySearchAlgorithm}: the vertices not yet visited with their weights, and the visit order. */
export interface CardinalitySearchState {
  /** Visited neighbours of each unvisited vertex. */
  readonly weight: Readonly<Record<string, number>>
  /** The vertices visited so far, in order. */
  readonly order: readonly string[]
}

/** The paths {@link maximumCardinalitySearchAlgorithm} logs, as a type. */
export type CardinalitySearchPath = 'mcs/vertex' | 'mcs/weight'

/** What {@link maximumCardinalitySearchAlgorithm} logs: the vertex visited (with the weights it won against) and its weight. */
export const cardinalitySearchContract: Contract<CardinalitySearchState, CardinalitySearchPath> = defineContract({
  'mcs/vertex': message({
    description: 'the vertex visited, with the weights of all unvisited vertices before it',
    payload: { vertex: 'the vertex', weights: 'visited-neighbour counts of the unvisited vertices' },
  }),
  'mcs/weight': scalar({ description: 'the number of visited neighbours of the vertex chosen' }),
})

/**
 * Maximum cardinality search (Tarjan and Yannakakis) as an {@link Algorithm}: each step visits the unvisited vertex with
 * the most visited neighbours (ties by name) and raises the weight of its unvisited neighbours.
 *
 * @param graph - The graph.
 * @returns The algorithm; it is done when every vertex is visited.
 * @concept triangulation
 * @example
 * ```ts
 * import { maximumCardinalitySearchAlgorithm, undirectedGraph } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = undirectedGraph([['a', 'b'], ['b', 'c']])
 * runAlgorithm(maximumCardinalitySearchAlgorithm(g), { maxSteps: 3 }).state.order // => ['a', 'b', 'c']
 * ```
 */
export function maximumCardinalitySearchAlgorithm(
  graph: UndirectedGraph,
): Algorithm<CardinalitySearchState, CardinalitySearchPath> {
  return {
    name: 'maximum-cardinality-search',
    config: { vertices: graph.size },
    init: () => ({ weight: Object.fromEntries([...graph.keys()].map((v) => [v, 0])), order: [] }),
    step: (s, ctx) => {
      let best = ''
      let bw = -1
      for (const v of Object.keys(s.weight).sort()) {
        const w = s.weight[v] ?? 0
        if (w > bw) {
          best = v
          bw = w
        }
      }
      ctx.emit('mcs/vertex', { vertex: best, weights: s.weight })
      ctx.log('mcs/weight', bw)
      const weight = { ...s.weight }
      delete weight[best]
      for (const w of graph.get(best) ?? []) if (w in weight) weight[w] = (weight[w] ?? 0) + 1
      return { weight, order: [...s.order, best] }
    },
    done: (s) => (Object.keys(s.weight).length === 0 ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CardinalitySearchState,
  }
}

/**
 * Maximum cardinality search (Tarjan and Yannakakis): visit next the vertex with the most visited neighbours, ties by
 * name ({@link maximumCardinalitySearchAlgorithm} run to the end). The reverse of the visit order is a perfect
 * elimination order exactly when the graph is chordal.
 *
 * @param graph - The graph.
 * @returns The visit order.
 * @example
 * ```ts
 * import { maximumCardinalitySearch, undirectedGraph } from 'omai-engine/pgm'
 * maximumCardinalitySearch(undirectedGraph([['a', 'b'], ['b', 'c']])) // => ['a', 'b', 'c']
 * ```
 */
export function maximumCardinalitySearch(graph: UndirectedGraph): string[] {
  return [...runAlgorithm(maximumCardinalitySearchAlgorithm(graph), { maxSteps: graph.size }).state.order]
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
 * import { isChordal, undirectedGraph } from 'omai-engine/pgm'
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
 * import { maximalCliques, undirectedGraph } from 'omai-engine/pgm'
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
 * import { inducedWidth, undirectedGraph } from 'omai-engine/pgm'
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
 * @throws OmaiError with code `value` for an exact request on more than 24 vertices.
 * @concept treewidth
 * @example
 * ```ts
 * import { treewidth, undirectedGraph } from 'omai-engine/pgm'
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
  if (n > 24) throw new OmaiError('value', `treewidth: exact treewidth of ${n} vertices is too expensive`)
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
 * @index function Junction tree from a model's moral graph.
 * @example
 * ```ts
 * import { junctionTree, undirectedGraph } from 'omai-engine/pgm'
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
 * import { hasRunningIntersection } from 'omai-engine/pgm'
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
