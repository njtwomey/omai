/**
 * Graph colouring and cliques: greedy colourings under NetworkX's ordering strategies with a check that a colouring is
 * proper, maximal independent sets, and maximal cliques by the Bron-Kerbosch algorithm with pivoting, from which the
 * maximum clique and clique number follow.
 */

import { WhiteboxError } from '../base/index.ts'
import { type Key, permutation } from '../random/index.ts'
import { connectedComponents } from './components.ts'
import { degree, type Graph, type NodeId, nodeIndex } from './graph.ts'
import { bfs, dfs } from './traversal.ts'

/** The order in which {@link greedyColouring} colours the nodes (NetworkX greedy_color strategies). */
export type ColouringStrategy =
  | 'largest-first'
  | 'smallest-last'
  | 'random-sequential'
  | 'independent-set'
  | 'connected-sequential-bfs'
  | 'connected-sequential-dfs'
  | 'dsatur'

/** Neighbour sets ignoring direction and self-loops. */
function neighbourSets(g: Graph): Set<number>[] {
  return Array.from({ length: g.n }, (_, u) => {
    const s = new Set<number>()
    for (const a of g.directed ? [g.out, g.in] : [g.out]) {
      for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) if (a.indices[k] !== u) s.add(a.indices[k] ?? 0)
    }
    return s
  })
}

function smallestFree(used: ReadonlySet<number>): number {
  let c = 0
  while (used.has(c)) c++
  return c
}

/** Repeatedly take the node of least degree in what remains (ties to the smallest index), dropping its neighbours. */
function minDegreeIndependent(nb: readonly Set<number>[], within: ReadonlySet<number>): number[] {
  const remaining = new Set(within)
  const out: number[] = []
  while (remaining.size > 0) {
    let best = -1
    let bestDegree = Number.POSITIVE_INFINITY
    for (const v of [...remaining].sort((p, q) => p - q)) {
      let d = 0
      for (const u of nb[v] ?? []) if (remaining.has(u)) d++
      if (d < bestDegree) {
        best = v
        bestDegree = d
      }
    }
    out.push(best)
    remaining.delete(best)
    for (const u of nb[best] ?? []) remaining.delete(u)
  }
  return out
}

function smallestLastOrder(nb: readonly Set<number>[]): number[] {
  const n = nb.length
  const deg = Int32Array.from(nb, (s) => s.size)
  const removed = new Uint8Array(n)
  const order: number[] = []
  for (let i = 0; i < n; i++) {
    let best = -1
    for (let v = 0; v < n; v++) if (removed[v] === 0 && (best < 0 || (deg[v] ?? 0) < (deg[best] ?? 0))) best = v
    removed[best] = 1
    order.push(best)
    for (const u of nb[best] ?? []) if (removed[u] === 0) deg[u] = (deg[u] ?? 0) - 1
  }
  return order.reverse()
}

/**
 * A greedy colouring (NetworkX greedy_color): nodes are coloured in the strategy's order, each with the smallest
 * colour not used by its neighbours. `'largest-first'` (default) orders by degree, `'smallest-last'` repeatedly
 * removes a node of least degree and colours in reverse, `'independent-set'` peels off maximal independent sets,
 * `'connected-sequential-bfs'`/`'-dfs'` follow a traversal of each component, `'random-sequential'` shuffles (with a
 * key), and `'dsatur'` (NetworkX saturation_largest_first, Brelaz 1979) always colours next the node with most
 * distinctly coloured neighbours, ties to the higher degree. Ties go to the smaller index, as in NetworkX. Edge
 * directions are ignored.
 *
 * @param g - The graph.
 * @param options - The strategy, and a key for `'random-sequential'`.
 * @returns A colour 0, 1, ... per node.
 * @throws WhiteboxError with code `value` when `'random-sequential'` has no key.
 * @concept graph-colouring
 * @reference D. Brelaz, New methods to color the vertices of a graph, Comm. ACM 22 (1979).
 * @example
 * ```ts
 * import { cycleGraph, greedyColouring } from 'whitebox-engine/graph'
 * greedyColouring(cycleGraph(5)) // => [0, 1, 0, 1, 2]
 * greedyColouring(cycleGraph(5), { strategy: 'dsatur' }) // => [0, 1, 0, 1, 2]
 * ```
 */
export function greedyColouring(
  g: Graph,
  options: { readonly strategy?: ColouringStrategy; readonly key?: Key } = {},
): number[] {
  const strategy = options.strategy ?? 'largest-first'
  const nb = neighbourSets(g)
  const n = g.n
  const colours = new Int32Array(n).fill(-1)
  const paint = (u: number): void => {
    const used = new Set<number>()
    for (const v of nb[u] ?? []) if ((colours[v] ?? -1) >= 0) used.add(colours[v] ?? 0)
    colours[u] = smallestFree(used)
  }
  if (strategy === 'dsatur') {
    const saturation: Set<number>[] = Array.from({ length: n }, () => new Set())
    const deg = degree(g)
    for (let i = 0; i < n; i++) {
      let best = -1
      for (let v = 0; v < n; v++) {
        if ((colours[v] ?? -1) >= 0) continue
        if (
          best < 0 ||
          (saturation[v]?.size ?? 0) > (saturation[best]?.size ?? 0) ||
          ((saturation[v]?.size ?? 0) === (saturation[best]?.size ?? 0) && (deg[v] ?? 0) > (deg[best] ?? 0))
        ) {
          best = v
        }
      }
      paint(best)
      for (const u of nb[best] ?? []) saturation[u]?.add(colours[best] ?? 0)
    }
    return Array.from(colours)
  }
  let order: number[]
  if (strategy === 'largest-first') {
    const deg = degree(g)
    order = Array.from({ length: n }, (_, v) => v).sort((p, q) => (deg[q] ?? 0) - (deg[p] ?? 0) || p - q)
  } else if (strategy === 'smallest-last') order = smallestLastOrder(nb)
  else if (strategy === 'random-sequential') {
    if (options.key === undefined) throw new WhiteboxError('value', "greedyColouring: 'random-sequential' needs a key")
    order = Array.from(permutation(options.key, n).data)
  } else if (strategy === 'independent-set') {
    order = []
    const remaining = new Set(Array.from({ length: n }, (_, v) => v))
    while (remaining.size > 0) {
      const set = minDegreeIndependent(nb, remaining)
      for (const v of set) {
        remaining.delete(v)
        order.push(v)
      }
    }
  } else {
    order = []
    for (const component of connectedComponents(g).components) {
      const source = component[0] ?? 0
      order.push(
        ...(strategy === 'connected-sequential-bfs'
          ? bfs(g, source, { direction: 'both' }).order
          : dfs(undirectedView(g, nb), source).preorder),
      )
    }
  }
  for (const u of order) paint(u)
  return Array.from(colours)
}

/** The graph itself when undirected; otherwise the same nodes with undirected neighbour sets, for traversal. */
function undirectedView(g: Graph, nb: readonly Set<number>[]): Graph {
  if (!g.directed) return g
  const indptr = new Int32Array(g.n + 1)
  const indices: number[] = []
  for (let u = 0; u < g.n; u++) {
    indices.push(...[...(nb[u] ?? [])].sort((p, q) => p - q))
    indptr[u + 1] = indices.length
  }
  const adj = {
    indptr,
    indices: Int32Array.from(indices),
    weights: new Float64Array(indices.length).fill(1),
    edges: new Int32Array(indices.length),
  }
  return { ...g, directed: false, out: adj, in: adj }
}

/**
 * Whether a colouring is proper: no edge joins two nodes of the same colour (self-loops ignored).
 *
 * @param g - The graph.
 * @param colours - A colour per node.
 * @returns True when proper.
 * @concept graph-colouring
 * @example
 * ```ts
 * import { isProperColouring, pathGraph } from 'whitebox-engine/graph'
 * [isProperColouring(pathGraph(3), [0, 1, 0]), isProperColouring(pathGraph(3), [0, 0, 1])] // => [true, false]
 * ```
 */
export function isProperColouring(g: Graph, colours: ArrayLike<number>): boolean {
  return g.edges.every((e) => e.source === e.target || colours[e.source] !== colours[e.target])
}

/**
 * A maximal independent set (NetworkX maximal_independent_set): no two of its nodes are adjacent and no node can be
 * added. Without a key, the greedy minimum-degree rule (repeatedly take a node of least remaining degree); with a key,
 * the nodes are considered in a random order. Nodes given in `nodes` are forced into the set.
 *
 * @param g - The graph (directions ignored).
 * @param options - Nodes to include, and a key for a random set.
 * @returns The set, ascending.
 * @throws WhiteboxError with code `value` when the forced nodes are not independent.
 * @concept independent-sets
 * @example
 * ```ts
 * import { maximalIndependentSet, starGraph } from 'whitebox-engine/graph'
 * maximalIndependentSet(starGraph(3)) // => [1, 2, 3]
 * maximalIndependentSet(starGraph(3), { nodes: [0] }) // => [0]
 * ```
 */
export function maximalIndependentSet(
  g: Graph,
  options: { readonly nodes?: readonly NodeId[]; readonly key?: Key } = {},
): number[] {
  const nb = neighbourSets(g)
  const forced = (options.nodes ?? []).map((v) => nodeIndex(g, v))
  const chosen = new Set<number>()
  const blocked = new Set<number>()
  for (const v of forced) {
    if (blocked.has(v)) throw new WhiteboxError('value', 'maximalIndependentSet: the given nodes are not independent')
    chosen.add(v)
    blocked.add(v)
    for (const u of nb[v] ?? []) blocked.add(u)
  }
  const free = new Set(Array.from({ length: g.n }, (_, v) => v).filter((v) => !blocked.has(v)))
  if (options.key !== undefined) {
    for (const v of Array.from(permutation(options.key, g.n).data)) {
      if (!free.has(v)) continue
      chosen.add(v)
      free.delete(v)
      for (const u of nb[v] ?? []) free.delete(u)
    }
  } else for (const v of minDegreeIndependent(nb, free)) chosen.add(v)
  return [...chosen].sort((p, q) => p - q)
}

/**
 * All maximal cliques by the Bron-Kerbosch algorithm with Tomita's pivoting (NetworkX find_cliques): each clique sorted,
 * the list in lexicographic order. Self-loops are ignored.
 *
 * @param g - An undirected graph.
 * @returns The maximal cliques.
 * @throws WhiteboxError with code `value` for a directed graph.
 * @concept cliques
 * @reference C. Bron and J. Kerbosch, Algorithm 457: finding all cliques of an undirected graph, Comm. ACM 16 (1973);
 *   E. Tomita, A. Tanaka and H. Takahashi, The worst-case time complexity for generating all maximal cliques, Theoretical
 *   Computer Science 363 (2006).
 * @example
 * ```ts
 * import { graph, maximalCliques } from 'whitebox-engine/graph'
 * maximalCliques(graph([[0, 1], [1, 2], [2, 0], [2, 3]])) // => [[0, 1, 2], [2, 3]]
 * ```
 */
export function maximalCliques(g: Graph): number[][] {
  if (g.directed) throw new WhiteboxError('value', 'maximalCliques: the graph must be undirected')
  const nb = neighbourSets(g)
  const out: number[][] = []
  // Iterative Bron-Kerbosch: frames of (R, P, X, the candidates P \ N(pivot) still to branch on).
  const stack: { r: number[]; p: Set<number>; x: Set<number>; todo: number[] }[] = []
  const push = (r: number[], p: Set<number>, x: Set<number>): void => {
    if (p.size === 0) {
      if (x.size === 0) out.push([...r].sort((a, b) => a - b))
      return
    }
    let pivot = -1
    let most = -1
    for (const u of [...p, ...x]) {
      let c = 0
      for (const v of nb[u] ?? []) if (p.has(v)) c++
      if (c > most) {
        most = c
        pivot = u
      }
    }
    const todo = [...p].filter((v) => !(nb[pivot]?.has(v) ?? false)).sort((a, b) => a - b)
    stack.push({ r, p, x, todo })
  }
  push([], new Set(Array.from({ length: g.n }, (_, v) => v)), new Set())
  while (stack.length > 0) {
    const frame = stack[stack.length - 1] as { r: number[]; p: Set<number>; x: Set<number>; todo: number[] }
    const v = frame.todo.shift()
    if (v === undefined) {
      stack.pop()
      continue
    }
    const nv = nb[v] ?? new Set<number>()
    push(
      [...frame.r, v],
      new Set([...frame.p].filter((u) => nv.has(u))),
      new Set([...frame.x].filter((u) => nv.has(u))),
    )
    frame.p.delete(v)
    frame.x.add(v)
  }
  const lexical = (a: readonly number[], b: readonly number[]): number => {
    for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return (a[i] ?? 0) - (b[i] ?? 0)
    return a.length - b.length
  }
  return out.sort(lexical)
}

/**
 * A maximum clique: a largest maximal clique (the lexicographically first among the largest), by Bron-Kerbosch.
 *
 * @param g - An undirected graph.
 * @returns The clique, ascending.
 * @concept cliques
 * @example
 * ```ts
 * import { graph, maximumClique } from 'whitebox-engine/graph'
 * maximumClique(graph([[0, 1], [1, 2], [2, 0], [2, 3], [3, 4]])) // => [0, 1, 2]
 * ```
 */
export function maximumClique(g: Graph): number[] {
  let best: number[] = []
  for (const c of maximalCliques(g)) if (c.length > best.length) best = c
  return best
}

/**
 * The clique number: the size of a maximum clique (NetworkX graph_clique_number, max(len(c) for c in find_cliques)).
 *
 * @param g - An undirected graph.
 * @returns The clique number (0 for the empty graph).
 * @concept cliques
 * @example
 * ```ts
 * import { cliqueNumber, petersenGraph } from 'whitebox-engine/graph'
 * cliqueNumber(petersenGraph()) // => 2
 * ```
 */
export function cliqueNumber(g: Graph): number {
  return maximumClique(g).length
}
