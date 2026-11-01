/**
 * Graph colouring and cliques: greedy colourings under NetworkX's ordering strategies with a check that a colouring is
 * proper, maximal independent sets, and maximal cliques by the Bron-Kerbosch algorithm with pivoting, from which the
 * maximum clique and clique number follow.
 */

import type { Key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  bronKerboschAlgorithm,
  type ColouringStrategy,
  greedyColouringAlgorithm,
  maximalIndependentSetAlgorithm,
} from './colouring-algorithms.ts'
import type { Graph, NodeId } from './graph.ts'

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
 * @throws MdmlError with code `value` when `'random-sequential'` has no key.
 * @concept graph-colouring
 * @reference D. Brelaz, New methods to color the vertices of a graph, Comm. ACM 22 (1979).
 * @example
 * ```ts
 * import { cycleGraph, greedyColouring } from 'mdml-engine/graph'
 * greedyColouring(cycleGraph(5)) // => [0, 1, 0, 1, 2]
 * greedyColouring(cycleGraph(5), { strategy: 'dsatur' }) // => [0, 1, 0, 1, 2]
 * ```
 */
export function greedyColouring(
  g: Graph,
  options: { readonly strategy?: ColouringStrategy; readonly key?: Key } = {},
): number[] {
  const alg = greedyColouringAlgorithm(g, options)
  return Array.from(runAlgorithm(alg, { maxSteps: g.n + 1 }).state.colours)
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
 * import { isProperColouring, pathGraph } from 'mdml-engine/graph'
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
 * @throws MdmlError with code `value` when the forced nodes are not independent.
 * @concept independent-sets
 * @example
 * ```ts
 * import { maximalIndependentSet, starGraph } from 'mdml-engine/graph'
 * maximalIndependentSet(starGraph(3)) // => [1, 2, 3]
 * maximalIndependentSet(starGraph(3), { nodes: [0] }) // => [0]
 * ```
 */
export function maximalIndependentSet(
  g: Graph,
  options: { readonly nodes?: readonly NodeId[]; readonly key?: Key } = {},
): number[] {
  const alg = maximalIndependentSetAlgorithm(g, options)
  return [...runAlgorithm(alg, { maxSteps: g.n + 1 }).state.chosen]
}

/**
 * All maximal cliques by the Bron-Kerbosch algorithm with Tomita's pivoting (NetworkX find_cliques): each clique sorted,
 * the list in lexicographic order. Self-loops are ignored.
 *
 * @param g - An undirected graph.
 * @returns The maximal cliques.
 * @throws MdmlError with code `value` for a directed graph.
 * @concept cliques
 * @reference C. Bron and J. Kerbosch, Algorithm 457: finding all cliques of an undirected graph, Comm. ACM 16 (1973);
 *   E. Tomita, A. Tanaka and H. Takahashi, The worst-case time complexity for generating all maximal cliques, Theoretical
 *   Computer Science 363 (2006).
 * @example
 * ```ts
 * import { graph, maximalCliques } from 'mdml-engine/graph'
 * maximalCliques(graph([[0, 1], [1, 2], [2, 0], [2, 3]])) // => [[0, 1, 2], [2, 3]]
 * ```
 */
export function maximalCliques(g: Graph): number[][] {
  const { cliques } = runAlgorithm(bronKerboschAlgorithm(g), { maxSteps: Number.MAX_SAFE_INTEGER }).state
  const out = cliques.map((c) => [...c])
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
 * import { graph, maximumClique } from 'mdml-engine/graph'
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
 * import { cliqueNumber, petersenGraph } from 'mdml-engine/graph'
 * cliqueNumber(petersenGraph()) // => 2
 * ```
 */
export function cliqueNumber(g: Graph): number {
  return maximumClique(g).length
}
