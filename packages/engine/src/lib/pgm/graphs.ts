import { WhiteboxError } from '../base/index.ts'
import { link, mutableGraph } from './_graph.ts'
import type { Factor } from './factor.ts'

/** An undirected graph: the neighbours of each vertex. */
export type UndirectedGraph = ReadonlyMap<string, ReadonlySet<string>>

/** A node of a directed acyclic graph: its name and its parents. */
export interface DagNode {
  readonly name: string
  readonly parents: readonly string[]
}

/**
 * Anything with named nodes and parents: a {@link Dag}, a Bayes network, a linear-Gaussian network or a causal model.
 *
 * @concept graphical-models
 */
export interface DirectedModel {
  readonly nodes: readonly DagNode[]
}

/**
 * A directed acyclic graph as plain data, nodes in a topological order.
 *
 * @concept graphical-models
 */
export interface Dag extends DirectedModel {
  readonly kind: 'dag'
}

/**
 * A directed acyclic graph from edges, validated and ordered topologically (ties kept in first-mention order).
 *
 * @param edges - `[parent, child]` pairs.
 * @param nodes - Extra (or all) nodes, including isolated ones.
 * @returns The graph.
 * @throws WhiteboxError with code `value` when the edges contain a cycle.
 * @concept graphical-models
 * @example
 * ```ts
 * import { dag } from 'whitebox-engine/pgm'
 * dag([['b', 'c'], ['a', 'b']]).nodes.map((n) => n.name) // => ['a', 'b', 'c']
 * ```
 */
export function dag(edges: readonly (readonly [string, string])[], nodes: readonly string[] = []): Dag {
  const names: string[] = []
  const seen = new Set<string>()
  const add = (v: string): void => {
    if (!seen.has(v)) {
      seen.add(v)
      names.push(v)
    }
  }
  for (const v of nodes) add(v)
  for (const [a, b] of edges) {
    add(a)
    add(b)
  }
  const parents = new Map<string, string[]>(names.map((v) => [v, []]))
  for (const [a, b] of edges) {
    const ps = parents.get(b) ?? []
    if (!ps.includes(a)) ps.push(a)
  }
  return { kind: 'dag', nodes: topologicalNodes(names.map((name) => ({ name, parents: parents.get(name) ?? [] }))) }
}

/** Kahn's algorithm, keeping the given order among ready nodes. */
function topologicalNodes<N extends DagNode>(nodes: readonly N[]): N[] {
  const ordered: N[] = []
  const placed = new Set<string>()
  const known = new Set(nodes.map((n) => n.name))
  for (const n of nodes) {
    for (const p of n.parents) if (!known.has(p)) throw new WhiteboxError('value', `${n.name} has unknown parent ${p}`)
  }
  while (ordered.length < nodes.length) {
    const next = nodes.find((n) => !placed.has(n.name) && n.parents.every((p) => placed.has(p)))
    if (next === undefined) throw new WhiteboxError('value', 'the graph has a cycle')
    ordered.push(next)
    placed.add(next.name)
  }
  return ordered
}

/**
 * Nodes in a topological order (parents before children), keeping the given order among ready nodes.
 *
 * @param nodes - The nodes, in any order.
 * @returns The same nodes, ordered.
 * @throws WhiteboxError with code `value` for unknown parents or a cycle.
 * @example
 * ```ts
 * import { topologicalOrder } from 'whitebox-engine/pgm'
 * topologicalOrder([{ name: 'b', parents: ['a'] }, { name: 'a', parents: [] }]).map((n) => n.name) // => ['a', 'b']
 * ```
 */
export function topologicalOrder<N extends DagNode>(nodes: readonly N[]): N[] {
  return topologicalNodes(nodes)
}

/**
 * The edges of a directed model, as `[parent, child]` pairs in node order.
 *
 * @param g - The graph or model.
 * @returns The edges.
 * @example
 * ```ts
 * import { dag, dagEdges } from 'whitebox-engine/pgm'
 * dagEdges(dag([['a', 'b']])) // => [['a', 'b']]
 * ```
 */
export function dagEdges(g: DirectedModel): [string, string][] {
  return g.nodes.flatMap((n) => n.parents.map((p): [string, string] => [p, n.name]))
}

function nodeOf(g: DirectedModel, name: string): DagNode {
  const n = g.nodes.find((x) => x.name === name)
  if (n === undefined) throw new WhiteboxError('value', `unknown variable ${name}`)
  return n
}

/**
 * The ancestors of some nodes, including the nodes themselves.
 *
 * @param g - The graph or model.
 * @param names - The nodes.
 * @returns The ancestral set, sorted.
 * @throws WhiteboxError with code `value` for an unknown node.
 * @example
 * ```ts
 * import { dag, ancestors } from 'whitebox-engine/pgm'
 * ancestors(dag([['a', 'b'], ['b', 'c'], ['d', 'c']]), ['b']) // => ['a', 'b']
 * ```
 */
export function ancestors(g: DirectedModel, names: readonly string[]): string[] {
  const out = new Set<string>()
  const stack = [...names]
  while (stack.length > 0) {
    const v = stack.pop() as string
    if (out.has(v)) continue
    out.add(v)
    stack.push(...nodeOf(g, v).parents)
  }
  return [...out].sort()
}

/**
 * The descendants of a node, including the node itself.
 *
 * @param g - The graph or model.
 * @param name - The node.
 * @returns The descendants, sorted.
 * @throws WhiteboxError with code `value` for an unknown node.
 * @example
 * ```ts
 * import { dag, descendants } from 'whitebox-engine/pgm'
 * descendants(dag([['a', 'b'], ['b', 'c'], ['d', 'c']]), 'b') // => ['b', 'c']
 * ```
 */
export function descendants(g: DirectedModel, name: string): string[] {
  nodeOf(g, name)
  const out = new Set<string>()
  const stack = [name]
  while (stack.length > 0) {
    const v = stack.pop() as string
    if (out.has(v)) continue
    out.add(v)
    for (const n of g.nodes) if (n.parents.includes(v)) stack.push(n.name)
  }
  return [...out].sort()
}

/**
 * An undirected graph from edges.
 *
 * @param edges - Vertex pairs.
 * @param vertices - Extra (or all) vertices, including isolated ones.
 * @returns The neighbours of each vertex.
 * @example
 * ```ts
 * import { undirectedGraph } from 'whitebox-engine/pgm'
 * [...(undirectedGraph([['a', 'b'], ['b', 'c']]).get('b') ?? [])] // => ['a', 'c']
 * ```
 */
export function undirectedGraph(
  edges: readonly (readonly [string, string])[],
  vertices: readonly string[] = [],
): Map<string, Set<string>> {
  const g = mutableGraph(vertices)
  for (const [a, b] of edges) link(g, a, b)
  return g
}

/**
 * The edges of an undirected graph, each once, as sorted pairs in sorted order.
 *
 * @param graph - The graph.
 * @returns The edges.
 * @example
 * ```ts
 * import { undirectedGraph, graphEdges } from 'whitebox-engine/pgm'
 * graphEdges(undirectedGraph([['b', 'a'], ['c', 'b']])) // => [['a', 'b'], ['b', 'c']]
 * ```
 */
export function graphEdges(graph: UndirectedGraph): [string, string][] {
  const out: [string, string][] = []
  for (const [v, ns] of graph) for (const w of ns) if (v < w) out.push([v, w])
  return out.sort((x, y) => x[0].localeCompare(y[0]) || x[1].localeCompare(y[1]))
}

/**
 * The moral graph of a directed model: undirected edges between each node and its parents, and between co-parents.
 *
 * @param g - The network or graph.
 * @returns The neighbours of each variable.
 * @concept moralisation
 * @example
 * ```ts
 * import { dag, moralGraph } from 'whitebox-engine/pgm'
 * [...(moralGraph(dag([['a', 'c'], ['b', 'c']])).get('a') ?? [])].sort() // => ['b', 'c']
 * ```
 */
export function moralGraph(g: DirectedModel): Map<string, Set<string>> {
  const out = mutableGraph(g.nodes.map((n) => n.name))
  for (const n of g.nodes) {
    for (const p of n.parents) link(out, p, n.name)
    for (let i = 0; i < n.parents.length; i++)
      for (let j = i + 1; j < n.parents.length; j++) link(out, n.parents[i] as string, n.parents[j] as string)
  }
  return out
}

/**
 * The interaction graph of a set of factors: an edge between every two variables that share a factor (for the CPTs of
 * a Bayes network this is its moral graph; for a Markov network, its graph).
 *
 * @param factors - The factors.
 * @param variables - Extra vertices, including variables in no factor.
 * @returns The neighbours of each variable.
 * @example
 * ```ts
 * import { factor, interactionGraph } from 'whitebox-engine/pgm'
 * [...(interactionGraph([factor(['a', 'b'], [2, 2], [1, 1, 1, 1])]).get('a') ?? [])] // => ['b']
 * ```
 */
export function interactionGraph(
  factors: readonly Factor[],
  variables: readonly string[] = [],
): Map<string, Set<string>> {
  const g = mutableGraph(variables)
  for (const f of factors) {
    for (const v of f.variables) if (!g.has(v)) g.set(v, new Set())
    for (let i = 0; i < f.variables.length; i++)
      for (let j = i + 1; j < f.variables.length; j++) link(g, f.variables[i] as string, f.variables[j] as string)
  }
  return g
}

/**
 * The connected components of an undirected graph.
 *
 * @param graph - The graph.
 * @returns The components (each sorted), ordered by their smallest vertex.
 * @example
 * ```ts
 * import { undirectedGraph, connectedComponents } from 'whitebox-engine/pgm'
 * connectedComponents(undirectedGraph([['a', 'b']], ['c'])) // => [['a', 'b'], ['c']]
 * ```
 */
export function connectedComponents(graph: UndirectedGraph): string[][] {
  const seen = new Set<string>()
  const out: string[][] = []
  for (const v of [...graph.keys()].sort()) {
    if (seen.has(v)) continue
    const comp: string[] = []
    const stack = [v]
    seen.add(v)
    while (stack.length > 0) {
      const x = stack.pop() as string
      comp.push(x)
      for (const w of graph.get(x) ?? []) {
        if (!seen.has(w)) {
          seen.add(w)
          stack.push(w)
        }
      }
    }
    out.push(comp.sort())
  }
  return out
}

/**
 * Whether two sets of vertices are separated by a third in an undirected graph (every path between them passes through
 * the separator): the global Markov property of Markov networks.
 *
 * @param graph - The graph.
 * @param x - First set.
 * @param y - Second set.
 * @param z - The separator.
 * @returns `true` when separated.
 * @concept markov-properties
 * @example
 * ```ts
 * import { undirectedGraph, separated } from 'whitebox-engine/pgm'
 * const g = undirectedGraph([['a', 'b'], ['b', 'c']])
 * separated(g, ['a'], ['c'], ['b']) // => true
 * separated(g, ['a'], ['c'], []) // => false
 * ```
 */
export function separated(
  graph: UndirectedGraph,
  x: readonly string[],
  y: readonly string[],
  z: readonly string[],
): boolean {
  const blocked = new Set(z)
  const targets = new Set(y)
  const seen = new Set(x.filter((v) => !blocked.has(v)))
  const queue = [...seen]
  while (queue.length > 0) {
    const v = queue.shift() as string
    if (targets.has(v)) return false
    for (const w of graph.get(v) ?? []) {
      if (!seen.has(w) && !blocked.has(w)) {
        seen.add(w)
        queue.push(w)
      }
    }
  }
  return true
}
