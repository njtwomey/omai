// Private helpers for the mutable undirected graphs used inside pgm (not re-exported).
import { type Graph, graph } from '../graph/index.ts'
import type { UndirectedGraph } from './graphs.ts'

/** An empty graph on some vertices. */
export function mutableGraph(vertices: Iterable<string>): Map<string, Set<string>> {
  return new Map([...vertices].map((v) => [v, new Set<string>()]))
}

/** Add an undirected edge (and its endpoints); self-loops are ignored. */
export function link(g: Map<string, Set<string>>, a: string, b: string): void {
  if (a === b) return
  if (!g.has(a)) g.set(a, new Set())
  if (!g.has(b)) g.set(b, new Set())
  g.get(a)?.add(b)
  g.get(b)?.add(a)
}

/** A deep copy of a graph. */
export function copyGraph(graph: UndirectedGraph): Map<string, Set<string>> {
  return new Map([...graph].map(([k, v]) => [k, new Set(v)]))
}

/**
 * A pgm graph as a `graph.Graph` for the graph domain's algorithms: vertex `i` is `names[i]`, the vertex names sorted
 * (code-unit order), so sorting by index is sorting by name. Self-loops are dropped.
 */
export function indexedGraph(u: UndirectedGraph): { readonly g: Graph; readonly names: readonly string[] } {
  const names = [...u.keys()].sort()
  const index = new Map(names.map((v, i) => [v, i]))
  const edges: [number, number][] = []
  for (const [a, ns] of u) {
    const i = index.get(a) ?? 0
    for (const b of ns) {
      const j = index.get(b)
      // Each edge once, as (smaller, larger); graph merges the repeat from the other endpoint's list.
      if (j !== undefined && j !== i) edges.push(i < j ? [i, j] : [j, i])
    }
  }
  return { g: graph(edges, { nodes: names.length }), names }
}
