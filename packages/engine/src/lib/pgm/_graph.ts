// Private helpers for the mutable undirected graphs used inside pgm (not re-exported).
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
