// Neighbour sets of a graph ignoring direction and self-loops, shared by the colouring and clique algorithms. Not
// re-exported.
import type { Graph } from './graph.ts'

/** Neighbour sets ignoring direction and self-loops. */
export function neighbourSets(g: Graph): Set<number>[] {
  return Array.from({ length: g.n }, (_, u) => {
    const s = new Set<number>()
    for (const a of g.directed ? [g.out, g.in] : [g.out]) {
      for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) if (a.indices[k] !== u) s.add(a.indices[k] ?? 0)
    }
    return s
  })
}
