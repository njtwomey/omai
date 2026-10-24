import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { arbitraryGraph } from '../../testing/graphs.ts'
import {
  bipartiteMatching,
  completeBipartiteGraph,
  cycleGraph,
  digraph,
  graph,
  maxFlow,
  minCostFlow,
  minCut,
} from './index.ts'

describe('flows (properties)', () => {
  it('max-flow equals min-cut, for Dinic and Edmonds-Karp, directed and undirected', () => {
    fc.assert(
      fc.property(
        fc.oneof(
          arbitraryGraph({ minNodes: 3, maxNodes: 9, maxWeight: 7, density: 0.4, directed: true }),
          arbitraryGraph({ minNodes: 3, maxNodes: 9, maxWeight: 7, density: 0.4 }),
        ),
        (g) => {
          const s = 0
          const t = g.n - 1
          const dinic = maxFlow(g, s, t)
          const ek = maxFlow(g, s, t, { method: 'edmonds-karp' })
          const cut = minCut(g, s, t)
          expect(ek.value).toBe(dinic.value)
          expect(cut.value).toBe(dinic.value)
          // The cut's capacity, read off the original edges, equals the flow (max-flow min-cut theorem).
          const side = new Set(cut.partition[0])
          let capacity = 0
          for (const e of g.edges) {
            if (side.has(e.source) && !side.has(e.target)) capacity += e.weight
            else if (!g.directed && side.has(e.target) && !side.has(e.source)) capacity += e.weight
          }
          expect(capacity).toBe(dinic.value)
          expect(side.has(s) && !side.has(t)).toBe(true)
        },
      ),
    )
  })

  it('matches bipartite matching to unit-capacity max flow', () => {
    fc.assert(
      fc.property(arbitraryGraph({ minNodes: 2, maxNodes: 10, density: 0.3 }), (h) => {
        // Split nodes: even on the left, odd on the right; keep the edges between the sides.
        const edges = h.edges
          .filter((e) => e.source % 2 !== e.target % 2)
          .map((e) => [e.source, e.target] as [number, number])
        const g = graph(edges, { nodes: h.n })
        const left = Array.from({ length: h.n }, (_, i) => i).filter((i) => i % 2 === 0)
        const m = bipartiteMatching(g, { left })
        const flowEdges: [number, number][] = [
          ...left.map((v): [number, number] => [h.n, v]),
          ...edges.map(([u, v]): [number, number] => (u % 2 === 0 ? [u, v] : [v, u])),
          ...Array.from({ length: h.n }, (_, i) => i)
            .filter((i) => i % 2 === 1)
            .map((v): [number, number] => [v, h.n + 1]),
        ]
        const f = maxFlow(digraph(flowEdges, { nodes: h.n + 2 }), h.n, h.n + 1)
        expect(m.size).toBe(f.value)
      }),
    )
  })

  it('handles infeasible, unbalanced and invalid inputs', () => {
    expect(() => maxFlow(graph([[0, 1]]), 0, 0)).toThrow()
    expect(() => maxFlow(digraph([[0, 1]]), 0, 1, { capacity: [-1] })).toThrow()
    expect(() => maxFlow(digraph([[0, 1]]), 0, 1, { capacity: [Number.POSITIVE_INFINITY] })).toThrow()
    expect(() => minCostFlow(graph([[0, 1]]), [-1, 1])).toThrow()
    expect(() => minCostFlow(digraph([[0, 1]]), [-1, 2])).toThrow()
    expect(() => minCostFlow(digraph([[0, 1]]), [-2, 2])).toThrow()
    expect(() => minCostFlow(digraph([[0, 1]]), [1])).toThrow()
    expect(() =>
      minCostFlow(digraph([[0, 1]]), [-1, 1], { cost: [-1], capacity: [Number.POSITIVE_INFINITY] }),
    ).toThrow()
    expect(() => bipartiteMatching(cycleGraph(3))).toThrow()
    expect(() => bipartiteMatching(graph([[0, 1]]), { left: [0, 1] })).toThrow()
    expect(bipartiteMatching(completeBipartiteGraph(3, 5)).size).toBe(3)
    // A negative-cost cycle with finite capacity is saturated.
    const r = minCostFlow(
      digraph([
        [0, 1, 2],
        [1, 0, 2],
      ]),
      [0, 0],
      { cost: [-1, -1] },
    )
    expect(r.cost).toBe(-4)
    expect(Array.from(r.flow)).toEqual([2, 2])
  })
})
