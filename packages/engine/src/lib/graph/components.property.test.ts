import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { arbitraryGraph } from '../../testing/graphs.ts'
import {
  articulationPoints,
  biconnectedComponents,
  bridges,
  condensation,
  connectedComponents,
  graph,
  isConnected,
  isDag,
  isStronglyConnected,
  stronglyConnectedComponents,
  subgraph,
} from './index.ts'

describe('components (properties)', () => {
  it('finds articulation points and bridges whose removal disconnects', () => {
    fc.assert(
      fc.property(arbitraryGraph({ maxNodes: 9, density: 0.35 }), (g) => {
        const base = connectedComponents(g).count
        const cuts = articulationPoints(g)
        for (let v = 0; v < g.n; v++) {
          const rest = Array.from({ length: g.n }, (_, i) => i).filter((i) => i !== v)
          const isolated = !g.edges.some((e) => e.source === v || e.target === v)
          const after = connectedComponents(subgraph(g, rest)).count
          expect(cuts.includes(v)).toBe(after > base - (isolated ? 1 : 0))
        }
        const bs = bridges(g)
        for (const [k, e] of g.edges.entries()) {
          const rest = graph(
            g.edges.filter((_, j) => j !== k).map((x) => [x.source, x.target] as [number, number]),
            { nodes: g.n },
          )
          const isBridge = connectedComponents(rest).count > base
          expect(bs.some(([u, v]) => u === e.source && v === e.target)).toBe(isBridge)
        }
        // Every edge lies in exactly one block.
        const blocks = biconnectedComponents(g)
        expect(blocks.reduce((s, b) => s + b.edges.length, 0)).toBe(g.edges.length)
      }),
    )
  })

  it('agrees between Tarjan and Kosaraju and condenses to a DAG', () => {
    fc.assert(
      fc.property(arbitraryGraph({ directed: true, maxNodes: 9, density: 0.25 }), (g) => {
        const t = stronglyConnectedComponents(g)
        const k = stronglyConnectedComponents(g, { method: 'kosaraju' })
        expect(k.components).toEqual(t.components)
        const c = condensation(g)
        expect(isDag(c.graph)).toBe(true)
        expect(c.graph.n).toBe(t.count)
        expect(isStronglyConnected(g)).toBe(t.count === 1)
      }),
    )
  })

  it('handles small and degenerate cases', () => {
    expect(isConnected(graph([], { nodes: 0 }))).toBe(false)
    expect(stronglyConnectedComponents(graph([[0, 1]])).count).toBe(1)
    expect(() => condensation(graph([[0, 1]]))).toThrow()
    expect(() => articulationPoints(condensation(graph([[0, 1]], { directed: true })).graph)).toThrow()
    expect(biconnectedComponents(graph([[0, 0]], { nodes: 2 }))).toEqual([])
  })
})
