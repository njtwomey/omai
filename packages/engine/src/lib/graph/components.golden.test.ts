import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import {
  articulationPoints,
  biconnectedComponents,
  bridges,
  condensation,
  connectedComponents,
  stronglyConnectedComponents,
} from './index.ts'

const lexical = (a: readonly number[], b: readonly number[]): number => {
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return (a[i] ?? 0) - (b[i] ?? 0)
  return a.length - b.length
}
const sortedSets = (sets: readonly (readonly number[])[]): number[][] =>
  sets.map((s) => [...s].sort((a, b) => a - b)).sort(lexical)

describe('components against networkx and scipy.sparse.csgraph', () => {
  for (const c of loadFixture('graph/components').cases) {
    it(c.name, () => {
      const g = graphFromCase(c.params)
      const p = c.params
      if (c.fn === 'undirected') {
        const cc = connectedComponents(g)
        expect(cc.count).toBe(p['count'])
        expect(sortedSets(cc.components)).toEqual(p['components'])
        expect(articulationPoints(g)).toEqual(p['articulation'])
        expect(bridges(g)).toEqual(p['bridges'])
        const blocks = biconnectedComponents(g)
        expect(sortedSets(blocks.map((b) => b.nodes))).toEqual(p['blocks'])
        const edgeSets = blocks
          .map((b) => b.edges.map((e) => [...e]).sort(lexical))
          .sort((x, y) => lexical(x.flat(), y.flat()))
        const want = (p['blockEdges'] as number[][][]).slice().sort((x, y) => lexical(x.flat(), y.flat()))
        expect(edgeSets).toEqual(want)
      } else {
        for (const method of ['tarjan', 'kosaraju'] as const) {
          const scc = stronglyConnectedComponents(g, { method })
          expect(scc.count).toBe(p['strongCount'])
          expect(sortedSets(scc.components)).toEqual(p['strong'])
        }
        expect(sortedSets(connectedComponents(g).components)).toEqual(p['weak'])
        const cond = condensation(g)
        const members = cond.components
        const edges = cond.graph.edges
          .map((e) => [members[e.source] ?? [], members[e.target] ?? []])
          .sort((x, y) => lexical(x.flat(), y.flat()))
        const want = (p['condensationEdges'] as number[][][]).slice().sort((x, y) => lexical(x.flat(), y.flat()))
        expect(edges).toEqual(want)
      }
    })
  }
})
