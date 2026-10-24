import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { maxOf } from '../base/index.ts'
import * as P from './index.ts'

describe('graph machinery against networkx', () => {
  for (const c of loadFixture('pgm/triangulation').cases) {
    it(c.name, () => {
      const g = P.undirectedGraph(c.params['edges'] as [string, string][], c.params['vertices'] as string[])
      expect(P.isChordal(g)).toBe(c.params['chordal'])
      expect(P.maximalCliques(g)).toEqual(c.params['cliques'])
      expect(P.connectedComponents(g)).toEqual(c.params['components'])
      const exact = P.treewidth(g)
      const heuristic = P.treewidth(g, 'heuristic')
      expect(exact).toBeLessThanOrEqual(c.params['minDegreeWidth'] as number)
      expect(exact).toBeLessThanOrEqual(c.params['minFillWidth'] as number)
      expect(heuristic).toBeGreaterThanOrEqual(exact)
      // A triangulation is chordal, its cliques are the maximal cliques of the chordal graph, and its junction tree has
      // the running intersection property.
      const t = P.triangulate(g)
      expect(P.isChordal(t.graph)).toBe(true)
      const byName = (a: readonly string[], b: readonly string[]): number => a.join().localeCompare(b.join())
      expect([...t.cliques].sort(byName)).toEqual([...P.maximalCliques(t.graph)].sort(byName))
      const jt = P.junctionTree(g)
      expect(P.hasRunningIntersection(jt)).toBe(true)
      expect(jt.edges).toHaveLength(jt.cliques.length - 1)
      if (c.params['chordal'] === true) {
        expect(t.fillIns).toHaveLength(0)
        expect(exact).toBe(maxOf((c.params['cliques'] as string[][]).map((q) => q.length)) - 1)
      }
    })
  }
})
