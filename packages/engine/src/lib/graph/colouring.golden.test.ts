import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import { maxOf } from '../base/index.ts'
import { key } from '../random/index.ts'
import {
  type ColouringStrategy,
  cliqueNumber,
  degree,
  greedyColouring,
  isProperColouring,
  maximalCliques,
  maximalIndependentSet,
  maximumClique,
} from './index.ts'

describe('colouring and cliques against networkx', () => {
  for (const c of loadFixture('graph/colouring').cases) {
    it(c.name, () => {
      const g = graphFromCase(c.params)
      const p = c.params
      expect(greedyColouring(g)).toEqual(p['largestFirst'])
      expect(greedyColouring(g, { strategy: 'dsatur' })).toEqual(p['dsatur'])
      const maxDegree = maxOf(degree(g))
      const strategies: ColouringStrategy[] = [
        'smallest-last',
        'independent-set',
        'connected-sequential-bfs',
        'connected-sequential-dfs',
        'random-sequential',
      ]
      for (const strategy of strategies) {
        const colours = greedyColouring(g, { strategy, key: key(3) })
        expect(isProperColouring(g, colours)).toBe(true)
        expect(maxOf(colours) + 1).toBeLessThanOrEqual(maxDegree + 1)
      }
      // Smallest-last never needs more than the degeneracy + 1 colours, which NetworkX's run also respects.
      const counts = p['counts'] as Record<string, number>
      expect(maxOf(greedyColouring(g, { strategy: 'smallest-last' })) + 1).toBeLessThanOrEqual(
        (counts['smallest-last'] ?? 0) + 1,
      )
      expect(maximalCliques(g)).toEqual(p['cliques'])
      expect(cliqueNumber(g)).toBe(p['cliqueNumber'])
      expect(maximumClique(g).length).toBe(p['cliqueNumber'])
      const mis = maximalIndependentSet(g)
      const inSet = new Set(mis)
      expect(g.edges.every((e) => !(inSet.has(e.source) && inSet.has(e.target)))).toBe(true)
      for (let v = 0; v < g.n; v++) {
        if (inSet.has(v)) continue
        expect(
          g.edges.some((e) => (e.source === v && inSet.has(e.target)) || (e.target === v && inSet.has(e.source))),
        ).toBe(true)
      }
    })
  }
})
