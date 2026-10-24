import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { arbitraryGraph } from '../../testing/graphs.ts'
import { maxOf } from '../base/index.ts'
import { key } from '../random/index.ts'
import {
  type ColouringStrategy,
  cliqueNumber,
  completeGraph,
  degree,
  digraph,
  greedyColouring,
  hasEdge,
  isProperColouring,
  maximalCliques,
  maximalIndependentSet,
  starGraph,
} from './index.ts'

const STRATEGIES: ColouringStrategy[] = [
  'largest-first',
  'smallest-last',
  'random-sequential',
  'independent-set',
  'connected-sequential-bfs',
  'connected-sequential-dfs',
  'dsatur',
]

describe('colouring and cliques (properties)', () => {
  it('colours properly with at most max degree + 1 colours, never fewer than the clique number', () => {
    fc.assert(
      fc.property(arbitraryGraph({ maxNodes: 10, density: 0.4 }), fc.nat(100), (g, seed) => {
        const bound = maxOf(degree(g).concat([0])) + 1
        const omega = cliqueNumber(g)
        for (const strategy of STRATEGIES) {
          const c = greedyColouring(g, { strategy, key: key(seed) })
          expect(isProperColouring(g, c)).toBe(true)
          const used = maxOf(c) + 1
          expect(used).toBeLessThanOrEqual(bound)
          expect(used).toBeGreaterThanOrEqual(omega)
        }
      }),
    )
  })

  it('lists maximal cliques that are cliques and cannot grow', () => {
    fc.assert(
      fc.property(arbitraryGraph({ maxNodes: 10, density: 0.45 }), (g) => {
        const cliques = maximalCliques(g)
        const covered = new Set(cliques.flat())
        expect(covered.size).toBe(g.n)
        for (const c of cliques) {
          for (let i = 0; i < c.length; i++)
            for (let j = i + 1; j < c.length; j++) expect(hasEdge(g, c[i] ?? 0, c[j] ?? 0)).toBe(true)
          for (let v = 0; v < g.n; v++) if (!c.includes(v)) expect(c.every((u) => hasEdge(g, u, v))).toBe(false)
        }
      }),
    )
  })

  it('draws maximal independent sets, greedy or random, with forced nodes', () => {
    fc.assert(
      fc.property(arbitraryGraph({ maxNodes: 10, density: 0.35 }), fc.nat(100), (g, seed) => {
        for (const set of [maximalIndependentSet(g), maximalIndependentSet(g, { key: key(seed) })]) {
          const s = new Set(set)
          for (const e of g.edges) expect(s.has(e.source) && s.has(e.target)).toBe(false)
          for (let v = 0; v < g.n; v++) if (!s.has(v)) expect(set.some((u) => hasEdge(g, u, v))).toBe(true)
        }
      }),
    )
    expect(() => maximalIndependentSet(completeGraph(3), { nodes: [0, 1] })).toThrow()
    expect(() => greedyColouring(starGraph(3), { strategy: 'random-sequential' })).toThrow()
    expect(() => maximalCliques(digraph([[0, 1]]))).toThrow()
    expect(
      isProperColouring(
        digraph([
          [0, 1],
          [1, 2],
        ]),
        greedyColouring(
          digraph([
            [0, 1],
            [1, 2],
          ]),
          { strategy: 'connected-sequential-dfs' },
        ),
      ),
    ).toBe(true)
  })
})
