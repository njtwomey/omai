import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import {
  balancedTree,
  barbellGraph,
  completeBipartiteGraph,
  completeGraph,
  cycleGraph,
  type Graph,
  gridGraph,
  hypercubeGraph,
  karateClubGraph,
  pathGraph,
  petersenGraph,
  starGraph,
  wheelGraph,
} from './index.ts'

const make: Record<string, (args: readonly number[]) => Graph> = {
  completeGraph: ([n = 0]) => completeGraph(n),
  completeDigraph: ([n = 0]) => completeGraph(n, { directed: true }),
  pathGraph: ([n = 0]) => pathGraph(n),
  cycleGraph: ([n = 0]) => cycleGraph(n),
  starGraph: ([n = 0]) => starGraph(n),
  wheelGraph: ([n = 0]) => wheelGraph(n),
  gridGraph: ([m = 0, n = 0]) => gridGraph(m, n),
  torus: ([m = 0, n = 0]) => gridGraph(m, n, { periodic: true }),
  hypercubeGraph: ([d = 0]) => hypercubeGraph(d),
  completeBipartiteGraph: ([m = 0, n = 0]) => completeBipartiteGraph(m, n),
  balancedTree: ([r = 0, h = 0]) => balancedTree(r, h),
  barbellGraph: ([m = 0, b = 0]) => barbellGraph(m, b),
  petersenGraph: () => petersenGraph(),
  karateClubGraph: () => karateClubGraph(),
}

describe('deterministic generators against networkx', () => {
  for (const c of loadFixture('graph/generators').cases) {
    it(c.name, () => {
      const build = make[c.fn]
      if (build === undefined) throw new Error(`unknown generator ${c.fn}`)
      const g = build(c.params['args'] as number[])
      expect(g.n).toBe(c.params['n'])
      expect(g.directed).toBe(c.params['directed'])
      const edges = g.edges
        .map((e) => [e.source, e.target])
        .sort((a, b) => (a[0] ?? 0) - (b[0] ?? 0) || (a[1] ?? 0) - (b[1] ?? 0))
      expect(edges).toEqual(c.params['edges'])
    })
  }
})
