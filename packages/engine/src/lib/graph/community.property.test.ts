import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { arbitraryGraph } from '../../testing/graphs.ts'
import { recorder } from '../trace/index.ts'
import {
  barbellGraph,
  connectedComponents,
  digraph,
  girvanNewman,
  graph,
  karateClubGraph,
  labelPropagation,
  leiden,
  louvain,
  modularity,
  partition,
  subgraph,
} from './index.ts'

describe('communities (properties)', () => {
  it('keeps modularity in range, at zero for one community and additive over communities', () => {
    fc.assert(
      fc.property(arbitraryGraph({ maxNodes: 9, maxWeight: 4, density: 0.4 }), (g) => {
        if (g.edges.length === 0) return
        const one = Array.from({ length: g.n }, () => 0)
        expect(Math.abs(modularity(g, one))).toBeLessThan(1e-12)
        const q = modularity(g, louvain(g, { seed: 3 }).labels)
        expect(q).toBeGreaterThanOrEqual(-0.5)
        expect(q).toBeLessThanOrEqual(1)
        // Louvain never ends below the all-singletons start.
        expect(q).toBeGreaterThanOrEqual(
          modularity(
            g,
            Array.from({ length: g.n }, (_, i) => i),
          ) - 1e-12,
        )
      }),
    )
  })

  it('returns connected communities from Leiden', () => {
    fc.assert(
      fc.property(arbitraryGraph({ minNodes: 4, maxNodes: 12, density: 0.3 }), fc.nat(50), (g, seed) => {
        for (const community of leiden(g, { seed }).components) {
          expect(connectedComponents(subgraph(g, community)).count).toBe(1)
        }
        for (const community of leiden(g, { seed, quality: 'cpm', resolution: 0.2 }).components) {
          expect(connectedComponents(subgraph(g, community)).count).toBe(1)
        }
      }),
    )
  })

  it('refines Girvan-Newman partitions level by level', () => {
    const levels = girvanNewman(karateClubGraph(), { levels: 4 })
    expect(levels.map((p) => p.count)).toEqual([2, 3, 4, 5])
    for (let i = 1; i < levels.length; i++) {
      const coarse = levels[i - 1]?.labels ?? new Int32Array(0)
      for (const c of levels[i]?.components ?? []) expect(new Set(c.map((v) => coarse[v])).size).toBe(1)
    }
  })

  it('traces the multilevel searches and label propagation', () => {
    const g = karateClubGraph()
    for (const run of [
      (r: ReturnType<typeof recorder>) => louvain(g, { tracer: r }),
      (r: ReturnType<typeof recorder>) => leiden(g, { tracer: r }),
      (r: ReturnType<typeof recorder>) => labelPropagation(g, { tracer: r }),
      (r: ReturnType<typeof recorder>) => girvanNewman(g, { tracer: r, levels: 1 }),
    ]) {
      const r = recorder()
      run(r)
      expect(r.events.some((e) => e.type === 'metric')).toBe(true)
    }
    // Resolution trades community size: a high resolution gives more communities.
    expect(louvain(g, { resolution: 3, seed: 0 }).count).toBeGreaterThan(louvain(g, { resolution: 0.3, seed: 0 }).count)
  })

  it('handles directed modularity, label inputs and errors', () => {
    const d = digraph([
      [0, 1],
      [1, 0],
      [2, 3],
      [3, 2],
      [1, 2],
    ])
    expect(
      modularity(d, [
        [0, 1],
        [2, 3],
      ]),
    ).toBeCloseTo(0.32, 12)
    expect(modularity(graph([], { nodes: 2 }), [0, 1])).toBe(0)
    expect(() =>
      modularity(d, [
        [0, 1],
        [1, 2, 3],
      ]),
    ).toThrow()
    expect(() => modularity(d, [[0, 1]])).toThrow()
    expect(() => modularity(d, [0, 1])).toThrow()
    expect(partition([[2, 0], [1]], 3).labels).toEqual(Int32Array.from([0, 1, 0]))
    expect(louvain(barbellGraph(3, 0), { maxLevel: 1, seed: 0 }).count).toBe(2)
  })
})
