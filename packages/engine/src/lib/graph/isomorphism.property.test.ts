import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { arbitraryGraph } from '../../testing/graphs.ts'
import { key, permutation } from '../random/index.ts'
import { recorder } from '../trace/index.ts'
import {
  cycleGraph,
  digraph,
  type Graph,
  graph,
  hasEdge,
  isIsomorphic,
  pathGraph,
  vf2,
  weisfeilerLehman,
  wlHash,
} from './index.ts'

function permuted(g: Graph, seed: number): { readonly h: Graph; readonly p: number[] } {
  const p = Array.from(permutation(key(seed), g.n).data)
  const h = graph(
    g.edges.map((e) => [p[e.source] ?? 0, p[e.target] ?? 0, e.weight] as [number, number, number]),
    { nodes: g.n, directed: g.directed },
  )
  return { h, p }
}

describe('isomorphism (properties)', () => {
  it('recognises relabelled graphs, with equal WL hashes and valid mappings', () => {
    fc.assert(
      fc.property(
        fc.oneof(
          arbitraryGraph({ maxNodes: 8, density: 0.4 }),
          arbitraryGraph({ maxNodes: 7, density: 0.3, directed: true }),
        ),
        fc.nat(1000),
        (g, seed) => {
          const { h, p } = permuted(g, seed)
          expect(isIsomorphic(g, h)).toBe(true)
          expect(wlHash(g)).toBe(wlHash(h))
          const mapping = vf2(h, g).next().value as number[]
          for (const e of g.edges) expect(hasEdge(h, mapping[e.source] ?? -1, mapping[e.target] ?? -1)).toBe(true)
          // The permutation itself is one of the isomorphisms.
          expect(Array.from(vf2(h, g)).some((m) => m.every((x, i) => x === p[i]))).toBe(true)
        },
      ),
    )
  })

  it('tells apart what 1-WL tells apart, and refines to a stable colouring', () => {
    expect(
      isIsomorphic(
        cycleGraph(6),
        graph([
          [0, 1],
          [1, 2],
          [2, 0],
          [3, 4],
          [4, 5],
          [5, 3],
        ]),
      ),
    ).toBe(false)
    expect(wlHash(cycleGraph(6))).toBe(
      wlHash(
        graph([
          [0, 1],
          [1, 2],
          [2, 0],
          [3, 4],
          [4, 5],
          [5, 3],
        ]),
      ),
    )
    expect(wlHash(pathGraph(4))).not.toBe(
      wlHash(
        graph([
          [0, 1],
          [0, 2],
          [0, 3],
        ]),
      ),
    )
    const rec = recorder()
    const r = weisfeilerLehman(pathGraph(7), { tracer: rec })
    expect(r.count).toBe(4)
    expect(r.history[0]?.length).toBe(7)
    expect(rec.events.some((e) => e.type === 'metric')).toBe(true)
    // Edge labels distinguish otherwise identical graphs.
    expect(wlHash(pathGraph(3), { edgeLabels: ['a', 'b'] })).not.toBe(wlHash(pathGraph(3), { edgeLabels: ['a', 'a'] }))
    expect(() => Array.from(vf2(digraph([[0, 1]]), pathGraph(2)))).toThrow()
    expect(Array.from(vf2(pathGraph(3), pathGraph(4)))).toEqual([])
    expect(Array.from(vf2(cycleGraph(5), cycleGraph(5), 'isomorphism', { limit: 3 })).length).toBe(3)
    expect(
      Array.from(vf2(pathGraph(3), pathGraph(3), 'isomorphism', { nodeMatch: (p, h) => (p === 1) === (h === 1) }))
        .length,
    ).toBe(2)
  })
})
