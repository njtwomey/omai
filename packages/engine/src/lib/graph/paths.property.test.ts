import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { arbitraryGraph } from '../../testing/graphs.ts'
import {
  aStar,
  bellmanFord,
  digraph,
  dijkstra,
  edgeWeight,
  floydWarshall,
  graph,
  johnson,
  kShortestPaths,
  reconstructPath,
  shortestPath,
} from './index.ts'

const weighted = fc.oneof(
  arbitraryGraph({ maxNodes: 8, maxWeight: 9, density: 0.4 }),
  arbitraryGraph({ maxNodes: 8, maxWeight: 9, density: 0.35, directed: true }),
)

describe('shortest paths (properties)', () => {
  it('satisfy the triangle inequality and agree across algorithms', () => {
    fc.assert(
      fc.property(weighted, (g) => {
        const fw = floydWarshall(g)
        const jo = johnson(g)
        const n = g.n
        const d = (i: number, j: number): number => fw.distance.data[i * n + j] ?? Number.NaN
        for (let s = 0; s < n; s++) {
          const dj = dijkstra(g, s)
          const bf = bellmanFord(g, s)
          for (let t = 0; t < n; t++) {
            expect(dj.distance[t]).toBe(d(s, t))
            expect(bf.distance[t]).toBe(d(s, t))
            expect(jo.distance.data[s * n + t]).toBe(d(s, t))
            for (let m = 0; m < n; m++) expect(d(s, t)).toBeLessThanOrEqual(d(s, m) + d(m, t))
            // The reconstructed path has the reported length.
            const path = reconstructPath(dj.predecessor, s, t)
            expect(path === null).toBe(d(s, t) === Number.POSITIVE_INFINITY)
            if (path !== null) {
              let len = 0
              for (let i = 0; i + 1 < path.length; i++)
                len += edgeWeight(g, path[i] ?? 0, path[i + 1] ?? 0) ?? Number.NaN
              expect(len).toBe(d(s, t))
              expect(reconstructPath(fw.predecessor, s, t)?.length ?? 0).toBeGreaterThan(0)
            }
          }
        }
      }),
    )
  })

  it('lists k shortest loopless paths in order, all distinct', () => {
    fc.assert(
      fc.property(weighted, (g) => {
        const t = g.n - 1
        const paths = kShortestPaths(g, 0, t, 5)
        const keys = new Set(paths.map((p) => p.path.join(',')))
        expect(keys.size).toBe(paths.length)
        for (let i = 0; i < paths.length; i++) {
          const p = paths[i]?.path ?? []
          expect(new Set(p).size).toBe(p.length)
          expect(p[0]).toBe(0)
          expect(p[p.length - 1]).toBe(t)
          if (i > 0) expect(paths[i]?.cost ?? 0).toBeGreaterThanOrEqual(paths[i - 1]?.cost ?? 0)
        }
        if (paths.length > 0) expect(paths[0]?.cost).toBe(dijkstra(g, 0).distance[t])
        const a = aStar(g, 0, t)
        expect(a.distance).toBe(dijkstra(g, 0).distance[t])
      }),
    )
  })

  it('reports negative weights and cycles', () => {
    expect(() => dijkstra(digraph([[0, 1, -1]]), 0)).toThrow()
    expect(() => aStar(digraph([[0, 1, -1]]), 0, 1)).toThrow()
    expect(() => bellmanFord(graph([[0, 1, -1]]), 0)).toThrow()
    expect(() => floydWarshall(digraph([[0, 0, -1]]))).toThrow()
    expect(
      shortestPath(
        digraph([
          [0, 1, 3],
          [1, 2, -2],
          [0, 2, 2],
        ]),
        0,
        2,
      ),
    ).toEqual({ path: [0, 1, 2], cost: 1 })
    expect(shortestPath(graph([[0, 1]], { nodes: 3 }), 0, 2)).toBeNull()
    expect(
      shortestPath(
        graph([
          [0, 1, 5],
          [1, 2, 5],
          [0, 2, 20],
        ]),
        0,
        2,
        { weighted: false },
      ),
    ).toEqual({
      path: [0, 2],
      cost: 1,
    })
    expect(reconstructPath([-1, 0], 0, 0)).toEqual([0])
    expect(kShortestPaths(graph([[0, 1]], { nodes: 3 }), 0, 2, 3)).toEqual([])
  })
})
