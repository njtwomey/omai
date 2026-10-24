import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { arbitraryGraph } from '../../testing/graphs.ts'
import { key, permutation } from '../random/index.ts'
import {
  balancedTree,
  connectedComponents,
  cycleGraph,
  dijkstra,
  eulerTour,
  graph,
  isForest,
  isTree,
  lcaIndex,
  lowestCommonAncestor,
  minimumSpanningTree,
  reconstructPath,
  rootTree,
  UnionFind,
} from './index.ts'

describe('trees (properties)', () => {
  it('finds minimum spanning forests: the cut and cycle properties hold, and no other forest is lighter', () => {
    fc.assert(
      fc.property(arbitraryGraph({ maxNodes: 9, maxWeight: 6, density: 0.45 }), fc.nat(1000), (g, seed) => {
        const t = minimumSpanningTree(g)
        for (const algorithm of ['prim', 'boruvka'] as const) {
          expect(minimumSpanningTree(g, { algorithm }).weight).toBe(t.weight)
        }
        expect(t.edges.length).toBe(g.n - connectedComponents(g).count)
        expect(isForest(t.graph) || g.n === 0).toBe(true)
        // Cycle property: every non-tree edge is at least as heavy as each edge on the tree path it closes.
        const inTree = new Set(t.edges.map((e) => `${e.source},${e.target}`))
        for (const e of g.edges) {
          if (inTree.has(`${e.source},${e.target}`)) continue
          const path = reconstructPath(dijkstra(t.graph, e.source, { weighted: false }).predecessor, e.source, e.target)
          if (path === null) continue
          for (let i = 0; i + 1 < path.length; i++) {
            const a = Math.min(path[i] ?? 0, path[i + 1] ?? 0)
            const b = Math.max(path[i] ?? 0, path[i + 1] ?? 0)
            const w = t.edges.find((x) => x.source === a && x.target === b)?.weight ?? Number.NaN
            expect(w).toBeLessThanOrEqual(e.weight)
          }
        }
        // Any spanning forest built in a random edge order weighs at least as much.
        const order = Array.from(permutation(key(seed), g.edges.length).data)
        const uf = new UnionFind(g.n)
        let other = 0
        for (const k of order) {
          const e = g.edges[k]
          if (e !== undefined && uf.union(e.source, e.target)) other += e.weight
        }
        expect(t.weight).toBeLessThanOrEqual(other)
        expect(minimumSpanningTree(g, { maximum: true }).weight).toBeGreaterThanOrEqual(other)
      }),
    )
  })

  it('answers LCA queries by binary lifting as the naive walk does, and tours each tree', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 30 }), fc.nat(1000), (n, seed) => {
        // A random recursive tree: node i attaches to a uniform earlier node.
        const p = Array.from(permutation(key(seed), Math.max(n, 2)).data)
        const edges = Array.from({ length: n - 1 }, (_, i) => [(p[i] ?? 0) % (i + 1), i + 1] as [number, number])
        const tree = rootTree(graph(edges, { nodes: n }), seed % n)
        const index = lcaIndex(tree)
        for (let a = 0; a < n; a += 3)
          for (let b = 0; b < n; b += 2) expect(index.lca(a, b)).toBe(lowestCommonAncestor(tree, a, b))
        expect(eulerTour(tree).length).toBe(2 * n - 1)
        expect(isTree(graph(edges, { nodes: n }))).toBe(true)
      }),
    )
  })

  it('rejects cycles, directed graphs and nodes outside the tree', () => {
    expect(() => rootTree(cycleGraph(4), 0)).toThrow()
    expect(() => rootTree(graph([[0, 1]], { directed: true }), 0)).toThrow()
    expect(() => minimumSpanningTree(graph([[0, 1]], { directed: true }))).toThrow()
    const t = rootTree(graph([[0, 1]], { nodes: 3 }), 0)
    expect(() => lowestCommonAncestor(t, 0, 2)).toThrow()
    expect(() => lcaIndex(t).lca(2, 0)).toThrow()
    expect(isTree(cycleGraph(3))).toBe(false)
    expect(rootTree(balancedTree(2, 1), 0).distance[1]).toBe(1)
  })
})
