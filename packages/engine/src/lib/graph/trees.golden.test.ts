import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import type { Tensor } from '../tensor/index.ts'
import { lcaIndex, lowestCommonAncestor, minimumSpanningTree, rootTree, subtreeSizes, treeOrder } from './index.ts'

describe('trees against scipy.sparse.csgraph and networkx', () => {
  for (const c of loadFixture('graph/trees').cases) {
    it(c.name, () => {
      const g = graphFromCase(c.params)
      const p = c.params
      if (c.fn === 'spanning') {
        const [minWeight, maxWeight] = Array.from((decode(c.output) as Tensor).data)
        for (const algorithm of ['kruskal', 'prim', 'boruvka'] as const) {
          const t = minimumSpanningTree(g, { algorithm })
          expect(t.weight).toBe(minWeight)
          expect(t.edges.map((e) => [e.source, e.target])).toEqual(p['treeEdges'])
          const m = minimumSpanningTree(g, { algorithm, maximum: true })
          expect(m.weight).toBe(maxWeight)
          expect(m.edges.map((e) => [e.source, e.target])).toEqual(p['maxTreeEdges'])
        }
        return
      }
      const tree = rootTree(g, p['root'] as number)
      expect(Array.from(tree.depth)).toEqual(p['depth'])
      expect(Array.from(subtreeSizes(tree))).toEqual(p['sizes'])
      expect(treeOrder(tree)).toEqual(p['preorder'])
      expect(treeOrder(tree, 'post')).toEqual(p['postorder'])
      const index = lcaIndex(tree)
      const pairs = p['pairs'] as [number, number][]
      const want = p['lca'] as number[]
      pairs.forEach(([a, b], i) => {
        expect(index.lca(a, b)).toBe(want[i])
        expect(lowestCommonAncestor(tree, a, b)).toBe(want[i])
      })
    })
  }
})
