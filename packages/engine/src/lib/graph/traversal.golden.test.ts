import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import { bfs, dfs, findCycle, isBipartite, isDag, kHopNeighbourhood, topologicalSort } from './index.ts'

describe('traversal against scipy.sparse.csgraph and networkx', () => {
  for (const c of loadFixture('graph/traversal').cases) {
    it(c.name, () => {
      const g = graphFromCase(c.params)
      const p = c.params
      if (c.fn === 'traversal') {
        const s = p['source'] as number
        const b = bfs(g, s)
        expect(b.order).toEqual(p['bfsOrder'])
        expect(Array.from(b.parent)).toEqual(p['bfsPredecessors'])
        expect(Array.from(b.distance)).toEqual(p['distance'])
        const d = dfs(g, s)
        expect(d.preorder).toEqual(p['preorder'])
        expect(d.preorder).toEqual(p['dfsOrder'])
        expect(Array.from(d.parent)).toEqual(p['dfsPredecessors'])
        expect(d.postorder).toEqual(p['postorder'])
        const tree = d.edges
          .filter((e) => e.kind === 'tree')
          .map((e) => [e.source, e.target])
          .sort((x, y) => (x[0] ?? 0) - (y[0] ?? 0) || (x[1] ?? 0) - (y[1] ?? 0))
        expect(tree).toEqual(p['treeEdges'])
        expect(kHopNeighbourhood(g, s, 2)).toEqual(p['kHop2'])
      } else if (c.fn === 'global') {
        expect(findCycle(g) !== null).toBe(p['hasCycle'])
        expect(isBipartite(g)).toBe(p['bipartite'])
        expect(isDag(g)).toBe(p['dag'])
      } else {
        expect(topologicalSort(g)).toEqual(p['order'])
      }
    })
  }
})
