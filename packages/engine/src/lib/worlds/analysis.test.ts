import { describe, expect, it } from 'vitest'
import { graph } from '../graph/index.ts'
import {
  cellDistance,
  deadEnds,
  distanceField,
  fromAscii,
  heuristicTo,
  regions,
  shortestPath,
  toAdjacency,
  world,
} from './index.ts'

describe('analysis', () => {
  it('computes fields from several sources and returns null paths to unreachable cells', () => {
    const w = fromAscii(['G..#.', '....G'])
    expect(Array.from(distanceField(w, w.goals))).toEqual([0, 1, 2, Infinity, 1, 1, 2, 2, 1, 0])
    expect(shortestPath(fromAscii(['.#.']), 0, 2)).toBeNull()
    expect(shortestPath(w, 4, 4)).toEqual([4])
    expect(() => distanceField(w, 10)).toThrow()
  })

  it('finds dead ends and regions with blocked cells', () => {
    const w = fromAscii(['....'])
    expect(deadEnds(w)).toEqual([0, 3])
    expect(regions(w, { blocked: (c) => c === 1 }).sizes).toEqual([1, 2])
  })

  it('builds weighted and directed adjacency that graph() accepts', () => {
    const w = fromAscii(['.~', '>.'])
    const adj = toAdjacency(w, { weighted: true })
    expect(adj.directed).toBe(true)
    const g = graph(adj.edges, { nodes: adj.labels, directed: adj.directed })
    expect([g.n, g.edges.length]).toEqual([4, adj.edges.length])
    expect(adj.edges).toContainEqual([0, 1, 5])
    expect(adj.edges).toContainEqual([1, 0, 1])
    const all = toAdjacency(fromAscii(['.#']), { cells: 'all' })
    expect([all.n, all.edges]).toEqual([2, []])
    const nodes = toAdjacency(world({ topology: { kind: 'graph', n: 2, neighbours: [[1], [0]] } }))
    expect(nodes.labels).toEqual(['0', '1'])
  })

  it('measures cells by Manhattan, Chebyshev, octile, Euclidean and hexagonal distance', () => {
    const w = fromAscii(['....', '....', '....'])
    expect(cellDistance(w, 0, 11, 'octile')).toBeCloseTo(1 + 2 * Math.SQRT2, 12)
    const h = fromAscii(['....', '....', '....'], { topology: 'hex' })
    expect([cellDistance(h, 0, 1), cellDistance(h, 0, 4), cellDistance(h, 0, 5), cellDistance(h, 0, 11)]).toEqual([
      1, 1, 2, 4,
    ])
    const g = world({
      topology: {
        kind: 'graph',
        n: 2,
        neighbours: [[1], [0]],
        positions: [
          [0, 0],
          [3, 4],
        ],
      },
    })
    expect(cellDistance(g, 0, 1)).toBe(5)
    expect(() => cellDistance(g, 0, 1, 'manhattan')).toThrow()
    expect(heuristicTo(w, { goals: [3, 8], metric: 'chebyshev' })(5)).toBe(1)
  })
})
