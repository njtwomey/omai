import { describe, expect, it } from 'vitest'
import { maxOf, meanOf, minOf } from '../base/index.ts'
import { foldIn, key } from '../random/index.ts'
import {
  balancedTree,
  barabasiAlbertGraph,
  barbellGraph,
  cycleGraph,
  degree,
  edgeList,
  erdosRenyiGraph,
  gnmRandomGraph,
  isDag,
  kNearestNeighbourGraph,
  randomDag,
  randomGeometricGraph,
  randomKnnGraph,
  stochasticBlockModel,
  wattsStrogatzGraph,
  wheelGraph,
} from './index.ts'

const k0 = key(42)
const draws = (n: number, f: (k: ReturnType<typeof key>) => number): number[] =>
  Array.from({ length: n }, (_, i) => f(foldIn(k0, i)))

describe('random graph models', () => {
  it('draws G(n, p) with the expected number of edges, reproducibly', () => {
    const n = 30
    const p = 0.2
    const m = meanOf(draws(60, (k) => erdosRenyiGraph(k, n, p).edges.length))
    const expected = (p * n * (n - 1)) / 2
    const sd = Math.sqrt((expected * (1 - p)) / 60)
    expect(Math.abs(m - expected)).toBeLessThan(5 * sd)
    expect(erdosRenyiGraph(k0, 10, 0.3).edges).toEqual(erdosRenyiGraph(k0, 10, 0.3).edges)
    expect(erdosRenyiGraph(k0, 6, 1, { directed: true }).edges.length).toBe(30)
    expect(() => erdosRenyiGraph(k0, 5, 1.5)).toThrow()
  })

  it('draws G(n, m) uniformly with exactly m edges', () => {
    expect(gnmRandomGraph(k0, 10, 0).edges.length).toBe(0)
    expect(gnmRandomGraph(k0, 5, 10).edges.length).toBe(10)
    expect(gnmRandomGraph(k0, 5, 7, { directed: true }).edges.length).toBe(7)
    expect(() => gnmRandomGraph(k0, 4, 7)).toThrow()
  })

  it('rewires Watts-Strogatz rings without changing the edge count', () => {
    for (const p of [0, 0.3, 1]) {
      const g = wattsStrogatzGraph(k0, 20, 4, p)
      expect(g.edges.length).toBe(40)
      expect(g.edges.every((e) => e.source !== e.target)).toBe(true)
    }
    const sorted = (g: ReturnType<typeof cycleGraph>) => edgeList(g).sort((a, b) => a[0] - b[0] || a[1] - b[1])
    expect(sorted(wattsStrogatzGraph(k0, 8, 2, 0))).toEqual(sorted(cycleGraph(8)))
    expect(() => wattsStrogatzGraph(k0, 4, 4, 0.1)).toThrow()
  })

  it('grows Barabasi-Albert graphs with heavy-tailed degrees', () => {
    const g = barabasiAlbertGraph(k0, 200, 2)
    expect(g.edges.length).toBe(2 + 2 * (200 - 3))
    const d = degree(g)
    expect(minOf(d.slice(3))).toBeGreaterThanOrEqual(2)
    // Early nodes collect far more links than average.
    expect(maxOf(d.slice(0, 5))).toBeGreaterThan(4 * meanOf(d))
    expect(() => barabasiAlbertGraph(k0, 3, 3)).toThrow()
  })

  it('draws stochastic block models with denser blocks inside', () => {
    const g = stochasticBlockModel(
      k0,
      [20, 20],
      [
        [0.5, 0.05],
        [0.05, 0.5],
      ],
    )
    let inside = 0
    for (const e of g.edges) if (g.nodeData[e.source]?.['block'] === g.nodeData[e.target]?.['block']) inside++
    expect(inside).toBeGreaterThan(4 * (g.edges.length - inside))
    expect(() =>
      stochasticBlockModel(
        k0,
        [2, 2],
        [
          [0.5, 0.1],
          [0.2, 0.5],
        ],
      ),
    ).toThrow()
    expect(() => stochasticBlockModel(k0, [2], [[0.5, 0.1]])).toThrow()
    expect(
      stochasticBlockModel(
        k0,
        [2, 2],
        [
          [0.5, 0.1],
          [0.2, 0.5],
        ],
        { directed: true },
      ).directed,
    ).toBe(true)
    expect(stochasticBlockModel(k0, [3], [[1]], { selfloops: true }).edges.length).toBe(6)
  })

  it('draws DAGs, geometric and nearest-neighbour graphs', () => {
    for (let i = 0; i < 5; i++) expect(isDag(randomDag(foldIn(k0, i), 12, 0.4))).toBe(true)
    const g = randomGeometricGraph(k0, 30, 0.3)
    for (const e of g.edges) {
      const a = g.nodeData[e.source]?.['pos'] as number[]
      const b = g.nodeData[e.target]?.['pos'] as number[]
      expect(Math.hypot((a[0] ?? 0) - (b[0] ?? 0), (a[1] ?? 0) - (b[1] ?? 0))).toBeLessThanOrEqual(0.3)
    }
    expect(randomGeometricGraph(k0, 5, 0.2, { dim: 3 }).nodeData[0]?.['pos']).toHaveLength(3)
    const knn = randomKnnGraph(k0, 25, 3)
    expect(degree(knn).every((x) => x >= 3)).toBe(true)
    const mutual = kNearestNeighbourGraph([[0], [1], [3], [7]], 1, { mutual: true, weighted: true })
    expect(mutual.edges).toEqual([{ source: 0, target: 1, weight: 1 }])
  })

  it('builds the small deterministic families', () => {
    expect(wheelGraph(3).edges.length).toBe(3)
    expect(wheelGraph(4).edges.length).toBe(6)
    expect(balancedTree(1, 3).n).toBe(4)
    expect(() => barbellGraph(1, 0)).toThrow()
    expect(cycleGraph(2).edges.length).toBe(1)
    expect(cycleGraph(1).edges.length).toBe(0)
  })
})
