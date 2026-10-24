import { describe, expect, it } from 'vitest'
import { kahanSum } from '../base/index.ts'
import { recorder } from '../trace/index.ts'
import {
  betweennessCentrality,
  completeGraph,
  cycleGraph,
  degreeCentrality,
  digraph,
  eigenvectorCentrality,
  eigenvectorCentralityAlgorithm,
  graph,
  hits,
  katzCentrality,
  pagerank,
  pagerankAlgorithm,
  pathGraph,
  shortestPathCounts,
  starGraph,
} from './index.ts'

describe('centrality', () => {
  it('ranks the centre of a star highest by every measure', () => {
    const g = starGraph(5)
    for (const c of [
      degreeCentrality(g),
      betweennessCentrality(g),
      eigenvectorCentrality(g),
      katzCentrality(g),
      pagerank(g),
    ]) {
      expect(Array.from(c).every((v, i) => i === 0 || v < (c[0] ?? 0))).toBe(true)
    }
  })

  it('gives PageRank a probability vector, with dangling mass redistributed', () => {
    const g = digraph(
      [
        [0, 1],
        [1, 2],
      ],
      { nodes: 4 },
    )
    const p = pagerank(g, { tol: 1e-12, maxIter: 1000 })
    expect(Math.abs(kahanSum(p) - 1)).toBeLessThan(1e-12)
    const personal = pagerank(g, { personalization: [1, 0, 0, 0], tol: 1e-12, maxIter: 1000 })
    expect(personal[3]).toBeCloseTo(0, 12)
    expect(() => pagerank(g, { personalization: [0, 0, 0, 0] })).toThrow()
    expect(() => pagerank(starGraph(4), { maxIter: 1, tol: 1e-15 })).toThrow()
  })

  it('traces the power iterations', () => {
    const rec = recorder()
    eigenvectorCentrality(cycleGraph(5), { tracer: rec })
    expect(rec.events.filter((e) => e.type === 'metric').length).toBeGreaterThan(0)
    expect(eigenvectorCentralityAlgorithm(cycleGraph(3)).schema?.metrics['error']).toBeDefined()
    expect(pagerankAlgorithm(cycleGraph(3)).name).toBe('pagerank')
    expect(() => eigenvectorCentrality(cycleGraph(3), { start: [0, 0, 0] })).toThrow()
  })

  it('diverges for Katz when alpha exceeds 1 / lambda_max', () => {
    expect(() => katzCentrality(completeGraph(5), { alpha: 0.5, maxIter: 200 })).toThrow()
    expect(
      Array.from(
        katzCentrality(pathGraph(3), { normalized: false, alpha: 0.1, tol: 1e-14 }),
        (v) => Math.round(v * 1e8) / 1e8,
      ),
    ).toEqual([1.12244898, 1.2244898, 1.12244898])
  })

  it('splits HITS into hubs pointing at authorities', () => {
    // Hubs are the leading eigenvector of A A^T = [[2, 1], [1, 1]] on nodes 0 and 1: (1, 1 / phi).
    const r = hits(
      digraph([
        [0, 2],
        [0, 3],
        [1, 2],
      ]),
      { normalized: false },
    )
    const phi = (1 + Math.sqrt(5)) / 2
    expect(r.hubs[0]).toBe(1)
    expect(r.hubs[1]).toBeCloseTo(1 / phi, 7)
    expect(r.authorities[2]).toBe(1)
    expect(r.authorities[3]).toBeCloseTo(1 / phi, 7)
    expect(r.hubs[2]).toBe(0)
  })

  it('counts shortest paths for Brandes, and handles tiny graphs', () => {
    expect(
      Array.from(
        shortestPathCounts(
          graph([
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
          ]),
          0,
        ).sigma,
      ),
    ).toEqual([1, 1, 1, 2])
    expect(Array.from(betweennessCentrality(pathGraph(2)))).toEqual([0, 0])
    expect(Array.from(degreeCentrality(graph([], { nodes: 1 })))).toEqual([0])
    expect(() => betweennessCentrality(graph([[0, 1, -1]]), { weighted: true })).toThrow()
  })
})
