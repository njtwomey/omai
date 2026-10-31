import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { iterate, runAlgorithm } from '../trace/index.ts'
import {
  bondPercolationAlgorithm,
  completeGraph,
  connectedComponents,
  erdosRenyiGiantFraction,
  giantComponentFraction,
  graph,
  latticeSides,
  PERCOLATION_THRESHOLDS,
  pathGraph,
  percolationAlgorithm,
  percolationSweep,
  sitePercolationAlgorithm,
  spanningCluster,
  spanningThresholds,
  subgraph,
  UnionFind,
} from './index.ts'

const mean = (x: readonly number[]): number => x.reduce((a, b) => a + b, 0) / x.length
const sd = (x: readonly number[]): number => Math.sqrt(mean(x.map((v) => (v - mean(x)) ** 2)))

describe('percolation: the union-find against recomputed components', () => {
  it('site mode: clusters and largest cluster match the components of the occupied subgraph at every step', () => {
    const l = latticeSides(7, 6)
    const alg = percolationAlgorithm(l.graph, { mode: 'site', top: l.top, bottom: l.bottom })
    for (const { t, state } of iterate(alg, { seed: 3, maxSteps: 42 })) {
      const on = Array.from({ length: 42 }, (_, v) => v).filter((v) => state.occupied[v] === 1)
      expect(on).toHaveLength(t)
      if (on.length === 0) continue
      const c = connectedComponents(subgraph(l.graph, on))
      expect(state.clusters).toBe(c.count)
      expect(state.largest).toBe(Math.max(...c.components.map((x) => x.length)))
      // Spanning means one component holds a node of the top row and one of the bottom row.
      const spans = c.components.some(
        (comp) => comp.some((i) => (on[i] as number) < 7) && comp.some((i) => (on[i] as number) >= 35),
      )
      expect(state.spans).toBe(spans)
    }
  })

  it('bond mode: clusters and largest cluster match the components of the open bonds at every step', () => {
    const l = latticeSides(6, 6)
    const alg = percolationAlgorithm(l.graph, { mode: 'bond', top: l.top, bottom: l.bottom })
    const edges = l.graph.edges
    for (const { state } of iterate(alg, { seed: 5, maxSteps: edges.length })) {
      const open = Array.from(
        state.order.subarray(0, state.added),
        (e) => [edges[e]?.source ?? 0, edges[e]?.target ?? 0] as [number, number],
      )
      const c = connectedComponents(graph(open, { nodes: 36 }))
      expect(state.clusters).toBe(c.count)
      expect(state.largest).toBe(Math.max(...c.components.map((x) => x.length)))
      const spans = c.components.some((comp) => comp.some((v) => v < 6) && comp.some((v) => v >= 30))
      expect(state.spans).toBe(spans)
    }
  })

  it('spans fires once, at the first spanning step, and the spanning cluster touches both sides', () => {
    const alg = bondPercolationAlgorithm(8)
    const run = runAlgorithm(alg, { seed: 1, maxSteps: 1000 })
    expect(run.status).toBe('completed')
    expect(run.state.spannedAt).toBeGreaterThan(7)
    const stopped = runAlgorithm(bondPercolationAlgorithm(8, 8, true), { seed: 1, maxSteps: 1000 })
    expect(stopped.status).toBe('converged')
    expect(stopped.state.added).toBe(run.state.spannedAt)
    const cluster = spanningCluster(stopped.state)
    expect(cluster.some((v) => v < 8)).toBe(true)
    expect(cluster.some((v) => v >= 56)).toBe(true)
    expect(spanningCluster(runAlgorithm(bondPercolationAlgorithm(8), { seed: 1, maxSteps: 3 }).state)).toEqual([])
  })

  it('on a path every bond must be open for the ends to span', () => {
    const g = pathGraph(6)
    const run = runAlgorithm(percolationAlgorithm(g, { mode: 'bond', top: [0], bottom: [5], stopAtSpanning: true }), {
      seed: 2,
      maxSteps: 100,
    })
    expect(run.state.added).toBe(5)
    expect(run.state.largest).toBe(6)
  })

  it('resumes exactly from a checkpoint', () => {
    const alg = sitePercolationAlgorithm(9)
    const straight = runAlgorithm(alg, { seed: 4, maxSteps: 60 })
    const early = runAlgorithm(alg, { seed: 4, maxSteps: 25 })
    const resumed = runAlgorithm(alg, {
      seed: 4,
      maxSteps: 35,
      start: { state: alg.serialise?.(early.state) ?? null, step: early.steps, serialised: true },
    })
    expect(Array.from(resumed.state.occupied)).toEqual(Array.from(straight.state.occupied))
    expect(resumed.state.largest).toBe(straight.state.largest)
    expect(resumed.state.clusters).toBe(straight.state.clusters)
  })

  it('rejects an unknown mode', () => {
    expect(() => percolationAlgorithm(pathGraph(3), { mode: 'edge' as 'site' })).toThrow(/mode/)
  })
})

describe('UnionFind copies and snapshots', () => {
  it('a clone is independent and a snapshot restores the sets', () => {
    const a = new UnionFind(6)
    a.union(0, 1)
    a.union(2, 3)
    const b = a.clone()
    b.union(1, 2)
    expect([a.count, b.count]).toEqual([4, 3])
    expect(a.connected(0, 3)).toBe(false)
    const c = UnionFind.fromJSON(JSON.parse(JSON.stringify(b.toJSON())))
    expect(c.sets()).toEqual(b.sets())
    expect(c.setSize(3)).toBe(4)
    expect(() => UnionFind.fromJSON({ parent: [0], rank: [], size: [1], count: 1 })).toThrow(/differ/)
  })
})

describe('the critical probabilities', () => {
  it('bond percolation on the square lattice spans near 1/2, and the window narrows as the lattice grows', () => {
    const small = spanningThresholds(8, 8, 'bond', 80, key(1))
    const large = spanningThresholds(32, 32, 'bond', 80, key(2))
    expect(PERCOLATION_THRESHOLDS['square-bond']).toBe(0.5)
    expect(Math.abs(mean(large) - 0.5)).toBeLessThan(0.02)
    expect(Math.abs(mean(small) - 0.5)).toBeLessThan(0.04)
    expect(sd(large)).toBeLessThan(sd(small) * 0.75)
  })

  it('site percolation spans near 0.5927', () => {
    const t = spanningThresholds(32, 32, 'site', 80, key(3))
    expect(Math.abs(mean(t) - PERCOLATION_THRESHOLDS['square-site'])).toBeLessThan(0.03)
  })

  it('the thresholds of the triangular and honeycomb lattices are dual', () => {
    expect(PERCOLATION_THRESHOLDS['triangular-bond'] + PERCOLATION_THRESHOLDS['honeycomb-bond']).toBeCloseTo(1, 14)
    expect(PERCOLATION_THRESHOLDS['triangular-site'] + 0.5).toBeCloseTo(1, 14)
  })

  it('the largest cluster is small below p_c and holds most sites well above it', () => {
    const l = latticeSides(40, 40)
    const s = percolationSweep(l.graph, key(4), { mode: 'site', top: l.top, bottom: l.bottom })
    expect(s.p).toHaveLength(1601)
    const at = (p: number): number => s.largest[Math.round(p * 1600)] as number
    expect(at(0.4)).toBeLessThan(0.05)
    expect(at(0.8)).toBeGreaterThan(0.7)
    expect(s.largest[1600]).toBe(1)
    expect(s.clusters[1600]).toBe(1)
    expect(s.spanningP).toBeGreaterThan(0.45)
    expect(s.spanningP).toBeLessThan(0.75)
  })
})

describe('the Erdos-Renyi giant component', () => {
  it('has the closed form S = 1 - exp(-c S) above c = 1', () => {
    expect(erdosRenyiGiantFraction(0.9)).toBe(0)
    expect(erdosRenyiGiantFraction(1)).toBe(0)
    const s = erdosRenyiGiantFraction(2)
    expect(s).toBeCloseTo(0.79681213, 8)
    expect(s).toBeCloseTo(1 - Math.exp(-2 * s), 14)
    expect(() => erdosRenyiGiantFraction(-1)).toThrow(/non-negative/)
  })

  it('bond percolation on the complete graph is the Erdos-Renyi process: the giant appears at mean degree 1', () => {
    const n = 400
    const s = percolationSweep(completeGraph(n), key(5), { mode: 'bond' })
    const edgesAt = (c: number): number => Math.round((c * n) / 2)
    expect(s.largest[edgesAt(0.5)] as number).toBeLessThan(0.1)
    expect(Math.abs((s.largest[edgesAt(2)] as number) - erdosRenyiGiantFraction(2))).toBeLessThan(0.05)
    expect(Math.abs((s.largest[edgesAt(3)] as number) - erdosRenyiGiantFraction(3))).toBeLessThan(0.04)
  })

  it('giantComponentFraction counts the largest component', () => {
    expect(
      giantComponentFraction(
        graph(
          [
            [0, 1],
            [1, 2],
          ],
          { nodes: 5 },
        ),
      ),
    ).toBe(0.6)
    expect(giantComponentFraction(graph([], { nodes: 0 }))).toBe(0)
  })
})
