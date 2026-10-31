import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import {
  averageClustering,
  averageShortestPathLength,
  barabasiAlbertDegreeLaw,
  blockGrowthAlgorithm,
  blockModelExpectedEdges,
  completeGraph,
  configurationModelGraph,
  connectedComponents,
  cycleGraph,
  edgeByEdgeAlgorithm,
  erdosRenyiConnectivityEdges,
  erdosRenyiGiantFraction,
  graph,
  growthGraph,
  pathGraph,
  preferentialAttachmentAlgorithm,
  rewiringAlgorithm,
  ringLatticePathLength,
  smallWorldSweep,
  starGraph,
  uniformAttachmentDegreeLaw,
  unrankPair,
  wattsStrogatzClustering,
  wattsStrogatzGraph,
} from './index.ts'

const degreeCounts = (n: number, flat: Int32Array): Int32Array => {
  const d = new Int32Array(n)
  for (const v of flat) d[v] = (d[v] ?? 0) + 1
  return d
}

describe('average shortest path length', () => {
  it('matches the closed forms', () => {
    expect(averageShortestPathLength(pathGraph(4))).toBeCloseTo(5 / 3, 14)
    expect(averageShortestPathLength(completeGraph(6))).toBe(1)
    expect(averageShortestPathLength(cycleGraph(6))).toBeCloseTo(9 / 5, 14)
    expect(averageShortestPathLength(starGraph(4))).toBeCloseTo((8 + 12 * 2) / 20, 14)
    expect(
      averageShortestPathLength(
        graph([
          [0, 1],
          [2, 3],
        ]),
      ),
    ).toBe(1)
    expect(averageShortestPathLength(graph([], { nodes: 3 }))).toBe(0)
  })
})

describe('unrankPair', () => {
  it('enumerates the pairs in order', () => {
    for (const n of [2, 3, 7, 12]) {
      const want: [number, number][] = []
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) want.push([i, j])
      expect(want.map((_, r) => unrankPair(n, r))).toEqual(want)
    }
  })
})

describe('Erdos-Renyi edge by edge', () => {
  it('the giant component appears as the mean degree passes 1 and matches S = 1 - exp(-c S)', () => {
    const n = 600
    const rec = recorder()
    const run = runAlgorithm(edgeByEdgeAlgorithm(n, 1.5 * n), {
      seed: 2,
      maxSteps: 10 * n,
      tracer: rec,
      policy: { ...policies.debug, schedule: { every: 25 } },
    })
    expect(run.steps).toBe(900)
    const share = (c: number): number => {
      const g = growthGraph(n, run.state.edges.subarray(0, 2 * Math.round((c * n) / 2)))
      return Math.max(...connectedComponents(g).components.map((x) => x.length)) / n
    }
    expect(share(0.5)).toBeLessThan(0.08)
    expect(Math.abs(share(3) - erdosRenyiGiantFraction(3))).toBeLessThan(0.04)
    const gs = rec.events.filter((e) => e.type === 'metric' && e.name === 'graph/giantShare')
    expect(gs.length).toBeGreaterThan(5)
  })

  it('every edge is a distinct pair and the graph is connected near (n / 2) ln n edges', () => {
    const n = 200
    const m = Math.round(2.5 * erdosRenyiConnectivityEdges(n))
    const run = runAlgorithm(edgeByEdgeAlgorithm(n, m), { seed: 7, maxSteps: m })
    const g = growthGraph(n, run.state.edges)
    expect(g.edges).toHaveLength(m)
    expect(connectedComponents(g).count).toBe(1)
    expect(
      connectedComponents(
        growthGraph(n, run.state.edges.subarray(0, 2 * Math.round(0.4 * erdosRenyiConnectivityEdges(n)))),
      ).count,
    ).toBeGreaterThan(5)
  })

  it('adding all pairs gives the complete graph, and bad sizes are rejected', () => {
    const run = runAlgorithm(edgeByEdgeAlgorithm(6), { seed: 0, maxSteps: 100 })
    expect(growthGraph(6, run.state.edges).edges).toHaveLength(15)
    expect(() => edgeByEdgeAlgorithm(1)).toThrow(/two nodes/)
    expect(() => edgeByEdgeAlgorithm(5, 11)).toThrow(/maxEdges/)
  })
})

describe('preferential attachment', () => {
  it('has the degree law 2m(m + 1) / (k (k + 1) (k + 2)) and uniform attachment the geometric one', () => {
    const n = 4000
    const m = 3
    const pa = runAlgorithm(preferentialAttachmentAlgorithm(n, m), { seed: 1, maxSteps: n }).state
    const un = runAlgorithm(preferentialAttachmentAlgorithm(n, m, 'uniform'), { seed: 1, maxSteps: n }).state
    expect(pa.nodes).toBe(n)
    expect(pa.edges.length / 2).toBe((n - m - 1) * m + m)
    const frac = (flat: Int32Array, k: number): number => degreeCounts(n, flat).filter((d) => d === k).length / n
    expect(Math.abs(frac(pa.edges, 3) - barabasiAlbertDegreeLaw(3, m))).toBeLessThan(0.025)
    expect(Math.abs(frac(pa.edges, 6) - barabasiAlbertDegreeLaw(6, m))).toBeLessThan(0.012)
    expect(Math.abs(frac(un.edges, 3) - uniformAttachmentDegreeLaw(3, m))).toBeLessThan(0.025)
    expect(Math.abs(frac(un.edges, 6) - uniformAttachmentDegreeLaw(6, m))).toBeLessThan(0.02)
    // Hubs: preferential attachment's biggest hub dwarfs uniform attachment's.
    expect(Math.max(...degreeCounts(n, pa.edges))).toBeGreaterThan(3 * Math.max(...degreeCounts(n, un.edges)))
  })

  it('the laws are probability distributions', () => {
    let pa = 0
    let un = 0
    for (let k = 2; k < 200000; k++) {
      pa += barabasiAlbertDegreeLaw(k, 2)
      un += uniformAttachmentDegreeLaw(k, 2)
    }
    expect(pa).toBeCloseTo(1, 4)
    expect(un).toBeCloseTo(1, 12)
    expect(barabasiAlbertDegreeLaw(1, 2)).toBe(0)
  })

  it('each new node brings m distinct neighbours, and the graph stays connected', () => {
    const run = runAlgorithm(preferentialAttachmentAlgorithm(60, 3), { seed: 4, maxSteps: 100 })
    const g = growthGraph(60, run.state.edges)
    expect(g.edges).toHaveLength(run.state.edges.length / 2)
    expect(connectedComponents(g).count).toBe(1)
    expect(() => preferentialAttachmentAlgorithm(5, 5)).toThrow(/1 <= m < n/)
    expect(() => preferentialAttachmentAlgorithm(5, 2, 'x' as 'uniform')).toThrow(/rule/)
  })

  it('resumes exactly from a checkpoint', () => {
    const alg = preferentialAttachmentAlgorithm(50, 2)
    const straight = runAlgorithm(alg, { seed: 3, maxSteps: 30 })
    const early = runAlgorithm(alg, { seed: 3, maxSteps: 10 })
    const resumed = runAlgorithm(alg, {
      seed: 3,
      maxSteps: 20,
      start: { state: alg.serialise?.(early.state) ?? null, step: early.steps, serialised: true },
    })
    expect(Array.from(resumed.state.edges)).toEqual(Array.from(straight.state.edges))
  })
})

describe('rewiring the ring', () => {
  it('without rewiring it is the ring lattice: clustering 3(k - 2) / (4(k - 1)) and the exact path length', () => {
    const run = runAlgorithm(rewiringAlgorithm(40, 6, 0), { seed: 0, maxSteps: 1000 })
    const g = growthGraph(40, run.state.edges)
    expect(run.steps).toBe(120)
    expect(run.state.rewired).toBe(0)
    expect(averageClustering(g)).toBeCloseTo(wattsStrogatzClustering(6, 0), 12)
    expect(wattsStrogatzClustering(6, 0)).toBeCloseTo(0.6, 14)
    expect(averageShortestPathLength(g)).toBeCloseTo(ringLatticePathLength(40, 6), 12)
    expect(ringLatticePathLength(6, 2)).toBeCloseTo(9 / 5, 14)
  })

  it('rewiring keeps the number of edges and every edge distinct', () => {
    const run = runAlgorithm(rewiringAlgorithm(60, 6, 0.3), { seed: 2, maxSteps: 1000 })
    const g = growthGraph(60, run.state.edges)
    expect(g.edges).toHaveLength(180)
    expect(run.state.rewired).toBeGreaterThan(30)
    expect(run.state.rewired).toBeLessThan(80)
  })

  it('clustering follows (1 - p)^3 while the path length collapses', () => {
    const [point] = smallWorldSweep(key(4), 400, 8, [0.1], 8)
    const ratio = (1 - 0.1) ** 3
    expect(Math.abs((point?.clustering ?? 0) - ratio)).toBeLessThan(0.08)
    expect(point?.pathLength).toBeLessThan(0.4)
    expect(smallWorldSweep(key(0), 30, 4, [0], 1)).toEqual([{ p: 0, clustering: 1, pathLength: 1 }])
  })

  it('the stepped algorithm and the generator rewire at the same rate', () => {
    const rates: number[] = []
    for (let s = 0; s < 10; s++)
      rates.push(runAlgorithm(rewiringAlgorithm(100, 4, 0.2), { seed: s, maxSteps: 1000 }).state.rewired / 200)
    expect(Math.abs(rates.reduce((a, b) => a + b, 0) / 10 - 0.2)).toBeLessThan(0.03)
    const g = wattsStrogatzGraph(key(0), 100, 4, 0.2)
    expect(g.edges).toHaveLength(200)
  })

  it('rejects an odd lattice degree and a probability outside [0, 1]', () => {
    expect(() => rewiringAlgorithm(10, 3, 0.1)).toThrow(/even/)
    expect(() => rewiringAlgorithm(10, 4, 1.1)).toThrow(/p must/)
  })
})

describe('block model growth', () => {
  it('the edge count matches the expectation and the communities show', () => {
    const shares = [0.5, 0.5]
    const probs = [
      [0.3, 0.02],
      [0.02, 0.3],
    ]
    const n = 200
    const counts: number[] = []
    let within = 0
    let total = 0
    for (let seed = 0; seed < 10; seed++) {
      const s = runAlgorithm(blockGrowthAlgorithm(n, shares, probs), { seed, maxSteps: n }).state
      counts.push(s.edges.length / 2)
      for (let i = 0; i < s.edges.length; i += 2) {
        total += 1
        if (s.blocks[s.edges[i] ?? 0] === s.blocks[s.edges[i + 1] ?? 0]) within += 1
      }
    }
    const want = blockModelExpectedEdges(n, shares, probs)
    expect(Math.abs(counts.reduce((a, b) => a + b, 0) / 10 / want - 1)).toBeLessThan(0.04)
    expect(within / total).toBeGreaterThan(0.85)
    expect(
      blockModelExpectedEdges(
        10,
        [0.5, 0.5],
        [
          [1, 0],
          [0, 1],
        ],
      ),
    ).toBe(22.5)
  })

  it('rejects shares that do not sum to 1', () => {
    expect(() =>
      blockGrowthAlgorithm(
        5,
        [0.5, 0.4],
        [
          [1, 0],
          [0, 1],
        ],
      ),
    ).toThrow(/sum to 1/)
    expect(() => blockGrowthAlgorithm(5, [1], [[2]])).toThrow(/probabilities/)
  })
})

describe('configuration model', () => {
  it('realises a regular sequence up to the erased ends, which are counted', () => {
    const degrees = Array<number>(1000).fill(3)
    const g = configurationModelGraph(key(0), degrees)
    const realised = Array.from({ length: 1000 }, (_, v) => (g.out.indptr[v + 1] ?? 0) - (g.out.indptr[v] ?? 0))
    const lost = (g.attributes['selfLoops'] as number) + (g.attributes['multiEdges'] as number)
    expect(realised.reduce((a, b) => a + b, 0) + 2 * lost).toBe(3000)
    expect(lost).toBeGreaterThan(0)
    expect(lost).toBeLessThan(30)
    expect(Math.max(...realised)).toBeLessThanOrEqual(3)
  })

  it('a sequence of ones pairs the nodes and only a self-loop can be erased', () => {
    const g = configurationModelGraph(key(1), Array<number>(20).fill(1))
    expect(g.attributes['multiEdges']).toBe(0)
    expect(g.edges).toHaveLength(10 - (g.attributes['selfLoops'] as number))
  })

  it('rejects odd totals and bad degrees', () => {
    expect(() => configurationModelGraph(key(0), [1, 1, 1])).toThrow(/even/)
    expect(() => configurationModelGraph(key(0), [1.5, 0.5])).toThrow(/non-negative integers/)
  })
})
