import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  barbellGraph,
  completeGraph,
  estimateSpread,
  exactIndependentCascadeSpread,
  influenceMaximisationAlgorithm,
  karateClubGraph,
  linearThresholdAlgorithm,
  liveEdgeReach,
  liveEdgeSamples,
  pathGraph,
  starGraph,
} from './index.ts'

const seedsOf = (alg: ReturnType<typeof influenceMaximisationAlgorithm>, k: number, seed = 0) =>
  runAlgorithm(alg, { seed, maxSteps: k }).state

describe('live-edge samples', () => {
  it('the mean reach converges to the exact independent-cascade spread', () => {
    const path = pathGraph(3)
    expect(exactIndependentCascadeSpread(path, [0], 0.5)).toBe(1.75)
    expect(estimateSpread(liveEdgeSamples(path, 'ic', key(1), 20000, { p: 0.5 }), [0])).toBeCloseTo(1.75, 1)
    const k4 = completeGraph(4)
    const exact = exactIndependentCascadeSpread(k4, [0, 1], 0.3)
    expect(Math.abs(estimateSpread(liveEdgeSamples(k4, 'ic', key(2), 20000, { p: 0.3 }), [0, 1]) - exact)).toBeLessThan(
      0.03,
    )
  })

  it('the linear threshold live-edge spread matches the threshold cascade simulated', () => {
    // Seeds are three of the six leaves of a star: the exact spread is 3 + (1 + 3) / 2 = 5.
    const g = starGraph(6)
    const live = estimateSpread(liveEdgeSamples(g, 'lt', key(3), 5000), [1, 2, 3])
    expect(Math.abs(live - 5)).toBeLessThan(0.1)
    let total = 0
    for (let seed = 0; seed < 500; seed++) {
      total += runAlgorithm(linearThresholdAlgorithm(g, [1, 2, 3]), { seed, maxSteps: 10 }).state.adopted
    }
    expect(Math.abs(total / 500 - live)).toBeLessThan(0.3)
  })

  it('every node keeps at most one in-edge under the linear threshold model', () => {
    const g = karateClubGraph()
    for (const s of liveEdgeSamples(g, 'lt', key(4), 5)) {
      const inDegree = new Int32Array(g.n)
      for (const v of s.indices) inDegree[v] = (inDegree[v] ?? 0) + 1
      expect(Math.max(...inDegree)).toBeLessThanOrEqual(1)
      expect(s.indices.length).toBe(g.n)
    }
  })

  it('reach counts the seeds and follows live edges', () => {
    const chain = { indptr: Int32Array.of(0, 1, 2, 2), indices: Int32Array.of(1, 2) }
    expect([
      liveEdgeReach(chain, [0]),
      liveEdgeReach(chain, [1]),
      liveEdgeReach(chain, [0, 2]),
      liveEdgeReach(chain, []),
    ]).toEqual([3, 2, 3, 0])
  })

  it('rejects bad arguments', () => {
    expect(() => liveEdgeSamples(pathGraph(3), 'xx' as 'ic', key(0), 3)).toThrow(/model/)
    expect(() => liveEdgeSamples(pathGraph(3), 'ic', key(0), 0)).toThrow(/samples/)
    expect(() => exactIndependentCascadeSpread(completeGraph(6), [0], 0.5)).toThrow(/enumerate/)
  })
})

describe('influence maximisation', () => {
  const karate = karateClubGraph()
  const options = { p: 0.15, samples: 150 } as const

  it('CELF returns the greedy seeds and gains with far fewer evaluations', () => {
    const greedy = seedsOf(influenceMaximisationAlgorithm(karate, 5, { ...options, method: 'greedy' }), 5)
    const celf = seedsOf(influenceMaximisationAlgorithm(karate, 5, { ...options, method: 'celf' }), 5)
    expect(celf.seeds).toEqual(greedy.seeds)
    expect(celf.gains).toEqual(greedy.gains)
    expect(greedy.evaluations).toBe(34 + 33 + 32 + 31 + 30)
    expect(celf.evaluations).toBeLessThan(greedy.evaluations / 2)
  })

  it('the marginal gains diminish (the spread is submodular) and the first seed is a hub', () => {
    const s = seedsOf(influenceMaximisationAlgorithm(karate, 6, { ...options, method: 'greedy' }), 6)
    for (let i = 1; i < s.gains.length; i++)
      expect(s.gains[i] as number).toBeLessThanOrEqual((s.gains[i - 1] as number) + 1e-12)
    expect([0, 33]).toContain(s.seeds[0])
    expect(s.gains.reduce((a, b) => a + b, 0)).toBeCloseTo(s.total / s.sampled.length, 10)
  })

  it('greedy reaches at least 1 - 1/e of the best pair, found by brute force on the same samples', () => {
    const s = seedsOf(influenceMaximisationAlgorithm(karate, 2, { ...options, method: 'greedy' }), 2)
    let best = 0
    for (let a = 0; a < karate.n; a++) {
      for (let b = a + 1; b < karate.n; b++) best = Math.max(best, estimateSpread(s.sampled, [a, b]))
    }
    const got = estimateSpread(s.sampled, s.seeds)
    expect(got).toBeLessThanOrEqual(best + 1e-9)
    expect(got).toBeGreaterThanOrEqual((1 - 1 / Math.E) * best)
    expect(got / best).toBeGreaterThan(0.9)
  })

  it('beats the highest-degree heuristic on two cliques joined by a bridge by diversifying', () => {
    const g = barbellGraph(6, 1)
    const s = seedsOf(influenceMaximisationAlgorithm(g, 2, { p: 0.3, samples: 400, method: 'celf' }), 2)
    const side = (v: number): number => (v < 6 ? 0 : v >= 7 ? 1 : 2)
    expect(new Set(s.seeds.map(side)).size).toBe(2)
    const degree = seedsOf(influenceMaximisationAlgorithm(g, 2, { p: 0.3, samples: 400, method: 'degree' }), 2)
    expect(s.total).toBeGreaterThanOrEqual(degree.total)
  })

  it('on a star the first seed is the hub; the linear threshold model works too', () => {
    expect(seedsOf(influenceMaximisationAlgorithm(starGraph(8), 1, { p: 0.5, samples: 50 }), 1).seeds).toEqual([0])
    const lt = seedsOf(influenceMaximisationAlgorithm(starGraph(8), 2, { model: 'lt', samples: 100 }), 2)
    expect(lt.seeds[0]).toBe(0)
    expect(lt.total / 100).toBeGreaterThan(5)
  })

  it('serialises and resumes', () => {
    const alg = influenceMaximisationAlgorithm(karate, 4, { ...options, method: 'celf' })
    const straight = runAlgorithm(alg, { seed: 2, maxSteps: 4 })
    const early = runAlgorithm(alg, { seed: 2, maxSteps: 2 })
    const resumed = runAlgorithm(alg, {
      seed: 2,
      maxSteps: 2,
      start: { state: alg.serialise?.(early.state) ?? null, step: early.steps, serialised: true },
    })
    expect(resumed.state.seeds).toEqual(straight.state.seeds)
    expect(resumed.state.evaluations).toBe(straight.state.evaluations)
  })

  it('rejects k outside [1, n] and an unknown method', () => {
    expect(() => influenceMaximisationAlgorithm(pathGraph(4), 5)).toThrow(/k must/)
    expect(() => influenceMaximisationAlgorithm(pathGraph(4), 2, { method: 'x' as 'celf' })).toThrow(/method/)
  })
})
