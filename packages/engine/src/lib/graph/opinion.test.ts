import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../trace/index.ts'
import {
  boundedConfidenceAlgorithm,
  completeGraph,
  degrootAlgorithm,
  degrootConsensus,
  friedkinJohnsenEquilibrium,
  karateClubGraph,
  pathGraph,
  starGraph,
  voterFixationProbability,
  voterModelAlgorithm,
} from './index.ts'

describe('voter model', () => {
  const winShare = (g: ReturnType<typeof starGraph>, initial: number[], trials: number): number => {
    let ones = 0
    for (let seed = 0; seed < trials; seed++) {
      const run = runAlgorithm(voterModelAlgorithm(g, { initial }), { seed, maxSteps: 20000 })
      expect(run.status).toBe('converged')
      ones += run.state.opinions[0] ?? 0
    }
    return ones / trials
  }

  it('on a star the hub holds half of the total degree, so its opinion wins half the time', () => {
    const g = starGraph(4)
    expect(voterFixationProbability(g, [1, 0, 0, 0, 0])).toBe(0.5)
    expect(Math.abs(winShare(g, [1, 0, 0, 0, 0], 400) - 0.5)).toBeLessThan(0.08)
  })

  it('on a regular graph the fixation probability is the initial share', () => {
    const g = completeGraph(6)
    expect(voterFixationProbability(g, [1, 1, 0, 0, 0, 0])).toBeCloseTo(1 / 3, 12)
    expect(Math.abs(winShare(g, [1, 1, 0, 0, 0, 0], 400) - 1 / 3)).toBeLessThan(0.08)
  })

  it('draws exactly the stated share, ends in consensus with no disagreeing edge, and fires once', () => {
    const g = karateClubGraph()
    const alg = voterModelAlgorithm(g, { initial: 0.25, updatesPerStep: 5 })
    expect(Array.from(runAlgorithm(alg, { seed: 1, maxSteps: 0 }).state.opinions).filter((x) => x === 1)).toHaveLength(
      Math.round(0.25 * 34),
    )
    const run = runAlgorithm(alg, { seed: 1, maxSteps: 100000 })
    expect(run.status).toBe('converged')
    expect(new Set(run.state.opinions).size).toBe(1)
    expect(run.state.updates % 5).toBe(0)
  })

  it('rejects a share outside [0, 1] and opinions of the wrong length', () => {
    expect(() => voterModelAlgorithm(pathGraph(3), { initial: 1.5 })).toThrow(/share/)
    expect(() => voterModelAlgorithm(pathGraph(3), { initial: [1, 0] })).toThrow(/one initial opinion/)
  })
})

describe('DeGroot and Friedkin-Johnsen', () => {
  const karate = karateClubGraph()
  const x0 = Array.from({ length: 34 }, (_, v) => (v * 7) % 11)

  it('converges to the degree-weighted mean of the initial opinions', () => {
    const run = runAlgorithm(degrootAlgorithm(karate, x0, { selfWeight: 0.5, tolerance: 1e-13 }), {
      seed: 0,
      maxSteps: 5000,
    })
    expect(run.status).toBe('converged')
    const want = degrootConsensus(karate, x0)
    for (const v of run.state.x) expect(v).toBeCloseTo(want, 8)
  })

  it('without self weight a bipartite graph oscillates for ever', () => {
    const run = runAlgorithm(degrootAlgorithm(pathGraph(4), [0, 0, 0, 4]), { seed: 0, maxSteps: 200 })
    expect(run.status).toBe('completed')
    expect(run.state.change).toBeGreaterThan(0.1)
  })

  it('stubborn agents converge to the Friedkin-Johnsen equilibrium and keep disagreeing', () => {
    const stub = Array.from({ length: 34 }, (_, v) => (v % 3 === 0 ? 0.6 : 0.1))
    const run = runAlgorithm(degrootAlgorithm(karate, x0, { stubbornness: stub, tolerance: 1e-13 }), {
      seed: 0,
      maxSteps: 5000,
    })
    const eq = friedkinJohnsenEquilibrium(karate, x0, stub)
    expect(run.status).toBe('converged')
    for (let v = 0; v < 34; v++) expect(run.state.x[v]).toBeCloseTo(eq[v] as number, 8)
    expect(Math.max(...eq) - Math.min(...eq)).toBeGreaterThan(1)
  })

  it('the equilibrium solves x = Lambda x0 + (I - Lambda) W x and is x0 for fully stubborn agents', () => {
    const g = pathGraph(2)
    const eq = friedkinJohnsenEquilibrium(g, [0, 1], [0.5, 0.5])
    expect(eq[0]).toBeCloseTo(1 / 3, 12)
    expect(eq[1]).toBeCloseTo(2 / 3, 12)
    expect(Array.from(friedkinJohnsenEquilibrium(karate, x0, 1))).toEqual(x0.map(Number))
  })

  it('the consensus of a star is its degree-weighted mean', () => {
    expect(degrootConsensus(starGraph(4), [4, 0, 0, 0, 0])).toBe(2)
    expect(() =>
      degrootConsensus(pathGraph(2).directed ? pathGraph(2) : { ...pathGraph(2), directed: true }, [0, 1]),
    ).toThrow(/undirected/)
  })

  it('serialises and resumes', () => {
    const alg = degrootAlgorithm(karate, x0, { selfWeight: 0.3 })
    const straight = runAlgorithm(alg, { seed: 0, maxSteps: 12 })
    const early = runAlgorithm(alg, { seed: 0, maxSteps: 5 })
    const resumed = runAlgorithm(alg, {
      seed: 0,
      maxSteps: 7,
      start: { state: alg.serialise?.(early.state) ?? null, step: early.steps, serialised: true },
    })
    expect(Array.from(resumed.state.x)).toEqual(Array.from(straight.state.x))
  })

  it('rejects wrong lengths and stubbornness outside [0, 1]', () => {
    expect(() => degrootAlgorithm(pathGraph(3), [0, 1])).toThrow(/one initial opinion/)
    expect(() => degrootAlgorithm(pathGraph(3), [0, 1, 2], { stubbornness: 2 })).toThrow(/stubbornness/)
  })
})

describe('bounded confidence', () => {
  it('a tolerance wider than the range gives consensus on the mean in one round (complete graph)', () => {
    const run = runAlgorithm(boundedConfidenceAlgorithm(completeGraph(3), [0, 0.5, 1], 1), { seed: 0, maxSteps: 5 })
    expect(Array.from(run.state.x)).toEqual([0.5, 0.5, 0.5])
    expect(run.status).toBe('converged')
  })

  it('a narrow tolerance freezes two clusters', () => {
    const run = runAlgorithm(boundedConfidenceAlgorithm(completeGraph(4), [0, 0.1, 0.9, 1], 0.2), {
      seed: 0,
      maxSteps: 50,
    })
    expect(Array.from(run.state.x).map((v) => Math.round(v * 1e9) / 1e9)).toEqual([0.05, 0.05, 0.95, 0.95])
    expect(run.status).toBe('converged')
  })

  it('rejects a negative confidence', () => {
    expect(() => boundedConfidenceAlgorithm(pathGraph(3), [0, 0, 0], -1)).toThrow(/confidence/)
  })
})
