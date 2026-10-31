import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import {
  barabasiAlbertGraph,
  completeGraph,
  cycleGraph,
  epidemicFinalSize,
  epidemicThresholds,
  erdosRenyiGraph,
  herdImmunityThreshold,
  homogeneousFinalSize,
  independentCascadeAlgorithm,
  linearThresholdAlgorithm,
  networkEpidemicAlgorithm,
  pathGraph,
  sirTransmissibility,
  starGraph,
  vaccinationSet,
  wattsCascadeCondition,
} from './index.ts'

const degreesOf = (g: { readonly out: { readonly indptr: Int32Array } }): number[] =>
  Array.from({ length: g.out.indptr.length - 1 }, (_, v) => (g.out.indptr[v + 1] ?? 0) - (g.out.indptr[v] ?? 0))
const mean = (x: readonly number[]): number => x.reduce((a, b) => a + b, 0) / x.length

describe('network epidemic: Gillespie on a graph', () => {
  it('an edge transmits with probability beta / (beta + gamma)', () => {
    const g = pathGraph(2)
    let reached = 0
    const trials = 3000
    for (let seed = 0; seed < trials; seed++) {
      const run = runAlgorithm(networkEpidemicAlgorithm(g, { beta: 1, gamma: 2, seeds: [0] }), { seed, maxSteps: 10 })
      if (run.state.counts[3] === 2) reached += 1
    }
    expect(sirTransmissibility(1, 2)).toBeCloseTo(1 / 3, 12)
    expect(Math.abs(reached / trials - 1 / 3)).toBeLessThan(0.035)
  })

  it('an infectious period is exponential with mean 1 / gamma', () => {
    const times: number[] = []
    for (let seed = 0; seed < 2000; seed++) {
      const run = runAlgorithm(networkEpidemicAlgorithm(pathGraph(1), { beta: 1, gamma: 2, seeds: [0] }), {
        seed,
        maxSteps: 5,
      })
      times.push(run.state.t)
    }
    expect(Math.abs(mean(times) - 0.5)).toBeLessThan(0.05)
  })

  it('the final size of an SIR epidemic on a random graph follows the generating-function relation', () => {
    const c = 4
    const results: number[] = []
    const predicted: number[] = []
    for (let s = 0; s < 3; s++) {
      const g = erdosRenyiGraph(key(100 + s), 1500, c / 1499)
      const t = sirTransmissibility(1, 1)
      predicted.push(epidemicFinalSize(degreesOf(g), t).finalSize)
      const run = runAlgorithm(networkEpidemicAlgorithm(g, { beta: 1, gamma: 1, seeds: 20 }), {
        seed: s,
        maxSteps: 100000,
      })
      expect(run.status).toBe('converged')
      results.push(run.state.ever / g.n)
    }
    expect(mean(predicted)).toBeGreaterThan(0.75)
    expect(Math.abs(mean(results) - mean(predicted))).toBeLessThan(0.04)
  })

  it('below the percolation threshold the epidemic stays small', () => {
    const g = erdosRenyiGraph(key(7), 1500, 4 / 1499)
    const t = epidemicThresholds(g).percolation
    expect(t).toBeGreaterThan(0.2)
    expect(t).toBeLessThan(0.3)
    const run = runAlgorithm(networkEpidemicAlgorithm(g, { beta: 1, gamma: 9, seeds: 5 }), {
      seed: 1,
      maxSteps: 100000,
    })
    expect(run.state.ever / g.n).toBeLessThan(0.05)
  })

  it('SIS settles at the endemic level of the complete graph', () => {
    // On K_N the infection of a susceptible node has rate beta I; the endemic share is 1 - gamma / (beta (N - 1)).
    const n = 100
    const beta = 0.05
    const shares: number[] = []
    for (let seed = 0; seed < 20; seed++) {
      const run = runAlgorithm(
        networkEpidemicAlgorithm(completeGraph(n), { model: 'SIS', beta, gamma: 1, seeds: 30, horizon: 25 }),
        {
          seed,
          maxSteps: 1000000,
        },
      )
      shares.push((run.state.counts[2] ?? 0) / n)
    }
    expect(Math.abs(mean(shares) - (1 - 1 / (beta * (n - 1))))).toBeLessThan(0.04)
  })

  it('SIS below the threshold dies out', () => {
    const run = runAlgorithm(
      networkEpidemicAlgorithm(cycleGraph(30), { model: 'SIS', beta: 0.2, gamma: 1, seeds: 10 }),
      {
        seed: 3,
        maxSteps: 1000000,
      },
    )
    expect(run.status).toBe('converged')
    expect(run.state.counts[2]).toBe(0)
  })

  it('SEIR passes through the exposed state and conserves the population', () => {
    const g = erdosRenyiGraph(key(2), 300, 0.02)
    const rec = recorder()
    const run = runAlgorithm(networkEpidemicAlgorithm(g, { model: 'SEIR', beta: 1, gamma: 1, sigma: 2, seeds: 3 }), {
      seed: 4,
      maxSteps: 100000,
      tracer: rec,
      policy: policies.debug,
    })
    expect(run.state.counts.reduce((a, b) => a + b, 0)).toBe(300)
    expect(run.state.counts[1]).toBe(0)
    expect((run.state.counts[0] ?? 0) + (run.state.counts[3] ?? 0)).toBe(300)
    expect(run.state.ever).toBe(run.state.counts[3])
    const exposed = rec.events.filter((e) => e.type === 'metric' && e.name === 'epidemic/infected')
    expect(exposed.length).toBeGreaterThan(0)
  })

  it('vaccinated nodes are never infected and a vaccinated hub protects its leaves', () => {
    const g = starGraph(20)
    const run = runAlgorithm(networkEpidemicAlgorithm(g, { beta: 5, gamma: 1, seeds: [1], immune: [0] }), {
      seed: 0,
      maxSteps: 1000,
    })
    expect(run.state.ever).toBe(1)
    expect(run.state.status[0]).toBe(3)
    expect(() =>
      runAlgorithm(networkEpidemicAlgorithm(g, { seeds: [0], immune: [0] }), { seed: 0, maxSteps: 1 }),
    ).toThrow(/immune/)
  })

  it('resumes exactly from a checkpoint', () => {
    const g = erdosRenyiGraph(key(5), 200, 0.03)
    const alg = networkEpidemicAlgorithm(g, { beta: 1, gamma: 0.5, seeds: 3 })
    const straight = runAlgorithm(alg, { seed: 9, maxSteps: 60 })
    const early = runAlgorithm(alg, { seed: 9, maxSteps: 25 })
    const resumed = runAlgorithm(alg, {
      seed: 9,
      maxSteps: 60 - early.steps,
      start: { state: alg.serialise?.(early.state) ?? null, step: early.steps, serialised: true },
    })
    expect(Array.from(resumed.state.status)).toEqual(Array.from(straight.state.status))
    expect(resumed.state.t).toBe(straight.state.t)
  })

  it('rejects bad rates and models', () => {
    expect(() => networkEpidemicAlgorithm(pathGraph(3), { beta: -1 })).toThrow(/beta/)
    expect(() => networkEpidemicAlgorithm(pathGraph(3), { model: 'SI' as 'SIR' })).toThrow(/model/)
    expect(() => sirTransmissibility(0, 0)).toThrow(/rates/)
  })
})

describe('epidemic results', () => {
  it('on a 3-regular graph the final size is 1 - ((1 - T) / T)^3 above T = 1/2', () => {
    const degrees = Array<number>(60).fill(3)
    const r = epidemicFinalSize(degrees, 0.75)
    expect(r.u).toBeCloseTo(1 / 3, 10)
    expect(r.finalSize).toBeCloseTo(1 - 1 / 27, 10)
    expect(r.threshold).toBeCloseTo(0.5, 12)
    expect(r.r0).toBeCloseTo(1.5, 12)
    expect(epidemicFinalSize(degrees, 0.5).finalSize).toBeLessThan(1e-3)
    expect(epidemicFinalSize(degrees, 0.3).finalSize).toBeCloseTo(0, 12)
  })

  it('for Poisson degrees it is the mean-field relation z = 1 - exp(-R0 z)', () => {
    const c = 4
    const degrees: number[] = []
    let pmf = Math.exp(-c)
    for (let k = 0; k < 40; k++) {
      for (let i = 0; i < Math.round(pmf * 4e5); i++) degrees.push(k)
      pmf = (pmf * c) / (k + 1)
    }
    const r = epidemicFinalSize(degrees, 0.5)
    const z = homogeneousFinalSize(2)
    expect(r.finalSize).toBeCloseTo(z, 3)
    expect(z).toBeCloseTo(1 - Math.exp(-2 * z), 13)
    expect(homogeneousFinalSize(1)).toBe(0)
    expect(homogeneousFinalSize(5)).toBeCloseTo(0.99303, 4)
  })

  it('hubs lower the thresholds: the star against the cycle', () => {
    const star = epidemicThresholds(starGraph(16))
    expect(star.lambdaMax).toBeCloseTo(4, 8)
    expect(star.spectral).toBeCloseTo(0.25, 8)
    const cycle = epidemicThresholds(cycleGraph(12))
    expect(cycle.lambdaMax).toBeCloseTo(2, 8)
    expect(cycle.heterogeneous).toBeCloseTo(0.5, 12)
    expect(cycle.percolation).toBeCloseTo(1, 12)
    expect(epidemicThresholds(pathGraph(2)).lambdaMax).toBe(1)
  })
})

describe('vaccination', () => {
  it('chooses the stated number of nodes by each strategy; targeting takes the hubs', () => {
    const g = barabasiAlbertGraph(key(3), 300, 2)
    const degree = (v: number): number => (g.out.indptr[v + 1] ?? 0) - (g.out.indptr[v] ?? 0)
    const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
    const by = (s: 'random' | 'degree' | 'acquaintance'): number[] => vaccinationSet(g, 30, s, key(1))
    for (const s of ['random', 'degree', 'acquaintance'] as const) expect(new Set(by(s)).size).toBe(30)
    expect(mean(by('degree').map(degree))).toBeGreaterThan(2 * mean(by('acquaintance').map(degree)))
    expect(mean(by('acquaintance').map(degree))).toBeGreaterThan(1.5 * mean(by('random').map(degree)))
    expect(vaccinationSet(starGraph(6), 1, 'degree', key(0))).toEqual([0])
    expect(() => vaccinationSet(starGraph(3), 9, 'random', key(0))).toThrow(/count/)
    expect(() => vaccinationSet(starGraph(3), 1, 'x' as 'random', key(0))).toThrow(/strategy/)
  })

  it('on a scale-free network vaccinating hubs stops an epidemic that random vaccination does not', () => {
    const g = barabasiAlbertGraph(key(4), 1500, 2)
    const attack = (immune: number[]): number => {
      let total = 0
      for (let s = 0; s < 4; s++) {
        const eligible = Array.from({ length: g.n }, (_, v) => v).filter((v) => !immune.includes(v))
        total +=
          runAlgorithm(
            networkEpidemicAlgorithm(g, { beta: 1, gamma: 1, seeds: eligible.slice(s * 50, s * 50 + 5), immune }),
            { seed: s, maxSteps: 200000 },
          ).state.ever / g.n
      }
      return total / 4
    }
    const doses = 150
    expect(attack(vaccinationSet(g, doses, 'degree', key(0)))).toBeLessThan(
      0.5 * attack(vaccinationSet(g, doses, 'random', key(0))),
    )
  })

  it('has the herd-immunity threshold 1 - 1 / R0', () => {
    expect(herdImmunityThreshold(4)).toBe(0.75)
    expect(herdImmunityThreshold(0.5)).toBe(0)
    expect(() => herdImmunityThreshold(-1)).toThrow(/R0/)
  })
})

describe('independent cascade', () => {
  it('with probability one it floods a path one node per wave, and with zero it stops at the seed', () => {
    const run = runAlgorithm(independentCascadeAlgorithm(pathGraph(6), [0], 1), { seed: 0, maxSteps: 20 })
    expect(run.state.adopted).toBe(6)
    expect(run.state.sizes).toEqual([1, 1, 1, 1, 1, 1, 0])
    expect(run.status).toBe('converged')
    const none = runAlgorithm(independentCascadeAlgorithm(pathGraph(6), [0], 0), { seed: 0, maxSteps: 20 })
    expect(none.state.adopted).toBe(1)
    expect(none.steps).toBe(1)
  })

  it('from the centre of a star the expected size is 1 + p (n - 1)', () => {
    const sizes: number[] = []
    for (let seed = 0; seed < 500; seed++) {
      sizes.push(
        runAlgorithm(independentCascadeAlgorithm(starGraph(10), [0], 0.3), { seed, maxSteps: 5 }).state.adopted,
      )
    }
    expect(Math.abs(mean(sizes) - (1 + 0.3 * 10))).toBeLessThan(0.2)
  })

  it('weights are the activation probabilities when asked', () => {
    const g = pathGraph(3)
    const heavy = runAlgorithm(independentCascadeAlgorithm(g, [0], 0, { weighted: true }), { seed: 0, maxSteps: 10 })
    expect(heavy.state.adopted).toBe(3)
  })

  it('checkpoints and resumes exactly', () => {
    const g = erdosRenyiGraph(key(1), 150, 0.03)
    const alg = independentCascadeAlgorithm(g, 2, 0.4)
    const straight = runAlgorithm(alg, { seed: 5, maxSteps: 8 })
    const early = runAlgorithm(alg, { seed: 5, maxSteps: 3 })
    const resumed = runAlgorithm(alg, {
      seed: 5,
      maxSteps: 8 - early.steps,
      start: { state: alg.serialise?.(early.state) ?? null, step: early.steps, serialised: true },
    })
    expect(Array.from(resumed.state.status)).toEqual(Array.from(straight.state.status))
  })
})

describe('linear threshold cascade', () => {
  it('Watts: on a clique a single seed tips everyone exactly when 1 / (n - 1) reaches the threshold', () => {
    const k5 = completeGraph(5)
    expect(
      runAlgorithm(linearThresholdAlgorithm(k5, [0], { threshold: 0.25 }), { seed: 0, maxSteps: 10 }).state.adopted,
    ).toBe(5)
    expect(
      runAlgorithm(linearThresholdAlgorithm(k5, [0], { threshold: 0.26 }), { seed: 0, maxSteps: 10 }).state.adopted,
    ).toBe(1)
  })

  it('on a path a threshold of one half carries the cascade to the end and 0.6 stops it', () => {
    const run = runAlgorithm(linearThresholdAlgorithm(pathGraph(6), [0], { threshold: 0.5 }), { seed: 0, maxSteps: 20 })
    expect(run.state.adopted).toBe(6)
    expect(run.state.sizes.slice(0, 6)).toEqual([1, 1, 1, 1, 1, 1])
    expect(
      runAlgorithm(linearThresholdAlgorithm(pathGraph(6), [0], { threshold: 0.6 }), { seed: 0, maxSteps: 20 }).state
        .adopted,
    ).toBe(1)
  })

  it('with uniform random thresholds the centre of a star adopts with probability (active leaves) / (leaves)', () => {
    const adopted: number[] = []
    for (let seed = 0; seed < 600; seed++) {
      adopted.push(
        runAlgorithm(linearThresholdAlgorithm(starGraph(6), [1, 2, 3]), { seed, maxSteps: 10 }).state.adopted,
      )
    }
    // Three seed leaves out of six: the centre joins with probability 1/2, and then every leaf follows.
    expect(Math.abs(mean(adopted) - (3 + 0.5 * 4))).toBeLessThan(0.3)
  })

  it('Watts window: sparse and dense random graphs do not cascade, intermediate ones do', () => {
    const phi = 0.18
    const poisson = (c: number): number[] => {
      const out: number[] = []
      let p = Math.exp(-c)
      for (let k = 0; k < 60; k++) {
        for (let i = 0; i < Math.round(p * 1e5); i++) out.push(k)
        p = (p * c) / (k + 1)
      }
      return out
    }
    expect(wattsCascadeCondition(poisson(0.5), phi)).toBeLessThan(1)
    expect(wattsCascadeCondition(poisson(4), phi)).toBeGreaterThan(1)
    expect(wattsCascadeCondition(poisson(15), phi)).toBeLessThan(1)
    const biggest = (c: number): number => {
      let best = 0
      for (let seed = 0; seed < 25; seed++) {
        const g = erdosRenyiGraph(key(300 + seed), 1000, c / 999)
        const run = runAlgorithm(linearThresholdAlgorithm(g, 1, { threshold: phi }), { seed, maxSteps: 200 })
        best = Math.max(best, run.state.adopted / g.n)
      }
      return best
    }
    expect(biggest(4)).toBeGreaterThan(0.3)
    expect(biggest(15)).toBeLessThan(0.02)
  })

  it('rejects a threshold outside [0, 1]', () => {
    expect(() => linearThresholdAlgorithm(pathGraph(3), [0], { threshold: 1.5 })).toThrow(/threshold/)
  })
})
