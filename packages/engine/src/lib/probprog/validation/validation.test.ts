// The validation kernel on cases with known answers: the grid against closed forms, the agreement table, simulation-based
// calibration (exact samplers pass, deliberately wrong ones fail) and Geweke's test. Fixed seeds; seconds.
import { describe, expect, it } from 'vitest'
import { expectAgreement, expectCalibrated, expectGeweke, expectMiscalibrated } from '../../../testing/probprog.ts'
import { bernoulli, beta, gamma, multivariateNormal, normal, poisson } from '../../dists/index.ts'
import { adam } from '../../optim/index.ts'
import { normal as drawNormal, foldIn, key as makeKey } from '../../random/index.ts'
import { item, sum, tensor, type Value } from '../../tensor/index.ts'
import { conjugateInference, mcmc, svi } from '../engines/index.ts'
import { autoDelta, autoNormal } from '../guides.ts'
import { randomWalkKernel, resimulationKernel } from '../inference/index.ts'
import { type Model, observe, plate, sample } from '../primitives.ts'
import {
  agree,
  agreementMarkdown,
  agreementWithin,
  betaBernoulli,
  dirichletCategorical,
  distributionOracle,
  fromPosterior,
  gammaPoisson,
  gewekeTest,
  gridPosterior,
  normalNormal,
  rankDistribution,
  rankOf,
  sbc,
  weightedOracle,
} from './index.ts'

const z = (k: ReturnType<typeof makeKey>): number => item(drawNormal(k, []))

describe('exact helpers agree with the conjugate engine', () => {
  it('normal-normal', () => {
    const y = [1.2, 0.4, 2.1]
    const model: Model = () => {
      const mu = sample('mu', normal(0.5, 2))
      plate('n', y.length, () => observe('y', normal(mu, 0.7), y))
    }
    const e = normalNormal({ priorMean: 0.5, priorSd: 2, sigma: 0.7, y })
    const post = conjugateInference(model, undefined)
    expect(post.logEvidence?.value).toBeCloseTo(e.logEvidence, 12)
    expect(item(post.mean('mu'))).toBeCloseTo(item(e.posterior.mean()), 12)
  })

  it('beta-Bernoulli, gamma-Poisson and Dirichlet-categorical evidence', () => {
    const flips = [1, 0, 1, 1]
    const coin: Model = () => {
      const p = sample('p', beta(2, 3))
      plate('n', flips.length, () => observe('y', bernoulli({ probs: p }), flips))
    }
    expect(conjugateInference(coin, undefined).logEvidence?.value).toBeCloseTo(
      betaBernoulli({ a: 2, b: 3, y: flips }).logEvidence,
      12,
    )
    const counts = [3, 1, 4]
    const rate: Model = () => {
      const r = sample('r', gamma(2, 1.5))
      plate('n', counts.length, () => observe('k', poisson(r), counts))
    }
    expect(conjugateInference(rate, undefined).logEvidence?.value).toBeCloseTo(
      gammaPoisson({ shape: 2, rate: 1.5, counts }).logEvidence,
      12,
    )
    // Dirichlet-categorical against its beta marginals: the first probability of Dirichlet(alpha + counts).
    const d = dirichletCategorical({ alpha: [1, 2, 3], y: [0, 0, 1, 2] })
    expect(d.oracles('p')[0]?.mean).toBeCloseTo(3 / 10, 14)
  })
})

describe('gridPosterior', () => {
  it('one coordinate: the evidence and every moment of the closed form', () => {
    const y = [1.2, 0.4, 2.1]
    const model: Model = () => {
      const mu = sample('mu', normal(0.5, 2))
      plate('n', y.length, () => observe('y', normal(mu, 0.7), y))
    }
    const e = normalNormal({ priorMean: 0.5, priorSd: 2, sigma: 0.7, y })
    const g = gridPosterior(model, undefined)
    expect(g.logEvidence).toBeCloseTo(e.logEvidence, 9)
    const m = g.marginal('mu')
    expect(m.mean).toBeCloseTo(item(e.posterior.mean()), 9)
    expect(m.sd).toBeCloseTo(item(e.posterior.stddev()), 9)
    expect(m.cdf(item(e.posterior.mean()))).toBeCloseTo(0.5, 4)
    // The CDF is the cumulative trapezoid of the marginal: second order in the spacing, unlike the evidence and moments.
    expect(m.quantile(0.975)).toBeCloseTo(item(e.posterior.ppf(0.975)), 3)
  })

  it('a constrained coordinate: the beta posterior on the probability scale', () => {
    const flips = [1, 0, 1, 1, 1, 0, 1]
    const model: Model = () => {
      const p = sample('p', beta(2, 3))
      plate('n', flips.length, () => observe('y', bernoulli({ probs: p }), flips))
    }
    const e = betaBernoulli({ a: 2, b: 3, y: flips })
    const g = gridPosterior(model, undefined)
    expect(g.logEvidence).toBeCloseTo(e.logEvidence, 8)
    const m = g.marginal('p')
    expect(m.mean).toBeCloseTo(item(e.posterior.mean()), 7)
    for (const x of [0.3, 0.55, 0.8]) expect(m.cdf(x)).toBeCloseTo(item(e.posterior.cdf(x)), 3)
    expect(g.unconstrained('p').sd).toBeGreaterThan(0)
  })

  it('two correlated coordinates: evidence and marginals of a bivariate normal', () => {
    // Prior N(0, [[1, 0.8], [0.8, 1]]) and one noisy observation of the sum: all Gaussian.
    const cov = tensor([
      [1, 0.8],
      [0.8, 1],
    ])
    const model: Model = () => {
      const w = sample('w', multivariateNormal([0, 0], { covariance: cov }))
      observe('y', normal(sum(w), 0.5), 1.0)
    }
    // Evidence: y = w0 + w1 + noise ~ N(0, var), var = 1 + 1 + 1.6 + 0.25.
    const v = 1 + 1 + 1.6 + 0.25
    const exact = -0.5 * Math.log(2 * Math.PI * v) - 0.5 / v
    expect(gridPosterior(model, undefined, { points: 81 }).logEvidence).toBeCloseTo(exact, 6)
  })

  it('refuses a model with more than four coordinates', () => {
    const big: Model = () => {
      plate('n', 5, () => sample('x', normal(0, 1)))
    }
    expect(() => gridPosterior(big, undefined)).toThrow(/too many/)
  })
})

describe('agree', () => {
  const exact = normal(0.6, Math.sqrt(0.5))
  const oracle = { m: distributionOracle('m', exact) }
  const draws = (n: number, shift: number, sd: number, seed: number): number[] =>
    Array.from({ length: n }, (_, i) => shift + sd * z(foldIn(makeKey(seed), i)))

  it('an exact sampler agrees; a narrow, a shifted and a wide one are each caught by their own measure', () => {
    const table = agree(
      [
        { name: 'exact', draws: { m: draws(3000, 0.6, Math.sqrt(0.5), 1) } },
        { name: 'narrow', draws: { m: draws(3000, 0.6, 0.4 * Math.sqrt(0.5), 2) } },
        { name: 'shifted', draws: { m: draws(3000, 0.6 + 0.5 * Math.sqrt(0.5), Math.sqrt(0.5), 3) } },
      ],
      { oracle },
    )
    const bad = agreementWithin(table)
    expect(bad.map((b) => b.row.method)).toEqual(['narrow', 'shifted'])
    expect(bad[0]?.failed).toContain('sdRatio')
    expect(bad[1]?.failed).toContain('meanError')
    const row = (m: string) => table.rows.find((r) => r.method === m)
    expect(row('exact')?.ks).toBeLessThan(0.04)
    expect(row('exact')?.kl).toBeLessThan(0.02)
    expect((row('narrow')?.kl ?? 0) > (row('exact')?.kl ?? 1)).toBe(true)
    expect(row('shifted')?.wasserstein ?? 0).toBeCloseTo(0.5, 1)
    expect(agreementMarkdown(table).split('\n')).toHaveLength(5)
  })

  it('without an oracle a reference approximation stands in, and weights are honoured', () => {
    const reference = { name: 'long run', draws: { m: draws(6000, 0.6, Math.sqrt(0.5), 4) } }
    // Importance weights turning N(0.6, s) draws taken from a wider proposal into the target.
    const proposal = draws(6000, 0.6, 1.5, 5)
    const sd = Math.sqrt(0.5)
    const weights = proposal.map(
      (x) => Math.exp(-0.5 * ((x - 0.6) / sd) ** 2 + 0.5 * ((x - 0.6) / 1.5) ** 2) * (1.5 / sd),
    )
    const table = agree([reference, { name: 'weighted', draws: { m: proposal }, weights }])
    expect(table.rows).toHaveLength(1)
    expectAgreement(table, { ks: 0.05, sdRatio: [0.9, 1.1] })
  })

  it('reads posteriors from the engines', () => {
    const model: Model = () => {
      const m = sample('m', normal(0, 1))
      observe('y', normal(m, 1), 1.2)
    }
    const a = fromPosterior('exact', conjugateInference(model, undefined), { key: makeKey(2) })
    const table = agree([a], { oracle })
    expectAgreement(table, { ks: 0.05 })
  })

  it('weightedOracle is a distribution: its quantile inverts its CDF', () => {
    const o = weightedOracle('x', draws(500, 1, 2, 9))
    for (const p of [0.1, 0.5, 0.9]) expect(o.cdf(o.quantile(p))).toBeCloseTo(p, 6)
  })
})

describe('simulation-based calibration', () => {
  // m ~ N(0, 1), y ~ N(m, 1): the posterior is N(y / 2, 1 / 2).
  const simulate = (k: ReturnType<typeof makeKey>): { data: number; truth: Record<string, number> } => {
    const m = z(foldIn(k, 0))
    return { data: m + z(foldIn(k, 1)), truth: { m } }
  }
  const sampler =
    (shift: number, scale: number, n = 19) =>
    (y: number, k: ReturnType<typeof makeKey>): Record<string, number[]> => ({
      m: Array.from({ length: n }, (_, i) => y / 2 + shift + scale * Math.SQRT1_2 * z(foldIn(k, i))),
    })
  const run = (fit: (y: number, k: ReturnType<typeof makeKey>) => Record<string, number[]>, seed = 1) =>
    sbc({ simulate, fit, key: makeKey(seed), replications: 200, numRanks: 19 })

  it('the exact distribution of a rank is uniform', () => {
    expect(rankDistribution(4)).toEqual([0.2, 0.2, 0.2, 0.2, 0.2])
    expect(rankOf(2.5, [1, 2, 3, 4])).toBe(2)
    expect(rankOf(0, [1, 2])).toBe(0)
    expect(rankOf(9, [1, 2])).toBe(2)
  })

  it('passes for the exact posterior sampler', () => {
    const r = run(sampler(0, 1))
    expectCalibrated(r)
    expect(r.quantities[0]?.diagnosis).toBe('uniform')
  })

  it('fails, and says why, for a guide that is too narrow, too wide, or biased', () => {
    const narrow = run(sampler(0, 0.4))
    expectMiscalibrated(narrow, /U-shaped/)
    const wide = run(sampler(0, 2.5))
    expectMiscalibrated(wide, /hump/)
    const low = run(sampler(-0.7, 1))
    expectMiscalibrated(low, /biased low/)
  })

  it('passes for NUTS on the conjugate program, and fails for NUTS given a wrong likelihood', () => {
    const program = (sigma: number) => (y: number) => (): void => {
      const m = sample('m', normal(0, 1))
      observe('y', normal(m, sigma), y)
    }
    const nuts =
      (sigma: number) =>
      (y: number, k: ReturnType<typeof makeKey>): Record<string, number[]> => {
        // 95 draws thinned to the 19 ranked: consecutive NUTS draws are autocorrelated, and SBC needs independent draws.
        const post = mcmc(program(sigma)(y), undefined, { key: k, numChains: 1, numWarmup: 60, numDraws: 95 })
        return { m: Array.from(post.draws?.['m']?.data ?? []) }
      }
    const small = (fit: typeof nuts extends (s: number) => infer F ? F : never) =>
      sbc({ simulate, fit, key: makeKey(3), replications: 60, numRanks: 19 })
    expectCalibrated(small(nuts(1)))
    // Fitting with the observation sd 0.5 instead of 1: the posterior is too narrow and too pulled to the data.
    expectMiscalibrated(small(nuts(0.4)))
  })
})

describe('simulation-based calibration of variational guides', () => {
  const simulate = (k: ReturnType<typeof makeKey>): { data: number; truth: Record<string, number> } => {
    const m = z(foldIn(k, 0))
    return { data: m + z(foldIn(k, 1)), truth: { m } }
  }
  const program = (y: number) => (): void => {
    const m = sample('m', normal(0, 1))
    observe('y', normal(m, 1), y)
  }
  const fit =
    (kind: 'delta' | 'normal') =>
    (y: number, k: ReturnType<typeof makeKey>): Record<string, number[]> => {
      const model = program(y)
      const guide = kind === 'delta' ? autoDelta(model) : autoNormal(model)
      const r = svi(model, undefined, {
        key: k,
        guide,
        numSteps: 120,
        optimizer: adam({ learningRate: (t: number) => 0.15 / (1 + t / 30) }),
      })
      return { m: Array.from(r.posterior.sample(foldIn(k, 9), 19)['m']?.data ?? []) }
    }

  const options = { simulate, key: makeKey(4), numRanks: 19 }

  it('a point-estimate guide fails: all its draws are one number', () => {
    const delta = sbc({ ...options, replications: 30, fit: fit('delta') })
    expectMiscalibrated(delta, /U-shaped/)
  })

  it('a mean-field normal guide on a Gaussian posterior is calibrated', () => {
    expectCalibrated(sbc({ ...options, replications: 60, fit: fit('normal') }))
  })
})

describe("Geweke's joint distribution test", () => {
  // m ~ N(0, 1), y | m ~ N(m, 1): the exact transition draws m from N(y / 2, 1 / 2).
  const base = {
    key: makeKey(5),
    numSamples: 4000,
    numIterations: 4000,
    simulate: (k: ReturnType<typeof makeKey>) => {
      const m = z(foldIn(k, 0))
      return { state: m, data: m + z(foldIn(k, 1)) }
    },
    resimulateData: (m: number, k: ReturnType<typeof makeKey>) => m + z(k),
    statistics: {
      m: (m: number) => m,
      'm^2': (m: number) => m * m,
      y: (_m: number, y: number) => y,
      'm y': (m: number, y: number) => m * y,
    },
  }

  it('passes for the exact transition', () => {
    expectGeweke(gewekeTest({ ...base, transition: (_m, y, k) => y / 2 + Math.SQRT1_2 * z(k) }))
  })

  it('fails for a transition with the wrong spread, and for one with the wrong centre', () => {
    const narrow = gewekeTest({ ...base, transition: (_m, y, k) => y / 2 + 0.5 * z(k) })
    expect(narrow.passes).toBe(false)
    const shifted = gewekeTest({ ...base, transition: (_m, y, k) => y / 2 + 0.3 + Math.SQRT1_2 * z(k) })
    expect(shifted.passes).toBe(false)
  })

  it('tests the kernels of the handler layer on programs', () => {
    const model = (y: number) => (): void => {
      const m = sample('m', normal(0, 1))
      observe('y', normal(m, 1), y)
    }
    for (const kernel of [resimulationKernel(), randomWalkKernel({ scale: 1.2 })]) {
      type S = ReturnType<ReturnType<typeof kernel.on<void>>['init']>
      const r = gewekeTest<S, number>({
        key: makeKey(8),
        numSamples: 3000,
        numIterations: 6000,
        simulate: (k) => {
          const bound = kernel.on(model(0), undefined)
          const state = bound.init(foldIn(k, 0))
          const m = item((state.trace.sites.get('m') as { value: Value }).value)
          return { state, data: m + z(foldIn(k, 1)) }
        },
        resimulateData: (s, k) => item((s.trace.sites.get('m') as { value: Value }).value) + z(k),
        transition: (s, y, k) => {
          const bound = kernel.on(model(y), undefined)
          const fresh = { trace: bound.target.update(s.trace, {}).trace, logDensity: 0 }
          return bound.step(k, fresh).state
        },
        statistics: {
          m: (s) => item((s.trace.sites.get('m') as { value: Value }).value),
          y: (_s, y) => y,
        },
      })
      expectGeweke(r)
    }
  })
})
