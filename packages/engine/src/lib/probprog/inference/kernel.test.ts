// The kernels of the handler layer: the invariance of the posterior under each (exact draws in, same law out),
// that a deliberately wrong kernel fails that test, and the plumbing (chains, thinning, MCMC adapter, discrete sites).
import { describe, expect, it } from 'vitest'
import { expectInvariant } from '../../../testing/sampling.ts'
import { bernoulli, gamma, normal, poisson } from '../../dists/index.ts'
import { foldIn, key as makeKey } from '../../random/index.ts'
import { type MCMCKernel, type MCMCState, sampleChain } from '../../sampling/index.ts'
import { item, mean, toFlat, type Value } from '../../tensor/index.ts'
import { type Model, observe, sample } from '../primitives.ts'
import { defineKernel, metropolis, randomWalkKernel, resimulationKernel } from './kernel.ts'

// The posterior of m given y = 1.2 under m ~ N(0, 1), y ~ N(m, 1) is N(0.6, 0.5).
const NORMAL: Model = () => {
  const m = sample('m', normal(0, 1))
  observe('y', normal(m, 1), 1.2)
}
const TRUTH = { mean: [0.6], sd: [Math.sqrt(0.5)] }

/** n exact draws of the posterior N(0.6, 0.5), `[n, 1]`. */
function exactDraws(n: number): Float64Array {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = 0.6 + Math.sqrt(0.5) * item(normal(0, 1).sample(foldIn(makeKey(31), i), []))
  return out
}

describe('invariance of the posterior', () => {
  const draws = exactDraws(1500)

  it('resimulation (lightweight Metropolis-Hastings)', () => {
    expectInvariant(resimulationKernel().on(NORMAL, undefined).asMcmcKernel(), draws, TRUTH, { steps: 2 })
  })

  it('random walk on the unconstrained scale', () => {
    expectInvariant(randomWalkKernel({ scale: 1 }).on(NORMAL, undefined).asMcmcKernel(), draws, TRUTH, { steps: 2 })
  })

  it('the kernel of the plan: a few lines on regenerate and metropolis', () => {
    const mine = defineKernel({
      name: 'mine',
      step: (k, trace, target) => metropolis(foldIn(k, 1), trace, target.regenerate(trace, ['m'], foldIn(k, 0))),
    })
    expectInvariant(mine.on(NORMAL, undefined).asMcmcKernel(), draws, TRUTH, { steps: 2 })
  })

  it('fails for a kernel that never rejects (it forgets the likelihood and returns the prior)', () => {
    const wrong = defineKernel({
      name: 'always-accept',
      step: (k, trace, target) => ({
        trace: target.regenerate(trace, ['m'], k).trace,
        accepted: true,
        acceptanceRate: 1,
      }),
    })
    expect(() => expectInvariant(wrong.on(NORMAL, undefined).asMcmcKernel(), draws, TRUTH, { steps: 2 })).toThrow()
  })

  it('a constrained site: the random walk keeps a positive rate and the gamma-Poisson posterior', () => {
    // Gamma(2, 1.5) prior, one count of 3: Gamma(5, 2.5) posterior (mean 2, variance 0.8).
    const model: Model = () => {
      const r = sample('r', gamma(2, 1.5))
      observe('k', poisson(r), 3)
    }
    const n = 1500
    const exact = new Float64Array(n)
    // Unconstrained draws are log r; gamma draws by the library's own sampler.
    const post = gamma(5, 2.5)
    for (let i = 0; i < n; i++) exact[i] = Math.log(item(post.sample(foldIn(makeKey(8), i), []) as Value))
    const um = randomWalkKernel({ scale: 1.5 }).on(model, undefined).asMcmcKernel()
    const out = expectInvariantLogScale(um, exact)
    expect(out.every((v) => Number.isFinite(v))).toBe(true)
  })
})

/** Invariance on the log scale: the exact draws are log r, and the sd of log r is estimated from the draws themselves. */
function expectInvariantLogScale(
  kernel: MCMCKernel<MCMCState & { readonly trace: unknown }>,
  exact: Float64Array,
): Float64Array {
  const n = exact.length
  const m0 = exact.reduce((a, b) => a + b, 0) / n
  const s0 = Math.sqrt(exact.reduce((a, b) => a + (b - m0) ** 2, 0) / n)
  return expectInvariant(kernel, exact, { mean: [m0], sd: [s0] }, { steps: 2, z: 5 })
}

describe('chains', () => {
  const kernel = resimulationKernel().on(NORMAL, undefined)

  it('is deterministic in the key, and keeps shapes, thinning and warm-up straight', () => {
    const a = kernel.run(makeKey(2), { numDraws: 40, numWarmup: 10, thin: 3 })
    const b = kernel.run(makeKey(2), { numDraws: 40, numWarmup: 10, thin: 3 })
    expect(a.draws['m']?.shape).toEqual([40])
    expect(toFlat(a.draws['m'] as Value)).toEqual(toFlat(b.draws['m'] as Value))
    expect(a.logDensity.length).toBe(40)
    expect(a.acceptanceRate).toBeGreaterThan(0)
    expect(a.acceptanceRate).toBeLessThanOrEqual(1)
  })

  it('a long chain recovers the posterior mean (Monte Carlo error from the chain)', () => {
    const c = kernel.run(makeKey(4), { numDraws: 6000, numWarmup: 200 })
    const m = item(mean(c.draws['m'] as Value))
    // Resimulation from the prior mixes slowly (acceptance about 0.5): allow an effective size of 1000.
    expect(Math.abs(m - 0.6)).toBeLessThan((4 * Math.sqrt(0.5)) / Math.sqrt(1000))
  })

  it('a resumed chain continues from the state it stopped in', () => {
    const first = kernel.run(makeKey(6), { numDraws: 5 })
    const second = kernel.run(makeKey(7), { numDraws: 5, state: first.state })
    expect(second.draws['m']?.shape).toEqual([5])
  })

  it('runs in the sampling library as an MCMCKernel', () => {
    const k = randomWalkKernel().on(NORMAL, undefined).asMcmcKernel()
    const chain = sampleChain(k, [0], { key: makeKey(5), numDraws: 30 })
    expect(chain.count).toBe(30)
    expect(chain.stats.acceptanceRate?.length).toBe(30)
  })

  it('moves discrete latent sites too (resimulation), against the exact posterior probability', () => {
    // c ~ Bernoulli(0.3); y ~ N(2c, 1) observed at 1.1.
    const model: Model = () => {
      const c = sample('c', bernoulli({ probs: 0.3 }))
      observe('y', normal(item(c) === 1 ? 2 : 0, 1), 1.1)
    }
    const w1 = 0.3 * Math.exp(-0.5 * (1.1 - 2) ** 2)
    const w0 = 0.7 * Math.exp(-0.5 * 1.1 ** 2)
    const exact = w1 / (w0 + w1)
    const c = resimulationKernel().on(model, undefined).run(makeKey(3), { numDraws: 4000, numWarmup: 100 })
    const p = item(mean(c.draws['c'] as Value))
    expect(Math.abs(p - exact)).toBeLessThan(4 * Math.sqrt((exact * (1 - exact)) / 1000))
    expect(() => resimulationKernel().on(model, undefined).asMcmcKernel()).toThrow(/continuous/)
  })

  it('a plain trace from step counts as an accepted move', () => {
    const stay = defineKernel({ name: 'stay', step: (_k, trace) => trace })
    const bound = stay.on(NORMAL, undefined)
    const s = bound.init(makeKey(1))
    const r = bound.step(makeKey(2), s)
    expect(r.move.accepted).toBe(false)
    expect(r.state).toBe(s)
  })
})
