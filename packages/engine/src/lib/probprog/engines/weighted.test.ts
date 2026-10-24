import { describe, expect, it } from 'vitest'
import { expectPosteriorMoments, MODELS } from '../../../testing/probprog.ts'
import { normal } from '../../dists/index.ts'
import { key, normal as normalDraws } from '../../random/index.ts'
import { add, item, mul, type Value } from '../../tensor/index.ts'
import { analyse } from '../analysis.ts'
import { factor, type Model, markov, observe, plate, sample } from '../primitives.ts'
import { abcEngine, abcRejection, abcSmc, simulator } from './abc.ts'
import { conjugateInference } from './conjugate.ts'
import { gaussianInference } from './gaussian.ts'
import { importanceInference } from './importance.ts'
import { particleFilter, smcEngine, temperedSmc } from './smc.ts'

const ys = [0.4, 1.1, 0.7, 1.8, 2.4, 1.9]
const walk: Model = () => {
  markov(
    't',
    ys.length,
    (t, x: Value | number) => {
      const z = sample(`x${t}`, normal(mul(x, 0.9), t === 0 ? 1.5 : 0.6))
      observe(`y${t}`, normal(z, 0.5), ys[t] ?? 0)
      return z
    },
    0 as Value | number,
  )
}

describe('sequential Monte Carlo', () => {
  const exact = gaussianInference(walk, undefined, { method: 'kalman' })

  it('filters a linear-Gaussian chain: evidence and smoothed means near the Kalman smoother', () => {
    const runs = [0, 1, 2].map((s) => particleFilter(walk, undefined, { key: key(s), numParticles: 1500 }))
    const post = runs[0] as (typeof runs)[number]
    expect(post.status).toBe('completed')
    // A bootstrap filter's log-evidence over these 6 informative steps has a standard error near 0.08 at 1500
    // particles; the mean of three runs is within 0.15 (about 3.5 standard errors).
    const mean = runs.reduce((a, r) => a + (r.logEvidence?.value ?? 0), 0) / runs.length
    expect(Math.abs(mean - (exact.logEvidence?.value ?? 0))).toBeLessThan(0.15)
    const last = `x${ys.length - 1}`
    // The final step's particles are a weighted sample from the filtering distribution, which is the smoothed one.
    expectPosteriorMoments(post, last, { mean: item(exact.mean(last)), sd: Math.sqrt(item(exact.variance(last))) })
  })

  it('ends with status failed when every particle has zero weight', () => {
    const impossible: Model = () => {
      markov(
        't',
        2,
        (t, carry: number) => {
          sample(`x${t}`, normal(0, 1))
          factor(`f${t}`, Number.NEGATIVE_INFINITY)
          return carry
        },
        0,
      )
    }
    expect(particleFilter(impossible, undefined, { key: key(1), numParticles: 20 }).status).toBe('failed')
  })

  it('tempers a static program to its posterior and evidence', () => {
    const model = MODELS['normalMean'] as Model
    const c = conjugateInference(model, undefined)
    const post = temperedSmc(model, undefined, { key: key(2), numParticles: 600 })
    expect(post.status).toBe('converged')
    expect(Math.abs((post.logEvidence?.value ?? 0) - (c.logEvidence?.value ?? 0))).toBeLessThan(0.1)
    expectPosteriorMoments(post, 'mu', { mean: item(c.mean('mu')) }, { z: 5 })
  })

  it('chooses filtering or tempering from the analysis', () => {
    expect(smcEngine().applies(analyse(walk, undefined)).reason).toMatch(/markov loop/)
    expect(smcEngine().applies(analyse(MODELS['normalMean'] as Model, undefined)).reason).toMatch(/tempering/)
  })
})

describe('importance sampling', () => {
  it('weights prior draws by the likelihood, estimating the evidence', () => {
    const model = MODELS['normalMean'] as Model
    const c = conjugateInference(model, undefined)
    const post = importanceInference(model, undefined, { key: key(3), numSamples: 4000 })
    expect(Math.abs((post.logEvidence?.value ?? 0) - (c.logEvidence?.value ?? 0))).toBeLessThan(
      4 * (post.logEvidence?.standardError ?? 1),
    )
    expectPosteriorMoments(post, 'mu', { mean: item(c.mean('mu')), sd: Math.sqrt(item(c.variance('mu'))) })
    expect(post.diagnostics['paretoK']).toBeLessThan(0.7)
  })

  it('fails clearly when no draw explains the data', () => {
    const model: Model = () => {
      sample('mu', normal(0, 1))
      factor('impossible', Number.NEGATIVE_INFINITY)
    }
    expect(importanceInference(model, undefined, { key: key(4), numSamples: 50 }).status).toBe('failed')
  })
})

describe('approximate Bayesian computation', () => {
  const y = [0.5, 1.2, 1.6, 1.9, 2, 2.1, 2.4, 2.8, 3.5, 2]
  // The same program with a simulator in place of the normal likelihood.
  const simulated: Model = () => {
    const m = sample('m', normal(0, 3))
    observe(
      'y',
      simulator('noise', (k) => add(m, normalDraws(k, [y.length])), { eventShape: [y.length] }),
      y,
    )
  }
  const exactModel: Model = () => {
    const m = sample('m', normal(0, 3))
    plate('n', y.length, () => observe('y', normal(m, 1), y))
  }
  const exact = conjugateInference(exactModel, undefined)

  it('finds no likelihood in a simulator program and plans ABC', () => {
    const a = analyse(simulated, undefined)
    expect(a.hasLikelihood).toBe(false)
    expect(a.simulators).toEqual(['y'])
    expect(abcEngine().applies(a).applies).toBe(true)
  })

  it('approximates the posterior by rejection and by SMC', () => {
    const m = item(exact.mean('m'))
    const rej = abcRejection(simulated, undefined, { key: key(5), numSamples: 2000, quantile: 0.05 })
    expect(Math.abs(item(rej.mean('m')) - m)).toBeLessThan(0.3)
    const smc = abcSmc(simulated, undefined, { key: key(6), numParticles: 200, numGenerations: 5 })
    expect(Math.abs(item(smc.mean('m')) - m)).toBeLessThan(0.3)
    expect(Number(smc.diagnostics['epsilon'])).toBeLessThan(Number(rej.diagnostics['epsilon']) * 2)
  })
})
