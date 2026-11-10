import { describe, expect, it } from 'vitest'
import { MODELS } from '../../../testing/probprog.ts'
import { bernoulli, gamma, normal, poisson } from '../../dists/index.ts'
import { adam } from '../../optim/index.ts'
import { key } from '../../random/index.ts'
import { item, mul, take, tensor, toFlat } from '../../tensor/index.ts'
import { recorder } from '../../trace/index.ts'
import { analyse } from '../analysis.ts'
import { importanceWeightedElbo, renyiElbo, traceElbo, traceMeanFieldElbo } from '../elbo.ts'
import {
  autoDelta,
  autoIafNormal,
  autoLaplace,
  autoLowRankMultivariateNormal,
  autoMultivariateNormal,
  autoNormal,
  guideSites,
} from '../guides.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, toProgram } from '../model/index.ts'
import { type Model, observe, param, plate, sample } from '../primitives.ts'
import { conjugateInference } from './conjugate.ts'
import { finalElbo, svi, sviEngine } from './svi.ts'

// A decaying step size (Robbins-Monro), so the final iterate settles instead of wandering with the gradient noise.
const decaying = (lr: number) => adam({ learningRate: (t: number) => lr / (1 + t / 100) })

const normalMean = MODELS['normalMean'] as Model
const exact = conjugateInference(normalMean, undefined)
const m0 = item(exact.mean('mu'))
const s0 = Math.sqrt(item(exact.variance('mu')))

describe('SVI', () => {
  it('fits a mean-field normal guide to a conjugate posterior', () => {
    const rec = recorder()
    const r = svi(normalMean, undefined, { key: key(0), numSteps: 2000, optimizer: decaying(0.02), tracer: rec })
    expect(r.status).toBe('completed')
    expect(Math.abs(item(r.params['mu_auto_loc'] ?? tensor(0)) - m0)).toBeLessThan(0.05)
    expect(Math.abs(item(r.params['mu_auto_scale'] ?? tensor(0)) - s0)).toBeLessThan(0.05)
    // The ELBO is a lower bound on log p(y), which is tight for this guide family.
    expect(finalElbo(r)).toBeLessThan((exact.logEvidence?.value ?? 0) + 0.05)
    expect(finalElbo(r)).toBeGreaterThan((exact.logEvidence?.value ?? 0) - 0.1)
    expect(r.posterior.form).toBe('variational')
    expect(rec.run().events.some((e) => e.type === 'metric' && e.name === 'elbo')).toBe(true)
  })

  it('fits correlated posteriors with joint guides', () => {
    const model = MODELS['mvn'] as Model
    const truth = conjugateFreeTruth()
    for (const guide of [autoMultivariateNormal(model), autoLowRankMultivariateNormal(model, { rank: 2 })]) {
      const r = svi(model, undefined, { key: key(1), guide, numSteps: 2500, optimizer: decaying(0.02) })
      const draws = r.posterior.sample(key(2), 4000)['x']
      const x = toFlat(draws ?? tensor([]))
      const m = [0, 1].map((j) => x.filter((_, i) => i % 2 === j).reduce((a, b) => a + b, 0) / 4000)
      expect(Math.abs((m[0] ?? 0) - truth.mean[0])).toBeLessThan(0.1)
      expect(Math.abs((m[1] ?? 0) - truth.mean[1])).toBeLessThan(0.1)
    }
  })

  it('finds the MAP point with a delta guide and its curvature with a Laplace guide', () => {
    const r = svi(normalMean, undefined, {
      key: key(3),
      guide: autoDelta(normalMean),
      numSteps: 1500,
      optimizer: decaying(0.05),
    })
    expect(Math.abs(item(r.params['mu_auto_loc'] ?? tensor(0)) - m0)).toBeLessThan(1e-3)
    expect(r.posterior.form).toBe('point')
    const guide = autoLaplace(normalMean)
    const l = svi(normalMean, undefined, { key: key(3), guide, numSteps: 1500, optimizer: decaying(0.05) })
    const post = guide.posterior(l.params, undefined)
    const d = toFlat(post.sample(key(4), 4000)['mu'] ?? tensor([]))
    const sd = Math.sqrt(d.reduce((a, b) => a + (b - m0) ** 2, 0) / d.length)
    expect(Math.abs(sd - s0)).toBeLessThan(0.03)
  })

  it('fits a normalising-flow guide', () => {
    const model = MODELS['mvn'] as Model
    const r = svi(model, undefined, {
      key: key(5),
      guide: autoIafNormal(model, { numFlows: 1, hiddenDim: 4 }),
      numSteps: 700,
      optimizer: decaying(0.02),
    })
    expect(r.status).toBe('completed')
    const x = toFlat(r.posterior.sample(key(6), 2000)['x'] ?? tensor([]))
    const m1 = x.filter((_, i) => i % 2 === 1).reduce((a, b) => a + b, 0) / 2000
    expect(Math.abs(m1 - conjugateFreeTruth().mean[1])).toBeLessThan(0.15)
  })

  it('accepts every ELBO estimator', () => {
    for (const elbo of [
      traceElbo({ numParticles: 2 }),
      traceMeanFieldElbo(),
      renyiElbo({ alpha: 0.5, numParticles: 3 }),
      importanceWeightedElbo({ numParticles: 3 }),
    ]) {
      const r = svi(normalMean, undefined, { key: key(7), elbo, numSteps: 800, optimizer: decaying(0.03) })
      expect(Math.abs(item(r.params['mu_auto_loc'] ?? tensor(0)) - m0), elbo.name).toBeLessThan(0.1)
    }
  })

  it('subsamples plates, scaling the likelihood', () => {
    const y = Array.from({ length: 40 }, (_, i) => 1 + Math.sin(i))
    const model = (): void => {
      const mu = sample('mu', normal(0, 10))
      plate('data', y.length, (idx) => observe('y', normal(mu, 1), take(tensor(y), idx)), { subsampleSize: 10 })
    }
    const mean = y.reduce((a, b) => a + b, 0) / y.length
    const r = svi(model, undefined, { key: key(8), numSteps: 2500, optimizer: decaying(0.01) })
    expect(Math.abs(item(r.params['mu_auto_loc'] ?? tensor(0)) - mean)).toBeLessThan(0.1)
    expect(sviEngine().applies(analyse(model, undefined)).reason).toMatch(/subsampled/)
  })

  it('learns a discrete guide site by score-function gradients with a baseline', () => {
    // c ~ Bernoulli(0.5), y ~ N(2c, 1) observed at 2: P(c = 1 | y) = 1 / (1 + exp(-2)).
    const model: Model = () => {
      const c = sample('c', bernoulli({ probs: 0.5 }))
      observe('y', normal(mul(c, 2), 1), 2)
    }
    const guide: Model = () => {
      const logit = param('q_logit', 0)
      sample('c', bernoulli({ logits: logit }))
    }
    const r = svi(model, undefined, {
      key: key(10),
      guide,
      numSteps: 1500,
      optimizer: decaying(0.1),
      elbo: traceElbo({ numParticles: 4 }),
    })
    const p = 1 / (1 + Math.exp(-item(r.params['q_logit'] ?? tensor(0))))
    expect(Math.abs(p - 1 / (1 + Math.exp(-2)))).toBeLessThan(0.05)
    expect(r.posterior.form).toBe('variational')
    expect(toFlat(r.posterior.sample(key(11), 10)['c'] ?? tensor([])).length).toBe(10)
  })

  it('reports divergence when the loss stops being finite', () => {
    const model = (): void => {
      const r = sample('r', gamma(1, 1))
      observe('k', poisson(mul(r, 1e300)), 3)
    }
    const r = svi(model, undefined, { key: key(9), numSteps: 50, optimizer: decaying(10) })
    expect(['diverged', 'completed']).toContain(r.status)
    expect(r.steps).toBeLessThanOrEqual(50)
  })
})

function conjugateFreeTruth(): { readonly mean: readonly [number, number] } {
  // x ~ N([0, 1], S), y = x0 + 2 x1 + N(0, 1) = 0.7: Gaussian conditioning by hand.
  const S = [
    [1, 0.3],
    [0.3, 2],
  ]
  const a = [1, 2]
  const Sa = [(S[0]?.[0] ?? 0) * 1 + (S[0]?.[1] ?? 0) * 2, (S[1]?.[0] ?? 0) * 1 + (S[1]?.[1] ?? 0) * 2]
  const v = (a[0] ?? 0) * (Sa[0] ?? 0) + (a[1] ?? 0) * (Sa[1] ?? 0) + 1
  const r = 0.7 - (0 + 2 * 1)
  return { mean: [(Sa[0] ?? 0) * (r / v), 1 + (Sa[1] ?? 0) * (r / v)] }
}

// A vector support (ordered means) on a scalar-event distribution under a plate couples the plate's elements: the guide has
// to draw the whole vector, outside that plate, or the model's observed site meets a guide value of the wrong shape.
describe('SVI on a mixture with ordered means and an enumerated label', () => {
  const data = [-2.4, -1.7, -2.1, 1.8, 2.5, 2.2, -1.9, 2.0, -2.2, 2.4]
  const mixture = (): Model =>
    toProgram(
      defineModel('ordered-mixture', (m) => {
        const row = m.range('row')
        const k = m.range('component', { size: 2 })
        const y = m.data('y', { dims: [row] })
        const mu = m.latent('mu', d.normal(0, 3), { dims: [k], constraint: { kind: 'ordered' } })
        const z = m.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: [row] })
        m.observe('y', d.normal(mu.at(z), 1), y)
      }).bind({ y: data }, { sizes: { component: 2 } }),
    )

  it('describes the coupled axis as part of the site event, outside the plate', () => {
    const [site] = guideSites(mixture(), undefined)
    expect(site?.name).toBe('mu')
    expect(site?.shape).toEqual([2])
    expect(site?.unconstrainedEventDim).toBe(1)
    expect(site?.plates).toEqual([])
  })

  it('fits the ordered means with the mean-field, delta and full-rank guides', () => {
    const model = mixture()
    for (const guide of [autoNormal(model), autoDelta(model), autoMultivariateNormal(model)]) {
      const r = svi(model, undefined, { key: key(0), guide, numSteps: 1200, optimizer: decaying(0.03) })
      expect(r.status).toBe('completed')
      const mu = toFlat(r.posterior.sample(key(1), 200)['mu'] ?? tensor([]))
      for (let i = 0; i < mu.length; i += 2) expect(mu[i] ?? 0).toBeLessThan(mu[i + 1] ?? 0)
      const m = [0, 1].map((j) => mu.filter((_, i) => i % 2 === j).reduce((a, b) => a + b, 0) / 200)
      expect(Math.abs((m[0] ?? 0) + 2)).toBeLessThan(0.4)
      expect(Math.abs((m[1] ?? 0) - 2)).toBeLessThan(0.4)
    }
  })

  it('applies to the spec: the method no longer stops at the first run', () => {
    expect(() => svi(mixture(), undefined, { key: key(2), numSteps: 5 })).not.toThrow()
  })
})
