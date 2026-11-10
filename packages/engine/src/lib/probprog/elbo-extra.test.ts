// Sticking the landing and the doubly reparameterised gradient against the exact ELBO of a conjugate normal model: both are
// unbiased, both have a gradient of zero variance at the exact posterior (the plain estimator does not), and the importance
// weighted bound rises with the number of particles towards the log evidence.
import { describe, expect, it } from 'vitest'
import { MODELS, Y4 } from '../../testing/probprog.ts'
import { valueAndGrad } from '../autodiff/index.ts'
import { bernoulli } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { add, log, mul, primalOf, square, sub, tensor, toFlat, type Value } from '../tensor/index.ts'
import { importanceWeightedElbo, traceElbo } from './elbo.ts'
import { gradientMoments } from './elbo-diagnostics.ts'
import { dregElbo, stlElbo } from './elbo-extra.ts'
import { conjugateInference } from './engines/conjugate.ts'
import { autoNormal } from './guides.ts'
import { type Model, param, sample } from './primitives.ts'

const model = MODELS['normalMean'] as Model
const guide = autoNormal(model)
const exact = conjugateInference(model, undefined)
const m0 = toFlat(primalOf(exact.mean('mu')))[0] ?? 0
const s0 = Math.sqrt(toFlat(primalOf(exact.variance('mu')))[0] ?? 1)
const logZ = exact.logEvidence?.value ?? 0
const at = (m: number, s: number) => ({ mu_auto_loc: tensor(m), mu_auto_scale: tensor(s) })

/** The exact ELBO of the normal-mean model with a normal guide, as a function of the guide's location and scale. */
function exactElbo(p: Record<string, Value>): Value {
  const m = p['mu_auto_loc'] as Value
  const s = p['mu_auto_scale'] as Value
  const prior = sub(-0.5 * Math.log(2 * Math.PI * 4), mul(add(square(m), square(s)), 1 / 8))
  let like: Value = tensor(0)
  for (const y of Y4)
    like = add(like, sub(-0.5 * Math.log(2 * Math.PI * 2.25), mul(add(square(sub(y, m)), square(s)), 1 / 4.5)))
  const entropy = add(0.5 * Math.log(2 * Math.PI * Math.E), log(s))
  return add(add(prior, like), entropy)
}

describe('sticking the landing', () => {
  it('has the same expected gradient as the plain estimator, which is the exact one', () => {
    const p = at(m0 + 0.4, s0 * 1.5)
    const { grad } = valueAndGrad(exactElbo)(p as Record<string, Value>)
    const g = grad as Record<string, Value>
    const stl = gradientMoments({
      estimator: stlElbo(),
      model,
      guide,
      data: undefined,
      params: p,
      key: key(1),
      repeats: 1500,
    })
    for (const k of ['mu_auto_loc', 'mu_auto_scale'] as const) {
      const exactGrad = -(toFlat(primalOf(g[k] as Value))[0] ?? 0)
      const se = Math.sqrt((stl.variance[k]?.[0] ?? 0) / 1500)
      expect(Math.abs((stl.mean[k]?.[0] ?? 0) - exactGrad), k).toBeLessThan(4.5 * se)
    }
  })

  it('has a gradient that vanishes at the exact posterior, where the plain one is noisy', () => {
    const p = at(m0, s0)
    const run = (estimator: Parameters<typeof gradientMoments>[0]['estimator']) =>
      gradientMoments({ estimator, model, guide, data: undefined, params: p, key: key(2), repeats: 200 })
    expect(run(stlElbo()).totalVariance).toBeLessThan(1e-18)
    expect(run(traceElbo()).totalVariance).toBeGreaterThan(1e-3)
  })

  it('refuses a guide with a site that has no reparameterised sampler', () => {
    const m: Model = () => sample('c', bernoulli({ probs: 0.5 }))
    const g: Model = () => sample('c', bernoulli({ logits: param('l', tensor(0)) }))
    expect(() => stlElbo().estimate(key(0), m, g, undefined, { l: tensor(0) }, null)).toThrow(/reparameterised/)
  })
})

describe('the importance-weighted bound', () => {
  const p = at(m0 + 0.3, s0 * 1.4)
  const mean = (estimator: ReturnType<typeof importanceWeightedElbo>, n: number): number => {
    let total = 0
    for (let i = 0; i < n; i++) total += estimator.estimate(key(100 + i), model, guide, undefined, p, null).elbo
    return total / n
  }

  it('rises with the number of particles and stays below the log evidence', () => {
    const bounds = [1, 2, 4, 8, 16].map((k) => mean(importanceWeightedElbo({ numParticles: k }), 300))
    for (let i = 1; i < bounds.length; i++) expect(bounds[i] as number).toBeGreaterThan(bounds[i - 1] as number)
    for (const b of bounds) expect(b).toBeLessThan(logZ + 1e-9)
  })

  it('the DReG estimator reports the same bound as the plain one on the same particles', () => {
    const a = dregElbo({ numParticles: 6 }).estimate(key(7), model, guide, undefined, p, null)
    const b = importanceWeightedElbo({ numParticles: 6 }).estimate(key(7), model, guide, undefined, p, null)
    expect(Math.abs(a.elbo - b.elbo)).toBeLessThan(1e-12)
    expect(a.diagnostics?.['ess']).toBeGreaterThan(1)
  })

  it('DReG has the plain IWAE gradient in expectation and less variance of it', () => {
    const repeats = 1200
    const plain = gradientMoments({
      estimator: importanceWeightedElbo({ numParticles: 8 }),
      model,
      guide,
      data: undefined,
      params: p,
      key: key(3),
      repeats,
    })
    const dreg = gradientMoments({
      estimator: dregElbo({ numParticles: 8 }),
      model,
      guide,
      data: undefined,
      params: p,
      key: key(3),
      repeats,
    })
    for (const k of ['mu_auto_loc', 'mu_auto_scale'] as const) {
      const se = Math.sqrt(((plain.variance[k]?.[0] ?? 0) + (dreg.variance[k]?.[0] ?? 0)) / repeats)
      expect(Math.abs((plain.mean[k]?.[0] ?? 0) - (dreg.mean[k]?.[0] ?? 0)), k).toBeLessThan(4.5 * se)
    }
    expect(dreg.totalVariance).toBeLessThan(plain.totalVariance)
    // At the exact posterior the doubly reparameterised gradient has no noise at all.
    const star = gradientMoments({
      estimator: dregElbo({ numParticles: 8 }),
      model,
      guide,
      data: undefined,
      params: at(m0, s0),
      key: key(4),
      repeats: 100,
    })
    expect(star.totalVariance).toBeLessThan(1e-18)
  })
})
