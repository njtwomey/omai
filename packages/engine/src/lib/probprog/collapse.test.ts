import { describe, expect, it } from 'vitest'
import { bernoulli, beta, gamma, halfNormal, multivariateNormal, normal, poisson } from '../dists/index.ts'
import { adaptiveSimpson } from '../numerics/quadrature/index.ts'
import { lbeta, lgamma } from '../numerics/special/index.ts'
import { foldIn, key as makeKey } from '../random/index.ts'
import { exp, item, mul, ones, tensor, toFlat, type Value } from '../tensor/index.ts'
import { collapse } from './collapse.ts'
import { logJoint, potentialEnergy } from './density.ts'
import { mcmc } from './engines/index.ts'
import { seed, traceOf } from './handlers.ts'
import { type Model, observe, plate, sample } from './primitives.ts'

const lj = (m: Model<undefined>, values: Record<string, number> = {}): number =>
  item(logJoint(m, undefined, values).logJoint)

describe('collapse equals the closed-form marginal likelihood', () => {
  it('normal mean, normal observations: y ~ N(m0 1, sigma^2 I + s0^2 11^T)', () => {
    const y = [1.2, 0.4, 2.1, -0.3]
    const m0 = 0.5
    const s0 = 2
    const sigma = 0.7
    const model: Model = () => {
      const mu = sample('mu', normal(m0, s0))
      plate('n', y.length, () => observe('y', normal(mu, sigma), y))
    }
    const n = y.length
    const cov = tensor(
      Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => s0 ** 2 + (i === j ? sigma ** 2 : 0))),
    )
    const exact = item(multivariateNormal(mul(ones([n]), m0), { covariance: cov }).logProb(tensor(y)))
    const c = collapse(model, undefined)
    expect(c.sites).toEqual(['mu'])
    expect(lj(c.model)).toBeCloseTo(exact, 11)
  })

  it('beta probability, Bernoulli trials: log B(a + k, b + n - k) - log B(a, b)', () => {
    const flips = [1, 0, 1, 1, 0, 1, 1]
    const [a, b] = [2, 3]
    const model: Model = () => {
      const p = sample('p', beta(a, b))
      plate('n', flips.length, () => observe('y', bernoulli({ probs: p }), flips))
    }
    const k = flips.reduce((s, v) => s + v, 0)
    const exact = item(lbeta(a + k, b + flips.length - k)) - item(lbeta(a, b))
    expect(lj(collapse(model, undefined).model)).toBeCloseTo(exact, 12)
  })

  it('gamma rate, Poisson counts: closed form through log-gamma', () => {
    const counts = [3, 1, 4, 1, 5]
    const [a, b] = [2, 1.5]
    const model: Model = () => {
      const r = sample('r', gamma(a, b))
      plate('n', counts.length, () => observe('k', poisson(r), counts))
    }
    const s = counts.reduce((x, v) => x + v, 0)
    const logFactorials = counts.reduce((x, v) => x + item(lgamma(v + 1)), 0)
    const exact =
      a * Math.log(b) - item(lgamma(a)) + item(lgamma(a + s)) - (a + s) * Math.log(b + counts.length) - logFactorials
    expect(lj(collapse(model, undefined).model)).toBeCloseTo(exact, 11)
  })
})

describe('collapse inside a larger model', () => {
  const y = [1.2, 0.4, 2.1, -0.3, 0.9]
  const model: Model = () => {
    const mu = sample('mu', normal(0, 3))
    const sigma = sample('sigma', halfNormal(2))
    plate('n', y.length, () => observe('y', normal(mu, sigma), y))
  }

  it('gives the marginal density of the sites that remain (quadrature over the removed one)', () => {
    const c = collapse(model, undefined)
    expect(c.sites).toEqual(['mu'])
    for (const sigma of [0.5, 1, 2.2]) {
      const integral = adaptiveSimpson((x) => tensor([Math.exp(lj(model, { mu: item(x), sigma }))]), -25, 25, {
        tol: 1e-13,
      }).value
      expect(lj(c.model, { sigma })).toBeCloseTo(Math.log(integral), 7)
    }
  })

  it('has the same gradient as the finite difference of its own density, on the unconstrained scale', () => {
    const um = potentialEnergy(collapse(model, undefined).model, undefined)
    expect(um.dim).toBe(1)
    for (const u of [-0.6, 0.2, 0.9]) {
      const h = 1e-5
      const fd = (um.logDensity.logDensity([u + h]) - um.logDensity.logDensity([u - h])) / (2 * h)
      expect(um.logDensity.valueAndGrad([u]).grad[0]).toBeCloseTo(fd, 6)
    }
  })

  it('recovers mu from its closed-form conditional given sigma', () => {
    const c = collapse(model, undefined)
    const sigma = 1.3
    const post = c.posteriors({ sigma }, undefined)['mu']
    // N(0, 3) prior and n observations of sd sigma: precision 1/9 + n / sigma^2.
    const prec = 1 / 9 + y.length / sigma ** 2
    expect(item(post?.mean() as Value)).toBeCloseTo(y.reduce((s, v) => s + v, 0) / sigma ** 2 / prec, 12)
    expect(item(post?.stddev() as Value)).toBeCloseTo(Math.sqrt(1 / prec), 12)
    const draws = c.recover({ sigma }, undefined, makeKey(4), 4000)['mu']
    const mean = toFlat(draws as Value).reduce((s, v) => s + v, 0) / 4000
    expect(Math.abs(mean - item(post?.mean() as Value))).toBeLessThan(4 * Math.sqrt(1 / prec / 4000))
  })

  it('samples what remains to the same posterior as the full model (NUTS, fixed seeds)', () => {
    const options = { key: makeKey(7), numChains: 2, numWarmup: 300, numDraws: 500 }
    const full = mcmc(model, undefined, options)
    const small = mcmc(collapse(model, undefined).model, undefined, options)
    const a = toFlat(full.draws?.['sigma'] as Value)
    const b = toFlat(small.draws?.['sigma'] as Value)
    const mean = (x: number[]): number => x.reduce((s, v) => s + v, 0) / x.length
    const sd = (x: number[]): number => Math.sqrt(x.reduce((s, v) => s + (v - mean(x)) ** 2, 0) / x.length)
    // Draws of a chain are correlated: an effective size of a fifth of the draws is generous.
    const se = Math.sqrt((sd(a) ** 2 + sd(b) ** 2) / (0.2 * a.length))
    expect(Math.abs(mean(a) - mean(b))).toBeLessThan(4 * se)
    expect(small.diagnostics['divergences']).toBe(0)
  })

  it('adds the marginal as a factor and keeps the removed site observed at its placeholder', () => {
    const t = traceOf(seed(collapse(model, undefined).model, foldIn(makeKey(0), 1)), undefined)
    expect(t.sites.has('mu__marginal')).toBe(true)
    const mu = t.sites.get('mu')
    expect(mu?.type === 'sample' && mu.isObserved).toBe(true)
  })
})

describe('collapse refuses what it cannot do, and says why', () => {
  it('a site used outside the conjugate parameter', () => {
    const model: Model = () => {
      const s = sample('s', halfNormal(1))
      plate('n', 2, () => observe('y', normal(0, s), [0.1, 0.4]))
    }
    expect(() => collapse(model, undefined)).toThrow(/no latent site can be collapsed/)
    expect(() => collapse(model, undefined, { sites: ['s'] })).toThrow(/not a conjugate likelihood/)
  })

  it('a site with a latent child and a site that is not latent', () => {
    const model: Model = () => {
      const mu = sample('mu', normal(0, 1))
      const x = sample('x', normal(mu, 1))
      observe('y', normal(x, 1), 0.3)
    }
    expect(() => collapse(model, undefined, { sites: ['mu'] })).toThrow(/not an observed sample site/)
    expect(() => collapse(model, undefined, { sites: ['y'] })).toThrow(/not a latent sample site/)
  })

  it('a child whose known parameter depends on the site', () => {
    const model: Model = () => {
      const mu = sample('mu', normal(0, 1))
      observe('y', normal(mu, exp(mu)), 0.3)
    }
    expect(() => collapse(model, undefined, { sites: ['mu'] })).toThrow(/depends on mu/)
  })
})
