/**
 * Fitted extended-family models on planted data: each family recovers the curves and the family parameters it was
 * simulated from, the Gaussian location-scale fit reduces to the Gaussian additive model when the scale is constant, and
 * the zero-inflated Poisson reduces to the maximum-likelihood fit of `linear` when nothing is smooth.
 */

import { describe, expect, it } from 'vitest'
import { families, gamma, poisson } from '../dists/index.ts'
import { key, normal, split, uniform } from '../random/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { extendedModel, fitExtended, linear, predictExtended, s } from './index.ts'

const logistic = (v: number): number => 1 / (1 + Math.exp(-v))
const grid = (n: number): number[][] => Array.from({ length: n }, (_, i) => [(i + 0.5) / n])
const flat = (v: Parameters<typeof toFlat>[0]): number[] => Array.from(toFlat(v))
const rmse = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let sq = 0
  for (let i = 0; i < a.length; i++) sq += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2
  return Math.sqrt(sq / a.length)
}

describe('negative binomial with the dispersion estimated jointly', () => {
  it('recovers the mean curve and theta', () => {
    const n = 300
    const theta = 3
    const x = grid(n)
    const mu = x.map((r) => Math.exp(1.2 + 0.9 * Math.sin(2 * Math.PI * (r[0] ?? 0))))
    const [k1, k2] = split(key(21), 2) as [ReturnType<typeof key>, ReturnType<typeof key>]
    const rate = flat(gamma(tensor(mu.map(() => theta)), tensor(mu.map((m) => theta / m))).sample(k1))
    const y = flat(poisson(tensor(rate)).sample(k2))
    const model = extendedModel(x, y, families.negativeBinomialTheta(), [[s(0, { k: 10 })]])
    const fit = fitExtended(model)
    expect(fit.converged).toBe(true)
    const pred = predictExtended(model, fit, x)
    expect(rmse(pred.eta[0] as Float64Array, mu.map(Math.log))).toBeLessThan(0.2)
    const est = Math.exp(fit.state.aux[0] ?? 0)
    expect(est / theta).toBeGreaterThan(0.6)
    expect(est / theta).toBeLessThan(1.7)
  }, 120000)
})

describe('Gaussian location-scale', () => {
  it('recovers a mean and a standard deviation that both vary', () => {
    const n = 300
    const x = grid(n)
    const z = flat(normal(key(31), [n]))
    const mean = (t: number): number => Math.sin(2 * Math.PI * t)
    const logsd = (t: number): number => -1 + 1.2 * t
    const y = x.map((r, i) => mean(r[0] ?? 0) + Math.exp(logsd(r[0] ?? 0)) * (z[i] ?? 0))
    const model = extendedModel(x, y, families.gaussianLocationScale(), [[s(0, { k: 8 })], [s(0, { k: 5 })]])
    const fit = fitExtended(model)
    expect(fit.converged).toBe(true)
    const pred = predictExtended(model, fit, x)
    expect(
      rmse(
        pred.eta[0] as Float64Array,
        x.map((r) => mean(r[0] ?? 0)),
      ),
    ).toBeLessThan(0.12)
    expect(
      rmse(
        pred.eta[1] as Float64Array,
        x.map((r) => logsd(r[0] ?? 0)),
      ),
    ).toBeLessThan(0.2)
  }, 120000)
})

describe('ordered categorical', () => {
  it('recovers the latent curve and the cut points', () => {
    const n = 400
    const cuts = [-1, 0.4, 1.9]
    const x = grid(n)
    const u = flat(uniform(key(41), [n]))
    const eta = (t: number): number => 1.4 * Math.sin(2 * Math.PI * t)
    const y = x.map((r, i) => {
      const noise = Math.log((u[i] ?? 0.5) / (1 - (u[i] ?? 0.5)))
      const latent = eta(r[0] ?? 0) + noise
      return cuts.filter((c) => c < latent).length
    })
    const model = extendedModel(x, y, families.orderedCategorical(4), [[s(0, { k: 8 })]])
    const fit = fitExtended(model)
    expect(fit.converged).toBe(true)
    const pred = predictExtended(model, fit, x)
    expect(
      rmse(
        pred.eta[0] as Float64Array,
        x.map((r) => eta(r[0] ?? 0)),
      ),
    ).toBeLessThan(0.35)
    const estCuts = [-1]
    for (const a of fit.state.aux) estCuts.push((estCuts[estCuts.length - 1] ?? 0) + Math.exp(a))
    expect(Math.abs((estCuts[1] ?? 0) - 0.4)).toBeLessThan(0.35)
    expect(Math.abs((estCuts[2] ?? 0) - 1.9)).toBeLessThan(0.5)
  }, 180000)
})

describe('zero-inflated Poisson', () => {
  it('recovers the count curve and the inflation curve from two predictors', () => {
    const n = 400
    const x = grid(n)
    const lam = (t: number): number => Math.exp(0.6 + 0.8 * Math.sin(2 * Math.PI * t))
    const pi = (t: number): number => logistic(-0.8 + 2 * t)
    const [k1, k2] = split(key(51), 2) as [ReturnType<typeof key>, ReturnType<typeof key>]
    const counts = flat(poisson(tensor(x.map((r) => lam(r[0] ?? 0)))).sample(k1))
    const u = flat(uniform(k2, [n]))
    const y = x.map((r, i) => ((u[i] ?? 1) < pi(r[0] ?? 0) ? 0 : (counts[i] ?? 0)))
    const model = extendedModel(x, y, families.zeroInflatedPoissonLogit(), [[s(0, { k: 8 })], [s(0, { k: 4 })]])
    const fit = fitExtended(model)
    expect(fit.converged).toBe(true)
    const pred = predictExtended(model, fit, x)
    expect(
      rmse(
        pred.eta[0] as Float64Array,
        x.map((r) => Math.log(lam(r[0] ?? 0))),
      ),
    ).toBeLessThan(0.35)
    expect(
      rmse(
        pred.parameters[1] as Float64Array,
        x.map((r) => pi(r[0] ?? 0)),
      ),
    ).toBeLessThan(0.12)
  }, 180000)
})

describe('multinomial additive model', () => {
  it('recovers the class probabilities of three classes', () => {
    const n = 400
    const x = grid(n)
    const eta1 = (t: number): number => 1.5 * Math.sin(2 * Math.PI * t)
    const eta2 = (t: number): number => 1 - 2 * t
    const prob = (t: number): number[] => {
      const e = [0, eta1(t), eta2(t)].map(Math.exp)
      const tot = e.reduce((a, b) => a + b, 0)
      return e.map((v) => v / tot)
    }
    const u = flat(uniform(key(61), [n]))
    const y = x.map((r, i) => {
      const p = prob(r[0] ?? 0)
      const v = u[i] ?? 0.5
      return v < (p[0] ?? 0) ? 0 : v < (p[0] ?? 0) + (p[1] ?? 0) ? 1 : 2
    })
    const model = extendedModel(x, y, families.multinomialLogit(3), [[s(0, { k: 8 })], [s(0, { k: 6 })]])
    const fit = fitExtended(model)
    expect(fit.converged).toBe(true)
    const pred = predictExtended(model, fit, x)
    const e1 = pred.eta[0] as Float64Array
    const e2 = pred.eta[1] as Float64Array
    let sq = 0
    x.forEach((r, i) => {
      const tot = 1 + Math.exp(e1[i] ?? 0) + Math.exp(e2[i] ?? 0)
      const p = prob(r[0] ?? 0)
      sq += (1 / tot - (p[0] ?? 0)) ** 2 + (Math.exp(e1[i] ?? 0) / tot - (p[1] ?? 0)) ** 2
    })
    expect(Math.sqrt(sq / (2 * n))).toBeLessThan(0.07)
  }, 180000)
})

describe('Tweedie with the power estimated', () => {
  it('recovers the mean curve and a power between 1 and 2', () => {
    const n = 350
    const power = 1.5
    const phi = 1.2
    const x = grid(n)
    const mu = x.map((r) => Math.exp(0.4 + 0.6 * Math.sin(2 * Math.PI * (r[0] ?? 0))))
    const [k1, k2] = split(key(71), 2) as [ReturnType<typeof key>, ReturnType<typeof key>]
    const lambda = mu.map((m) => m ** (2 - power) / (phi * (2 - power)))
    const counts = flat(poisson(tensor(lambda)).sample(k1))
    const alpha = (2 - power) / (power - 1)
    const g = flat(
      gamma(
        tensor(counts.map((c) => Math.max(c, 1e-9) * alpha)),
        tensor(mu.map((m) => 1 / (phi * (power - 1) * m ** (power - 1)))),
      ).sample(k2),
    )
    const y = counts.map((c, i) => (c === 0 ? 0 : (g[i] ?? 0)))
    const model = extendedModel(x, y, families.tweedieEstimated(), [[s(0, { k: 8 })]])
    const fit = fitExtended(model)
    expect(fit.converged).toBe(true)
    const pred = predictExtended(model, fit, x)
    expect(rmse(pred.eta[0] as Float64Array, mu.map(Math.log))).toBeLessThan(0.3)
    const p = 1 + logistic(fit.state.aux[1] ?? 0)
    expect(Math.abs(p - power)).toBeLessThan(0.2)
  }, 300000)
})

describe('reductions to models the library already fits', () => {
  it('Gaussian location-scale with a constant scale is the Gaussian additive model at the matching penalty', async () => {
    const { fitGam, GAM_DEFAULTS, gamProblem } = await import('./index.ts')
    const n = 80
    const x = grid(n)
    const z = flat(normal(key(81), [n]))
    const y = x.map((r, i) => Math.sin(5 * (r[0] ?? 0)) + 0.3 * (z[i] ?? 0))
    const lambda = 0.7
    const model = extendedModel(x, y, families.gaussianLocationScale(), [[s(0, { k: 7 })], []])
    const fit = fitExtended(model, { method: 'fixed', sp: [lambda], aux: undefined, maxIter: 5, solver: undefined })
    const sd2 = Math.exp(2 * (fit.eta[1]?.[0] ?? 0))
    // penalised log-likelihood: -rss / (2 sd^2) - lambda beta'S beta / 2  <=>  rss + sd^2 lambda beta'S beta
    const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 7 })], method: 'fixed' as const, sp: [sd2 * lambda] }
    const reference = fitGam(gamProblem(x, y, hyper), hyper).state
    const mean = fit.eta[0] as Float64Array
    expect(rmse(mean, reference.eta)).toBeLessThan(1e-6)
  }, 60000)

  it('zero-inflated Poisson with nothing smooth is the maximum-likelihood fit of linear.zeroInflated', async () => {
    const { zeroInflated } = await import('../linear/index.ts')
    const n = 300
    const x = grid(n)
    const [k1, k2] = split(key(91), 2) as [ReturnType<typeof key>, ReturnType<typeof key>]
    const counts = flat(poisson(tensor(x.map((r) => Math.exp(0.5 + (r[0] ?? 0))))).sample(k1))
    const u = flat(uniform(k2, [n]))
    const y = counts.map((c, i) => ((u[i] ?? 1) < 0.25 ? 0 : c))
    const model = extendedModel(x, y, families.zeroInflatedPoissonLogit(), [[linear(0)], []])
    const fit = fitExtended(model, { method: 'fixed', sp: undefined, aux: undefined, maxIter: 5, solver: undefined })
    const ref = zeroInflated({ family: 'poisson' }).fit({ x, y })
    const lam = flat(ref.predictCount(x))
    const mean0 = fit.eta[0] as Float64Array
    expect(
      rmse(
        mean0.map((e) => Math.exp(e)),
        lam,
      ),
    ).toBeLessThan(5e-3)
    const pi = flat(ref.predictInflation(x))
    expect(Math.abs(logistic(fit.eta[1]?.[0] ?? 0) - (pi[0] ?? 0))).toBeLessThan(5e-3)
  }, 60000)
})
