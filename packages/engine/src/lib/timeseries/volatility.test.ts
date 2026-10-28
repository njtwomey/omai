import { describe, expect, it } from 'vitest'
import { expectResumable } from '../../testing/series.ts'
import { key } from '../random/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../trace/index.ts'
import * as T from './index.ts'

const truth: T.GarchParameters = { mu: 0, omega: 0.1, alpha: [0.1], gamma: [], beta: [0.8] }
const zero: T.GarchSpec = { mean: 'zero' }
const y = T.garchSimulate(key(7), 3000, zero, truth)

describe('GARCH recursion and likelihood', () => {
  it('is the i.i.d. density when the coefficients vanish', () => {
    const p = { mu: 0, omega: 2, alpha: [0], gamma: [], beta: [] }
    const ll = T.garchLogLikelihood([1, -2], { q: 0, mean: 'zero' }, p)
    expect(ll).toBeCloseTo(-Math.log(2 * Math.PI) - Math.log(2) - (1 + 4) / 4, 12)
  })

  it('follows the defining recursion with the arch backcast', () => {
    const e = [0.5, -1.2, 0.3, 2.1, -0.4]
    const v = T.garchVariance(e, zero, truth)
    // The backcast: the weighted mean of the squared residuals with weights 0.94^i.
    let w = 0
    let s = 0
    e.forEach((x, i) => {
      w += 0.94 ** i
      s += 0.94 ** i * x * x
    })
    const back = s / w
    expect(v[0]).toBeCloseTo(0.1 + 0.1 * back + 0.8 * back, 12)
    expect(v[1]).toBeCloseTo(0.1 + 0.1 * 0.25 + 0.8 * (v[0] ?? 0), 12)
    expect(v[3]).toBeCloseTo(0.1 + 0.1 * 0.09 + 0.8 * (v[2] ?? 0), 12)
  })

  it('adds the asymmetric term for negative residuals only', () => {
    const e = [-1, 1, -1, 1]
    const sym = T.garchVariance(e, { ...zero, o: 0 }, { ...truth, alpha: [0.1] })
    const gjr = T.garchVariance(e, { ...zero, o: 1 }, { ...truth, gamma: [0.2] })
    expect((gjr[1] ?? 0) - (sym[1] ?? 0)).toBeGreaterThan(0.1)
    expect((gjr[2] ?? 0) - (sym[2] ?? 0)).toBeLessThan((gjr[1] ?? 0) - (sym[1] ?? 0) + 1)
  })

  it('has a Student-t density that integrates to one with unit variance', () => {
    const nu = 7
    let mass = 0
    let second = 0
    const h = 0.005
    for (let x = -60; x < 60; x += h) {
      const d = Math.exp(
        T.garchLogLikelihood(
          [x],
          { q: 0, mean: 'zero', distribution: 't' },
          { mu: 0, omega: 1, alpha: [0], gamma: [], beta: [], nu },
        ),
      )
      mass += d * h
      second += x * x * d * h
    }
    expect(mass).toBeCloseTo(1, 3)
    expect(second).toBeCloseTo(1, 2)
  })

  it('runs as an algorithm that accumulates the same likelihood and serialises', () => {
    const params = { ...truth, alpha: [0.1], beta: [0.8] }
    const alg = T.garchFilterAlgorithm(y.slice(0, 200), zero, params)
    const rec = recorder()
    const run = runAlgorithm(alg, { maxSteps: 1000, tracer: rec })
    expect(run.steps).toBe(200)
    expect(run.state.logLikelihood).toBeCloseTo(T.garchLogLikelihood(y.slice(0, 200), zero, params), 9)
    expect(metricSeries(rec.run(), 'logLikelihood').values.at(-1)).toBeCloseTo(run.state.logLikelihood, 12)
    expectResumable(T.garchFilterAlgorithm(y.slice(0, 50), { ...zero, o: 1 }, { ...truth, gamma: [0.05] }), 20, 20)
  })
})

describe('GARCH fits and forecasts', () => {
  it('recovers a GARCH(1,1) from simulated data', () => {
    const fit = T.garchFit(y, zero)
    expect(fit.converged).toBe(true)
    expect(Math.abs((fit.params.alpha[0] ?? 0) - 0.1)).toBeLessThan(0.04)
    expect(Math.abs((fit.params.beta[0] ?? 0) - 0.8)).toBeLessThan(0.08)
    expect(Math.abs(fit.persistence - 0.9)).toBeLessThan(0.06)
    expect(fit.stdErrors.every((v) => v > 0)).toBe(true)
    expect(fit.logLikelihood).toBeGreaterThan(T.garchLogLikelihood(y, zero, { ...truth, omega: 0.5 }))
    expect(fit.logLikelihood).toBeGreaterThan(T.garchLogLikelihood(y, zero, truth) - 5)
  })

  it('fits a mean and Student-t innovations', () => {
    const ty = T.garchSimulate(key(8), 4000, { distribution: 't' }, { ...truth, mu: 0.3, nu: 6 })
    const fit = T.garchFit(ty, { distribution: 't' })
    expect(Math.abs(fit.params.mu - 0.3)).toBeLessThan(0.08)
    expect(fit.params.nu).toBeGreaterThan(3.5)
    expect(fit.params.nu).toBeLessThan(11)
  })

  it('detects leverage with a GJR term', () => {
    const gy = T.garchSimulate(
      key(9),
      4000,
      { ...zero, o: 1 },
      { mu: 0, omega: 0.1, alpha: [0.03], gamma: [0.2], beta: [0.8] },
    )
    const fit = T.garchFit(gy, { ...zero, o: 1 })
    expect(fit.params.gamma[0]).toBeGreaterThan(0.08)
  })

  it('records the log-likelihood per iteration', () => {
    const { algorithm } = T.garchFitAlgorithm(y.slice(0, 600), zero)
    const rec = recorder()
    runAlgorithm(algorithm, { maxSteps: 100, tracer: rec })
    const ll = metricSeries(rec.run(), 'logLikelihood').values
    expect(ll.length).toBeGreaterThan(3)
    expect(ll.at(-1)).toBeGreaterThan(ll[0] ?? 0)
  })

  it('forecasts variances that relax geometrically to the unconditional variance', () => {
    const fit = { spec: zero, params: truth }
    const f = T.garchForecast(fit, y.slice(0, 500), 30)
    const v0 = f.variance[0] ?? 0
    const bar = 0.1 / (1 - 0.9)
    for (let h = 1; h < 10; h++) expect((f.variance[h] ?? 0) - bar).toBeCloseTo(0.9 ** h * (v0 - bar), 9)
    expect(f.volatility[0]).toBeCloseTo(Math.sqrt(v0), 12)
  })

  it('refuses short series and models without a residual term', () => {
    expect(() => T.garchFit(y.slice(0, 20), zero)).toThrow()
    expect(() => T.garchFit(y, { p: 0, o: 0 })).toThrow()
  })
})
