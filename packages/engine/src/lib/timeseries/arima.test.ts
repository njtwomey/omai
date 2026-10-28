import { describe, expect, it } from 'vitest'
import { expectResumable, simulateArma } from '../../testing/series.ts'
import { ljungBox } from '../stats/tests/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../trace/index.ts'
import { expandPolynomials } from './_arima-model.ts'
import * as T from './index.ts'

const near = (a: number, b: number, tol: number) => expect(Math.abs(a - b)).toBeLessThan(tol)

describe('ARIMA fits', () => {
  const y = simulateArma(11, 400, [0.6], [0.3], 2)

  it('recovers the parameters of a simulated ARMA(1, 1) with a mean', () => {
    const fit = T.arimaFit(y, { order: [1, 0, 1] })
    near(fit.ar[0] ?? 0, 0.6, 0.12)
    near(fit.ma[0] ?? 0, 0.3, 0.15)
    near(fit.regression[0] ?? 0, 2, 0.3)
    near(fit.sigma2, 1, 0.2)
    expect(fit.nobs).toBe(400)
    expect(fit.stdErrors).toHaveLength(4)
    expect(fit.aic).toBeCloseTo(-2 * fit.logLikelihood + 8, 9)
    expect(fit.converged).toBe(true)
  })

  it('leaves white residuals when the model is right', () => {
    const fit = T.arimaFit(y, { order: [1, 0, 1] })
    const { innovations, relativeVariances } = T.arimaResiduals(fit, y)
    const std = Array.from(innovations, (v, t) => v / Math.sqrt(relativeVariances[t] ?? 1))
    expect(ljungBox(std, { lags: 10, modelDf: 2 }).pValue).toBeGreaterThan(0.01)
  })

  it('is plain data that survives JSON', () => {
    const fit = T.arimaFit(y, { order: [1, 0, 0] })
    expect(JSON.parse(JSON.stringify(fit))).toEqual(fit)
  })

  it('agrees across the likelihood methods', () => {
    const mle = T.arimaFit(y, { order: [2, 0, 0] }, { method: 'mle' })
    const css = T.arimaFit(y, { order: [2, 0, 0] }, { method: 'css' })
    const both = T.arimaFit(y, { order: [2, 0, 0] }, { method: 'css-mle' })
    near(css.ar[0] ?? 0, mle.ar[0] ?? 0, 0.05)
    near(both.logLikelihood, mle.logLikelihood, 1e-6)
  })

  it('differences once for an integrated series and takes the drift from a linear trend', () => {
    const walk = simulateArma(12, 300, [0.5], []).reduce<number[]>((acc, v, t) => {
      acc.push((acc[t - 1] ?? 0) + v + 0.2)
      return acc
    }, [])
    const fit = T.arimaFit(walk, { order: [1, 1, 0], trend: 't' })
    expect(fit.nobs).toBe(299)
    const drift = T.difference(walk).reduce((a, b) => a + b, 0) / 299
    near(fit.regression[0] ?? 0, drift, 0.05)
    expect(() => T.arimaFit(walk, { order: [1, 1, 0], trend: 'c' })).toThrow()
  })

  it('fits a regression with ARMA errors', () => {
    const x = Array.from({ length: 300 }, (_, t) => [Math.sin(t / 5), (t % 7) - 3])
    const e = simulateArma(13, 300, [0.5], [])
    const z = x.map((row, t) => 1.5 * (row[0] ?? 0) - 0.5 * (row[1] ?? 0) + (e[t] ?? 0))
    const fit = T.arimaFit(z, { order: [1, 0, 0], trend: 'n', exog: x })
    near(fit.regression[0] ?? 0, 1.5, 0.2)
    near(fit.regression[1] ?? 0, -0.5, 0.1)
    near(fit.ar[0] ?? 0, 0.5, 0.12)
  })

  it('runs as an algorithm with the log-likelihood at each iteration, and resumes from a saved state', () => {
    const { algorithm, fit } = T.arimaFitAlgorithm(y, { order: [2, 0, 1] })
    const rec = recorder()
    const run = runAlgorithm(algorithm, { maxSteps: 100, tracer: rec })
    const ll = metricSeries(rec.run(), 'logLikelihood').values
    expect(ll.length).toBeGreaterThan(2)
    expect(ll[ll.length - 1] ?? 0).toBeGreaterThan(ll[0] ?? 0)
    expect(fit(run.state).logLikelihood).toBeCloseTo(ll[ll.length - 1] ?? 0, 9)
    expectResumable(algorithm, 2, 3)
  })

  it('refuses specifications it cannot fit', () => {
    expect(() => T.arimaFit([1, 2, 3], { order: [3, 0, 2] })).toThrow()
    expect(() => T.arimaFit(y, { order: [1, 0, 0], seasonal: { order: [1, 0, 0], period: 1 } })).toThrow()
  })
})

describe('seasonal polynomials', () => {
  it('multiplies the seasonal factors into the polynomials', () => {
    const { ar, ma } = expandPolynomials([0.5], [0.4], [0.3], [0.2], 4)
    // (1 - 0.5 B)(1 - 0.3 B^4) = 1 - 0.5 B - 0.3 B^4 + 0.15 B^5.
    ;[0.5, 0, 0, 0.3, -0.15].forEach((v, i) => {
      expect(ar[i]).toBeCloseTo(v, 12)
    })
    // (1 + 0.4 B)(1 + 0.2 B^4) = 1 + 0.4 B + 0.2 B^4 + 0.08 B^5.
    ma.forEach((v, i) => {
      expect(v).toBeCloseTo([0.4, 0, 0, 0.2, 0.08][i] ?? 0, 12)
    })
  })

  it('fits a seasonal AR on a seasonal series', () => {
    const e = simulateArma(14, 240, [], [])
    const s = e.reduce<number[]>((acc, v, t) => {
      acc.push(v + 0.7 * (acc[t - 4] ?? 0))
      return acc
    }, [])
    const fit = T.arimaFit(s, { order: [0, 0, 0], seasonal: { order: [1, 0, 0], period: 4 }, trend: 'n' })
    near(fit.seasonalAr[0] ?? 0, 0.7, 0.1)
  })
})

describe('ARIMA forecasts', () => {
  it('forecasts an AR(1) with the known mean reversion and variance', () => {
    const fit = T.arimaFit(simulateArma(15, 300, [0.7], [], 1), { order: [1, 0, 0] })
    const history = simulateArma(15, 300, [0.7], [], 1)
    const f = T.arimaForecast(fit, history, 6)
    const phi = fit.ar[0] ?? 0
    const mu = fit.regression[0] ?? 0
    const last = history[299] ?? 0
    f.mean.forEach((m, h) => {
      expect(m).toBeCloseTo(mu + phi ** (h + 1) * (last - mu), 6)
    })
    f.se.forEach((s, h) => {
      let v = 0
      for (let j = 0; j <= h; j++) v += phi ** (2 * j)
      expect(s).toBeCloseTo(Math.sqrt(fit.sigma2 * v), 6)
    })
    expect(f.intervals[0]?.lower[0]).toBeLessThan(f.mean[0] ?? 0)
  })

  it('widens the intervals with the order of integration', () => {
    const base = {
      seasonalOrder: [0, 0, 0],
      period: 1,
      trend: 'n',
      regressionNames: [],
      regression: [],
      ar: [],
      ma: [],
    } as const
    const rest = {
      seasonalAr: [],
      seasonalMa: [],
      sigma2: 1,
      logLikelihood: 0,
      aic: 0,
      bic: 0,
      nobs: 10,
      stdErrors: [],
      method: 'mle',
      converged: true,
      iterations: 0,
    } as const
    const y = Array.from({ length: 30 }, (_, t) => t * t)
    const d1 = T.arimaForecast({ ...base, ...rest, order: [0, 1, 0] }, y, 5)
    const d2 = T.arimaForecast({ ...base, ...rest, order: [0, 2, 0] }, y, 5)
    d1.se.forEach((s, h) => {
      expect(s).toBeCloseTo(Math.sqrt(h + 1), 9)
    })
    // The second-order random walk: the variance after h steps is the sum of squared partial sums 1, 2, ..., h.
    d2.se.forEach((s, h) => {
      expect(s * s).toBeCloseTo(((h + 1) * (h + 2) * (2 * h + 3)) / 6, 6)
    })
    // The second differences are zero, so the forecast extrapolates the last slope: 2 * 29^2 - 28^2.
    expect(d2.mean[0]).toBeCloseTo(2 * 29 * 29 - 28 * 28, 9)
  })

  it('needs the exogenous rows of the future', () => {
    const fit = T.arimaFit([1, 2, 1, 3, 2, 4, 3, 5, 4, 6], {
      order: [0, 0, 0],
      trend: 'n',
      exog: [[1], [2], [1], [3], [2], [4], [3], [5], [4], [6]],
    })
    expect(() => T.arimaForecast(fit, [1, 2, 3], 2)).toThrow()
  })
})
