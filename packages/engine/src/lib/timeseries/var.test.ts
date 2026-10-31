import { describe, expect, it } from 'vitest'
import { simulateArma } from '../../testing/series.ts'
import * as T from './index.ts'

// Two independent AR(1) series: no Granger causality, a diagonal coefficient matrix.
const a = simulateArma(71, 400, [0.6], [])
const b = simulateArma(72, 400, [-0.4], [])
const y = a.map((v, t) => [v, b[t] ?? 0])

describe('VAR', () => {
  it('recovers a diagonal system and agrees with a univariate AR for one variable', () => {
    const f = T.varFit(y, { lags: 1, trend: 'n' })
    expect(f.coefficients[0]?.[0]?.[0]).toBeCloseTo(0.6, 1)
    expect(f.coefficients[0]?.[1]?.[1]).toBeCloseTo(-0.4, 1)
    expect(Math.abs(f.coefficients[0]?.[0]?.[1] ?? 1)).toBeLessThan(0.15)
    const single = T.varFit(
      a.map((v) => [v]),
      { lags: 2, trend: 'c' },
    )
    const ar = T.fitAr(a, 2, { method: 'ols' })
    expect(single.coefficients[0]?.[0]?.[0]).toBeCloseTo(ar.ar[0] ?? 0, 9)
    expect(single.coefficients[1]?.[0]?.[0]).toBeCloseTo(ar.ar[1] ?? 0, 9)
    expect(single.sigmaMle[0]?.[0]).toBeCloseTo(ar.sigma2, 9)
  })

  it('does not find Granger causality between independent series, and finds it when present', () => {
    expect(T.grangerCausality(T.varFit(y, { lags: 2 }), [1], [0]).pValue).toBeGreaterThan(0.01)
    const lead = a.map((v, t) => [v, 0.8 * (a[t - 1] ?? 0) + 0.5 * (b[t] ?? 0)])
    expect(T.grangerCausality(T.varFit(lead, { lags: 1 }), [1], [0]).pValue).toBeLessThan(1e-6)
  })

  it('has orthogonal responses that start at the Cholesky factor and variance shares that sum to one', () => {
    const f = T.varFit(y, { lags: 2 })
    const irf = T.varImpulseResponse(f, 5, { orthogonal: true })
    expect(irf[0]?.[0]?.[0]).toBeCloseTo(Math.sqrt(f.sigma[0]?.[0] ?? 0), 12)
    expect(irf[0]?.[0]?.[1]).toBeCloseTo(0, 12)
    for (const eq of T.varForecastErrorVarianceDecomposition(f, 6))
      for (const shares of eq) expect(shares.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12)
    const cum = T.varImpulseResponse(f, 3, { cumulative: true })
    expect(cum[1]?.[0]?.[0]).toBeCloseTo(
      (T.varImpulseResponse(f, 3)[0]?.[0]?.[0] ?? 0) + (T.varImpulseResponse(f, 3)[1]?.[0]?.[0] ?? 0),
      12,
    )
  })

  it('forecasts towards the mean with a widening interval', () => {
    const f = T.varFit(y, { lags: 1 })
    const fc = T.varForecast(f, y, 20)
    expect(Math.abs(fc.mean[19]?.[0] ?? 1)).toBeLessThan(0.3)
    expect((fc.upper[19]?.[0] ?? 0) - (fc.lower[19]?.[0] ?? 0)).toBeGreaterThan(
      (fc.upper[0]?.[0] ?? 0) - (fc.lower[0]?.[0] ?? 0),
    )
    expect(T.varStability(f).stable).toBe(true)
    expect(T.varStability({ coefficients: [[[1.2]]], nVars: 1 }).stable).toBe(false)
  })

  it('selects the true order of a VAR(1) and refuses short samples', () => {
    const sel = T.varSelectOrder(y, 4)
    expect(sel.selected.bic).toBe(1)
    expect(() => T.varFit(y.slice(0, 4), { lags: 2 })).toThrow()
  })
})
