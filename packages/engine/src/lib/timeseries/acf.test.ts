import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { simulateArma } from '../../testing/series.ts'
import * as T from './index.ts'

const series = fc.array(fc.double({ min: -50, max: 50, noNaN: true, noDefaultInfinity: true }), {
  minLength: 10,
  maxLength: 60,
})

describe('sample correlation functions', () => {
  it('starts at one and stays inside [-1, 1] for the biased estimator', () => {
    fc.assert(
      fc.property(series, (y) => {
        fc.pre(new Set(y).size > 2)
        // The correlations divide by the variance, which underflows to 0 (a NaN, as in statsmodels) for a series of
        // denormal numbers such as [0, ..., 5e-324, -5e-324] (a counterexample that fast-check found): ask for a
        // variance a double can carry.
        const mean = y.reduce((a, v) => a + v, 0) / y.length
        fc.pre(y.reduce((a, v) => a + (v - mean) ** 2, 0) > 1e-12)
        const { acf } = T.acf(y, 8)
        expect(acf[0]).toBeCloseTo(1, 12)
        for (const v of acf) expect(Math.abs(v)).toBeLessThanOrEqual(1 + 1e-9)
      }),
    )
  })

  it('relates the adjusted and biased autocovariances by n / (n - k)', () => {
    const y = simulateArma(21, 50, [0.5], [])
    const biased = T.autocovariance(y, 5)
    const adjusted = T.autocovariance(y, 5, { adjusted: true })
    adjusted.forEach((v, k) => {
      expect(v).toBeCloseTo(((biased[k] ?? 0) * 50) / (50 - k), 12)
    })
  })

  it('has a partial autocorrelation that cuts off for an autoregression', () => {
    const y = simulateArma(22, 2000, [0.6, -0.3], [])
    const p = T.pacf(y, 6)
    expect(Math.abs(p[2] ?? 0)).toBeGreaterThan(0.2)
    for (let k = 3; k <= 6; k++) expect(Math.abs(p[k] ?? 1)).toBeLessThan(0.1)
  })

  it('is the autocorrelation when a series is cross-correlated with itself', () => {
    const y = simulateArma(23, 80, [0.4], [0.2])
    const c = T.ccf(y, y, 6, false)
    T.acf(y, 6).acf.forEach((v, k) => {
      expect(c[k]).toBeCloseTo(v, 10)
    })
  })

  it('widens the Bartlett band with the lag', () => {
    const band = T.acf(simulateArma(24, 100, [0.7], []), 6, { alpha: 0.05 }).band ?? new Float64Array()
    expect(band[2]).toBeGreaterThan(band[1] ?? 0)
    expect(band[1]).toBeCloseTo(1.959964 / 10, 5)
  })

  it('refuses a lag beyond the series', () => {
    expect(() => T.acf([1, 2, 3], 3)).toThrow()
  })
})
