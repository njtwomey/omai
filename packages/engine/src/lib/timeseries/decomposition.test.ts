import { describe, expect, it } from 'vitest'
import { expectResumable, simulateArma } from '../../testing/series.ts'
import { metricSeries, recorder, runAlgorithm } from '../trace/index.ts'
import * as T from './index.ts'

const season4 = [3, 0, -3, 0]
const series = (n: number) => Array.from({ length: n }, (_, t) => 10 + 0.5 * t + (season4[t % 4] ?? 0))

describe('classical decomposition', () => {
  it('recovers an exact trend and seasonal pattern', () => {
    const d = T.seasonalDecompose(series(40), 4)
    d.seasonal.forEach((v, t) => {
      expect(v).toBeCloseTo(season4[t % 4] ?? 0, 9)
    })
    for (let t = 2; t < 38; t++) expect(d.trend[t]).toBeCloseTo(10 + 0.5 * t, 9)
    expect(Number.isNaN(d.trend[0])).toBe(true)
    d.resid.slice(2, 38).forEach((v) => {
      expect(Math.abs(v)).toBeLessThan(1e-9)
    })
  })

  it('reconstructs the series from its components in either model', () => {
    const y = simulateArma(41, 60, [0.3], [], 30).map((v) => Math.abs(v) + 5)
    const add = T.seasonalDecompose(y, 6)
    const mul = T.seasonalDecompose(y, 6, { model: 'multiplicative' })
    y.forEach((v, t) => {
      const a = (add.trend[t] ?? Number.NaN) + (add.seasonal[t] ?? 0) + (add.resid[t] ?? 0)
      const m = (mul.trend[t] ?? Number.NaN) * (mul.seasonal[t] ?? 1) * (mul.resid[t] ?? 1)
      if (!Number.isNaN(a)) expect(a).toBeCloseTo(v, 9)
      if (!Number.isNaN(m)) expect(m).toBeCloseTo(v, 9)
    })
    expect(mul.seasonal.slice(0, 6).reduce((a, b) => a + b, 0) / 6).toBeCloseTo(1, 12)
  })

  it('extrapolates the trend over the ends', () => {
    const d = T.seasonalDecompose(series(40), 4, { extrapolateTrend: 'period' })
    expect(d.trend.every(Number.isFinite)).toBe(true)
    expect(d.trend[0]).toBeCloseTo(10, 6)
  })

  it('refuses short series and non-positive multiplicative ones', () => {
    expect(() => T.seasonalDecompose([1, 2, 3, 4, 5], 4)).toThrow()
    expect(() => T.seasonalDecompose([1, -2, 3, 4, 5, 6, 7, 8], 4, { model: 'multiplicative' })).toThrow()
  })
})

describe('STL', () => {
  const y = series(48).map((v, t) => v + 0.3 * Math.sin(t * 1.7))

  it('splits a series exactly into trend, seasonal and residual', () => {
    const d = T.stl(y, { period: 4 })
    y.forEach((v, t) => {
      expect((d.trend[t] ?? 0) + (d.seasonal[t] ?? 0) + (d.resid[t] ?? 0)).toBeCloseTo(v, 9)
    })
    expect(d.seasonal[0]).toBeGreaterThan(2)
    expect(d.seasonal[2]).toBeLessThan(-2)
  })

  it('down-weights an outlier when robust', () => {
    const spiked = [...y]
    spiked[20] = (spiked[20] ?? 0) + 40
    const plain = T.stl(spiked, { period: 4 })
    const robust = T.stl(spiked, { period: 4, robust: true })
    expect(robust.weights?.[20]).toBeLessThan(0.05)
    expect(Math.abs(robust.resid[20] ?? 0)).toBeGreaterThan(Math.abs(plain.resid[20] ?? 0))
  })

  it('runs one inner loop per step and records the residual variance', () => {
    const { algorithm, steps } = T.stlAlgorithm(y, { period: 4 })
    expect(steps).toBe(5)
    const rec = recorder()
    const run = runAlgorithm(algorithm, { maxSteps: 50, tracer: rec })
    expect(run.steps).toBe(5)
    const v = metricSeries(rec.run(), 'residualVariance').values
    expect(v).toHaveLength(6)
    expect(v[5]).toBeLessThanOrEqual((v[1] ?? 0) + 1e-9)
    expect(T.stlAlgorithm(y, { period: 4, robust: true }).steps).toBe(2 * 16)
    expectResumable(algorithm, 2, 3)
    expectResumable(T.stlAlgorithm(y, { period: 4, robust: true, outerIterations: 3 }).algorithm, 3, 4)
  })

  it('refuses invalid smoothers and short series', () => {
    expect(() => T.stl(y, { period: 4, seasonal: 8 })).toThrow()
    expect(() => T.stl(y, { period: 1 })).toThrow()
    expect(() => T.stl(y.slice(0, 6), { period: 4 })).toThrow()
    expect(() => T.stl(y, { period: 4, trend: 3 })).toThrow()
  })
})

describe('MSTL', () => {
  const n = 120
  const y = Array.from(
    { length: n },
    (_, t) => 0.05 * t + 2 * Math.sin((2 * Math.PI * t) / 6) + Math.sin((2 * Math.PI * t) / 24),
  )

  it('separates two seasonal periods and reconstructs the series', () => {
    const d = T.mstl(y, { periods: [24, 6] })
    expect(d.periods).toEqual([6, 24])
    expect(d.seasonal).toHaveLength(2)
    y.forEach((v, t) => {
      const total = (d.trend[t] ?? 0) + (d.seasonal[0]?.[t] ?? 0) + (d.seasonal[1]?.[t] ?? 0) + (d.resid[t] ?? 0)
      expect(total).toBeCloseTo(v, 8)
    })
    const amplitude = (s: readonly number[]) => Math.max(...s) - Math.min(...s)
    expect(amplitude(d.seasonal[0] ?? [])).toBeGreaterThan(3)
  })

  it('steps one STL fit at a time and serialises', () => {
    const { algorithm, steps } = T.mstlAlgorithm(y, { periods: [6, 24] })
    expect(steps).toBe(4)
    expectResumable(algorithm, 1, 2)
  })

  it('drops periods that are too long and refuses mismatched windows', () => {
    expect(T.mstl(y, { periods: [6, 100] }).periods).toEqual([6])
    expect(() => T.mstl(y, { periods: [6, 24], windows: [7] })).toThrow()
    expect(() => T.mstl(y, { periods: [100] })).toThrow()
  })
})
