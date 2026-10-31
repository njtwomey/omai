import { describe, expect, it } from 'vitest'
import { boxcoxNormmax } from '../stats/index.ts'
import * as T from './index.ts'

const multiplicative = Array.from(
  { length: 120 },
  (_, t) => Math.exp(0.025 * t) * (3 + Math.sin((2 * Math.PI * t) / 12)),
)

describe('Guerrero lambda', () => {
  it('is the logarithm for a spread proportional to the level and one for a constant spread', () => {
    expect(T.guerreroLambda([1, 2, 3, 2, 10, 20, 30, 20, 100, 200, 300, 200], 4)).toBeCloseTo(0, 4)
    // Blocks with equal spread and increasing level: sd / mean^(1 - lambda) is constant when lambda = 1.
    const additive = [1, 2, 3, 2, 11, 12, 13, 12, 21, 22, 23, 22]
    expect(T.guerreroLambda(additive, 4)).toBeGreaterThan(0.9)
  })
  it('rejects non-positive data and too few blocks', () => {
    expect(() => T.guerreroLambda([1, -1, 2, 3, 4, 5, 6, 7], 4)).toThrow()
    expect(() => T.guerreroLambda([1, 2, 3, 4], 4)).toThrow()
  })
})

describe('Box-Cox MSTL', () => {
  it('chooses a lambda near the logarithm for multiplicative seasonality and decomposes the log scale', () => {
    const r = T.boxCoxMstl(multiplicative, { periods: [12] })
    expect(Math.abs(r.lambda)).toBeLessThan(0.3)
    const log = T.boxCoxMstl(multiplicative, { periods: [12], lambda: 0 })
    // On the log scale the seasonal amplitude is constant from the first cycle to the last.
    const s = log.decomposition.seasonal[0] as readonly number[]
    const amplitude = (from: number): number =>
      Math.max(...s.slice(from, from + 12)) - Math.min(...s.slice(from, from + 12))
    expect(amplitude(96) / amplitude(12)).toBeCloseTo(1, 1)
    // The additive decomposition of the raw series instead lets the amplitude grow with the level.
    const raw = T.mstl(multiplicative, { periods: [12] }).seasonal[0] as readonly number[]
    const rawAmp = (from: number): number =>
      Math.max(...raw.slice(from, from + 12)) - Math.min(...raw.slice(from, from + 12))
    expect(rawAmp(96) / rawAmp(12)).toBeGreaterThan(3)
  })
  it('carries the components back: the fit reproduces the series and the factors are the seasonal ratios', () => {
    const r = T.boxCoxMstl(multiplicative, { periods: [12], lambda: 0 })
    for (let t = 0; t < multiplicative.length; t += 7)
      expect(Math.abs((r.fitted[t] ?? 0) / (multiplicative[t] ?? 1) - 1)).toBeLessThan(0.03)
    // The seasonal factor at the log scale is exp(s): trend * factor = trend * exp(s).
    const s = r.decomposition.seasonal[0]?.[30] ?? 0
    expect(r.seasonalFactors[0]?.[30]).toBeCloseTo(Math.exp(s), 10)
  })
  it('takes the profile-likelihood lambda on request', () => {
    const r = T.boxCoxMstl(multiplicative, { periods: [12], lambda: 'mle' })
    expect(r.lambda).toBeCloseTo(boxcoxNormmax(multiplicative), 10)
  })
})
