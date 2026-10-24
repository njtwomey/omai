import { describe, expect, it } from 'vitest'
import { exponential, key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const integrate = (f: (x: number) => number, lo: number, hi: number, n = 4000): number => {
  const h = (hi - lo) / n
  let s = 0.5 * (f(lo) + f(hi))
  for (let i = 1; i < n; i++) s += f(lo + i * h)
  return s * h
}

describe('kernel density estimation', () => {
  it('integrates to one, with and without boundary correction', () => {
    const x = toFlat(normal(key(1), [60]))
    const kde = S.gaussianKde(x)
    expect(integrate((t) => toFlat(kde.evaluate([t]))[0] ?? 0, -8, 8)).toBeCloseTo(1, 4)
    const y = toFlat(exponential(key(2), [60]))
    const reflected = S.gaussianKde(y, { boundary: { lower: 0 } })
    expect(integrate((t) => toFlat(reflected.evaluate([t]))[0] ?? 0, 0, 12)).toBeCloseTo(1, 4)
    expect(toFlat(reflected.evaluate([-0.1]))[0]).toBe(0)
  })

  it('resamples from the estimate and cross-validates a bandwidth near the rule of thumb', () => {
    const x = toFlat(normal(key(3), [80]))
    const kde = S.gaussianKde(x, { bandwidth: 'silverman' })
    const draws = toFlat(kde.resample(key(4), 4000))
    expect(S.mean(draws)).toBeCloseTo(S.mean(x), 1)
    for (const method of ['likelihood', 'least-squares'] as const) {
      const h = S.bandwidthCrossValidation(x, { method })
      const rule = S.bandwidthSilverman(x)
      expect(h / rule).toBeGreaterThan(0.3)
      expect(h / rule).toBeLessThan(3)
      expect(S.bandwidthCriterion(x, h, { method })).toBeGreaterThanOrEqual(
        S.bandwidthCriterion(x, h * 1.1, { method }),
      )
    }
    expect(S.gaussianKde(x, { bandwidth: 'cv-likelihood' }).factor).toBeGreaterThan(0)
  })

  it('handles weights and rejects bad input', () => {
    const kde = S.gaussianKde([0, 1, 2], { weights: [1, 0, 1] })
    expect(kde.neff).toBeCloseTo(2, 12)
    expect(() => S.gaussianKde([1])).toThrow()
    expect(() => S.gaussianKde([1, 1, 1])).toThrow()
  })
})
