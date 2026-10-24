import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const sampleArb = fc.array(
  fc.integer({ min: -1000, max: 1000 }).map((v) => v / 10),
  { minLength: 4, maxLength: 30 },
)
const spread = (xs: readonly number[]): boolean => new Set(xs).size > 2

describe('descriptive statistics', () => {
  it('locate and scale as their definitions say', () => {
    fc.assert(
      fc.property(
        sampleArb,
        fc.double({ min: -50, max: 50, noNaN: true }),
        fc.double({ min: 0.1, max: 10, noNaN: true }),
        (x, c, a) => {
          fc.pre(spread(x))
          const y = x.map((v) => a * v + c)
          expect(S.mean(y)).toBeCloseTo(a * S.mean(x) + c, 6)
          expect(S.variance(y, { ddof: 1 }) / (a * a * S.variance(x, { ddof: 1 }))).toBeCloseTo(1, 6)
          expect(S.skewness(y)).toBeCloseTo(S.skewness(x), 5)
          expect(S.kurtosis(y)).toBeCloseTo(S.kurtosis(x), 5)
        },
      ),
    )
  })

  it('treats integer frequency weights as repeated observations', () => {
    fc.assert(
      fc.property(sampleArb, fc.array(fc.integer({ min: 1, max: 4 }), { minLength: 30, maxLength: 30 }), (x, w0) => {
        fc.pre(spread(x))
        const w = w0.slice(0, x.length)
        const repeated = x.flatMap((v, i) => new Array<number>(w[i] ?? 1).fill(v))
        expect(S.mean(x, { weights: w })).toBeCloseTo(S.mean(repeated), 8)
        expect(S.variance(x, { weights: w, ddof: 1 })).toBeCloseTo(S.variance(repeated, { ddof: 1 }), 6)
        expect(S.skewness(x, { weights: w })).toBeCloseTo(S.skewness(repeated), 6)
      }),
    )
  })

  it('standardises to zero mean and unit variance', () => {
    fc.assert(
      fc.property(sampleArb, (x) => {
        fc.pre(spread(x))
        const z = toFlat(S.zscore(x))
        expect(S.mean(z)).toBeCloseTo(0, 8)
        expect(S.variance(z)).toBeCloseTo(1, 8)
      }),
    )
  })

  it('orders the means of positive data: harmonic <= geometric <= arithmetic', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ min: 0.01, max: 100, noNaN: true }), { minLength: 1, maxLength: 20 }), (x) => {
        const h = S.harmonicMean(x)
        const g = S.geometricMean(x)
        expect(h).toBeLessThanOrEqual(g * (1 + 1e-12))
        expect(g).toBeLessThanOrEqual(S.mean(x) * (1 + 1e-12))
      }),
    )
  })

  it('finds the mode and rejects invalid input', () => {
    expect(S.mode([1, 1, 2, 2, 3])).toEqual({ mode: 1, count: 2 })
    expect(() => S.geometricMean([1, -1])).toThrow()
    expect(() => S.moment([1, 2], 1.5)).toThrow()
    expect(S.moment([1, 2, 3], 0)).toBe(1)
    expect(Number.isNaN(S.skewness([2, 2, 2]))).toBe(true)
  })
})
