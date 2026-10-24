import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, uniform } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const values = fc.array(
  fc.integer({ min: -10000, max: 10000 }).map((v) => v / 100),
  { minLength: 2, maxLength: 40 },
)

describe('running statistics', () => {
  it('Welford moments agree with the batch formulas, and merging chunks is exact', () => {
    fc.assert(
      fc.property(values, fc.nat(), (x, cut) => {
        fc.pre(new Set(x).size > 2)
        const k = cut % x.length
        const whole = S.momentsOf(S.pushAll(S.momentState(), x), { ddof: 1 })
        const merged = S.momentsOf(
          S.merge(S.pushAll(S.momentState(), x.slice(0, k)), S.pushAll(S.momentState(), x.slice(k))),
          { ddof: 1 },
        )
        expect(whole.mean).toBeCloseTo(S.mean(x), 9)
        expect(whole.variance).toBeCloseTo(S.variance(x, { ddof: 1 }), 6)
        expect(whole.skewness).toBeCloseTo(S.skewness(x), 6)
        expect(whole.kurtosis).toBeCloseTo(S.kurtosis(x), 6)
        expect(merged.variance).toBeCloseTo(whole.variance, 6)
        expect(merged.kurtosis).toBeCloseTo(whole.kurtosis, 6)
      }),
    )
  })

  it('P-squared tracks quantiles of a long uniform stream', () => {
    const x = toFlat(uniform(key(5), [4000]))
    for (const p of [0.1, 0.5, 0.9]) {
      let s = S.p2Quantile(p)
      for (const v of x) s = S.p2Push(s, v)
      expect(Math.abs(S.p2Estimate(s) - p)).toBeLessThan(0.02)
    }
  })

  it('the recursive and adjusted EWM means converge for long series of a constant', () => {
    const x = new Array<number>(200).fill(3)
    expect(toFlat(S.ewmMean(x, { alpha: 0.2 })).at(-1)).toBeCloseTo(3, 12)
    expect(toFlat(S.ewmMean(x, { alpha: 0.2, adjust: false })).at(-1)).toBeCloseTo(3, 12)
    expect(() => S.ewmState({ alpha: 0.2, span: 3 })).toThrow()
  })
})
