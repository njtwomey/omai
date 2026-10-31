import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { gini } from './descriptive.ts'

describe('gini', () => {
  it('is 0 for equal values and (n - 1) / n for one holder', () => {
    expect(gini([3, 3, 3])).toBeCloseTo(0, 14)
    expect(gini([0, 0, 0, 0, 10])).toBeCloseTo(0.8, 14)
    expect(gini([0, 0])).toBe(0)
  })
  it('matches the textbook value for 1, 2, 3, 4 (mean absolute difference 5/4 over 2 mean = 0.25)', () => {
    expect(gini([1, 2, 3, 4])).toBeCloseTo(0.25, 14)
  })
  it('equals the double-sum definition and is scale-free', () => {
    fc.assert(
      fc.property(fc.array(fc.integer({ min: 0, max: 1000 }), { minLength: 1, maxLength: 30 }), (x) => {
        const n = x.length
        const mean = x.reduce((a, b) => a + b, 0) / n
        let d = 0
        for (const a of x) for (const b of x) d += Math.abs(a - b)
        const direct = mean === 0 ? 0 : d / (2 * n * n * mean)
        expect(gini(x)).toBeCloseTo(direct, 9)
        expect(gini(x.map((v) => 7 * v))).toBeCloseTo(gini(x), 9)
      }),
    )
  })
  it('rejects negative values', () => {
    expect(() => gini([1, -1])).toThrow(/non-negative/)
  })
})
