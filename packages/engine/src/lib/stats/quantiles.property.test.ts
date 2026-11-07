import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const values = fc.array(
  fc.integer({ min: -1000, max: 1000 }).map((v) => v / 10),
  { minLength: 1, maxLength: 30 },
)
const METHODS: readonly S.QuantileMethod[] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

describe('quantiles', () => {
  it('are monotone in the level and bounded by the extremes for every method', () => {
    fc.assert(
      fc.property(values, fc.constantFrom(...METHODS), (x, method) => {
        const q = toFlat(S.quantile(x, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1], { method }))
        for (let i = 1; i < q.length; i++) expect(q[i] ?? 0).toBeGreaterThanOrEqual((q[i - 1] ?? 0) - 1e-12)
        const lo = x.reduce((m, v) => Math.min(m, v), Number.POSITIVE_INFINITY)
        const hi = x.reduce((m, v) => Math.max(m, v), Number.NEGATIVE_INFINITY)
        expect(q[0] ?? 0).toBeGreaterThanOrEqual(lo)
        expect(q.at(-1) ?? 0).toBeLessThanOrEqual(hi)
      }),
    )
  })

  it('the ECDF steps by 1/n at each observation and matches percentileOfScore', () => {
    fc.assert(
      fc.property(values, (x) => {
        const F = S.ecdf(x)
        for (const v of x)
          expect(toFlat(F.evaluate([v]))[0]).toBeCloseTo(S.percentileOfScore(x, v, { kind: 'weak' }) / 100, 12)
        expect(toFlat(F.probabilities).at(-1)).toBeCloseTo(1, 12)
      }),
    )
  })

  it('MAD and IQR are invariant to shifts and scale with the data', () => {
    fc.assert(
      fc.property(values, fc.integer({ min: -50, max: 50 }), (x, c) => {
        const y = x.map((v) => 3 * v + c)
        expect(S.medianAbsoluteDeviation(y)).toBeCloseTo(3 * S.medianAbsoluteDeviation(x), 8)
        expect(S.iqr(y)).toBeCloseTo(3 * S.iqr(x), 8)
      }),
    )
  })

  it('rejects bad levels and methods', () => {
    expect(() => S.quantile([1, 2], 1.5)).toThrow()
    expect(() => S.quantile([1, 2], 0.5, { method: 'nope' as S.QuantileMethod })).toThrow()
    expect(Number.isNaN(S.quantile([1, Number.NaN], 0.5))).toBe(true)
  })
})

describe('ranks', () => {
  it('sum to n (n + 1) / 2 and are invariant to monotone transforms', () => {
    fc.assert(
      fc.property(values, fc.constantFrom<S.RankMethod>('average', 'ordinal'), (x, method) => {
        const r = toFlat(S.rankdata(x, { method }))
        expect(r.reduce((s, v) => s + v, 0)).toBeCloseTo((x.length * (x.length + 1)) / 2, 9)
        expect(
          toFlat(
            S.rankdata(
              x.map((v) => Math.exp(v / 50)),
              { method },
            ),
          ),
        ).toEqual(r)
      }),
    )
  })

  it('tie corrections are 1 without ties and below 1 with them', () => {
    expect(S.tieCorrection([3, 1, 2])).toBe(1)
    expect(S.tieCorrection([1, 1, 2])).toBeLessThan(1)
    expect(Array.from(S.tieCounts([2, 2, 2]).data)).toEqual([3])
    expect(() => S.rankdata([1, Number.NaN])).toThrow()
  })
})
