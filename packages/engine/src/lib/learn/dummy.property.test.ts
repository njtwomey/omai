import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toArray, toFlat } from '../tensor/index.ts'
import { dummyClassifier, dummyRegressor } from './dummy.ts'

const labels = fc.array(fc.integer({ min: 0, max: 4 }), { minLength: 2, maxLength: 40 })
const values = fc.array(fc.double({ min: -100, max: 100, noNaN: true }), { minLength: 2, maxLength: 40 })
const rowsOf = (n: number): number[][] => Array.from({ length: n }, (_, i) => [i])

describe('dummy properties', () => {
  it('the prior sums to one and prior and most_frequent predict the same, most frequent class', () => {
    fc.assert(
      fc.property(labels, (y) => {
        const x = rowsOf(y.length)
        const f = dummyClassifier().fit({ x, y })
        expect(f.params.prior.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 12)
        const top = Math.max(...f.params.prior)
        const predicted = toFlat(f.predict(x))[0] ?? -1
        expect(f.params.prior[f.classes.indexOf(predicted)]).toBe(top)
        expect(toFlat(dummyClassifier({ strategy: 'most_frequent' }).fit({ x, y }).predict(x))).toEqual(
          toFlat(f.predict(x)),
        )
        for (const r of toArray(f.predictProba(x)) as number[][])
          expect(r.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 12)
      }),
    )
  })

  it('the quantile is monotone in q and stays within the targets; the mean strategy has zero R^2', () => {
    fc.assert(
      fc.property(
        values,
        fc.double({ min: 0, max: 1, noNaN: true }),
        fc.double({ min: 0, max: 1, noNaN: true }),
        (y, a, b) => {
          const x = rowsOf(y.length)
          const [lo, hi] = a <= b ? [a, b] : [b, a]
          const qlo = dummyRegressor({ strategy: 'quantile', quantile: lo }).fit({ x, y }).params.constant
          const qhi = dummyRegressor({ strategy: 'quantile', quantile: hi }).fit({ x, y }).params.constant
          expect(qlo).toBeLessThanOrEqual(qhi + 1e-12)
          expect(qlo).toBeGreaterThanOrEqual(Math.min(...y) - 1e-12)
          expect(qhi).toBeLessThanOrEqual(Math.max(...y) + 1e-12)
          if (Math.max(...y) - Math.min(...y) > 1e-3) expect(dummyRegressor().fit({ x, y }).score({ x, y })).toBeCloseTo(0, 8)
        },
      ),
    )
  })
})
