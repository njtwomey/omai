import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { frobeniusDistance } from './_matrix.ts'
import { clipCorrelation, isCorrelationMatrix, nearestCorrelation } from './nearest.ts'

/** A symmetric unit-diagonal matrix of arbitrary entries in [-1, 1]: usually not positive semidefinite. */
const unitSymmetric = fc.integer({ min: 2, max: 7 }).chain((p) =>
  fc
    .array(fc.double({ min: -1, max: 1, noNaN: true }), { minLength: (p * (p - 1)) / 2, maxLength: (p * (p - 1)) / 2 })
    .map((u) => {
      const m = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => (i === j ? 1 : 0)))
      let k = 0
      for (let i = 0; i < p; i++)
        for (let j = i + 1; j < p; j++) {
          const v = u[k++] ?? 0
          ;(m[i] as number[])[j] = v
          ;(m[j] as number[])[i] = v
        }
      return m
    }),
)

describe('nearestCorrelation on arbitrary symmetric unit-diagonal matrices', () => {
  it('always returns a valid correlation matrix no farther than the clipped one', () => {
    fc.assert(
      fc.property(unitSymmetric, (a) => {
        const r = nearestCorrelation(a, { tol: 1e-12 })
        expect(isCorrelationMatrix(r.matrix, 1e-9)).toBe(true)
        const clipped = clipCorrelation(a, 1e-9)
        expect(r.distance).toBeLessThanOrEqual(clipped.distance + 1e-6)
      }),
      { numRuns: 40 },
    )
  })

  it('is idempotent: the nearest matrix of a valid one is itself', () => {
    fc.assert(
      fc.property(unitSymmetric, (a) => {
        const once = nearestCorrelation(a, { tol: 1e-13 }).matrix
        expect(frobeniusDistance(nearestCorrelation(once).matrix, once)).toBeLessThan(1e-9)
      }),
      { numRuns: 30 },
    )
  })

  it('leaves a matrix that is already valid unchanged', () => {
    fc.assert(
      fc.property(unitSymmetric, (a) => {
        const valid = isCorrelationMatrix(a, 1e-12)
        fc.pre(valid)
        expect(frobeniusDistance(nearestCorrelation(a).matrix, a)).toBeLessThan(1e-10)
      }),
      { numRuns: 200 },
    )
  })
})
