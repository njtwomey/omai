import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { asTensor, sum, toFlat } from '../../tensor/index.ts'
import { bsplineBasis, extendKnots, knotsFromLogits } from './bspline.ts'

describe('B-spline basis properties', () => {
  const interior = fc
    .array(fc.double({ min: 0.02, max: 0.98, noNaN: true }), { minLength: 1, maxLength: 6 })
    .map((v) => [...v].sort((a, b) => a - b))
  const points = fc.array(fc.double({ min: -0.5, max: 1.5, noNaN: true }), { minLength: 1, maxLength: 8 })

  it('is non-negative and a partition of unity for any ordered knots (repeated ones included), clamped outside the range', () => {
    fc.assert(
      fc.property(interior, fc.integer({ min: 1, max: 4 }), points, (knots, degree, x) => {
        const basis = bsplineBasis(x, extendKnots(knots, 0, 1, degree), degree)
        expect(Math.min(...toFlat(basis))).toBeGreaterThanOrEqual(-1e-12)
        for (const total of toFlat(sum(basis, 1))) expect(total).toBeCloseTo(1, 10)
      }),
      { numRuns: 60 },
    )
  })

  it('maps any logits to strictly increasing interior knots inside the range', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ min: -30, max: 30, noNaN: true }), { minLength: 3, maxLength: 9 }), (logits) => {
        const knots = toFlat(knotsFromLogits(asTensor(logits), -1, 2, 1e-3))
        expect(knots.every((k, i) => k > -1 && k < 2 && (i === 0 || k > (knots[i - 1] as number)))).toBe(true)
      }),
      { numRuns: 60 },
    )
  })
})
