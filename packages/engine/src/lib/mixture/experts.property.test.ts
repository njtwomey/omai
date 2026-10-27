import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { moeGate } from './experts.ts'

describe('mixture of experts: properties', () => {
  it('gate probabilities are non-negative and sum to one for any weights and points', () => {
    const weight = fc.double({ min: -50, max: 50, noNaN: true })
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 6 }),
        fc.integer({ min: 1, max: 3 }),
        fc.array(fc.array(fc.double({ min: -20, max: 20, noNaN: true }), { minLength: 3, maxLength: 3 }), {
          minLength: 1,
          maxLength: 8,
        }),
        fc.array(fc.array(weight, { minLength: 4, maxLength: 4 }), { minLength: 6, maxLength: 6 }),
        (k, d, rows, weights) => {
          const x = rows.map((r) => r.slice(0, d))
          const gate = weights.slice(0, k).map((w) => w.slice(0, d + 1))
          for (const row of moeGate(x, gate)) {
            for (const p of row) expect(p >= 0 && p <= 1).toBe(true)
            expect(row.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 12)
          }
        },
      ),
      { numRuns: 100 },
    )
  })
})
