import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { tensor, toFlat, type Value } from '../tensor/index.ts'
import { clipPerExample, dpEpsilon } from './dp-sgd.ts'

describe('clipping', () => {
  it('bounds every example’s norm by C and keeps its direction, for any gradients', () => {
    fc.assert(
      fc.property(
        fc.array(fc.array(fc.double({ min: -50, max: 50, noNaN: true }), { minLength: 4, maxLength: 4 }), {
          minLength: 1,
          maxLength: 8,
        }),
        fc.double({ min: 0.01, max: 20, noNaN: true }),
        (rows, c) => {
          const clipped = clipPerExample({ w: tensor(rows) }, c)
          const out = toFlat((clipped.grads as { w: Value }).w)
          rows.forEach((row, i) => {
            const got = out.slice(4 * i, 4 * i + 4)
            const before = Math.hypot(...row)
            expect(Math.hypot(...got)).toBeLessThanOrEqual(c + 1e-9)
            if (before <= c * 0.999) {
              const factor = Math.min(1, c / (before + 1e-6))
              for (const [k, v] of got.entries()) expect(v).toBeCloseTo((row[k] as number) * factor, 9)
            }
            if (before > 1e-6) {
              // Same direction: the clipped row is a non-negative multiple of the original.
              const dot = got.reduce((a, v, k) => a + v * (row[k] as number), 0)
              expect(dot).toBeGreaterThanOrEqual(-1e-9)
            }
          })
        },
      ),
    )
  })
})

describe('the accountant is monotone', () => {
  it('epsilon rises with steps and falls with noise', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0.001, max: 0.2, noNaN: true }),
        fc.double({ min: 0.8, max: 4, noNaN: true }),
        fc.integer({ min: 10, max: 500 }),
        (q, sigma, steps) => {
          const e = (s: number, t: number): number =>
            dpEpsilon({ sampleRate: q, noiseMultiplier: s, steps: t, delta: 1e-5 }).epsilon
          expect(e(sigma, steps)).toBeLessThanOrEqual(e(sigma, steps * 2) + 1e-12)
          expect(e(sigma * 1.5, steps)).toBeLessThanOrEqual(e(sigma, steps) + 1e-12)
        },
      ),
      { numRuns: 25 },
    )
  })
})
