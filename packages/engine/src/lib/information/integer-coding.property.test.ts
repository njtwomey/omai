import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import * as I from './index.ts'

describe('integer codes (property)', () => {
  it('Elias codes round-trip and have their textbook lengths', () => {
    fc.assert(
      fc.property(fc.array(fc.integer({ min: 1, max: 2 ** 40 }), { maxLength: 20 }), (xs) => {
        for (const kind of ['gamma', 'delta', 'omega'] as const) {
          expect(I.eliasDecode(I.eliasEncode(xs, kind), kind)).toEqual(xs)
        }
        for (const n of xs) {
          const l = Math.floor(Math.log2(n))
          expect(I.eliasEncode([n], 'gamma').length).toBe(2 * l + 1)
          expect(I.eliasEncode([n], 'delta').length).toBe(l + 2 * Math.floor(Math.log2(l + 1)) + 1)
        }
      }),
      { numRuns: RUNS },
    )
  })

  it('Elias codewords are prefix-free (each set of codewords for 1..64)', () => {
    for (const kind of ['gamma', 'delta', 'omega'] as const) {
      const words = Array.from({ length: 64 }, (_, i) => I.eliasEncode([i + 1], kind))
      expect(I.isPrefixFree(words)).toBe(true)
      expect(I.kraftSum(words.map((w) => w.length))).toBeLessThanOrEqual(1)
    }
  })

  it('Golomb and Rice codes round-trip, and Rice remainders take exactly k bits', () => {
    fc.assert(
      fc.property(fc.array(fc.nat({ max: 5000 }), { maxLength: 30 }), fc.integer({ min: 1, max: 70 }), (xs, m) => {
        expect(I.golombDecode(I.golombEncode(xs, m), m)).toEqual(xs)
      }),
      { numRuns: RUNS },
    )
    fc.assert(
      fc.property(fc.nat({ max: 1000 }), fc.integer({ min: 0, max: 6 }), (n, k) => {
        const m = 2 ** k
        expect(I.golombEncode([n], m).length).toBe(Math.floor(n / m) + 1 + k)
      }),
      { numRuns: RUNS },
    )
  })

  it('run-length coding round-trips', () => {
    fc.assert(
      fc.property(fc.array(fc.integer({ min: 0, max: 2 }), { maxLength: 60 }), (xs) => {
        const runs = I.runLengthEncode(xs)
        expect(I.runLengthDecode(runs)).toEqual(xs)
        expect(runs.values.every((v, i) => i === 0 || v !== runs.values[i - 1])).toBe(true)
      }),
      { numRuns: RUNS },
    )
  })
})
