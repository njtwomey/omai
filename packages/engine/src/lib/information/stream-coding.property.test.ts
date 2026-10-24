import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import * as I from './index.ts'

const freqs = fc.array(fc.integer({ min: 1, max: 500 }), { minLength: 2, maxLength: 12 })

describe('arithmetic and range coding (property)', () => {
  it('decoding inverts encoding, for static and adaptive models', () => {
    fc.assert(
      fc.property(freqs, fc.array(fc.nat(), { maxLength: 300 }), (f, raw) => {
        const symbols = raw.map((r) => r % f.length)
        const fixed = { kind: 'static', frequencies: f } as const
        const adaptive = { kind: 'adaptive', alphabetSize: f.length, increment: 8 } as const
        for (const spec of [fixed, adaptive]) {
          expect(I.arithmeticDecode(I.arithmeticEncode(symbols, spec), symbols.length, spec)).toEqual(symbols)
          expect(I.rangeDecode(I.rangeEncode(symbols, spec), symbols.length, spec)).toEqual(symbols)
        }
      }),
      { numRuns: RUNS },
    )
  })

  it('the output length is within a few bits of the information content', () => {
    fc.assert(
      fc.property(freqs, fc.array(fc.nat(), { minLength: 1, maxLength: 400 }), (f, raw) => {
        const total = f.reduce((a, b) => a + b, 0)
        const symbols = raw.map((r) => r % f.length)
        const spec = { kind: 'static', frequencies: f } as const
        const information = symbols.reduce((s, x) => s - Math.log2((f[x] ?? 0) / total), 0)
        const bits = I.arithmeticEncode(symbols, spec).length
        expect(bits).toBeLessThanOrEqual(Math.ceil(information) + 2 + 1e-9 * symbols.length)
        expect(bits).toBeGreaterThanOrEqual(Math.floor(information) - 1)
        // the carry-less range coder loses a little to its clipped ranges and four flush bytes
        expect(8 * I.rangeEncode(symbols, spec).length).toBeLessThanOrEqual(1.01 * information + 48)
      }),
      { numRuns: RUNS },
    )
  })

  it('frequency tables keep every positive probability and hit the total', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: 1, maxLength: 50 }), (p) => {
        if (!p.some((v) => v > 0)) return
        const f = I.frequencyTable(p, 4096)
        expect(f.reduce((a, b) => a + b, 0)).toBe(4096)
        p.forEach((v, i) => {
          expect((f[i] ?? 0) > 0).toBe(v > 0)
        })
      }),
      { numRuns: RUNS },
    )
  })
})
