import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import * as I from './index.ts'

const text = fc.array(fc.integer({ min: 0, max: 3 }), { maxLength: 200 })
const bytes = fc.array(fc.integer({ min: 0, max: 255 }), { maxLength: 200 })

describe('dictionary codes (property)', () => {
  it('LZ77, LZ78 and LZW round-trip', () => {
    fc.assert(
      fc.property(fc.oneof(text, bytes), fc.integer({ min: 1, max: 64 }), (xs, window) => {
        expect(I.lz77Decode(I.lz77Encode(xs, { window, maxLength: 16 }))).toEqual(xs)
        expect(I.lz78Decode(I.lz78Encode(xs))).toEqual(xs)
        expect(I.lzwDecode(I.lzwEncode(xs))).toEqual(xs)
        expect(I.lzwDecode(I.lzwEncode(xs, { maxCodes: 260 }), { maxCodes: 260 })).toEqual(xs)
      }),
      { numRuns: RUNS },
    )
  })

  it('every LZ77 token respects the window and the length limit', () => {
    fc.assert(
      fc.property(text, (xs) => {
        for (const t of I.lz77Encode(xs, { window: 8, maxLength: 5 })) {
          expect(t.offset).toBeLessThanOrEqual(8)
          expect(t.length).toBeLessThanOrEqual(5)
        }
      }),
      { numRuns: RUNS },
    )
  })

  it('BWT and move-to-front are invertible permutations and codes', () => {
    fc.assert(
      fc.property(fc.oneof(text, bytes), (xs) => {
        const t = I.bwt(xs)
        expect([...t.last].sort((a, b) => a - b)).toEqual([...xs].sort((a, b) => a - b))
        expect(I.inverseBwt(t)).toEqual(xs)
        expect(I.inverseMoveToFront(I.moveToFront(xs))).toEqual(xs)
      }),
      { numRuns: RUNS },
    )
  })

  it('periodic inputs (equal rotations) still invert', () => {
    for (const s of ['abab', 'aaaa', 'abcabcabc', 'a'])
      expect(I.inverseBwt(I.bwt(s))).toEqual(Array.from(new TextEncoder().encode(s)))
  })
})
