import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  alignedText,
  caseFold,
  collapseWhitespace,
  filterAligned,
  mapAligned,
  normaliseAligned,
  originalSpan,
  replaceAligned,
  stripAccents,
  stripAligned,
} from './index.ts'

const unicodeText = fc.string({ unit: 'grapheme', maxLength: 12 })
const accented = fc
  .array(fc.constantFrom('a', 'e', 'é', 'é', 'ﬁ', 'Å', ' ', 'ß', '가', '가', 'Ω', '½', 'ẋ̣'), {
    maxLength: 10,
  })
  .map((xs) => xs.join(''))

describe('aligned text invariants', () => {
  it('normalising aligned text gives the plain normal form, with monotone alignments inside the original', () => {
    fc.assert(
      fc.property(
        fc.oneof(unicodeText, accented),
        fc.constantFrom('NFC', 'NFD', 'NFKC', 'NFKD' as const),
        (s, form) => {
          const r = normaliseAligned(alignedText(s), form)
          expect(r.text).toBe(s.normalize(form))
          expect(r.alignments.length).toBe(r.text.length)
          let prev = 0
          for (const [a, b] of r.alignments) {
            expect(a).toBeGreaterThanOrEqual(prev)
            expect(b).toBeGreaterThan(a)
            expect(b).toBeLessThanOrEqual(s.length)
            prev = a
          }
        },
      ),
    )
  })

  it('maps, filters, replacements and strips keep one alignment per code unit and the original intact', () => {
    fc.assert(
      fc.property(fc.oneof(unicodeText, accented), (s) => {
        const steps = [
          mapAligned(alignedText(s), (c) => c.toUpperCase()),
          filterAligned(alignedText(s), (c) => c !== ' '),
          replaceAligned(alignedText(s), /\s+/u, '_'),
          stripAligned(alignedText(s)),
        ]
        for (const r of steps) {
          expect(r.original).toBe(s)
          expect(r.alignments.length).toBe(r.text.length)
          const [a, b] = originalSpan(r, 0, r.text.length)
          expect(a).toBeGreaterThanOrEqual(0)
          expect(b).toBeLessThanOrEqual(s.length)
        }
      }),
    )
  })

  it('identity alignment spans recover the original slice', () => {
    fc.assert(
      fc.property(fc.string({ maxLength: 20 }), fc.nat(20), fc.nat(20), (s, i, j) => {
        const a = Math.min(i, j, s.length)
        const b = Math.min(Math.max(i, j), s.length)
        const [x, y] = originalSpan(alignedText(s), a, b)
        expect(x).toBeLessThanOrEqual(a)
        expect(y).toBeGreaterThanOrEqual(b)
      }),
    )
  })

  it('case folding is idempotent and caseless', () => {
    fc.assert(
      fc.property(unicodeText, (s) => {
        expect(caseFold(caseFold(s))).toBe(caseFold(s))
        expect(caseFold(s.toUpperCase())).toBe(caseFold(caseFold(s).toUpperCase()))
      }),
    )
  })

  it('accent stripping and whitespace collapsing are idempotent', () => {
    fc.assert(
      fc.property(fc.oneof(unicodeText, accented), (s) => {
        expect(stripAccents(stripAccents(s))).toBe(stripAccents(s))
        expect(collapseWhitespace(collapseWhitespace(s))).toBe(collapseWhitespace(s))
      }),
    )
  })
})
