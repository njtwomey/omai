import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import * as I from './index.ts'

const probs = (min: number, max: number): fc.Arbitrary<number[]> =>
  fc.array(fc.double({ min: 0.01, max: 1, noNaN: true }), { minLength: min, maxLength: max }).map((w) => {
    const s = w.reduce((a, b) => a + b, 0)
    return w.map((v) => v / s)
  })
const entropyIn = (p: readonly number[], arity: number): number =>
  -p.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0) / Math.log(arity)

/** The least expected length over all binary length vectors (lengths 1..m-1) that satisfy Kraft's inequality. */
function bruteForceOptimum(p: readonly number[]): number {
  const m = p.length
  let best = Number.POSITIVE_INFINITY
  const lengths = new Array<number>(m).fill(1)
  const visit = (i: number): void => {
    if (i === m) {
      if (I.kraftSum(lengths) <= 1 + 1e-12) best = Math.min(best, I.expectedLength(lengths, p))
      return
    }
    for (let l = 1; l < Math.max(2, m); l++) {
      lengths[i] = l
      visit(i + 1)
    }
  }
  visit(0)
  return best
}

describe('source codes (property)', () => {
  it('Huffman codes are prefix-free, complete (Kraft sum 1) and within one digit of the entropy', () => {
    fc.assert(
      fc.property(probs(2, 30), fc.integer({ min: 2, max: 4 }), (p, arity) => {
        const code = I.huffman(p, { arity })
        expect(I.isPrefixFree(code.codewords)).toBe(true)
        expect(I.isUniquelyDecodable(code.codewords)).toBe(true)
        const kraft = I.kraftSum(code.lengths, arity)
        expect(kraft).toBeLessThanOrEqual(1 + 1e-12)
        if (arity === 2) expect(kraft).toBeCloseTo(1, 12)
        const l = I.expectedLength(code, p)
        const h = entropyIn(p, arity)
        expect(l).toBeGreaterThanOrEqual(h - 1e-12)
        expect(l).toBeLessThan(h + 1)
      }),
      { numRuns: RUNS },
    )
  })

  it('Huffman codes are optimal among all prefix codes (brute force over length vectors)', () => {
    fc.assert(
      fc.property(probs(2, 5), (p) => {
        expect(I.expectedLength(I.huffman(p), p)).toBeCloseTo(bruteForceOptimum(p), 12)
        // and never worse than Fano's or Shannon's codes
        expect(I.expectedLength(I.huffman(p), p)).toBeLessThanOrEqual(I.expectedLength(I.shannonFano(p), p) + 1e-12)
        expect(I.expectedLength(I.huffman(p), p)).toBeLessThanOrEqual(I.expectedLength(I.shannonCode(p), p) + 1e-12)
      }),
      { numRuns: RUNS },
    )
  })

  it('canonical codes realise any lengths that satisfy Kraft (the converse of Kraft-McMillan)', () => {
    fc.assert(
      fc.property(fc.array(fc.integer({ min: 1, max: 8 }), { minLength: 1, maxLength: 20 }), (lengths) => {
        const kraft = I.kraftSum(lengths)
        if (kraft > 1) {
          expect(() => I.canonicalCode(lengths)).toThrow()
          return
        }
        const code = I.canonicalCode(lengths)
        expect(code.codewords.map((w) => w.length)).toEqual(lengths)
        expect(I.isPrefixFree(code.codewords)).toBe(true)
      }),
      { numRuns: RUNS },
    )
  })

  it('uniquely decodable codes satisfy McMillan; decoding inverts encoding', () => {
    fc.assert(
      fc.property(
        fc.array(fc.array(fc.integer({ min: 0, max: 1 }), { minLength: 1, maxLength: 4 }), {
          minLength: 1,
          maxLength: 6,
        }),
        (words) => {
          if (I.isUniquelyDecodable(words))
            expect(I.kraftSum(words.map((w) => w.length))).toBeLessThanOrEqual(1 + 1e-12)
          if (I.isPrefixFree(words)) expect(I.isUniquelyDecodable(words)).toBe(true)
        },
      ),
      { numRuns: RUNS * 5 },
    )
    fc.assert(
      fc.property(probs(2, 12), fc.array(fc.nat(), { maxLength: 40 }), (p, raw) => {
        const symbols = raw.map((r) => r % p.length)
        for (const code of [I.huffman(p), I.shannonFano(p), I.shannonCode(p), I.huffman(p, { arity: 3 })]) {
          expect(I.prefixDecode(code, I.prefixEncode(code, symbols))).toEqual(symbols)
        }
      }),
      { numRuns: RUNS },
    )
  })
})
