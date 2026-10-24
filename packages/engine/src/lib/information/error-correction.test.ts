import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import * as I from './index.ts'

const bits = (n: number): fc.Arbitrary<number[]> =>
  fc.array(fc.integer({ min: 0, max: 1 }), { minLength: n, maxLength: n })

describe('linear block codes', () => {
  it('Hamming codes correct every single error and the extended code detects double errors', () => {
    for (const r of [2, 3, 4]) {
      const code = I.hammingCode(r)
      expect([code.n, code.k]).toEqual([2 ** r - 1, 2 ** r - 1 - r])
      const decode = I.syndromeDecoder(code)
      const message = Array.from({ length: code.k }, (_, i) => (i * 7 + r) % 2)
      const word = I.linearEncode(code, message)
      expect(I.syndrome(code, word).every((v) => v === 0)).toBe(true)
      for (let e = 0; e < code.n; e++) {
        const received = word.map((b, i) => (i === e ? 1 - b : b))
        // Hamming's arrangement: the syndrome is the position, least significant bit first.
        expect(I.syndrome(code, received).reduce((s, v, b) => s + v * 2 ** b, 0)).toBe(e + 1)
        const result = decode(received)
        expect(result.message).toEqual(message)
        expect(result.corrected).toEqual([e])
      }
    }
    const extended = I.hammingCode(3, { extended: true })
    const word = I.linearEncode(extended, [1, 1, 0, 1])
    const double = word.map((b, i) => (i === 0 || i === 3 ? 1 - b : b))
    const s = I.syndrome(extended, double)
    // Overall parity holds but the Hamming syndrome does not vanish: a detected double error.
    expect(s[s.length - 1]).toBe(0)
    expect(s.slice(0, -1).some((v) => v === 1)).toBe(true)
    expect(() => I.hammingCode(1)).toThrow()
  })

  it('codewords of any linear code have zero syndrome; minimum distance bounds correction (property)', () => {
    fc.assert(
      fc.property(fc.array(bits(8), { minLength: 2, maxLength: 4 }), fc.nat(), (rows, seed) => {
        let code: I.LinearCode
        try {
          code = I.linearCode(rows)
        } catch {
          return
        }
        const message = Array.from({ length: code.k }, (_, i) => (seed >> i) & 1)
        const word = I.linearEncode(code, message)
        expect(I.syndrome(code, word).every((v) => v === 0)).toBe(true)
        const d = I.minimumDistance(code)
        const t = Math.floor((d - 1) / 2)
        const received = word.map((b, i) => (i < t ? 1 - b : b))
        expect(I.syndromeDecode(code, received).codeword).toEqual(word)
        expect(I.syndromeDecode(code, word).message).toEqual(message)
        // Singleton and Hamming bounds hold for the code.
        expect(2 ** code.k).toBeLessThanOrEqual(I.singletonBound(code.n, d))
        expect(2 ** code.k).toBeLessThanOrEqual(I.hammingBound(code.n, d))
      }),
      { numRuns: RUNS },
    )
    expect(() =>
      I.linearCode([
        [1, 1],
        [1, 1],
      ]),
    ).toThrow()
    expect(() => I.linearCode([[1, 2]])).toThrow()
    expect(() => I.linearCode([])).toThrow()
  })

  it('parity and repetition codes', () => {
    const parity = I.parityCode(4)
    expect(I.minimumDistance(parity)).toBe(2)
    expect(I.parityValid(I.parityEncode([1, 1, 1]))).toBe(true)
    expect(I.parityValid(I.parityEncode([1, 1, 1], { odd: true }), { odd: true })).toBe(true)
    expect(I.parityValid([1, 0, 0, 0, 1, 1])).toBe(false)
    const rep = I.repetitionCode(5)
    expect(I.minimumDistance(rep)).toBe(5)
    expect(I.syndromeDecode(rep, [0, 1, 1, 0, 1]).message).toEqual([1])
    expect(I.syndromeDecode(rep, [0, 0, 1, 0, 1]).message).toEqual([0])
    expect(() => I.repetitionCode(1)).toThrow()
    expect(() => I.parityCode(0)).toThrow()
    expect(() => I.linearEncode(rep, [1, 0])).toThrow()
    expect(() => I.syndrome(rep, [1])).toThrow()
  })

  it('coding bounds: perfect codes meet the Hamming bound, and lower bounds stay below upper bounds', () => {
    expect(I.hammingBound(15, 3)).toBe(2 ** 11)
    expect(I.hammingBound(3, 3)).toBe(2)
    expect(I.hammingBound(11, 5, 3)).toBe(729)
    expect(I.plotkinBound(10, 5)).toBe(12)
    expect(I.plotkinBound(8, 4)).toBe(16)
    for (let n = 3; n <= 16; n++)
      for (let d = 1; d <= n; d++) {
        // The guaranteed size never exceeds any upper bound.
        const upper = Math.min(I.hammingBound(n, d), I.singletonBound(n, d), I.plotkinBound(n, d))
        expect(I.gilbertVarshamovBound(n, d)).toBeLessThanOrEqual(upper)
      }
    expect(() => I.hammingBound(3, 4)).toThrow()
    expect(() => I.singletonBound(3, 2, 1)).toThrow()
  })
})
