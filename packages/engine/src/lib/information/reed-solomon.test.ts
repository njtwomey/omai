import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import * as I from './index.ts'

describe('Reed-Solomon codes', () => {
  it('field arithmetic: every non-zero element has an inverse, and alpha has full order', () => {
    for (const m of [3, 4, 8]) {
      const f = I.galoisField(m)
      for (let a = 1; a < f.size; a++) {
        expect(f.mul(a, f.inverse(a))).toBe(1)
        expect(f.div(f.mul(a, 5 % f.size || 1), a)).toBe(5 % f.size || 1)
      }
      expect(f.pow(2, f.size - 1)).toBe(1)
      expect(f.pow(0, 0)).toBe(1)
      expect(f.pow(0, 3)).toBe(0)
      expect(f.add(3, 3)).toBe(0)
    }
    expect(() => I.galoisField(8, 0x11b)).toThrow()
    expect(() => I.galoisField(1)).toThrow()
    expect(() => I.galoisField(8).div(1, 0)).toThrow()
  })

  it('corrects any errors and erasures with 2e + s <= nsym (property)', () => {
    const code = I.reedSolomon(8)
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: 255 }), { minLength: 1, maxLength: 40 }),
        fc.integer({ min: 0, max: 4 }),
        fc.nat(),
        (message, errors, seed) => {
          const word = I.rsEncode(code, message)
          const erasures = 8 - 2 * errors
          const n = word.length
          const positions: number[] = []
          for (let i = 0; positions.length < Math.min(n, errors + erasures); i++) {
            const p = (seed + i * 7919) % n
            if (!positions.includes(p)) positions.push(p)
          }
          const corrupt = [...word]
          positions.forEach((p, i) => {
            corrupt[p] = (corrupt[p] ?? 0) ^ (1 + ((seed + i) % 255))
          })
          const result = I.rsDecode(code, corrupt, { erasures: positions.slice(errors) })
          expect(result.codeword).toEqual(word)
          expect(result.message).toEqual(message)
        },
      ),
      { numRuns: RUNS },
    )
  })

  it('reports uncorrectable words and bad input', () => {
    const code = I.reedSolomon(4)
    const word = I.rsEncode(code, [1, 2, 3, 4, 5, 6])
    const corrupt = word.map((v, i) => (i < 3 ? v ^ 0x55 : v))
    expect(() => I.rsDecode(code, corrupt)).toThrow()
    expect(I.rsDecode(code, word).errata).toEqual([])
    expect(I.rsDecode(code, word, { erasures: [0] }).codeword).toEqual(word)
    expect(() => I.rsDecode(code, word, { erasures: [0, 1, 2, 3, 4] })).toThrow()
    expect(() => I.rsDecode(code, word, { erasures: [99] })).toThrow()
    expect(() => I.rsDecode(code, [1, 2])).toThrow()
    expect(() => I.rsEncode(code, new Array<number>(252).fill(1))).toThrow()
    expect(() => I.rsEncode(code, [256])).toThrow()
    expect(() => I.reedSolomon(0)).toThrow()
  })
})
