import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { eye, item, matmul, norm, sub, toFlat, transpose } from '../../tensor/index.ts'
import { det } from './index.ts'
import { haarOrthogonal } from './randomized.ts'

const seed = fc.integer({ min: 0, max: 2 ** 31 })

describe('haarOrthogonal', () => {
  it('is orthogonal for every seed and size', () => {
    fc.assert(
      fc.property(seed, fc.integer({ min: 1, max: 12 }), (s, n) => {
        const q = haarOrthogonal(key(s), n)
        expect(q.shape).toEqual([n, n])
        expect(item(norm(sub(matmul(transpose(q), q), eye(n))))).toBeLessThan(1e-12)
      }),
      { numRuns: 40 },
    )
  })

  it('gives an orthonormal frame of the first columns', () => {
    fc.assert(
      fc.property(seed, fc.integer({ min: 2, max: 15 }), (s, n) => {
        const k = 1 + (s % n)
        const q = haarOrthogonal(key(s), n, k)
        expect(q.shape).toEqual([n, k])
        expect(item(norm(sub(matmul(transpose(q), q), eye(k))))).toBeLessThan(1e-12)
      }),
      { numRuns: 40 },
    )
  })

  it('is reproducible from the key', () => {
    expect(toFlat(haarOrthogonal(key(3), 5))).toEqual(toFlat(haarOrthogonal(key(3), 5)))
    expect(toFlat(haarOrthogonal(key(3), 5))).not.toEqual(toFlat(haarOrthogonal(key(4), 5)))
  })

  it('has both determinant signs and a squared entry of mean 1 / n, as the Haar law has', () => {
    const n = 4
    const draws = 600
    let plus = 0
    let entry = 0
    for (let s = 0; s < draws; s++) {
      const q = haarOrthogonal(key(s), n)
      if (item(det(q)) > 0) plus++
      entry += (toFlat(q)[0] ?? 0) ** 2
    }
    expect(Math.abs(plus / draws - 0.5)).toBeLessThan(0.08)
    expect(Math.abs(entry / draws - 1 / n)).toBeLessThan(0.03)
  })

  it('rejects a bad dimension', () => {
    expect(() => haarOrthogonal(key(0), 0)).toThrow(/positive integer/)
    expect(() => haarOrthogonal(key(0), 3, 4)).toThrow(/columns/)
  })
})
