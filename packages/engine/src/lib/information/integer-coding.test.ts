import { describe, expect, it } from 'vitest'
import * as I from './index.ts'

describe('integer codes', () => {
  it('matches textbook codewords', () => {
    expect(I.eliasEncode([1, 2, 3, 4], 'gamma')).toEqual([1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0])
    expect(I.eliasEncode([1], 'delta')).toEqual([1])
    expect(I.eliasEncode([17], 'delta')).toEqual([0, 0, 1, 0, 1, 0, 0, 0, 1])
    expect(I.eliasEncode([1], 'omega')).toEqual([0])
    expect(I.eliasEncode([2], 'omega')).toEqual([1, 0, 0])
    expect(I.eliasEncode([16], 'omega')).toEqual([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    expect(I.golombEncode([0, 1, 2], 1)).toEqual([0, 1, 0, 1, 1, 0])
    expect(I.golombEncode([0, 1, 2, 3, 4], 5)).toEqual([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1])
  })

  it('rejects invalid input', () => {
    expect(() => I.eliasEncode([0], 'gamma')).toThrow()
    expect(() => I.eliasEncode([1.5], 'delta')).toThrow()
    expect(() => I.eliasDecode([0, 0], 'gamma')).toThrow()
    expect(() => I.eliasDecode([1], 'omega')).toThrow()
    expect(() => I.golombEncode([-1], 3)).toThrow()
    expect(() => I.golombEncode([1], 0)).toThrow()
    expect(() => I.golombDecode([1, 1], 3)).toThrow()
    expect(() => I.golombParameter(1)).toThrow()
    expect(() => I.runLengthDecode({ values: [1], lengths: [] })).toThrow()
    expect(() => I.runLengthDecode({ values: [1], lengths: [-1] })).toThrow()
  })

  it('the optimal Golomb parameter minimises the expected length of a geometric source', () => {
    for (const theta of [0.3, 0.7, 0.85, 0.95]) {
      const cost = (m: number): number => {
        let total = 0
        for (let n = 0; n < 2000; n++) total += (1 - theta) * theta ** n * I.golombEncode([n], m).length
        return total
      }
      const best = I.golombParameter(theta)
      for (const m of [best - 1, best + 1].filter((v) => v >= 1)) expect(cost(best)).toBeLessThanOrEqual(cost(m) + 1e-9)
    }
  })
})
