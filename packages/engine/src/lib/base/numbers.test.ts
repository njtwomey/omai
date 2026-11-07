import { describe, expect, it } from 'vitest'
import { maxOf, minOf, quantileOf, range, toRows } from './index.ts'

describe('numbers at degenerate arguments', () => {
  it('rejects a zero step and a quantile level outside [0, 1], as numpy and the tensor arange do', () => {
    for (const make of [
      () => range(0, 5, 0),
      () => range(0, 5, Number.NaN),
      () => quantileOf([1, 2], 1.5),
      () => quantileOf([1, 2], -0.1),
      () => quantileOf([1, 2], Number.NaN),
    ])
      expect(make).toThrow(expect.objectContaining({ code: 'value' }))
    expect(range(5, 0, -2)).toEqual([5, 3, 1])
    expect(quantileOf([1, 2, 3, 4], 1)).toBe(4)
    expect(quantileOf([], 0.5)).toBeNaN()
  })

  it('maxOf and minOf take 200,000 values where Math.max(...xs) overflows the stack, and match it on short lists', () => {
    const n = 200_000
    const long = Array.from({ length: n }, (_, i) => ((i * 7919) % n) - 100_000)
    const typed = Float64Array.from(long)
    expect(() => Math.max(...long)).toThrow(RangeError)
    for (const xs of [long, typed, [...long, Number.NaN]]) {
      const hasNaN = xs.length > n
      expect(maxOf(xs)).toBe(hasNaN ? Number.NaN : 99_999)
      expect(minOf(xs)).toBe(hasNaN ? Number.NaN : -100_000)
    }
    for (const xs of [[], [3], [-1, 4, 2], [1, Number.NaN, 2]]) {
      expect(maxOf(xs)).toBe(Math.max(...xs))
      expect(minOf(xs)).toBe(Math.min(...xs))
    }
  })

  it('toRows reshapes a flat row-major array of any array type, reading a missing entry as 0', () => {
    expect(toRows([1, 2, 3, 4, 5, 6], 2, 3)).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ])
    expect(toRows(Float64Array.of(1, 2, 3, 4), 2, 2)).toEqual([
      [1, 2],
      [3, 4],
    ])
    expect(toRows([1, 2, 3], 2, 2)).toEqual([
      [1, 2],
      [3, 0],
    ])
    expect(toRows([], 0, 3)).toEqual([])
    expect(toRows([], 2, 0)).toEqual([[], []])
  })
})
