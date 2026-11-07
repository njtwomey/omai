import { describe, expect, it } from 'vitest'
import { quantileOf, range } from './index.ts'

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
})
