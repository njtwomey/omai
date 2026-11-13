import { describe, expect, it } from 'vitest'
import { largestOf } from './_restart.ts'

describe('the largest residual', () => {
  it('is a loop: a million of them do not overflow the call stack, and a NaN is a NaN', () => {
    const many = new Float64Array(1_000_000)
    many[777_777] = 3
    expect(largestOf(many, 0)).toBe(3)
    expect(largestOf([], 0)).toBe(0)
    expect(largestOf([-2, -1])).toBe(-1)
    expect(largestOf([1, Number.NaN, 2], 0)).toBeNaN()
  })
})
