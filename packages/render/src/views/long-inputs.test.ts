import { describe, expect, it } from 'vitest'
import { limitsOf, spanOf, wantsLog } from './_view.ts'

// Axis decisions over a long run (a metric logged at every step of 200,000): Math.max(...values) throws RangeError there.
describe('views at 200,000 values', () => {
  it('decides the axis of a long series without spreading it into Math.max', () => {
    const n = 200_000
    const values = Array.from({ length: n }, (_, i) => 1 + ((i * 7919) % n))
    expect(wantsLog(values)).toBe(true)
    expect(wantsLog(values.map((v) => 1 + v / n))).toBe(false)
    expect(limitsOf(values, false)).toEqual([1, n])
    expect(limitsOf(values, true)).toEqual([1, 10 ** 6])
    expect(spanOf(values)).toEqual([1, n])
  })
})
