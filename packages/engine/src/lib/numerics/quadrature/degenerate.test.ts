import { describe, expect, it } from 'vitest'
import { div, mul, sin, type Value } from '../../tensor/index.ts'
import * as Q from './index.ts'

describe('adaptiveSimpson on integrands and limits that cannot be integrated', () => {
  // Each of these refined every panel to the depth limit (2^50 evaluations) before, so a run that returns is the test.
  it.each([
    ['a NaN integrand', (x: Value) => mul(x, Number.NaN), 0, 1],
    ['an integrand that is infinite at the lower limit (1 / x)', (x: Value) => div(1, x), 0, 1],
    ['an infinite limit', (x: Value) => x, 0, Number.POSITIVE_INFINITY],
    ['a NaN limit', (x: Value) => x, Number.NaN, 1],
  ])('throws a value error for %s', (_what, f, a, b) => {
    expect(() => Q.adaptiveSimpson(f, a, b)).toThrow(expect.objectContaining({ code: 'value' }))
  })

  it('stops subdividing after a million evaluations when no panel meets the tolerance', () => {
    // sin(1000 x) with tol 0: no panel ever satisfies |delta| <= 0, which was 2^50 evaluations.
    const r = Q.adaptiveSimpson((x) => sin(mul(x, 1000)), 0, 1, { tol: 0 })
    expect(Number.isFinite(r.value)).toBe(true)
    expect(r.evaluations).toBeGreaterThanOrEqual(1_000_000)
    expect(r.evaluations).toBeLessThan(1_000_500)
  })
})
