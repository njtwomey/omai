import { expect } from 'vitest'
import { toFlat, type Value } from '../lib/tensor/index.ts'

const flat = (x: Value | ArrayLike<number>): number[] =>
  typeof (x as Value).shape === 'undefined' ? Array.from(x as ArrayLike<number>) : toFlat(x as Value)

/**
 * Assert that two tensors (or number arrays) have the same length and agree entry by entry within `tolerance`
 * (relative to `1 + |want|`).
 */
export function expectClose(got: Value | ArrayLike<number>, want: Value | ArrayLike<number>, tolerance = 1e-10): void {
  const a = flat(got)
  const b = flat(want)
  expect(a.length).toBe(b.length)
  for (const [i, v] of a.entries()) {
    const w = b[i] as number
    expect(Math.abs(v - w), `entry ${i}: ${v} against ${w}`).toBeLessThanOrEqual(tolerance * (1 + Math.abs(w)))
  }
}
