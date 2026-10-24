// Shared helpers of the stats property tests: simulated null p-values and a uniformity check. Node only; not published.
import { expect } from 'vitest'
import { foldIn, type Key, key } from '../lib/random/index.ts'

/** p-values of a test on `m` datasets simulated under its null hypothesis, each from its own key. */
export function nullPValues(m: number, seed: number, pValue: (k: Key) => number): number[] {
  const root = key(seed)
  return Array.from({ length: m }, (_, i) => pValue(foldIn(root, i)))
}

/** The Kolmogorov distance of a sample of p-values from the uniform distribution. */
export function uniformDistance(ps: readonly number[]): number {
  const s = [...ps].sort((a, b) => a - b)
  const n = s.length
  let d = 0
  for (let i = 0; i < n; i++) d = Math.max(d, (i + 1) / n - (s[i] ?? 0), (s[i] ?? 0) - i / n)
  return d
}

/**
 * Assert that p-values look uniform: the rejection rate at 5% within [lo, hi] and the Kolmogorov distance below
 * `maxDistance` (conservative and discrete tests pass `discrete` to check only that they do not over-reject).
 */
export function expectUniformPValues(
  ps: readonly number[],
  options: {
    readonly maxDistance?: number
    readonly discrete?: boolean
    readonly lo?: number
    readonly hi?: number
  } = {},
): void {
  for (const p of ps) {
    expect(p).toBeGreaterThanOrEqual(0)
    expect(p).toBeLessThanOrEqual(1)
  }
  const rate = ps.filter((p) => p < 0.05).length / ps.length
  expect(rate).toBeLessThanOrEqual(options.hi ?? 0.1)
  if (options.discrete === true) return
  expect(rate).toBeGreaterThanOrEqual(options.lo ?? 0.01)
  expect(uniformDistance(ps)).toBeLessThan(options.maxDistance ?? 0.12)
}

/** The fraction of intervals that contain a value. */
export function coverage(intervals: readonly (readonly [number, number])[], value: number): number {
  return intervals.filter(([lo, hi]) => lo <= value && value <= hi).length / intervals.length
}
