import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { permutationInvariant } from './audio.ts'

// The best mean over every permutation, by brute force.
function exhaustive(m: readonly (readonly number[])[], sign: number): number {
  const s = m.length
  let best = Number.NEGATIVE_INFINITY
  const used = new Array<boolean>(s).fill(false)
  const visit = (i: number, acc: number): void => {
    if (i === s) {
      best = Math.max(best, sign * acc)
      return
    }
    for (let j = 0; j < s; j++) {
      if (used[j] === true) continue
      used[j] = true
      visit(i + 1, acc + (m[i]?.[j] ?? 0))
      used[j] = false
    }
  }
  visit(0, 0)
  return (sign * best) / s
}

describe('permutation-invariant matching', () => {
  it('finds the best permutation, as exhaustive search does', () => {
    const square = fc.integer({ min: 1, max: 5 }).chain((s) =>
      fc.array(fc.array(fc.double({ min: -10, max: 10, noNaN: true }), { minLength: s, maxLength: s }), {
        minLength: s,
        maxLength: s,
      }),
    )
    fc.assert(
      fc.property(square, fc.constantFrom<'max' | 'min'>('max', 'min'), (m, direction) => {
        // Each signal is one sample holding its row index, so the metric can look the pair up in m.
        const s = m.length
        const idx = Array.from({ length: s }, (_, i) => [i])
        const metric = (a: Float64Array, b: Float64Array): number => m[a[0] ?? 0]?.[b[0] ?? 0] ?? 0
        const r = permutationInvariant(idx, idx, metric, { direction })
        expect(r.value).toBeCloseTo(exhaustive(m, direction === 'min' ? -1 : 1), 10)
        expect([...r.permutation].sort((a, b) => a - b)).toEqual(idx.map(([i]) => i))
        const achieved = r.permutation.reduce((acc, j, i) => acc + (m[i]?.[j] ?? 0), 0) / s
        expect(achieved).toBeCloseTo(r.value, 10)
      }),
    )
  })
})
