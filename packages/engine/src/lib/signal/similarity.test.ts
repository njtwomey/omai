import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import * as S from './index.ts'

const series = (n: number): fc.Arbitrary<number[]> =>
  fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: n, maxLength: n })

/** DTW by brute recursion over all monotone paths (symmetric1), for tiny series, with a cost and a band. */
function brute(
  a: readonly number[],
  b: readonly number[],
  cost: (u: number, v: number) => number = (u, v) => Math.abs(u - v),
  band = Number.POSITIVE_INFINITY,
): number {
  const memo = new Map<string, number>()
  const go = (i: number, j: number): number => {
    const key = `${i},${j}`
    const hit = memo.get(key)
    if (hit !== undefined) return hit
    const d = Math.abs(i - j) > band ? Number.POSITIVE_INFINITY : cost(a[i] ?? 0, b[j] ?? 0)
    let v: number
    if (i === 0 && j === 0) v = d
    else {
      const opts: number[] = []
      if (i > 0 && j > 0) opts.push(go(i - 1, j - 1))
      if (i > 0) opts.push(go(i - 1, j))
      if (j > 0) opts.push(go(i, j - 1))
      v = d + opts.reduce((m, o) => Math.min(m, o), Number.POSITIVE_INFINITY)
    }
    memo.set(key, v)
    return v
  }
  return go(a.length - 1, b.length - 1)
}

describe('similarity', () => {
  it('matches brute-force DTW and is bounded below by LB_Kim and LB_Keogh', () => {
    fc.assert(
      fc.property(series(6), series(6), (a, b) => {
        const d = S.dtw(a, b, { stepPattern: 'symmetric1' })
        expect(d.distance).toBeCloseTo(brute(a, b), 10)
        expect(S.lbKim(a, b)).toBeLessThanOrEqual(d.distance + 1e-12)
        expect(S.dtw(a, b, { stepPattern: 'symmetric1', window: 2 }).distance).toBeCloseTo(
          brute(a, b, undefined, 2),
          10,
        )
        // LB_Keogh bounds the banded DTW with squared costs (its square root).
        const squared = brute(a, b, (u, v) => (u - v) ** 2, 2)
        expect(S.lbKeogh(a, b, 2)).toBeLessThanOrEqual(Math.sqrt(squared) + 1e-9)
      }),
      { numRuns: RUNS },
    )
  })

  it('computes PAA, SAX and lags', () => {
    expect(S.paa([1, 2, 3, 4, 5], 2)).toEqual([1.8, 4.2])
    const s = S.sax([0, 0, 0, 0, 5, 5, 5, 5], 2, 2)
    expect(s.word).toBe('ab')
    expect(S.saxMindist([0, 0], [2, 2], S.sax([1, 2, 3], 3, 3).breakpoints, 8)).toBeGreaterThan(0)
    expect(S.crossCorrelationLag([1, 2, 1, 0, 0, 0], [0, 0, 1, 2, 1, 0])).toBe(-2)
    expect(() => S.paa([1, 2], 3)).toThrow()
  })
})
