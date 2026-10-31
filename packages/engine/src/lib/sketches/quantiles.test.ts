/**
 * Tests for the quantile sketches against exact ranks: the Greenwald-Khanna guarantee (every quantile within epsilon n
 * ranks, for any input order), the t-digest's accuracy and its tighter tails, KLL's rank error of about 1 / k that does not
 * grow with n, its unbiasedness, merge, determinism from the key, and monotonicity of the answers.
 */

import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { key, normal } from '../random/index.ts'
import { gkSummary, kllSketch, tDigest } from './index.ts'

function exactRanks(values: readonly number[]): (v: number) => number {
  const sorted = [...values].sort((a, b) => a - b)
  return (v) => {
    let lo = 0
    let hi = sorted.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if ((sorted[mid] ?? 0) <= v) lo = mid + 1
      else hi = mid
    }
    return lo / sorted.length
  }
}

const gaussian = (n: number, seed: number): number[] => Array.from(normal(key(seed), [n]).data)
const grid = Array.from({ length: 199 }, (_, i) => (i + 1) / 200)

describe('gkSummary', () => {
  for (const epsilon of [0.05, 0.01]) {
    for (const order of ['random', 'ascending', 'descending'] as const) {
      it(`answers every quantile within epsilon n ranks (epsilon ${epsilon}, ${order} input)`, () => {
        const n = 10_000
        let xs = gaussian(n, 3)
        if (order === 'ascending') xs = xs.sort((a, b) => a - b)
        if (order === 'descending') xs = xs.sort((a, b) => b - a)
        const gk = gkSummary(epsilon)
        for (const x of xs) gk.add(x)
        const rank = exactRanks(xs)
        for (const q of grid) {
          expect(Math.abs(rank(gk.quantile(q)) - q)).toBeLessThanOrEqual(epsilon + 1 / n)
        }
        expect(gk.tuples().length).toBeLessThan(n / 10)
        expect(gk.quantile(0)).toBe(Math.min(...xs))
        expect(gk.max).toBe(Math.max(...xs))
      })
    }
  }

  it('estimates ranks and the cdf within epsilon', () => {
    const xs = gaussian(10_000, 4)
    const gk = gkSummary(0.02)
    for (const x of xs) gk.add(x)
    const rank = exactRanks(xs)
    for (const x of [-2, -1, 0, 1, 2]) expect(Math.abs(gk.cdf(x) - rank(x))).toBeLessThanOrEqual(0.02)
    expect(gk.cdf(-100)).toBe(0)
    expect(gk.cdf(100)).toBe(1)
  })

  it('validates its input', () => {
    expect(() => gkSummary(0)).toThrow(/epsilon/)
    expect(() => gkSummary(0.1).add(Number.NaN)).toThrow(/finite/)
    expect(gkSummary(0.1).quantile(0.5)).toBeNaN()
  })
})

describe('tDigest', () => {
  const xs = gaussian(50_000, 1)
  const rank = exactRanks(xs)
  const td = tDigest({ compression: 100 })
  for (const x of xs) td.add(x)

  it('stays small and accurate everywhere', () => {
    expect(td.centroids().length).toBeLessThan(150)
    expect(td.totalWeight).toBe(50_000)
    for (const q of grid) expect(Math.abs(rank(td.quantile(q)) - q)).toBeLessThan(0.004)
  })

  it('is far more accurate in the tails than in the middle', () => {
    const tail = Math.max(...[0.0005, 0.001, 0.999, 0.9995].map((q) => Math.abs(rank(td.quantile(q)) - q)))
    expect(tail).toBeLessThan(0.0006)
    // the relative error at q = 0.001 is a small fraction of q itself
    expect(Math.abs(rank(td.quantile(0.001)) - 0.001) / 0.001).toBeLessThan(0.3)
  })

  it('has a cdf that inverts its quantile and matches the exact ranks', () => {
    for (const x of [-3, -1, 0, 1, 2.5]) expect(Math.abs(td.cdf(x) - rank(x))).toBeLessThan(0.004)
    for (const q of [0.1, 0.5, 0.9]) expect(Math.abs(td.cdf(td.quantile(q)) - q)).toBeLessThan(0.002)
  })

  it('merges two digests into one as accurate as a single pass', () => {
    const a = tDigest({ compression: 100 })
    const b = tDigest({ compression: 100 })
    for (const x of xs.slice(0, 25_000)) a.add(x)
    for (const x of xs.slice(25_000)) b.add(x)
    const merged = a.merge(b)
    expect(merged.totalWeight).toBe(50_000)
    for (const q of [0.001, 0.1, 0.5, 0.9, 0.999]) expect(Math.abs(rank(merged.quantile(q)) - q)).toBeLessThan(0.004)
  })

  it('honours weights: a weight-3 value counts three times', () => {
    const w = tDigest({ compression: 100 })
    w.add(1, 3)
    w.add(2, 1)
    expect(w.totalWeight).toBe(4)
    expect(w.quantile(0.5)).toBeLessThanOrEqual(1.5)
    expect(() => w.add(1, 0)).toThrow(/weight/)
    expect(() => tDigest({ compression: 2 })).toThrow(/compression/)
  })

  it('answers monotonically in q (property)', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ min: -1e6, max: 1e6, noNaN: true }), { minLength: 1, maxLength: 300 }), (vs) => {
        const d = tDigest({ compression: 20 })
        for (const v of vs) d.add(v)
        const answers = grid.map((q) => d.quantile(q))
        return answers.every((a, i) => i === 0 || a >= (answers[i - 1] ?? Number.NEGATIVE_INFINITY) - 1e-9)
      }),
      { numRuns: RUNS },
    )
  })
})

describe('kllSketch', () => {
  const k = 200
  const maxError = (n: number, seed: number): number => {
    const xs = gaussian(n, seed + 100)
    const kll = kllSketch({ k, key: key(seed) })
    for (const x of xs) kll.add(x)
    const rank = exactRanks(xs)
    return Math.max(...grid.map((q) => Math.abs(rank(kll.quantile(q)) - q)))
  }

  it('has rank error of about 1 / k, not growing with n', () => {
    for (const n of [3000, 30_000]) {
      for (const seed of [1, 2, 3]) expect(maxError(n, seed)).toBeLessThan(2 / k)
    }
  })

  it('stays within about three k values', () => {
    const kll = kllSketch({ k, key: key(5) })
    for (const x of gaussian(100_000, 6)) kll.add(x)
    expect(kll.size).toBeLessThan(4 * k)
    expect(kll.count).toBe(100_000)
    expect(kll.levels().length).toBeGreaterThan(5)
  })

  it('is unbiased: the signed rank error at the median averages to zero over keys', () => {
    const xs = gaussian(5000, 9)
    const rank = exactRanks(xs)
    const errors = Array.from({ length: 80 }, (_, s) => {
      const kll = kllSketch({ k: 32, key: key(s) })
      for (const x of xs) kll.add(x)
      return kll.cdf(0) - rank(0)
    })
    const mean = errors.reduce((a, b) => a + b, 0) / errors.length
    const sd = Math.sqrt(errors.reduce((a, b) => a + (b - mean) ** 2, 0) / (errors.length - 1))
    expect(Math.abs(mean)).toBeLessThan(4 * (sd / Math.sqrt(errors.length)))
    expect(sd).toBeGreaterThan(0)
  })

  it('preserves the total weight and merges two sketches', () => {
    const xs = gaussian(20_000, 11)
    const a = kllSketch({ k, key: key(1) })
    const b = kllSketch({ k, key: key(2) })
    for (const x of xs.slice(0, 10_000)) a.add(x)
    for (const x of xs.slice(10_000)) b.add(x)
    const merged = a.merge(b)
    const weight = merged.levels().reduce((s, l, h) => s + l.length * 2 ** h, 0)
    expect(weight).toBe(20_000)
    expect(merged.count).toBe(20_000)
    const rank = exactRanks(xs)
    expect(Math.max(...grid.map((q) => Math.abs(rank(merged.quantile(q)) - q)))).toBeLessThan(2 / k)
  })

  it('is deterministic given the key and varies with it', () => {
    const run = (s: number): number[] => {
      const kll = kllSketch({ k: 16, key: key(s) })
      for (let i = 0; i < 2000; i++) kll.add((i * 37) % 1000)
      return kll.levels().flat()
    }
    expect(run(1)).toEqual(run(1))
    expect(run(1)).not.toEqual(run(2))
    expect(() => kllSketch({ k: 4, key: key(0) })).toThrow(/at least 8/)
  })
})
