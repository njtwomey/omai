import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { hierarchicalSamples, samplePdf, stratifiedSamples } from './sampling.ts'

describe('stratifiedSamples', () => {
  it('draws one sample per bin: increasing, within the planes and in its own bin', () => {
    const t = stratifiedSamples(key(0), 5, 16, 2, 6)
    expect(t.shape).toEqual([5, 16])
    const v = toFlat(t)
    for (let r = 0; r < 5; r++)
      for (let i = 0; i < 16; i++) {
        const x = v[r * 16 + i] as number
        expect(x).toBeGreaterThanOrEqual(2 + (4 * i) / 16)
        expect(x).toBeLessThan(2 + (4 * (i + 1)) / 16)
        if (i > 0) expect(x).toBeGreaterThan(v[r * 16 + i - 1] as number)
      }
  })

  it('takes bin midpoints without perturbation, one draw per key, and per-ray planes', () => {
    expect(toArray(stratifiedSamples(key(0), 2, 2, [0, 10], [2, 14], false))).toEqual([
      [0.5, 1.5],
      [11, 13],
    ])
    expect(toFlat(stratifiedSamples(key(3), 2, 4, 1, 2))).toEqual(toFlat(stratifiedSamples(key(3), 2, 4, 1, 2)))
    expect(toFlat(stratifiedSamples(key(3), 2, 4, 1, 2))).not.toEqual(toFlat(stratifiedSamples(key(4), 2, 4, 1, 2)))
  })

  it('refuses planes that are not increasing and no samples', () => {
    expect(() => stratifiedSamples(key(0), 1, 4, 3, 3)).toThrow(/far must exceed near/)
    expect(() => stratifiedSamples(key(0), 1, 0, 1, 2)).toThrow(/at least one/)
  })
})

describe('samplePdf', () => {
  it('is the quantile function of a piecewise-constant density: evenly spaced quantiles of a known CDF', () => {
    // Bins [0,1), [1,3), [3,4] with masses 0.2, 0.5, 0.3: the CDF is 0.2 x on [0,1], 0.2 + 0.25 (x - 1) on [1,3], and so on.
    const edges = [[0, 1, 3, 4]]
    const weights = [[0.2, 0.5, 0.3]]
    const quantile = (q: number): number => (q <= 0.2 ? q / 0.2 : q <= 0.7 ? 1 + (q - 0.2) / 0.25 : 3 + (q - 0.7) / 0.3)
    const n = 10
    const got = toFlat(samplePdf(edges, weights, n))
    for (let i = 0; i < n; i++) expect(got[i]).toBeCloseTo(quantile((i + 0.5) / n), 12)
  })

  it('has random samples that follow the analytic CDF (Kolmogorov-Smirnov distance)', () => {
    const edges = [[0, 1, 3, 4]]
    const weights = [[0.2, 0.5, 0.3]]
    const cdf = (x: number): number => (x <= 1 ? 0.2 * x : x <= 3 ? 0.2 + 0.25 * (x - 1) : 0.7 + 0.3 * (x - 3))
    const n = 2000
    const x = toFlat(samplePdf(edges, weights, n, key(5)))
    expect(x.every((v, i) => i === 0 || v >= (x[i - 1] as number))).toBe(true)
    let d = 0
    for (let i = 0; i < n; i++) d = Math.max(d, Math.abs(cdf(x[i] as number) - (i + 0.5) / n))
    // The 5% critical value of the KS statistic is 1.36 / sqrt(n) = 0.030.
    expect(d).toBeLessThan(0.03)
  })

  it('concentrates on a peak, works row by row, and samples uniformly when a row has no weight', () => {
    const edges = [
      [0, 1, 2, 3, 4, 5],
      [0, 1, 2, 3, 4, 5],
    ]
    const weights = [
      [0, 0, 100, 0, 0],
      [0, 0, 0, 0, 0],
    ]
    const x = toFlat(samplePdf(edges, weights, 8, key(1)))
    expect(x.slice(0, 8).every((v) => v >= 2 && v <= 3)).toBe(true)
    expect(Math.max(...x.slice(8)) - Math.min(...x.slice(8))).toBeGreaterThan(2.5)
  })

  it('checks shapes and signs', () => {
    expect(() => samplePdf([[0, 1, 2]], [[1, 1, 1]], 2)).toThrow(/edges need 2 weights/)
    expect(() => samplePdf([[0, 1, 2]], [[1, -1]], 2)).toThrow(/non-negative/)
  })
})

describe('hierarchicalSamples', () => {
  it('merges fine samples with the coarse ones in order, concentrated where the coarse weights were', () => {
    const coarse = [Array.from({ length: 16 }, (_, i) => 2 + i * 0.25)]
    const weights = [coarse[0]?.map((t) => Math.exp(-(((t as number) - 4) ** 2) / 0.05)) as number[]]
    const merged = toFlat(hierarchicalSamples(coarse, weights, 32, key(0)))
    expect(merged.length).toBe(48)
    expect(merged.every((v, i) => i === 0 || v >= (merged[i - 1] as number))).toBe(true)
    for (const t of coarse[0] as number[]) expect(merged).toContain(t)
    const fine = merged.filter((v) => !(coarse[0] as number[]).includes(v))
    expect(fine.filter((v) => Math.abs(v - 4) < 0.5).length).toBeGreaterThan(24)
  })

  it('refuses mismatched weights', () => {
    expect(() => hierarchicalSamples([[1, 2, 3]], [[1, 1]], 2)).toThrow(/same shape/)
  })
})
