import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import * as T from './index.ts'

const n = 256
const noise = Array.from(toFlat(normal(key(4), [n])))
const sine = Array.from({ length: n }, (_, t) => Math.sin((2 * Math.PI * t) / 16))
const walk: number[] = []
for (const v of noise) walk.push((walk.at(-1) ?? 0) + v)
const trend = Array.from({ length: n }, (_, t) => 0.05 * t + 0.1 * (noise[t] ?? 0))

describe('permutation entropy', () => {
  it('is 0 for a monotone series, near 1 for noise, and by hand for a short pattern', () => {
    expect(T.permutationEntropy([1, 2, 3, 4, 5, 6], 3)).toBe(0)
    expect(T.permutationEntropy(noise, 3)).toBeGreaterThan(0.95)
    // Windows (1,3,2) and (3,2,4): two different patterns, one each: H = ln 2, normalised by ln 6.
    expect(T.permutationEntropy([1, 3, 2, 4], 3)).toBeCloseTo(Math.log(2) / Math.log(6), 12)
  })
})

describe('features', () => {
  const f = (x: number[]) => T.timeSeriesFeatures(x, { set: 'all' }).byName
  it('returns a labelled vector with stable, unique names', () => {
    const v = T.timeSeriesFeatures(noise)
    expect(new Set(v.names).size).toBe(v.names.length)
    expect(v.values).toHaveLength(v.names.length)
    expect(v.names.length).toBeGreaterThan(40)
    expect(v.values.every(Number.isFinite)).toBe(true)
    const m = T.featureMatrix([noise, sine])
    expect(m.rows.every((r) => r.length === m.names.length)).toBe(true)
  })
  it('tells a periodic series from noise and from a random walk', () => {
    const [a, b, c] = [f(sine), f(noise), f(walk)]
    expect(a['dominant_frequency']).toBeCloseTo(1 / 16, 2)
    expect(a['sample_entropy'] ?? 9).toBeLessThan((b['sample_entropy'] ?? 0) / 3)
    expect(a['spectral_entropy'] ?? 9).toBeLessThan(0.3)
    expect(b['spectral_entropy'] ?? 0).toBeGreaterThan(0.85)
    expect(Math.abs(b['acf_lag1'] ?? 9)).toBeLessThan(0.2)
    expect(c['acf_lag1'] ?? 0).toBeGreaterThan(0.9)
    expect(c['spectral_low_power_ratio'] ?? 0).toBeGreaterThan(0.9)
    expect(a['mean_crossings'] ?? 0).toBeCloseTo(2 / 16, 1)
  })
  it('measures trend and the Hurst exponent where they are planted', () => {
    const t = f(trend)
    expect(t['trend_slope']).toBeCloseTo(0.05, 3)
    expect(t['trend_r2'] ?? 0).toBeGreaterThan(0.98)
    expect(f(noise)['trend_r2'] ?? 1).toBeLessThan(0.05)
    expect(f(walk)['hurst_rs'] ?? 0).toBeGreaterThan(0.7)
    expect(Math.abs((f(noise)['hurst_rs'] ?? 0) - 0.55)).toBeLessThan(0.2)
  })
  it('matches closed forms on a short series', () => {
    const x = [1, 3, 2, 5, 4, 6, 5, 8, 7, 9, 8, 10, 9, 12, 11, 13]
    const v = T.timeSeriesFeatures(x).byName
    expect(v['mean_change']).toBeCloseTo(12 / 15, 12)
    expect(v['absolute_energy']).toBe(x.reduce((a, b) => a + b * b, 0))
    expect(v['cid_ce']).toBeCloseTo(Math.sqrt(x.slice(1).reduce((a, b, k) => a + (b - (x[k] ?? 0)) ** 2, 0)), 12)
    expect(v['count_above_mean']).toBeCloseTo(8 / 16, 12)
  })
  it('rejects series that are too short', () => {
    expect(() => T.timeSeriesFeatures([1, 2, 3])).toThrow()
  })
})
