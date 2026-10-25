import { describe, expect, it } from 'vitest'
import { demographicParityDifference, equalizedOddsDifference } from './group-metrics.ts'
import {
  bootstrapDisparity,
  groupRateIntervals,
  minimumDetectableDifference,
  permutationTestDisparity,
  smallGroupWarnings,
} from './inference.ts'

// Two groups of 60 with selection rates 0.6 and 0.3.
const groups = Array.from({ length: 120 }, (_, i) => (i < 60 ? 'a' : 'b'))
const yPred = groups.map((g, i) => (g === 'a' ? (i % 5 < 3 ? 1 : 0) : i % 10 < 3 ? 1 : 0))
const yTrue = groups.map((_, i) => (i % 3 === 0 ? 1 : 0))
const dp = (t: number[], p: number[], g: string[]): number => demographicParityDifference(t, p, g).estimate.value

describe('uncertainty of fairness metrics', () => {
  it('bootstraps an interval around the gap, reproducibly, narrowing with more rows', () => {
    const r = bootstrapDisparity(yTrue, yPred, groups, dp, { seed: 3, nResamples: 400 })
    const [lo, hi] = r.estimate.interval ?? [0, 0]
    expect(r.estimate.value).toBeCloseTo(0.3)
    expect(lo).toBeLessThan(0.3)
    expect(hi).toBeGreaterThan(0.3)
    expect(r.distribution).toHaveLength(400)
    expect(bootstrapDisparity(yTrue, yPred, groups, dp, { seed: 3, nResamples: 400 }).estimate.interval).toEqual(
      r.estimate.interval,
    )
    const big = <T>(x: readonly T[]): T[] => [...x, ...x, ...x, ...x]
    const wide = bootstrapDisparity(big(yTrue), big(yPred), big(groups), dp, { seed: 3, nResamples: 400 })
    const width = (e: typeof r) => (e.estimate.interval?.[1] ?? 0) - (e.estimate.interval?.[0] ?? 0)
    expect(width(wide)).toBeLessThan(width(r))
  })

  it('a permutation test finds a real gap and not a null one', () => {
    expect(permutationTestDisparity(yTrue, yPred, groups, dp, { seed: 1, nResamples: 199 }).pValue).toBeLessThan(0.02)
    const fair = groups.map((_, i) => (i % 2 === 0 ? 1 : 0))
    const r = permutationTestDisparity(yTrue, fair, groups, (t, p, g) => equalizedOddsDifference(t, p, g), {
      seed: 1,
      nResamples: 199,
    })
    expect(r.pValue).toBeGreaterThan(0.05)
    expect(r.distribution).toHaveLength(199)
  })

  it('gives each group Wilson intervals, with no interval for an empty denominator', () => {
    const r = groupRateIntervals([0, 0, 1, 1], [0, 0, 1, 0], ['a', 'a', 'b', 'b'])
    expect(r.map((x) => x.group)).toEqual(['a', 'b'])
    expect(r[0]?.rates.truePositiveRate.value).toBeNaN()
    const [lo, hi] = r[1]?.rates.selectionRate.interval ?? [0, 0]
    expect(lo).toBeLessThan(0.5)
    expect(hi).toBeGreaterThan(0.5)
  })

  it('needs larger gaps to detect between smaller groups, and warns of them', () => {
    const small = minimumDetectableDifference({ n1: 50, n2: 50 })
    const large = minimumDetectableDifference({ n1: 500, n2: 500 })
    expect(small.difference).toBeGreaterThan(large.difference)
    expect(large.difference).toBeGreaterThan(0)
    expect(() => minimumDetectableDifference({ n1: 0, n2: 3 })).toThrow(/rows/)
    expect(() => minimumDetectableDifference({ n1: 10, n2: 3, baseRate: 1 })).toThrow(/baseRate/)
    const w = smallGroupWarnings(yTrue, groups, { minSize: 61 })
    expect(w).toHaveLength(2)
    expect(w[0]?.message).toContain('fewer than 61')
    expect(smallGroupWarnings(yTrue, groups)).toEqual([])
  })
})
