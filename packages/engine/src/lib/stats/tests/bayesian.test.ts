import { describe, expect, it } from 'vitest'
import { gamma, key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

describe('Bayesian tests', () => {
  it('Bayes factors grow with the evidence and favour the null for null data', () => {
    expect(T.bayesFactorTTest(0.1, 50).statistic).toBeLessThan(1 / 3)
    expect(T.bayesFactorTTest(5, 50).statistic).toBeGreaterThan(100)
    expect(T.bayesFactorPearson(0.02, 100).statistic).toBeLessThan(1 / 3)
    const two = T.bayesFactorPearson(0.4, 30).statistic
    const up = T.bayesFactorPearson(0.4, 30, { alternative: 'greater' }).statistic
    const down = T.bayesFactorPearson(0.4, 30, { alternative: 'less' }).statistic
    // The two-sided factor averages the one-sided ones.
    expect((up + down) / 2).toBeCloseTo(two, 8)
    expect(up).toBeGreaterThan(two)
    expect(T.bayesFactorBinomial(10, 20).statistic).toBeLessThan(1)
    expect(Number.isNaN(T.bayesFactorBinomial(10, 20).pValue)).toBe(true)
  })

  it('the A/B probability is symmetric and its losses are non-negative', () => {
    const ab = T.bayesianAbTest(30, 100, 20, 100)
    const ba = T.bayesianAbTest(20, 100, 30, 100)
    expect(ab.statistic + ba.statistic).toBeCloseTo(1, 8)
    expect(Number(ab.assumptions?.['expectedLossChoosingB'])).toBeGreaterThan(0)
    expect(ab.estimate?.interval?.[0]).toBeLessThan(ab.estimate?.value ?? 0)
  })

  it('the HDI is the narrowest interval of its mass and drives the ROPE decision', () => {
    const draws = toFlat(gamma(key(1), [4000], 2))
    const [lo, hi] = T.highestDensityInterval(draws, 0.9)
    const inside = draws.filter((v) => v >= lo && v <= hi).length / draws.length
    expect(inside).toBeGreaterThanOrEqual(0.9 - 1e-3)
    expect(T.rope(draws, { low: -100, high: 100 }).assumptions?.['decision']).toBe('accept')
    expect(T.rope(draws, { low: 50, high: 60 }).assumptions?.['decision']).toBe('reject')
    expect(T.rope(draws, { low: 0, high: 2 }).assumptions?.['decision']).toBe('undecided')
  })
})
