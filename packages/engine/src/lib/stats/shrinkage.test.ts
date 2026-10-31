import { describe, expect, it } from 'vitest'
import { empiricalBayesNormalMeans, jamesStein, positivePartJamesStein } from './shrinkage.ts'

describe('James-Stein and empirical Bayes shrinkage', () => {
  it('computes classic and positive-part James-Stein towards zero, grand mean, and vector targets', () => {
    const res = jamesStein([2, 2, 2, 2], { sigma: 1, target: 'zero' })
    expect(res.dfShrinkage).toBe(2)
    expect(res.squaredNorm).toBeCloseTo(16, 12)
    expect(res.shrinkageFactor).toBeCloseTo(0.125, 12)
    expect(res.multiplier).toBeCloseTo(0.875, 12)
    expect(Array.from(res.estimates.data)).toEqual([1.75, 1.75, 1.75, 1.75])
    expect(res.mleRisk).toBe(4)
    expect(res.sureRiskEstimate).toBeCloseTo(4 - 4 / 16, 12)
    expect(res.priorVarianceEstimate).toBeCloseTo(7, 12)

    // Small norm triggers positive-part truncation at 0 whereas classic flips sign
    const small = [0.2, -0.2, 0.2]
    const pos = positivePartJamesStein(small, { sigma: 1 })
    const classic = jamesStein(small, { sigma: 1, positivePart: false })
    expect(pos.multiplier).toBe(0)
    expect(Array.from(pos.estimates.data)).toEqual([0, 0, 0])
    expect(classic.multiplier).toBeLessThan(0)
    expect((classic.estimates.data[0] ?? 0) * (small[0] ?? 0)).toBeLessThan(0)

    // Shrinking towards the grand mean requires p >= 4 and uses df = p - 3
    const gm = jamesStein([1, 2, 3, 6], { sigma: 1, target: 'grand-mean' })
    expect(gm.dfShrinkage).toBe(1)
    expect(Array.from(gm.target.data)).toEqual([3, 3, 3, 3])
    expect(() => jamesStein([1, 2], { target: 'zero' })).toThrow(/p >= 3/)
    expect(() => jamesStein([1, 2, 3], { target: 'grand-mean' })).toThrow(/p >= 4/)
  })

  it('computes empirical Bayes normal means for homoscedastic and heteroscedastic observations', () => {
    const eb = empiricalBayesNormalMeans([-3, -1, 1, 3], { sigma: 1, priorMean: 0 })
    // ss = 9 + 1 + 1 + 9 = 20, df = 4 - 2 = 2 -> tau^2 = 20 / 2 - 1 = 9
    expect(eb.priorVariance).toBeCloseTo(9, 12)
    expect(eb.shrinkageFactors.data[0]).toBeCloseTo(0.1, 12)
    expect(eb.posteriorMeans.data[3]).toBeCloseTo(2.7, 12)
    expect(eb.posteriorVariances.data[0]).toBeCloseTo(0.9, 12)
    expect(eb.intervals).toHaveLength(4)

    // Heteroscedastic: noisier observations shrink more strongly towards the prior mean
    const het = empiricalBayesNormalMeans([-4, -1, 1, 4], {
      variances: [4, 0.5, 0.5, 4],
    })
    expect(het.shrinkageFactors.data[0] ?? 0).toBeGreaterThan(het.shrinkageFactors.data[1] ?? 1)
    expect(Math.abs(het.posteriorMeans.data[0] ?? 0)).toBeLessThan(4)
  })
})
