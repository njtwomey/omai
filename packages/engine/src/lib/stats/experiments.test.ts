import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import {
  clusterRandomisation,
  clusterRandomisedTest,
  cuped,
  deltaMethodRatio,
  postStratification,
  postStratify,
  regressionAdjustment,
  switchbackAnalysis,
  switchbackDesign,
} from './experiments.ts'

describe('online experimentation estimators', () => {
  it('reduces variance and recovers the true ATE with CUPED', () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    const t = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    const y = x.map((xi, i) => 3 * xi + (t[i] === 1 ? 2 : 0))
    const res = cuped(y, x, { treatment: t })
    // Pooled theta is Cov(y, x) / Var(x); the ATE is the adjusted difference in arm means.
    const mean = (v: number[]): number => v.reduce((a, b) => a + b, 0) / v.length
    const xBar = mean(x)
    const yBar = mean(y)
    let sxy = 0
    let sxx = 0
    for (let i = 0; i < x.length; i++) {
      sxy += ((x[i] ?? 0) - xBar) * ((y[i] ?? 0) - yBar)
      sxx += ((x[i] ?? 0) - xBar) ** 2
    }
    expect(res.theta).toBeCloseTo(sxy / sxx, 12)
    const arm = (v: number[], a: number): number[] => v.filter((_, i) => t[i] === a)
    const expected = mean(arm(y, 1)) - mean(arm(y, 0)) - res.theta * (mean(arm(x, 1)) - mean(arm(x, 0)))
    expect(res.ate?.value).toBeCloseTo(expected, 10)
    // With the true slope supplied the effect is recovered exactly and the adjusted outcomes are noise-free.
    const exact = cuped(y, x, { treatment: t, theta: 3 })
    expect(exact.ate?.value).toBeCloseTo(2, 10)
    expect(res.varianceReduction).toBeGreaterThan(0.95)
    expect(res.ate?.standardError ?? 1).toBeLessThan(res.unadjustedAte?.standardError ?? 0)
  })

  it('eliminates between-stratum variance via postStratify', () => {
    const y = [10, 10.1, 12, 12.1, 100, 100.1, 102, 102.1]
    const t = [0, 0, 1, 1, 0, 0, 1, 1]
    const s = [1, 1, 1, 1, 2, 2, 2, 2]
    const res = postStratify(y, t, s)
    expect(res.ate.value).toBeCloseTo(2, 10)
    expect(res.ate.standardError ?? 1).toBeLessThan(0.1)
    expect(res.unadjustedSe).toBeGreaterThan(20)
    expect(res.varianceReduction).toBeGreaterThan(0.999)
    expect(postStratification(y, t, s).ate.value).toBeCloseTo(2, 10)
  })

  it('estimates ATE with Lin demeaned covariate-interaction regression adjustment', () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    const t = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    // Heterogeneous slope: control slope = 2, treated slope = 4 around mean(x) = 6.5, ATE at mean(x) = 3.0
    const xBar = 6.5
    const y = x.map((xi, i) => 5 + 2 * (xi - xBar) + (t[i] === 1 ? 3 + 2 * (xi - xBar) : 0))
    const res = regressionAdjustment(y, t, x)
    expect(res.ate.value).toBeCloseTo(3, 10)
    expect(res.intercept).toBeCloseTo(5, 10)
    expect(res.controlSlopes.data[0]).toBeCloseTo(2, 10)
    expect(res.interactionSlopes.data[0]).toBeCloseTo(2, 10)
    expect(res.rSquared).toBeCloseTo(1, 10)
  })

  it('computes single-sample and two-sample ratio metrics via the delta method', () => {
    const num = [2, 5, 3, 8, 12, 18, 9, 21]
    const den = [10, 20, 10, 20, 30, 40, 20, 40]
    const treat = [0, 0, 0, 0, 1, 1, 1, 1]
    const single = deltaMethodRatio(num, den)
    expect(single.ratio).toBeCloseTo(78 / 190, 12)
    // Mean of linearized scores equals the ratio
    const linMean = Array.from(single.linearized.data).reduce((a, b) => a + b, 0) / num.length
    expect(linMean).toBeCloseTo(single.ratio, 12)
    // The delta-method variance is the variance of the linearised scores over n.
    const linVar = Array.from(single.linearized.data).reduce((a, b) => a + (b - linMean) ** 2, 0) / (num.length - 1)
    expect(single.variance).toBeCloseTo(linVar / num.length, 12)

    const two = deltaMethodRatio(num, den, { treatment: treat })
    // Control ratio = 18 / 60 = 0.3; Treatment ratio = 60 / 130 = 6/13 ≈ 0.461538
    expect(two.controlRatio?.value).toBeCloseTo(0.3, 12)
    expect(two.treatmentRatio?.value).toBeCloseTo(60 / 130, 12)
    expect(two.estimate.value).toBeCloseTo(60 / 130 - 0.3, 12)
    expect(two.test.pValue).toBeLessThan(0.05)
  })

  it('computes ICC, design effect, and cluster-robust standard errors in clusterRandomisedTest', () => {
    // 6 clusters (3 control, 3 treated), 4 units per cluster with high within-cluster correlation
    const y = [
      1.0, 1.1, 0.9, 1.0, 2.0, 2.1, 1.9, 2.0, 1.5, 1.6, 1.4, 1.5, 5.0, 5.1, 4.9, 5.0, 6.0, 6.1, 5.9, 6.0, 5.5, 5.6, 5.4,
      5.5,
    ]
    const t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    const c = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6]
    const res = clusterRandomisedTest(y, t, c)
    expect(res.ate.value).toBeCloseTo(4, 10)
    expect(res.nClusters).toBe(6)
    expect(res.meanClusterSize).toBe(4)
    expect(res.icc).toBeGreaterThan(0.9)
    expect(res.designEffect).toBeGreaterThan(3.5)
    expect(res.clusterRobustSe).toBeGreaterThan(res.naiveSe)
    expect(clusterRandomisation(y, t, c).ate.value).toBeCloseTo(4, 10)
  })

  it('analyzes switchback experiments with Newey-West HAC SE and randomization inference', () => {
    const y = [10, 10.2, 13, 12.8, 10.1, 9.9, 13.1, 12.9, 9.9, 10.1, 12.9, 13.1, 10.0, 10.0, 13.0, 13.0]
    const t = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    const p = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    const res = switchbackAnalysis(y, t, p, { key: key(42), nPermutations: 200 })
    expect(res.nPeriods).toBe(8)
    expect(res.ate.value).toBeCloseTo(2.95, 10)
    expect(res.pValue).toBeLessThan(0.001)
    expect(res.permutationPValue ?? 1).toBeLessThan(0.05)
    expect(switchbackDesign(y, t, p).ate.value).toBeCloseTo(2.95, 10)
  })
})
