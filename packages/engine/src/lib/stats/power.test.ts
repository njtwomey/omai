import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  powerAnova,
  powerChisquare,
  powerProportions,
  powerTTest,
  sampleSizeAnova,
  sampleSizeChisquare,
  sampleSizeProportions,
  sampleSizeTTest,
  simulatePower,
} from './power.ts'
import { ttest1samp } from './tests/index.ts'

describe('power analysis', () => {
  it('power grows with the effect and the sample, and the sample size inverts it', () => {
    fc.assert(
      fc.property(fc.double({ min: 0.1, max: 1.2, noNaN: true }), fc.integer({ min: 5, max: 200 }), (d, n) => {
        const p = powerTTest({ effectSize: d, nobs: n, kind: 'two-sample' })
        expect(powerTTest({ effectSize: d * 1.2, nobs: n, kind: 'two-sample' })).toBeGreaterThanOrEqual(p)
        expect(powerTTest({ effectSize: d, nobs: n + 5, kind: 'two-sample' })).toBeGreaterThanOrEqual(p)
        if (p > 0.1 && p < 0.99)
          expect(sampleSizeTTest({ effectSize: d, power: p, kind: 'two-sample' })).toBeCloseTo(n, 4)
      }),
      { numRuns: 30 },
    )
  })

  it('sample sizes invert the power of proportion, ANOVA and chi-square tests', () => {
    const pp = powerProportions({ effectSize: 0.3, nobs: 80 })
    expect(sampleSizeProportions({ effectSize: 0.3, power: pp })).toBeCloseTo(80, 4)
    const pa = powerAnova({ effectSize: 0.3, nobs: 60, kGroups: 4 })
    expect(sampleSizeAnova({ effectSize: 0.3, power: pa, kGroups: 4 })).toBeCloseTo(60, 4)
    const pc = powerChisquare({ effectSize: 0.3, nobs: 70, nBins: 3 })
    expect(sampleSizeChisquare({ effectSize: 0.3, power: pc, nBins: 3 })).toBeCloseTo(70, 4)
  })

  it('has the size of the test at zero effect and rejects bad alphas', () => {
    expect(powerTTest({ effectSize: 0, nobs: 20 })).toBeCloseTo(0.05, 10)
    expect(() => powerTTest({ effectSize: 0.5, nobs: 20, alpha: 1 })).toThrow()
  })

  it('simulated power brackets the analytic power', () => {
    const analytic = powerTTest({ effectSize: 0.5, nobs: 30 })
    const sim = simulatePower((k) => ttest1samp(toFlat(normal(k, [30])).map((v) => v + 0.5)).pValue, {
      nSimulations: 400,
      seed: 3,
    })
    expect(sim.interval?.[0]).toBeLessThan(analytic)
    expect(sim.interval?.[1]).toBeGreaterThan(analytic)
  })
})
