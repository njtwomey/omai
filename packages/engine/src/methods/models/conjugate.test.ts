import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { item } from '../../lib/tensor/index.ts'
import { betaBinomialModel, gammaPoissonModel, normalNormalModel } from './conjugate.ts'

describe('conjugate models', () => {
  it('beta-binomial: Beta(1, 1) and 7 heads in 10 flips gives Beta(8, 4)', () => {
    const r = betaBinomialModel([1, 1, 0, 1, 1, 0, 1, 1, 0, 1])
    expect(r.posteriorMean).toBeCloseTo(8 / 12, 12)
    expect(r.sequence.n).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expect(r.sequence.mle.at(-1)).toBeCloseTo(0.7, 12)
    const [curve] = recordsOf(r.exposition, 'learningCurve')
    expect(curve?.trainSpread?.length).toBe(11)
    const [densities] = recordsOf(r.exposition, 'densities')
    expect(densities?.curves.map((c) => c.role)).toEqual(['prior', 'posterior'])
    expect(densities?.steps?.length).toBe(11)
    expect(densities?.steps?.at(-1)?.density).toEqual(densities?.curves[1]?.density)
    const [table] = recordsOf(r.exposition, 'coefficientTable')
    const [lo, hi] = table?.intervals?.[1] ?? [0, 0]
    expect(lo).toBeLessThan(8 / 12)
    expect(hi).toBeGreaterThan(8 / 12)
  })

  it('binomial trials, gamma-Poisson and normal-normal posteriors are exact', () => {
    expect(betaBinomialModel([3, 4], { trials: 5, a: 2, b: 2 }).posteriorMean).toBeCloseTo(9 / 14, 12)
    // Gamma(1, 1) and counts 2, 3, 4: Gamma(10, 4).
    expect(gammaPoissonModel([2, 3, 4]).posteriorMean).toBeCloseTo(2.5, 12)
    // N(0, 1) prior, sigma 1, data mean 2 over 3: precision 4, mean 6 / 4.
    const r = normalNormalModel([1, 2, 3])
    expect(r.posteriorMean).toBeCloseTo(1.5, 12)
    expect(item(r.posterior.stddev())).toBeCloseTo(0.5, 12)
  })
})
