// The grid kernels of the references against closed forms: each recovers the moments of a density whose answer is known.
import { describe, expect, it } from 'vitest'
import { gridMarginals, hyperPosterior, logWeights, quadratureMarginal } from './grid.ts'
import { gaussianMixtureMarginal, probabilityMarginal, weightedMarginal } from './marginal.ts'
import { normalCdf, normalCdfFast, normalPdf } from './special.ts'

describe('the grid kernels', () => {
  it('quadratureMarginal and gridMarginals recover a normal', () => {
    const q = quadratureMarginal((x) => -0.5 * ((x - 1) / 2) ** 2, -15, 17)
    expect(q.mean).toBeCloseTo(1, 6)
    expect(q.sd).toBeCloseTo(2, 6)
    const g = gridMarginals(
      (x) => -0.5 * (x[0] ?? 0) ** 2 - 0.5 * ((x[1] ?? 0) / 2) ** 2,
      [
        [-8, 8],
        [-16, 16],
      ],
      61,
    )
    expect(g.marginal(1).sd).toBeCloseTo(2, 4)
  })

  it('hyperPosterior recovers a gamma and mixes conditionals', () => {
    const post = hyperPosterior((h) => 2 * Math.log(h) - 3 * h, 1e-3, 40)
    expect(post.hyper.mean).toBeCloseTo(1, 3)
    const m = post.mix((h) => ({ mean: h, variance: 0, pdf: () => 0 }))
    expect(m.mean).toBeCloseTo(post.hyper.mean, 6)
  })

  it('weightedMarginal reads the same moments under each density', () => {
    const points = [0, 1, 2, 3, 4]
    const weights = [1, 4, 6, 4, 1]
    for (const density of ['cell', 'nearest', 'linear'] as const) {
      const m = weightedMarginal(points, weights, density)
      expect(m.mean).toBeCloseTo(2, 12)
      expect(m.sd).toBeCloseTo(1, 12)
    }
    expect(weightedMarginal(points, weights, 'cell').pdf(2)).toBeCloseTo(6 / 16, 12)
  })

  it('gaussianMixtureMarginal has the total variance and density of its parts', () => {
    const m = gaussianMixtureMarginal([
      { weight: 0.5, mean: -1, variance: 1 },
      { weight: 0.5, mean: 1, variance: 1 },
    ])
    expect(m.mean).toBeCloseTo(0, 12)
    expect(m.sd).toBeCloseTo(Math.SQRT2, 12)
    expect(m.pdf(1)).toBeCloseTo(0.5 * normalPdf(1, 1, 1) + 0.5 * normalPdf(1, -1, 1), 12)
  })

  it('logWeights puts the largest weight at one; probabilityMarginal has the spread of a coin; the cdfs agree to 1e-7', () => {
    expect(logWeights([-1000, -1001]).weights[0]).toBe(1)
    expect(probabilityMarginal(0.25).sd).toBeCloseTo(Math.sqrt(0.1875), 12)
    for (const x of [-3, -0.5, 0, 1.2, 2.5]) expect(Math.abs(normalCdfFast(x) - normalCdf(x))).toBeLessThan(1e-7)
  })
})
