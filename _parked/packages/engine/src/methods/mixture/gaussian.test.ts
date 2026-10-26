import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { blobs } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { adjustedRandScore } from '../../lib/metrics/index.ts'
import { key } from '../../lib/random/index.ts'
import { sum, toFlat } from '../../lib/tensor/index.ts'
import { BayesianGaussianMixture } from './bayesian.ts'
import { GaussianMixture, gaussianMixture, mixtureCriteria } from './gaussian.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 90, centers: 3, clusterStd: 0.6 } })

describe('Gaussian mixtures', () => {
  for (const covarianceType of ['full', 'tied', 'diag', 'spherical'] as const) {
    it(`${covarianceType}: EM recovers the blobs and never lowers the log likelihood`, () => {
      const f = gaussianMixture({ nComponents: 3, covarianceType }).fit(data, { key: key(0) })
      expect(adjustedRandScore(data.y, toFlat(f.predict(data.x))).estimate.value).toBeGreaterThan(0.95)
      const [steps] = recordsOf(f.exposition, 'mixtureSteps')
      const obj = steps?.objective ?? []
      for (let t = 1; t < obj.length; t++) expect(obj[t] ?? 0).toBeGreaterThanOrEqual((obj[t - 1] ?? 0) - 1e-9)
      expect(steps?.means.length).toBe(obj.length)
      expect(steps?.responsibilities?.[0]?.length).toBe(90)
      for (const s of toFlat(sum(f.predictProba(data.x), 1))) expect(s).toBeCloseTo(1, 12)
    })
  }

  it('BIC prefers the true number of components; the class form fits', () => {
    const bic = [1, 3, 6].map(
      (k) => mixtureCriteria(gaussianMixture({ nComponents: k }).fit(data, { key: key(1) }), data.x).bic,
    )
    expect(bic[1]).toBeLessThan(bic[0] ?? 0)
    expect(bic[1]).toBeLessThan(bic[2] ?? 0)
    expect(
      new GaussianMixture({ nComponents: 3 }).fit(data.x, undefined, { key: key(2) }).predict(data.x).shape,
    ).toEqual([90])
  })

  it('the Bayesian mixture switches off unneeded components', () => {
    const f = new BayesianGaussianMixture({ nComponents: 6, weightConcentrationPrior: 0.01 }).fit(data.x, undefined, {
      key: key(4),
    }).model
    expect(f.params.weights.filter((w) => w > 0.05).length).toBe(3)
    expect(adjustedRandScore(data.y, toFlat(f.predict(data.x))).estimate.value).toBeGreaterThan(0.95)
    const [steps] = recordsOf(f.exposition, 'mixtureSteps')
    expect(steps?.weights[0]?.length).toBe(6)
    expect(Number.isFinite(f.score({ x: data.x }))).toBe(true)
  })
})

describe('component selection', () => {
  it('BIC picks three components for three blobs, with a curve record and density surfaces', async () => {
    const { selectComponents } = await import('./selection.ts')
    const s = selectComponents(data.x, [1, 2, 3, 4, 5], { key: key(9) })
    expect(s.best).toBe(3)
    expect(s.curve.values).toEqual([1, 2, 3, 4, 5])
    const f = gaussianMixture({ nComponents: 3 }).fit(data, { key: key(0) })
    expect(recordsOf(f.exposition, 'decisionSurface')[0]?.values?.length).toBe(40)
  })
})
