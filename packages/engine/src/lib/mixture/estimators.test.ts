import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { adjustedRandScore } from '../metrics/index.ts'
import { foldIn, key } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { bayesianGaussianMixture, gaussianMixture } from './estimators.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 90, centers: 3, clusterStd: 0.6 } })
const ari = (labels: ArrayLike<number>): number => adjustedRandScore(data.y, Array.from(labels)).estimate.value

describe('gaussianMixture', () => {
  it('fits the blobs, and its verbs agree: probabilities sum to one, labels are their argmax', () => {
    const f = gaussianMixture.with({ nComponents: 3 }).fit(data, { key: key(0) })
    expect(ari(toFlat(f.predict(data.x)))).toBeGreaterThan(0.95)
    const proba = toArray(f.predictProba(data.x)) as number[][]
    for (const r of proba) expect(r.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 12)
    expect(proba.map((r) => r.indexOf(Math.max(...r)))).toEqual(Array.from(toFlat(f.predict(data.x))))
    expect(f.score(data)).toBeCloseTo(f.params.logLikelihood, 12)
    const ss = toFlat(f.scoreSamples(data.x))
    expect(ss.reduce((a, v) => a + v, 0) / ss.length).toBeCloseTo(f.score(data), 12)
  })

  it('BIC and AIC follow their definitions and prefer the true number of components', () => {
    const fits = [1, 2, 3, 6].map((k) => gaussianMixture.with({ nComponents: k }).fit(data, { key: key(1) }))
    const n = 90
    const f3 = fits[2]
    if (f3 === undefined) throw new Error('no fit')
    const params = 3 * 2 + 3 * 3 + 2 // means, full covariances (3 per component), weights
    expect(f3.aic(data.x)).toBeCloseTo(-2 * n * f3.score(data) + 2 * params, 8)
    expect(f3.bic(data.x)).toBeCloseTo(-2 * n * f3.score(data) + params * Math.log(n), 8)
    const bics = fits.map((f) => f.bic(data.x))
    expect(bics.indexOf(Math.min(...bics))).toBe(2)
  })

  it('keeps the restart with the highest likelihood', () => {
    const many = gaussianMixture.with({ nComponents: 4, init: 'random', nInit: 6 }).fit(data, { key: key(5) })
    const singles = Array.from(
      { length: 6 },
      (_, r) =>
        gaussianMixture.with({ nComponents: 4, init: 'random' }).fit(data, { key: foldIn(key(5), r) }).params
          .logLikelihood,
    )
    expect(many.params.logLikelihood).toBeCloseTo(Math.max(...singles), 9)
  })

  it('revive rebuilds the verbs, and a session steps the EM with assignments probes', () => {
    const f = gaussianMixture.with({ nComponents: 3, covarianceType: 'diag' }).fit(data, { key: key(2) })
    expect(toArray(gaussianMixture.revive(f.params).predict(data.x))).toEqual(toArray(f.predict(data.x)))
    const s = gaussianMixture.with({ nComponents: 3 }).session(data, { key: key(2), logs: 'all' })
    s.advance({ steps: 2 })
    expect((s.probe(2, 'model/assignments') as number[]).length).toBe(90)
  })
})

describe('bayesianGaussianMixture', () => {
  it('a small weight concentration lets the data switch off surplus components', () => {
    const f = bayesianGaussianMixture
      .with({ nComponents: 6, weightConcentrationPrior: 0.001, maxIter: 200 })
      .fit(data, { key: key(0) })
    const used = f.params.weights.filter((w) => w > 0.05).length
    expect(used).toBe(3)
    expect(ari(toFlat(f.predict(data.x)))).toBeGreaterThan(0.9)
    expect(f.params.weights.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 10)
    const proba = toArray(f.predictProba(data.x)) as number[][]
    for (const r of proba) expect(r.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 10)
    expect(Number.isFinite(f.score(data))).toBe(true)
  })

  it('revive gives the same predictions', () => {
    const f = bayesianGaussianMixture.with({ nComponents: 3, weightPrior: 'dirichlet_distribution' }).fit(data)
    expect(toArray(bayesianGaussianMixture.revive(f.params).predict(data.x))).toEqual(toArray(f.predict(data.x)))
  })
})
