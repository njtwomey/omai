import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { adjustedRandScore } from '../metrics/index.ts'
import { key } from '../random/index.ts'
import { sum, tensor, toFlat } from '../tensor/index.ts'
import { runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { componentLabels, gaussianEmAlgorithm, initialMixture, mixturePosterior } from './gaussian.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 90, centers: 3, clusterStd: 0.6 } })

describe('Gaussian mixture EM', () => {
  for (const covarianceType of ['full', 'tied', 'diag', 'spherical'] as const) {
    it(`${covarianceType}: recovers the blobs and never lowers the log likelihood`, () => {
      const start = initialMixture(data.x, { nComponents: 3, covarianceType }, key(0))
      const out = runAlgorithm(gaussianEmAlgorithm(data.x, start, { covarianceType }), { maxSteps: 100 })
      expect(out.status).toBe('converged')
      const labels = componentLabels(mixturePosterior(data.x, out.state).resp)
      expect(adjustedRandScore(data.y, labels).estimate.value).toBeGreaterThan(0.95)
      for (const s of toFlat(sum(tensor(out.state.resp), 1))) expect(s).toBeCloseTo(1, 12)
      // More steps never reach a lower likelihood than fewer.
      let last = Number.NEGATIVE_INFINITY
      for (const maxSteps of [2, 4, 8]) {
        const ll = runAlgorithm(gaussianEmAlgorithm(data.x, start, { covarianceType, tol: 0 }), { maxSteps }).state
          .logLikelihood
        expect(ll).toBeGreaterThanOrEqual(last - 1e-9)
        last = ll
      }
    })
  }

  it('keeps a plain state that round-trips through JSON', () => {
    const start = initialMixture(data.x, { nComponents: 3 }, key(0))
    const s = gaussianEmAlgorithm(data.x, start).init(key(0))
    expect(Object.keys(s).sort()).toEqual(['change', 'covariances', 'logLikelihood', 'means', 'resp', 'weights'])
    expect(stateFromJson(JSON.parse(JSON.stringify(stateToJson(s))))).toEqual(s)
  })
})
