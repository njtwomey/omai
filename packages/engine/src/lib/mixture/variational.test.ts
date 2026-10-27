import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { adjustedRandScore } from '../metrics/index.ts'
import { key } from '../random/index.ts'
import { runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { componentLabels, initialResponsibilities } from './gaussian.ts'
import {
  expectedWeights,
  type VariationalState,
  variationalMixtureAlgorithm,
  variationalPriors,
  variationalResponsibilities,
} from './variational.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 90, centers: 3, clusterStd: 0.6 } })

describe('variational Gaussian mixture', () => {
  for (const weightPrior of ['dirichlet_process', 'dirichlet_distribution'] as const) {
    for (const covarianceType of ['full', 'tied', 'diag', 'spherical'] as const) {
      it(`${weightPrior}, ${covarianceType}: the lower bound never falls`, () => {
        const start = initialResponsibilities(data.x, 6, 'random', key(1))
        const priors = variationalPriors(data.x, 6, { covarianceType, weightConcentration: 0.1 })
        const alg = variationalMixtureAlgorithm(data.x, start, priors, { weightPrior, covarianceType, tol: 0 })
        let state = alg.init(key(0))
        const bounds: number[] = []
        for (let t = 0; t < 25; t++) {
          state = alg.step(state, undefined as never)
          bounds.push(state.lowerBound)
        }
        for (let t = 1; t < bounds.length; t++)
          expect(bounds[t] ?? 0).toBeGreaterThanOrEqual((bounds[t - 1] ?? 0) - 1e-9 * Math.abs(bounds[t - 1] ?? 1))
      })
    }
  }

  it('switches off the unneeded components of eight on three separated blobs', () => {
    const start = initialResponsibilities(data.x, 8, 'kmeans', key(1))
    const priors = variationalPriors(data.x, 8, { weightConcentration: 0.01 })
    const out = runAlgorithm(variationalMixtureAlgorithm(data.x, start, priors), { maxSteps: 200 })
    expect(out.status).toBe('converged')
    const weights = expectedWeights(out.state.posterior)
    expect(weights.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 12)
    expect(weights.filter((w) => w > 0.05).length).toBe(3)
    expect(weights.filter((w) => w < 1e-2).length).toBe(5)
    const labels = componentLabels(variationalResponsibilities(data.x, out.state.posterior).resp)
    expect(adjustedRandScore(data.y, labels).estimate.value).toBeGreaterThan(0.95)
  })

  it('also does it from a random start, and with a finite Dirichlet prior', () => {
    for (const weightPrior of ['dirichlet_process', 'dirichlet_distribution'] as const) {
      const start = initialResponsibilities(data.x, 8, 'random', key(4))
      const priors = variationalPriors(data.x, 8, { weightConcentration: 0.01 })
      const out = runAlgorithm(variationalMixtureAlgorithm(data.x, start, priors, { weightPrior }), { maxSteps: 300 })
      const weights = expectedWeights(out.state.posterior)
      expect(weights.filter((w) => w > 0.05).length).toBeLessThanOrEqual(3)
      expect(weights.filter((w) => w < 0.02).length).toBeGreaterThanOrEqual(5)
    }
  })

  it('keeps a plain state that round-trips through JSON', () => {
    const start = initialResponsibilities(data.x, 3, 'kmeans', key(0))
    const alg = variationalMixtureAlgorithm(data.x, start, variationalPriors(data.x, 3))
    const s: VariationalState = alg.step(alg.init(key(0)), undefined as never)
    expect(Object.keys(s).sort()).toEqual(['change', 'lowerBound', 'posterior', 'resp'])
    expect(stateFromJson(JSON.parse(JSON.stringify(stateToJson(s))))).toEqual(s)
    const first = alg.init(key(0))
    expect(stateFromJson(JSON.parse(JSON.stringify(stateToJson(first))))).toEqual(first)
  })
})
