import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { rbfKernel, scaleKernel } from '../../lib/kernels/index.ts'
import { key } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import { classificationSet, regressionSet } from '../../testing/supervised.ts'
import { GaussianProcessClassifier, gaussianProcessClassifier } from './classification.ts'
import { GaussianProcessRegressor, logMarginalLikelihood } from './regression.ts'
import { sparseGpRegressor } from './sparse.ts'

const sine = regressionSet('sineCurve', { n: 30 }, 4)

describe('Gaussian processes', () => {
  it('regression returns its posterior grid, samples and marginal-likelihood surface', () => {
    const f = new GaussianProcessRegressor({ alpha: 0.05, surfaceResolution: 8 }).fit(sine.x, sine.y, { key: key(1) })
    expect(f.score(sine.x, sine.y)).toBeGreaterThan(0.8)
    const grid = recordsOf(f.exposition, 'posteriorGrid')[0]
    expect(grid?.samples).toHaveLength(3)
    expect(grid?.mean).toHaveLength(60)
    const surface = recordsOf(f.exposition, 'hyperparameterSurface')[0]
    expect(surface?.values).toHaveLength(8)
    expect(surface?.path?.length).toBeGreaterThan(1)
    expect(f.model.predictDistribution(sine.x).batchShape).toEqual([30])
  })

  it('regression draws a chosen number of keyed samples from the posterior or the prior', () => {
    const kernel = scaleKernel(rbfKernel({ lengthScale: 0.7 }), 1)
    const fit = (draw: 'posterior' | 'prior', seed: number) =>
      recordsOf(
        new GaussianProcessRegressor({
          kernel,
          alpha: 0.01,
          optimize: false,
          surfaceResolution: 0,
          nSamples: 10,
          draw,
          gridPoints: 120,
        }).fit(sine.x, sine.y, { key: key(seed) }).exposition,
        'posteriorGrid',
      )[0]
    const post = fit('posterior', 3)
    const prior = fit('prior', 3)
    expect(post?.samples).toHaveLength(10)
    expect(post?.samples?.[0]).toHaveLength(120)
    expect(prior?.mean.every((m) => m === 0)).toBe(true)
    expect(prior?.variance.every((v) => Math.abs(v - 1) < 1e-9)).toBe(true)
    expect(prior?.observed?.targets).toHaveLength(30)
    expect(Math.max(...(post?.variance ?? []))).toBeLessThan(1)
    expect(fit('posterior', 3)?.samples).toEqual(post?.samples)
    expect(fit('posterior', 4)?.samples).not.toEqual(post?.samples)
  })

  it('classification by the Laplace approximation', () => {
    const moons = classificationSet('moons', { n: 40, noise: 0.15 }, 2)
    const f = new GaussianProcessClassifier({ surfaceResolution: 5, maxIter: 40 }).fit(moons.x, moons.y)
    expect(f.score(moons.x, moons.y)).toBeGreaterThan(0.8)
    expect(recordsOf(f.exposition, 'hyperparameterSurface')).toHaveLength(1)
    expect(recordsOf(f.exposition, 'decisionSurface')).toHaveLength(1)
    const surface = recordsOf(f.exposition, 'decisionSurface')[0]
    expect(surface?.stepName).toBe('optimiser step')
    expect(surface?.steps?.length).toBeGreaterThan(2)
    expect(surface?.objective?.lowerIsBetter).toBe(false)
    const blobs = classificationSet('blobs', { n: 45 }, 3)
    expect(gaussianProcessClassifier({ optimize: false }).fit(blobs).predictProba(blobs.x).shape).toEqual([45, 3])
  })

  it('sparse approximations: VFE with every row inducing is the exact likelihood', () => {
    const kernel = scaleKernel(rbfKernel({ lengthScale: 1 }), 1)
    const exact = toFlat(logMarginalLikelihood(kernel, sine.x, sine.y, 0.1))[0] ?? 0
    const all = sparseGpRegressor({ kernel, nInducing: 30, noise: 0.1, optimize: false }).fit(sine)
    expect(all.params.objective).toBeCloseTo(exact, 4)
    for (const method of ['sor', 'dtc', 'fitc', 'vfe'] as const) {
      const f = sparseGpRegressor({ method, nInducing: 10 }).fit(sine, { key: key(2) })
      expect(f.score?.(sine), method).toBeGreaterThan(0.6)
      expect(recordsOf(f.exposition, 'posteriorGrid')).toHaveLength(1)
    }
  })
})
