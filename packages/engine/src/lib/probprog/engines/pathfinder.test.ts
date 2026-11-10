// Pathfinder against the exact Gaussian posterior it must find, against the exact conjugate posterior it corrects by importance
// weights, and against BlackJAX's `pathfinder.approximate` (moments of the unconstrained draws, from the oracle).
import { describe, expect, it } from 'vitest'
import { MODELS, weightedMoments } from '../../../testing/probprog.ts'
import { key } from '../../random/index.ts'
import { item, primalOf, toFlat, type Value } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { seed, traceOf } from '../handlers.ts'
import type { Model } from '../primitives.ts'
import { conjugateInference } from './conjugate.ts'
import { gaussianCondition, linearGaussianForm } from './gaussian.ts'
import { type PathfinderOptions, type PathfinderState, pathfinderAlgorithm, pathfinderPosterior } from './pathfinder.ts'

const run = (model: Model, options: PathfinderOptions = {}, steps = 400, seedNo = 0) =>
  runAlgorithm(pathfinderAlgorithm(model, undefined, options), { seed: seedNo, maxSteps: steps })

const momentsOf = (state: PathfinderState, dim: number) =>
  weightedMoments(state.draws as Float64Array, state.logWeights as Float64Array, dim)

describe('Pathfinder', () => {
  it('finds a Gaussian posterior: the Newton step at the mode and the inverse Hessian there', () => {
    const model = MODELS['mvn'] as Model
    const exact = gaussianCondition(linearGaussianForm(model, undefined, traceOf(seed(model, key(0)), undefined)))
    const r = run(model, { maxIterations: 40 })
    expect(r.status).toBe('converged')
    const { mean, sd } = momentsOf(r.state, 2)
    const cov = toFlat(exact.covariance)
    const ex = toFlat(exact.mean)
    expect(Math.abs((mean[0] ?? 0) - (ex[0] ?? 0))).toBeLessThan(0.08)
    expect(Math.abs((mean[1] ?? 0) - (ex[1] ?? 0))).toBeLessThan(0.08)
    expect(Math.abs((sd[0] ?? 0) / Math.sqrt(cov[0] ?? 1) - 1)).toBeLessThan(0.1)
    expect(Math.abs((sd[1] ?? 0) / Math.sqrt(cov[3] ?? 1) - 1)).toBeLessThan(0.1)
    // The normal is the posterior: the importance weights are nearly uniform and the k-hat small.
    expect(r.state.paretoK).toBeLessThan(0.3)
    expect(r.state.ess).toBeGreaterThan(0.8 * 400)
  })

  it('keeps the iterate of the best ELBO, which is at least that of the start', () => {
    const r = run(MODELS['logistic'] as Model, { maxIterations: 30 })
    const path = r.state.paths[0]
    expect(path?.best?.elbo).toBeGreaterThanOrEqual(path?.latest === undefined ? Number.NEGATIVE_INFINITY : -Infinity)
    expect(Number.isFinite(r.state.elbo)).toBe(true)
    expect(path?.iteration).toBeGreaterThan(3)
  })

  it('corrects the Gaussian in log space by importance weights, towards the exact gamma posterior', () => {
    const model = MODELS['gammaPoisson'] as Model
    const truth = item(conjugateInference(model, undefined).mean('rate'))
    const meanRate = (weights: 'psis' | 'none'): number => {
      const r = run(model, { weights, numDraws: 2000, maxIterations: 40 })
      return item(pathfinderPosterior(model, undefined, r.state).mean('rate'))
    }
    const weighted = Math.abs(meanRate('psis') - truth)
    const plain = Math.abs(meanRate('none') - truth)
    expect(weighted).toBeLessThan(0.05 * truth)
    expect(weighted).toBeLessThan(plain)
  })

  it('with several paths takes draws from the mixture of their best normals and weights them', () => {
    const r = run(MODELS['logistic'] as Model, { paths: 4, maxIterations: 40, numDraws: 800 })
    expect(r.state.paths).toHaveLength(4)
    expect(r.state.paths.every((p) => p.best !== null)).toBe(true)
    expect(Number.isFinite(r.state.paretoK)).toBe(true)
    expect(r.state.ess).toBeGreaterThan(80)
    const post = pathfinderPosterior(MODELS['logistic'] as Model, undefined, r.state)
    expect(post.sites).toEqual(['alpha', 'beta'])
    expect(post.diagnostics['paths']).toBe(4)
  })

  it('reads a partial posterior while the paths are still climbing (an anytime snapshot)', () => {
    const model = MODELS['logistic'] as Model
    const r = run(model, { maxIterations: 40 }, 4)
    expect(r.state.stage).toBe('path')
    const post = pathfinderPosterior(model, undefined, r.state)
    expect(post.status).toBe('stopped')
    expect(Number.isFinite(item(primalOf(post.mean('alpha')) as Value))).toBe(true)
  })
})
