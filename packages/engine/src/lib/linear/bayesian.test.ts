import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { sineCurve } from '../datasets/index.ts'
import { multivariateNormal } from '../dists/index.ts'
import { key, permutation } from '../random/index.ts'
import { item, toFlat } from '../tensor/index.ts'
import { runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  type BayesianPosterior,
  bayesianLinearAlgorithm,
  bayesianLinearFit,
  maximiseEvidence,
  polynomialBasis,
  posteriorPredictive,
  priorPosterior,
} from './bayesian.ts'
import { designOf } from './design.ts'

const { data } = generate(sineCurve, { key: key(4), knobs: { n: 40, noise: 0.2 } })
const basis = polynomialBasis(4, data.x)
const design = { x: basis.expand(data.x), y: Float64Array.from(toFlat(data.y ?? [])) }
const prior = { alpha: 2, beta: 25 }
const order = Array.from(toFlat(permutation(key(9), 40)))

const close = (a: readonly number[], b: readonly number[], tol: number): void => {
  expect(a.length).toBe(b.length)
  for (const [i, v] of a.entries())
    expect(Math.abs(v - (b[i] ?? 0))).toBeLessThanOrEqual(tol * (1 + Math.abs(b[i] ?? 0)))
}

describe('Bayesian linear regression', () => {
  it('reaches the batch posterior after the last row, whatever the order', () => {
    const batch = bayesianLinearFit(design, prior)
    for (const rows of [Array.from({ length: 40 }, (_, i) => i), order]) {
      const out = runAlgorithm(bayesianLinearAlgorithm(design, prior, rows), { maxSteps: 100 })
      expect(out.status).toBe('converged')
      expect(out.steps).toBe(40)
      close(out.state.mean, batch.mean, 1e-8)
      close(out.state.covariance.flat(), batch.covariance.flat(), 1e-8)
      close(out.state.precision.flat(), batch.precision.flat(), 1e-12)
      expect(out.state.logEvidence).toBeCloseTo(batch.logEvidence, 8)
      expect(out.state.n).toBe(batch.n)
    }
  })

  it('has the batch posterior of the rows seen at every step', () => {
    const alg = bayesianLinearAlgorithm(design, prior, order)
    const p = basis.expand(data.x).d
    let state: BayesianPosterior = alg.init(key(0))
    for (let t = 1; t <= 8; t++) {
      state = alg.step(state, undefined as never)
      const rows = order.slice(0, t)
      const seen = {
        x: {
          n: t,
          d: p,
          data: Float64Array.from(rows.flatMap((i) => Array.from(design.x.data.subarray(i * p, (i + 1) * p)))),
        },
        y: Float64Array.from(rows.map((i) => design.y[i] ?? 0)),
      }
      const batch = bayesianLinearFit(seen, prior)
      close(state.mean, batch.mean, 1e-8)
      expect(state.logEvidence).toBeCloseTo(batch.logEvidence, 8)
    }
  })

  it('shrinks the predictive uncertainty as data arrive and starts at the prior', () => {
    const alg = bayesianLinearAlgorithm(design, prior, order)
    const spread = (t: number, state: BayesianPosterior): number =>
      alg.observe?.(state, t).metrics?.['meanPredictiveStd'] ?? Number.NaN
    let state = alg.init(key(0))
    let last = spread(0, state)
    expect(last).toBeGreaterThan(1.5 / Math.sqrt(prior.beta))
    for (let t = 0; t < 40; t++) {
      state = alg.step(state, undefined as never)
      const now = spread(t + 1, state)
      expect(now).toBeLessThanOrEqual(last + 1e-12)
      last = now
    }
    expect(last).toBeLessThan(1.2 / Math.sqrt(prior.beta))
  })

  it('predicts the noise plus the weights uncertainty, and the prior alone before data', () => {
    const empty = priorPosterior(2, { alpha: 4, beta: 9 })
    const { mean, variance } = posteriorPredictive(
      empty,
      [
        [1, 0],
        [0, 2],
      ],
      9,
    )
    expect(mean).toEqual([0, 0])
    close(variance, [1 / 9 + 1 / 4, 1 / 9 + 4 / 4], 1e-14)
    const fit = bayesianLinearFit(design, prior)
    const at = posteriorPredictive(fit, basis.expand(data.x), prior.beta)
    expect(at.mean.length).toBe(40)
    for (const v of at.variance) expect(v).toBeGreaterThanOrEqual(1 / prior.beta)
  })

  it('agrees with the conjugate Gaussian formulas on one weight', () => {
    // One weight, three observations of phi = 1: precision alpha + 3 beta, mean beta sum(y) / precision.
    const one = designOf({ x: [[1], [1], [1]], y: [1, 2, 6] }, 'test')
    const post = bayesianLinearFit(one, { alpha: 2, beta: 4 })
    expect(post.precision).toEqual([[14]])
    expect(post.mean[0]).toBeCloseTo((4 * 9) / 14, 12)
    expect(post.covariance[0]?.[0]).toBeCloseTo(1 / 14, 12)
    // The evidence is a Gaussian marginal: y ~ N(0, I / beta + 1 1^T / alpha).
    const cov = [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? 1 / 4 : 0) + 1 / 2))
    const direct = item(multivariateNormal([0, 0, 0], { covariance: cov }).logProb([1, 2, 6]))
    expect(post.logEvidence).toBeCloseTo(direct, 10)
  })

  it('keeps a plain state that round-trips through JSON', () => {
    const alg = bayesianLinearAlgorithm(design, prior, order)
    const s = alg.step(alg.init(key(0)), undefined as never)
    expect(stateFromJson(JSON.parse(JSON.stringify(stateToJson(s))))).toEqual(s)
    expect(alg.deserialise?.(alg.serialise?.(s) as never)).toEqual(s)
  })

  it('maximises the evidence over alpha and beta, and finds the noise level', () => {
    const fit = maximiseEvidence(design, { alpha: 1, beta: 1 })
    expect(fit.converged).toBe(true)
    expect(fit.gamma).toBeGreaterThan(0)
    expect(fit.gamma).toBeLessThanOrEqual(5)
    // The noise has standard deviation 0.2 (beta = 25); the estimate is close.
    expect(fit.beta).toBeGreaterThan(12)
    expect(fit.beta).toBeLessThan(50)
    // The evidence at the optimum beats the evidence at the start.
    expect(fit.posterior.logEvidence).toBeGreaterThan(bayesianLinearFit(design, { alpha: 1, beta: 1 }).logEvidence)
  })

  it('rejects invalid precisions and rows', () => {
    expect(() => bayesianLinearFit(design, { alpha: 0, beta: 1 })).toThrow()
    expect(() => bayesianLinearAlgorithm(design, prior, [40])).toThrow()
  })
})

describe('polynomialBasis', () => {
  it('scales the features to [-1, 1] and expands them', () => {
    const b = polynomialBasis(3, [[10], [20]])
    expect(Array.from(b.expand([[10]]).data)).toEqual([1, -1, 1, -1])
    expect(Array.from(b.expand([[20]]).data)).toEqual([1, 1, 1, 1])
    expect(b.expand([[15]]).d).toBe(4)
  })
})
