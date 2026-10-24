import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { meanOf, varianceOf } from '../base/index.ts'
import { key } from '../random/index.ts'
import { item, sliceAxis, sum, tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  bridgeCovariance,
  brownianCovariance,
  brownianKarhunenLoeve,
  conditionGaussianProcess,
  karhunenLoeve,
  ouCovariance,
  sampleGaussianProcess,
  sampleKarhunenLoeve,
  samplePosterior,
} from './gaussian-processes.ts'
import { timeGrid } from './paths.ts'

describe('Gaussian processes as random functions', () => {
  it('sample paths with the covariance of the process', () => {
    const t = [0, 0.5, 1]
    const s = sampleGaussianProcess(key(1), brownianCovariance(), t, { paths: 4000 })
    expect(s.shape).toEqual([4000, 3])
    // W_0 = 0 exactly up to the jitter; Var W_1 = 1; Cov(W_0.5, W_1) = 0.5
    const w0 = toFlat(sliceAxis(s, 1, 0, 1))
    expect(w0.every((v) => Math.abs(v) < 1e-3)).toBe(true)
    const a = toFlat(sliceAxis(s, 1, 1, 2))
    const b = toFlat(sliceAxis(s, 1, 2, 3))
    expect(Math.abs(varianceOf(b) - 1)).toBeLessThan(4 * Math.sqrt(2 / 4000))
    const cov = meanOf(a.map((x, i) => x * (b[i] ?? 0))) - meanOf(a) * meanOf(b)
    expect(Math.abs(cov - 0.5)).toBeLessThan(0.06)
  })

  it('differentiates samples in covariance parameters (reparameterisation)', () => {
    const f = (theta: Value, sigma: Value): Value =>
      sampleGaussianProcess(key(2), ouCovariance({ theta, sigma }), [0, 0.3, 1], { paths: 2 })
    expectConformant(f, [tensor(1.2), tensor(0.7)])
  })

  it('conditions Brownian motion into a Brownian bridge', () => {
    const t = [0.25, 0.5, 0.75]
    const post = conditionGaussianProcess(brownianCovariance(), [1], [0], t, { noise: 0 })
    const bridge = toFlat(bridgeCovariance()(tensor([[0.25], [0.5], [0.75]]), tensor([[0.25], [0.5], [0.75]])))
    toFlat(post.covariance).forEach((v, i) => {
      expect(v).toBeCloseTo(bridge[i] ?? 0, 12)
    })
    expect(toFlat(post.mean)).toEqual([0, 0, 0])
    const draws = samplePosterior(
      key(3),
      conditionGaussianProcess(brownianCovariance(), [1], [2], [0.5], { noise: 1e-9 }),
      3000,
    )
    expect(Math.abs(meanOf(toFlat(draws)) - 1)).toBeLessThan(4 * Math.sqrt(0.25 / 3000))
    expect(() => conditionGaussianProcess(brownianCovariance(), [1, 2], [1], [0.5])).toThrow()
  })

  it('differentiates the posterior and the marginal likelihood', () => {
    const f = (y: Value, theta: Value): Value =>
      conditionGaussianProcess(ouCovariance({ theta, sigma: 1 }), [0, 1, 2], y, [0.5, 1.5], { noise: 0.1 }).mean
    expectConformant(f, [tensor([0.3, -0.2, 0.5]), tensor(0.8)])
    const g = (theta: Value): Value =>
      conditionGaussianProcess(ouCovariance({ theta, sigma: 1 }), [0, 1, 2], [0.3, -0.2, 0.5], [0.5], { noise: 0.1 })
        .logMarginalLikelihood
    expectConformant(g, [tensor(0.8)])
  })
})

describe('Karhunen-Loeve expansions', () => {
  it('the Nystrom eigenvalues of Brownian motion approach the closed form', () => {
    const grid = timeGrid(1, 200)
    const numeric = toFlat(karhunenLoeve(brownianCovariance(), grid, 4).eigenvalues)
    const exact = toFlat(brownianKarhunenLoeve(grid, 4).eigenvalues)
    for (let k = 0; k < 4; k++)
      expect(Math.abs((numeric[k] ?? 0) - (exact[k] ?? 0)) / (exact[k] ?? 1)).toBeLessThan(1e-3)
    // the eigenfunctions are orthonormal under the trapezoid rule and match sqrt(2) sin((k - 1/2) pi t)
    const phi = karhunenLoeve(brownianCovariance(), grid, 2).eigenfunctions
    const first = toFlat(sliceAxis(phi, 0, 0, 1))
    const closed = toFlat(sliceAxis(brownianKarhunenLoeve(grid, 1).eigenfunctions, 0, 0, 1))
    first.forEach((v, j) => {
      expect(Math.abs(v - (closed[j] ?? 0))).toBeLessThan(5e-3)
    })
  })

  it('truncated expansions reproduce the variance t minus the discarded tail', () => {
    const kl = brownianKarhunenLoeve(timeGrid(1, 20), 50)
    const s = sampleKarhunenLoeve(key(4), kl, 4000)
    const end = toFlat(sliceAxis(s, 1, 20, 21))
    expect(Math.abs(varianceOf(end) - 1)).toBeLessThan(4 * Math.sqrt(2 / 4000) + 0.01)
    expect(item(sum(kl.eigenvalues))).toBeCloseTo(0.5, 2)
    expect(toFlat(sliceAxis(s, 1, 0, 1)).every((v) => v === 0)).toBe(true)
  })
})
