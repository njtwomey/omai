import { describe, expect, it } from 'vitest'
import { expectInvariant, expectMoments, gaussian2, moments, standardNormal } from '../../testing/sampling.ts'
import { key, normal } from '../random/index.ts'
import { mul, square, sub, sum, take, tensor } from '../tensor/index.ts'
import { metricSeries, recorder } from '../trace/index.ts'
import { parameterDraws, sampleChain, sampleChains } from './chains.ts'
import { mcse } from './diagnostics/index.ts'
import {
  malaKernel,
  medianBandwidth,
  minibatchGradient,
  polynomialStepSize,
  sgldKernel,
  svgd,
  svgdDirection,
  ulaKernel,
} from './langevin.ts'
import { numericTarget } from './target.ts'

describe('malaKernel', () => {
  it('leaves a correlated Gaussian invariant, with and without a preconditioner', () => {
    const t = gaussian2([0, 0], [1, 2], 0.7)
    const z = normal(key(1), [2000, 2]).data
    const draws = Float64Array.from({ length: 4000 }, (_, k) => {
      const i = Math.floor(k / 2)
      const a = z[2 * i] ?? 0
      return k % 2 === 0 ? a : 2 * (0.7 * a + Math.sqrt(1 - 0.49) * (z[2 * i + 1] ?? 0))
    })
    expectInvariant(malaKernel(t, { stepSize: 0.3 }), draws, { mean: [0, 0], sd: [1, 2] }, { key: key(2), steps: 2 })
    const pre = [
      [1, 1.4],
      [1.4, 4],
    ]
    expectInvariant(
      malaKernel(t, { stepSize: 0.5, preconditioner: pre }),
      draws,
      { mean: [0, 0], sd: [1, 2] },
      { key: key(3) },
    )
  })

  it('samples a Gaussian with acceptance near the optimum for a tuned step', () => {
    const c = sampleChains(malaKernel(standardNormal(3), { stepSize: 0.6 }), {
      key: key(4),
      numChains: 2,
      numDraws: 1500,
    })
    expectMoments(c, { mean: [0, 0, 0], sd: [1, 1, 1] })
    const a = c.stats.acceptanceRate?.data ?? new Float64Array()
    expect(a.reduce((x, y) => x + y, 0) / a.length).toBeGreaterThan(0.5)
  })

  it('rejects proposals outside the support and validates its settings', () => {
    const half = numericTarget((x) => ((x[0] ?? 0) > 0 ? -(x[0] ?? 0) : Number.NEGATIVE_INFINITY), 1, {
      grad: () => [-1],
    })
    const r = sampleChain(malaKernel(half, { stepSize: 0.5 }), [0.5], { key: key(5), numDraws: 200 })
    expect(r.draws.every((v) => v > 0)).toBe(true)
    expect(() => malaKernel(standardNormal(1), { stepSize: 0 })).toThrow()
    expect(() => malaKernel(standardNormal(2), { stepSize: 0.1, preconditioner: [1, -1] })).toThrow()
    expect(() =>
      malaKernel(
        numericTarget(() => 0, 1),
        { stepSize: 0.1 },
      ).init([0]),
    ).toThrow()
  })
})

describe('ulaKernel', () => {
  it('is biased by its step size exactly as theory says: variance 1 / (1 - eps / 2) for a standard normal', () => {
    const eps = 0.4
    const c = sampleChains(ulaKernel(standardNormal(1), { stepSize: eps }), {
      key: key(6),
      numChains: 2,
      numDraws: 4000,
    })
    const x = parameterDraws(c, 0)
    const { sd } = moments(x.data)
    const se = mcse(x, { method: 'sd' })
    expect(Math.abs(sd - Math.sqrt(1 / (1 - eps / 2)))).toBeLessThan(4 * se)
    expect(Math.abs(sd - 1)).toBeGreaterThan(4 * se)
  })
})

describe('sgldKernel', () => {
  it('samples the posterior of a Gaussian mean from minibatch gradients', () => {
    // y_i ~ N(mu, 1) with a N(0, 10^2) prior: the posterior is N(sum y / (n + 0.01), 1 / (n + 0.01)).
    const n = 40
    const y = normal(key(7), [n], { loc: 2 }).data
    const ys = tensor(Array.from(y))
    const gradient = minibatchGradient({
      logPrior: (m) => mul(-0.005, sum(square(m))),
      logLikelihood: (m, batch) => mul(-0.5, sum(square(sub(take(ys, Array.from(batch)), m)))),
      numData: n,
      batchSize: 10,
    })
    const postMean = y.reduce((a, b) => a + b, 0) / (n + 0.01)
    const k = sgldKernel(gradient, 1, { stepSize: polynomialStepSize(0.01, 10, 0.55) })
    const r = sampleChain(k, [0], { key: key(8), numDraws: 2000, numWarmup: 300 })
    const { mean, sd } = moments(r.draws)
    expect(Math.abs(mean - postMean)).toBeLessThan(0.1)
    // Decreasing steps keep the spread close to the posterior sd 1 / sqrt(n) = 0.158.
    expect(sd).toBeGreaterThan(0.08)
    expect(sd).toBeLessThan(0.3)
    expect(r.state.t).toBe(2300)
  })

  it('traces step sizes from its schedule and reports log densities with a target', () => {
    const rec = recorder()
    const k = sgldKernel((_, x) => x.map((v) => -v), 1, { stepSize: (t) => 0.1 / (1 + t), target: standardNormal(1) })
    sampleChain(k, [1], { key: key(9), numDraws: 4, tracer: rec })
    const s = metricSeries(rec.run(), 'stepSize').values
    expect(s[0]).toBeCloseTo(0.1, 14)
    expect(s[1]).toBeCloseTo(0.05, 14)
    expect(Number.isFinite(metricSeries(rec.run(), 'logDensity').values[1] ?? Number.NaN)).toBe(true)
    expect(() => sgldKernel((_, x) => x, 1, { stepSize: -1 })).toThrow()
    expect(() => polynomialStepSize(1, 1, 2)).toThrow()
    expect(() => minibatchGradient({ logPrior: (m) => m, logLikelihood: (m) => m, numData: 3, batchSize: 4 })).toThrow()
  })
})

describe('svgd', () => {
  it('moves particles to a Gaussian target: mean and spread', () => {
    const t = numericTarget((x) => -0.5 * (((x[0] ?? 0) - 2) / 0.5) ** 2, 1, {
      grad: (x) => [-((x[0] ?? 0) - 2) / 0.25],
    })
    const start = Array.from(normal(key(10), [40]).data, (v) => [v])
    const r = svgd(t, start, { stepSize: 0.05, maxIter: 1500, optimizer: 'adagrad' })
    const { mean, sd } = moments(r.particles.data)
    expect(mean).toBeCloseTo(2, 1)
    expect(sd).toBeGreaterThan(0.35)
    expect(sd).toBeLessThan(0.65)
  })

  it('spreads particles apart: the repulsive term vanishes for one particle and acts for two', () => {
    const one = svgdDirection(new Float64Array([0]), new Float64Array([0]), 1, 1)
    expect(Array.from(one)).toEqual([0])
    const two = svgdDirection(new Float64Array([-0.1, 0.1]), new Float64Array([0, 0]), 1, 1)
    expect(two[0]).toBeLessThan(0)
    expect(two[1]).toBeGreaterThan(0)
    expect(() => medianBandwidth(new Float64Array([1]), 1)).toThrow()
    expect(() => svgd(standardNormal(1), [[1]])).toThrow()
  })

  it('settles on the target mean in two dimensions', () => {
    const t = gaussian2([1, -1], [1, 1], 0)
    const start = Array.from({ length: 30 }, (_, i) => [Math.cos(i), Math.sin(i)])
    const r = svgd(t, start, { stepSize: 0.2, maxIter: 800 })
    const coord = (k: number): Float64Array =>
      Float64Array.from({ length: 30 }, (_, i) => r.particles.data[2 * i + k] ?? 0)
    expect(moments(coord(0)).mean).toBeCloseTo(1, 1)
    expect(moments(coord(1)).mean).toBeCloseTo(-1, 1)
  })
})
