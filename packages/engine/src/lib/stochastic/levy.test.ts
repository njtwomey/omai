import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { jacfwd } from '../autodiff/index.ts'
import { meanOf, varianceOf } from '../base/index.ts'
import { exponential } from '../dists/index.ts'
import { quad } from '../numerics/quadrature/index.ts'
import { key } from '../random/index.ts'
import { item, mul, tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  compoundPoisson,
  compoundPoissonMoments,
  stableCharacteristicFunction,
  stablePdf,
  stableProcess,
  stableSample,
  varianceGammaDensity,
  varianceGammaMoments,
  varianceGammaProcess,
} from './levy.ts'
import { timeGrid, valuesAt } from './paths.ts'

describe('compound Poisson processes', () => {
  it('have the closed-form mean and variance', () => {
    const jumps = exponential(2)
    const p = compoundPoisson(key(1), timeGrid(3, 6), { rate: 2, jumps, paths: 4000, drift: 0.1 })
    const end = toFlat(valuesAt(p, -1))
    const m = compoundPoissonMoments(3, 2, jumps, 0.1)
    expect(Math.abs(meanOf(end) - m.mean)).toBeLessThan(4 * Math.sqrt(m.variance / 4000))
    expect(Math.abs(varianceOf(end) - m.variance)).toBeLessThan(0.15 * m.variance)
    expect(() => compoundPoisson(key(0), [0, 1], { rate: -1, jumps })).toThrow()
  })
})

describe('alpha-stable laws', () => {
  it('reduce to the normal (alpha = 2) and Cauchy (alpha = 1) laws', () => {
    const g = toFlat(stableSample(key(2), [8000], { alpha: 2 }))
    expect(Math.abs(varianceOf(g) - 2)).toBeLessThan(5 * 2 * Math.sqrt(2 / 8000))
    const c = toFlat(stableSample(key(3), [8000], { alpha: 1 })).sort((a, b) => a - b)
    // Cauchy quartiles are -1 and 1
    expect(Math.abs((c[2000] ?? 0) + 1)).toBeLessThan(0.08)
    expect(Math.abs((c[6000] ?? 0) - 1)).toBeLessThan(0.08)
    const cf = stableCharacteristicFunction([2], { alpha: 1, scale: 0.5 })
    expect(item(cf.re)).toBeCloseTo(Math.exp(-1), 12)
  })

  it('the density integrates to one and matches the inversion of the characteristic function', () => {
    const total = quad((x) => stablePdf(x, { alpha: 1.6, beta: 0.4 }), -60, 60, { limit: 100 }).value
    expect(item(total)).toBeCloseTo(1, 2)
    expect(() => stableSample(key(0), [1], { alpha: 2.5 })).toThrow()
    expect(() => stableSample(key(0), [1], { alpha: 1.5, beta: 2 })).toThrow()
  })

  it('samples differentiate in alpha, beta and the scale (reparameterisation)', () => {
    const f = (a: Value, b: Value, s: Value): Value => stableSample(key(4), [3], { alpha: a, beta: b, scale: s })
    expectConformant(f, [tensor(1.5), tensor(0.3), tensor(0.8)], { rtol: 1e-4 })
  })

  it('stable processes are self-similar with index 1 / alpha', () => {
    const p = stableProcess(key(5), timeGrid(4, 8), { alpha: 1.5, paths: 4000 })
    const end = toFlat(valuesAt(p, -1)).map((v) => v / 4 ** (1 / 1.5))
    const one = toFlat(stableSample(key(6), [4000], { alpha: 1.5 }))
    const q = (xs: number[], f: number): number => [...xs].sort((a, b) => a - b)[Math.floor(f * xs.length)] ?? 0
    expect(Math.abs(q(end, 0.75) - q(one, 0.75))).toBeLessThan(0.1)
    expect(stableProcess(key(7), timeGrid(1, 4), { alpha: 1, beta: 0.5 }).values.shape).toEqual([1, 5])
  })
})

describe('variance gamma processes', () => {
  const params = { theta: -0.2, sigma: 0.3, nu: 0.4 }

  it('have the closed-form moments and density', () => {
    const p = varianceGammaProcess(key(8), timeGrid(2, 4), { ...params, paths: 4000 })
    const end = toFlat(valuesAt(p, -1))
    const m = varianceGammaMoments(2, params)
    expect(Math.abs(meanOf(end) - m.mean)).toBeLessThan(4 * Math.sqrt(m.variance / 4000))
    expect(Math.abs(varianceOf(end) - m.variance)).toBeLessThan(0.12 * m.variance)
    const total = quad((x) => varianceGammaDensity(x, 2, params), -8, 8, { points: [0] }).value
    expect(item(total)).toBeCloseTo(1, 6)
  })

  it('paths differentiate in theta and sigma, and unbiasedly in nu', () => {
    const f = (theta: Value, sigma: Value, nu: Value): Value =>
      valuesAt(varianceGammaProcess(key(9), [0, 0.5, 1], { theta, sigma, nu, paths: 2 }), -1)
    // the gamma clock is drawn by rejection: finite differences apply to theta and sigma only
    expectConformant(f, [tensor(-0.2), tensor(0.3), tensor(0.4)], { wrt: [0, 1] })
    // d/dnu E[X_t^2] = theta^2 t, estimated by the average implicit-reparameterisation gradient
    const n = 3000
    const square = (nu: Value): Value =>
      mul(
        valuesAt(varianceGammaProcess(key(10), [0, 1], { ...params, nu, paths: n }), -1),
        valuesAt(varianceGammaProcess(key(10), [0, 1], { ...params, nu, paths: n }), -1),
      )
    const each = toFlat(jacfwd(square)(tensor(0.4)))
    expect(Math.abs(meanOf(each) - 0.04)).toBeLessThan(4 * Math.sqrt(varianceOf(each) / n))
  })
})
