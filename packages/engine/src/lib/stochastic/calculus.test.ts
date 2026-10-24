import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { grad } from '../autodiff/index.ts'
import { meanOf, varianceOf } from '../base/index.ts'
import { ndtr } from '../numerics/special/index.ts'
import { key } from '../random/index.ts'
import {
  add,
  exp,
  item,
  maximum,
  mean,
  mul,
  sin,
  square,
  sub,
  sum,
  tensor,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { brownianMotion, quadraticVariation } from './brownian.ts'
import {
  blackScholes,
  feynmanKac,
  girsanovWeights,
  itoIntegral,
  itoIsometry,
  itoLemma,
  stratonovichIntegral,
  timeIntegral,
  weightedExpectation,
} from './calculus.ts'
import { timeGrid, valuesAt } from './paths.ts'
import type { Sde } from './sde.ts'

const w = brownianMotion(key(1), timeGrid(1, 200), { paths: 2000 })
const wT = toFlat(valuesAt(w, -1))

describe('stochastic integrals', () => {
  it('int W dW is (W_T^2 - [W]_T) / 2 exactly and int W o dW is W_T^2 / 2 exactly', () => {
    const ito = toFlat(itoIntegral(w.values, w))
    const strat = toFlat(stratonovichIntegral(w.values, w))
    const qv = toFlat(quadraticVariation(w))
    for (let j = 0; j < 20; j++) {
      expect(ito[j]).toBeCloseTo(((wT[j] ?? 0) ** 2 - (qv[j] ?? 0)) / 2, 12)
      expect(strat[j]).toBeCloseTo((wT[j] ?? 0) ** 2 / 2, 12)
    }
  })

  it('has the moments of the Ito integral: mean 0 and variance T^2 / 2 for int W dW', () => {
    const ito = toFlat(itoIntegral(w.values, w))
    expect(Math.abs(meanOf(ito))).toBeLessThan(4 * Math.sqrt(0.5 / 2000))
    // Var of the sample variance for this non-Gaussian integral; E[I^4] = 7/4 + ... use a loose band
    expect(Math.abs(varianceOf(ito) - 0.5)).toBeLessThan(0.08)
    // the Stratonovich integral has mean T / 2
    expect(Math.abs(meanOf(toFlat(stratonovichIntegral(w.values, w))) - 0.5)).toBeLessThan(0.07)
  })

  it('satisfies the Ito isometry for int t dW (variance T^3 / 3)', () => {
    const t = add(zeros(w.values.shape), tensor(Array.from(w.times)))
    const c = itoIsometry(t, w)
    expect(Math.abs(c.lhs.value - 1 / 3)).toBeLessThan(4 * (c.lhs.standardError ?? 0))
    expect(c.rhs.value).toBeCloseTo((1 / 3) * (1 - 0.5 / 200) ** 1, 2)
  })

  it('computes running integrals and time integrals', () => {
    const run = itoIntegral([[0, 1, 3]], [[0, 1, 3]], { cumulative: true })
    expect(toFlat(run)).toEqual([0, 0, 2])
    expect(toFlat(timeIntegral([[2, 2, 9]], [0, 1, 3]))).toEqual([6])
    expect(() => itoIntegral([[1, 2, 3, 4]], [[0, 1]])).toThrow()
    expect(() => stratonovichIntegral([[1]], [[0, 1]])).toThrow()
  })

  it('checks Ito lemma on f(t, x) = exp(t / 2) sin(x) along Brownian paths', () => {
    const fine = brownianMotion(key(2), timeGrid(1, 4000), { paths: 3 })
    const coarse = brownianMotion(key(2), timeGrid(1, 250), { paths: 3 })
    const f = (t: Value, x: Value): Value => mul(exp(mul(t, 0.5)), sin(x))
    const a = itoLemma(f, fine)
    const b = itoLemma(f, coarse)
    expect(a.residual).toBeLessThan(0.05)
    expect(a.residual).toBeLessThan(b.residual)
    // f(x) = x^2 is exact: d(W^2) = 2 W dW + dt with the realised quadratic variation
    expect(itoLemma((_t, x) => square(x), coarse).residual).toBeLessThan(1e-12)
  })

  it('differentiates integrals in the integrand', () => {
    expectConformant((h: Value) => itoIntegral(h, [[0, 0.3, -0.1, 0.4]]), [tensor([[1, 2, 3, 4]])])
    expectConformant((h: Value) => stratonovichIntegral(h, [[0, 0.3, -0.1, 0.4]]), [tensor([[1, 2, 3, 4]])])
  })
})

describe('Girsanov reweighting', () => {
  it('turns P-Brownian paths into Brownian motion with drift theta', () => {
    const z = girsanovWeights(0.7, w)
    // E_Q[W_T] = theta T and E_Q[W_T^2] = 1 + theta^2
    const m = weightedExpectation(wT, z)
    expect(Math.abs(m.value - 0.7)).toBeLessThan(4 * (m.standardError ?? 0))
    const s = weightedExpectation(
      wT.map((x) => x * x),
      z,
      { selfNormalised: true },
    )
    expect(Math.abs(s.value - 1.49)).toBeLessThan(4 * (s.standardError ?? 0) + 1e-3)
    // a constant theta gives the closed form exp(theta W_T - theta^2 T / 2)
    expect(toFlat(z)[0]).toBeCloseTo(Math.exp(0.7 * (wT[0] ?? 0) - 0.245), 10)
  })

  it('accepts adapted drifts on the grid and differentiates in theta', () => {
    const theta = mul(w.values, 0)
    expect(toFlat(girsanovWeights(theta, w)).every((v) => Math.abs(v - 1) < 1e-15)).toBe(true)
    const small = brownianMotion(key(3), timeGrid(1, 4), { paths: 3 })
    expectConformant((th: Value) => girsanovWeights(th, small), [tensor(0.3)])
    expect(() => weightedExpectation([1], [1, 2])).toThrow()
  })
})

describe('Feynman-Kac and Black-Scholes', () => {
  const r = 0.05
  const sigma = 0.2
  const gbm: Sde = { drift: (_t, y) => mul(y, r), diffusion: (_t, y) => mul(y, sigma) }

  it('prices a call by Monte Carlo within four standard errors of Black-Scholes', () => {
    const call = feynmanKac(
      key(4),
      {
        sde: gbm,
        payoff: (y) => sum(maximum(sub(y, 100), 0), 1),
        potential: (_t, y) => add(zeros([y.shape[0] ?? 1]), r),
      },
      [100],
      1,
      { steps: 20, paths: 8000, method: 'milstein' },
    )
    const exact = item(blackScholes(100, 100, 1, r, sigma))
    expect(Math.abs(call.estimate.value - exact)).toBeLessThan(4 * (call.estimate.standardError ?? 0) + 0.1)
  })

  it('handles a source term: u = E[int k ds] for k = 1 and no potential is T', () => {
    const zero: Sde = { drift: (_t, y) => zeros(y.shape), diffusion: (_t, y) => zeros(y.shape) }
    const u = feynmanKac(
      key(5),
      { sde: zero, payoff: (y) => zeros([y.shape[0] ?? 1]), source: (_t, y) => add(zeros([y.shape[0] ?? 1]), 1) },
      [0],
      2,
      { steps: 10, paths: 4 },
    )
    expect(u.estimate.value).toBeCloseTo(2, 12)
  })

  it('has Greeks by autodiff that match the closed forms', () => {
    const d1 = (Math.log(1) + (r + sigma ** 2 / 2) * 1) / sigma
    const delta = item(grad((s: Value) => blackScholes(s, 100, 1, r, sigma))(tensor(100)))
    expect(delta).toBeCloseTo(item(ndtr(d1)), 10)
    const vega = item(grad((v: Value) => blackScholes(100, 100, 1, r, v))(tensor(sigma)))
    expect(vega).toBeCloseTo((100 * Math.exp(-(d1 ** 2) / 2)) / Math.sqrt(2 * Math.PI), 9)
    // put-call parity
    const c = item(blackScholes(95, 100, 0.5, r, 0.3))
    const p = item(blackScholes(95, 100, 0.5, r, 0.3, { kind: 'put' }))
    expect(c - p).toBeCloseTo(95 - 100 * Math.exp(-r * 0.5), 12)
  })

  it('differentiates the Monte Carlo price pathwise (the delta)', () => {
    const price = (s0: Value): Value =>
      feynmanKac(key(6), { sde: gbm, payoff: (y) => sum(maximum(sub(y, 100), 0), 1) }, s0, 1, {
        steps: 10,
        paths: 4000,
      }).value
    const delta = item(grad((s: Value) => mean(price(s)))(tensor([100]))) * Math.exp(-r)
    expect(Math.abs(delta - item(ndtr((r + sigma ** 2 / 2) / sigma)))).toBeLessThan(0.05)
  })
})
