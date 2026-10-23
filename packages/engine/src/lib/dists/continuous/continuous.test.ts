import { describe, expect, it } from 'vitest'
import { checkPrimitive, jvp } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import { item, mean, square, sub, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { integrate, numericMoments } from '../_numeric.ts'
import * as D from '../index.ts'

const N = 20000

describe('sampling matches the analytic moments', () => {
  const families: readonly (readonly [string, D.Distribution])[] = [
    ['skewNormal', D.skewNormal(0.5, 2, 3)],
    ['skewNormal (left)', D.skewNormal(-1, 0.7, -1.5)],
    ['johnsonSU', D.johnsonSU(1.2, 2, 0.5, 1.5)],
    ['sinhArcsinh', D.sinhArcsinh(0.5, 1.2, 0.4, 0.8)],
    ['logitNormal', D.logitNormal(0.3, 0.8)],
    ['kumaraswamy', D.kumaraswamy(2, 5)],
    ['triangular', D.triangular(-2, -1.5, 4)],
    ['erlang', D.erlang(4, 0.5)],
    ['inverseGaussian', D.inverseGaussian(2, 4)],
    ['generalisedExtremeValue', D.generalisedExtremeValue(0.5, 2, -0.2)],
    ['generalisedPareto', D.generalisedPareto(0.5, 2, 0.1)],
    ['frechet', D.frechet(5, -0.5, 1.5)],
    ['fDistribution', D.fDistribution(10, 20)],
    ['noncentralChi2', D.noncentralChi2(3, 2)],
    ['noncentralT', D.noncentralT(12, 1.5)],
  ]
  for (const [name, d] of families) {
    it(name, () => {
      const x = d.sample(key(41), [N])
      const m = item(mean(x))
      const v = item(mean(square(sub(x, m))))
      const sd = Math.sqrt(item(d.variance()))
      expect(Math.abs(m - item(d.mean()))).toBeLessThan((5 * sd) / Math.sqrt(N))
      expect(Math.abs(v / item(d.variance()) - 1)).toBeLessThan(0.1)
    })
  }

  it('the studentised range samples its mean', () => {
    const d = D.studentisedRange(3, 10)
    const x = d.sample(key(42), [4000])
    const sd = Math.sqrt(item(d.variance()))
    expect(Math.abs(item(mean(x)) - item(d.mean()))).toBeLessThan((5 * sd) / Math.sqrt(4000))
    expect(d.hasRsample).toBe(false)
  })
})

describe('pathwise gradients', () => {
  // A reparameterised sample keeps its CDF value fixed: dx/dtheta = d ppf(u; theta) / dtheta at u = F(x).
  const quantileDerivative = (make: (p: number) => D.Distribution, p: number, x: Tensor): number[] => {
    const u = make(p).cdf(x)
    const h = 1e-5 * Math.max(1, Math.abs(p))
    return toFlat(sub(make(p + h).ppf(u), make(p - h).ppf(u))).map((v) => v / (2 * h))
  }
  const cases: readonly (readonly [string, (p: Value) => D.Distribution, number])[] = [
    ['inverseGaussian mean (implicit)', (p) => D.inverseGaussian(p, 3), 1.5],
    ['inverseGaussian shape (implicit)', (p) => D.inverseGaussian(1.5, p), 3],
    ['noncentralChi2 noncentrality (implicit)', (p) => D.noncentralChi2(3, p), 2],
    ['kumaraswamy a (inverse CDF)', (p) => D.kumaraswamy(p, 3), 2],
    ['generalisedPareto shape (inverse CDF)', (p) => D.generalisedPareto(0, 1, p), 0.2],
    ['generalisedExtremeValue shape (inverse CDF)', (p) => D.generalisedExtremeValue(0, 1, p), -0.2],
    ['johnsonSU a (transform)', (p) => D.johnsonSU(p, 2), 0.5],
    ['triangular mode (inverse CDF)', (p) => D.triangular(0, p, 3), 1],
  ]
  for (const [name, make, p] of cases) {
    it(name, () => {
      const k = key(43)
      const x = make(tensor(p)).sample(k, [16]) as Tensor
      const got = toFlat(jvp((v: Value) => make(v).rsample(k, [16]), tensor(p), tensor(1)).tangent)
      const want = quantileDerivative((v) => make(tensor(v)), p, x)
      for (const [i, g] of got.entries()) expect(g).toBeCloseTo(want[i] ?? Number.NaN, 4)
    })
  }

  it('the skew normal is reparameterised: d/dalpha E[X] = sqrt(2 / pi) / (1 + alpha^2)^(3/2)', () => {
    const a = 1.5
    const g = jvp((v: Value) => mean(D.skewNormal(0, 1, v).rsample(key(44), [N])), tensor(a), tensor(1)).tangent
    expect(item(g)).toBeCloseTo(Math.sqrt(2 / Math.PI) / (1 + a * a) ** 1.5, 2)
  })

  it('numerical quantiles differentiate as the implicit function', () => {
    const q = tensor([0.1, 0.5, 0.9])
    const make = (a: Value): D.Distribution => D.skewNormal(0, 1, a)
    const got = toFlat(jvp((a: Value) => make(a).ppf(q), tensor(2), tensor(1)).tangent)
    const h = 1e-5
    const want = toFlat(sub(make(tensor(2 + h)).ppf(q), make(tensor(2 - h)).ppf(q))).map((v) => v / (2 * h))
    for (const [i, g] of got.entries()) expect(g).toBeCloseTo(want[i] ?? Number.NaN, 6)
  })
})

describe('densities differentiate in their parameters', () => {
  const x = tensor([0.3, 1.2, 2.5])
  const unit = tensor([0.2, 0.5, 0.9])
  const fs: readonly (readonly [string, (a: Value, b: Value) => Value, readonly [number, number]])[] = [
    ['skewNormal cdf', (a, b) => D.skewNormal(a, 1.3, b).cdf(x), [0.4, 2]],
    ['johnsonSU logProb', (a, b) => D.johnsonSU(a, b, 0.2, 1.1).logProb(x), [0.7, 1.4]],
    ['sinhArcsinh logProb', (a, b) => D.sinhArcsinh(0.2, 1.1, a, b).logProb(x), [0.4, 1.3]],
    ['kumaraswamy logProb', (a, b) => D.kumaraswamy(a, b).logProb(unit), [1.7, 2.3]],
    ['triangular cdf', (a, b) => D.triangular(a, 1, b).cdf(x), [-0.5, 3]],
    ['logitNormal logProb', (a, b) => D.logitNormal(a, b).logProb(unit), [0.3, 0.9]],
    ['generalisedExtremeValue logProb', (a, b) => D.generalisedExtremeValue(a, 1.5, b).logProb(x), [0.2, 0.15]],
    ['generalisedPareto cdf', (a, b) => D.generalisedPareto(0, a, b).cdf(x), [1.4, 0.3]],
    ['frechet logProb', (a, b) => D.frechet(a, -0.5, b).logProb(x), [2.5, 1.2]],
    ['fDistribution logProb', (a, b) => D.fDistribution(a, b).logProb(x), [4, 7]],
    ['inverseGaussian cdf', (a, b) => D.inverseGaussian(a, b).cdf(x), [1.5, 2.5]],
    ['noncentralChi2 cdf', (a, b) => D.noncentralChi2(a, b).cdf(x), [2.5, 1.5]],
    ['noncentralT logProb', (a, b) => D.noncentralT(a, b).logProb(x), [6, 0.8]],
  ]
  for (const [name, f, [a, b]] of fs) {
    it(name, () => {
      expect(checkPrimitive(f, [tensor(a), tensor(b)], { rtol: 1e-4, atol: 1e-7 }).failures).toEqual([])
    })
  }
})

describe('families without a reference library', () => {
  it('the sinh-arcsinh reduces to the normal and its moments match quadrature', () => {
    const s = D.sinhArcsinh(0.5, 2, 0, 1)
    const n = D.normal(0.5, 2)
    expect(item(s.logProb(1.3))).toBeCloseTo(item(n.logProb(1.3)), 12)
    expect(item(s.cdf(-0.4))).toBeCloseTo(item(n.cdf(-0.4)), 12)
    const d = D.sinhArcsinh(0.5, 1.2, 0.4, 0.8)
    const m = numericMoments(d, { kind: 'real', center: 0.5, scale: 1.2 })
    expect(item(d.mean())).toBeCloseTo(item(m.mean), 9)
    expect(item(d.variance())).toBeCloseTo(item(m.variance), 9)
    // The quantile function inverts the CDF.
    expect(toFlat(d.cdf(d.ppf([0.1, 0.5, 0.9])))).toEqual(
      expect.arrayContaining([expect.closeTo(0.1, 12), expect.closeTo(0.5, 12), expect.closeTo(0.9, 12)]),
    )
  })

  it('the logit-normal density integrates to one and its moments match sampling', () => {
    const d = D.logitNormal(0.3, 0.8)
    const mass = integrate((u) => d.prob(u), { kind: 'finite', low: 0, high: 1 }, 0)
    expect(item(mass)).toBeCloseTo(1, 10)
    expect(item(d.cdf(item(d.ppf(0.3))))).toBeCloseTo(0.3, 12)
    // A bimodal logit-normal has its (symmetric pair of) modes near the ends.
    const m = item(D.logitNormal(0, 3).mode())
    expect(Math.min(m, 1 - m)).toBeLessThan(0.1)
  })

  it('numerical modes are stationary points of the density', () => {
    for (const d of [D.skewNormal(0.5, 2, 3), D.johnsonSU(1.2, 2, 0.5, 1.5), D.noncentralT(8, 1.2)]) {
      const m = d.mode()
      const score = jvp((x: Value) => d.logProb(x), m, tensor(1)).tangent
      expect(Math.abs(item(score))).toBeLessThan(1e-8)
    }
  })

  it('validates parameters', () => {
    expect(() => D.normal(0, -1)).toThrow(/scale must be positive/)
    expect(() => D.gamma(0, 1)).toThrow(/concentration/)
    expect(() => D.erlang(2.5, 1)).toThrow(/positiveInteger/)
    expect(() => D.studentisedRange(1, 4)).toThrow(/at least 2/)
  })
})
