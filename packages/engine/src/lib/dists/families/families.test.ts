import { describe, expect, it } from 'vitest'
import { checkPrimitive, jvp } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import { item, mean, square, sub, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as F from './index.ts'

describe('links', () => {
  const links = [
    F.identity(),
    F.log(),
    F.logit(),
    F.probit(),
    F.cauchit(),
    F.cloglog(),
    F.loglog(),
    F.inverse(),
    F.sqrt(),
  ]
  it('invert and agree with their derivatives', () => {
    const mu = tensor([0.15, 0.4, 0.8])
    for (const l of links) {
      const back = toFlat(l.inverse(l.link(mu)))
      for (const [i, v] of back.entries()) expect(v).toBeCloseTo(toFlat(mu)[i] ?? 0, 12)
      const d = toFlat(jvp((m: Value) => l.link(m), mu, tensor([1, 1, 1])).tangent)
      for (const [i, v] of toFlat(l.derivative(mu)).entries()) expect(v).toBeCloseTo(d[i] ?? 0, 8)
      const d2 = toFlat(jvp((m: Value) => l.derivative(m), mu, tensor([1, 1, 1])).tangent)
      for (const [i, v] of toFlat(l.secondDerivative(mu)).entries())
        expect(Math.abs(v - (d2[i] ?? 0))).toBeLessThan(1e-7 * (1 + Math.abs(d2[i] ?? 0)))
    }
    expect(F.linkByName('cloglog').name).toBe('cloglog')
    expect(() => F.power(0)).toThrow(/log link/)
  })
})

describe('dispersion families', () => {
  it("IRLS weights are 1 / (g'(mu)^2 V(mu)): mu for the canonical Poisson link", () => {
    const mu = tensor([0.5, 2, 4])
    expect(toFlat(F.poisson().irlsWeights(mu))).toEqual([
      expect.closeTo(0.5, 14),
      expect.closeTo(2, 14),
      expect.closeTo(4, 14),
    ])
  })

  it('family log-likelihoods differentiate in the mean', () => {
    const y = tensor([0.3, 1.2, 2.5])
    for (const fam of [F.gamma(), F.inverseGaussian(), F.tweedie(1.4)]) {
      const f = (mu: Value): Value => fam.logLikelihoodObs(y, mu, 0.7)
      expect(checkPrimitive(f, [tensor([0.8, 1.1, 1.9])], { rtol: 1e-5, atol: 1e-8 }).failures).toEqual([])
    }
  })

  it('the Tweedie distribution samples its mean and variance; its CDF has the atom at zero', () => {
    const d = F.tweedieDistribution(1.5, 0.8, 1.4)
    const x = d.sample(key(1), [20000]) as Tensor
    expect(item(mean(x))).toBeCloseTo(1.5, 1)
    const v = item(mean(square(sub(x, item(mean(x))))))
    expect(v / item(d.variance())).toBeCloseTo(1, 1)
    const zeros = toFlat(x).filter((value) => value === 0).length / 20000
    expect(zeros).toBeCloseTo(item(d.prob(0)), 2)
    expect(item(d.cdf(d.ppf(0.7)))).toBeCloseTo(0.7, 10)
    expect(item(d.ppf(item(d.prob(0)) / 2))).toBe(0)
  })

  it('distributions match the family moments', () => {
    const g = F.gamma().distribution(2, 0.5)
    expect(item(g.mean())).toBeCloseTo(2, 14)
    expect(item(g.variance())).toBeCloseTo(0.5 * 4, 14)
    const nb = F.negativeBinomial(0.5).distribution(3)
    expect(item(nb.variance())).toBeCloseTo(item(F.negativeBinomial(0.5).variance(3)), 12)
  })
})

describe('distributional families', () => {
  it('log-likelihoods differentiate in every predictor', () => {
    const y = tensor([0.4, 1.3, 2.2])
    const fam = F.gamlssGamma()
    const f = (a: Value, b: Value): Value => fam.logLikelihood(y, [a, b])
    expect(
      checkPrimitive(f, [tensor([0.1, 0.2, 0.3]), tensor([-0.5, -0.2, 0])], { rtol: 1e-5, atol: 1e-8 }).failures,
    ).toEqual([])
    expect(() => fam.fromPredictors([0])).toThrow(/needs 2 parameters/)
  })

  it('ordinal families give probabilities that sum to one for every link', () => {
    for (const link of ['logit', 'probit', 'cloglog', 'loglog', 'cauchit'] as const) {
      const d = F.ordinalFamily(link).distribution(0.3, [-1, 0.5])
      const total = toFlat(d.prob(tensor([0, 1, 2]))).reduce((s, v) => s + v, 0)
      expect(total).toBeCloseTo(1, 13)
    }
  })
})
