import { describe, expect, it } from 'vitest'
import { digamma } from '../numerics/special/index.ts'
import { key } from '../random/index.ts'
import { add, item, mean, mul, sub, sum, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as D from './index.ts'

const families: readonly (readonly [string, D.Distribution])[] = [
  ['normal', D.normal(tensor([0.5, -1]), tensor([1.2, 0.7]))],
  ['logNormal', D.logNormal(0.3, 0.6)],
  ['halfNormal', D.halfNormal(1.4)],
  ['exponential', D.exponential(2)],
  ['gamma', D.gamma(tensor([2.5, 0.7]), 1.5)],
  ['chi2', D.chi2(4)],
  ['erlang', D.erlang(3, 2)],
  ['inverseGamma', D.inverseGamma(3, 2)],
  ['inverseGaussian', D.inverseGaussian(1.5, 2)],
  ['beta', D.beta(2, 3)],
  ['weibull', D.weibull(1.7, 2)],
  ['pareto', D.pareto(3, 1.5)],
  ['laplace', D.laplace(0.5, 1.2)],
  ['bernoulli', D.bernoulli({ probs: tensor([0.2, 0.7]) })],
  ['binomial', D.binomial(10, 0.3)],
  ['poisson', D.poisson(3.5)],
  ['geometric', D.geometric(0.3)],
  ['negativeBinomial', D.negativeBinomial(2.5, 0.4)],
  ['categorical', D.categorical({ probs: [0.2, 0.5, 0.3] })],
  ['oneHotCategorical', D.oneHotCategorical({ probs: [0.2, 0.5, 0.3] })],
  ['multinomial', D.multinomial(5, [0.2, 0.5, 0.3])],
  ['dirichlet', D.dirichlet([1.5, 2, 0.8])],
  [
    'multivariateNormal',
    D.multivariateNormal([0.5, -1], {
      covariance: [
        [1, 0.3],
        [0.3, 0.8],
      ],
    }),
  ],
  [
    'wishart',
    D.wishart(5, {
      covariance: [
        [1, 0.2],
        [0.2, 0.6],
      ],
    }),
  ],
  ['vonMises', D.vonMises(0.4, 2.5)],
]

const dotParts = (eta: readonly Value[], t: readonly Value[], ranks: readonly number[]): Value => {
  let total: Value = tensor(0)
  for (const [i, e] of eta.entries()) {
    let term = mul(e, t[i] as Value)
    for (let r = 0; r < (ranks[i] ?? 0); r++) term = sum(term, -1)
    total = add(total, term)
  }
  return total
}

describe('exponential-family forms', () => {
  for (const [name, d] of families) {
    it(`${name}: h(x) exp(eta . T(x) - A(eta)) is the density, and fromNatural round-trips`, () => {
      const ef = D.exponentialFamily(d)
      const x = d.sample(key(1), [5])
      const lp = add(
        sub(dotParts(ef.natural, ef.sufficientStatistics(x), ef.partRanks), ef.logPartition()),
        ef.logBaseMeasure(x),
      )
      const want = toFlat(d.logProb(x))
      for (const [i, v] of toFlat(lp).entries()) expect(v).toBeCloseTo(want[i] ?? Number.NaN, 9)
      const back = toFlat(ef.fromNatural(ef.natural).logProb(x))
      for (const [i, v] of back.entries()) expect(v).toBeCloseTo(want[i] ?? Number.NaN, 9)
    })
  }

  it('mean parameters are the expected sufficient statistics', () => {
    const ef = D.exponentialFamily(D.normal(0.5, 1.2))
    const [m1, m2] = ef.meanParameters() as [Value, Value]
    expect(item(m1)).toBeCloseTo(0.5, 12)
    expect(item(m2)).toBeCloseTo(0.25 + 1.44, 12)
    const g = D.exponentialFamily(D.gamma(2.5, 1.5)).meanParameters()
    // E[X] = alpha / beta.
    expect(item(g[1] as Value)).toBeCloseTo(2.5 / 1.5, 12)
    const d = D.exponentialFamily(D.dirichlet([1, 2, 3])).meanParameters()[0] as Value
    // E[log X_i] = psi(alpha_i) - psi(alpha_0).
    expect(toFlat(d)[0]).toBeCloseTo(item(digamma(1)) - item(digamma(6)), 12)
  })

  it('the Fisher information is the covariance of the sufficient statistics', () => {
    expect(toFlat(D.exponentialFamily(D.bernoulli({ probs: 0.3 })).fisherInformation())[0]).toBeCloseTo(0.21, 12)
    const f = toFlat(D.exponentialFamily(D.normal(0.5, 1.2)).fisherInformation())
    // Var[X] = s^2, Cov[X, X^2] = 2 mu s^2, Var[X^2] = 2 s^4 + 4 mu^2 s^2.
    const s2 = 1.44
    expect(f[0]).toBeCloseTo(s2, 10)
    expect(f[1]).toBeCloseTo(2 * 0.5 * s2, 10)
    expect(f[3]).toBeCloseTo(2 * s2 * s2 + 4 * 0.25 * s2, 10)
    // A Monte Carlo check for the gamma.
    const g = D.gamma(2.5, 1.5)
    const x = g.sample(key(2), [40000]) as Tensor
    const t0 = D.exponentialFamily(g).sufficientStatistics(x)[0] as Value
    const centred = sub(t0, item(mean(t0)))
    expect(toFlat(D.exponentialFamily(g).fisherInformation())[0]).toBeCloseTo(item(mean(mul(centred, centred))), 2)
  })

  it('refuses families that are not exponential families', () => {
    expect(() => D.exponentialFamily(D.studentT(3))).toThrow(/no exponential-family form/)
  })
})
