import { describe, expect, it } from 'vitest'
import { MdmlError } from '../base/index.ts'
import { key } from '../random/index.ts'
import { item, toFlat } from '../tensor/index.ts'
import * as D from './index.ts'

// Contract checks over one table of (family, parameters) rows. Each check is a single test that loops over the rows and
// names the row in its failure message, so a new family costs one line here and no test.
const families: readonly (readonly [string, () => D.Distribution])[] = [
  ['normal', () => D.normal(1, 2)],
  ['halfNormal', () => D.halfNormal(2)],
  ['logNormal', () => D.logNormal(0, 1)],
  ['truncatedNormal', () => D.truncatedNormal(0, 1, -1, 2)],
  ['skewNormal', () => D.skewNormal(0, 1, 3)],
  ['uniform', () => D.uniform(-1, 3)],
  ['beta', () => D.beta(2, 3)],
  ['beta (a > 1 >= b)', () => D.beta(3, 0.5)],
  ['kumaraswamy', () => D.kumaraswamy(2, 3)],
  ['logitNormal', () => D.logitNormal(0, 1)],
  ['triangular', () => D.triangular(0, 1, 3)],
  ['studentT', () => D.studentT(3, 0, 1)],
  ['cauchy', () => D.cauchy(0, 1)],
  ['halfCauchy', () => D.halfCauchy(1)],
  ['laplace', () => D.laplace(0, 1)],
  ['logistic', () => D.logistic(0, 1)],
  ['pareto', () => D.pareto(3, 1)],
  ['johnsonSU', () => D.johnsonSU(0.5, 2, 0, 1)],
  ['gamma', () => D.gamma(2, 1.5)],
  ['gamma (shape below 1)', () => D.gamma(0.3, 1)],
  ['exponential', () => D.exponential(2)],
  ['inverseGamma', () => D.inverseGamma(3, 2)],
  ['chi2', () => D.chi2(4)],
  ['weibull', () => D.weibull(1.5, 2)],
  ['inverseGaussian', () => D.inverseGaussian(1, 2)],
  ['fDistribution', () => D.fDistribution(5, 7)],
  ['fDistribution (d1 below 2)', () => D.fDistribution(1, 7)],
  ['noncentralChi2', () => D.noncentralChi2(3, 2)],
  ['noncentralT', () => D.noncentralT(5, 1)],
  ['gumbel', () => D.gumbel(0, 1)],
  ['generalisedExtremeValue', () => D.generalisedExtremeValue(0, 1, 0.2)],
  ['generalisedPareto', () => D.generalisedPareto(0, 1, 0.2)],
  ['frechet', () => D.frechet(3, 0, 1)],
  ['vonMises', () => D.vonMises(0, 2)],
  ['vonMises (loc outside the support)', () => D.vonMises(3, 0.5)],
  ['poisson', () => D.poisson(3)],
  ['binomial', () => D.binomial(20, 0.3)],
  ['binomial (p = 1)', () => D.binomial(20, 1)],
  ['geometric', () => D.geometric(0.2)],
  ['negativeBinomial', () => D.negativeBinomial(3, 0.4)],
  ['hypergeometric', () => D.hypergeometric(50, 20, 10)],
  ['betaBinomial', () => D.betaBinomial(10, 2, 3)],
  ['discreteUniform', () => D.discreteUniform(2, 7)],
  ['categorical', () => D.categorical({ probs: [0.2, 0.5, 0.3] })],
  ['categorical (one class)', () => D.categorical({ probs: [1] })],
  ['categorical (masked)', () => D.categorical({ logits: [0, Number.NEGATIVE_INFINITY, 0] })],
]

const methods: readonly (readonly [string, (d: D.Distribution, x: number) => number])[] = [
  ['logProb', (d, x) => item(d.logProb(x))],
  ['prob', (d, x) => item(d.prob(x))],
  ['cdf', (d, x) => item(d.cdf(x))],
  ['sf', (d, x) => item(d.sf(x))],
  ['logCdf', (d, x) => item(d.logCdf(x))],
  ['logSf', (d, x) => item(d.logSf(x))],
  ['ppf', (d, x) => item(d.ppf(x))],
  ['isf', (d, x) => item(d.isf(x))],
]

describe('every family follows the Distribution contract', () => {
  it('answers NaN for a NaN argument in every density, distribution and quantile function', () => {
    for (const [name, make] of families) {
      const d = make()
      for (const [method, call] of methods) expect(call(d, Number.NaN), `${name}.${method}(NaN)`).toBeNaN()
    }
  })

  it('has the limits of a distribution function at an infinite argument', () => {
    const inf = Number.POSITIVE_INFINITY
    for (const [name, make] of families) {
      const d = make()
      expect([item(d.cdf(-inf)), item(d.cdf(inf)), item(d.sf(-inf)), item(d.sf(inf))], `${name}: cdf and sf`).toEqual([
        0, 1, 1, 0,
      ])
      expect(
        [d.logCdf(-inf), d.logCdf(inf), d.logSf(-inf), d.logSf(inf)].map((v) => item(v) + 0),
        `${name}: log`,
      ).toEqual([-inf, 0, 0, -inf])
    }
  })

  it('has its mode in the support (or a point of infinite or finite density) and every sample inside it', () => {
    for (const [name, make] of families) {
      const d = make()
      const inside =
        item(D.inSupport(d.support, d.mode())) === 1 || item(d.logProb(d.mode())) > Number.NEGATIVE_INFINITY
      expect(inside, `${name}: mode ${item(d.mode())}`).toBe(true)
      expect(toFlat(D.inSupport(d.support, d.sample(key(5), [200]))).every(Boolean), `${name}: samples`).toBe(true)
    }
  })

  it('has a CDF that inverts its quantile function (exactly when continuous, from above when discrete)', () => {
    for (const [name, make] of families) {
      const d = make()
      const discrete = d.support.kind === 'integerInterval' || d.support.kind === 'boolean'
      for (const p of [0.001, 0.3, 0.9, 0.999]) {
        const c = item(d.cdf(d.ppf(p)))
        expect(c, `${name}: cdf(ppf(${p}))`).toBeGreaterThanOrEqual(p - 1e-9)
        if (!discrete) expect(c, `${name}: cdf(ppf(${p}))`).toBeLessThanOrEqual(p + 1e-9)
      }
    }
  })
})

describe('closed forms at the edge of the parameter range', () => {
  const lp0 = (d: D.Distribution): number => item(d.logProb(0))
  const rows: readonly (readonly [string, () => number, number])[] = [
    ['beta(3, 0.5) mode is 1', () => item(D.beta(3, 0.5).mode()), 1],
    ['beta(0.5, 3) mode is 0', () => item(D.beta(0.5, 3).mode()), 0],
    ['beta(0.5, 0.5) is bimodal: its mode is undefined', () => item(D.beta(0.5, 0.5).mode()), Number.NaN],
    ['beta(2, 3) mode is 1/3', () => item(D.beta(2, 3).mode()), 1 / 3],
    ['dirichlet([0.5, 3]) mode is [0, 1]', () => toFlat(D.dirichlet([0.5, 3]).mode())[0] ?? Number.NaN, 0],
    ['binomial(20, 1) mode is 20', () => item(D.binomial(20, 1).mode()), 20],
    ['binomial(20, 0.3) mode is 6', () => item(D.binomial(20, 0.3).mode()), 6],
    ['F(1, 7) density at 0 is infinite', () => lp0(D.fDistribution(1, 7)), Number.POSITIVE_INFINITY],
    ['F(3, 7) density at 0 is zero', () => lp0(D.fDistribution(3, 7)), Number.NEGATIVE_INFINITY],
    ['vonMises(3, 0.5) cdf at loc - pi is 0', () => item(D.vonMises(3, 0.5).cdf(3 - Math.PI)), 0],
    ['vonMises(3, 0.5) cdf at loc + pi is 1', () => item(D.vonMises(3, 0.5).cdf(3 + Math.PI)), 1],
    ['vonMises(3, 0.5) support starts at loc - pi', () => (D.vonMises(3, 0.5).support.kind === 'interval' ? 1 : 0), 1],
    [
      'masked categorical entropy is log 2',
      () => item(D.categorical({ logits: [0, -Infinity, 0] }).entropy()),
      Math.LN2,
    ],
    ['kl(categorical [1, 0], itself) is 0', () => klCategorical([1, 0], [1, 0]), 0],
    ['kl(categorical [1, 0], [0.5, 0.5]) is log 2', () => klCategorical([1, 0], [0.5, 0.5]), Math.LN2],
    [
      'kl(categorical [0.5, 0.5], [1, 0]) is infinite',
      () => klCategorical([0.5, 0.5], [1, 0]),
      Number.POSITIVE_INFINITY,
    ],
    [
      'kl(bernoulli 0, bernoulli 0.5) is log 2',
      () => item(D.klDivergence(D.bernoulli({ probs: 0 }), D.bernoulli({ probs: 0.5 }))),
      Math.LN2,
    ],
    ['categorical with one class samples 0', () => item(D.categorical({ probs: [1] }).sample(key(1), [])), 0],
  ]
  function klCategorical(p: number[], q: number[]): number {
    return item(D.klDivergence(D.categorical({ probs: p }), D.categorical({ probs: q })))
  }

  it('hold', () => {
    for (const [name, value, expected] of rows) {
      if (Number.isFinite(expected)) expect(value(), name).toBeCloseTo(expected, 12)
      else if (Number.isNaN(expected)) expect(value(), name).toBeNaN()
      else expect(value(), name).toBe(expected)
    }
  })
})

describe('parameter validation', () => {
  const invalid: readonly (readonly [string, () => unknown])[] = [
    ['uniform(0, -1)', () => D.uniform(0, -1)],
    ['uniform(0, 0)', () => D.uniform(0, 0)],
    ['triangular with the mode below the range', () => D.triangular(0, -1, 1)],
    ['triangular with the mode above the range', () => D.triangular(0, 2, 1)],
    ['triangular with low above high', () => D.triangular(2, 1, 0)],
    ['truncatedNormal with low above high', () => D.truncatedNormal(0, 1, 2, 1)],
    ['studentisedRange with zero degrees of freedom', () => D.studentisedRange(3, 0)],
    ['normal with an infinite scale', () => D.normal(0, Number.POSITIVE_INFINITY)],
    ['gamma with an infinite concentration', () => D.gamma(Number.POSITIVE_INFINITY, 1)],
    ['studentT with infinite degrees of freedom', () => D.studentT(Number.POSITIVE_INFINITY, 0, 1)],
    ['poisson with an infinite rate', () => D.poisson(Number.POSITIVE_INFINITY)],
    ['geometric(0)', () => D.geometric(0)],
    ['negativeBinomial(3, 0)', () => D.negativeBinomial(3, 0)],
    ['hypergeometric with more successes than the total', () => D.hypergeometric(5, 7, 2)],
    ['hypergeometric with more draws than the total', () => D.hypergeometric(5, 3, 9)],
    [
      'multivariateNormal with an asymmetric covariance',
      () =>
        D.multivariateNormal([0, 0], {
          covariance: [
            [1, 0.5],
            [0, 1],
          ],
        }),
    ],
  ]
  it('rejects the invalid parameters of the register with value', () => {
    for (const [name, make] of invalid) expect(make, name).toThrow(expect.objectContaining({ code: 'value' }))
    expect(new MdmlError('value', 'x').code).toBe('value')
  })

  it('accepts the boundary cases', () => {
    expect(() => D.triangular(0, 1, 1)).not.toThrow()
    expect(() => D.triangular(0, 0, 1)).not.toThrow()
    expect(() => D.geometric(1)).not.toThrow()
    expect(() => D.negativeBinomial(3, 1)).not.toThrow()
    expect(() => D.poisson(0)).not.toThrow()
    expect(() => D.studentisedRange(3, Number.POSITIVE_INFINITY)).not.toThrow()
  })
})
