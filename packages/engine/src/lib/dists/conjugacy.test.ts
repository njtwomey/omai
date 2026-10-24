import { describe, expect, it } from 'vitest'
import { asTensor, item, sqrt, sum, type TensorLike, type Value } from '../tensor/index.ts'
import * as D from './index.ts'

interface Case {
  readonly likelihood: string
  readonly parameter: string
  readonly prior: D.Distribution
  readonly data: TensorLike
  readonly fixed?: Readonly<Record<string, TensorLike>>
  /** The likelihood at a parameter value. */
  readonly at: (theta: Value) => D.Distribution
  /** Two parameter values in the support. */
  readonly thetas: readonly [TensorLike, TensorLike]
}

const spd = (a: number, b: number, c: number): number[][] => [
  [a, b],
  [b, c],
]

const cases: readonly Case[] = [
  {
    likelihood: 'bernoulli',
    parameter: 'probs',
    prior: D.beta(2, 3),
    data: [1, 0, 1, 1],
    at: (p) => D.bernoulli({ probs: p }),
    thetas: [0.3, 0.7],
  },
  {
    likelihood: 'binomial',
    parameter: 'probs',
    prior: D.beta(2, 3),
    data: [3, 5, 2],
    fixed: { n: 8 },
    at: (p) => D.binomial(8, p),
    thetas: [0.3, 0.6],
  },
  {
    likelihood: 'geometric',
    parameter: 'probs',
    prior: D.beta(1.5, 2),
    data: [1, 4, 2],
    at: (p) => D.geometric(p),
    thetas: [0.2, 0.5],
  },
  {
    likelihood: 'negativeBinomial',
    parameter: 'probs',
    prior: D.beta(1.5, 2),
    data: [1, 4, 2],
    fixed: { n: 3 },
    at: (p) => D.negativeBinomial(3, p),
    thetas: [0.2, 0.5],
  },
  {
    likelihood: 'poisson',
    parameter: 'rate',
    prior: D.gamma(2, 1),
    data: [3, 1, 4],
    at: (l) => D.poisson(l),
    thetas: [1.5, 3],
  },
  {
    likelihood: 'exponential',
    parameter: 'rate',
    prior: D.gamma(2, 1),
    data: [0.3, 1.1, 0.7],
    at: (l) => D.exponential(l),
    thetas: [0.8, 2],
  },
  {
    likelihood: 'gamma',
    parameter: 'rate',
    prior: D.gamma(2, 1),
    data: [0.3, 1.1, 0.7],
    fixed: { concentration: 2.5 },
    at: (r) => D.gamma(2.5, r),
    thetas: [0.8, 2],
  },
  {
    likelihood: 'pareto',
    parameter: 'b',
    prior: D.gamma(2, 1),
    data: [1.6, 2.4, 3.1],
    fixed: { scale: 1.5 },
    at: (b) => D.pareto(b, 1.5),
    thetas: [1.2, 2.5],
  },
  {
    likelihood: 'normal',
    parameter: 'loc',
    prior: D.normal(0, 2),
    data: [0.4, 1.3, -0.2],
    fixed: { scale: 0.7 },
    at: (m) => D.normal(m, 0.7),
    thetas: [0.1, 0.9],
  },
  {
    likelihood: 'logNormal',
    parameter: 'mu',
    prior: D.normal(0, 2),
    data: [0.4, 1.3, 2.2],
    fixed: { sigma: 0.7 },
    at: (m) => D.logNormal(m, 0.7),
    thetas: [0.1, 0.9],
  },
  {
    likelihood: 'normal',
    parameter: 'variance',
    prior: D.inverseGamma(3, 2),
    data: [0.4, 1.3, -0.2],
    fixed: { loc: 0.2 },
    at: (v) => D.normal(0.2, sqrt(v)),
    thetas: [0.5, 1.4],
  },
  {
    likelihood: 'categorical',
    parameter: 'probs',
    prior: D.dirichlet([1, 2, 1.5]),
    data: [0, 2, 2, 1],
    at: (p) => D.categorical({ probs: p }),
    thetas: [
      [0.2, 0.5, 0.3],
      [0.4, 0.4, 0.2],
    ],
  },
  {
    likelihood: 'multinomial',
    parameter: 'probs',
    prior: D.dirichlet([1, 2, 1.5]),
    data: [
      [1, 2, 0],
      [0, 1, 2],
    ],
    at: (p) => D.multinomial(3, p),
    thetas: [
      [0.2, 0.5, 0.3],
      [0.4, 0.4, 0.2],
    ],
  },
  {
    likelihood: 'multivariateNormal',
    parameter: 'loc',
    prior: D.multivariateNormal([0, 0], { covariance: spd(2, 0.3, 1) }),
    data: [
      [0.4, -0.2],
      [1.1, 0.3],
    ],
    fixed: { covariance: spd(0.5, 0.1, 0.4) },
    at: (m) => D.multivariateNormal(m, { covariance: spd(0.5, 0.1, 0.4) }),
    thetas: [
      [0.2, 0.1],
      [0.6, -0.3],
    ],
  },
  {
    likelihood: 'multivariateNormal',
    parameter: 'covariance',
    prior: D.inverseWishart(5, spd(1, 0.2, 0.8)),
    data: [
      [0.4, -0.2],
      [1.1, 0.3],
      [-0.5, 0.6],
    ],
    fixed: { loc: [0.1, 0] },
    at: (s) => D.multivariateNormal([0.1, 0], { covariance: s }),
    thetas: [spd(0.6, 0.1, 0.5), spd(1.2, -0.2, 0.9)],
  },
  {
    likelihood: 'multivariateNormal',
    parameter: 'precision',
    prior: D.wishart(5, { covariance: spd(1, 0.2, 0.8) }),
    data: [
      [0.4, -0.2],
      [1.1, 0.3],
      [-0.5, 0.6],
    ],
    fixed: { loc: [0.1, 0] },
    at: (p) => D.multivariateNormal([0.1, 0], { precision: p }),
    thetas: [spd(1.6, 0.1, 1.5), spd(0.9, -0.2, 1.2)],
  },
  {
    likelihood: 'uniform',
    parameter: 'high',
    prior: D.pareto(2, 1),
    data: [0.4, 1.7, 1.2],
    at: (h) => D.uniform(0, h),
    thetas: [2, 3.5],
  },
]

describe('conjugate posteriors', () => {
  for (const c of cases) {
    it(`${c.prior.name} for ${c.likelihood}.${c.parameter} is proportional to prior times likelihood`, () => {
      const post = D.conjugatePosterior(c.prior, c.likelihood, c.parameter, c.data, c.fixed)
      const unnormalised = (theta: TensorLike): number =>
        item(c.prior.logProb(theta)) + item(sum(c.at(asTensor(theta)).logProb(c.data)))
      const [a, b] = c.thetas
      const want = unnormalised(a) - unnormalised(b)
      const got = item(post.logProb(a)) - item(post.logProb(b))
      expect(got).toBeCloseTo(want, 9)
    })
  }

  it('looks up pairs and refuses mismatches', () => {
    expect(D.conjugatePrior('poisson', 'rate')?.prior).toBe('gamma')
    expect(D.conjugatePrior('studentT', 'df')).toBeUndefined()
    expect(() => D.conjugatePosterior(D.normal(0, 1), 'poisson', 'rate', [1])).toThrow(/gamma prior/)
    expect(() => D.conjugatePosterior(D.normal(0, 1), 'normal', 'loc', [1])).toThrow(/known scale/)
  })
})
