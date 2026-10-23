import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Tensor, Value } from '../tensor/index.ts'
import * as D from './index.ts'

type Factory = (params: Tensor[]) => D.Distribution
const FAMILIES: Readonly<Record<string, Factory>> = {
  normal: ([loc, scale]) => D.normal(loc as Tensor, scale as Tensor),
  logNormal: ([mu, sigma]) => D.logNormal(mu as Tensor, sigma as Tensor),
  halfNormal: ([scale]) => D.halfNormal(scale as Tensor),
  truncatedNormal: ([loc, scale, lo, hi]) =>
    D.truncatedNormal(loc as Tensor, scale as Tensor, lo as Tensor, hi as Tensor),
  gamma: ([a, r]) => D.gamma(a as Tensor, r as Tensor),
  exponential: ([r]) => D.exponential(r as Tensor),
  beta: ([a, b]) => D.beta(a as Tensor, b as Tensor),
  uniform: ([lo, hi]) => D.uniform(lo as Tensor, hi as Tensor),
  bernoulli: ([p]) => D.bernoulli({ probs: p as Tensor }),
  studentT: ([df, loc, scale]) => D.studentT(df as Tensor, loc as Tensor, scale as Tensor),
  cauchy: ([loc, scale]) => D.cauchy(loc as Tensor, scale as Tensor),
  halfCauchy: ([scale]) => D.halfCauchy(scale as Tensor),
  laplace: ([loc, scale]) => D.laplace(loc as Tensor, scale as Tensor),
  logistic: ([loc, scale]) => D.logistic(loc as Tensor, scale as Tensor),
  pareto: ([b, xm]) => D.pareto(b as Tensor, xm as Tensor),
  gumbel: ([loc, scale]) => D.gumbel(loc as Tensor, scale as Tensor),
  inverseGamma: ([a, b]) => D.inverseGamma(a as Tensor, b as Tensor),
  chi2: ([k]) => D.chi2(k as Tensor),
  weibull: ([k, l]) => D.weibull(k as Tensor, l as Tensor),
  binomial: ([n, p]) => D.binomial(n as Tensor, p as Tensor),
  poisson: ([l]) => D.poisson(l as Tensor),
  geometric: ([p]) => D.geometric(p as Tensor),
  negativeBinomial: ([r, p]) => D.negativeBinomial(r as Tensor, p as Tensor),
  categorical: ([p]) => D.categorical({ probs: p as Tensor }),
}
const ARITY: Readonly<Record<string, number>> = {
  normal: 2,
  logNormal: 2,
  halfNormal: 1,
  truncatedNormal: 4,
  gamma: 2,
  exponential: 1,
  beta: 2,
  uniform: 2,
  bernoulli: 1,
  categorical: 1,
  studentT: 3,
  cauchy: 2,
  halfCauchy: 1,
  laplace: 2,
  logistic: 2,
  pareto: 2,
  gumbel: 2,
  inverseGamma: 2,
  chi2: 1,
  weibull: 2,
  binomial: 2,
  poisson: 1,
  geometric: 1,
  negativeBinomial: 2,
}
const MOMENTS = new Set(['mean', 'variance', 'entropy', 'mode'])

for (const file of ['continuous', 'discrete']) {
  describe(`dists/${file} against scipy.stats`, () => {
    for (const c of loadFixture(`dists/${file}`).cases) {
      it(c.name, () => {
        const [family, method] = c.fn.split('.') as [string, string]
        const inputs = c.inputs.map((x) => decode(x) as Tensor)
        const n = ARITY[family] ?? 0
        const dist = (FAMILIES[family] as Factory)(inputs.slice(0, n))
        const call = dist[method as keyof D.Distribution] as unknown as (x?: Tensor) => Value
        const got = MOMENTS.has(method) ? call() : call(inputs[n])
        expectMatches(got, c.output, c.rtol, c.atol, c.name)
      })
    }
  })
}

describe('dists/multivariate against scipy.stats', () => {
  for (const c of loadFixture('dists/multivariate').cases) {
    it(c.name, () => {
      const xs = c.inputs.map((x) => decode(x) as Tensor)
      const [family, method] = c.fn.split('.') as [string, string]
      let got: Value
      if (family === 'multivariateNormal') {
        const d = D.multivariateNormal(xs[0] as Tensor, { covariance: xs[1] as Tensor })
        const conditioned = (): D.MultivariateNormal => d.condition([2], [1.5])
        got =
          method === 'logProb'
            ? d.logProb(xs[2] as Tensor)
            : method === 'conditionMean'
              ? conditioned().loc
              : method === 'conditionCov'
                ? conditioned().covariance()
                : (d[method as 'entropy' | 'variance'] as () => Value)()
      } else if (family === 'dirichlet') {
        const d = D.dirichlet(xs[0] as Tensor)
        got = method === 'logProb' ? d.logProb(xs[1] as Tensor) : (d[method as 'mean'] as () => Value)()
      } else {
        const d = D.multinomial(c.params['n'] as number, xs[0] as Tensor)
        got = method === 'logProb' ? d.logProb(xs[1] as Tensor) : (d[method as 'entropy'] as () => Value)()
      }
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})
