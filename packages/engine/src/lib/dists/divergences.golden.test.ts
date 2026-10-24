import { type Custom, describeDistributions } from '../../testing/distributions.ts'
import type { Tensor } from '../tensor/index.ts'
import * as C from './compose/index.ts'
import * as D from './index.ts'

type Make = (xs: readonly Tensor[]) => D.Distribution
const t = (xs: readonly Tensor[], i: number): Tensor => xs[i] as Tensor

const MAKERS: Readonly<Record<string, readonly [Make, Make]>> = {
  normal: [(x) => D.normal(t(x, 0), t(x, 1)), (x) => D.normal(t(x, 0), t(x, 1))],
  logNormal: [(x) => D.logNormal(t(x, 0), t(x, 1)), (x) => D.logNormal(t(x, 0), t(x, 1))],
  halfNormal: [(x) => D.halfNormal(t(x, 0)), (x) => D.halfNormal(t(x, 0))],
  gamma: [(x) => D.gamma(t(x, 0), t(x, 1)), (x) => D.gamma(t(x, 0), t(x, 1))],
  chi2: [(x) => D.chi2(t(x, 0)), (x) => D.chi2(t(x, 0))],
  exponential: [(x) => D.exponential(t(x, 0)), (x) => D.exponential(t(x, 0))],
  exponentialGamma: [(x) => D.exponential(t(x, 0)), (x) => D.gamma(t(x, 0), t(x, 1))],
  beta: [(x) => D.beta(t(x, 0), t(x, 1)), (x) => D.beta(t(x, 0), t(x, 1))],
  dirichlet: [(x) => D.dirichlet(t(x, 0)), (x) => D.dirichlet(t(x, 0))],
  bernoulli: [(x) => D.bernoulli({ probs: t(x, 0) }), (x) => D.bernoulli({ probs: t(x, 0) })],
  binomial: [(x) => D.binomial(t(x, 0), t(x, 1)), (x) => D.binomial(t(x, 0), t(x, 1))],
  categorical: [(x) => D.categorical({ probs: t(x, 0) }), (x) => D.categorical({ probs: t(x, 0) })],
  oneHotCategorical: [(x) => D.oneHotCategorical({ probs: t(x, 0) }), (x) => D.oneHotCategorical({ probs: t(x, 0) })],
  poisson: [(x) => D.poisson(t(x, 0)), (x) => D.poisson(t(x, 0))],
  geometric: [(x) => D.geometric(t(x, 0)), (x) => D.geometric(t(x, 0))],
  laplace: [(x) => D.laplace(t(x, 0), t(x, 1)), (x) => D.laplace(t(x, 0), t(x, 1))],
  gumbel: [(x) => D.gumbel(t(x, 0), t(x, 1)), (x) => D.gumbel(t(x, 0), t(x, 1))],
  uniform: [(x) => D.uniform(t(x, 0), t(x, 1)), (x) => D.uniform(t(x, 0), t(x, 1))],
  pareto: [(x) => D.pareto(t(x, 0), t(x, 1)), (x) => D.pareto(t(x, 0), t(x, 1))],
  multivariateNormal: [
    (x) => D.multivariateNormal(t(x, 0), { covariance: t(x, 1) }),
    (x) => D.multivariateNormal(t(x, 0), { covariance: t(x, 1) }),
  ],
  lowRankMultivariateNormal: [
    (x) => D.lowRankMultivariateNormal(t(x, 0), t(x, 1), t(x, 2)),
    (x) => D.multivariateNormal(t(x, 0), { covariance: t(x, 1) }),
  ],
  independentNormal: [
    (x) => C.independent(D.normal(t(x, 0), t(x, 1))),
    (x) => C.independent(D.normal(t(x, 0), t(x, 1))),
  ],
}

const custom: Record<string, Custom> = {}
for (const [name, [makeP, makeQ]] of Object.entries(MAKERS))
  custom[`kl.${name}`] = (xs, c) => {
    const arity = c.params['arity'] as number
    return D.klDivergence(makeP(xs.slice(0, arity)), makeQ(xs.slice(arity)))
  }

describeDistributions('dists/divergences', {}, custom)
