/**
 * Conjugate prior and likelihood pairs, where the posterior stays in the prior's family and has a closed form: the
 * table `CONJUGATES` (beta for Bernoulli and binomial probabilities, gamma for Poisson and exponential rates, normal
 * and inverse gamma for normal means and variances, Dirichlet for categorical probabilities, Wishart and inverse
 * Wishart for covariances, Pareto for a uniform bound), with `conjugatePrior` to look one up and `conjugatePosterior`
 * to update it from observations, as probabilistic programs use to skip sampling.
 */

import { OmaiError } from '../base/index.ts'
import { inv } from '../numerics/linalg/index.ts'
import {
  add,
  cast,
  div,
  equal,
  expandDims,
  log,
  matmul,
  max,
  maximum,
  mul,
  promoteFloat,
  sqrt,
  square,
  squeeze,
  stack,
  sub,
  sum,
  swapaxes,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'
import { beta } from './continuous/bounded.ts'
import { gamma, inverseGamma } from './continuous/gamma.ts'
import { pareto } from './continuous/heavy-tailed.ts'
import { normal } from './continuous/normal.ts'
import type { Distribution } from './distribution.ts'
import { inverseWishart, wishart } from './multivariate/matrix.ts'
import { multivariateNormal } from './multivariate/normal.ts'
import { dirichlet } from './multivariate/simplex.ts'

/**
 * One conjugate pair: a prior family for one parameter of a likelihood family, with the closed-form posterior given
 * observations (stacked on the first axis) and the likelihood's other, known parameters.
 *
 * @concept conjugate-priors
 */
export interface ConjugatePair {
  readonly likelihood: string
  readonly parameter: string
  readonly prior: string
  /** The names of the likelihood parameters that must be known (passed in `fixed`). */
  readonly fixed: readonly string[]
  posterior(prior: Distribution, data: Value, fixed: Readonly<Record<string, Value>>): Distribution
}

const count = (data: Value): number => data.shape[0] ?? 0
const known = (fixed: Readonly<Record<string, Value>>, name: string, pair: string): Value => {
  const v = fixed[name]
  if (v === undefined) throw new OmaiError('value', `conjugatePosterior: ${pair} needs the known ${name}`)
  return v
}
const priorParam = (prior: Distribution, name: string): Value => {
  const v = prior.params[name]
  if (v === undefined) throw new OmaiError('value', `conjugatePosterior: the prior has no parameter ${name}`)
  return v
}
const transposeLast = (m: Value): Value => swapaxes(m, -1, -2)

/**
 * The conjugate pairs the library knows, as `probprog` uses them to detect closed-form posteriors: beta for Bernoulli,
 * binomial, geometric and negative binomial probabilities; gamma for Poisson, exponential and gamma rates and the
 * Pareto shape; normal for a normal or log-normal mean with known scale; inverse gamma for a normal variance with
 * known mean; Dirichlet for categorical and multinomial probabilities; multivariate normal for a multivariate normal
 * mean, inverse Wishart for its covariance and Wishart for its precision; Pareto for the upper end of a uniform.
 *
 * @concept conjugate-priors
 */
export const CONJUGATES: readonly ConjugatePair[] = Object.freeze([
  {
    likelihood: 'bernoulli',
    parameter: 'probs',
    prior: 'beta',
    fixed: [],
    posterior: (prior, x) =>
      beta(add(priorParam(prior, 'a'), sum(x, 0)), add(priorParam(prior, 'b'), sub(count(x), sum(x, 0)))),
  },
  {
    likelihood: 'binomial',
    parameter: 'probs',
    prior: 'beta',
    fixed: ['n'],
    posterior: (prior, x, f) => {
      const trials = mul(known(f, 'n', 'binomial'), count(x))
      return beta(add(priorParam(prior, 'a'), sum(x, 0)), add(priorParam(prior, 'b'), sub(trials, sum(x, 0))))
    },
  },
  {
    likelihood: 'geometric',
    parameter: 'probs',
    prior: 'beta',
    fixed: [],
    // Trials up to and including the first success: n successes and sum(x) - n failures.
    posterior: (prior, x) =>
      beta(add(priorParam(prior, 'a'), count(x)), add(priorParam(prior, 'b'), sub(sum(x, 0), count(x)))),
  },
  {
    likelihood: 'negativeBinomial',
    parameter: 'probs',
    prior: 'beta',
    fixed: ['n'],
    posterior: (prior, x, f) =>
      beta(
        add(priorParam(prior, 'a'), mul(known(f, 'n', 'negativeBinomial'), count(x))),
        add(priorParam(prior, 'b'), sum(x, 0)),
      ),
  },
  {
    likelihood: 'poisson',
    parameter: 'rate',
    prior: 'gamma',
    fixed: [],
    posterior: (prior, x) =>
      gamma(add(priorParam(prior, 'concentration'), sum(x, 0)), add(priorParam(prior, 'rate'), count(x))),
  },
  {
    likelihood: 'exponential',
    parameter: 'rate',
    prior: 'gamma',
    fixed: [],
    posterior: (prior, x) =>
      gamma(add(priorParam(prior, 'concentration'), count(x)), add(priorParam(prior, 'rate'), sum(x, 0))),
  },
  {
    likelihood: 'gamma',
    parameter: 'rate',
    prior: 'gamma',
    fixed: ['concentration'],
    posterior: (prior, x, f) =>
      gamma(
        add(priorParam(prior, 'concentration'), mul(known(f, 'concentration', 'gamma'), count(x))),
        add(priorParam(prior, 'rate'), sum(x, 0)),
      ),
  },
  {
    likelihood: 'pareto',
    parameter: 'b',
    prior: 'gamma',
    fixed: ['scale'],
    posterior: (prior, x, f) =>
      gamma(
        add(priorParam(prior, 'concentration'), count(x)),
        add(priorParam(prior, 'rate'), sum(log(div(x, known(f, 'scale', 'pareto'))), 0)),
      ),
  },
  {
    likelihood: 'normal',
    parameter: 'loc',
    prior: 'normal',
    fixed: ['scale'],
    posterior: (prior, x, f) => normalMeanPosterior(prior, x, known(f, 'scale', 'normal')),
  },
  {
    likelihood: 'logNormal',
    parameter: 'mu',
    prior: 'normal',
    fixed: ['sigma'],
    posterior: (prior, x, f) => normalMeanPosterior(prior, log(x), known(f, 'sigma', 'logNormal')),
  },
  {
    likelihood: 'normal',
    parameter: 'variance',
    prior: 'inverseGamma',
    fixed: ['loc'],
    posterior: (prior, x, f) =>
      inverseGamma(
        add(priorParam(prior, 'concentration'), count(x) / 2),
        add(priorParam(prior, 'scale'), mul(sum(square(sub(x, known(f, 'loc', 'normal'))), 0), 0.5)),
      ),
  },
  {
    likelihood: 'categorical',
    parameter: 'probs',
    prior: 'dirichlet',
    fixed: [],
    posterior: (prior, x) => {
      const alpha = priorParam(prior, 'concentration')
      const k = alpha.shape[alpha.shape.length - 1] ?? 0
      const counts = Array.from({ length: k }, (_, j) => sum(cast(equal(x, j), 'float64'), 0))
      return dirichlet(add(alpha, stack(counts, -1)))
    },
  },
  {
    likelihood: 'multinomial',
    parameter: 'probs',
    prior: 'dirichlet',
    fixed: [],
    posterior: (prior, x) => dirichlet(add(priorParam(prior, 'concentration'), sum(x, 0))),
  },
  {
    likelihood: 'multivariateNormal',
    parameter: 'loc',
    prior: 'multivariateNormal',
    fixed: ['covariance'],
    posterior: (prior, x, f) => {
      // Lambda_n = Lambda_0 + n Sigma^{-1}, mu_n = Lambda_n^{-1} (Lambda_0 mu_0 + Sigma^{-1} sum x).
      const l0 = priorParam(prior, 'scaleTril')
      const prec0 = inv(matmul(l0, transposeLast(l0)))
      const sigmaInv = inv(known(f, 'covariance', 'multivariateNormal'))
      const precN = add(prec0, mul(sigmaInv, count(x)))
      const covN = inv(precN)
      const rhs = add(
        squeeze(matmul(prec0, expandDims(priorParam(prior, 'loc'), -1)), -1),
        squeeze(matmul(sigmaInv, expandDims(sum(x, 0), -1)), -1),
      )
      return multivariateNormal(squeeze(matmul(covN, expandDims(rhs, -1)), -1), { covariance: symmetrise(covN) })
    },
  },
  {
    likelihood: 'multivariateNormal',
    parameter: 'covariance',
    prior: 'inverseWishart',
    fixed: ['loc'],
    posterior: (prior, x, f) => {
      const centred = sub(x, known(f, 'loc', 'multivariateNormal'))
      const scatter = sum(mul(expandDims(centred, -1), expandDims(centred, -2)), 0)
      return inverseWishart(add(priorParam(prior, 'df'), count(x)), add(priorParam(prior, 'scale'), scatter))
    },
  },
  {
    likelihood: 'multivariateNormal',
    parameter: 'precision',
    prior: 'wishart',
    fixed: ['loc'],
    posterior: (prior, x, f) => {
      const centred = sub(x, known(f, 'loc', 'multivariateNormal'))
      const scatter = sum(mul(expandDims(centred, -1), expandDims(centred, -2)), 0)
      const l = priorParam(prior, 'scaleTril')
      const scaleInv = inv(matmul(l, transposeLast(l)))
      return wishart(add(priorParam(prior, 'df'), count(x)), { precision: symmetrise(add(scaleInv, scatter)) })
    },
  },
  {
    likelihood: 'uniform',
    parameter: 'high',
    prior: 'pareto',
    fixed: [],
    posterior: (prior, x) =>
      pareto(add(priorParam(prior, 'b'), count(x)), maximum(priorParam(prior, 'scale'), max(x, 0))),
  },
])

function normalMeanPosterior(prior: Distribution, x: Value, sigma: Value): Distribution {
  const m0 = priorParam(prior, 'loc')
  const s0 = priorParam(prior, 'scale')
  const prec = add(div(1, square(s0)), div(count(x), square(sigma)))
  const meanN = div(add(div(m0, square(s0)), div(sum(x, 0), square(sigma))), prec)
  return normal(meanN, div(1, sqrt(prec)))
}

const symmetrise = (m: Value): Value => mul(add(m, transposeLast(m)), 0.5)

/**
 * The conjugate prior family for a parameter of a likelihood, if there is one in {@link CONJUGATES}.
 *
 * @param likelihood - The likelihood's distribution name (as in `dist.name`).
 * @param parameter - The parameter (`probs`, `rate`, `loc`, `variance`, `covariance`, `precision`, ...).
 * @returns The conjugate pair, or undefined.
 * @example
 * ```ts
 * import { conjugatePrior } from 'omai-engine/dists'
 * conjugatePrior('poisson', 'rate')?.prior // => 'gamma'
 * ```
 */
export function conjugatePrior(likelihood: string, parameter: string): ConjugatePair | undefined {
  return CONJUGATES.find((c) => c.likelihood === likelihood && c.parameter === parameter)
}

/**
 * The closed-form posterior of a conjugate prior given observations.
 *
 * @param prior - The prior (its family must match the pair's).
 * @param likelihood - The likelihood's name.
 * @param parameter - The parameter the prior is on.
 * @param data - Observations along the first axis.
 * @param fixed - The likelihood's known parameters.
 * @returns The posterior distribution.
 * @throws OmaiError with code `not-implemented` when the pair is not conjugate here, or `value` when the prior's
 *   family does not match or a known parameter is missing.
 * @example
 * ```ts
 * import { beta, conjugatePosterior } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * // Beta(1, 1) and three successes in four Bernoulli trials: Beta(4, 2), with mean 2/3.
 * item(conjugatePosterior(beta(1, 1), 'bernoulli', 'probs', [1, 1, 0, 1]).mean()) // => 0.6666666666666666
 * ```
 */
export function conjugatePosterior(
  prior: Distribution,
  likelihood: string,
  parameter: string,
  data: TensorLike,
  fixed: Readonly<Record<string, TensorLike>> = {},
): Distribution {
  const pair = conjugatePrior(likelihood, parameter)
  if (pair === undefined)
    throw new OmaiError('not-implemented', `conjugatePosterior: no conjugate prior for ${likelihood}.${parameter}`)
  if (pair.prior !== prior.name)
    throw new OmaiError('value', `conjugatePosterior: ${likelihood}.${parameter} takes a ${pair.prior} prior`)
  const values: Record<string, Value> = {}
  for (const [k, v] of Object.entries(fixed)) values[k] = promoteFloat(v)[0] as Value
  return pair.posterior(prior, promoteFloat(data)[0] as Value, values)
}
