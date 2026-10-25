/**
 * Distributional regression families in the GAMLSS style (Rigby and Stasinopoulos), where every parameter of a
 * distribution (location, scale, shape) has its own link and can depend on covariates: normal, Student-t, skew-normal,
 * gamma, log-normal, Weibull, beta, Poisson, negative binomial and zero-inflated Poisson, plus cumulative-link families
 * for ordinal regression.
 */

import { WhiteboxError } from '../../base/index.ts'
import { log1mexp, logNdtr, logSigmoid } from '../../numerics/special/index.ts'
import {
  add,
  arctan,
  div,
  exp,
  log,
  mul,
  neg,
  promoteFloat,
  square,
  sub,
  sum,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'
import { beta } from '../continuous/bounded.ts'
import { gamma, weibull } from '../continuous/gamma.ts'
import { studentT } from '../continuous/heavy-tailed.ts'
import { logNormal, normal, skewNormal } from '../continuous/normal.ts'
import { orderedCategorical } from '../discrete/categorical.ts'
import { negativeBinomial, poisson, zeroInflatedPoisson } from '../discrete/counts.ts'
import type { Distribution } from '../distribution.ts'
import { identity, type Link, logit, log as logLink } from './links.ts'

/** One parameter of a distributional family: its name and the link from its natural scale to the predictor scale. */
export interface DistributionalParameter {
  readonly name: string
  readonly link: Link
}

/**
 * A GAMLSS-style distributional family (Rigby and Stasinopoulos, 2005): a distribution whose every parameter has its
 * own link and linear predictor, so location, scale and shape can all depend on covariates.
 *
 * @concept distributional-regression
 */
export interface DistributionalFamily {
  readonly name: string
  readonly parameters: readonly DistributionalParameter[]
  /** The distribution at parameter values on their natural scales. */
  distribution(params: readonly TensorLike[]): Distribution
  /** The distribution at linear predictors (one per parameter, on the link scale). */
  fromPredictors(predictors: readonly TensorLike[]): Distribution
  /** The summed log-likelihood of responses at linear predictors. */
  logLikelihood(y: TensorLike, predictors: readonly TensorLike[]): Value
}

/**
 * Build a distributional family from its parameters and their links and a distribution constructor.
 *
 * @param name - The family's name.
 * @param parameters - Names and links, in order.
 * @param make - The distribution at natural-scale parameter values.
 * @returns The family.
 * @example
 * ```ts
 * import { families, laplace } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * const lap = families.defineDistributionalFamily('laplace', [
 *   { name: 'mu', link: families.identity() }, { name: 'b', link: families.log() },
 * ], ([m, b]) => laplace(m, b))
 * item(lap.fromPredictors([0, 0]).variance()) // => 2
 * ```
 */
export function defineDistributionalFamily(
  name: string,
  parameters: readonly DistributionalParameter[],
  make: (params: readonly Value[]) => Distribution,
): DistributionalFamily {
  const count = parameters.length
  const check = (xs: readonly TensorLike[]): Value[] => {
    if (xs.length !== count)
      throw new WhiteboxError(
        'value',
        `${name}: needs ${count} parameters (${parameters.map((p) => p.name).join(', ')})`,
      )
    return xs.map((x) => promoteFloat(x)[0] as Value)
  }
  const fromPredictors = (predictors: readonly TensorLike[]): Distribution =>
    make(check(predictors).map((eta, i) => (parameters[i] as DistributionalParameter).link.inverse(eta)))
  return {
    name,
    parameters,
    distribution: (params) => make(check(params)),
    fromPredictors,
    logLikelihood: (y, predictors) => sum(fromPredictors(predictors).logProb(y)),
  }
}

const p = (name: string, link: Link): DistributionalParameter => ({ name, link })

/**
 * The normal family NO(mu, sigma): mean mu (identity link) and standard deviation sigma (log link).
 *
 * @returns The family.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.gamlssNormal().fromPredictors([1, 0]).stddev()) // => 1
 * ```
 */
export function gamlssNormal(): DistributionalFamily {
  return defineDistributionalFamily('normal', [p('mu', identity()), p('sigma', logLink())], ([m, s]) =>
    normal(m as Value, s as Value),
  )
}

/**
 * The Student-t family TF(mu, sigma, nu): location (identity), scale (log) and degrees of freedom (log).
 *
 * @returns The family.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.gamlssStudentT().distribution([0, 1, 3]).cdf(0)) // => 0.5
 * ```
 */
export function gamlssStudentT(): DistributionalFamily {
  return defineDistributionalFamily(
    'studentT',
    [p('mu', identity()), p('sigma', logLink()), p('nu', logLink())],
    ([m, s, n]) => studentT(n as Value, m as Value, s as Value),
  )
}

/**
 * The skew-normal family SN(mu, sigma, nu): location (identity), scale (log) and skewness (identity).
 *
 * @returns The family.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.gamlssSkewNormal().distribution([0, 1, 0]).cdf(0)) // => 0.5
 * ```
 */
export function gamlssSkewNormal(): DistributionalFamily {
  return defineDistributionalFamily(
    'skewNormal',
    [p('mu', identity()), p('sigma', logLink()), p('nu', identity())],
    ([m, s, a]) => skewNormal(m as Value, s as Value, a as Value),
  )
}

/**
 * The gamma family GA(mu, sigma) in GAMLSS's parameterisation: mean mu and coefficient of variation sigma (both log
 * links), so the shape is 1 / sigma^2 and the variance sigma^2 mu^2.
 *
 * @returns The family.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.gamlssGamma().distribution([2, 0.5]).variance()) // => 1
 * ```
 */
export function gamlssGamma(): DistributionalFamily {
  return defineDistributionalFamily('gamma', [p('mu', logLink()), p('sigma', logLink())], ([m, s]) => {
    const shape = div(1, square(s as Value))
    return gamma(shape, div(shape, m as Value))
  })
}

/**
 * The log-normal family LOGNO(mu, sigma): the mean (identity) and standard deviation (log) of log Y.
 *
 * @returns The family.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.gamlssLogNormal().distribution([0, 1]).cdf(1)) // => 0.5
 * ```
 */
export function gamlssLogNormal(): DistributionalFamily {
  return defineDistributionalFamily('logNormal', [p('mu', identity()), p('sigma', logLink())], ([m, s]) =>
    logNormal(m as Value, s as Value),
  )
}

/**
 * The Weibull family WEI(mu, sigma): scale mu and shape sigma (both log links).
 *
 * @returns The family.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(families.gamlssWeibull().distribution([2, 1]).mean()) - 2) < 1e-14 // => true
 * ```
 */
export function gamlssWeibull(): DistributionalFamily {
  return defineDistributionalFamily('weibull', [p('mu', logLink()), p('sigma', logLink())], ([m, s]) =>
    weibull(s as Value, m as Value),
  )
}

/**
 * The beta family BE(mu, sigma) in GAMLSS's mean-precision form: mean mu and sigma with variance
 * sigma^2 mu (1 - mu) (both logit links), i.e. a = mu (1 - sigma^2) / sigma^2, b = (1 - mu) (1 - sigma^2) / sigma^2.
 *
 * @returns The family.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(families.gamlssBeta().distribution([0.3, 0.5]).variance()) - 0.25 * 0.21) < 1e-15 // => true
 * ```
 */
export function gamlssBeta(): DistributionalFamily {
  return defineDistributionalFamily('beta', [p('mu', logit()), p('sigma', logit())], ([m, s]) => {
    const s2 = square(s as Value)
    const k = div(sub(1, s2), s2)
    return beta(mul(m as Value, k), mul(sub(1, m as Value), k))
  })
}

/**
 * The Poisson family PO(mu): the mean (log link).
 *
 * @returns The family.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.gamlssPoisson().fromPredictors([0]).mean()) // => 1
 * ```
 */
export function gamlssPoisson(): DistributionalFamily {
  return defineDistributionalFamily('poisson', [p('mu', logLink())], ([m]) => poisson(m as Value))
}

/**
 * The negative binomial family NBI(mu, sigma): mean mu and dispersion sigma (both log links), variance
 * mu + sigma mu^2.
 *
 * @returns The family.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.gamlssNegativeBinomial().distribution([2, 0.5]).variance()) // => 4
 * ```
 */
export function gamlssNegativeBinomial(): DistributionalFamily {
  return defineDistributionalFamily('negativeBinomial', [p('mu', logLink()), p('sigma', logLink())], ([m, s]) =>
    negativeBinomial(div(1, s as Value), div(1, add(mul(s as Value, m as Value), 1))),
  )
}

/**
 * The zero-inflated Poisson family ZIP(mu, sigma): Poisson mean mu (log link) and structural-zero probability sigma
 * (logit link).
 *
 * @returns The family.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.gamlssZeroInflatedPoisson().distribution([4, 0.25]).mean()) // => 3
 * ```
 */
export function gamlssZeroInflatedPoisson(): DistributionalFamily {
  return defineDistributionalFamily('zeroInflatedPoisson', [p('mu', logLink()), p('sigma', logit())], ([m, s]) =>
    zeroInflatedPoisson(m as Value, s as Value),
  )
}

/** The cumulative links of ordinal regression: the CDF F with P(Y <= k) = F(cutpoint_k - predictor). */
export type CumulativeLink = 'logit' | 'probit' | 'cloglog' | 'loglog' | 'cauchit'

const LOG_CDFS: Readonly<Record<CumulativeLink, (x: Value) => Value>> = {
  logit: logSigmoid,
  probit: logNdtr,
  // F = 1 - exp(-exp(x)), log F = log1mexp(-exp(x)).
  cloglog: (x) => log1mexp(neg(exp(x))),
  // F = exp(-exp(-x)).
  loglog: (x) => neg(exp(neg(x))),
  cauchit: (x) => log(add(div(arctan(x), Math.PI), 0.5)),
}

/** A cumulative-link (ordinal regression) family: the predictor and the increasing cutpoints. */
export interface OrdinalFamily {
  readonly name: string
  readonly link: CumulativeLink
  /** The distribution over {0, ..., K-1} for a predictor and K - 1 increasing cutpoints. */
  distribution(predictor: TensorLike, cutpoints: TensorLike): Distribution
  /** The summed log-likelihood of ordinal responses. */
  logLikelihood(y: TensorLike, predictor: TensorLike, cutpoints: TensorLike): Value
}

/**
 * A cumulative-link ordinal family (proportional odds for logit, ordered probit, and the cloglog, loglog and cauchit
 * variants; statsmodels' OrderedModel, R's MASS::polr): P(Y <= k) = F(cutpoint_k - predictor).
 *
 * @param link - The cumulative link (default logit).
 * @returns The family.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.ordinalFamily('probit').distribution(0, [0]).prob(0)) // => 0.5
 * ```
 */
export function ordinalFamily(link: CumulativeLink = 'logit'): OrdinalFamily {
  const logCdf = LOG_CDFS[link]
  const distribution = (predictor: TensorLike, cutpoints: TensorLike): Distribution =>
    orderedCategorical(predictor, cutpoints, logCdf, `ordinal(${link})`)
  return {
    name: `ordinal(${link})`,
    link,
    distribution,
    logLikelihood: (y, predictor, cutpoints) => sum(distribution(predictor, cutpoints).logProb(y)),
  }
}
