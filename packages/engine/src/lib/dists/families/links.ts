/**
 * Link functions for generalised linear models, which map a distribution's mean to the scale of the linear predictor,
 * as statsmodels' links: identity, log, logit, probit, cauchit, complementary log-log, log-log, inverse, inverse
 * squared, square root, power and negative binomial, each with its inverse and derivatives guarded at the edges of its
 * domain.
 */

import { MdmlError } from '../../base/index.ts'
import { ndtr, ndtri, normalPdf } from '../../numerics/special/index.ts'
import {
  abs,
  add,
  arctan,
  clip,
  cos,
  div,
  exp,
  expm1,
  log1p,
  log as logOp,
  maximum,
  mul,
  neg,
  pow,
  promoteFloat,
  sigmoid,
  sin,
  square,
  sub,
  type TensorLike,
  tan,
  type Value,
} from '../../tensor/index.ts'

/** The machine epsilon statsmodels clips probabilities and means with. */
const EPS = Number.EPSILON

/**
 * A GLM link function g with eta = g(mu) (statsmodels.genmod.families.links): the link, its inverse, and first and
 * second derivatives, each guarded against the edges of its domain (probabilities clipped to [eps, 1 - eps], positive
 * means to at least eps, as statsmodels does).
 *
 * @concept generalized-linear-models
 */
export interface Link {
  readonly name: string
  /** eta = g(mu). */
  link(mu: TensorLike): Value
  /** mu = g^{-1}(eta). */
  inverse(eta: TensorLike): Value
  /** g'(mu) = d eta / d mu. */
  derivative(mu: TensorLike): Value
  /** d mu / d eta at eta. */
  inverseDerivative(eta: TensorLike): Value
  /** g''(mu). */
  secondDerivative(mu: TensorLike): Value
}

interface LinkSpec {
  readonly name: string
  link(mu: Value): Value
  inverse(eta: Value): Value
  derivative(mu: Value): Value
  inverseDerivative(eta: Value): Value
  secondDerivative(mu: Value): Value
}

function defineLink(spec: LinkSpec): Link {
  const v = (x: TensorLike): Value => promoteFloat(x)[0] as Value
  return {
    name: spec.name,
    link: (mu) => spec.link(v(mu)),
    inverse: (eta) => spec.inverse(v(eta)),
    derivative: (mu) => spec.derivative(v(mu)),
    inverseDerivative: (eta) => spec.inverseDerivative(v(eta)),
    secondDerivative: (mu) => spec.secondDerivative(v(mu)),
  }
}

const probability = (p: Value): Value => clip(p, EPS, 1 - EPS)
const positive = (x: Value): Value => maximum(x, EPS)

/**
 * The power link eta = mu^p (statsmodels' Power): identity at p = 1, inverse at -1, inverse squared at -2, square
 * root at 1/2.
 *
 * @param p - The power (nonzero).
 * @returns The link.
 * @throws MdmlError with code `value` for p = 0 (use the log link).
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(families.power(2).inverse(9)) // => 3
 * ```
 */
export function power(p: number): Link {
  if (p === 0) throw new MdmlError('value', 'power link: p must be nonzero (the limit is the log link)')
  return defineLink({
    name: p === 1 ? 'identity' : p === -1 ? 'inverse' : p === -2 ? 'inverseSquared' : p === 0.5 ? 'sqrt' : 'power',
    link: (mu) => (p === 1 ? mu : pow(mu, p)),
    inverse: (eta) => (p === 1 ? eta : pow(eta, 1 / p)),
    derivative: (mu) => (p === 1 ? add(mul(mu, 0), 1) : mul(pow(mu, p - 1), p)),
    inverseDerivative: (eta) => (p === 1 ? add(mul(eta, 0), 1) : div(pow(eta, 1 / p - 1), p)),
    secondDerivative: (mu) => (p === 1 ? mul(mu, 0) : mul(pow(mu, p - 2), p * (p - 1))),
  })
}

/**
 * The identity link eta = mu (the canonical link of the Gaussian family).
 *
 * @returns The link.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(families.identity().link(2.5)) // => 2.5
 * ```
 */
export function identity(): Link {
  return power(1)
}

/**
 * The inverse (reciprocal) link eta = 1 / mu (the canonical link of the gamma family).
 *
 * @returns The link.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(families.inverse().link(4)) // => 0.25
 * ```
 */
export function inverse(): Link {
  return power(-1)
}

/**
 * The inverse squared link eta = 1 / mu^2 (the canonical link of the inverse Gaussian family).
 *
 * @returns The link.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(families.inverseSquared().inverse(0.25)) // => 2
 * ```
 */
export function inverseSquared(): Link {
  return power(-2)
}

/**
 * The square-root link eta = sqrt(mu), variance stabilising for counts.
 *
 * @returns The link.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(families.sqrt().link(9)) // => 3
 * ```
 */
export function sqrt(): Link {
  return power(0.5)
}

/**
 * The log link eta = log(mu) (the canonical link of the Poisson family), with mu clipped below at eps.
 *
 * @returns The link.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(families.log().inverse(0)) // => 1
 * ```
 */
export function log(): Link {
  return defineLink({
    name: 'log',
    link: (mu) => logOp(positive(mu)),
    inverse: exp,
    derivative: (mu) => div(1, positive(mu)),
    inverseDerivative: exp,
    secondDerivative: (mu) => neg(div(1, square(positive(mu)))),
  })
}

/**
 * The logit link eta = log(mu / (1 - mu)) (the canonical link of the binomial family).
 *
 * @returns The link; the inverse is the stable sigmoid.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(families.logit().inverse(0)) // => 0.5
 * ```
 */
export function logit(): Link {
  return defineLink({
    name: 'logit',
    link: (mu) => {
      const p = probability(mu)
      return sub(logOp(p), log1p(neg(p)))
    },
    inverse: sigmoid,
    derivative: (mu) => {
      const p = probability(mu)
      return div(1, mul(p, sub(1, p)))
    },
    inverseDerivative: (eta) => {
      // exp(-|z|) / (1 + exp(-|z|))^2, without overflow.
      const t = exp(neg(abs(eta)))
      return div(t, square(add(t, 1)))
    },
    secondDerivative: (mu) => {
      const p = probability(mu)
      return div(sub(mul(p, 2), 1), square(mul(p, sub(1, p))))
    },
  })
}

/**
 * The probit link eta = Phi^{-1}(mu).
 *
 * @returns The link.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(families.probit().link(0.5)) // => 0
 * ```
 */
export function probit(): Link {
  return defineLink({
    name: 'probit',
    link: (mu) => ndtri(probability(mu)),
    inverse: ndtr,
    derivative: (mu) => div(1, normalPdf(ndtri(probability(mu)))),
    inverseDerivative: normalPdf,
    secondDerivative: (mu) => {
      const z = ndtri(probability(mu))
      return div(z, square(normalPdf(z)))
    },
  })
}

/**
 * The cauchit link eta = tan(pi (mu - 1/2)), the Cauchy quantile function: heavier-tailed than logit or probit.
 *
 * @returns The link.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(families.cauchit().inverse(1)) // => 0.75
 * ```
 */
export function cauchit(): Link {
  const angle = (mu: Value): Value => mul(sub(probability(mu), 0.5), Math.PI)
  return defineLink({
    name: 'cauchit',
    link: (mu) => tan(angle(mu)),
    inverse: (eta) => add(div(arctan(eta), Math.PI), 0.5),
    derivative: (mu) => div(Math.PI, square(cos(angle(mu)))),
    inverseDerivative: (eta) => div(1 / Math.PI, add(square(eta), 1)),
    secondDerivative: (mu) => {
      const a = angle(mu)
      return div(mul(sin(a), 2 * Math.PI * Math.PI), pow(cos(a), 3))
    },
  })
}

/**
 * The complementary log-log link eta = log(-log(1 - mu)), the Gumbel-minimum CDF (asymmetric, for rare events and
 * grouped survival).
 *
 * @returns The link.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(families.cloglog().inverse(0)) === 1 - Math.exp(-1) // => true
 * ```
 */
export function cloglog(): Link {
  return defineLink({
    name: 'cloglog',
    link: (mu) => logOp(neg(log1p(neg(probability(mu))))),
    inverse: (eta) => neg(expm1(neg(exp(eta)))),
    derivative: (mu) => {
      const p = probability(mu)
      return div(1, mul(sub(p, 1), log1p(neg(p))))
    },
    inverseDerivative: (eta) => exp(sub(eta, exp(eta))),
    secondDerivative: (mu) => {
      const p = probability(mu)
      const fl = log1p(neg(p))
      return mul(neg(div(1, mul(square(sub(1, p)), fl))), add(div(1, fl), 1))
    },
  })
}

/**
 * The log-log link eta = -log(-log(mu)), the Gumbel-maximum CDF (the mirror of cloglog).
 *
 * @returns The link.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(families.loglog().inverse(0)) === Math.exp(-1) // => true
 * ```
 */
export function loglog(): Link {
  return defineLink({
    name: 'loglog',
    link: (mu) => neg(logOp(neg(logOp(probability(mu))))),
    inverse: (eta) => exp(neg(exp(neg(eta)))),
    derivative: (mu) => {
      const p = probability(mu)
      return neg(div(1, mul(p, logOp(p))))
    },
    inverseDerivative: (eta) => exp(sub(neg(exp(neg(eta))), eta)),
    secondDerivative: (mu) => {
      const p = probability(mu)
      return div(add(logOp(p), 1), square(mul(p, logOp(p))))
    },
  })
}

/**
 * The negative binomial link eta = log(mu / (mu + 1 / alpha)) (the canonical link of the negative binomial family
 * with dispersion alpha; statsmodels' NegativeBinomial link).
 *
 * @param alpha - The dispersion (positive).
 * @returns The link.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(families.negativeBinomialLink(1).link(1)) === Math.log(0.5) // => true
 * ```
 */
export function negativeBinomialLink(alpha = 1): Link {
  return defineLink({
    name: 'negativeBinomial',
    link: (mu) => {
      const m = positive(mu)
      return logOp(div(m, add(m, 1 / alpha)))
    },
    inverse: (eta) => div(-1 / alpha, neg(expm1(neg(eta)))),
    derivative: (mu) => div(1, add(mu, mul(square(mu), alpha))),
    inverseDerivative: (eta) => div(exp(eta), mul(square(neg(expm1(eta))), alpha)),
    secondDerivative: (mu) => neg(div(add(mul(mu, 2 * alpha), 1), square(add(mu, mul(square(mu), alpha))))),
  })
}

/** The names of the built-in links. */
export type LinkName =
  | 'identity'
  | 'log'
  | 'logit'
  | 'probit'
  | 'cloglog'
  | 'loglog'
  | 'cauchit'
  | 'inverse'
  | 'inverseSquared'
  | 'sqrt'

/**
 * A built-in link by name.
 *
 * @param name - The link's name.
 * @returns The link.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * families.linkByName('probit').name // => 'probit'
 * ```
 */
export function linkByName(name: LinkName): Link {
  switch (name) {
    case 'identity':
      return identity()
    case 'log':
      return log()
    case 'logit':
      return logit()
    case 'probit':
      return probit()
    case 'cloglog':
      return cloglog()
    case 'loglog':
      return loglog()
    case 'cauchit':
      return cauchit()
    case 'inverse':
      return inverse()
    case 'inverseSquared':
      return inverseSquared()
    case 'sqrt':
      return sqrt()
  }
}
