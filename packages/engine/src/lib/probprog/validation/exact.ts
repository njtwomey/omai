/**
 * Closed-form posteriors and evidence of the conjugate models, as oracles: a normal mean with normal observations, a
 * beta probability with Bernoulli or binomial trials, a gamma rate with Poisson counts and a Dirichlet with category
 * counts. The posteriors come from the conjugate table of `dists`; what this file adds is the log evidence, written
 * out from the standard formulas, and the one-dimensional oracles of each posterior.
 */

import { OmaiError } from '../../base/index.ts'
import { beta, conjugatePosterior, type Distribution, dirichlet, gamma, normal } from '../../dists/index.ts'
import { lbeta, lgamma } from '../../numerics/special/index.ts'
import { item, tensor } from '../../tensor/index.ts'
import { distributionOracle, type MarginalOracle } from './oracle.ts'

/** A conjugate posterior with its log evidence. */
export interface ExactPosterior {
  /** The posterior distribution of the parameter. */
  readonly posterior: Distribution
  /** log p(data) under the prior. */
  readonly logEvidence: number
  /** The scalar marginals of the posterior (one per coordinate), as oracles. */
  oracles(name: string): MarginalOracle[]
}

function withOracles(posterior: Distribution, logEvidence: number): ExactPosterior {
  return {
    posterior,
    logEvidence,
    oracles: (name) => {
      if (posterior.eventShape.length === 0 && posterior.batchShape.length === 0)
        return [distributionOracle(name, posterior)]
      throw new OmaiError(
        'shape',
        `exact oracles: the posterior of ${name} is not scalar; use its coordinate marginals`,
      )
    },
  }
}

const sumOf = (y: readonly number[]): number => y.reduce((a, b) => a + b, 0)

/**
 * A normal mean with known observation sd: prior N(m0, s0^2), observations y_i ~ N(mu, sigma^2). The posterior is
 * N(m_n, s_n^2) and log p(y) = log N(y; m0 1, sigma^2 I + s0^2 11').
 *
 * @param options - The prior, the known sd and the observations.
 * @returns The posterior and the log evidence.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { normalNormal } from 'omai-engine/probprog/validation'
 * const e = normalNormal({ priorMean: 0, priorSd: 1, sigma: 1, y: [1.2] })
 * [e.posterior.mean().data[0], e.posterior.variance().data[0]] // => [0.6, 0.5]
 * ```
 */
export function normalNormal(options: {
  readonly priorMean: number
  readonly priorSd: number
  readonly sigma: number
  readonly y: readonly number[]
}): ExactPosterior {
  const { priorMean: m0, priorSd: s0, sigma, y } = options
  const n = y.length
  const posterior = conjugatePosterior(normal(m0, s0), 'normal', 'loc', tensor([...y]), { scale: sigma })
  const ybar = sumOf(y) / n
  const ss = y.reduce((a, v) => a + (v - ybar) ** 2, 0)
  const v = sigma ** 2 + n * s0 ** 2
  // log N(y; m0 1, sigma^2 I + s0^2 11') through the matrix determinant lemma and Sherman-Morrison.
  const logEvidence =
    -0.5 * n * Math.log(2 * Math.PI) -
    0.5 * ((n - 1) * Math.log(sigma ** 2) + Math.log(v)) -
    0.5 * (ss / sigma ** 2 + (n * (ybar - m0) ** 2) / v)
  return withOracles(posterior, logEvidence)
}

/**
 * A beta probability with Bernoulli trials: prior Beta(a, b), k successes in n trials. The posterior is
 * Beta(a + k, b + n - k) and log p(y) = log B(a + k, b + n - k) - log B(a, b) (for the sequence, not the count).
 *
 * @param options - The prior and the trials (0 or 1).
 * @returns The posterior and the log evidence.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { betaBernoulli } from 'omai-engine/probprog/validation'
 * betaBernoulli({ a: 1, b: 1, y: [1, 0, 1] }).logEvidence // => -2.4849066497880004
 * ```
 */
export function betaBernoulli(options: {
  readonly a: number
  readonly b: number
  readonly y: readonly number[]
}): ExactPosterior {
  const { a, b, y } = options
  const k = sumOf(y)
  const posterior = conjugatePosterior(beta(a, b), 'bernoulli', 'probs', tensor([...y]))
  return withOracles(posterior, item(lbeta(a + k, b + y.length - k)) - item(lbeta(a, b)))
}

/**
 * A gamma rate with Poisson counts: prior Gamma(a, b) (shape and rate), counts k_i. The posterior is
 * Gamma(a + sum k, b + n) and log p(k) = a log b - log G(a) + log G(a + sum k) - (a + sum k) log(b + n) - sum log k_i!.
 *
 * @param options - The prior (shape and rate) and the counts.
 * @returns The posterior and the log evidence.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { gammaPoisson } from 'omai-engine/probprog/validation'
 * gammaPoisson({ shape: 2, rate: 1, counts: [3] }).posterior.mean().data[0] // => 2.5
 * ```
 */
export function gammaPoisson(options: {
  readonly shape: number
  readonly rate: number
  readonly counts: readonly number[]
}): ExactPosterior {
  const { shape: a, rate: b, counts } = options
  const n = counts.length
  const s = sumOf(counts)
  const posterior = conjugatePosterior(gamma(a, b), 'poisson', 'rate', tensor([...counts]))
  const logFactorials = counts.reduce((acc, k) => acc + item(lgamma(k + 1)), 0)
  const logEvidence =
    a * Math.log(b) - item(lgamma(a)) + item(lgamma(a + s)) - (a + s) * Math.log(b + n) - logFactorials
  return withOracles(posterior, logEvidence)
}

/**
 * A Dirichlet probability vector with categorical observations: prior Dirichlet(alpha), category counts c. The
 * posterior is Dirichlet(alpha + c) and log p(sequence) = log B(alpha + c) - log B(alpha), with the multivariate
 * beta function through log-gamma.
 *
 * @param options - The prior concentration and the observed categories (0-based).
 * @returns The posterior (its oracles are the beta marginals of each probability) and the log evidence.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { dirichletCategorical } from 'omai-engine/probprog/validation'
 * dirichletCategorical({ alpha: [1, 1, 1], y: [0, 0, 2] }).oracles('p').map((o) => o.mean) // => [0.5, 0.16666666666666666, 0.3333333333333333]
 * ```
 */
export function dirichletCategorical(options: {
  readonly alpha: readonly number[]
  readonly y: readonly number[]
}): ExactPosterior {
  const { alpha, y } = options
  const counts = alpha.map((_, k) => y.filter((c) => c === k).length)
  const post = alpha.map((a, k) => a + (counts[k] ?? 0))
  const logB = (v: readonly number[]): number => v.reduce((s, x) => s + item(lgamma(x)), 0) - item(lgamma(sumOf(v)))
  const posterior = dirichlet(post)
  const total = sumOf(post)
  return {
    posterior,
    logEvidence: logB(post) - logB(alpha),
    oracles: (name) => post.map((a, k) => distributionOracle(`${name}[${k}]`, beta(a, total - a))),
  }
}
