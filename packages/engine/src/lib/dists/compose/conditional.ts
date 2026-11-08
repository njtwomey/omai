/**
 * Conditional distributions p(y | x), functions from inputs to distributions: the linear-Gaussian conditional with its
 * closed-form marginals and posteriors, generalised linear model and distributional (GAMLSS) regression conditionals,
 * and heads that turn a network's raw outputs into valid normal, Bernoulli, categorical, Poisson, negative binomial,
 * beta, gamma, mixture-density and multivariate normal distributions.
 */

import { OmaiError } from '../../base/index.ts'
import { solve } from '../../numerics/linalg/index.ts'
import type { Key } from '../../random/index.ts'
import {
  add,
  concat,
  div,
  exp,
  expandDims,
  matmul,
  mul,
  promoteFloat,
  sliceAxis,
  softplus,
  squeeze,
  sub,
  swapaxes,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'
import { lowerCholesky } from '../bijectors/structured.ts'
import { beta } from '../continuous/bounded.ts'
import { gamma } from '../continuous/gamma.ts'
import { normal } from '../continuous/normal.ts'
import { bernoulli } from '../discrete/binary.ts'
import { categorical } from '../discrete/categorical.ts'
import { negativeBinomial, poisson } from '../discrete/counts.ts'
import type { Distribution } from '../distribution.ts'
import type { DispersionFamily } from '../families/dispersion.ts'
import type { DistributionalFamily } from '../families/distributional.ts'
import type { Link } from '../families/links.ts'
import { type CovarianceParams, type MultivariateNormal, multivariateNormal } from '../multivariate/normal.ts'
import { type Mixture, mixtureSameFamily } from './mixture.ts'

/**
 * A conditional distribution p(y | x): a function from inputs to distributions, the common form of GLM likelihoods,
 * network heads, mixture density networks and predictive distributions.
 *
 * @concept distributions
 */
export interface ConditionalDistribution {
  readonly name: string
  /** The distribution of y given inputs x. */
  condition(x: TensorLike): Distribution
  /** log p(y | x). */
  logProb(y: TensorLike, x: TensorLike): Value
  /** Draws from p(y | x). */
  sample(key: Key, x: TensorLike, shape?: readonly number[]): Value
  /** Reparameterised draws from p(y | x) (when the conditioned distribution has them). */
  rsample(key: Key, x: TensorLike, shape?: readonly number[]): Value
}

/**
 * A conditional distribution from a function of the inputs.
 *
 * @param name - Its name.
 * @param condition - Maps inputs to the distribution of the response.
 * @returns The conditional distribution.
 * @example
 * ```ts
 * import { normal } from 'omai-engine/dists'
 * import { conditional } from 'omai-engine/dists/compose'
 * import { item, mul } from 'omai-engine/tensor'
 * const regression = conditional('linear', (x) => normal(mul(x, 2), 1))
 * item(regression.condition(1.5).mean()) // => 3
 * ```
 */
export function conditional(name: string, condition: (x: Value) => Distribution): ConditionalDistribution {
  const at = (x: TensorLike): Distribution => condition(promoteFloat(x)[0] as Value)
  return {
    name,
    condition: at,
    logProb: (y, x) => at(x).logProb(y),
    sample: (key, x, shape = []) => at(x).sample(key, shape),
    rsample: (key, x, shape = []) => at(x).rsample(key, shape),
  }
}

/** A linear-Gaussian conditional y | x ~ N(A x + b, R), with the Gaussian identities it admits. */
export interface LinearGaussian extends ConditionalDistribution {
  readonly weights: Value
  readonly bias: Value
  /** The noise covariance R. */
  readonly noiseCovariance: Value
  /** p(y) for a Gaussian prior on x: N(A mu + b, A Sigma A^T + R). */
  marginal(prior: MultivariateNormal): MultivariateNormal
  /** p(x | y) for a Gaussian prior on x (the Kalman update). */
  posterior(prior: MultivariateNormal, y: TensorLike): MultivariateNormal
  /** The joint p(x, y) as one normal over the concatenation [x, y]. */
  joint(prior: MultivariateNormal): MultivariateNormal
}

const transposeLast = (m: Value): Value => swapaxes(m, -1, -2)
const symmetrise = (m: Value): Value => mul(add(m, transposeLast(m)), 0.5)

/**
 * The linear-Gaussian conditional y | x ~ N(A x + b, R) (Bishop, section 2.3.3), the building block of linear
 * dynamical systems, factor analysis and Gaussian process regression, with closed-form marginals, posteriors (in the
 * numerically stable Kalman-gain form) and joints.
 *
 * @param weights - A `[m, n]`.
 * @param bias - b `[m]`.
 * @param noise - The noise covariance R as `{ covariance }`, `{ precision }` or `{ scaleTril }`.
 * @returns The conditional distribution.
 * @throws OmaiError with code `shape` when the shapes disagree.
 * @example
 * ```ts
 * import { multivariateNormal } from 'omai-engine/dists'
 * import { linearGaussian } from 'omai-engine/dists/compose'
 * import { toArray } from 'omai-engine/tensor'
 * const obs = linearGaussian([[1]], [0], { covariance: [[1]] })
 * // A unit prior and a unit-noise observation of 2: the posterior mean is 1 and its variance 1/2.
 * toArray(obs.posterior(multivariateNormal([0], { covariance: [[1]] }), [2]).loc) // => [1]
 * ```
 */
export function linearGaussian(weights: TensorLike, bias: TensorLike, noise: CovarianceParams): LinearGaussian {
  const [a, b] = promoteFloat(weights, bias) as [Value, Value]
  if (a.shape.length !== 2) throw new OmaiError('shape', 'linearGaussian: weights must be [m, n]')
  const m = a.shape[0] ?? 0
  if ((b.shape[b.shape.length - 1] ?? -1) !== m)
    throw new OmaiError('shape', `linearGaussian: the bias must end in ${m}`)
  const noiseDist = multivariateNormal(b, noise)
  const r = noiseDist.covariance()
  const mean = (x: Value): Value => add(squeeze(matmul(a, expandDims(x, -1)), -1), b)
  const base = conditional('linearGaussian', (x) => multivariateNormal(mean(x), { scaleTril: noiseDist.scaleTril }))
  const predictive = (
    prior: MultivariateNormal,
  ): { readonly mean: Value; readonly cov: Value; readonly cross: Value } => {
    const sigma = prior.covariance()
    const cross = matmul(sigma, transposeLast(a))
    return { mean: mean(prior.loc), cov: symmetrise(add(matmul(a, cross), r)), cross }
  }
  return {
    ...base,
    weights: a,
    bias: b,
    noiseCovariance: r,
    marginal: (prior) => {
      const p = predictive(prior)
      return multivariateNormal(p.mean, { covariance: p.cov })
    },
    posterior: (prior, y) => {
      const p = predictive(prior)
      const residual = sub(promoteFloat(y)[0] as Value, p.mean)
      // K = Sigma A^T S^{-1}, mean mu + K r, covariance Sigma - K S K^T = Sigma - K (A Sigma).
      const gain = transposeLast(solve(p.cov, transposeLast(p.cross)))
      const loc = add(prior.loc, squeeze(matmul(gain, expandDims(residual, -1)), -1))
      const cov = symmetrise(sub(prior.covariance(), matmul(gain, transposeLast(p.cross))))
      return multivariateNormal(loc, { covariance: cov })
    },
    joint: (prior) => {
      const p = predictive(prior)
      const top = concat([prior.covariance(), p.cross], -1)
      const bottom = concat([transposeLast(p.cross), p.cov], -1)
      return multivariateNormal(concat([prior.loc, p.mean], -1), { covariance: concat([top, bottom], -2) })
    },
  }
}

/** Options of {@link glmConditional}. */
export interface GlmOptions {
  /** Coefficients beta `[p]`. */
  readonly coefficients: TensorLike
  /** Intercept (default 0). */
  readonly intercept?: TensorLike
  /** The link (default the family's). */
  readonly link?: Link
  /** The dispersion phi (default 1). */
  readonly scale?: TensorLike
}

/**
 * A generalised linear model as a conditional distribution: x `[..., p]` maps to the family's distribution with mean
 * g^{-1}(x . beta + beta_0) and dispersion phi.
 *
 * @param family - The exponential dispersion family.
 * @param options - Coefficients, intercept, link and scale.
 * @returns The conditional distribution.
 * @example
 * ```ts
 * import { families } from 'omai-engine/dists'
 * import { glmConditional } from 'omai-engine/dists/compose'
 * import { item } from 'omai-engine/tensor'
 * const logistic = glmConditional(families.binomial(), { coefficients: [2], intercept: -1 })
 * item(logistic.condition([0.5]).mean()) // => 0.5
 * ```
 */
export function glmConditional(family: DispersionFamily, options: GlmOptions): ConditionalDistribution {
  const [beta0, intercept, scale] = promoteFloat(options.coefficients, options.intercept ?? 0, options.scale ?? 1) as [
    Value,
    Value,
    Value,
  ]
  const link = options.link ?? family.defaultLink
  return conditional(`glm(${family.name})`, (x) => {
    const eta = add(squeeze(matmul(x, expandDims(beta0, -1)), -1), intercept)
    return family.distribution(link.inverse(eta), scale)
  })
}

/**
 * A distributional (GAMLSS) regression as a conditional distribution: each parameter's linear predictor is its own
 * function of the inputs.
 *
 * @param family - The distributional family.
 * @param predictors - One function per parameter, from inputs to that parameter's linear predictor.
 * @returns The conditional distribution.
 * @example
 * ```ts
 * import { families } from 'omai-engine/dists'
 * import { distributionalConditional } from 'omai-engine/dists/compose'
 * import { item, mul } from 'omai-engine/tensor'
 * const het = distributionalConditional(families.gamlssNormal(), [(x) => x, (x) => mul(x, 0)])
 * item(het.condition(2).mean()) // => 2
 * ```
 */
export function distributionalConditional(
  family: DistributionalFamily,
  predictors: readonly ((x: Value) => Value)[],
): ConditionalDistribution {
  return conditional(`distributional(${family.name})`, (x) => family.fromPredictors(predictors.map((f) => f(x))))
}

const part = (raw: Value, i: number): Value => squeeze(sliceAxis(raw, -1, i, i + 1), -1)

function lastAxis(raw: TensorLike, size: number, what: string): Value {
  const [v] = promoteFloat(raw) as [Value]
  if ((v.shape[v.shape.length - 1] ?? -1) !== size)
    throw new OmaiError('shape', `${what}: the output's last axis must have ${size} entries`)
  return v
}

/**
 * A normal distribution from a network output `[..., 2]`: loc = raw[..., 0], scale = softplus(raw[..., 1]) +
 * minScale.
 *
 * @param raw - The output.
 * @param minScale - A floor on the scale (default 1e-6).
 * @returns The distribution.
 * @example
 * ```ts
 * import { normalFromOutput } from 'omai-engine/dists/compose'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(normalFromOutput([1, 0], 0).stddev()) - Math.LN2) < 1e-15 // => true
 * ```
 */
export function normalFromOutput(raw: TensorLike, minScale = 1e-6): Distribution {
  const v = lastAxis(raw, 2, 'normalFromOutput')
  return normal(part(v, 0), add(softplus(part(v, 1)), minScale))
}

/**
 * A Bernoulli distribution from logits.
 *
 * @param logits - The output.
 * @returns The distribution.
 * @example
 * ```ts
 * import { bernoulliFromOutput } from 'omai-engine/dists/compose'
 * import { item } from 'omai-engine/tensor'
 * item(bernoulliFromOutput(0).mean()) // => 0.5
 * ```
 */
export function bernoulliFromOutput(logits: TensorLike): Distribution {
  return bernoulli({ logits })
}

/**
 * A categorical distribution from logits along the last axis.
 *
 * @param logits - The output `[..., K]`.
 * @returns The distribution.
 * @example
 * ```ts
 * import { categoricalFromOutput } from 'omai-engine/dists/compose'
 * import { item } from 'omai-engine/tensor'
 * item(categoricalFromOutput([0, 0]).prob(1)) // => 0.5
 * ```
 */
export function categoricalFromOutput(logits: TensorLike): Distribution {
  return categorical({ logits })
}

/**
 * A Poisson distribution from a log rate.
 *
 * @param logRate - The output.
 * @returns The distribution.
 * @example
 * ```ts
 * import { poissonFromOutput } from 'omai-engine/dists/compose'
 * import { item } from 'omai-engine/tensor'
 * item(poissonFromOutput(0).mean()) // => 1
 * ```
 */
export function poissonFromOutput(logRate: TensorLike): Distribution {
  return poisson(exp(promoteFloat(logRate)[0] as Value))
}

/**
 * A negative binomial (NB2) distribution from an output `[..., 2]`: mean exp(raw[..., 0]) and dispersion
 * softplus(raw[..., 1]), so the variance is mu + alpha mu^2.
 *
 * @param raw - The output.
 * @returns The distribution.
 * @example
 * ```ts
 * import { negativeBinomialFromOutput } from 'omai-engine/dists/compose'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(negativeBinomialFromOutput([0, 0]).mean()) - 1) < 1e-14 // => true
 * ```
 */
export function negativeBinomialFromOutput(raw: TensorLike): Distribution {
  const v = lastAxis(raw, 2, 'negativeBinomialFromOutput')
  const mu = exp(part(v, 0))
  const alpha = add(softplus(part(v, 1)), 1e-8)
  const r = div(1, alpha)
  return negativeBinomial(r, div(r, add(r, mu)))
}

/**
 * A beta distribution from an output `[..., 2]`: a = softplus(raw[..., 0]) + eps, b = softplus(raw[..., 1]) + eps.
 *
 * @param raw - The output.
 * @returns The distribution.
 * @example
 * ```ts
 * import { betaFromOutput } from 'omai-engine/dists/compose'
 * import { item } from 'omai-engine/tensor'
 * item(betaFromOutput([0, 0]).mean()) // => 0.5
 * ```
 */
export function betaFromOutput(raw: TensorLike): Distribution {
  const v = lastAxis(raw, 2, 'betaFromOutput')
  return beta(add(softplus(part(v, 0)), 1e-6), add(softplus(part(v, 1)), 1e-6))
}

/**
 * A gamma distribution from an output `[..., 2]`: concentration softplus(raw[..., 0]) and rate softplus(raw[..., 1])
 * (each plus eps).
 *
 * @param raw - The output.
 * @returns The distribution.
 * @example
 * ```ts
 * import { gammaFromOutput } from 'omai-engine/dists/compose'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(gammaFromOutput([0, 0]).mean()) - 1) < 1e-5 // => true
 * ```
 */
export function gammaFromOutput(raw: TensorLike): Distribution {
  const v = lastAxis(raw, 2, 'gammaFromOutput')
  return gamma(add(softplus(part(v, 0)), 1e-6), add(softplus(part(v, 1)), 1e-6))
}

/**
 * A mixture density network head (Bishop, 1994): an output `[..., 3K]` read as K mixture logits, K means and K raw
 * scales (through softplus) of a mixture of normals.
 *
 * @param raw - The output.
 * @param components - K.
 * @returns The mixture (reparameterised through its CDF).
 * @example
 * ```ts
 * import { mixtureDensityFromOutput } from 'omai-engine/dists/compose'
 * import { item } from 'omai-engine/tensor'
 * item(mixtureDensityFromOutput([0, 0, -1, 1, 0, 0], 2).mean()) // => 0
 * ```
 */
export function mixtureDensityFromOutput(raw: TensorLike, components: number): Mixture {
  const k = components
  const v = lastAxis(raw, 3 * k, 'mixtureDensityFromOutput')
  const logits = sliceAxis(v, -1, 0, k)
  const locs = sliceAxis(v, -1, k, 2 * k)
  const scales = add(softplus(sliceAxis(v, -1, 2 * k, 3 * k)), 1e-6)
  return mixtureSameFamily({ logits }, normal(locs, scales))
}

/**
 * A multivariate normal from an output `[..., d + d (d + 1) / 2]`: the mean, then the lower Cholesky factor of the
 * covariance filled row by row (diagonal through exp), so every output is a valid distribution.
 *
 * @param raw - The output.
 * @param dim - d.
 * @returns The distribution.
 * @example
 * ```ts
 * import { multivariateNormalFromOutput } from 'omai-engine/dists/compose'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(multivariateNormalFromOutput([1, 2, 0, 0, 0], 2).covariance()) // => [[1, 0], [0, 1]]
 * ```
 */
export function multivariateNormalFromOutput(raw: TensorLike, dim: number): MultivariateNormal {
  const v = lastAxis(raw, dim + (dim * (dim + 1)) / 2, 'multivariateNormalFromOutput')
  const loc = sliceAxis(v, -1, 0, dim)
  const tril = lowerCholesky().forward(sliceAxis(v, -1, dim, dim + (dim * (dim + 1)) / 2))
  return multivariateNormal(loc, { scaleTril: tril })
}
