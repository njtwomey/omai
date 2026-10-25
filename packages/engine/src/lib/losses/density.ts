/**
 * Likelihood losses for probabilistic models: the negative log-likelihood of observations under any distribution or of
 * targets under a conditional distribution (a likelihood head), the mixture density network loss, the heteroscedastic
 * Gaussian loss with a predicted log-variance, and the likelihood of censored observations known only to lie in an
 * interval.
 */

import { MdmlError } from '../base/index.ts'
import type { Distribution } from '../dists/index.ts'
import { log1mexp } from '../numerics/special/index.ts'
import {
  add,
  asTensor,
  div,
  exp,
  expandDims,
  greater,
  log,
  logSoftmax,
  logsumexp,
  mul,
  neg,
  promoteFloat,
  reshape,
  square,
  stopGradient,
  sub,
  sum,
  type TensorLike,
  tensor,
  toFlat,
  type Value,
  where,
} from '../tensor/index.ts'
import { type LossOptions, reduce } from './_reduce.ts'

/**
 * Negative log-likelihood of observations under any distribution: -log p(x), reduced. Gradients flow to the
 * distribution's parameters (and to x where the density is differentiable in it).
 *
 * @param dist - The distribution (its batch shape broadcasts against x).
 * @param x - Observations.
 * @param options - Reduction and weights.
 * @returns The loss, differentiable.
 * @concept maximum-likelihood
 * @example
 * ```ts
 * import { negativeLogLikelihood } from 'mdml-engine/losses'
 * import { normal } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(negativeLogLikelihood(normal(0, 1), [0])) - 0.5 * Math.log(2 * Math.PI)) < 1e-15 // => true
 * ```
 */
export function negativeLogLikelihood(dist: Distribution, x: TensorLike, options: LossOptions = {}): Value {
  return reduce(neg(dist.logProb(x)), options)
}

/** A conditional distribution: a function from inputs to the distribution of the targets (a likelihood head). */
export type ConditionalLike = (inputs: Value) => Distribution

/**
 * Negative log-likelihood of targets under a conditional distribution of the inputs, -log p(y | x): the loss of any
 * probabilistic regressor or classifier head (a GLM likelihood, a network that outputs distribution parameters).
 *
 * @param model - Inputs to distribution.
 * @param inputs - Inputs x.
 * @param targets - Targets y.
 * @param options - Reduction and weights.
 * @returns The loss, differentiable in whatever the model closes over.
 * @example
 * ```ts
 * import { conditionalNll } from 'mdml-engine/losses'
 * import { normal } from 'mdml-engine/dists'
 * import { item, type Value } from 'mdml-engine/tensor'
 * const model = (x: Value) => normal(x, 1)
 * Math.abs(item(conditionalNll(model, [1, 2], [1, 2])) - 0.5 * Math.log(2 * Math.PI)) < 1e-15 // => true
 * ```
 */
export function conditionalNll(
  model: ConditionalLike,
  inputs: TensorLike,
  targets: TensorLike,
  options: LossOptions = {},
): Value {
  return negativeLogLikelihood(model(asTensor(inputs)), targets, options)
}

/**
 * Mixture-density-network loss (Bishop, 1994): the negative log-likelihood of targets under a mixture of Gaussians with
 * diagonal covariances, -log sum_k pi_k N(y; mu_k, diag(sigma_k^2)), computed with log-sum-exp.
 *
 * @param logits - Mixture logits `[N, K]` (softmax gives the weights pi).
 * @param means - Component means `[N, K]` (scalar targets) or `[N, K, D]`.
 * @param scales - Component standard deviations, the shape of `means` (positive).
 * @param target - Targets `[N]` or `[N, D]`.
 * @param options - Reduction and weights.
 * @returns The loss, differentiable in logits, means and scales.
 * @example
 * ```ts
 * import { mixtureDensityNll } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(mixtureDensityNll([[0, 0]], [[0, 0]], [[1, 1]], [0])) - 0.5 * Math.log(2 * Math.PI)) < 1e-15 // => true
 * ```
 */
export function mixtureDensityNll(
  logits: TensorLike,
  means: TensorLike,
  scales: TensorLike,
  target: TensorLike,
  options: LossOptions = {},
): Value {
  const [z, mu, sigma, y] = promoteFloat(logits, means, scales, target) as [Value, Value, Value, Value]
  const multivariate = mu.shape.length === 3
  const yk = expandDims(y, 1)
  const standardised = div(sub(yk, mu), sigma)
  let logN = sub(mul(square(standardised), -0.5), add(log(sigma), 0.5 * Math.log(2 * Math.PI)))
  if (multivariate) logN = sum(logN, -1)
  return reduce(neg(logsumexp(add(logSoftmax(z, -1), logN), -1)), options)
}

/** Options of {@link heteroscedasticGaussianNll}. */
export interface HeteroscedasticOptions extends LossOptions {
  /** Add the constant 0.5 log(2 pi). */
  readonly full?: boolean
  /**
   * Beta-NLL (Seitzer et al., 2022): weight each term by stopGradient(variance)^beta, so that badly fitted means are
   * not ignored by inflating the variance (0 is the plain NLL, 1 gives mean gradients like the squared error).
   */
  readonly beta?: number
}

/**
 * Heteroscedastic Gaussian negative log-likelihood with a predicted log-variance (Nix and Weigend, 1994; Kendall and
 * Gal, 2017): 0.5 (log sigma^2 + (y - mu)^2 / sigma^2), parameterised by s = log sigma^2 for stable training, with
 * the beta-NLL weighting as an option.
 *
 * @param mean - Predicted means.
 * @param logVariance - Predicted log-variances.
 * @param target - Targets.
 * @param options - full, beta, reduction and weights.
 * @returns The loss, differentiable in means and log-variances.
 * @example
 * ```ts
 * import { heteroscedasticGaussianNll } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(heteroscedasticGaussianNll([0], [0], [2])) // => 2
 * ```
 */
export function heteroscedasticGaussianNll(
  mean: TensorLike,
  logVariance: TensorLike,
  target: TensorLike,
  options: HeteroscedasticOptions = {},
): Value {
  const [mu, s, y] = promoteFloat(mean, logVariance, target) as [Value, Value, Value]
  let terms = mul(add(s, mul(square(sub(y, mu)), exp(neg(s)))), 0.5)
  if (options.full === true) terms = add(terms, 0.5 * Math.log(2 * Math.PI))
  const beta = options.beta ?? 0
  if (beta !== 0) terms = mul(terms, stopGradient(exp(mul(s, beta))))
  return reduce(terms, options)
}

/**
 * Negative log-likelihood of censored observations: each observation is known to lie in [lower, upper]. lower =
 * upper is an exact observation (log density), upper = +inf is right-censored (log survival at lower), lower = -inf
 * is left-censored (log CDF at upper), and otherwise the interval's log probability log(F(upper) - F(lower)), computed
 * as log F(u) + log(1 - F(l) / F(u)). The survival-analysis and Tobit likelihoods are special cases.
 *
 * @param dist - The distribution (its batch shape broadcasts against the bounds).
 * @param lower - Lower bounds (-Infinity for left-censored).
 * @param upper - Upper bounds (+Infinity for right-censored; equal to lower for exact).
 * @param options - Reduction and weights.
 * @returns The loss, differentiable in the distribution's parameters.
 * @throws MdmlError with code `value` for an interval that is unbounded on both sides, or with lower > upper.
 * @example
 * ```ts
 * import { censoredNll } from 'mdml-engine/losses'
 * import { normal } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(censoredNll(normal(0, 1), [0], [Infinity])) === Math.LN2 // => true
 * ```
 */
export function censoredNll(
  dist: Distribution,
  lower: TensorLike,
  upper: TensorLike,
  options: LossOptions = {},
): Value {
  const lo = asTensor(lower, 'float64')
  const hi = asTensor(upper, 'float64')
  const l = toFlat(lo)
  const u = toFlat(hi)
  if (l.length !== u.length) throw new MdmlError('shape', 'censoredNll: lower and upper must have the same shape')
  const kinds = l.map((a, i): Kind => {
    const b = u[i] ?? Number.NaN
    if (!(a <= b)) throw new MdmlError('value', `censoredNll: lower ${a} exceeds upper ${b}`)
    if (a === -Infinity && b === Infinity)
      throw new MdmlError('value', 'censoredNll: (-inf, inf) carries no information')
    if (a === b) return 'exact'
    if (b === Infinity) return 'right'
    if (a === -Infinity) return 'left'
    return 'interval'
  })
  const shape = lo.shape
  // Each branch sees, at positions of other kinds, the bounds of one of its own observations, so its value and
  // gradient stay finite there before `where` discards them.
  const fill = (kind: Kind, values: readonly number[]): Value | undefined => {
    const first = kinds.indexOf(kind)
    if (first < 0) return undefined
    const ref = values[first] ?? 0
    return reshape(tensor(values.map((v, i) => (kinds[i] === kind ? v : ref))), shape)
  }
  const mask = (kind: Kind): Value => reshape(tensor(kinds.map((k) => k === kind)), shape)
  let terms: Value | undefined
  const accumulate = (kind: Kind, value: Value): void => {
    terms = where(mask(kind), value, terms ?? 0)
  }
  const exact = fill('exact', l)
  if (exact !== undefined) accumulate('exact', dist.logProb(exact))
  const right = fill('right', l)
  if (right !== undefined) accumulate('right', dist.logSf(right))
  const left = fill('left', u)
  if (left !== undefined) accumulate('left', dist.logCdf(left))
  const a = fill('interval', l)
  const b = fill('interval', u)
  if (a !== undefined && b !== undefined) {
    // log(F(b) - F(a)) from the CDF below the median and from the survival function above it.
    const logFa = dist.logCdf(a)
    const logFb = dist.logCdf(b)
    const logSa = dist.logSf(a)
    const logSb = dist.logSf(b)
    const fromCdf = add(logFb, log1mexp(sub(logFa, logFb)))
    const fromSf = add(logSa, log1mexp(sub(logSb, logSa)))
    accumulate('interval', where(greater(logFa, -Math.LN2), fromSf, fromCdf))
  }
  return reduce(neg(terms ?? tensor(0)), options)
}

type Kind = 'exact' | 'right' | 'left' | 'interval'
