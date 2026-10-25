/**
 * Training objectives of generative models: the GAN discriminator and generator losses (minimax, non-saturating,
 * Wasserstein, hinge and least squares) with the WGAN-GP and R1 gradient penalties, the VAE's KL terms, reconstruction
 * likelihoods, negative ELBO and IWAE bound, denoising score matching, the diffusion loss with noise, clean-sample or
 * velocity targets, and conditional flow matching.
 */

import { grad } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import type { Distribution } from '../dists/index.ts'
import { softplus } from '../numerics/special/index.ts'
import { type Key, uniform } from '../random/index.ts'
import {
  add,
  div,
  exp,
  logsumexp,
  mean,
  mul,
  neg,
  promoteFloat,
  relu,
  reshape,
  sqrt,
  square,
  sub,
  sum,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'
import { type Reduction, reduce } from './_reduce.ts'

/** The GAN objectives: the original minimax game, its non-saturating generator, WGAN, hinge and least squares. */
export type GanObjective = 'minimax' | 'non-saturating' | 'wasserstein' | 'hinge' | 'least-squares'

/**
 * The discriminator's (critic's) loss on real and generated samples, from its raw outputs (logits):
 * minimax and non-saturating: softplus(-D(x)) + softplus(D(G(z))) (Goodfellow et al., 2014); wasserstein:
 * D(G(z)) - D(x) (Arjovsky et al., 2017); hinge: relu(1 - D(x)) + relu(1 + D(G(z))) (Miyato et al., 2018);
 * least-squares: 0.5 (D(x) - 1)^2 + 0.5 D(G(z))^2 (Mao et al., 2017). Each term is averaged over its batch.
 *
 * @param realLogits - Discriminator outputs on real samples.
 * @param fakeLogits - Discriminator outputs on generated samples.
 * @param options - The objective (default 'non-saturating').
 * @returns The scalar loss, differentiable in both outputs.
 * @concept generative-adversarial-networks
 * @example
 * ```ts
 * import { discriminatorLoss } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(discriminatorLoss([2, 0], [-1, 1], { objective: 'hinge' })) // => 1.5
 * ```
 */
export function discriminatorLoss(
  realLogits: TensorLike,
  fakeLogits: TensorLike,
  options: { readonly objective?: GanObjective } = {},
): Value {
  const [r, f] = promoteFloat(realLogits, fakeLogits) as [Value, Value]
  switch (options.objective ?? 'non-saturating') {
    case 'minimax':
    case 'non-saturating':
      return add(mean(softplus(neg(r))), mean(softplus(f)))
    case 'wasserstein':
      return sub(mean(f), mean(r))
    case 'hinge':
      return add(mean(relu(sub(1, r))), mean(relu(add(1, f))))
    case 'least-squares':
      return mul(add(mean(square(sub(r, 1))), mean(square(f))), 0.5)
    default:
      throw new MdmlError('value', `unknown GAN objective ${String(options.objective)}`)
  }
}

/**
 * The generator's loss from the discriminator's outputs on generated samples: minimax: -softplus(D(G(z))) (that is,
 * log(1 - sigma(D))); non-saturating: softplus(-D(G(z))); wasserstein and hinge: -D(G(z)); least-squares:
 * 0.5 (D(G(z)) - 1)^2. Averaged over the batch.
 *
 * @param fakeLogits - Discriminator outputs on generated samples.
 * @param options - The objective (default 'non-saturating').
 * @returns The scalar loss, differentiable in the outputs.
 * @example
 * ```ts
 * import { generatorLoss } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(generatorLoss([0], { objective: 'non-saturating' })) === Math.LN2 // => true
 * ```
 */
export function generatorLoss(fakeLogits: TensorLike, options: { readonly objective?: GanObjective } = {}): Value {
  const [f] = promoteFloat(fakeLogits) as [Value]
  switch (options.objective ?? 'non-saturating') {
    case 'minimax':
      return neg(mean(softplus(f)))
    case 'non-saturating':
      return mean(softplus(neg(f)))
    case 'wasserstein':
    case 'hinge':
      return neg(mean(f))
    case 'least-squares':
      return mul(mean(square(sub(f, 1))), 0.5)
    default:
      throw new MdmlError('value', `unknown GAN objective ${String(options.objective)}`)
  }
}

/** A critic: a function from a batch of samples `[N, ...]` to one score per sample `[N]`. */
export type Critic = (x: Value) => Value

/** The per-sample squared input-gradient norms ||grad_x D(x)||^2 `[N]` of a critic (samples are independent). */
function squaredGradientNorms(critic: Critic, x: Value): Value {
  const g = grad((v: Value) => sum(critic(v)))(x)
  const n = x.shape[0] ?? 1
  return sum(square(reshape(g, [n, -1])), 1)
}

/**
 * WGAN-GP gradient penalty (Gulrajani et al., 2017): weight E[(||grad D(x_hat)||_2 - target)^2] at random
 * interpolates x_hat = alpha x + (1 - alpha) x~ between real and generated samples. The penalty is differentiable
 * in the critic's parameters (a gradient of a gradient).
 *
 * @param critic - Samples to scores.
 * @param real - Real samples `[N, ...]`.
 * @param fake - Generated samples `[N, ...]`.
 * @param options - weight (default 10), target norm (default 1), and the interpolation: `alpha` `[N]` given, or
 *   drawn uniformly with `key`.
 * @returns The scalar penalty.
 * @throws MdmlError with code `value` when neither `alpha` nor `key` is given.
 * @example
 * ```ts
 * import { gradientPenalty } from 'mdml-engine/losses'
 * import { item, mul, sum, type Value } from 'mdml-engine/tensor'
 * const critic = (x: Value) => sum(mul(x, 3), 1)
 * item(gradientPenalty(critic, [[0, 0]], [[1, 1]], { alpha: [0.5], weight: 1 })) === (Math.sqrt(18) - 1) ** 2 // => true
 * ```
 */
export function gradientPenalty(
  critic: Critic,
  real: TensorLike,
  fake: TensorLike,
  options: { readonly weight?: number; readonly target?: number; readonly alpha?: TensorLike; readonly key?: Key } = {},
): Value {
  const [x, y] = promoteFloat(real, fake) as [Value, Value]
  const n = x.shape[0] ?? 1
  let alpha: Value
  if (options.alpha !== undefined) alpha = promoteFloat(options.alpha)[0] as Value
  else if (options.key !== undefined) alpha = uniform(options.key, [n])
  else throw new MdmlError('value', 'gradientPenalty: pass `alpha` or a `key`')
  const a = reshape(alpha, [n, ...x.shape.slice(1).map(() => 1)])
  const interpolates = add(mul(a, x), mul(sub(1, a), y))
  const norms = sqrt(squaredGradientNorms(critic, interpolates))
  return mul(mean(square(sub(norms, options.target ?? 1))), options.weight ?? 10)
}

/**
 * R1 regularisation (Mescheder, Geiger and Nowozin, 2018): (gamma / 2) E||grad D(x)||^2 on real samples, the
 * zero-centred penalty of StyleGAN. Pass generated samples for the R2 variant.
 *
 * @param critic - Samples to scores.
 * @param samples - Real samples (R1) or generated samples (R2) `[N, ...]`.
 * @param options - gamma (default 10).
 * @returns The scalar penalty, differentiable in the critic's parameters.
 * @example
 * ```ts
 * import { r1Penalty } from 'mdml-engine/losses'
 * import { item, mul, sum, type Value } from 'mdml-engine/tensor'
 * item(r1Penalty((x: Value) => sum(mul(x, 2), 1), [[1, 1]], { gamma: 1 })) // => 4
 * ```
 */
export function r1Penalty(critic: Critic, samples: TensorLike, options: { readonly gamma?: number } = {}): Value {
  const [x] = promoteFloat(samples) as [Value]
  return mul(mean(squaredGradientNorms(critic, x)), (options.gamma ?? 10) / 2)
}

/** Sum over every axis but the first: one value per sample. */
const perSample = (v: Value): Value => (v.shape.length <= 1 ? v : sum(reshape(v, [v.shape[0] ?? 1, -1]), 1))

/**
 * KL(N(mu, exp(s)) || N(0, I)) of a diagonal Gaussian posterior against the standard normal prior, the VAE's
 * regulariser (Kingma and Welling, 2014): 0.5 sum_d (mu_d^2 + exp(s_d) - 1 - s_d), summed over the latent axes of each
 * sample.
 *
 * @param mean - Posterior means `[N, ...]`.
 * @param logVariance - Posterior log-variances, the shape of `mean`.
 * @param options - Reduction over samples (default 'mean').
 * @returns The KL term, differentiable.
 * @concept variational-autoencoders
 * @example
 * ```ts
 * import { gaussianKl } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(gaussianKl([[0, 0]], [[0, 0]])) // => 0
 * ```
 */
export function gaussianKl(
  mean: TensorLike,
  logVariance: TensorLike,
  options: { readonly reduction?: Reduction } = {},
): Value {
  const [mu, s] = promoteFloat(mean, logVariance) as [Value, Value]
  const terms = mul(sub(sub(add(square(mu), exp(s)), 1), s), 0.5)
  return reduce(perSample(terms), { reduction: options.reduction ?? 'mean' })
}

/**
 * Monte Carlo KL(q || p) from samples z ~ q: mean of log q(z) - log p(z), the estimator for pairs without a closed form.
 * With reparameterised samples (`q.rsample`) it is differentiable in q's parameters.
 *
 * @param q - The approximating distribution.
 * @param p - The reference distribution.
 * @param z - Samples from q (leading axes are averaged).
 * @returns The estimate, differentiable.
 * @example
 * ```ts
 * import { sampledKl } from 'mdml-engine/losses'
 * import { normal } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(sampledKl(normal(0, 1), normal(0, 1), [0.3, -1])) // => 0
 * ```
 */
export function sampledKl(q: Distribution, p: Distribution, z: TensorLike): Value {
  return mean(sub(q.logProb(z), p.logProb(z)))
}

/**
 * The negative evidence lower bound, -E[log p(x | z)] + beta KL(q(z | x) || p(z)), from per-sample reconstruction
 * log-likelihoods and KL terms (beta-VAE weighting; beta = 1 is the ELBO).
 *
 * @param reconstructionLogLik - log p(x | z) per sample `[N]`.
 * @param kl - KL per sample `[N]` (or already reduced).
 * @param options - beta (default 1) and reduction over samples.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { negativeElbo } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(negativeElbo([-3, -1], [0.5, 0.5], { beta: 2 })) // => 3
 * ```
 */
export function negativeElbo(
  reconstructionLogLik: TensorLike,
  kl: TensorLike,
  options: { readonly beta?: number; readonly reduction?: Reduction } = {},
): Value {
  const [ll, k] = promoteFloat(reconstructionLogLik, kl) as [Value, Value]
  return reduce(sub(mul(k, options.beta ?? 1), ll), { reduction: options.reduction ?? 'mean' })
}

/**
 * Bernoulli reconstruction log-likelihood of binary (or [0, 1]) data from decoder logits, summed over each sample's
 * event axes: sum x log sigma(l) + (1 - x) log sigma(-l).
 *
 * @param logits - Decoder logits `[N, ...]`.
 * @param x - Data `[N, ...]` in [0, 1].
 * @returns log p(x | z) per sample `[N]`, differentiable.
 * @example
 * ```ts
 * import { bernoulliLogLikelihood } from 'mdml-engine/losses'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(bernoulliLogLikelihood([[0, 0]], [[1, 0]])).map((v) => v / Math.LN2) // => [-2]
 * ```
 */
export function bernoulliLogLikelihood(logits: TensorLike, x: TensorLike): Value {
  const [l, y] = promoteFloat(logits, x) as [Value, Value]
  return perSample(neg(add(mul(y, softplus(neg(l))), mul(sub(1, y), softplus(l)))))
}

/**
 * Gaussian reconstruction log-likelihood with a fixed or predicted log-variance, summed over each sample's event axes:
 * -0.5 sum ((x - mu)^2 / sigma^2 + log(2 pi sigma^2)).
 *
 * @param mean - Decoder means `[N, ...]`.
 * @param x - Data `[N, ...]`.
 * @param options - logVariance (default 0, unit variance; broadcastable).
 * @returns log p(x | z) per sample `[N]`, differentiable.
 * @example
 * ```ts
 * import { gaussianLogLikelihood } from 'mdml-engine/losses'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(gaussianLogLikelihood([[0]], [[0]])).map((v) => v + 0.5 * Math.log(2 * Math.PI)) // => [0]
 * ```
 */
export function gaussianLogLikelihood(
  mean: TensorLike,
  x: TensorLike,
  options: { readonly logVariance?: TensorLike } = {},
): Value {
  const [mu, y] = promoteFloat(mean, x) as [Value, Value]
  const s = options.logVariance === undefined ? 0 : (promoteFloat(options.logVariance)[0] as Value)
  const terms = add(mul(square(sub(y, mu)), exp(neg(s))), add(s, Math.log(2 * Math.PI)))
  return perSample(mul(terms, -0.5))
}

/**
 * The importance-weighted bound of the IWAE (Burda, Grosse and Salakhutdinov, 2016), negated:
 * -(logsumexp_k log w_k - log K) with log w_k = log p(x, z_k) - log q(z_k | x) for K samples per data point.
 * K = 1 is the negative ELBO.
 *
 * @param logWeights - Log importance weights `[N, K]`.
 * @param options - Reduction over data points.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { importanceWeightedBound } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(importanceWeightedBound([[-2, -2]])) - 2) < 1e-15 // => true
 * ```
 */
export function importanceWeightedBound(
  logWeights: TensorLike,
  options: { readonly reduction?: Reduction } = {},
): Value {
  const [w] = promoteFloat(logWeights) as [Value]
  const k = w.shape[w.shape.length - 1] ?? 1
  return reduce(neg(sub(logsumexp(w, -1), Math.log(k))), { reduction: options.reduction ?? 'mean' })
}

/**
 * Denoising score matching (Vincent, 2011; the noise-conditional objective of Song and Ermon, 2019): for x~ = x +
 * sigma eps, the target score is -eps / sigma and the loss is lambda(sigma) 0.5 ||s(x~) - target||^2 summed over each
 * sample, with the usual weighting lambda = sigma^2 (that is, 0.5 ||sigma s + eps||^2).
 *
 * @param score - Predicted scores s(x~, sigma) `[N, ...]`.
 * @param noise - The standard normal noise eps that was added `[N, ...]`.
 * @param sigma - Noise levels, per sample `[N]` or a scalar.
 * @param options - weighting ('sigma2', default, or 'none') and reduction over samples.
 * @returns The loss, differentiable in the scores.
 * @example
 * ```ts
 * import { denoisingScoreMatching } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(denoisingScoreMatching([[-2]], [[1]], [0.5])) // => 0
 * ```
 */
export function denoisingScoreMatching(
  score: TensorLike,
  noise: TensorLike,
  sigma: TensorLike,
  options: { readonly weighting?: 'sigma2' | 'none'; readonly reduction?: Reduction } = {},
): Value {
  const [s, e, sg] = promoteFloat(score, noise, sigma) as [Value, Value, Value]
  const sig = sg.shape.length === 0 ? sg : reshape(sg, [sg.shape[0] ?? 1, ...s.shape.slice(1).map(() => 1)])
  let terms = perSample(mul(square(add(s, div(e, sig))), 0.5))
  if ((options.weighting ?? 'sigma2') === 'sigma2') terms = mul(terms, square(sg))
  return reduce(terms, { reduction: options.reduction ?? 'mean' })
}

/** What a diffusion model predicts. */
export type DiffusionParameterisation = 'epsilon' | 'x0' | 'v'

/**
 * The regression target of a diffusion model at x_t = alpha_t x0 + sigma_t eps: the noise eps (DDPM, Ho et al., 2020),
 * the clean sample x0, or the velocity v = alpha_t eps - sigma_t x0 (Salimans and Ho, 2022).
 *
 * @param parameterisation - 'epsilon', 'x0' or 'v'.
 * @param x0 - Clean samples `[N, ...]`.
 * @param noise - Noise eps `[N, ...]`.
 * @param alpha - Signal scales alpha_t (broadcastable).
 * @param sigma - Noise scales sigma_t (broadcastable).
 * @returns The target, the shape of x0.
 * @example
 * ```ts
 * import { diffusionTarget } from 'mdml-engine/losses'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(diffusionTarget('v', [1], [2], 0.5, 0.25)) // => [0.75]
 * ```
 */
export function diffusionTarget(
  parameterisation: DiffusionParameterisation,
  x0: TensorLike,
  noise: TensorLike,
  alpha: TensorLike,
  sigma: TensorLike,
): Value {
  const [x, e, a, s] = promoteFloat(x0, noise, alpha, sigma) as [Value, Value, Value, Value]
  if (parameterisation === 'epsilon') return e
  if (parameterisation === 'x0') return x
  return sub(mul(a, e), mul(s, x))
}

/**
 * The simple diffusion loss: the mean squared error between a model's prediction and the target of its
 * parameterisation ({@link diffusionTarget}), averaged over every element as in DDPM's L_simple.
 *
 * @param prediction - Model output `[N, ...]`.
 * @param x0 - Clean samples.
 * @param noise - Noise eps.
 * @param alpha - Signal scales alpha_t.
 * @param sigma - Noise scales sigma_t.
 * @param options - parameterisation (default 'epsilon').
 * @returns The scalar loss, differentiable in the prediction.
 * @example
 * ```ts
 * import { diffusionLoss } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(diffusionLoss([1], [2], [1], 0.6, 0.8)) // => 0
 * ```
 */
export function diffusionLoss(
  prediction: TensorLike,
  x0: TensorLike,
  noise: TensorLike,
  alpha: TensorLike,
  sigma: TensorLike,
  options: { readonly parameterisation?: DiffusionParameterisation } = {},
): Value {
  const target = diffusionTarget(options.parameterisation ?? 'epsilon', x0, noise, alpha, sigma)
  const [p] = promoteFloat(prediction) as [Value]
  return mean(square(sub(p, target)))
}

/**
 * The conditional flow-matching path between noise x0 and data x1 (Lipman et al., 2023; rectified flow at
 * sigmaMin = 0): x_t = (1 - (1 - sigmaMin) t) x0 + t x1 with target velocity u = x1 - (1 - sigmaMin) x0.
 *
 * @param x0 - Source (noise) samples `[N, ...]`.
 * @param x1 - Data samples `[N, ...]`.
 * @param t - Times in [0, 1], per sample `[N]`.
 * @param options - sigmaMin (default 0).
 * @returns The points x_t and target velocities.
 * @example
 * ```ts
 * import { flowMatchingPath } from 'mdml-engine/losses'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(flowMatchingPath([[0]], [[2]], [0.25]).xt) // => [[0.5]]
 * ```
 */
export function flowMatchingPath(
  x0: TensorLike,
  x1: TensorLike,
  t: TensorLike,
  options: { readonly sigmaMin?: number } = {},
): { readonly xt: Value; readonly velocity: Value } {
  const [a, b, tt] = promoteFloat(x0, x1, t) as [Value, Value, Value]
  const k = 1 - (options.sigmaMin ?? 0)
  const time = reshape(tt, [tt.shape[0] ?? 1, ...a.shape.slice(1).map(() => 1)])
  return { xt: add(mul(sub(1, mul(time, k)), a), mul(time, b)), velocity: sub(b, mul(a, k)) }
}

/**
 * The conditional flow-matching loss: the mean squared error between predicted velocities v(x_t, t) and the path's
 * target velocities ({@link flowMatchingPath}), averaged over every element (as torchcfm).
 *
 * @param velocity - Predicted velocities `[N, ...]`.
 * @param x0 - Source samples.
 * @param x1 - Data samples.
 * @param options - sigmaMin (default 0).
 * @returns The scalar loss, differentiable in the predictions.
 * @example
 * ```ts
 * import { flowMatching } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(flowMatching([[2]], [[0]], [[2]])) // => 0
 * ```
 */
export function flowMatching(
  velocity: TensorLike,
  x0: TensorLike,
  x1: TensorLike,
  options: { readonly sigmaMin?: number } = {},
): Value {
  const [v, a, b] = promoteFloat(velocity, x0, x1) as [Value, Value, Value]
  return mean(square(sub(v, sub(b, mul(a, 1 - (options.sigmaMin ?? 0))))))
}
