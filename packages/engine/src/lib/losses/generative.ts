/**
 * Training objectives of generative models: the GAN discriminator and generator losses (minimax, non-saturating,
 * Wasserstein, hinge and least squares) with the WGAN-GP and R1 gradient penalties, the VAE's KL terms, reconstruction
 * likelihoods, negative ELBO and IWAE bound, denoising score matching, the diffusion loss with noise, clean-sample or
 * velocity targets, and conditional flow matching; noise-contrastive estimation of unnormalised densities, exact and
 * sliced score matching, and vector quantisation (nearest code, codebook and commitment losses, the straight-through
 * estimator and the exponential-moving-average codebook update of the VQ-VAE).
 */

import { grad, jvp } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import type { Distribution } from '../dists/index.ts'
import { softplus } from '../numerics/special/index.ts'
import { type Key, normal, split, uniform } from '../random/index.ts'
import {
  add,
  argmin,
  broadcastTo,
  cast,
  div,
  exp,
  eye,
  greater,
  log,
  logsumexp,
  matmul,
  mean,
  mul,
  neg,
  oneHot,
  promoteFloat,
  relu,
  reshape,
  sqrt,
  square,
  stopGradient,
  sub,
  sum,
  type TensorLike,
  take,
  transpose,
  type Value,
  where,
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

/** Options of {@link noiseContrastive}. */
export interface NoiseContrastiveOptions {
  /** A learned log-normaliser c subtracted from the model's log-density (so the model need not be normalised). */
  readonly logNormaliser?: TensorLike
  /** The number of noise samples per data sample, k; default the ratio of the two sample sizes. */
  readonly k?: number
}

/**
 * Noise-contrastive estimation (Gutmann and Hyvarinen, 2010, 2012): fit an unnormalised model by logistic regression
 * of data against noise with a known density q. With k noise samples per data sample the classifier's logit is
 * G(x) = s(x) - c - log k - log q(x), where s is the model's unnormalised log-density and c a learned log-normaliser,
 * and the loss is -mean log sigma(G(data)) - (1/N) sum log sigma(-G(noise)). The minimiser has c = log of the
 * partition function, so normalisation is learned along with the density. The same loss with a language model's score
 * of the true word against k sampled words is NCE for large vocabularies.
 *
 * @param dataScores - The model's unnormalised log-density at the N data samples.
 * @param dataLogNoise - The noise log-density at the data samples.
 * @param noiseScores - The model's unnormalised log-density at the M noise samples.
 * @param noiseLogNoise - The noise log-density at the noise samples.
 * @param options - logNormaliser and k.
 * @returns The scalar loss, differentiable in the scores and the normaliser.
 * @concept noise-contrastive-estimation
 * @example
 * ```ts
 * import { noiseContrastive } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * // The model equal to the noise (and k = 1) cannot tell them apart: 2 log 2.
 * Math.abs(item(noiseContrastive([0], [0], [0], [0])) - 2 * Math.LN2) < 1e-15 // => true
 * ```
 */
export function noiseContrastive(
  dataScores: TensorLike,
  dataLogNoise: TensorLike,
  noiseScores: TensorLike,
  noiseLogNoise: TensorLike,
  options: NoiseContrastiveOptions = {},
): Value {
  const [sd, qd, sn, qn] = promoteFloat(dataScores, dataLogNoise, noiseScores, noiseLogNoise) as [
    Value,
    Value,
    Value,
    Value,
  ]
  const nData = sd.shape.reduce((a, b) => a * b, 1)
  const nNoise = sn.shape.reduce((a, b) => a * b, 1)
  const k = options.k ?? nNoise / nData
  const c = options.logNormaliser === undefined ? 0 : (promoteFloat(options.logNormaliser)[0] as Value)
  const shift = Math.log(k)
  const logit = (s: Value, q: Value): Value => sub(sub(sub(s, c), shift), q)
  const dataTerm = mean(softplus(neg(logit(sd, qd))))
  const noiseTerm = div(sum(softplus(logit(sn, qn))), nData)
  return add(dataTerm, noiseTerm)
}

/** The kind of random projection direction of {@link slicedScoreMatching}. */
export type ProjectionKind = 'gaussian' | 'rademacher'

/**
 * Exact (Hyvarinen) score matching, E[tr(J_s(x)) + 0.5 ||s(x)||^2] with J_s the Jacobian of the score model, which
 * equals the Fisher divergence to the data score up to a constant. The trace costs one forward-mode derivative per
 * dimension, so it suits low dimensions; {@link slicedScoreMatching} estimates it with random projections.
 *
 * @param score - The score model s(x) mapping `[N, D]` to `[N, D]` (rows independent).
 * @param x - Samples `[N, D]`.
 * @returns The scalar loss, differentiable in whatever the score model closes over.
 * @concept score-matching
 * @example
 * ```ts
 * import { scoreMatching } from 'mdml-engine/losses'
 * import { mul, item, type Value } from 'mdml-engine/tensor'
 * // s(x) = -x is the score of a standard normal: tr(J) = -1 per sample, 0.5 |x|^2 averages to 0.5 on [[1], [-1]].
 * item(scoreMatching((x: Value) => mul(x, -1), [[1], [-1]])) // => -0.5
 * ```
 */
export function scoreMatching(score: (x: Value) => Value, x: TensorLike): Value {
  const [data] = promoteFloat(x) as [Value]
  const d = data.shape[1] ?? 1
  const s = score(data)
  const n = data.shape[0] ?? 1
  const traces: Value[] = []
  for (let i = 0; i < d; i++) {
    const e = broadcastTo(reshape(take(eye(d), i), [1, d]), [n, d])
    traces.push(sum(mul(jvp(score, data, e).tangent, e), -1))
  }
  return mean(
    add(
      traces.reduce((a, b) => add(a, b)),
      mul(sum(square(s), -1), 0.5),
    ),
  )
}

/**
 * Sliced score matching (Song et al., 2019): E_v[v^T J_s(x) v + 0.5 (v^T s(x))^2] over random directions v, which
 * estimates the score-matching objective without the Jacobian's trace, each projection costing one forward-mode
 * derivative. With the variance-reduced form (Gaussian v only) the second term is 0.5 ||s(x)||^2.
 *
 * @param score - The score model s(x) mapping `[N, D]` to `[N, D]` (rows independent).
 * @param x - Samples `[N, D]`.
 * @param k - Random key for the directions.
 * @param options - projections per sample (default 1), kind ('gaussian' or 'rademacher') and varianceReduced.
 * @returns The scalar loss, differentiable in whatever the score model closes over.
 * @example
 * ```ts
 * import { slicedScoreMatching } from 'mdml-engine/losses'
 * import { key } from 'mdml-engine/random'
 * import { mul, item, type Value } from 'mdml-engine/tensor'
 * // The score of a standard normal, -x, at the data [[1], [-1]]: v^T J v = -v^2 and (v s)^2 / 2 = (v x)^2 / 2 with v^2 = 1.
 * item(slicedScoreMatching((x: Value) => mul(x, -1), [[1], [-1]], key(0), { kind: 'rademacher' })) // => -0.5
 * ```
 */
export function slicedScoreMatching(
  score: (x: Value) => Value,
  x: TensorLike,
  k: Key,
  options: { readonly projections?: number; readonly kind?: ProjectionKind; readonly varianceReduced?: boolean } = {},
): Value {
  const [data] = promoteFloat(x) as [Value]
  const [n, d] = [data.shape[0] ?? 1, data.shape[1] ?? 1]
  const m = options.projections ?? 1
  const kind = options.kind ?? 'gaussian'
  const s = score(data)
  const terms = split(k, m).map((projectionKey) => {
    const raw = normal(projectionKey, [n, d])
    const v = kind === 'gaussian' ? raw : where(greater(raw, 0), 1, -1)
    const quadratic = sum(mul(v, jvp(score, data, v).tangent), -1)
    const second =
      options.varianceReduced === true && kind === 'gaussian' ? sum(square(s), -1) : square(sum(mul(v, s), -1))
    return add(quadratic, mul(second, 0.5))
  })
  return mean(
    div(
      terms.reduce((a, b) => add(a, b)),
      m,
    ),
  )
}

/** The result of {@link vectorQuantise}. */
export interface VectorQuantisation {
  /** The nearest code of every latent `[N, D]`, with the straight-through gradient to the latents. */
  readonly quantised: Value
  /** The index of the nearest code of every latent `[N]`. */
  readonly indices: Value
  /** mean ||sg(z) - e||^2: moves the codes to the latents (zero when the codebook is updated by EMA). */
  readonly codebookLoss: Value
  /** mean ||z - sg(e)||^2: keeps the encoder's outputs near their codes. */
  readonly commitmentLoss: Value
  /** codebookLoss + beta commitmentLoss, differentiable in the latents and the codebook. */
  readonly loss: Value
  /** The share of the batch assigned to each code `[K]`. */
  readonly usage: Value
  /** exp of the entropy of the usage: the number of codes effectively used (K when uniform, 1 when collapsed). */
  readonly perplexity: Value
}

/**
 * Vector quantisation of the VQ-VAE (van den Oord et al., 2017): each latent is replaced by its nearest code in a
 * codebook; the loss has a codebook term ||sg(z) - e||^2 (moves the codes), and a commitment term beta ||z - sg(e)||^2
 * (keeps the encoder near its codes), where sg stops gradients. The quantised output passes the decoder's gradient
 * straight through to the latents, z + sg(e - z).
 *
 * @param latents - Encoder outputs `[N, D]`.
 * @param codebook - The codes `[K, D]`.
 * @param options - beta, the commitment weight (default 0.25), and `ema` to drop the codebook term because the codes
 *   are updated by {@link vqEmaUpdate} instead.
 * @returns The quantised latents, indices, the loss terms, the code usage and its perplexity.
 * @concept vector-quantisation
 * @example
 * ```ts
 * import { vectorQuantise } from 'mdml-engine/losses'
 * import { toArray } from 'mdml-engine/tensor'
 * const out = vectorQuantise([[0.9, 0], [0, 1.2]], [[1, 0], [0, 1]])
 * toArray(out.indices) // => [0, 1]
 * ```
 */
export function vectorQuantise(
  latents: TensorLike,
  codebook: TensorLike,
  options: { readonly beta?: number; readonly ema?: boolean } = {},
): VectorQuantisation {
  const [z, book] = promoteFloat(latents, codebook) as [Value, Value]
  const k = book.shape[0] ?? 1
  const squared = sub(
    add(sum(square(z), -1, true), transpose(sum(square(book), -1, true))),
    mul(matmul(z, transpose(book)), 2),
  )
  const indices = argmin(stopGradient(squared), -1)
  const e = take(book, indices)
  const codebookLoss =
    options.ema === true
      ? mul(mean(square(sub(stopGradient(z), stopGradient(e)))), 0)
      : mean(square(sub(stopGradient(z), e)))
  const commitmentLoss = mean(square(sub(z, stopGradient(e))))
  const usage = mean(oneHot(indices, k), 0)
  const perplexity = exp(neg(sum(mul(usage, log(add(usage, 1e-10))))))
  return {
    quantised: add(z, stopGradient(sub(e, z))),
    indices,
    codebookLoss,
    commitmentLoss,
    loss: add(codebookLoss, mul(commitmentLoss, options.beta ?? 0.25)),
    usage,
    perplexity,
  }
}

/** The running statistics of an exponential-moving-average codebook. */
export interface VqEmaState {
  /** The smoothed number of latents assigned to each code `[K]`. */
  readonly counts: Value
  /** The smoothed sum of the latents assigned to each code `[K, D]`. */
  readonly sums: Value
  /** The codes `[K, D]`, sums over (smoothed) counts. */
  readonly codebook: Value
}

/**
 * The exponential-moving-average codebook update of the VQ-VAE (appendix A of van den Oord et al., 2017; Sonnet's
 * VectorQuantizerEMA): the smoothed assignment counts and latent sums per code decay at `decay` and absorb the batch,
 * and the code is their ratio with Laplace smoothing of the counts (a code no latent picks decays toward zero, the dead-code
 * problem that codebook restarts address).
 *
 * @param state - The previous counts, sums and codebook (start from zeros counts and the initial codebook as sums).
 * @param latents - Encoder outputs `[N, D]`.
 * @param indices - The nearest-code indices of the batch `[N]` (from {@link vectorQuantise}).
 * @param options - decay (default 0.99) and eps of the Laplace smoothing (default 1e-5).
 * @returns The next state.
 * @example
 * ```ts
 * import { vqEmaUpdate } from 'mdml-engine/losses'
 * import { toArray } from 'mdml-engine/tensor'
 * const start = { counts: [1, 1], sums: [[1, 0], [0, 1]], codebook: [[1, 0], [0, 1]] }
 * const next = vqEmaUpdate(start, [[3, 0]], [0], { decay: 0.5, eps: 0 })
 * toArray(next.codebook)[0] // => [2, 0]
 * ```
 */
export function vqEmaUpdate(
  state: { readonly counts: TensorLike; readonly sums: TensorLike; readonly codebook: TensorLike },
  latents: TensorLike,
  indices: TensorLike,
  options: { readonly decay?: number; readonly eps?: number } = {},
): VqEmaState {
  const decay = options.decay ?? 0.99
  const eps = options.eps ?? 1e-5
  const [counts, sums, z] = promoteFloat(state.counts, state.sums, latents) as [Value, Value, Value]
  const k = counts.shape[0] ?? 1
  const hot = oneHot(cast(promoteFloat(indices)[0] as Value, 'int32'), k)
  const batchCounts = sum(hot, 0)
  const batchSums = matmul(transpose(hot), z)
  const nextCounts = add(mul(counts, decay), mul(batchCounts, 1 - decay))
  const nextSums = add(mul(sums, decay), mul(batchSums, 1 - decay))
  const total = sum(nextCounts)
  const smoothed = mul(div(add(nextCounts, eps), add(total, k * eps)), total)
  return { counts: nextCounts, sums: nextSums, codebook: div(nextSums, reshape(smoothed, [k, 1])) }
}
