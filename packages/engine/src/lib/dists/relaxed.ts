/**
 * Continuous relaxations of discrete distributions, so that sampling categorical outcomes can be differentiated: the
 * relaxed Bernoulli (binary Concrete) and the relaxed one-hot categorical (Gumbel-softmax), whose temperature controls
 * how close draws are to hard choices, and straight-through estimators that return hard samples with the gradients of
 * the relaxed ones.
 */

import { WhiteboxError } from '../base/index.ts'
import { lgamma, logit, logSigmoid } from '../numerics/special/index.ts'
import {
  add,
  argmax,
  broadcastShapes,
  cast,
  div,
  expandDims,
  greater,
  isfinite,
  lessEqual,
  log,
  logicalAnd,
  logSoftmax,
  logsumexp,
  mul,
  neg,
  oneHot,
  promoteFloat,
  sigmoid,
  softmax,
  softplus,
  stopGradient,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  type Value,
  where,
} from '../tensor/index.ts'
import { concreteTwin, numericMode, quantileExpectation } from './_numeric.ts'
import { uniformOpen } from './_sampling.ts'
import { batchShapeOf, type Distribution, defineDistribution, unavailable, validateParams } from './distribution.ts'

/** Probabilities or logits of a binary or categorical outcome. */
export type RelaxedParams = { readonly probs: TensorLike } | { readonly logits: TensorLike }

function logitsFrom(params: RelaxedParams, family: string, categorical: boolean): Value {
  if ('logits' in params) return promoteFloat(params.logits)[0] as Value
  const [p] = promoteFloat(params.probs) as [Value]
  validateParams(family, { probs: [p, categorical ? 'nonNegative' : 'unitInterval'] })
  return categorical ? log(p) : sub(log(p), log(sub(1, p)))
}

/**
 * The relaxed Bernoulli (binary Concrete) distribution on (0, 1) (Maddison et al., 2017; Jang et al., 2017;
 * torch.distributions.RelaxedBernoulli): X = sigmoid((logits + logit(U)) / temperature), whose logit is logistic with
 * location logits / temperature and scale 1 / temperature. As the temperature falls it concentrates on {0, 1} and
 * approaches the Bernoulli.
 *
 * @param temperature - Temperature (positive).
 * @param params - `{ probs }` or `{ logits }`.
 * @returns The distribution with closed-form density, CDF and quantiles and a reparameterised `rsample`; the moments
 *   and entropy by quadrature over the quantile function, the mode by search (at an end for temperatures below 1).
 * @example
 * ```ts
 * import { relaxedBernoulli } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(relaxedBernoulli(0.5, { logits: 0 }).cdf(0.5)) // => 0.5
 * ```
 */
export function relaxedBernoulli(temperature: TensorLike, params: RelaxedParams): Distribution {
  const [t] = promoteFloat(temperature) as [Value]
  validateParams('relaxedBernoulli', { temperature: [t, 'positive'] })
  const l = logitsFrom(params, 'relaxedBernoulli', false)
  const batchShape = batchShapeOf(t, l)
  const inside = (x: Value): Value => logicalAnd(greater(x, 0), greater(sub(1, x), 0))
  const safe = (x: Value): Value => where(inside(x), x, 0.5)
  // The logistic CDF of the logit: sigmoid(t logit(x) - l).
  const cdf = (x: Value): Value => sigmoid(sub(mul(t, logit(safe(x))), l))
  const ppf = (q: Value): Value => sigmoid(div(add(l, logit(q)), t))
  const twin = concreteTwin([t, l], (a, b) => relaxedBernoulli(a as Tensor, { logits: b as Tensor }))
  const self: Distribution = defineDistribution({
    name: 'relaxedBernoulli',
    params: { temperature: t, logits: l },
    support: { kind: 'unitInterval' },
    batchShape,
    logProb: (x) => {
      const v = safe(x)
      const diff = sub(l, mul(t, logit(v)))
      const lp = sub(sub(add(log(t), diff), mul(softplus(diff), 2)), add(log(v), log(sub(1, v))))
      return where(inside(x), lp, Number.NEGATIVE_INFINITY)
    },
    cdf: (x) => where(lessEqual(x, 0), 0, where(lessEqual(sub(1, x), 0), 1, cdf(x))),
    ppf,
    rsample: (key, shape) => ppf(uniformOpen(key, [...shape, ...batchShape])),
    mean: () => quantileExpectation(self, (x) => x),
    variance: () => {
      const m = quantileExpectation(self, (x) => x)
      return quantileExpectation(self, (x) => mul(sub(x, m), sub(x, m)))
    },
    mode: () => numericMode(self, twin, 1e-9, 1 - 1e-9),
    // H(X) = H(Y) + E[log sigmoid'(Y)] for the logistic logit Y (entropy 2 - log t).
    entropy: () =>
      add(
        sub(2, log(t)),
        quantileExpectation(self, (x) => {
          const y = logit(x)
          const term = add(logSigmoid(y), logSigmoid(neg(y)))
          return where(isfinite(term), term, 0)
        }),
      ),
  })
  return self
}

/**
 * The relaxed one-hot categorical (Concrete, Gumbel-softmax) distribution on the simplex (Maddison et al., 2017;
 * Jang et al., 2017; torch.distributions.RelaxedOneHotCategorical): X = softmax((logits + G) / temperature) for
 * independent Gumbel noise G. Its density is Gamma(K) t^(K-1) prod_i (pi_i x_i^(-t-1)) / (sum_i pi_i x_i^(-t))^K.
 *
 * @param temperature - Temperature (positive).
 * @param params - `{ probs }` or `{ logits }`, categories along the last axis.
 * @returns The distribution with event shape `[K]` and a reparameterised `rsample`; its moments, mode and entropy have
 *   no closed form and throw (estimate them from samples).
 * @example
 * ```ts
 * import { relaxedOneHotCategorical } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * // Two categories: the density at (1/2, 1/2) with equal logits is t (temperature 1 gives 1).
 * Math.abs(item(relaxedOneHotCategorical(1, { logits: [0, 0] }).logProb([0.5, 0.5]))) < 1e-15 // => true
 * ```
 */
export function relaxedOneHotCategorical(temperature: TensorLike, params: RelaxedParams): Distribution {
  const [t] = promoteFloat(temperature) as [Value]
  validateParams('relaxedOneHotCategorical', { temperature: [t, 'positive'] })
  const raw = logitsFrom(params, 'relaxedOneHotCategorical', true)
  if (raw.shape.length === 0)
    throw new WhiteboxError('shape', 'relaxedOneHotCategorical: parameters need a category axis')
  const l = logSoftmax(raw, -1)
  const k = l.shape[l.shape.length - 1] ?? 0
  const batchShape = broadcastShapes(l.shape.slice(0, -1), t.shape)
  // The temperature lifted over the category axis.
  const tk = expandDims(t, -1)
  const logNormaliser = add(lgamma(k), mul(log(t), k - 1))
  return defineDistribution({
    name: 'relaxedOneHotCategorical',
    params: { temperature: t, logits: l },
    support: { kind: 'simplex' },
    batchShape,
    eventShape: [k],
    logProb: (x) => {
      const logX = log(x)
      const score = sub(l, mul(tk, logX))
      return add(sub(sub(sum(score, -1), mul(logsumexp(score, -1), k)), sum(logX, -1)), logNormaliser)
    },
    rsample: (key, shape) => {
      const u = uniformOpen(key, [...shape, ...batchShape, k])
      const gumbel = neg(log(neg(log(u))))
      return softmax(div(add(l, gumbel), tk), -1)
    },
    mean: () => unavailable('relaxedOneHotCategorical', 'mean (no closed form)'),
    variance: () => unavailable('relaxedOneHotCategorical', 'variance (no closed form)'),
    mode: () => unavailable('relaxedOneHotCategorical', 'mode (no closed form)'),
    entropy: () => unavailable('relaxedOneHotCategorical', 'entropy (no closed form)'),
  })
}

/**
 * A straight-through estimator (Bengio et al., 2013): `rsample` returns hard samples (0/1 for a binary distribution,
 * one-hot vectors for a categorical one) whose gradients are those of a differentiable surrogate. For a relaxed
 * distribution the surrogate is its relaxed draw (straight-through Gumbel-softmax); for a Bernoulli or one-hot
 * categorical it is the probability vector (torch's OneHotCategoricalStraightThrough). Densities and other methods
 * are the given distribution's.
 *
 * @param dist - A `relaxedBernoulli`, `relaxedOneHotCategorical`, `bernoulli` or `oneHotCategorical` distribution.
 * @returns The distribution with the straight-through `rsample`.
 * @throws WhiteboxError with code `value` for other distributions.
 * @example
 * ```ts
 * import { relaxedBernoulli, straightThrough } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(straightThrough(relaxedBernoulli(0.5, { probs: 0.3 })).rsample(key(0), [4])).every((v) => v === 0 || v === 1) // => true
 * ```
 */
export function straightThrough(dist: Distribution): Distribution {
  const hardBinary = (soft: Value): Value => cast(greater(soft, 0.5), 'float64')
  const hardOneHot = (soft: Value): Value =>
    oneHot(cast(argmax(soft, -1), 'int32'), soft.shape[soft.shape.length - 1] ?? 0)
  // hard + soft - stopGradient(soft): the value of hard, the gradient of soft.
  const through = (hard: Value, soft: Value): Value => add(stopGradient(sub(hard, soft)), soft)
  let rsample: (key: Parameters<Distribution['rsample']>[0], shape?: readonly number[]) => Value
  switch (dist.name) {
    case 'relaxedBernoulli':
      rsample = (key, shape = []) => {
        const soft = dist.rsample(key, shape)
        return through(hardBinary(soft), soft)
      }
      break
    case 'relaxedOneHotCategorical':
      rsample = (key, shape = []) => {
        const soft = dist.rsample(key, shape)
        return through(hardOneHot(soft), soft)
      }
      break
    case 'bernoulli':
    case 'oneHotCategorical':
      rsample = (key, shape = []) => {
        const hard = dist.sample(key, shape)
        const p = dist.params['probs'] as Value
        return add(hard, sub(p, stopGradient(p)))
      }
      break
    default:
      throw new WhiteboxError('value', `straightThrough: no straight-through estimator for ${dist.name}`)
  }
  return { ...dist, name: `straightThrough(${dist.name})`, hasRsample: true, rsample }
}
