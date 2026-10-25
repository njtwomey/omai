/**
 * Distributions over a finite set of categories: the categorical and one-hot categorical (from probabilities or
 * logits), the discrete uniform on a range of integers, and ordinal (cumulative-link) distributions such as the ordered
 * logistic and ordered probit, where increasing cutpoints divide a latent scale into ordered classes.
 */

import { MdmlError, maxOf, minOf } from '../../base/index.ts'
import { log1mexp, logNdtr, logSigmoid } from '../../numerics/special/index.ts'
import { type Key, uniform as uniformNoise } from '../../random/index.ts'
import {
  add,
  all,
  arange,
  argmax,
  broadcastShapes,
  broadcastTo,
  cast,
  ceil,
  clip,
  concat,
  cumsum,
  div,
  equal,
  exp,
  expandDims,
  floor,
  full,
  greaterEqual,
  less,
  lessEqual,
  log,
  logicalAnd,
  logicalOr,
  logSoftmax,
  mul,
  neg,
  oneHot,
  primalOf,
  promoteFloat,
  square,
  squeeze,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  takeAlongAxis,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { discreteSample } from '../_sampling.ts'
import { batchShapeOf, broadcastBatch, type Distribution, defineDistribution, validateParams } from '../distribution.ts'

/** Parameters of a categorical distribution: probabilities (normalised along the last axis) or logits. */
export type CategoricalParams = { readonly probs: TensorLike } | { readonly logits: TensorLike }

/** Normalised log-probabilities along the last axis from `{ probs }` or `{ logits }`. */
function logProbsOf(params: CategoricalParams, family: string): Value {
  let raw: Value
  if ('logits' in params) raw = promoteFloat(params.logits)[0] as Value
  else {
    const p = promoteFloat(params.probs)[0] as Value
    validateParams(family, { probs: [p, 'nonNegative'] })
    raw = log(p)
  }
  if (raw.shape.length === 0) throw new MdmlError('shape', `${family}: parameters need a category axis`)
  return logSoftmax(raw, -1)
}

/** Draw category indices `[...shape, ...batch]` from cumulative probabilities `[...batch, K]`. */
function drawIndices(key: Key, cumulative: Value, shape: readonly number[], batchShape: readonly number[]): Value {
  return discreteSample([cumulative], [...shape, ...batchShape], 'float64', ([cdf]) => {
    const u = expandDims(uniformNoise(key, [...shape, ...batchShape]), -1)
    return sum(cast(less(cdf as Tensor, u), 'float64'), -1) as Tensor
  })
}

/**
 * The categorical distribution over {0, ..., K-1}, with categories along the last axis (torch.distributions.Categorical).
 *
 * @param params - `{ probs }` (normalised for you) or `{ logits }`.
 * @returns The distribution (no reparameterisation: `rsample` throws; use `relaxedOneHotCategorical`).
 * @concept distributions
 * @example
 * ```ts
 * import { categorical } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(categorical({ probs: [0.2, 0.8] }).logProb(1)) - Math.log(0.8)) < 1e-15 // => true
 * ```
 */
export function categorical(params: CategoricalParams): Distribution {
  const logp = logProbsOf(params, 'categorical')
  return categoricalFromLogProbs(logp, 'categorical', {})
}

/** A categorical distribution from normalised log-probabilities (shared by the ordinal families). */
function categoricalFromLogProbs(logp: Value, name: string, extra: Readonly<Record<string, Value>>): Distribution {
  const probs = exp(logp)
  const k = logp.shape[logp.shape.length - 1] ?? 0
  const batchShape = logp.shape.slice(0, -1)
  const index = arange(k)
  // Broadcast the per-category values to the shape of x and gather at x.
  const gather = (values: Value, x: Value): Value => {
    const shape = broadcastShapes(x.shape, batchShape)
    const table = broadcastTo(values, [...shape, k])
    const at = expandDims(broadcastTo(clip(x, 0, k - 1), shape), -1)
    return squeeze(takeAlongAxis(table, cast(at, 'int32'), -1), -1)
  }
  const cumulative = cumsum(probs, -1)
  const tail = cumsum(probs, -1, true)
  return defineDistribution({
    name,
    params: { ...extra, logits: logp, probs },
    support: { kind: 'integerInterval', low: 0, high: k - 1 },
    batchShape,
    logProb: (x) => {
      const v = promoteFloat(x)[0] as Value
      const invalid = logicalOr(logicalOr(less(v, 0), greaterEqual(v, k)), less(floor(v), v))
      return where(invalid, Number.NEGATIVE_INFINITY, gather(logp, v))
    },
    cdf: (x) => {
      const v = floor(promoteFloat(x)[0] as Value)
      return where(less(v, 0), 0, where(greaterEqual(v, k - 1), 1, gather(cumulative, v)))
    },
    sf: (x) => {
      // P(X > v) = sum_{j > v} p_j, the reversed cumulative sum at v + 1.
      const v = floor(promoteFloat(x)[0] as Value)
      return where(less(v, 0), 1, where(greaterEqual(v, k - 1), 0, gather(tail, add(v, 1))))
    },
    ppf: (q) => {
      // The smallest k with cdf(k) >= q: the number of cumulative probabilities below q (-1 at q = 0).
      const qv = promoteFloat(q)[0] as Value
      const count = sum(cast(less(cumulative, expandDims(qv, -1)), 'float64'), -1)
      return where(lessEqual(qv, 0), -1, clip(count, 0, k - 1))
    },
    sample: (key: Key, shape) => drawIndices(key, cumulative, shape, batchShape),
    mean: () => sum(mul(probs, index), -1),
    variance: () => {
      const m = sum(mul(probs, index), -1)
      return sub(sum(mul(probs, square(index)), -1), square(m))
    },
    mode: () => cast(argmax(logp, -1), 'float64'),
    entropy: () => neg(sum(mul(probs, logp), -1)),
  })
}

/**
 * The one-hot categorical distribution: categorical outcomes as one-hot vectors (torch.distributions.OneHotCategorical),
 * with event shape `[K]`.
 *
 * @param params - `{ probs }` or `{ logits }`, categories along the last axis.
 * @returns The distribution; the mean is the probability vector and the variance p (1 - p). No reparameterisation
 *   (use `relaxedOneHotCategorical` or `straightThrough`).
 * @example
 * ```ts
 * import { oneHotCategorical } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(oneHotCategorical({ probs: [0.2, 0.8] }).logProb([0, 1])) - Math.log(0.8)) < 1e-15 // => true
 * ```
 */
export function oneHotCategorical(params: CategoricalParams): Distribution {
  const logp = logProbsOf(params, 'oneHotCategorical')
  const probs = exp(logp)
  const k = logp.shape[logp.shape.length - 1] ?? 0
  const batchShape = logp.shape.slice(0, -1)
  const cumulative = cumsum(probs, -1)
  return defineDistribution({
    name: 'oneHotCategorical',
    params: { logits: logp, probs },
    support: { kind: 'oneHot' },
    batchShape,
    eventShape: [k],
    logProb: (x) => {
      const v = promoteFloat(x)[0] as Value
      const binary = all(logicalOr(equal(v, 0), equal(v, 1)), -1)
      const single = equal(sum(v, -1), 1)
      // log p at the hot entry (sum(x log p) with 0 log 0 taken as 0).
      const lp = sum(where(equal(v, 1), logp, 0), -1)
      return where(logicalAnd(binary, single), lp, Number.NEGATIVE_INFINITY)
    },
    sample: (key: Key, shape) => oneHot(cast(drawIndices(key, cumulative, shape, batchShape), 'int32'), k),
    mean: () => probs,
    variance: () => mul(probs, sub(1, probs)),
    mode: () => oneHot(cast(argmax(logp, -1), 'int32'), k),
    entropy: () => neg(sum(mul(probs, logp), -1)),
  })
}

/**
 * The discrete uniform distribution on the integers low, low + 1, ..., high, both ends included (NumPyro's
 * DiscreteUniform; scipy.stats.randint(low, high + 1)).
 *
 * @param low - Lowest value (an integer).
 * @param high - Highest value (an integer, at least low).
 * @returns The distribution (no reparameterisation).
 * @example
 * ```ts
 * import { discreteUniform } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(discreteUniform(1, 6).mean()) // => 3.5
 * item(discreteUniform(1, 6).cdf(2)) // => 0.3333333333333333
 * ```
 */
export function discreteUniform(low: TensorLike, high: TensorLike): Distribution {
  const [lo, hi] = promoteFloat(low, high) as [Value, Value]
  validateParams('discreteUniform', { low: [lo, 'integer'], high: [hi, 'integer'] })
  const batchShape = batchShapeOf(lo, hi)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const n = add(sub(hi, lo), 1)
  const inside = (x: Value): Value => logicalAnd(logicalAnd(greaterEqual(x, lo), lessEqual(x, hi)), equal(floor(x), x))
  return defineDistribution({
    name: 'discreteUniform',
    params: { low: lo, high: hi },
    support: { kind: 'integerInterval', low: minOf(toFlat(primalOf(lo))), high: maxOf(toFlat(primalOf(hi))) },
    batchShape,
    logProb: (x) => where(inside(x), neg(log(b(n))), Number.NEGATIVE_INFINITY),
    cdf: (x) => clip(div(add(sub(floor(x), lo), 1), n), 0, 1),
    sf: (x) => clip(div(sub(hi, floor(x)), n), 0, 1),
    ppf: (q) => where(lessEqual(q, 0), sub(lo, 1), add(sub(lo, 1), ceil(mul(q, n)))),
    sample: (key: Key, shape) =>
      discreteSample([lo, n], [...shape, ...batchShape], 'float64', ([l, m]) => {
        const u = uniformNoise(key, [...shape, ...batchShape])
        return add(l as Tensor, floor(mul(u, m as Tensor))) as Tensor
      }),
    mean: () => b(mul(add(lo, hi), 0.5)),
    variance: () => b(div(sub(square(n), 1), 12)),
    mode: () => b(lo),
    entropy: () => b(log(n)),
  })
}

/**
 * An ordinal (cumulative-link) distribution over {0, ..., K-1}: P(Y <= k) = F(cutpoints_k - predictor) for an
 * increasing vector of K - 1 cutpoints and a link CDF F, given in log form for accuracy in the tails.
 *
 * @param predictor - The linear predictor (batch shape).
 * @param cutpoints - Increasing cutpoints `[..., K - 1]`.
 * @param logCdf - log F.
 * @param name - The distribution's name.
 * @returns A categorical distribution with `predictor` and `cutpoints` among its parameters.
 * @example
 * ```ts
 * import { orderedCategorical } from 'mdml-engine/dists'
 * import { logSigmoid } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(orderedCategorical(0, [0], logSigmoid, 'ordinal').prob(0)) // => 0.5
 * ```
 */
export function orderedCategorical(
  predictor: TensorLike,
  cutpoints: TensorLike,
  logCdf: (x: Value) => Value,
  name: string,
): Distribution {
  const [eta, c] = promoteFloat(predictor, cutpoints) as [Value, Value]
  if (c.shape.length === 0) throw new MdmlError('shape', `${name}: cutpoints need a category axis`)
  const z = sub(c, expandDims(eta, -1))
  const lead = z.shape.slice(0, -1)
  // log P(Y <= j) for j = 0 .. K-2, with log P(Y <= K-1) = 0, and log P(Y = j) as log(F_j - F_{j-1}).
  const upper = concat([logCdf(z), full([...lead, 1], 0)], -1)
  const lower = concat([full([...lead, 1], Number.NEGATIVE_INFINITY), logCdf(z)], -1)
  const safeDiff = where(less(lower, upper), sub(lower, upper), -1)
  const logp = where(less(lower, upper), add(upper, log1mexp(safeDiff)), Number.NEGATIVE_INFINITY)
  return categoricalFromLogProbs(logp, name, { predictor: eta, cutpoints: c })
}

/**
 * The ordered logistic distribution (proportional odds; NumPyro's OrderedLogistic): P(Y <= k) =
 * sigmoid(cutpoints_k - predictor).
 *
 * @param predictor - The linear predictor.
 * @param cutpoints - Increasing cutpoints `[..., K - 1]`.
 * @returns The distribution over {0, ..., K-1}.
 * @example
 * ```ts
 * import { orderedLogistic } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(orderedLogistic(0, [-1, 1]).prob(1)) - (1 / (1 + Math.exp(-1)) - 1 / (1 + Math.exp(1)))) < 1e-15 // => true
 * ```
 */
export function orderedLogistic(predictor: TensorLike, cutpoints: TensorLike): Distribution {
  return orderedCategorical(predictor, cutpoints, logSigmoid, 'orderedLogistic')
}

/**
 * The ordered probit distribution: P(Y <= k) = Phi(cutpoints_k - predictor).
 *
 * @param predictor - The linear predictor.
 * @param cutpoints - Increasing cutpoints `[..., K - 1]`.
 * @returns The distribution over {0, ..., K-1}.
 * @example
 * ```ts
 * import { orderedProbit } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(orderedProbit(0, [0]).prob(1)) // => 0.5
 * ```
 */
export function orderedProbit(predictor: TensorLike, cutpoints: TensorLike): Distribution {
  return orderedCategorical(predictor, cutpoints, logNdtr, 'orderedProbit')
}
