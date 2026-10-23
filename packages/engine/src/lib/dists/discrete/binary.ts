import { betainc, lbinom, logSigmoid, xlog1py, xlogy } from '../../numerics/special/index.ts'
import { type Key, uniform as uniformNoise } from '../../random/index.ts'
import {
  add,
  cast,
  equal,
  floor,
  greater,
  greaterEqual,
  less,
  log,
  log1p,
  logicalAnd,
  mul,
  neg,
  promoteFloat,
  sigmoid,
  softplus,
  sub,
  type Tensor,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { discreteSample, elementwiseSample } from '../_sampling.ts'
import { batchShapeOf, type Distribution, defineDistribution } from '../distribution.ts'
import { entropyBySum, isCount, largest } from './_support.ts'

/** Parameters of a Bernoulli distribution: probabilities or logits. */
export type BernoulliParams = { readonly probs: TensorLike } | { readonly logits: TensorLike }

/**
 * The Bernoulli distribution on {0, 1} (scipy.stats.bernoulli, torch.distributions.Bernoulli).
 *
 * @param params - `{ probs }` or `{ logits }` (logits keep log-probabilities accurate near 0 and 1).
 * @returns The distribution (no reparameterisation: `rsample` throws; use `relaxedBernoulli`).
 * @concept distributions
 * @example
 * ```ts
 * import { bernoulli } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(bernoulli({ probs: 0.25 }).mean()) // => 0.25
 * ```
 */
export function bernoulli(params: BernoulliParams): Distribution {
  const logits =
    'logits' in params
      ? (promoteFloat(params.logits)[0] as Value)
      : ((): Value => {
          const [p] = promoteFloat(params.probs) as [Value]
          return sub(log(p), log1p(neg(p)))
        })()
  const p = sigmoid(logits)
  const batchShape = batchShapeOf(logits)
  return defineDistribution({
    name: 'bernoulli',
    params: { logits, probs: p },
    support: { kind: 'boolean' },
    batchShape,
    logProb: (x) => {
      const [v] = promoteFloat(x) as [Value]
      return where(
        equal(v, 1),
        logSigmoid(logits),
        where(equal(v, 0), logSigmoid(neg(logits)), Number.NEGATIVE_INFINITY),
      )
    },
    cdf: (x) => {
      const [v] = promoteFloat(x) as [Value]
      return where(greater(0, v), 0, where(greater(1, v), sub(1, p), 1))
    },
    sample: (key: Key, shape) =>
      discreteSample([p], [...shape, ...batchShape], 'float64', ([probs]) => {
        const u = uniformNoise(key, [...shape, ...batchShape])
        return cast(greater(probs as Tensor, u), 'float64') as Tensor
      }),
    mean: () => p,
    variance: () => mul(p, sub(1, p)),
    mode: () => cast(greater(logits, 0), 'float64'),
    entropy: () => sub(softplus(logits), mul(p, logits)),
  })
}

/**
 * The binomial distribution: successes in n trials with success probability p (scipy.stats.binom).
 *
 * @param n - Number of trials (a non-negative integer).
 * @param probs - Success probability.
 * @returns The distribution (no reparameterisation).
 * @example
 * ```ts
 * import { binomial } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(binomial(10, 0.3).mean()) // => 3
 * ```
 */
export function binomial(n: TensorLike, probs: TensorLike): Distribution {
  const [trials, p] = promoteFloat(n, probs) as [Value, Value]
  const batchShape = batchShapeOf(trials, p)
  const inside = (x: Value): Value => logicalAnd(isCount(x), less(x, add(trials, 0.5)))
  const logPmf = (x: Value): Value => {
    const safe = where(inside(x), x, 0)
    return where(
      inside(x),
      add(add(lbinom(trials, safe), xlogy(safe, p)), xlog1py(sub(trials, safe), neg(p))),
      Number.NEGATIVE_INFINITY,
    )
  }
  return defineDistribution({
    name: 'binomial',
    params: { n: trials, probs: p },
    support: { kind: 'integerInterval', low: 0, high: Number.POSITIVE_INFINITY },
    batchShape,
    logProb: logPmf,
    cdf: (x) => {
      // P(X <= k) = I_{1-p}(n - k, k + 1) for 0 <= k < n.
      const k = floor(x)
      const mid = logicalAnd(greaterEqual(k, 0), less(k, trials))
      const safe = where(mid, k, 0)
      return where(less(k, 0), 0, where(mid, betainc(sub(trials, safe), add(safe, 1), sub(1, p)), 1))
    },
    sample: (key: Key, shape) =>
      elementwiseSample(key, [trials, p], shape, batchShape, (k, nn, pp) => {
        let c = 0
        const u = uniformNoise(k, [nn ?? 0]).data
        for (const v of u) if (v < (pp ?? 0)) c++
        return c
      }),
    mean: () => mul(trials, p),
    variance: () => mul(mul(trials, p), sub(1, p)),
    mode: () => floor(mul(add(trials, 1), p)),
    entropy: () => entropyBySum(logPmf, largest(trials, 'binomial'), batchShape.length),
  })
}
