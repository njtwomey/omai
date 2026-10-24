import { betainc, lbeta, lbinom, logSigmoid, xlog1py, xlogy } from '../../numerics/special/index.ts'
import {
  beta as betaDraw,
  binomial as binomialDraw,
  foldIn,
  type Key,
  uniform as uniformNoise,
} from '../../random/index.ts'
import {
  add,
  cast,
  div,
  equal,
  floor,
  greater,
  greaterEqual,
  less,
  logicalAnd,
  mul,
  neg,
  promoteFloat,
  sigmoid,
  softplus,
  square,
  sub,
  type Tensor,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { concreteTwin, discretePpf } from '../_numeric.ts'
import { discreteSample, elementwiseSample } from '../_sampling.ts'
import { batchShapeOf, broadcastBatch, type Distribution, defineDistribution, validateParams } from '../distribution.ts'
import { binaryLogits, cdfBySum, entropyBySum, isCount, largest, modeBySearch } from './_support.ts'

/** Parameters of a Bernoulli distribution: probabilities or logits. */
export type BernoulliParams = { readonly probs: TensorLike } | { readonly logits: TensorLike }

/**
 * The Bernoulli distribution on {0, 1} (scipy.stats.bernoulli, torch.distributions.Bernoulli).
 *
 * @param params - `{ probs }` or `{ logits }` (logits keep log-probabilities accurate near 0 and 1).
 * @returns The distribution (no reparameterisation: `rsample` throws; use `relaxedBernoulli` or `straightThrough`).
 * @concept distributions
 * @example
 * ```ts
 * import { bernoulli } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(bernoulli({ probs: 0.25 }).mean()) // => 0.25
 * ```
 */
export function bernoulli(params: BernoulliParams): Distribution {
  const logits = binaryLogits(params, 'bernoulli')
  const p = sigmoid(logits)
  const batchShape = batchShapeOf(logits)
  const value = (x: Value): Value => promoteFloat(x)[0] as Value
  return defineDistribution({
    name: 'bernoulli',
    params: { logits, probs: p },
    support: { kind: 'boolean' },
    batchShape,
    logProb: (x) => {
      const v = value(x)
      return where(
        equal(v, 1),
        logSigmoid(logits),
        where(equal(v, 0), logSigmoid(neg(logits)), Number.NEGATIVE_INFINITY),
      )
    },
    cdf: (x) => {
      const v = value(x)
      return where(greater(0, v), 0, where(greater(1, v), sigmoid(neg(logits)), 1))
    },
    sf: (x) => {
      const v = value(x)
      return where(greater(0, v), 1, where(greater(1, v), p, 0))
    },
    // The smallest k with F(k) >= q: -1 at q = 0, 0 up to 1 - p, then 1.
    ppf: (q) => {
      const v = value(q)
      return where(equal(v, 0), -1, where(greater(v, sigmoid(neg(logits))), 1, 0))
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
 * The binomial distribution: successes in n trials with success probability p (scipy.stats.binom,
 * torch.distributions.Binomial).
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
  validateParams('binomial', { n: [trials, 'nonNegativeInteger'], probs: [p, 'unitInterval'] })
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
  const twin = concreteTwin([trials, p], (a, b) => binomial(a as Tensor, b as Tensor))
  return defineDistribution({
    name: 'binomial',
    params: { n: trials, probs: p },
    support: { kind: 'integerInterval', low: 0, high: largest(trials) },
    batchShape,
    logProb: logPmf,
    cdf: (x) => {
      // P(X <= k) = I_{1-p}(n - k, k + 1) for 0 <= k < n.
      const k = floor(x)
      const mid = logicalAnd(greaterEqual(k, 0), less(k, trials))
      const safe = where(mid, k, 0)
      return where(less(k, 0), 0, where(mid, betainc(sub(trials, safe), add(safe, 1), sub(1, p)), 1))
    },
    sf: (x) => {
      // P(X > k) = I_p(k + 1, n - k).
      const k = floor(x)
      const mid = logicalAnd(greaterEqual(k, 0), less(k, trials))
      const safe = where(mid, k, 0)
      return where(less(k, 0), 1, where(mid, betainc(add(safe, 1), sub(trials, safe), p), 0))
    },
    ppf: (q) => discretePpf(twin(), q, 0, broadcastBatch(trials, batchShape)),
    sample: (key: Key, shape) =>
      elementwiseSample(key, [trials, p], shape, batchShape, (k, nn, pp) => {
        return binomialDraw(k, nn ?? 0, pp ?? 0, []).data[0] ?? 0
      }),
    mean: () => mul(trials, p),
    variance: () => mul(mul(trials, p), sub(1, p)),
    mode: () => floor(mul(add(trials, 1), p)),
    entropy: () => entropyBySum(logPmf, largest(trials), batchShape.length),
  })
}

/**
 * The beta-binomial distribution: binomial counts whose success probability is Beta(a, b) (scipy.stats.betabinom),
 * an overdispersed binomial.
 *
 * @param n - Number of trials (a non-negative integer).
 * @param a - First beta shape (positive).
 * @param b - Second beta shape (positive).
 * @returns The distribution (no reparameterisation); the CDF, mode and entropy sum the pmf over 0 .. n.
 * @example
 * ```ts
 * import { betaBinomial } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(betaBinomial(10, 2, 3).mean()) // => 4
 * ```
 */
export function betaBinomial(n: TensorLike, a: TensorLike, b: TensorLike): Distribution {
  const [trials, alpha, bet] = promoteFloat(n, a, b) as [Value, Value, Value]
  validateParams('betaBinomial', { n: [trials, 'nonNegativeInteger'], a: [alpha, 'positive'], b: [bet, 'positive'] })
  const batchShape = batchShapeOf(trials, alpha, bet)
  const total = add(alpha, bet)
  const inside = (x: Value): Value => logicalAnd(isCount(x), less(x, add(trials, 0.5)))
  const logPmf = (x: Value): Value => {
    const k = where(inside(x), x, 0)
    const lp = sub(add(lbinom(trials, k), lbeta(add(k, alpha), add(sub(trials, k), bet))), lbeta(alpha, bet))
    return where(inside(x), lp, Number.NEGATIVE_INFINITY)
  }
  const nmax = (): number => largest(trials)
  const twin = concreteTwin([trials, alpha, bet], (x, y, z) => betaBinomial(x as Tensor, y as Tensor, z as Tensor))
  return defineDistribution({
    name: 'betaBinomial',
    params: { n: trials, a: alpha, b: bet },
    support: { kind: 'integerInterval', low: 0, high: largest(trials) },
    batchShape,
    logProb: logPmf,
    cdf: (x) => where(greaterEqual(floor(x), trials), 1, cdfBySum(logPmf, x, nmax(), batchShape)),
    ppf: (q) => discretePpf(twin(), q, 0, broadcastBatch(trials, batchShape)),
    sample: (key: Key, shape) =>
      elementwiseSample(key, [trials, alpha, bet], shape, batchShape, (k, nn, aa, bb) => {
        const pr = betaDraw(foldIn(k, 0), [], aa ?? 1, bb ?? 1).data[0] ?? 0
        return binomialDraw(foldIn(k, 1), nn ?? 0, pr, []).data[0] ?? 0
      }),
    mean: () => broadcastBatch(div(mul(trials, alpha), total), batchShape),
    variance: () =>
      broadcastBatch(
        div(mul(mul(mul(trials, alpha), bet), add(total, trials)), mul(square(total), add(total, 1))),
        batchShape,
      ),
    mode: () => modeBySearch(logPmf, nmax(), batchShape.length),
    entropy: () => entropyBySum(logPmf, nmax(), batchShape.length),
  })
}
