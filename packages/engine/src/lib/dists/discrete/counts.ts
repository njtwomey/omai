/**
 * Count distributions on the non-negative integers: Poisson, geometric, negative binomial and hypergeometric (as
 * scipy.stats), and zero-inflated versions of a count distribution, which add structural zeros, with the zero-inflated
 * Poisson and negative binomial ready-made.
 */

import {
  betainc,
  expit,
  gammainc,
  gammaincc,
  lbinom,
  lgamma,
  logaddexp,
  logSigmoid,
  xlog1py,
  xlogy,
} from '../../numerics/special/index.ts'
import {
  foldIn,
  gammaVariates,
  geometric as geometricDraw,
  type Key,
  poisson as poissonDraw,
  uniform as uniformNoise,
} from '../../random/index.ts'
import {
  add,
  ceil,
  div,
  equal,
  exp,
  expm1,
  floor,
  greaterEqual,
  less,
  lessEqual,
  log,
  log1p,
  logicalAnd,
  maximum,
  minimum,
  mul,
  neg,
  promoteFloat,
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

/**
 * The Poisson distribution with rate lambda (scipy.stats.poisson, torch.distributions.Poisson).
 *
 * @param rate - The rate (non-negative).
 * @returns The distribution (no reparameterisation).
 * @concept distributions
 * @example
 * ```ts
 * import { poisson } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(poisson(2).prob(0)) - Math.exp(-2)) < 1e-15 // => true
 * ```
 */
export function poisson(rate: TensorLike): Distribution {
  const [lambda] = promoteFloat(rate) as [Value]
  validateParams('poisson', { rate: [lambda, 'nonNegative'] })
  const batchShape = batchShapeOf(lambda)
  const logPmf = (x: Value): Value =>
    where(isCount(x), sub(sub(xlogy(x, lambda), lambda), lgamma(add(x, 1))), Number.NEGATIVE_INFINITY)
  const twin = concreteTwin([lambda], (l) => poisson(l as Tensor))
  const k = (x: Value): Value => floor(where(less(x, 0), 0, x))
  return defineDistribution({
    name: 'poisson',
    params: { rate: lambda },
    support: { kind: 'integerInterval', low: 0, high: Number.POSITIVE_INFINITY },
    batchShape,
    logProb: logPmf,
    cdf: (x) => where(less(x, 0), 0, gammaincc(add(k(x), 1), lambda)),
    sf: (x) => where(less(x, 0), 1, gammainc(add(k(x), 1), lambda)),
    ppf: (q) => discretePpf(twin(), q, 0, Number.POSITIVE_INFINITY),
    sample: (key: Key, shape) =>
      elementwiseSample(key, [lambda], shape, batchShape, (kk, l) => poissonDraw(kk, l ?? 0, []).data[0] ?? 0),
    mean: () => lambda,
    variance: () => lambda,
    mode: () => floor(lambda),
    entropy: () => {
      const lmax = largest(lambda)
      return entropyBySum(logPmf, Math.ceil(lmax + 12 * Math.sqrt(lmax + 1) + 20), batchShape.length)
    },
  })
}

/**
 * The geometric distribution: the number of trials up to and including the first success, on {1, 2, ...}
 * (scipy.stats.geom).
 *
 * @param probs - Success probability (in (0, 1]).
 * @returns The distribution (no reparameterisation).
 * @example
 * ```ts
 * import { geometric } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(geometric(0.25).mean()) // => 4
 * ```
 */
export function geometric(probs: TensorLike): Distribution {
  const [p] = promoteFloat(probs) as [Value]
  validateParams('geometric', { probs: [p, 'unitInterval'] })
  const batchShape = batchShapeOf(p)
  const inside = (x: Value): Value => logicalAnd(isCount(x), greaterEqual(x, 1))
  // log(1 - p) and log P(X > k) = k log(1 - p).
  const logFail = log1p(neg(p))
  return defineDistribution({
    name: 'geometric',
    params: { probs: p },
    support: { kind: 'integerInterval', low: 1, high: Number.POSITIVE_INFINITY },
    batchShape,
    logProb: (x) => {
      const safe = where(inside(x), x, 1)
      return where(inside(x), add(xlog1py(sub(safe, 1), neg(p)), log(p)), Number.NEGATIVE_INFINITY)
    },
    cdf: (x) => {
      const k = floor(x)
      return where(less(k, 1), 0, neg(expm1(mul(k, logFail))))
    },
    sf: (x) => {
      const k = floor(x)
      return where(less(k, 1), 1, exp(mul(k, logFail)))
    },
    logSf: (x) => {
      const k = floor(x)
      return where(less(k, 1), 0, mul(k, logFail))
    },
    // The smallest k >= 1 with 1 - (1 - p)^k >= q.
    ppf: (q) => where(lessEqual(q, 0), 0, maximum(ceil(sub(div(log1p(neg(q)), logFail), 1e-12)), 1)),
    sample: (key: Key, shape) =>
      elementwiseSample(key, [p], shape, batchShape, (k, pp) => geometricDraw(k, pp ?? 0, []).data[0] ?? 0),
    mean: () => div(1, p),
    variance: () => div(sub(1, p), square(p)),
    mode: () => add(mul(p, 0), 1),
    entropy: () => div(neg(add(xlog1py(sub(1, p), neg(p)), xlogy(p, p))), p),
  })
}

/**
 * The negative binomial distribution: failures before the n-th success with success probability p
 * (scipy.stats.nbinom; n need not be an integer).
 *
 * @param n - Number of successes (positive).
 * @param probs - Success probability.
 * @returns The distribution (no reparameterisation).
 * @example
 * ```ts
 * import { negativeBinomial } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(negativeBinomial(3, 0.5).mean()) // => 3
 * ```
 */
export function negativeBinomial(n: TensorLike, probs: TensorLike): Distribution {
  const [r, p] = promoteFloat(n, probs) as [Value, Value]
  validateParams('negativeBinomial', { n: [r, 'positive'], probs: [p, 'unitInterval'] })
  const batchShape = batchShapeOf(r, p)
  const logPmf = (x: Value): Value => {
    const safe = where(isCount(x), x, 0)
    const lp = add(
      add(sub(sub(lgamma(add(safe, r)), lgamma(add(safe, 1))), lgamma(r)), mul(r, log(p))),
      xlog1py(safe, neg(p)),
    )
    return where(isCount(x), lp, Number.NEGATIVE_INFINITY)
  }
  const twin = concreteTwin([r, p], (a, b) => negativeBinomial(a as Tensor, b as Tensor))
  return defineDistribution({
    name: 'negativeBinomial',
    params: { n: r, probs: p },
    support: { kind: 'integerInterval', low: 0, high: Number.POSITIVE_INFINITY },
    batchShape,
    logProb: logPmf,
    cdf: (x) => {
      const k = floor(x)
      const ok = greaterEqual(k, 0)
      return where(ok, betainc(r, add(where(ok, k, 0), 1), p), 0)
    },
    sf: (x) => {
      // P(X > k) = I_{1-p}(k + 1, n).
      const k = floor(x)
      const ok = greaterEqual(k, 0)
      return where(ok, betainc(add(where(ok, k, 0), 1), r, sub(1, p)), 1)
    },
    ppf: (q) => discretePpf(twin(), q, 0, Number.POSITIVE_INFINITY),
    sample: (key: Key, shape) =>
      elementwiseSample(key, [r, p], shape, batchShape, (k, rr, pp) => {
        // A gamma-Poisson mixture: lambda ~ Gamma(r, rate p / (1 - p)), X ~ Poisson(lambda).
        const g = (gammaVariates(k, [rr ?? 1])[0] ?? 0) * ((1 - (pp ?? 0.5)) / (pp ?? 0.5))
        return poissonDraw(foldIn(k, 1), g, []).data[0] ?? 0
      }),
    mean: () => div(mul(r, sub(1, p)), p),
    variance: () => div(mul(r, sub(1, p)), square(p)),
    mode: () => where(less(1, r), floor(div(mul(sub(r, 1), sub(1, p)), p)), mul(r, 0)),
    entropy: () => {
      const m = largest(div(mul(r, sub(1, p)), p))
      const v = largest(div(mul(r, sub(1, p)), square(p)))
      // The tail is geometric-like, so sum far out: 40 standard deviations past the mean.
      return entropyBySum(logPmf, Math.ceil(m + 40 * Math.sqrt(v) + 100), batchShape.length)
    },
  })
}

/**
 * The hypergeometric distribution: successes in `draws` draws without replacement from `total` objects of which
 * `successes` are successes (scipy.stats.hypergeom(M=total, n=successes, N=draws)).
 *
 * @param total - Population size (a non-negative integer).
 * @param successes - Successes in the population (an integer, at most total).
 * @param draws - Number of draws (an integer, at most total).
 * @returns The distribution (no reparameterisation); the CDF and entropy sum the pmf over its finite support.
 * @example
 * ```ts
 * import { hypergeometric } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(hypergeometric(20, 7, 12).mean()) // => 4.2
 * ```
 */
export function hypergeometric(total: TensorLike, successes: TensorLike, draws: TensorLike): Distribution {
  const [m, n, nn] = promoteFloat(total, successes, draws) as [Value, Value, Value]
  validateParams('hypergeometric', {
    total: [m, 'nonNegativeInteger'],
    successes: [n, 'nonNegativeInteger'],
    draws: [nn, 'nonNegativeInteger'],
  })
  const batchShape = batchShapeOf(m, n, nn)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const low = maximum(sub(nn, sub(m, n)), 0)
  const high = minimum(n, nn)
  const inside = (x: Value): Value => logicalAnd(isCount(x), logicalAnd(greaterEqual(x, low), lessEqual(x, high)))
  const logPmf = (x: Value): Value => {
    const k = where(inside(x), x, low)
    const lp = sub(add(lbinom(n, k), lbinom(sub(m, n), sub(nn, k))), lbinom(m, nn))
    return where(inside(x), lp, Number.NEGATIVE_INFINITY)
  }
  const kmax = (): number => largest(high)
  const twin = concreteTwin([m, n, nn], (a, c, d) => hypergeometric(a as Tensor, c as Tensor, d as Tensor))
  return defineDistribution({
    name: 'hypergeometric',
    params: { total: m, successes: n, draws: nn },
    support: { kind: 'integerInterval', low: 0, high: largest(nn) },
    batchShape,
    logProb: logPmf,
    cdf: (x) => where(greaterEqual(floor(x), high), 1, cdfBySum(logPmf, x, kmax(), batchShape)),
    ppf: (q) => discretePpf(twin(), q, 0, b(high)),
    sample: (key: Key, shape) =>
      elementwiseSample(key, [m, n, nn], shape, batchShape, (k, mm, ss, dd) => {
        // Draw without replacement: each draw is a success with probability (successes left) / (objects left).
        const u = uniformNoise(k, [dd ?? 0]).data
        let left = mm ?? 0
        let good = ss ?? 0
        let count = 0
        for (const v of u) {
          if (v * left < good) {
            count++
            good--
          }
          left--
        }
        return count
      }),
    mean: () => b(div(mul(nn, n), m)),
    variance: () => b(div(mul(mul(mul(nn, n), sub(m, n)), sub(m, nn)), mul(square(m), sub(m, 1)))),
    mode: () => b(floor(div(mul(add(nn, 1), add(n, 1)), add(m, 2)))),
    entropy: () => entropyBySum(logPmf, kmax(), batchShape.length),
  })
}

/** The zero-inflation gate: the probability of a structural zero, or its logit. */
export type GateParams = { readonly probs: TensorLike } | { readonly logits: TensorLike }

/**
 * A zero-inflated count distribution: a structural zero with probability `gate`, otherwise a draw from `base`
 * (NumPyro's ZeroInflatedDistribution): P(0) = gate + (1 - gate) p(0) and P(k) = (1 - gate) p(k) for k > 0.
 *
 * @param base - The count distribution (its support must contain 0).
 * @param gate - `{ probs }` or `{ logits }` of the structural zero.
 * @returns The distribution (no reparameterisation); quantiles, mode and entropy from the pmf.
 * @example
 * ```ts
 * import { poisson, zeroInflated } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(zeroInflated(poisson(2), { probs: 0.3 }).prob(0)) - (0.3 + 0.7 * Math.exp(-2))) < 1e-15 // => true
 * ```
 */
export function zeroInflated(base: Distribution, gate: GateParams): Distribution {
  const g = binaryLogits(gate, 'zeroInflated')
  const pi = expit(g)
  const logPi = logSigmoid(g)
  const logKeep = logSigmoid(neg(g))
  const batchShape = batchShapeOf(g, promoteFloat(base.mean())[0] as Value)
  const logPmf = (x: Value): Value => {
    const lp = add(logKeep, base.logProb(x))
    return where(equal(x, 0), logaddexp(logPi, add(logKeep, base.logProb(0))), lp)
  }
  const kmax = (): number => largest(base.ppf(1 - 1e-15)) + 10
  const self: Distribution = defineDistribution({
    name: 'zeroInflated',
    params: { ...base.params, gate: pi },
    support: base.support,
    batchShape,
    logProb: logPmf,
    cdf: (x) => where(less(x, 0), 0, add(pi, mul(sub(1, pi), base.cdf(x)))),
    sf: (x) => where(less(x, 0), 1, mul(sub(1, pi), base.sf(x))),
    ppf: (q) => discretePpf(self, q, 0, Number.POSITIVE_INFINITY),
    sample: (key: Key, shape) => {
      const x = base.sample(foldIn(key, 0), shape)
      return discreteSample([x, pi], [...shape, ...batchShape], 'float64', ([draw, gp]) => {
        const u = uniformNoise(foldIn(key, 1), [...shape, ...batchShape])
        return where(less(u, gp as Tensor), 0, draw as Tensor) as Tensor
      })
    },
    mean: () => mul(sub(1, pi), base.mean()),
    variance: () => {
      const mu = base.mean()
      return mul(sub(1, pi), add(base.variance(), mul(pi, square(mu))))
    },
    mode: () => modeBySearch(logPmf, kmax(), batchShape.length),
    entropy: () => entropyBySum(logPmf, kmax(), batchShape.length),
  })
  return self
}

/**
 * The zero-inflated Poisson distribution (statsmodels' zipoisson(rate, gate)).
 *
 * @param rate - Poisson rate.
 * @param gate - Probability of a structural zero.
 * @returns The distribution.
 * @example
 * ```ts
 * import { zeroInflatedPoisson } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(zeroInflatedPoisson(4, 0.25).mean()) // => 3
 * ```
 */
export function zeroInflatedPoisson(rate: TensorLike, gate: TensorLike): Distribution {
  const d = zeroInflated(poisson(rate), { probs: gate })
  return { ...d, name: 'zeroInflatedPoisson' }
}

/**
 * The zero-inflated negative binomial distribution: a structural zero with probability `gate`, otherwise
 * `negativeBinomial(n, probs)`.
 *
 * @param n - Number of successes of the negative binomial.
 * @param probs - Its success probability.
 * @param gate - Probability of a structural zero.
 * @returns The distribution.
 * @example
 * ```ts
 * import { zeroInflatedNegativeBinomial } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(zeroInflatedNegativeBinomial(3, 0.5, 0.5).mean()) // => 1.5
 * ```
 */
export function zeroInflatedNegativeBinomial(n: TensorLike, probs: TensorLike, gate: TensorLike): Distribution {
  const d = zeroInflated(negativeBinomial(n, probs), { probs: gate })
  return { ...d, name: 'zeroInflatedNegativeBinomial' }
}
