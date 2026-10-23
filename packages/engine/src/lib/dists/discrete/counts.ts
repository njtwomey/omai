import { betainc, gammaincc, lgamma, xlog1py, xlogy } from '../../numerics/special/index.ts'
import {
  foldIn,
  gammaVariates,
  geometric as geometricDraw,
  type Key,
  poisson as poissonDraw,
} from '../../random/index.ts'
import {
  add,
  div,
  expm1,
  floor,
  greaterEqual,
  less,
  log,
  log1p,
  logicalAnd,
  mul,
  neg,
  promoteFloat,
  square,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { elementwiseSample } from '../_sampling.ts'
import { batchShapeOf, type Distribution, defineDistribution } from '../distribution.ts'
import { entropyBySum, isCount, largest } from './_support.ts'

/**
 * The Poisson distribution with rate lambda (scipy.stats.poisson).
 *
 * @param rate - The rate (non-negative).
 * @returns The distribution (no reparameterisation).
 * @concept distributions
 * @example
 * ```ts
 * import { poisson } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(poisson(2).prob(0)) - Math.exp(-2)) < 1e-15 // => true
 * ```
 */
export function poisson(rate: TensorLike): Distribution {
  const [lambda] = promoteFloat(rate) as [Value]
  const batchShape = batchShapeOf(lambda)
  const logPmf = (x: Value): Value =>
    where(isCount(x), sub(sub(xlogy(x, lambda), lambda), lgamma(add(x, 1))), Number.NEGATIVE_INFINITY)
  return defineDistribution({
    name: 'poisson',
    params: { rate: lambda },
    support: { kind: 'integerInterval', low: 0, high: Number.POSITIVE_INFINITY },
    batchShape,
    logProb: logPmf,
    cdf: (x) => where(less(x, 0), 0, gammaincc(add(floor(where(less(x, 0), 0, x)), 1), lambda)),
    sample: (key: Key, shape) =>
      elementwiseSample(key, [lambda], shape, batchShape, (k, l) => poissonDraw(k, l ?? 0, []).data[0] ?? 0),
    mean: () => lambda,
    variance: () => lambda,
    mode: () => floor(lambda),
    entropy: () => {
      const lmax = largest(lambda, 'poisson')
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
 * import { geometric } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(geometric(0.25).mean()) // => 4
 * ```
 */
export function geometric(probs: TensorLike): Distribution {
  const [p] = promoteFloat(probs) as [Value]
  const batchShape = batchShapeOf(p)
  const inside = (x: Value): Value => logicalAnd(isCount(x), greaterEqual(x, 1))
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
      return where(less(k, 1), 0, neg(expm1(mul(k, log1p(neg(p))))))
    },
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
 * import { negativeBinomial } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(negativeBinomial(3, 0.5).mean()) // => 3
 * ```
 */
export function negativeBinomial(n: TensorLike, probs: TensorLike): Distribution {
  const [r, p] = promoteFloat(n, probs) as [Value, Value]
  const batchShape = batchShapeOf(r, p)
  const logPmf = (x: Value): Value => {
    const safe = where(isCount(x), x, 0)
    const lp = add(
      add(sub(sub(lgamma(add(safe, r)), lgamma(add(safe, 1))), lgamma(r)), mul(r, log(p))),
      xlog1py(safe, neg(p)),
    )
    return where(isCount(x), lp, Number.NEGATIVE_INFINITY)
  }
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
      const m = largest(div(mul(r, sub(1, p)), p), 'negativeBinomial')
      const v = largest(div(mul(r, sub(1, p)), square(p)), 'negativeBinomial')
      // The tail is geometric-like, so sum far out: 40 standard deviations past the mean.
      return entropyBySum(logPmf, Math.ceil(m + 40 * Math.sqrt(v) + 100), batchShape.length)
    },
  })
}
