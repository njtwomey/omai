import {
  digamma,
  gammainc,
  gammaincc,
  gammainccinv,
  gammaincinv,
  lgamma,
  logGammainc,
  logGammaincc,
} from '../../numerics/special/index.ts'
import { exponential as exponentialNoise } from '../../random/index.ts'
import {
  add,
  div,
  expm1,
  greaterEqual,
  log,
  log1p,
  maximum,
  mul,
  neg,
  promoteFloat,
  square,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { gammaSample } from '../_sampling.ts'
import { batchShapeOf, type Distribution, defineDistribution } from '../distribution.ts'

/**
 * The gamma distribution with shape `concentration` and inverse scale `rate` (scipy.stats.gamma with scale = 1 / rate,
 * torch.distributions.Gamma).
 *
 * @param concentration - Shape alpha (positive).
 * @param rate - Rate beta (positive).
 * @returns The distribution; `rsample` uses implicit reparameterisation, so samples differentiate in both parameters.
 * @concept distributions
 * @example
 * ```ts
 * import { gamma } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(gamma(2, 4).mean()) // => 0.5
 * ```
 */
export function gamma(concentration: TensorLike, rate: TensorLike = 1): Distribution {
  const [alpha, beta] = promoteFloat(concentration, rate) as [Value, Value]
  const batchShape = batchShapeOf(alpha, beta)
  const positive = (x: Value): Value => greaterEqual(x, 0)
  const safe = (x: Value): Value => where(positive(x), x, 1)
  return defineDistribution({
    name: 'gamma',
    params: { concentration: alpha, rate: beta },
    support: { kind: 'positive' },
    batchShape,
    logProb: (x) =>
      where(
        positive(x),
        sub(sub(add(mul(alpha, log(beta)), mul(sub(alpha, 1), log(safe(x)))), mul(beta, safe(x))), lgamma(alpha)),
        Number.NEGATIVE_INFINITY,
      ),
    cdf: (x) => where(positive(x), gammainc(alpha, mul(beta, safe(x))), 0),
    logCdf: (x) => where(positive(x), logGammainc(alpha, mul(beta, safe(x))), Number.NEGATIVE_INFINITY),
    sf: (x) => where(positive(x), gammaincc(alpha, mul(beta, safe(x))), 1),
    logSf: (x) => where(positive(x), logGammaincc(alpha, mul(beta, safe(x))), 0),
    ppf: (q) => div(gammaincinv(alpha, q), beta),
    isf: (q) => div(gammainccinv(alpha, q), beta),
    rsample: (key, shape) => div(gammaSample(key, add(alpha, mul(beta, 0)), shape), beta),
    mean: () => div(alpha, beta),
    variance: () => div(alpha, square(beta)),
    mode: () => div(maximum(sub(alpha, 1), 0), beta),
    entropy: () => add(sub(add(alpha, lgamma(alpha)), log(beta)), mul(sub(1, alpha), digamma(alpha))),
  })
}

/**
 * The exponential distribution with rate `rate` (scipy.stats.expon with scale = 1 / rate).
 *
 * @param rate - Rate (positive).
 * @returns The distribution; `rsample` is E / rate for unit exponential noise E.
 * @example
 * ```ts
 * import { exponential } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(exponential(2).mean()) // => 0.5
 * ```
 */
export function exponential(rate: TensorLike = 1): Distribution {
  const [lambda] = promoteFloat(rate) as [Value]
  const batchShape = batchShapeOf(lambda)
  const positive = (x: Value): Value => greaterEqual(x, 0)
  return defineDistribution({
    name: 'exponential',
    params: { rate: lambda },
    support: { kind: 'nonNegative' },
    batchShape,
    logProb: (x) => where(positive(x), sub(log(lambda), mul(lambda, x)), Number.NEGATIVE_INFINITY),
    cdf: (x) => where(positive(x), neg(expm1(neg(mul(lambda, x)))), 0),
    logSf: (x) => where(positive(x), neg(mul(lambda, x)), 0),
    ppf: (q) => div(neg(log1p(neg(q))), lambda),
    isf: (q) => div(neg(log(q)), lambda),
    rsample: (key, shape) => div(exponentialNoise(key, [...shape, ...batchShape]), lambda),
    mean: () => div(1, lambda),
    variance: () => div(1, square(lambda)),
    mode: () => mul(lambda, 0),
    entropy: () => sub(1, log(lambda)),
  })
}
