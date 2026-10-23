import {
  digamma,
  gamma as gammaFn,
  gammainc,
  gammaincc,
  gammainccinv,
  gammaincinv,
  lgamma,
  logGammainc,
  logGammaincc,
} from '../../numerics/special/index.ts'
import { exponential as exponentialNoise, uniform as uniformNoise } from '../../random/index.ts'
import {
  add,
  div,
  expm1,
  greater,
  greaterEqual,
  log,
  log1p,
  maximum,
  mul,
  neg,
  pow,
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

/**
 * The inverse gamma distribution with shape alpha and scale beta (scipy.stats.invgamma(alpha, scale=beta)): 1 / X for
 * X ~ Gamma(alpha, rate beta).
 *
 * @param concentration - Shape alpha (positive).
 * @param scale - Scale beta (positive).
 * @returns The distribution; `rsample` is beta / G with G an implicitly reparameterised gamma sample.
 * @example
 * ```ts
 * import { inverseGamma } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(inverseGamma(3, 4).mean()) // => 2
 * ```
 */
export function inverseGamma(concentration: TensorLike, scale: TensorLike = 1): Distribution {
  const [alpha, beta] = promoteFloat(concentration, scale) as [Value, Value]
  const batchShape = batchShapeOf(alpha, beta)
  const positive = (x: Value): Value => greater(x, 0)
  const safe = (x: Value): Value => where(positive(x), x, 1)
  return defineDistribution({
    name: 'inverseGamma',
    params: { concentration: alpha, scale: beta },
    support: { kind: 'positive' },
    batchShape,
    logProb: (x) =>
      where(
        positive(x),
        sub(sub(sub(mul(alpha, log(beta)), lgamma(alpha)), mul(add(alpha, 1), log(safe(x)))), div(beta, safe(x))),
        Number.NEGATIVE_INFINITY,
      ),
    cdf: (x) => where(positive(x), gammaincc(alpha, div(beta, safe(x))), 0),
    sf: (x) => where(positive(x), gammainc(alpha, div(beta, safe(x))), 1),
    ppf: (q) => div(beta, gammainccinv(alpha, q)),
    isf: (q) => div(beta, gammaincinv(alpha, q)),
    rsample: (key, shape) => div(beta, gammaSample(key, add(alpha, mul(beta, 0)), shape)),
    mean: () => where(greater(alpha, 1), div(beta, sub(alpha, 1)), Number.POSITIVE_INFINITY),
    variance: () =>
      where(greater(alpha, 2), div(square(beta), mul(square(sub(alpha, 1)), sub(alpha, 2))), Number.POSITIVE_INFINITY),
    mode: () => div(beta, add(alpha, 1)),
    entropy: () => sub(add(add(alpha, log(beta)), lgamma(alpha)), mul(add(alpha, 1), digamma(alpha))),
  })
}

/**
 * The chi-square distribution with k degrees of freedom (scipy.stats.chi2): Gamma(k / 2, rate 1/2).
 *
 * @param df - Degrees of freedom (positive).
 * @returns The distribution (named `chi2`), with implicit reparameterisation in df.
 * @example
 * ```ts
 * import { chi2 } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(chi2(4).variance()) // => 8
 * ```
 */
export function chi2(df: TensorLike): Distribution {
  const [k] = promoteFloat(df) as [Value]
  return { ...gamma(mul(k, 0.5), 0.5), name: 'chi2', params: { df: k } }
}

/**
 * The Weibull distribution with shape k and scale lambda (scipy.stats.weibull_min(k, scale=lambda)).
 *
 * @param concentration - Shape k (positive).
 * @param scale - Scale lambda (positive).
 * @returns The distribution; `rsample` is the quantile of uniform noise.
 * @example
 * ```ts
 * import { weibull } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(weibull(1, 2).mean()) // => 2
 * ```
 */
export function weibull(concentration: TensorLike, scale: TensorLike = 1): Distribution {
  const [k, lambda] = promoteFloat(concentration, scale) as [Value, Value]
  const batchShape = batchShapeOf(k, lambda)
  const positive = (x: Value): Value => greaterEqual(x, 0)
  const safe = (x: Value): Value => where(positive(x), x, 1)
  const ppf = (q: Value): Value => mul(lambda, pow(neg(log1p(neg(q))), div(1, k)))
  const g1 = gammaFn(add(div(1, k), 1))
  return defineDistribution({
    name: 'weibull',
    params: { concentration: k, scale: lambda },
    support: { kind: 'nonNegative' },
    batchShape,
    logProb: (x) => {
      const r = div(safe(x), lambda)
      return where(
        positive(x),
        sub(add(log(div(k, lambda)), mul(sub(k, 1), log(r))), pow(r, k)),
        Number.NEGATIVE_INFINITY,
      )
    },
    cdf: (x) => where(positive(x), neg(expm1(neg(pow(div(safe(x), lambda), k)))), 0),
    logSf: (x) => where(positive(x), neg(pow(div(safe(x), lambda), k)), 0),
    ppf,
    rsample: (key, shape) => ppf(uniformNoise(key, [...shape, ...batchShape])),
    mean: () => mul(lambda, g1),
    variance: () => mul(square(lambda), sub(gammaFn(add(div(2, k), 1)), square(g1))),
    mode: () => where(greater(k, 1), mul(lambda, pow(div(sub(k, 1), k), div(1, k))), mul(lambda, 0)),
    entropy: () => add(add(mul(sub(1, div(1, k)), EULER), log(div(lambda, k))), 1),
  })
}

const EULER = 0.5772156649015329
