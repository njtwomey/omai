/**
 * The gamma family and its relatives on the positive reals: gamma, Erlang, exponential, inverse gamma, chi-square,
 * Weibull and inverse Gaussian (Wald), with scipy.stats parameterisations noted on each.
 */

import {
  digamma,
  gamma as gammaFn,
  gammainc,
  gammaincc,
  gammainccinv,
  gammaincinv,
  lgamma,
  log1mexp,
  logaddexp,
  logGammainc,
  logGammaincc,
  logNdtr,
  ndtr,
} from '../../numerics/special/index.ts'
import {
  exponential as exponentialNoise,
  foldIn,
  type Key,
  normal as normalNoise,
  uniform,
} from '../../random/index.ts'
import {
  add,
  div,
  exp,
  expm1,
  greater,
  greaterEqual,
  lessEqual,
  log,
  log1p,
  maximum,
  mul,
  neg,
  pow,
  promoteFloat,
  sqrt,
  square,
  sub,
  type Tensor,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { concreteTwin, numericEntropy, numericPpf } from '../_numeric.ts'
import { gammaSample, implicitReparam, uniformOpen } from '../_sampling.ts'
import { batchShapeOf, broadcastBatch, type Distribution, defineDistribution, validateParams } from '../distribution.ts'

const EULER = 0.5772156649015329

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
  validateParams('gamma', { concentration: [alpha, 'positive'], rate: [beta, 'positive'] })
  const batchShape = batchShapeOf(alpha, beta)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
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
    rsample: (key, shape) => div(gammaSample(key, b(alpha), shape), beta),
    mean: () => b(div(alpha, beta)),
    variance: () => b(div(alpha, square(beta))),
    mode: () => b(div(maximum(sub(alpha, 1), 0), beta)),
    entropy: () => b(add(sub(add(alpha, lgamma(alpha)), log(beta)), mul(sub(1, alpha), digamma(alpha)))),
  })
}

/**
 * The Erlang distribution: the waiting time for `k` events of a Poisson process with rate `rate`, a gamma with integer
 * shape (scipy.stats.erlang(k, scale=1 / rate)).
 *
 * @param k - Number of events (a positive integer).
 * @param rate - Rate (positive).
 * @returns The distribution (the gamma's methods, named `erlang`).
 * @throws WhiteboxError with code `value` when k is not a positive integer.
 * @example
 * ```ts
 * import { erlang } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(erlang(3, 2).mean()) // => 1.5
 * ```
 */
export function erlang(k: TensorLike, rate: TensorLike = 1): Distribution {
  const [shape, r] = promoteFloat(k, rate) as [Value, Value]
  validateParams('erlang', { k: [shape, 'positiveInteger'] })
  return { ...gamma(shape, r), name: 'erlang', params: { k: shape, rate: r } }
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
  validateParams('exponential', { rate: [lambda, 'positive'] })
  const batchShape = batchShapeOf(lambda)
  const positive = (x: Value): Value => greaterEqual(x, 0)
  const safe = (x: Value): Value => where(positive(x), x, 0)
  return defineDistribution({
    name: 'exponential',
    params: { rate: lambda },
    support: { kind: 'nonNegative' },
    batchShape,
    logProb: (x) => where(positive(x), sub(log(lambda), mul(lambda, x)), Number.NEGATIVE_INFINITY),
    cdf: (x) => where(positive(x), neg(expm1(neg(mul(lambda, safe(x))))), 0),
    logCdf: (x) => where(positive(x), log1mexp(neg(mul(lambda, safe(x)))), Number.NEGATIVE_INFINITY),
    sf: (x) => where(positive(x), exp(neg(mul(lambda, safe(x)))), 1),
    logSf: (x) => where(positive(x), neg(mul(lambda, safe(x))), 0),
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
  validateParams('inverseGamma', { concentration: [alpha, 'positive'], scale: [beta, 'positive'] })
  const batchShape = batchShapeOf(alpha, beta)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
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
    logCdf: (x) => where(positive(x), logGammaincc(alpha, div(beta, safe(x))), Number.NEGATIVE_INFINITY),
    sf: (x) => where(positive(x), gammainc(alpha, div(beta, safe(x))), 1),
    logSf: (x) => where(positive(x), logGammainc(alpha, div(beta, safe(x))), 0),
    ppf: (q) => div(beta, gammainccinv(alpha, q)),
    isf: (q) => div(beta, gammaincinv(alpha, q)),
    rsample: (key, shape) => div(beta, gammaSample(key, b(alpha), shape)),
    mean: () => b(where(greater(alpha, 1), div(beta, sub(alpha, 1)), Number.POSITIVE_INFINITY)),
    variance: () =>
      b(
        where(
          greater(alpha, 2),
          div(square(beta), mul(square(sub(alpha, 1)), sub(alpha, 2))),
          Number.POSITIVE_INFINITY,
        ),
      ),
    mode: () => b(div(beta, add(alpha, 1))),
    entropy: () => b(sub(add(add(alpha, log(beta)), lgamma(alpha)), mul(add(alpha, 1), digamma(alpha)))),
  })
}

/**
 * The chi-square distribution with k degrees of freedom (scipy.stats.chi2, torch.distributions.Chi2): Gamma(k / 2,
 * rate 1/2).
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
  validateParams('chi2', { df: [k, 'positive'] })
  return { ...gamma(mul(k, 0.5), 0.5), name: 'chi2', params: { df: k } }
}

/**
 * The Weibull distribution with shape k and scale lambda (scipy.stats.weibull_min(k, scale=lambda),
 * torch.distributions.Weibull).
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
  validateParams('weibull', { concentration: [k, 'positive'], scale: [lambda, 'positive'] })
  const batchShape = batchShapeOf(k, lambda)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const positive = (x: Value): Value => greaterEqual(x, 0)
  const safe = (x: Value): Value => where(positive(x), x, 1)
  const hazard = (x: Value): Value => pow(div(safe(x), lambda), k)
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
    cdf: (x) => where(positive(x), neg(expm1(neg(hazard(x)))), 0),
    logCdf: (x) => where(positive(x), log1mexp(neg(hazard(x))), Number.NEGATIVE_INFINITY),
    sf: (x) => where(positive(x), exp(neg(hazard(x))), 1),
    logSf: (x) => where(positive(x), neg(hazard(x)), 0),
    ppf,
    isf: (q) => mul(lambda, pow(neg(log(q)), div(1, k))),
    rsample: (key, shape) => ppf(uniformOpen(key, [...shape, ...batchShape])),
    mean: () => b(mul(lambda, g1)),
    variance: () => b(mul(square(lambda), sub(gammaFn(add(div(2, k), 1)), square(g1)))),
    mode: () => b(where(greater(k, 1), mul(lambda, pow(div(sub(k, 1), k), div(1, k))), mul(lambda, 0))),
    entropy: () => b(add(add(mul(sub(1, div(1, k)), EULER), log(div(lambda, k))), 1)),
  })
}

/**
 * The inverse Gaussian (Wald) distribution with mean mu and shape lambda (scipy.stats.invgauss(mu / lambda,
 * scale=lambda)): the first passage time of a Brownian motion with drift, and the GLM family with variance mu^3.
 *
 * @param mean - Mean mu (positive).
 * @param shape - Shape lambda (positive).
 * @returns The distribution. `sample` uses the transformation with multiple roots (Michael, Schucany and Haas, 1976),
 *   and `rsample` differentiates it by implicit reparameterisation through the closed-form CDF; quantiles and the
 *   entropy are numerical.
 * @example
 * ```ts
 * import { inverseGaussian } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(inverseGaussian(2, 4).variance()) // => 2
 * ```
 */
export function inverseGaussian(mean: TensorLike, shape: TensorLike = 1): Distribution {
  const [mu, lambda] = promoteFloat(mean, shape) as [Value, Value]
  validateParams('inverseGaussian', { mean: [mu, 'positive'], shape: [lambda, 'positive'] })
  const batchShape = batchShapeOf(mu, lambda)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const positive = (x: Value): Value => greater(x, 0)
  const safe = (x: Value): Value => where(positive(x), x, 1)
  // Phi(r (x / mu - 1)) + exp(2 lambda / mu) Phi(-r (x / mu + 1)), r = sqrt(lambda / x); the second term in logs.
  const parts = (x: Value): [Value, Value] => {
    const r = sqrt(div(lambda, safe(x)))
    const t = div(safe(x), mu)
    return [mul(r, sub(t, 1)), add(div(mul(lambda, 2), mu), logNdtr(neg(mul(r, add(t, 1)))))]
  }
  const twin = concreteTwin([mu, lambda], (m, l) => inverseGaussian(m as Tensor, l as Tensor))
  // Concrete draws by the transformation with multiple roots.
  const draw = (key: Key, shape: readonly number[]): Value => {
    const c = twin()
    const [m, l] = [c.params['mean'] as Value, c.params['shape'] as Value]
    const full = [...shape, ...batchShape]
    const y = square(normalNoise(foldIn(key, 0), full))
    const my = mul(m, y)
    const x1 = add(m, div(mul(m, sub(my, sqrt(add(mul(mul(l, my), 4), square(my))))), mul(l, 2)))
    const u = uniform(foldIn(key, 1), full)
    return where(lessEqual(u, div(m, add(m, x1))), x1, div(square(m), x1))
  }
  const self: Distribution = defineDistribution({
    name: 'inverseGaussian',
    params: { mean: mu, shape: lambda },
    support: { kind: 'positive' },
    batchShape,
    logProb: (x) => {
      const v = safe(x)
      const lp = sub(
        mul(sub(log(lambda), add(Math.log(2 * Math.PI), mul(log(v), 3))), 0.5),
        div(mul(lambda, square(sub(v, mu))), mul(mul(square(mu), v), 2)),
      )
      return where(positive(x), lp, Number.NEGATIVE_INFINITY)
    },
    cdf: (x) => {
      const [a, lb] = parts(x)
      return where(positive(x), add(ndtr(a), exp(lb)), 0)
    },
    logCdf: (x) => {
      const [a, lb] = parts(x)
      return where(positive(x), logaddexp(logNdtr(a), lb), Number.NEGATIVE_INFINITY)
    },
    sf: (x) => {
      const [a, lb] = parts(x)
      return where(positive(x), sub(ndtr(neg(a)), exp(lb)), 1)
    },
    ppf: (q) => numericPpf(self, twin, q),
    sample: (key, shape) => draw(key, shape),
    rsample: (key, shape) => implicitReparam(draw(key, shape), self.cdf, self.prob),
    mean: () => b(mu),
    variance: () => b(div(pow(mu, 3), lambda)),
    mode: () => {
      const r = div(mul(mu, 1.5), lambda)
      return b(mul(mu, sub(sqrt(add(square(r), 1)), r)))
    },
    entropy: () => numericEntropy(self, { kind: 'halfLine', low: 0, scale: b(div(square(mu), add(mu, lambda))) }),
  })
  return self
}
