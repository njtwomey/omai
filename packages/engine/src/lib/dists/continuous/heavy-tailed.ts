/**
 * Heavy-tailed and flexible-shape continuous distributions: Student's t, Cauchy and half-Cauchy, Laplace, logistic,
 * Pareto, Johnson's S_U and the sinh-arcsinh family of Jones and Pewsey, whose tails and skewness can be tuned, as
 * scipy.stats, torch.distributions and TFP.
 */

import {
  digamma,
  kv,
  lbeta,
  lgamma,
  logaddexp,
  logit,
  logNdtr,
  logSigmoid,
  ndtr,
  ndtri,
  normalLogPdf,
  stdtr,
  stdtrit,
} from '../../numerics/special/index.ts'
import { foldIn, normal as normalNoise } from '../../random/index.ts'
import {
  abs,
  add,
  arcsinh,
  arctan,
  cosh,
  div,
  exp,
  expm1,
  greaterEqual,
  less,
  log,
  log1p,
  mul,
  neg,
  pow,
  promoteFloat,
  sigmoid,
  sign,
  sinh,
  sqrt,
  square,
  sub,
  type Tensor,
  type TensorLike,
  tan,
  type Value,
  where,
} from '../../tensor/index.ts'
import { concreteTwin, numericEntropy, numericMode } from '../_numeric.ts'
import { gammaSample, uniformOpen } from '../_sampling.ts'
import { batchShapeOf, broadcastBatch, type Distribution, defineDistribution, validateParams } from '../distribution.ts'

const LOG_PI = Math.log(Math.PI)
const nanOutside = (ok: Value, v: Value): Value => where(ok, v, Number.NaN)

/**
 * Student's t distribution with `df` degrees of freedom, location and scale (scipy.stats.t,
 * torch.distributions.StudentT).
 *
 * @param df - Degrees of freedom (positive).
 * @param loc - Location.
 * @param scale - Scale (positive).
 * @returns The distribution; `rsample` is loc + scale eps sqrt(df / V) with V chi-square, differentiable in every
 *   parameter (through the implicitly reparameterised gamma).
 * @concept distributions
 * @index distribution Student's t distribution.
 * @example
 * ```ts
 * import { studentT } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(studentT(3).cdf(0)) // => 0.5
 * ```
 */
export function studentT(df: TensorLike, loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [nu, mu, sigma] = promoteFloat(df, loc, scale) as [Value, Value, Value]
  validateParams('studentT', { df: [nu, 'positive'], loc: [mu, 'real'], scale: [sigma, 'positive'] })
  const batchShape = batchShapeOf(nu, mu, sigma)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const z = (x: Value): Value => div(sub(x, mu), sigma)
  const half = mul(add(nu, 1), 0.5)
  return defineDistribution({
    name: 'studentT',
    params: { df: nu, loc: mu, scale: sigma },
    support: { kind: 'real' },
    batchShape,
    logProb: (x) =>
      sub(
        sub(sub(sub(lgamma(half), lgamma(mul(nu, 0.5))), mul(log(mul(nu, Math.PI)), 0.5)), log(sigma)),
        mul(half, log1p(div(square(z(x)), nu))),
      ),
    cdf: (x) => stdtr(nu, z(x)),
    sf: (x) => stdtr(nu, neg(z(x))),
    ppf: (q) => add(mu, mul(sigma, stdtrit(nu, q))),
    isf: (q) => sub(mu, mul(sigma, stdtrit(nu, q))),
    rsample: (key, shape) => {
      const eps = normalNoise(foldIn(key, 0), [...shape, ...batchShape])
      const v = mul(gammaSample(foldIn(key, 1), b(mul(nu, 0.5)), shape), 2)
      return add(mu, mul(mul(sigma, eps), sqrt(div(nu, v))))
    },
    mean: () => b(nanOutside(less(1, nu), b(mu))),
    variance: () => {
      const v = div(mul(square(sigma), nu), sub(nu, 2))
      return b(where(less(2, nu), v, where(less(1, nu), Number.POSITIVE_INFINITY, Number.NaN)))
    },
    mode: () => b(mu),
    entropy: () =>
      b(
        add(
          add(add(mul(half, sub(digamma(half), digamma(mul(nu, 0.5)))), mul(log(nu), 0.5)), lbeta(mul(nu, 0.5), 0.5)),
          log(sigma),
        ),
      ),
  })
}

/**
 * The Cauchy distribution (scipy.stats.cauchy): Student's t with one degree of freedom.
 *
 * @param loc - Location (the median).
 * @param scale - Scale (positive).
 * @returns The distribution; `rsample` is the quantile of uniform noise. Its mean and variance are undefined (NaN).
 * @example
 * ```ts
 * import { cauchy } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(cauchy(0, 1).cdf(1)) // => 0.75
 * ```
 */
export function cauchy(loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [mu, sigma] = promoteFloat(loc, scale) as [Value, Value]
  validateParams('cauchy', { loc: [mu, 'real'], scale: [sigma, 'positive'] })
  const batchShape = batchShapeOf(mu, sigma)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const z = (x: Value): Value => div(sub(x, mu), sigma)
  const ppf = (q: Value): Value => add(mu, mul(sigma, tan(mul(sub(q, 0.5), Math.PI))))
  return defineDistribution({
    name: 'cauchy',
    params: { loc: mu, scale: sigma },
    support: { kind: 'real' },
    batchShape,
    logProb: (x) => neg(add(add(log(sigma), LOG_PI), log1p(square(z(x))))),
    cdf: (x) => add(0.5, div(arctan(z(x)), Math.PI)),
    sf: (x) => sub(0.5, div(arctan(z(x)), Math.PI)),
    ppf,
    isf: (q) => sub(mu, mul(sigma, tan(mul(sub(q, 0.5), Math.PI)))),
    rsample: (key, shape) => ppf(uniformOpen(key, [...shape, ...batchShape])),
    mean: () => b(mul(mu, Number.NaN)),
    variance: () => b(mul(mu, Number.NaN)),
    mode: () => b(mu),
    entropy: () => b(add(log(sigma), Math.log(4 * Math.PI))),
  })
}

/**
 * The half-Cauchy distribution |Cauchy(0, scale)| (scipy.stats.halfcauchy), a common prior for scale parameters.
 *
 * @param scale - Scale (positive).
 * @returns The distribution; its mean and variance are infinite.
 * @example
 * ```ts
 * import { halfCauchy } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(halfCauchy(1).cdf(1)) // => 0.5
 * ```
 */
export function halfCauchy(scale: TensorLike = 1): Distribution {
  const [sigma] = promoteFloat(scale) as [Value]
  validateParams('halfCauchy', { scale: [sigma, 'positive'] })
  const batchShape = batchShapeOf(sigma)
  const inside = (x: Value): Value => greaterEqual(x, 0)
  const ppf = (q: Value): Value => mul(sigma, tan(mul(q, Math.PI / 2)))
  return defineDistribution({
    name: 'halfCauchy',
    params: { scale: sigma },
    support: { kind: 'nonNegative' },
    batchShape,
    logProb: (x) =>
      where(
        inside(x),
        neg(add(add(log(sigma), Math.log(Math.PI / 2)), log1p(square(div(x, sigma))))),
        Number.NEGATIVE_INFINITY,
      ),
    cdf: (x) => where(inside(x), mul(arctan(div(x, sigma)), 2 / Math.PI), 0),
    // 1 - (2 / pi) atan(x / s) = (2 / pi) atan(s / x), without cancellation in the upper tail.
    sf: (x) => where(inside(x), mul(arctan(div(sigma, where(inside(x), x, 1))), 2 / Math.PI), 1),
    ppf,
    isf: (q) => div(sigma, tan(mul(q, Math.PI / 2))),
    rsample: (key, shape) => ppf(uniformOpen(key, [...shape, ...batchShape])),
    mean: () => add(mul(sigma, 0), Number.POSITIVE_INFINITY),
    variance: () => add(mul(sigma, 0), Number.POSITIVE_INFINITY),
    mode: () => mul(sigma, 0),
    entropy: () => add(log(sigma), Math.log(2 * Math.PI)),
  })
}

/**
 * The Laplace (double exponential) distribution (scipy.stats.laplace, torch.distributions.Laplace).
 *
 * @param loc - Location.
 * @param scale - Scale b (positive).
 * @returns The distribution; `rsample` is the quantile of uniform noise.
 * @example
 * ```ts
 * import { laplace } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(laplace(0, 2).variance()) // => 8
 * ```
 */
export function laplace(loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [mu, s] = promoteFloat(loc, scale) as [Value, Value]
  validateParams('laplace', { loc: [mu, 'real'], scale: [s, 'positive'] })
  const batchShape = batchShapeOf(mu, s)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const z = (x: Value): Value => div(sub(x, mu), s)
  const ppf = (q: Value): Value => {
    const d = sub(q, 0.5)
    return sub(mu, mul(mul(s, sign(d)), log1p(mul(abs(d), -2))))
  }
  // The lower tail exp(z) / 2 for z < 0 and its mirror.
  const lowerTail = (v: Value): Value => mul(exp(neg(abs(v))), 0.5)
  return defineDistribution({
    name: 'laplace',
    params: { loc: mu, scale: s },
    support: { kind: 'real' },
    batchShape,
    logProb: (x) => neg(add(log(mul(s, 2)), abs(z(x)))),
    cdf: (x) => {
      const v = z(x)
      return where(less(v, 0), lowerTail(v), sub(1, lowerTail(v)))
    },
    sf: (x) => {
      const v = z(x)
      return where(less(v, 0), sub(1, lowerTail(v)), lowerTail(v))
    },
    logCdf: (x) => {
      const v = z(x)
      return where(less(v, 0), sub(v, Math.LN2), log1p(neg(lowerTail(v))))
    },
    logSf: (x) => {
      const v = z(x)
      return where(less(v, 0), log1p(neg(lowerTail(v))), sub(neg(v), Math.LN2))
    },
    ppf,
    isf: (q) => sub(mul(mu, 2), ppf(q)),
    rsample: (key, shape) => ppf(uniformOpen(key, [...shape, ...batchShape])),
    mean: () => b(mu),
    variance: () => b(mul(square(s), 2)),
    mode: () => b(mu),
    entropy: () => b(add(log(mul(s, 2)), 1)),
  })
}

/**
 * The logistic distribution (scipy.stats.logistic), whose CDF is the sigmoid.
 *
 * @param loc - Location.
 * @param scale - Scale s (positive).
 * @returns The distribution; `rsample` is loc + s logit(U).
 * @example
 * ```ts
 * import { logistic } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(logistic(0, 1).cdf(0)) // => 0.5
 * ```
 */
export function logistic(loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [mu, s] = promoteFloat(loc, scale) as [Value, Value]
  validateParams('logistic', { loc: [mu, 'real'], scale: [s, 'positive'] })
  const batchShape = batchShapeOf(mu, s)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const z = (x: Value): Value => div(sub(x, mu), s)
  const ppf = (q: Value): Value => add(mu, mul(s, logit(q)))
  return defineDistribution({
    name: 'logistic',
    params: { loc: mu, scale: s },
    support: { kind: 'real' },
    batchShape,
    logProb: (x) => {
      const v = z(x)
      return sub(sub(neg(v), log(s)), mul(logaddexp(0, neg(v)), 2))
    },
    cdf: (x) => sigmoid(z(x)),
    sf: (x) => sigmoid(neg(z(x))),
    logCdf: (x) => logSigmoid(z(x)),
    logSf: (x) => logSigmoid(neg(z(x))),
    ppf,
    isf: (q) => sub(mu, mul(s, logit(q))),
    rsample: (key, shape) => ppf(uniformOpen(key, [...shape, ...batchShape])),
    mean: () => b(mu),
    variance: () => b(mul(square(s), (Math.PI * Math.PI) / 3)),
    mode: () => b(mu),
    entropy: () => b(add(log(s), 2)),
  })
}

/**
 * The Pareto distribution with shape b and scale x_m (scipy.stats.pareto(b, scale=x_m), torch.distributions.Pareto),
 * on [x_m, inf).
 *
 * @param b - Shape (positive).
 * @param scale - Scale x_m (positive), the minimum value.
 * @returns The distribution; `rsample` is the quantile of uniform noise.
 * @example
 * ```ts
 * import { pareto } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(pareto(3, 1).mean()) // => 1.5
 * ```
 */
export function pareto(b: TensorLike, scale: TensorLike = 1): Distribution {
  const [alpha, xm] = promoteFloat(b, scale) as [Value, Value]
  validateParams('pareto', { b: [alpha, 'positive'], scale: [xm, 'positive'] })
  const batchShape = batchShapeOf(alpha, xm)
  const bb = (v: Value): Value => broadcastBatch(v, batchShape)
  const inside = (x: Value): Value => greaterEqual(x, xm)
  const safe = (x: Value): Value => where(inside(x), x, xm)
  const logSf = (x: Value): Value => mul(alpha, log(div(xm, safe(x))))
  const ppf = (q: Value): Value => mul(xm, pow(sub(1, q), div(-1, alpha)))
  return defineDistribution({
    name: 'pareto',
    params: { b: alpha, scale: xm },
    support: { kind: 'greaterThan', low: xm },
    batchShape,
    logProb: (x) =>
      where(
        inside(x),
        sub(add(log(alpha), mul(alpha, log(xm))), mul(add(alpha, 1), log(safe(x)))),
        Number.NEGATIVE_INFINITY,
      ),
    cdf: (x) => where(inside(x), neg(expm1(logSf(x))), 0),
    sf: (x) => where(inside(x), exp(logSf(x)), 1),
    logSf: (x) => where(inside(x), logSf(x), 0),
    ppf,
    isf: (q) => mul(xm, pow(q, div(-1, alpha))),
    rsample: (key, shape) => ppf(uniformOpen(key, [...shape, ...batchShape])),
    mean: () => bb(where(less(1, alpha), div(mul(alpha, xm), sub(alpha, 1)), Number.POSITIVE_INFINITY)),
    variance: () =>
      bb(
        where(
          less(2, alpha),
          div(mul(square(xm), alpha), mul(square(sub(alpha, 1)), sub(alpha, 2))),
          Number.POSITIVE_INFINITY,
        ),
      ),
    mode: () => bb(xm),
    entropy: () => bb(add(add(log(div(xm, alpha)), div(1, alpha)), 1)),
  })
}

/**
 * Johnson's S_U distribution (scipy.stats.johnsonsu(a, b, loc, scale)): z = a + b asinh((x - loc) / scale) is
 * standard normal. A four-parameter family covering any skewness and kurtosis above the lognormal line.
 *
 * @param a - Shape gamma (skewness; real).
 * @param b - Shape delta (tail weight; positive).
 * @param loc - Location.
 * @param scale - Scale (positive).
 * @returns The distribution with closed-form density, CDF, quantiles and moments; `rsample` is
 *   loc + scale sinh((eps - a) / b); the mode and entropy are numerical (as in scipy).
 * @example
 * ```ts
 * import { johnsonSU } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(johnsonSU(0, 1).cdf(0)) // => 0.5
 * ```
 */
export function johnsonSU(a: TensorLike, b: TensorLike, loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [ga, de, mu, s] = promoteFloat(a, b, loc, scale) as [Value, Value, Value, Value]
  validateParams('johnsonSU', { a: [ga, 'finite'], b: [de, 'positive'], loc: [mu, 'real'], scale: [s, 'positive'] })
  const batchShape = batchShapeOf(ga, de, mu, s)
  const y = (x: Value): Value => div(sub(x, mu), s)
  const z = (x: Value): Value => add(ga, mul(de, arcsinh(y(x))))
  const fromNormal = (e: Value): Value => add(mu, mul(s, sinh(div(sub(e, ga), de))))
  const w = exp(div(1, square(de)))
  const mean = sub(mu, mul(mul(s, sqrt(w)), sinh(div(ga, de))))
  const variance = mul(mul(square(s), 0.5), mul(sub(w, 1), add(mul(w, cosh(mul(div(ga, de), 2))), 1)))
  const twin = concreteTwin([ga, de, mu, s], (p, q, r, t) =>
    johnsonSU(p as Tensor, q as Tensor, r as Tensor, t as Tensor),
  )
  const self: Distribution = defineDistribution({
    name: 'johnsonSU',
    params: { a: ga, b: de, loc: mu, scale: s },
    support: { kind: 'real' },
    batchShape,
    logProb: (x) => sub(add(normalLogPdf(z(x)), log(de)), add(log(s), mul(log1p(square(y(x))), 0.5))),
    cdf: (x) => ndtr(z(x)),
    sf: (x) => ndtr(neg(z(x))),
    logCdf: (x) => logNdtr(z(x)),
    logSf: (x) => logNdtr(neg(z(x))),
    ppf: (q) => fromNormal(ndtri(q)),
    isf: (q) => fromNormal(neg(ndtri(q))),
    rsample: (key, shape) => fromNormal(normalNoise(key, [...shape, ...batchShape])),
    mean: () => broadcastBatch(mean, batchShape),
    variance: () => broadcastBatch(variance, batchShape),
    mode: () => {
      const sd = sqrt(variance)
      return numericMode(self, twin, sub(mean, mul(sd, 3)), add(mean, mul(sd, 3)))
    },
    entropy: () => numericEntropy(self, { kind: 'real', center: mean, scale: sqrt(variance) }),
  })
  return self
}

/**
 * The sinh-arcsinh distribution of Jones and Pewsey (2009): X = loc + scale sinh((asinh(Z) + skewness) tailweight)
 * for standard normal Z (their epsilon = skewness, delta = 1 / tailweight; TensorFlow Probability's SinhArcsinh without
 * its 2 / F_0(2) rescaling). Tailweight 1 and skewness 0 give the normal; tailweight above 1 thickens the tails.
 *
 * @param loc - Location.
 * @param scale - Scale (positive).
 * @param skewness - Skewness (real; positive skews right).
 * @param tailweight - Tail weight (positive).
 * @returns The distribution with closed-form density, CDF and quantiles, `rsample` through the transform, and moments
 *   through Bessel functions K (Jones and Pewsey's P_q); the mode and entropy are numerical.
 * @reference Jones, M. C. and Pewsey, A. (2009). Sinh-arcsinh distributions. Biometrika 96(4).
 * @example
 * ```ts
 * import { sinhArcsinh } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(sinhArcsinh(0, 1, 0, 1).cdf(0)) // => 0.5
 * Math.abs(item(sinhArcsinh(0, 1, 0, 1).variance()) - 1) < 1e-12 // => true
 * ```
 */
export function sinhArcsinh(
  loc: TensorLike = 0,
  scale: TensorLike = 1,
  skewness: TensorLike = 0,
  tailweight: TensorLike = 1,
): Distribution {
  const [mu, s, eps, tw] = promoteFloat(loc, scale, skewness, tailweight) as [Value, Value, Value, Value]
  validateParams('sinhArcsinh', {
    loc: [mu, 'real'],
    scale: [s, 'positive'],
    skewness: [eps, 'finite'],
    tailweight: [tw, 'positive'],
  })
  const batchShape = batchShapeOf(mu, s, eps, tw)
  const delta = div(1, tw)
  const y = (x: Value): Value => div(sub(x, mu), s)
  // The standard normal value of x: sinh(delta asinh(y) - epsilon).
  const inner = (x: Value): Value => sub(mul(delta, arcsinh(y(x))), eps)
  const z = (x: Value): Value => sinh(inner(x))
  const fromNormal = (e: Value): Value => add(mu, mul(s, sinh(mul(add(arcsinh(e), eps), tw))))
  // P_q = e^{1/4} / sqrt(8 pi) (K_{(q+1)/2}(1/4) + K_{(q-1)/2}(1/4)).
  const p = (q: Value): Value =>
    mul(add(kv(mul(add(q, 1), 0.5), 0.25), kv(mul(sub(q, 1), 0.5), 0.25)), Math.exp(0.25) / Math.sqrt(8 * Math.PI))
  const m1 = mul(sinh(mul(eps, tw)), p(tw))
  const m2 = mul(sub(mul(cosh(mul(mul(eps, tw), 2)), p(mul(tw, 2))), 1), 0.5)
  const mean = add(mu, mul(s, m1))
  const variance = mul(square(s), sub(m2, square(m1)))
  const twin = concreteTwin([mu, s, eps, tw], (a, b, c, d) =>
    sinhArcsinh(a as Tensor, b as Tensor, c as Tensor, d as Tensor),
  )
  const self: Distribution = defineDistribution({
    name: 'sinhArcsinh',
    params: { loc: mu, scale: s, skewness: eps, tailweight: tw },
    support: { kind: 'real' },
    batchShape,
    logProb: (x) =>
      sub(add(add(normalLogPdf(z(x)), log(delta)), log(cosh(inner(x)))), add(log(s), mul(log1p(square(y(x))), 0.5))),
    cdf: (x) => ndtr(z(x)),
    sf: (x) => ndtr(neg(z(x))),
    logCdf: (x) => logNdtr(z(x)),
    logSf: (x) => logNdtr(neg(z(x))),
    ppf: (q) => fromNormal(ndtri(q)),
    isf: (q) => fromNormal(neg(ndtri(q))),
    rsample: (key, shape) => fromNormal(normalNoise(key, [...shape, ...batchShape])),
    mean: () => broadcastBatch(mean, batchShape),
    variance: () => broadcastBatch(variance, batchShape),
    mode: () => {
      const sd = sqrt(variance)
      return numericMode(self, twin, sub(mean, mul(sd, 3)), add(mean, mul(sd, 3)))
    },
    entropy: () => numericEntropy(self, { kind: 'real', center: mean, scale: sqrt(variance) }),
  })
  return self
}
