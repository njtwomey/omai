import { digamma, lbeta, lgamma, logaddexp, logit, stdtr, stdtrit } from '../../numerics/special/index.ts'
import { normal as normalNoise, split, uniform as uniformNoise } from '../../random/index.ts'
import {
  abs,
  add,
  arctan,
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
  sqrt,
  square,
  sub,
  type TensorLike,
  tan,
  type Value,
  where,
} from '../../tensor/index.ts'
import { gammaSample } from '../_sampling.ts'
import { batchShapeOf, type Distribution, defineDistribution } from '../distribution.ts'

const LOG_PI = Math.log(Math.PI)
const like = (v: Value, ...others: readonly Value[]): Value => others.reduce((a, o) => add(a, mul(o, 0)), v)
const nanOutside = (ok: Value, v: Value): Value => where(ok, v, Number.NaN)

/**
 * Student's t distribution with `df` degrees of freedom, location and scale (scipy.stats.t).
 *
 * @param df - Degrees of freedom (positive).
 * @param loc - Location.
 * @param scale - Scale (positive).
 * @returns The distribution; `rsample` is loc + scale eps sqrt(df / V) with V chi-square, differentiable in every
 *   parameter (through the implicitly reparameterised gamma).
 * @concept distributions
 * @example
 * ```ts
 * import { studentT } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(studentT(3).cdf(0)) // => 0.5
 * ```
 */
export function studentT(df: TensorLike, loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [nu, mu, sigma] = promoteFloat(df, loc, scale) as [Value, Value, Value]
  const batchShape = batchShapeOf(nu, mu, sigma)
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
    rsample: (key, shape) => {
      const [k1, k2] = split(key)
      const eps = normalNoise(k1 as never, [...shape, ...batchShape])
      const v = mul(gammaSample(k2 as never, like(mul(nu, 0.5), mu, sigma), shape), 2)
      return add(mu, mul(mul(sigma, eps), sqrt(div(nu, v))))
    },
    mean: () => nanOutside(less(1, nu), like(mu, nu, sigma)),
    variance: () => {
      const v = div(mul(square(sigma), nu), sub(nu, 2))
      return where(less(2, nu), v, where(less(1, nu), Number.POSITIVE_INFINITY, Number.NaN))
    },
    mode: () => like(mu, nu, sigma),
    entropy: () =>
      add(
        add(add(mul(half, sub(digamma(half), digamma(mul(nu, 0.5)))), mul(log(nu), 0.5)), lbeta(mul(nu, 0.5), 0.5)),
        log(sigma),
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
 * import { cauchy } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(cauchy(0, 1).cdf(1)) // => 0.75
 * ```
 */
export function cauchy(loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [mu, sigma] = promoteFloat(loc, scale) as [Value, Value]
  const batchShape = batchShapeOf(mu, sigma)
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
    rsample: (key, shape) => ppf(uniformNoise(key, [...shape, ...batchShape])),
    mean: () => mul(like(mu, sigma), Number.NaN),
    variance: () => mul(like(mu, sigma), Number.NaN),
    mode: () => like(mu, sigma),
    entropy: () => add(log(sigma), Math.log(4 * Math.PI)),
  })
}

/**
 * The half-Cauchy distribution |Cauchy(0, scale)| (scipy.stats.halfcauchy), a common prior for scale parameters.
 *
 * @param scale - Scale (positive).
 * @returns The distribution; its mean and variance are infinite.
 * @example
 * ```ts
 * import { halfCauchy } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(halfCauchy(1).cdf(1)) // => 0.5
 * ```
 */
export function halfCauchy(scale: TensorLike = 1): Distribution {
  const [sigma] = promoteFloat(scale) as [Value]
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
    ppf,
    rsample: (key, shape) => ppf(uniformNoise(key, [...shape, ...batchShape])),
    mean: () => add(mul(sigma, 0), Number.POSITIVE_INFINITY),
    variance: () => add(mul(sigma, 0), Number.POSITIVE_INFINITY),
    mode: () => mul(sigma, 0),
    entropy: () => add(log(sigma), Math.log(2 * Math.PI)),
  })
}

/**
 * The Laplace (double exponential) distribution (scipy.stats.laplace).
 *
 * @param loc - Location.
 * @param scale - Scale b (positive).
 * @returns The distribution; `rsample` is the quantile of uniform noise.
 * @example
 * ```ts
 * import { laplace } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(laplace(0, 2).variance()) // => 8
 * ```
 */
export function laplace(loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [mu, b] = promoteFloat(loc, scale) as [Value, Value]
  const batchShape = batchShapeOf(mu, b)
  const z = (x: Value): Value => div(sub(x, mu), b)
  const ppf = (q: Value): Value => {
    const d = sub(q, 0.5)
    return sub(mu, mul(mul(b, sign(d)), log1p(mul(abs(d), -2))))
  }
  return defineDistribution({
    name: 'laplace',
    params: { loc: mu, scale: b },
    support: { kind: 'real' },
    batchShape,
    logProb: (x) => neg(add(log(mul(b, 2)), abs(z(x)))),
    cdf: (x) => {
      const v = z(x)
      return where(less(v, 0), mul(exp(v), 0.5), sub(1, mul(exp(neg(v)), 0.5)))
    },
    ppf,
    rsample: (key, shape) => ppf(uniformNoise(key, [...shape, ...batchShape])),
    mean: () => like(mu, b),
    variance: () => like(mul(square(b), 2), mu),
    mode: () => like(mu, b),
    entropy: () => like(add(log(mul(b, 2)), 1), mu),
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
 * import { logistic } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(logistic(0, 1).cdf(0)) // => 0.5
 * ```
 */
export function logistic(loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [mu, s] = promoteFloat(loc, scale) as [Value, Value]
  const batchShape = batchShapeOf(mu, s)
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
    ppf,
    rsample: (key, shape) => ppf(uniformNoise(key, [...shape, ...batchShape])),
    mean: () => like(mu, s),
    variance: () => like(mul(square(s), (Math.PI * Math.PI) / 3), mu),
    mode: () => like(mu, s),
    entropy: () => like(add(log(s), 2), mu),
  })
}

/**
 * The Pareto distribution with shape b and scale x_m (scipy.stats.pareto(b, scale=x_m)), on [x_m, inf).
 *
 * @param b - Shape (positive).
 * @param scale - Scale x_m (positive), the minimum value.
 * @returns The distribution; `rsample` is the quantile of uniform noise.
 * @example
 * ```ts
 * import { pareto } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(pareto(3, 1).mean()) // => 1.5
 * ```
 */
export function pareto(b: TensorLike, scale: TensorLike = 1): Distribution {
  const [alpha, xm] = promoteFloat(b, scale) as [Value, Value]
  const batchShape = batchShapeOf(alpha, xm)
  const inside = (x: Value): Value => greaterEqual(x, xm)
  const safe = (x: Value): Value => where(inside(x), x, xm)
  const ppf = (q: Value): Value => mul(xm, pow(sub(1, q), div(-1, alpha)))
  return defineDistribution({
    name: 'pareto',
    params: { b: alpha, scale: xm },
    support: { kind: 'positive' },
    batchShape,
    logProb: (x) =>
      where(
        inside(x),
        sub(add(log(alpha), mul(alpha, log(xm))), mul(add(alpha, 1), log(safe(x)))),
        Number.NEGATIVE_INFINITY,
      ),
    cdf: (x) => where(inside(x), neg(expm1(mul(alpha, log(div(xm, safe(x)))))), 0),
    ppf,
    rsample: (key, shape) => ppf(uniformNoise(key, [...shape, ...batchShape])),
    mean: () => where(less(1, alpha), div(mul(alpha, xm), sub(alpha, 1)), Number.POSITIVE_INFINITY),
    variance: () =>
      where(
        less(2, alpha),
        div(mul(square(xm), alpha), mul(square(sub(alpha, 1)), sub(alpha, 2))),
        Number.POSITIVE_INFINITY,
      ),
    mode: () => like(xm, alpha),
    entropy: () => add(add(log(div(xm, alpha)), div(1, alpha)), 1),
  })
}
