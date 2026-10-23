import {
  erf,
  erfinv,
  logIntervalProbability,
  logNdtr,
  ndtr,
  ndtri,
  normalLogPdf,
  normalPdf,
} from '../../numerics/special/index.ts'
import { type Key, normal as normalNoise, uniform as uniformNoise } from '../../random/index.ts'
import {
  abs,
  add,
  clip,
  div,
  exp,
  expm1,
  greaterEqual,
  less,
  log,
  logicalOr,
  mul,
  neg,
  promoteFloat,
  square,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { batchShapeOf, type Distribution, defineDistribution } from '../distribution.ts'

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)
const noise = (key: Key, shape: readonly number[], batch: readonly number[]): Value =>
  normalNoise(key, [...shape, ...batch])

/**
 * The normal distribution N(loc, scale^2) (scipy.stats.norm, torch.distributions.Normal).
 *
 * @param loc - Mean.
 * @param scale - Standard deviation (positive).
 * @returns The distribution; `rsample` is loc + scale eps.
 * @concept distributions
 * @example
 * ```ts
 * import { normal } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(normal(0, 1).cdf(0)) // => 0.5
 * ```
 */
export function normal(loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [mu, sigma] = promoteFloat(loc, scale) as [Value, Value]
  const z = (x: Value): Value => div(sub(x, mu), sigma)
  const batchShape = batchShapeOf(mu, sigma)
  return defineDistribution({
    name: 'normal',
    params: { loc: mu, scale: sigma },
    support: { kind: 'real' },
    batchShape,
    logProb: (x) => sub(normalLogPdf(z(x)), log(sigma)),
    cdf: (x) => ndtr(z(x)),
    logCdf: (x) => logNdtr(z(x)),
    sf: (x) => ndtr(neg(z(x))),
    logSf: (x) => logNdtr(neg(z(x))),
    ppf: (q) => add(mu, mul(sigma, ndtri(q))),
    isf: (q) => sub(mu, mul(sigma, ndtri(q))),
    rsample: (key, shape) => add(mu, mul(sigma, noise(key, shape, batchShape))),
    mean: () => add(mu, mul(sigma, 0)),
    variance: () => add(square(sigma), mul(mu, 0)),
    mode: () => add(mu, mul(sigma, 0)),
    entropy: () => add(add(log(sigma), 0.5 + LOG_SQRT_2PI), mul(mu, 0)),
  })
}

/**
 * The half-normal distribution |N(0, scale^2)| (scipy.stats.halfnorm).
 *
 * @param scale - Scale (positive).
 * @returns The distribution; `rsample` is |scale eps|.
 * @example
 * ```ts
 * import { halfNormal } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(halfNormal(1).cdf(0)) // => 0
 * ```
 */
export function halfNormal(scale: TensorLike = 1): Distribution {
  const [sigma] = promoteFloat(scale) as [Value]
  const batchShape = batchShapeOf(sigma)
  const inside = (x: Value): Value => greaterEqual(x, 0)
  return defineDistribution({
    name: 'halfNormal',
    params: { scale: sigma },
    support: { kind: 'nonNegative' },
    batchShape,
    logProb: (x) =>
      where(inside(x), add(sub(normalLogPdf(div(x, sigma)), log(sigma)), Math.LN2), Number.NEGATIVE_INFINITY),
    cdf: (x) => where(inside(x), erf(div(x, mul(sigma, Math.SQRT2))), 0),
    ppf: (q) => mul(mul(sigma, Math.SQRT2), erfinv(q)),
    rsample: (key, shape) => abs(mul(sigma, noise(key, shape, batchShape))),
    mean: () => mul(sigma, Math.sqrt(2 / Math.PI)),
    variance: () => mul(square(sigma), 1 - 2 / Math.PI),
    mode: () => mul(sigma, 0),
    entropy: () => add(mul(log(mul(square(sigma), Math.PI / 2)), 0.5), 0.5),
  })
}

/**
 * The log-normal distribution of exp(N(mu, sigma^2)) (scipy.stats.lognorm with s = sigma, scale = exp(mu)).
 *
 * @param mu - Mean of the logarithm.
 * @param sigma - Standard deviation of the logarithm (positive).
 * @returns The distribution; `rsample` is exp(mu + sigma eps).
 * @example
 * ```ts
 * import { logNormal } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(logNormal(0, 1).cdf(1)) // => 0.5
 * ```
 */
export function logNormal(mu: TensorLike = 0, sigma: TensorLike = 1): Distribution {
  const [m, s] = promoteFloat(mu, sigma) as [Value, Value]
  const batchShape = batchShapeOf(m, s)
  const z = (x: Value): Value => div(sub(log(x), m), s)
  return defineDistribution({
    name: 'logNormal',
    params: { mu: m, sigma: s },
    support: { kind: 'positive' },
    batchShape,
    logProb: (x) => sub(sub(normalLogPdf(z(x)), log(s)), log(x)),
    cdf: (x) => ndtr(z(x)),
    logCdf: (x) => logNdtr(z(x)),
    sf: (x) => ndtr(neg(z(x))),
    logSf: (x) => logNdtr(neg(z(x))),
    ppf: (q) => exp(add(m, mul(s, ndtri(q)))),
    rsample: (key, shape) => exp(add(m, mul(s, noise(key, shape, batchShape)))),
    mean: () => exp(add(m, mul(square(s), 0.5))),
    variance: () => mul(expm1(square(s)), exp(add(mul(m, 2), square(s)))),
    mode: () => exp(sub(m, square(s))),
    entropy: () => add(add(m, log(s)), 0.5 + LOG_SQRT_2PI),
  })
}

/**
 * The normal distribution N(loc, scale^2) truncated to [low, high] (scipy.stats.truncnorm with standardised bounds).
 *
 * @param loc - Location of the parent normal.
 * @param scale - Scale of the parent normal.
 * @param low - Lower bound (may be -Infinity).
 * @param high - Upper bound (may be Infinity).
 * @returns The distribution; `rsample` is the inverse CDF of uniform noise, differentiable in every parameter.
 * @example
 * ```ts
 * import { truncatedNormal } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(truncatedNormal(0, 1, -1, 1).cdf(0)) // => 0.5
 * ```
 */
export function truncatedNormal(loc: TensorLike, scale: TensorLike, low: TensorLike, high: TensorLike): Distribution {
  const [mu, sigma, lo, hi] = promoteFloat(loc, scale, low, high) as [Value, Value, Value, Value]
  const batchShape = batchShapeOf(mu, sigma, lo, hi)
  const a = div(sub(lo, mu), sigma)
  const b = div(sub(hi, mu), sigma)
  const logZ = logIntervalProbability(a, b)
  const z = (x: Value): Value => div(sub(x, mu), sigma)
  const outside = (x: Value): Value => logicalOr(less(x, lo), less(hi, x))
  const ppf = (q: Value): Value => {
    // Phi^{-1}(Phi(a) + q (Phi(b) - Phi(a))).
    const pa = ndtr(a)
    return add(mu, mul(sigma, ndtri(add(pa, mul(q, sub(ndtr(b), pa))))))
  }
  const pdfA = normalPdf(a)
  const pdfB = normalPdf(b)
  const zMass = exp(logZ)
  // a phi(a) and b phi(b), zero at infinite bounds.
  const aPhi = where(logicalOr(less(a, -1e300), less(1e300, a)), 0, mul(a, pdfA))
  const bPhi = where(logicalOr(less(b, -1e300), less(1e300, b)), 0, mul(b, pdfB))
  return defineDistribution({
    name: 'truncatedNormal',
    params: { loc: mu, scale: sigma, low: lo, high: hi },
    support: { kind: 'interval', low: lo, high: hi },
    batchShape,
    logProb: (x) => where(outside(x), Number.NEGATIVE_INFINITY, sub(sub(normalLogPdf(z(x)), log(sigma)), logZ)),
    cdf: (x) => clip(exp(sub(logIntervalProbability(a, clip(z(x), a, b)), logZ)), 0, 1),
    ppf,
    rsample: (key, shape) => ppf(uniformNoise(key, [...shape, ...batchShape])),
    mean: () => add(mu, mul(sigma, div(sub(pdfA, pdfB), zMass))),
    variance: () => {
      const r = div(sub(pdfA, pdfB), zMass)
      return mul(square(sigma), sub(add(1, div(sub(aPhi, bPhi), zMass)), square(r)))
    },
    mode: () => clip(mu, lo, hi),
    entropy: () => add(add(add(log(sigma), logZ), 0.5 + LOG_SQRT_2PI), div(sub(aPhi, bPhi), mul(zMass, 2))),
  })
}
