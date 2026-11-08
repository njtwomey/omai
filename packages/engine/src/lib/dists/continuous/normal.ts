/**
 * The normal distribution and its direct relatives: normal, half-normal, log-normal, truncated normal and Azzalini's
 * skew-normal, each with densities, CDFs, quantiles, moments, entropy and reparameterised sampling, as scipy.stats and
 * torch.distributions.
 */

import { LOG_SQRT_2PI } from '../../base/index.ts'
import {
  erf,
  erfc,
  erfinv,
  logIntervalProbability,
  logNdtr,
  ndtr,
  ndtri,
  normalLogPdf,
  normalPdf,
  owensT,
} from '../../numerics/special/index.ts'
import { foldIn, type Key, normal as normalNoise } from '../../random/index.ts'
import {
  abs,
  add,
  clip,
  div,
  exp,
  expm1,
  greater,
  greaterEqual,
  less,
  log,
  logicalAnd,
  logicalOr,
  mul,
  neg,
  promoteFloat,
  sqrt,
  square,
  sub,
  type Tensor,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { concreteTwin, numericEntropy, numericMode, numericPpf, tailProbability } from '../_numeric.ts'
import { uniformOpen } from '../_sampling.ts'
import { validateOrdered } from '../_validate.ts'
import { batchShapeOf, broadcastBatch, type Distribution, defineDistribution, validateParams } from '../distribution.ts'

const noise = (key: Key, shape: readonly number[], batch: readonly number[]): Value =>
  normalNoise(key, [...shape, ...batch])

/**
 * The normal distribution N(loc, scale^2) (scipy.stats.norm, torch.distributions.Normal).
 *
 * @param loc - Mean.
 * @param scale - Standard deviation (positive).
 * @returns The distribution; `rsample` is loc + scale eps.
 * @throws OmaiError with code `value` when the scale is not positive.
 * @concept distributions
 * @index distribution Normal distribution, scipy.stats.norm and torch Normal in one.
 * @example
 * ```ts
 * import { normal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * item(normal(0, 1).cdf(0)) // => 0.5
 * ```
 */
export function normal(loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [mu, sigma] = promoteFloat(loc, scale) as [Value, Value]
  validateParams('normal', { loc: [mu, 'real'], scale: [sigma, 'positive'] })
  const z = (x: Value): Value => div(sub(x, mu), sigma)
  const batchShape = batchShapeOf(mu, sigma)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
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
    mean: () => b(mu),
    variance: () => b(square(sigma)),
    stddev: () => b(sigma),
    mode: () => b(mu),
    entropy: () => b(add(log(sigma), 0.5 + LOG_SQRT_2PI)),
  })
}

/**
 * The half-normal distribution |N(0, scale^2)| (scipy.stats.halfnorm).
 *
 * @param scale - Scale (positive).
 * @returns The distribution; `rsample` is |scale eps|.
 * @example
 * ```ts
 * import { halfNormal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * item(halfNormal(1).cdf(0)) // => 0
 * ```
 */
export function halfNormal(scale: TensorLike = 1): Distribution {
  const [sigma] = promoteFloat(scale) as [Value]
  validateParams('halfNormal', { scale: [sigma, 'positive'] })
  const batchShape = batchShapeOf(sigma)
  const inside = (x: Value): Value => greaterEqual(x, 0)
  const u = (x: Value): Value => div(where(inside(x), x, 0), mul(sigma, Math.SQRT2))
  return defineDistribution({
    name: 'halfNormal',
    params: { scale: sigma },
    support: { kind: 'nonNegative' },
    batchShape,
    logProb: (x) =>
      where(inside(x), add(sub(normalLogPdf(div(x, sigma)), log(sigma)), Math.LN2), Number.NEGATIVE_INFINITY),
    cdf: (x) => where(inside(x), erf(u(x)), 0),
    sf: (x) => where(inside(x), erfc(u(x)), 1),
    logSf: (x) => where(inside(x), add(logNdtr(neg(div(where(inside(x), x, 0), sigma))), Math.LN2), 0),
    ppf: (q) => mul(mul(sigma, Math.SQRT2), erfinv(q)),
    isf: (q) => mul(sigma, neg(ndtri(mul(q, 0.5)))),
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
 * import { logNormal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * item(logNormal(0, 1).cdf(1)) // => 0.5
 * ```
 */
export function logNormal(mu: TensorLike = 0, sigma: TensorLike = 1): Distribution {
  const [m, s] = promoteFloat(mu, sigma) as [Value, Value]
  validateParams('logNormal', { mu: [m, 'real'], sigma: [s, 'positive'] })
  const batchShape = batchShapeOf(m, s)
  const positive = (x: Value): Value => greaterEqual(x, Number.MIN_VALUE)
  const z = (x: Value): Value => div(sub(log(where(positive(x), x, 1)), m), s)
  return defineDistribution({
    name: 'logNormal',
    params: { mu: m, sigma: s },
    support: { kind: 'positive' },
    batchShape,
    logProb: (x) => where(positive(x), sub(sub(normalLogPdf(z(x)), log(s)), log(x)), Number.NEGATIVE_INFINITY),
    cdf: (x) => where(positive(x), ndtr(z(x)), 0),
    logCdf: (x) => where(positive(x), logNdtr(z(x)), Number.NEGATIVE_INFINITY),
    sf: (x) => where(positive(x), ndtr(neg(z(x))), 1),
    logSf: (x) => where(positive(x), logNdtr(neg(z(x))), 0),
    ppf: (q) => exp(add(m, mul(s, ndtri(q)))),
    isf: (q) => exp(sub(m, mul(s, ndtri(q)))),
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
 * import { truncatedNormal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(truncatedNormal(0, 1, -1, 1).cdf(0)) - 0.5) < 1e-14 // => true
 * ```
 */
export function truncatedNormal(loc: TensorLike, scale: TensorLike, low: TensorLike, high: TensorLike): Distribution {
  const [mu, sigma, lo, hi] = promoteFloat(loc, scale, low, high) as [Value, Value, Value, Value]
  validateParams('truncatedNormal', { loc: [mu, 'real'], scale: [sigma, 'positive'] })
  validateOrdered('truncatedNormal', 'low', lo, 'high', hi)
  const batchShape = batchShapeOf(mu, sigma, lo, hi)
  const a = div(sub(lo, mu), sigma)
  const b = div(sub(hi, mu), sigma)
  const logZ = logIntervalProbability(a, b)
  const z = (x: Value): Value => div(sub(x, mu), sigma)
  const outside = (x: Value): Value => logicalOr(less(x, lo), less(hi, x))
  const ppf = (q: Value): Value => {
    // Phi^{-1}(Phi(a) + q (Phi(b) - Phi(a))), through the upper tail when a > 0 for accuracy.
    const pa = ndtr(a)
    return add(mu, mul(sigma, ndtri(add(pa, mul(q, sub(ndtr(b), pa))))))
  }
  const pdfA = normalPdf(a)
  const pdfB = normalPdf(b)
  const zMass = exp(logZ)
  // a phi(a) and b phi(b), zero at infinite bounds.
  const aPhi = where(logicalOr(less(a, -1e300), less(1e300, a)), 0, mul(a, pdfA))
  const bPhi = where(logicalOr(less(b, -1e300), less(1e300, b)), 0, mul(b, pdfB))
  const logMass = (x: Value): Value => logIntervalProbability(a, clip(z(x), a, b))
  const logUpper = (x: Value): Value => logIntervalProbability(clip(z(x), a, b), b)
  return defineDistribution({
    name: 'truncatedNormal',
    params: { loc: mu, scale: sigma, low: lo, high: hi },
    support: { kind: 'interval', low: lo, high: hi },
    batchShape,
    logProb: (x) => where(outside(x), Number.NEGATIVE_INFINITY, sub(sub(normalLogPdf(z(x)), log(sigma)), logZ)),
    cdf: (x) => clip(exp(sub(logMass(x), logZ)), 0, 1),
    logCdf: (x) => where(less(x, lo), Number.NEGATIVE_INFINITY, sub(logMass(x), logZ)),
    sf: (x) => clip(exp(sub(logUpper(x), logZ)), 0, 1),
    logSf: (x) => where(less(hi, x), Number.NEGATIVE_INFINITY, sub(logUpper(x), logZ)),
    ppf,
    rsample: (key, shape) => ppf(uniformOpen(key, [...shape, ...batchShape])),
    mean: () => add(mu, mul(sigma, div(sub(pdfA, pdfB), zMass))),
    variance: () => {
      const r = div(sub(pdfA, pdfB), zMass)
      return mul(square(sigma), sub(add(1, div(sub(aPhi, bPhi), zMass)), square(r)))
    },
    mode: () => clip(broadcastBatch(mu, batchShape), lo, hi),
    entropy: () => add(add(add(log(sigma), logZ), 0.5 + LOG_SQRT_2PI), div(sub(aPhi, bPhi), mul(zMass, 2))),
  })
}

/**
 * The skew-normal distribution of Azzalini (scipy.stats.skewnorm(alpha, loc, scale)): density
 * 2 phi(z) Phi(alpha z) / scale with z = (x - loc) / scale. `alpha = 0` is the normal; as alpha grows it approaches a
 * half-normal.
 *
 * @param loc - Location.
 * @param scale - Scale (positive).
 * @param alpha - Shape (skewness); positive skews right.
 * @returns The distribution. The CDF is Phi(z) - 2 T(z, alpha) with Owen's T; `rsample` is the pathwise
 *   loc + scale (delta |u| + sqrt(1 - delta^2) v), delta = alpha / sqrt(1 + alpha^2); quantiles, the mode and the
 *   entropy are numerical (as in scipy) and differentiable.
 * @example
 * ```ts
 * import { skewNormal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * item(skewNormal(0, 1, 0).cdf(0)) // => 0.5
 * Math.abs(item(skewNormal(0, 1, 3).mean()) - (3 / Math.sqrt(10)) * Math.sqrt(2 / Math.PI)) < 1e-15 // => true
 * ```
 */
export function skewNormal(loc: TensorLike = 0, scale: TensorLike = 1, alpha: TensorLike = 0): Distribution {
  const [mu, sigma, a] = promoteFloat(loc, scale, alpha) as [Value, Value, Value]
  validateParams('skewNormal', { loc: [mu, 'real'], scale: [sigma, 'positive'], alpha: [a, 'finite'] })
  const batchShape = batchShapeOf(mu, sigma, a)
  const z = (x: Value): Value => div(sub(x, mu), sigma)
  const delta = div(a, sqrt(add(square(a), 1)))
  const meanZ = mul(delta, Math.sqrt(2 / Math.PI))
  const mean = add(mu, mul(sigma, meanZ))
  const variance = mul(square(sigma), sub(1, square(meanZ)))
  const twin = concreteTwin([mu, sigma, a], (m, s, al) => skewNormal(m as Tensor, s as Tensor, al as Tensor))
  // The density decays at rate about (1 + alpha^2) |z| / scale in the thin tail.
  const tailScale = (v: Value): Value => div(sigma, add(mul(add(square(a), 1), abs(v)), 1))
  const self: Distribution = defineDistribution({
    name: 'skewNormal',
    params: { loc: mu, scale: sigma, alpha: a },
    support: { kind: 'real' },
    batchShape,
    logProb: (x) => {
      const v = z(x)
      return add(sub(add(normalLogPdf(v), logNdtr(mul(a, v))), log(sigma)), Math.LN2)
    },
    // Phi(z) - 2 T(z, alpha) cancels in the tail that the skew thins (as in scipy, which integrates there too).
    cdf: (x) => {
      const v = z(x)
      const closed = clip(sub(ndtr(v), mul(owensT(v, a), 2)), 0, 1)
      const thin = logicalAnd(less(closed, 1e-6), greater(a, 0))
      return where(thin, tailProbability(self, x, 'lower', tailScale(v)), closed)
    },
    sf: (x) => {
      const v = z(x)
      const closed = clip(add(ndtr(neg(v)), mul(owensT(v, a), 2)), 0, 1)
      const thin = logicalAnd(less(closed, 1e-6), less(a, 0))
      return where(thin, tailProbability(self, x, 'upper', tailScale(v)), closed)
    },
    ppf: (q) => numericPpf(self, twin, q),
    rsample: (key, shape) => {
      const full = [...shape, ...batchShape]
      const u = abs(normalNoise(foldIn(key, 0), full))
      const v = normalNoise(foldIn(key, 1), full)
      return add(mu, mul(sigma, add(mul(delta, u), mul(sqrt(sub(1, square(delta))), v))))
    },
    mean: () => broadcastBatch(mean, batchShape),
    variance: () => broadcastBatch(variance, batchShape),
    mode: () => numericMode(self, twin, sub(mu, mul(sigma, 1.5)), add(mu, mul(sigma, 1.5))),
    entropy: () => numericEntropy(self, { kind: 'real', center: mean, scale: sqrt(variance) }),
  })
  return self
}
