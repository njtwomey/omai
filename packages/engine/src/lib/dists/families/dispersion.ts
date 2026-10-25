/**
 * Exponential dispersion families for generalised linear models, as statsmodels.genmod.families: Gaussian, binomial,
 * Poisson, gamma, inverse Gaussian, negative binomial and Tweedie, each with its variance function, unit deviance,
 * log-likelihood and default link, plus the Tweedie compound Poisson-gamma distribution itself.
 */

import { maxOf, minOf, WhiteboxError } from '../../base/index.ts'
import { gammainc, lbinom, lgamma, xlogy } from '../../numerics/special/index.ts'
import { foldIn, gammaVariates, type Key, poisson as poissonDraw } from '../../random/index.ts'
import {
  add,
  arange,
  broadcastShapes,
  broadcastTo,
  div,
  exp,
  fromStorage,
  greater,
  less,
  lessEqual,
  log,
  log1p,
  logsumexp,
  maximum,
  mean,
  mul,
  neg,
  pow,
  primalOf,
  promoteFloat,
  reshape,
  sign,
  sqrt,
  square,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { numericPpf } from '../_numeric.ts'
import { gamma as gammaDistribution, inverseGaussian as inverseGaussianDistribution } from '../continuous/gamma.ts'
import { normal as normalDistribution } from '../continuous/normal.ts'
import { binomial as binomialDistribution } from '../discrete/binary.ts'
import { negativeBinomial as negativeBinomialDistribution, poisson as poissonDistribution } from '../discrete/counts.ts'
import { type Distribution, defineDistribution, unavailable, validateParams } from '../distribution.ts'
import { identity, inverse, inverseSquared, type Link, logit, log as logLink } from './links.ts'

const EPS = 2.220446049250313e-16

/**
 * An exponential dispersion family for generalised linear models (statsmodels.genmod.families): the variance function
 * V(mu), the unit deviance d(y, mu), the log-likelihood with dispersion (scale) phi and variance weights, starting
 * means for IRLS, and the distribution of a response with a given mean.
 *
 * @concept generalized-linear-models
 */
export interface DispersionFamily {
  readonly name: string
  readonly defaultLink: Link
  /** V(mu). */
  variance(mu: TensorLike): Value
  /** V'(mu). */
  varianceDerivative(mu: TensorLike): Value
  /** The unit deviance d(y, mu), per observation. */
  unitDeviance(y: TensorLike, mu: TensorLike): Value
  /** The deviance sum_i w_i d(y_i, mu_i). */
  deviance(y: TensorLike, mu: TensorLike, weights?: TensorLike): Value
  /** Signed deviance residuals sign(y - mu) sqrt(w d(y, mu)). */
  devianceResiduals(y: TensorLike, mu: TensorLike, weights?: TensorLike): Value
  /** The log-likelihood of each observation. */
  logLikelihoodObs(y: TensorLike, mu: TensorLike, scale?: TensorLike, weights?: TensorLike): Value
  /** The total log-likelihood. */
  logLikelihood(y: TensorLike, mu: TensorLike, scale?: TensorLike, weights?: TensorLike): Value
  /** Starting means for IRLS (statsmodels' starting_mu). */
  startingMu(y: TensorLike): Value
  /** The IRLS working weights w / (g'(mu)^2 V(mu)) for a link g. */
  irlsWeights(mu: TensorLike, link?: Link, weights?: TensorLike): Value
  /** The distribution of a response with mean mu and dispersion phi. */
  distribution(mu: TensorLike, scale?: TensorLike): Distribution
}

interface FamilySpec {
  readonly name: string
  readonly defaultLink: Link
  variance(mu: Value): Value
  varianceDerivative(mu: Value): Value
  unitDeviance(y: Value, mu: Value): Value
  logLikelihoodObs(y: Value, mu: Value, scale: Value, weights: Value): Value
  startingMu?(y: Value): Value
  distribution(mu: Value, scale: Value): Distribution
}

function defineFamily(spec: FamilySpec): DispersionFamily {
  const v = (x: TensorLike): Value => promoteFloat(x)[0] as Value
  const deviance = (y: TensorLike, mu: TensorLike, weights: TensorLike = 1): Value =>
    sum(mul(v(weights), spec.unitDeviance(v(y), v(mu))))
  const obs = (y: TensorLike, mu: TensorLike, scale: TensorLike = 1, weights: TensorLike = 1): Value =>
    spec.logLikelihoodObs(v(y), v(mu), v(scale), v(weights))
  return {
    name: spec.name,
    defaultLink: spec.defaultLink,
    variance: (mu) => spec.variance(v(mu)),
    varianceDerivative: (mu) => spec.varianceDerivative(v(mu)),
    unitDeviance: (y, mu) => spec.unitDeviance(v(y), v(mu)),
    deviance,
    devianceResiduals: (y, mu, weights = 1) => {
      const yy = v(y)
      const m = v(mu)
      return mul(sign(sub(yy, m)), sqrt(maximum(mul(v(weights), spec.unitDeviance(yy, m)), 0)))
    },
    logLikelihoodObs: obs,
    logLikelihood: (y, mu, scale = 1, weights = 1) => sum(obs(y, mu, scale, weights)),
    startingMu: (y) => {
      const yy = v(y)
      return spec.startingMu ? spec.startingMu(yy) : mul(add(yy, mean(yy)), 0.5)
    },
    irlsWeights: (mu, link = spec.defaultLink, weights = 1) => {
      const m = v(mu)
      return div(v(weights), mul(square(link.derivative(m)), spec.variance(m)))
    },
    distribution: (mu, scale = 1) => spec.distribution(v(mu), v(scale)),
  }
}

// y log(y / mu) with 0 log 0 = 0.
const yLogRatio = (y: Value, mu: Value): Value => sub(xlogy(y, y), xlogy(y, mu))

/**
 * The Gaussian family: V(mu) = 1, d = (y - mu)^2, default identity link.
 *
 * @param link - The link (default identity).
 * @returns The family.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.gaussian().deviance([1, 2], [0, 0])) // => 5
 * ```
 */
export function gaussian(link: Link = identity()): DispersionFamily {
  return defineFamily({
    name: 'gaussian',
    defaultLink: link,
    variance: (mu) => add(mul(mu, 0), 1),
    varianceDerivative: (mu) => mul(mu, 0),
    unitDeviance: (y, mu) => square(sub(y, mu)),
    logLikelihoodObs: (y, mu, scale, w) =>
      mul(add(div(mul(neg(w), square(sub(y, mu))), scale), sub(neg(log(div(scale, w))), Math.log(2 * Math.PI))), 0.5),
    distribution: (mu, scale) => normalDistribution(mu, sqrt(scale)),
  })
}

/**
 * The binomial family for proportions y in [0, 1] of `trials` trials: V(mu) = mu (1 - mu),
 * d = 2 [y log(y / mu) + (1 - y) log((1 - y) / (1 - mu))], default logit link, starting means (y + 1/2) / 2; the
 * log-likelihood is the binomial log pmf of the y n successes (statsmodels' Binomial, with weights multiplying it).
 *
 * @param link - The link (default logit).
 * @param trials - Trials per observation (default 1).
 * @returns The family; `distribution(mu)` is binomial(trials, mu) on counts.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.binomial().variance(0.5)) // => 0.25
 * ```
 */
export function binomial(link: Link = logit(), trials = 1): DispersionFamily {
  return defineFamily({
    name: 'binomial',
    defaultLink: link,
    variance: (mu) => mul(mu, sub(1, mu)),
    varianceDerivative: (mu) => sub(1, mul(mu, 2)),
    unitDeviance: (y, mu) => mul(add(yLogRatio(y, mu), yLogRatio(sub(1, y), sub(1, mu))), 2),
    logLikelihoodObs: (y, mu, _scale, w) => {
      const k = mul(y, trials)
      const ll = add(
        add(lbinom(trials, k), mul(k, log(div(add(mu, 1e-20), add(sub(1, mu), 1e-20))))),
        mul(log(add(sub(1, mu), 1e-20)), trials),
      )
      return mul(ll, w)
    },
    startingMu: (y) => mul(add(y, 0.5), 0.5),
    distribution: (mu) => binomialDistribution(trials, mu),
  })
}

/**
 * The Poisson family: V(mu) = mu, d = 2 [y log(y / mu) - (y - mu)], default log link.
 *
 * @param link - The link (default log).
 * @returns The family.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.poisson().unitDeviance(2, 2)) // => 0
 * ```
 */
export function poisson(link: Link = logLink()): DispersionFamily {
  return defineFamily({
    name: 'poisson',
    defaultLink: link,
    variance: (mu) => mu,
    varianceDerivative: (mu) => add(mul(mu, 0), 1),
    unitDeviance: (y, mu) => mul(sub(yLogRatio(y, mu), sub(y, mu)), 2),
    logLikelihoodObs: (y, mu, scale, w) => div(mul(w, sub(sub(xlogy(y, mu), mu), lgamma(add(y, 1)))), scale),
    distribution: (mu) => poissonDistribution(mu),
  })
}

/**
 * The gamma family: V(mu) = mu^2, d = 2 [(y - mu) / mu - log(y / mu)], default inverse link (statsmodels'
 * default; the log link is common in practice).
 *
 * @param link - The link (default inverse).
 * @returns The family; `distribution(mu, phi)` is the gamma with mean mu and squared coefficient of variation phi.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.gamma().variance(3)) // => 9
 * ```
 */
export function gamma(link: Link = inverse()): DispersionFamily {
  return defineFamily({
    name: 'gamma',
    defaultLink: link,
    variance: square,
    varianceDerivative: (mu) => mul(mu, 2),
    unitDeviance: (y, mu) => {
      const ratio = div(maximum(y, EPS), mu)
      return mul(sub(div(sub(y, mu), mu), log(ratio)), 2)
    },
    logLikelihoodObs: (y, mu, scale, w) => {
      const ratio = div(maximum(y, EPS), mu)
      const ws = div(w, scale)
      return sub(sub(sub(mul(ws, log(mul(ws, ratio))), mul(ws, ratio)), lgamma(ws)), log(y))
    },
    distribution: (mu, phi) => gammaDistribution(div(1, phi), div(1, mul(phi, mu))),
  })
}

/**
 * The inverse Gaussian family: V(mu) = mu^3, d = (y - mu)^2 / (y mu^2), default inverse-squared link.
 *
 * @param link - The link (default inverse squared).
 * @returns The family; `distribution(mu, phi)` is the inverse Gaussian with mean mu and shape 1 / phi.
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.inverseGaussian().variance(2)) // => 8
 * ```
 */
export function inverseGaussian(link: Link = inverseSquared()): DispersionFamily {
  return defineFamily({
    name: 'inverseGaussian',
    defaultLink: link,
    variance: (mu) => pow(mu, 3),
    varianceDerivative: (mu) => mul(square(mu), 3),
    unitDeviance: (y, mu) => div(square(sub(y, mu)), mul(y, square(mu))),
    logLikelihoodObs: (y, mu, scale, w) =>
      mul(
        add(
          div(mul(neg(w), square(sub(y, mu))), mul(mul(scale, y), square(mu))),
          sub(neg(log(div(mul(scale, pow(y, 3)), w))), Math.log(2 * Math.PI)),
        ),
        0.5,
      ),
    distribution: (mu, phi) => inverseGaussianDistribution(mu, div(1, phi)),
  })
}

/**
 * The negative binomial (NB2) family with fixed dispersion alpha: V(mu) = mu + alpha mu^2, d = 2 [y log(y / mu) -
 * (y + 1/alpha) log((1 + alpha y) / (1 + alpha mu))], default log link.
 *
 * @param alpha - The dispersion (positive).
 * @param link - The link (default log).
 * @returns The family; `distribution(mu)` is negativeBinomial(1 / alpha, 1 / (1 + alpha mu)).
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.negativeBinomial(0.5).variance(2)) // => 4
 * ```
 */
export function negativeBinomial(alpha = 1, link: Link = logLink()): DispersionFamily {
  if (!(alpha > 0)) throw new WhiteboxError('value', 'negativeBinomial family: alpha must be positive')
  const r = 1 / alpha
  return defineFamily({
    name: 'negativeBinomial',
    defaultLink: link,
    variance: (mu) => add(mu, mul(square(mu), alpha)),
    varianceDerivative: (mu) => add(mul(mu, 2 * alpha), 1),
    unitDeviance: (y, mu) =>
      mul(sub(yLogRatio(y, mu), mul(add(y, r), log(div(add(mul(y, alpha), 1), add(mul(mu, alpha), 1))))), 2),
    logLikelihoodObs: (y, mu, scale, w) => {
      const ll = add(
        sub(xlogy(y, mul(mu, alpha)), mul(add(y, r), log1p(mul(mu, alpha)))),
        sub(sub(lgamma(add(y, r)), lgamma(r)), lgamma(add(y, 1))),
      )
      return div(mul(w, ll), scale)
    },
    distribution: (mu) => negativeBinomialDistribution(r, div(1, add(mul(mu, alpha), 1))),
  })
}

/** The log density of the Tweedie compound Poisson-gamma distribution, 1 < p < 2 (Dunn and Smyth's series). */
function tweedieLogProb(y: Value, mu: Value, phi: Value, p: number): Value {
  const alpha = (2 - p) / (1 - p)
  const theta = div(pow(mu, 1 - p), 1 - p)
  const kappa = div(pow(mu, 2 - p), 2 - p)
  const base = div(sub(mul(y, theta), kappa), phi)
  const positive = greater(y, 0)
  const ys = where(positive, y, 1)
  // c(y, phi) = (1 / y) sum_{j >= 1} x^j / (j! Gamma(-alpha j)), x = ((p - 1) phi / y)^alpha / ((2 - p) phi).
  const logX = sub(mul(log(div(mul(phi, p - 1), ys)), alpha), log(mul(phi, 2 - p)))
  // The terms peak near j = y^(2-p) / ((2 - p) phi).
  const peak = toFlat(primalOf(div(pow(ys, 2 - p), mul(phi, 2 - p))))
  const lo = Math.max(1, Math.floor(minOf(peak) - 12 * Math.sqrt(minOf(peak)) - 20))
  const hi = Math.ceil(maxOf(peak) + 12 * Math.sqrt(maxOf(peak)) + 40)
  const rank = broadcastShapes(broadcastShapes(y.shape, mu.shape), phi.shape).length
  const j = reshape(arange(lo, hi + 1), [hi - lo + 1, ...new Array<number>(rank).fill(1)])
  const terms = sub(sub(mul(j, logX), lgamma(add(j, 1))), lgamma(mul(j, -alpha)))
  const logC = sub(logsumexp(terms, 0), log(ys))
  return add(base, where(positive, logC, 0))
}

/**
 * The Tweedie distribution with mean mu, dispersion phi and power 1 < p < 2: the compound Poisson-gamma (a sum of
 * N ~ Poisson(lambda) gamma variables), with a point mass exp(-lambda) at zero and a continuous density above, where
 * lambda = mu^(2-p) / (phi (2 - p)). Variance phi mu^p.
 *
 * @param mean - Mean mu (positive).
 * @param dispersion - Dispersion phi (positive).
 * @param power - Variance power p in (1, 2).
 * @returns The distribution: log density by Dunn and Smyth's series (the mass at zero for y = 0), CDF as a Poisson
 *   mixture of gamma CDFs, quantiles (0 inside the atom, numerical above), and sampling by the compound construction.
 * @throws WhiteboxError with code `value` for powers outside (1, 2).
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * // P(Y = 0) = exp(-mu^(2-p) / (phi (2-p))).
 * Math.abs(item(families.tweedieDistribution(1, 1, 1.5).prob(0)) - Math.exp(-2)) < 1e-15 // => true
 * ```
 */
export function tweedieDistribution(mean: TensorLike, dispersion: TensorLike, power: number): Distribution {
  if (!(power > 1 && power < 2)) throw new WhiteboxError('value', 'tweedie: the power must lie in (1, 2)')
  const [mu, phi] = promoteFloat(mean, dispersion) as [Value, Value]
  validateParams('tweedie', { mean: [mu, 'positive'], dispersion: [phi, 'positive'] })
  const batchShape = broadcastShapes(mu.shape, phi.shape)
  const p = power
  const lambda = div(pow(mu, 2 - p), mul(phi, 2 - p))
  const shape = (2 - p) / (p - 1)
  const scale = mul(mul(phi, p - 1), pow(mu, p - 1))
  const cdf = (y: Value): Value => {
    const lmax = maxOf(toFlat(primalOf(lambda)))
    const n = Math.ceil(lmax + 12 * Math.sqrt(lmax) + 30)
    const rank = broadcastShapes(y.shape, batchShape).length
    const k = reshape(arange(1, n + 1), [n, ...new Array<number>(rank).fill(1)])
    const logW = sub(sub(mul(k, log(lambda)), lambda), lgamma(add(k, 1)))
    const ys = maximum(y, 0)
    const continuous = sum(mul(exp(logW), gammainc(mul(k, shape), div(ys, scale))), 0)
    return where(less(y, 0), 0, add(exp(neg(lambda)), continuous))
  }
  const self: Distribution = defineDistribution({
    name: 'tweedie',
    params: { mean: mu, dispersion: phi, power: promoteFloat(p)[0] as Value },
    support: { kind: 'nonNegative' },
    batchShape,
    logProb: (y) => where(less(y, 0), Number.NEGATIVE_INFINITY, tweedieLogProb(y, mu, phi, p)),
    cdf,
    ppf: (q) =>
      where(
        lessEqual(q, exp(neg(lambda))),
        0,
        numericPpf(self, () => self, q),
      ),
    sample: (key: Key, sampleShape) => {
      const full = [...sampleShape, ...batchShape]
      const lam = toFlat(primalOf(broadcastTo(lambda, batchShape)))
      const sc = toFlat(primalOf(broadcastTo(scale, batchShape)))
      const size = full.reduce((a, b) => a * b, 1)
      const out = new Float64Array(size)
      for (let i = 0; i < size; i++) {
        const b = i % Math.max(1, lam.length)
        const kn = foldIn(key, i)
        const count = poissonDraw(foldIn(kn, 0), lam[b] ?? 0, []).data[0] ?? 0
        out[i] = count === 0 ? 0 : (gammaVariates(foldIn(kn, 1), [count * shape])[0] ?? 0) * (sc[b] ?? 1)
      }
      return fromStorage(out, full, 'float64')
    },
    mean: () => broadcastTo(mu, batchShape),
    variance: () => broadcastTo(mul(phi, pow(mu, p)), batchShape),
    mode: () => unavailable('tweedie', 'mode (a mixed distribution)'),
    entropy: () => unavailable('tweedie', 'entropy (a mixed distribution has none)'),
  })
  return self
}

/**
 * The Tweedie family with variance power p: V(mu) = mu^p, with unit deviance
 * 2 [y^(2-p) / ((1-p)(2-p)) - y mu^(1-p) / (1-p) + mu^(2-p) / (2-p)] (Poisson at p = 1, gamma at p = 2), default log
 * link; the log-likelihood is exact for 1 < p < 2 (Dunn and Smyth's series, as statsmodels with eql = false).
 *
 * @param power - The variance power p (1 <= p <= 2).
 * @param link - The link (default log).
 * @returns The family; `distribution(mu, phi)` is {@link tweedieDistribution} (Poisson at p = 1, gamma at p = 2).
 * @example
 * ```ts
 * import { families } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(families.tweedie(1.5).variance(4)) // => 8
 * ```
 */
export function tweedie(power: number, link: Link = logLink()): DispersionFamily {
  if (!(power >= 1 && power <= 2)) throw new WhiteboxError('value', 'tweedie family: the power must lie in [1, 2]')
  if (power === 1) return { ...poisson(link), name: 'tweedie' }
  if (power === 2) return { ...gamma(link), name: 'tweedie' }
  const p = power
  return defineFamily({
    name: 'tweedie',
    defaultLink: link,
    variance: (mu) => pow(mu, p),
    varianceDerivative: (mu) => mul(pow(mu, p - 1), p),
    unitDeviance: (y, mu) =>
      mul(
        add(sub(div(pow(y, 2 - p), (1 - p) * (2 - p)), div(mul(y, pow(mu, 1 - p)), 1 - p)), div(pow(mu, 2 - p), 2 - p)),
        2,
      ),
    logLikelihoodObs: (y, mu, scale, w) => tweedieLogProb(y, mu, div(scale, w), p),
    distribution: (mu, phi) => tweedieDistribution(mu, phi, p),
  })
}
