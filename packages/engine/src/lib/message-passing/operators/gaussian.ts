/**
 * The Gaussian-family operators: a fixed prior message (any family), a Gaussian with a fixed precision, a Gaussian
 * with a gamma precision, a Gaussian observation with a normal-gamma mean and precision, a product of Gaussians, and
 * the equality of two variables. Moment matching is exact for the first two and for the equality; the product's
 * messages to its factors are numerical (one-dimensional quadrature, the rest in closed form).
 */

import { OmaiError } from '../../base/index.ts'
import { gaussianGrid, tiltedGrid, tiltedLogGrid } from '../_quadrature.ts'
import { digammaOf, trigammaOf } from '../_scalar.ts'
import { gamma, gammaShapeRate } from '../families.ts'
import type { Arg, FactorOperator, FamilyKind, OperatorContext, OperatorResult, Params } from '../types.ts'
import {
  allProper,
  etaOf,
  gaussianMessage,
  gaussianOf,
  gaussianSiteMessage,
  ImproperMessageError,
  logNormal,
  massOf,
  numberParam,
  rawSum,
  siteMessage,
  vectorParam,
} from './args.ts'

const ALL: readonly FamilyKind[] = [
  'gaussian',
  'gaussian-mv',
  'gamma',
  'beta',
  'bernoulli',
  'discrete',
  'dirichlet',
  'normal-gamma',
]

const arg = (c: OperatorContext, i: number): Arg => c.args[i] as Arg

/**
 * A fixed message on one variable: the factor that is a known density, the prior. Its parameter `eta` is the natural
 * parameters (in the family's layout) of a normalised density, so a Gaussian prior N(m, v) is `{ eta: [m / v, 1 / v] }`
 * and the factor's mass under the incoming message is the ratio of two partitions. Exact under every rule.
 *
 * @concept factor-operators
 * @index structure The prior: a fixed message on a variable, in any family.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, prior } from 'omai-engine/message-passing'
 * const eta = gaussianEta(1, 4)
 * prior.ep?.({ mode: 'ep', to: 0, args: [messageArg(gaussian, gaussian.flat)], params: { eta }, power: 1 }).message // => [0.25, 0.25]
 * ```
 */
export const prior: FactorOperator = {
  name: 'prior',
  description: 'a normalised density with the given natural parameters on one variable',
  ports: [{ name: 'x', accepts: ALL }],
  support: { ep: 'exact', vmp: 'exact', bp: 'exact' },
  check: (params, families) =>
    vectorParam(params, 'eta').length === families[0]?.size
      ? undefined
      : 'prior: eta must have the length of the family',
  ep: (c) => ({ message: vectorParam(c.params, 'eta') }),
  vmp: (c) => ({ message: vectorParam(c.params, 'eta') }),
  bp: (c) => ({ message: vectorParam(c.params, 'eta') }),
  logEvidence: (c) => {
    const a = arg(c, 0)
    const p = vectorParam(c.params, 'eta')
    return a.family.logPartition(rawSum(etaOf(a), p)) - a.family.logPartition(p)
  },
  expectedLog: (c) => {
    const a = arg(c, 0)
    const p = vectorParam(c.params, 'eta')
    return a.family.pair(p, a.family.moments(etaOf(a))) - a.family.logPartition(p)
  },
}

const gaussianPorts = [
  { name: 'x', accepts: ['gaussian'] as const, constant: true },
  { name: 'mean', accepts: ['gaussian'] as const, constant: true },
]

/** E[(x - mean)^2] for two independent arguments under their moments. */
function expectedSquare(x: Arg, mean: Arg): number {
  const a = gaussianOf(x)
  const b = gaussianOf(mean)
  return a.variance + b.variance + (a.mean - b.mean) ** 2
}

/**
 * x ~ N(mean, 1 / precision) with the precision a parameter: the Gaussian likelihood and the Gaussian prior of a
 * mean. EP adds the variances (a message to x is N(E[mean], Var[mean] + 1 / precision)); VMP uses the expected log
 * density, whose message to x is N(E[mean], 1 / precision). Both are closed forms.
 *
 * @concept factor-operators
 * @index structure A Gaussian with a fixed precision: likelihood or prior of a mean.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, gaussianOperator, messageArg } from 'omai-engine/message-passing'
 * const c = { mode: 'ep' as const, to: 0, power: 1, params: { precision: 4 }, args: [messageArg(gaussian, gaussian.flat), messageArg(gaussian, gaussianEta(2, 1))] }
 * gaussianOperator.ep?.(c).message // => [1.6, 0.8]
 * ```
 */
export const gaussianOperator: FactorOperator = {
  name: 'gaussian',
  description: 'x ~ N(mean, 1 / precision) with a fixed precision',
  ports: gaussianPorts,
  support: { ep: 'exact', vmp: 'exact' },
  ep: (c): OperatorResult => {
    const o = gaussianOf(arg(c, 1 - c.to))
    return { message: gaussianMessage(o.mean, o.variance + 1 / numberParam(c.params, 'precision')) }
  },
  vmp: (c): OperatorResult => {
    const tau = numberParam(c.params, 'precision')
    const o = gaussianOf(arg(c, 1 - c.to))
    return { message: [tau * o.mean, tau] }
  },
  logEvidence: (c) => {
    if (!allProper(c.args)) return Number.NaN
    const x = gaussianOf(arg(c, 0))
    const m = gaussianOf(arg(c, 1))
    return (
      massOf(arg(c, 0)) +
      massOf(arg(c, 1)) +
      logNormal(x.mean, m.mean, x.variance + m.variance + 1 / numberParam(c.params, 'precision'))
    )
  },
  expectedLog: (c) => {
    const tau = numberParam(c.params, 'precision')
    return 0.5 * Math.log(tau / (2 * Math.PI)) - 0.5 * tau * expectedSquare(arg(c, 0), arg(c, 1))
  },
}

const TILT_POINTS = 801
const TILT_SPAN = 20
/** The variance a flat cavity is read as where a tilt needs a density: wide enough that the other factors decide. */
const WIDE = 1e6

/**
 * The moments of a Gaussian cavity where a flat one (no information yet) is read as a very wide normal, so a factor on an
 * unobserved value can send its first message (a latent value with a learned spread and a likelihood that needs a proper
 * cavity, as the logistic factor does, would otherwise never start). A cavity with a negative precision is still improper.
 */
function cavityOf(a: Arg): { readonly mean: number; readonly variance: number } {
  if (a.value === undefined) {
    const eta = etaOf(a)
    if (Math.abs(eta[1] ?? 0) < 1e-12 && Math.abs(eta[0] ?? 0) < 1e-12) return { mean: 0, variance: WIDE }
  }
  return gaussianOf(a)
}

/**
 * The moments of the tilted distribution of the factor `x ~ N(u, 1 / lambda)` with the two Gaussian cavities `x ~ N(mx, vx)`
 * and `u ~ N(mu, vu)` (either may be observed: a variance of zero) and the gamma cavity of `lambda`. The factor depends on
 * `x - u` only, so given `lambda` it is a Gaussian observation of that difference and each of `x` and `u` has the Kalman
 * update; integrating both out leaves `Gamma(lambda) N(0; mx - mu, vx + vu + 1 / lambda)`, a Student-t shaped function of
 * the precision whose moments (`E log lambda`, `E lambda`) are computed on a grid in `log lambda`, and the Gaussians' moments
 * are the mixtures of the updates over it. This is what expectation propagation needs of a Gaussian with an unknown
 * noise level or population spread. With a `scale` and an `offset` the precision of the Gaussian is `scale lambda + offset`
 * (a coefficient of a penalised spline: `lambda d_j + n_j`), the gamma variable `lambda` being the shared one.
 */
function precisionTilt(
  x: { readonly mean: number; readonly variance: number },
  u: { readonly mean: number; readonly variance: number },
  lambda: readonly number[],
  scale = 1,
  offset = 0,
): {
  readonly x: { readonly mean: number; readonly variance: number }
  readonly u: { readonly mean: number; readonly variance: number }
  readonly logLambda: number
  readonly lambda: number
} {
  const { shape, rate } = gammaShapeRate(lambda)
  if (!(shape > 0 && rate > 0 && Number.isFinite(shape) && Number.isFinite(rate)))
    throw new ImproperMessageError('a gamma message has no density (its shape and rate are not positive)')
  const gap = x.mean - u.mean
  const joint = x.variance + u.variance
  const centre = digammaOf(shape) - Math.log(rate)
  const half = TILT_SPAN * Math.sqrt(trigammaOf(shape))
  const lo = Math.max(centre - half, -60)
  const hi = Math.min(centre + half, 60)
  const step = (hi - lo) / (TILT_POINTS - 1)
  const logw = new Float64Array(TILT_POINTS)
  let top = Number.NEGATIVE_INFINITY
  for (let i = 0; i < TILT_POINTS; i++) {
    const t = lo + i * step
    // The gamma density in t = log lambda is exp(shape t - rate e^t); the difference integrated out is N(0; gap, joint + e^-t).
    const lw = shape * t - rate * Math.exp(t) + logNormal(0, gap, joint + 1 / (scale * Math.exp(t) + offset))
    logw[i] = lw
    if (lw > top) top = lw
  }
  let z = 0
  let eLog = 0
  let eLam = 0
  let eX = 0
  let eX2 = 0
  let eU = 0
  let eU2 = 0
  for (let i = 0; i < TILT_POINTS; i++) {
    const t = lo + i * step
    const w = Math.exp((logw[i] ?? 0) - top)
    const s = joint + 1 / (scale * Math.exp(t) + offset)
    const mx = x.mean - (x.variance * gap) / s
    const vx = x.variance - (x.variance * x.variance) / s
    const mu = u.mean + (u.variance * gap) / s
    const vu = u.variance - (u.variance * u.variance) / s
    z += w
    eLog += w * t
    eLam += w * Math.exp(t)
    eX += w * mx
    eX2 += w * (vx + mx * mx)
    eU += w * mu
    eU2 += w * (vu + mu * mu)
  }
  const meanX = eX / z
  const meanU = eU / z
  return {
    x: { mean: meanX, variance: eX2 / z - meanX * meanX },
    u: { mean: meanU, variance: eU2 / z - meanU * meanU },
    logLambda: eLog / z,
    lambda: eLam / z,
  }
}

/**
 * x ~ N(mean, 1 / lambda) with a gamma-distributed precision lambda: the factor of a Gaussian with unknown noise.
 * Variational message passing (Winn and Bishop) uses the expected log density, closed forms that make the model
 * conjugate-exponential. Expectation propagation matches the moments of the tilted distribution (the cavities of `x`, of the
 * mean and of the precision times the factor), computed by quadrature in `log lambda`: `x` and the mean each get a Gaussian
 * site, the precision a gamma site (the projection of `E log lambda` and `E lambda`), for an observed or a latent `x`. This
 * is what the linear regression with an unknown noise level needs, and a population of latent values with an unknown spread
 * (the abilities of an item response model). A Gaussian port whose cavity is still flat is read as very wide, so the first
 * message to a latent value is the prior predictive and a logistic likelihood on it can start. The optional parameters
 * `scale` and `offset` make the precision `scale lambda + offset` (a coefficient of a penalised spline has the precision
 * `lambda d_j` of one shared smoothing parameter `lambda`; the variational rule has `scale` only).
 *
 * @concept factor-operators
 * @index structure A Gaussian with a gamma precision (variational and expectation propagation, observed or latent).
 * @example
 * ```ts
 * import { gamma, gammaEta, gaussian, gaussianEta, gaussianGammaPrecision, messageArg, valueArg } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussian, 1), messageArg(gaussian, gaussianEta(0, 1)), messageArg(gamma, gammaEta(2, 1))]
 * gaussianGammaPrecision.ep?.({ mode: 'ep', to: 1, args, params: {}, power: 1 }).message.map((v) => v.toFixed(2)) // => ['1.48', '1.41']
 * ```
 */
export const gaussianGammaPrecision: FactorOperator = {
  name: 'gaussian-gamma-precision',
  description: 'x ~ N(mean, 1 / (scale lambda + offset)) with lambda a gamma-distributed precision',
  ports: [...gaussianPorts, { name: 'precision', accepts: ['gamma'] }],
  support: { ep: 'numeric', vmp: 'exact' },
  // The precision's own cavity must be a gamma density; a Gaussian port whose cavity is still flat is read as very wide.
  readsTarget: (to) => to === 2,
  ep: (c): OperatorResult => {
    const { scale, offset } = precisionParams(c.params)
    const tilt = precisionTilt(cavityOf(arg(c, 0)), cavityOf(arg(c, 1)), etaOf(arg(c, 2)), scale, offset)
    if (c.to === 0) return { message: gaussianSiteMessage(etaOf(arg(c, 0)), tilt.x.mean, tilt.x.variance) }
    if (c.to === 1) return { message: gaussianSiteMessage(etaOf(arg(c, 1)), tilt.u.mean, tilt.u.variance) }
    return { message: siteMessage(gamma, etaOf(arg(c, 2)), [tilt.logLambda, tilt.lambda]) }
  },
  vmp: (c): OperatorResult => {
    const { scale, offset } = precisionParams(c.params)
    if (offset !== 0)
      throw new OmaiError('not-implemented', 'gaussian-gamma-precision: a precision offset has no variational rule')
    const lambda = arg(c, 2)
    const mo = lambda.family.moments(etaOf(lambda))
    const meanLambda = scale * (mo[1] ?? 0)
    if (c.to === 2) return { message: [0.5, -0.5 * scale * expectedSquare(arg(c, 0), arg(c, 1))] }
    const o = gaussianOf(arg(c, c.to === 0 ? 1 : 0))
    return { message: [meanLambda * o.mean, meanLambda] }
  },
  expectedLog: (c) => {
    const { scale, offset } = precisionParams(c.params)
    if (offset !== 0)
      throw new OmaiError('not-implemented', 'gaussian-gamma-precision: a precision offset has no variational rule')
    const lambda = arg(c, 2)
    const mo = lambda.family.moments(etaOf(lambda))
    return (
      0.5 * (Math.log(scale) + (mo[0] ?? 0) - Math.log(2 * Math.PI)) -
      0.5 * scale * (mo[1] ?? 0) * expectedSquare(arg(c, 0), arg(c, 1))
    )
  },
}

/** The `scale` (default 1) and `offset` (default 0) of the precision `scale lambda + offset` of a `gaussian-gamma-precision` factor. */
function precisionParams(params: Params): { readonly scale: number; readonly offset: number } {
  const scale = numberParam(params, 'scale', 1)
  const offset = numberParam(params, 'offset', 0)
  if (!(scale >= 0 && offset >= 0 && scale + offset > 0))
    throw new OmaiError('value', 'gaussian-gamma-precision: the scale and the offset are non-negative, not both zero')
  return { scale, offset }
}

/** log N(y; loc, exp(2 s)) as a function of the log scale s. */
const logLikelihoodOfLogScale = (y: number, loc: number, s: number): number =>
  -0.5 * Math.log(2 * Math.PI) - s - 0.5 * (y - loc) ** 2 * Math.exp(-2 * s)

/**
 * x ~ N(loc, exp(2 s)) with the log of the scale `s` a Gaussian variable and the value `x` observed: the observation of a
 * stochastic volatility model (the returns have scale `exp(h / 2)` with `h` a Gaussian log-variance). The log likelihood
 * `-s - (x - loc)^2 exp(-2 s) / 2` is concave in `s`, so the tilted density (the cavity of `s` times the likelihood) is
 * unimodal; expectation propagation matches its moments by quadrature on a grid over the cavity and sends the Gaussian
 * site that turns the cavity into them. There is no variational rule (the expected likelihood has no closed form in a Gaussian
 * `s`).
 *
 * @concept factor-operators
 * @index structure An observation with a Gaussian log scale, x ~ N(loc, exp(2 s)): quadrature expectation propagation.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, gaussianLogScale, messageArg, valueArg } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussian, 2), messageArg(gaussian, gaussianEta(0, 1))]
 * gaussianLogScale.ep?.({ mode: 'ep', to: 1, args, params: { loc: 0 }, power: 1 }).message.map((v) => v.toFixed(3)) // => ['2.348', '2.500']
 * ```
 */
export const gaussianLogScale: FactorOperator = {
  name: 'gaussian-log-scale',
  description: 'x ~ N(loc, exp(2 s)) with a Gaussian log scale s and an observed x',
  ports: [
    { name: 'x', accepts: ['gaussian'] as const, constant: true },
    { name: 'logScale', accepts: ['gaussian'] as const },
  ],
  support: { ep: 'numeric' },
  readsTarget: (to) => to === 1,
  noRule: {
    vmp: 'the expected Gaussian likelihood of a log scale has no closed form (use expectation propagation or a sampler)',
  },
  ep: (c): OperatorResult => {
    if (c.to !== 1) throw new OmaiError('value', 'gaussian-log-scale: the value x must be observed')
    const y = gaussianOf(arg(c, 0)).mean
    const loc = c.params['loc'] === undefined ? 0 : numberParam(c.params, 'loc')
    const s = gaussianOf(arg(c, 1))
    const t = tiltedLogGrid(s.mean, s.variance, (v) => logLikelihoodOfLogScale(y, loc, v))
    return { message: gaussianSiteMessage(etaOf(arg(c, 1)), t.mean, t.variance) }
  },
  logEvidence: (c) => {
    if (!allProper(c.args)) return Number.NaN
    const y = gaussianOf(arg(c, 0)).mean
    const loc = c.params['loc'] === undefined ? 0 : numberParam(c.params, 'loc')
    const s = gaussianOf(arg(c, 1))
    return massOf(arg(c, 1)) + tiltedLogGrid(s.mean, s.variance, (v) => logLikelihoodOfLogScale(y, loc, v)).logMass
  },
}

/**
 * x ~ N(mu, 1 / (c lambda)) with the mean and the precision one normal-gamma variable (mu, lambda) and `c` a known
 * multiple (the parameter `precision`, default 1): the observation of a Gaussian with unknown mean and precision, the
 * likelihood the normal-gamma prior is conjugate to. For an observed value the factor is itself in the family, so the
 * message to the normal-gamma is the same under every rule and exact: the coefficients of the sufficient statistics
 * (lambda mu, lambda mu^2, lambda, log lambda), `[c x, -c/2, -c x^2 / 2, 1/2]` (the posterior of `n` observations is the
 * normal-gamma with `beta + c n`, `a + n/2` and the usual `b`, and EP reaches it in one sweep). Variational message
 * passing also takes a Gaussian value that is itself a random variable (the expectation of the statistics); expectation
 * propagation does not, it needs the value observed.
 *
 * @concept factor-operators
 * @index structure A Gaussian observation of a normal-gamma mean and precision (conjugate, exact for an observed value).
 * @example
 * ```ts
 * import { gaussianNormalGamma, normalGamma, normalGammaEta, valueArg, messageArg } from 'omai-engine/message-passing'
 * import { gaussian } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussian, 2), messageArg(normalGamma, normalGammaEta(0, 1, 2, 1))]
 * gaussianNormalGamma.ep?.({ mode: 'ep', to: 1, args, params: {}, power: 1 }).message // => [2, -0.5, -2, 0.5]
 * ```
 */
export const gaussianNormalGamma: FactorOperator = {
  name: 'gaussian-normal-gamma',
  description: 'x ~ N(mu, 1 / (c lambda)) with (mu, lambda) normal-gamma',
  ports: [
    { name: 'x', accepts: ['gaussian'], constant: true },
    { name: 'theta', accepts: ['normal-gamma'] },
  ],
  support: { ep: 'exact', vmp: 'exact' },
  ep: (c): OperatorResult => {
    const x = arg(c, 0)
    if (x.value === undefined)
      throw new OmaiError('value', 'gaussian-normal-gamma: expectation propagation needs the observation observed')
    if (c.to !== 1) throw new OmaiError('value', 'gaussian-normal-gamma: the value is observed, nothing is sent to it')
    const k = numberParam(c.params, 'precision', 1)
    const v = gaussianOf(x).mean
    return { message: [k * v, -k / 2, (-k * v * v) / 2, 0.5] }
  },
  vmp: (c): OperatorResult => {
    const x = gaussianOf(arg(c, 0))
    const k = numberParam(c.params, 'precision', 1)
    if (c.to === 1) return { message: [k * x.mean, -k / 2, (-k * (x.variance + x.mean ** 2)) / 2, 0.5] }
    const t = arg(c, 1)
    const mo = t.family.moments(etaOf(t))
    return { message: [mo[0] ?? 0, mo[2] ?? 0] }
  },
  logEvidence: (c) => {
    const x = arg(c, 0)
    const t = arg(c, 1)
    if (x.value === undefined || !allProper(c.args)) return Number.NaN
    const k = numberParam(c.params, 'precision', 1)
    const v = gaussianOf(x).mean
    // N(x | mu, 1 / (k lambda)) = sqrt(k / 2 pi) exp(message . statistics): the mass of cavity times it is the new log partition.
    return (
      0.5 * Math.log(k / (2 * Math.PI)) +
      t.family.logPartition(rawSum(etaOf(t), [k * v, -k / 2, (-k * v * v) / 2, 0.5]))
    )
  },
  expectedLog: (c) => {
    const x = gaussianOf(arg(c, 0))
    const k = numberParam(c.params, 'precision', 1)
    const t = arg(c, 1)
    const mo = t.family.moments(etaOf(t))
    const quad = (x.variance + x.mean ** 2) * (mo[2] ?? 0) - 2 * x.mean * (mo[0] ?? 0) + (mo[1] ?? 0)
    return 0.5 * ((mo[3] ?? 0) + Math.log(k) - Math.log(2 * Math.PI)) - 0.5 * k * quad
  },
}

/**
 * y = a b for Gaussian a and b: the product of two Gaussian variables. The message to y matches the exact mean and
 * variance of the product of independent Gaussians; the messages to a and b are moment-matched by one-dimensional
 * quadrature against the other argument's closed-form integral.
 *
 * @concept factor-operators
 * @index structure The product of two Gaussian variables (moment matching).
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, product } from 'omai-engine/message-passing'
 * const a = messageArg(gaussian, gaussianEta(2, 1))
 * const b = messageArg(gaussian, gaussianEta(3, 4))
 * const message = product.ep?.({ mode: 'ep', to: 0, power: 1, params: {}, args: [messageArg(gaussian, gaussian.flat), a, b] }).message
 * message?.map((v) => Math.round(v * 1e9) / 1e9) // => [0.206896552, 0.034482759]
 * ```
 */
export const product: FactorOperator = {
  name: 'product',
  description: 'y = a b for Gaussian a and b',
  ports: [
    { name: 'y', accepts: ['gaussian'], constant: true },
    { name: 'a', accepts: ['gaussian'], constant: true },
    { name: 'b', accepts: ['gaussian'], constant: true },
  ],
  support: { ep: 'numeric' },
  readsTarget: (to) => to !== 0,
  ep: (c): OperatorResult => {
    const a = gaussianOf(arg(c, 1))
    const b = gaussianOf(arg(c, 2))
    if (c.to === 0) {
      const incoming = etaOf(arg(c, 0))
      const closed = {
        mean: a.mean * b.mean,
        variance: a.variance * b.variance + a.variance * b.mean ** 2 + b.variance * a.mean ** 2,
      }
      // With nothing known about y the message is the matched pushforward; otherwise the product's distribution is
      // weighted by the incoming message of y before matching, as EP asks (by a product rule over a and b).
      if ((incoming[0] ?? 0) === 0 && (incoming[1] ?? 0) === 0)
        return { message: gaussianMessage(closed.mean, closed.variance) }
      const y = gaussianOf(arg(c, 0))
      // Given a, y is Gaussian (mean a m_b, variance a^2 v_b): the weight of a node is the overlap with the incoming
      // message of y, and the tilted y of that node a product of two Gaussians, in closed form.
      const na = gaussianGrid(a.mean, a.variance)
      let mass = 0
      let m1 = 0
      let m2 = 0
      for (let i = 0; i < na.x.length; i++) {
        const node = na.x[i] ?? 0
        const s2 = node * node * b.variance
        const center = node * b.mean
        const w = (na.w[i] ?? 0) * Math.exp(logNormal(center, y.mean, y.variance + s2))
        const post = s2 > 0 ? 1 / (1 / s2 + 1 / y.variance) : 0
        const mp = s2 > 0 ? post * (center / s2 + y.mean / y.variance) : center
        mass += w
        m1 += w * mp
        m2 += w * (post + mp * mp)
      }
      const mean = m1 / mass
      const variance = m2 / mass - mean * mean
      return { message: gaussianSiteMessage(incoming, mean, variance) }
    }
    const [self, other] = c.to === 1 ? [a, b] : [b, a]
    const y = gaussianOf(arg(c, 0))
    const tilted = tiltedGrid(self.mean, self.variance, productDensity(other, y))
    return { message: gaussianSiteMessage(etaOf(arg(c, c.to)), tilted.mean, tilted.variance) }
  },
  logEvidence: (c) => {
    // A product whose output has no other factor (nothing observed downstream) has a flat cavity at y: no mass to report.
    if (!allProper(c.args)) return Number.NaN
    const a = gaussianOf(arg(c, 1))
    const b = gaussianOf(arg(c, 2))
    const y = gaussianOf(arg(c, 0))
    const mass = tiltedGrid(a.mean, a.variance, productDensity(b, y)).mass
    return massOf(arg(c, 0)) + massOf(arg(c, 1)) + massOf(arg(c, 2)) + Math.log(mass)
  },
}

/** g(a) = integral of the other factor against the output's incoming message: N(a m_b; m_y, a^2 v_b + v_y). */
function productDensity(
  other: { readonly mean: number; readonly variance: number },
  y: { readonly mean: number; readonly variance: number },
): (a: number) => number {
  return (a) => {
    const v = a * a * other.variance + y.variance
    if (!(v > 0))
      throw new OmaiError('value', 'product: the output and one factor are both observed, which determines the other')
    return Math.exp(logNormal(a * other.mean, y.mean, v))
  }
}

/**
 * x = y: two variables that are the same quantity (a copy). The message to one is the message of the other, under
 * every rule.
 *
 * @concept factor-operators
 * @index structure Equality of two variables of one family: a copy.
 * @example
 * ```ts
 * import { equal, gaussian, gaussianEta, messageArg } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussian.flat), messageArg(gaussian, gaussianEta(1, 2))]
 * equal.ep?.({ mode: 'ep', to: 0, args, params: {}, power: 1 }).message // => [0.5, 0.5]
 * ```
 */
export const equal: FactorOperator = {
  name: 'equal',
  description: 'x = y',
  ports: [
    { name: 'x', accepts: ALL },
    { name: 'y', accepts: ALL },
  ],
  support: { ep: 'exact', vmp: 'exact', bp: 'exact' },
  // A copy defines x from y: under the mean field x is y, so its belief is y's and what x's other factors say goes to y.
  defines: 0,
  check: (_, families) =>
    families[0]?.kind === families[1]?.kind && families[0]?.size === families[1]?.size
      ? undefined
      : 'equal: both variables must be of one family',
  ep: (c) => ({ message: etaOf(arg(c, 1 - c.to)) }),
  vmp: (c) => ({ message: etaOf(arg(c, 1 - c.to)) }),
  bp: (c) => ({ message: etaOf(arg(c, 1 - c.to)) }),
  logEvidence: (c) => {
    const a = arg(c, 0)
    return a.family.logPartition(rawSum(etaOf(a), etaOf(arg(c, 1))))
  },
}
