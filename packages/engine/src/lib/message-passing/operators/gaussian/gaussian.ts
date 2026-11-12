/**
 * The Gaussian-family operators: a fixed prior message (any family), a Gaussian with a fixed precision, a Gaussian
 * with a gamma precision, a Gaussian observation with a normal-gamma mean and precision, a product of Gaussians, and
 * the equality of two variables. Moment matching is exact for the first two and for the equality; the product's
 * messages to its factors are numerical (one-dimensional quadrature, the rest in closed form).
 */

import { OmaiError } from '../../../base/index.ts'
import { arg, errorText, keyOf, memoised } from '../../core/_messages.ts'
import {
  allProper,
  etaOf,
  gaussianMessage,
  gaussianOf,
  gaussianSiteMessage,
  logGaussianIntegral,
  logNormal,
  massOf,
  numberParam,
  rawSum,
  scalarOf,
  vectorParam,
} from '../../core/args.ts'
import { defineOperator, messageRule, siteRule } from '../../core/define.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { gammaShapeRate } from '../../core/families.ts'
import { param } from '../../core/params.ts'
import type { Arg, FactorOperator, FamilyKind, OperatorContext, OperatorResult, Params } from '../../core/types.ts'
import { centredMoments } from '../../numerics/_moments.ts'
import { gaussianGrid, tiltedGrid, tiltedLogGrid } from '../../numerics/_quadrature.ts'
import { gammaLogWindow, tiltGrid, windowProfiles } from '../../numerics/_windows.ts'
import { numericExpectedLog, numericMessage, quadraturePorts } from '../kernel/variational-numeric.ts'

const ALL: readonly FamilyKind[] = [
  'gaussian',
  'gaussian-mv',
  'gamma',
  'beta',
  'bernoulli',
  'discrete',
  'dirichlet',
  'normal-gamma',
  'wishart',
  'normal-wishart',
]

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
export const prior: FactorOperator = defineOperator({
  name: 'prior',
  description: 'a normalised density with the given natural parameters on one variable',
  ports: [{ name: 'x', accepts: ALL }],
  params: { eta: param.vector() },
  check: (params, families) =>
    vectorParam(params, 'eta').length === families[0]?.size
      ? undefined
      : 'prior: eta must have the length of the family',
  rules: {
    ep: messageRule((c) => ({ message: vectorParam(c.params, 'eta') })),
    vmp: messageRule((c) => ({ message: vectorParam(c.params, 'eta') })),
    bp: messageRule((c) => ({ message: vectorParam(c.params, 'eta') })),
  },
  evidence: {
    ep: (c) => {
      const a = arg(c, 0)
      const p = vectorParam(c.params, 'eta')
      return a.family.logPartition(rawSum(etaOf(a), p)) - a.family.logPartition(p)
    },
    vmp: (c) => {
      const a = arg(c, 0)
      const p = vectorParam(c.params, 'eta')
      return a.family.pair(p, a.family.moments(etaOf(a))) - a.family.logPartition(p)
    },
  },
})

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
export const gaussianOperator: FactorOperator = defineOperator({
  name: 'gaussian',
  description: 'x ~ N(mean, 1 / precision) with a fixed precision',
  ports: gaussianPorts,
  params: { precision: param.positive() },
  rules: {
    ep: messageRule((c): OperatorResult => {
      const o = gaussianOf(arg(c, 1 - c.to))
      return { message: gaussianMessage(o.mean, o.variance + 1 / numberParam(c.params, 'precision')) }
    }),
    vmp: messageRule((c): OperatorResult => {
      const tau = numberParam(c.params, 'precision')
      const o = gaussianOf(arg(c, 1 - c.to))
      return { message: [tau * o.mean, tau] }
    }),
  },
  evidence: {
    // The two cavities need only be proper together (a message that is the site of a likelihood can have a negative
    // precision), so the integral is taken over the pair in natural parameters.
    ep: (c) => {
      const tau = numberParam(c.params, 'precision')
      const x = arg(c, 0)
      const m = arg(c, 1)
      const base = 0.5 * Math.log(tau / (2 * Math.PI))
      if (x.value !== undefined && m.value !== undefined) return base - 0.5 * tau * (scalarOf(x) - scalarOf(m)) ** 2
      if (x.value !== undefined || m.value !== undefined) {
        const [v, e] = x.value !== undefined ? [scalarOf(x), etaOf(m)] : [scalarOf(m), etaOf(x)]
        return base - 0.5 * tau * v * v + logGaussianIntegral([(e[0] ?? 0) + tau * v], [(e[1] ?? 0) + tau], 1)
      }
      const ex = etaOf(x)
      const em = etaOf(m)
      const p = [(ex[1] ?? 0) + tau, -tau, -tau, (em[1] ?? 0) + tau]
      return base + logGaussianIntegral([ex[0] ?? 0, em[0] ?? 0], p, 2)
    },
    vmp: (c) => {
      const tau = numberParam(c.params, 'precision')
      return 0.5 * Math.log(tau / (2 * Math.PI)) - 0.5 * tau * expectedSquare(arg(c, 0), arg(c, 1))
    },
  },
})

const TILT_POINTS = 801
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
  scale: number,
  offset: number,
  c: OperatorContext,
): Tilt {
  // A gate asks for the same tilt of a branch several times in a row (its evidence, then the message to each variable), and
  // each is a grid of hundreds of points: the engine's cache for the call keeps it under its inputs.
  return memoised(c, `precision-tilt:${keyOf([x.mean, x.variance, u.mean, u.variance, lambda, scale, offset])}`, () =>
    precisionTiltGrid(x, u, lambda, scale, offset),
  )
}

/** What the grid over the precision gives: the moments of both Gaussians and of the precision, and the integral. */
interface Tilt {
  readonly x: { readonly mean: number; readonly variance: number }
  readonly u: { readonly mean: number; readonly variance: number }
  readonly logLambda: number
  readonly lambda: number
  /** log of the integral of the (unnormalised) gamma cavity times the normal density of the difference, over `log lambda`. */
  readonly logZ: number
}

function precisionTiltGrid(
  x: { readonly mean: number; readonly variance: number },
  u: { readonly mean: number; readonly variance: number },
  lambda: readonly number[],
  scale: number,
  offset: number,
): Tilt {
  const { shape, rate } = gammaShapeRate(lambda)
  if (!(shape > 0 && rate > 0 && Number.isFinite(shape) && Number.isFinite(rate)))
    throw new ImproperMessageError('a gamma message has no density (its shape and rate are not positive)')
  const gap = x.mean - u.mean
  const joint = x.variance + u.variance
  // The gamma density in t = log lambda is exp(shape t - rate e^t); the difference integrated out is N(0; gap, joint + e^-t).
  const {
    t: ts,
    step,
    top,
    weights,
    total: z,
  } = tiltGrid(
    gammaLogWindow(shape, rate, windowProfiles.logTilt),
    TILT_POINTS,
    (t) => shape * t - rate * Math.exp(t) + logNormal(0, gap, joint + 1 / (scale * Math.exp(t) + offset)),
  )
  // Given the precision, each Gaussian has its Kalman update: the tilted Gaussian is the mixture of the updates over the grid, with
  // the mean and variance of the mixture (the mean of the variances plus the variance of the means, centred).
  const mx = new Float64Array(TILT_POINTS)
  const vx = new Float64Array(TILT_POINTS)
  const mu = new Float64Array(TILT_POINTS)
  const vu = new Float64Array(TILT_POINTS)
  for (let i = 0; i < TILT_POINTS; i++) {
    const s = joint + 1 / (scale * Math.exp(ts[i] as number) + offset)
    mx[i] = x.mean - (x.variance * gap) / s
    vx[i] = x.variance - (x.variance * x.variance) / s
    mu[i] = u.mean + (u.variance * gap) / s
    vu[i] = u.variance - (u.variance * u.variance) / s
  }
  const mixture = (means: Float64Array, variances: Float64Array): { mean: number; variance: number } => {
    const m = centredMoments(weights, z, means)
    let within = 0
    for (let i = 0; i < TILT_POINTS; i++) within += (weights[i] as number) * (variances[i] as number)
    return { mean: m.mean, variance: within / z + m.variance }
  }
  let eLog = 0
  let eLam = 0
  for (let i = 0; i < TILT_POINTS; i++) {
    eLog += (weights[i] as number) * (ts[i] as number)
    eLam += (weights[i] as number) * Math.exp(ts[i] as number)
  }
  return {
    x: mixture(mx, vx),
    u: mixture(mu, vu),
    logLambda: eLog / z,
    lambda: eLam / z,
    logZ: top + Math.log(z * step),
  }
}

/**
 * x ~ N(mean, 1 / lambda) with a gamma-distributed precision lambda: the factor of a Gaussian with unknown noise.
 * Variational message passing (Winn and Bishop) uses the expected log density, closed forms that make the model
 * conjugate-exponential. Expectation propagation matches the moments of the tilted distribution (the cavities of `x`, of the
 * mean and of the precision times the factor), computed by quadrature in `log lambda`: `x` and the mean each get a Gaussian
 * site, the precision a gamma site (the projection of `E log lambda` and `E lambda`), for an observed or a latent `x`. This
 * is what the linear regression with an unknown noise level needs, and a population of latent values with an unknown spread
 * (the abilities of an item response model). A Gaussian port whose cavity is still flat is shown to it as very wide by the engine (`needsDensity`), so the first
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
export const gaussianGammaPrecision: FactorOperator = defineOperator({
  name: 'gaussian-gamma-precision',
  description: 'x ~ N(mean, 1 / (scale lambda + offset)) with lambda a gamma-distributed precision',
  ports: [...gaussianPorts, { name: 'precision', accepts: ['gamma'] }],
  params: { scale: param.nonNegative({ optional: true }), offset: param.nonNegative({ optional: true }) },
  check: (params) => {
    try {
      precisionParams(params)
      return undefined
    } catch (e) {
      return errorText(e)
    }
  },
  rules: {
    // The tilted distribution of the factor against the three cavities: a Gaussian for `x` and for the mean, the moments of
    // `log lambda` and `lambda` for the precision (the engine projects them).
    ep: siteRule((c) => {
      const { scale, offset } = precisionParams(c.params)
      const tilt = precisionTilt(gaussianOf(arg(c, 0)), gaussianOf(arg(c, 1)), etaOf(arg(c, 2)), scale, offset, c)
      if (c.to === 0) return tilt.x
      if (c.to === 1) return tilt.u
      return { moments: [tilt.logLambda, tilt.lambda] }
    }),
    vmp: messageRule((c): OperatorResult => {
      const { scale, offset } = precisionParams(c.params)
      if (offset !== 0)
        throw new OmaiError('not-implemented', 'gaussian-gamma-precision: a precision offset has no variational rule')
      const lambda = arg(c, 2)
      const mo = lambda.family.moments(etaOf(lambda))
      const meanLambda = scale * (mo[1] ?? 0)
      if (c.to === 2) return { message: [0.5, -0.5 * scale * expectedSquare(arg(c, 0), arg(c, 1))] }
      const o = gaussianOf(arg(c, c.to === 0 ? 1 : 0))
      return { message: [meanLambda * o.mean, meanLambda] }
    }),
  },
  evidence: {
    improperIsNaN: true,
    // The integral of the cavities (unnormalised, so their masses are in it) times the factor: the Gaussian cavities' masses and the
    // grid's integral over the precision, in which the gamma cavity's unnormalised density is already part of the weight.
    ep: (c) => {
      const { scale, offset } = precisionParams(c.params)
      const tilt = precisionTilt(gaussianOf(arg(c, 0)), gaussianOf(arg(c, 1)), etaOf(arg(c, 2)), scale, offset, c)
      return massOf(arg(c, 0)) + massOf(arg(c, 1)) + tilt.logZ
    },
    vmp: (c) => {
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
  },
  // The precision's own cavity must be a gamma density; a Gaussian port whose cavity is still flat is shown very wide by the engine.
  traits: { readsTarget: (to) => to === 2, needsDensity: true },
})

/** The `scale` (default 1) and `offset` (default 0) of the precision `scale lambda + offset` of a `gaussian-gamma-precision` factor. */
function precisionParams(params: Params): { readonly scale: number; readonly offset: number } {
  const scale = numberParam(params, 'scale', 1)
  const offset = numberParam(params, 'offset', 0)
  if (!(scale >= 0 && offset >= 0 && scale + offset > 0))
    throw new ParameterError('gaussian-gamma-precision: the scale and the offset are non-negative, not both zero')
  return { scale, offset }
}

/** log N(y; loc, exp(2 s)) as a function of the log scale s. */
const logLikelihoodOfLogScale = (y: number, loc: number, s: number): number =>
  -0.5 * Math.log(2 * Math.PI) - s - 0.5 * (y - loc) ** 2 * Math.exp(-2 * s)

/** The log density of the factor at the coordinates (x, s): the location is the parameter `loc`. */
const logScaleAt =
  (c: OperatorContext) =>
  (v: readonly number[]): number =>
    logLikelihoodOfLogScale(v[0] as number, numberParam(c.params, 'loc', 0), v[1] as number)

/**
 * x ~ N(loc, exp(2 s)) with the log of the scale `s` a Gaussian variable and the value `x` observed: the observation of a
 * stochastic volatility model (the returns have scale `exp(h / 2)` with `h` a Gaussian log-variance). The log likelihood
 * `-s - (x - loc)^2 exp(-2 s) / 2` is concave in `s`, so the tilted density (the cavity of `s` times the likelihood) is
 * unimodal; expectation propagation matches its moments by quadrature on a grid over the cavity and sends the Gaussian
 * site that turns the cavity into them. Variational message passing is non-conjugate: the expected log is closed form in a
 * Gaussian `s` (`E exp(-2 s)` is a log-normal moment) but `exp(-2 s)` is not a statistic of the Gaussian, so the message is the
 * natural gradient of the expectation (log-concave in `s`: the precision is positive).
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
export const gaussianLogScale: FactorOperator = defineOperator({
  name: 'gaussian-log-scale',
  description: 'x ~ N(loc, exp(2 s)) with a Gaussian log scale s and an observed x',
  ports: [
    { name: 'x', accepts: ['gaussian'] as const, constant: true },
    { name: 'logScale', accepts: ['gaussian'] as const },
  ],
  params: { loc: param.number({ optional: true }) },
  rules: {
    ep: messageRule((c): OperatorResult => {
      if (c.to !== 1) throw new ParameterError('gaussian-log-scale: the value x must be observed')
      const y = gaussianOf(arg(c, 0)).mean
      const loc = c.params['loc'] === undefined ? 0 : numberParam(c.params, 'loc')
      const s = gaussianOf(arg(c, 1))
      const t = tiltedLogGrid(s.mean, s.variance, (v) => logLikelihoodOfLogScale(y, loc, v))
      return { message: gaussianSiteMessage(etaOf(arg(c, 1)), t.mean, t.variance) }
    }, 'numeric'),
    // The mean field: the non-conjugate message of log N(x; loc, exp(2 s)) in the log scale (and in a latent x), by quadrature.
    vmp: messageRule(
      (c): OperatorResult => ({ message: numericMessage(quadraturePorts(c.args), c.to, logScaleAt(c)) }),
      'numeric',
    ),
  },
  evidence: {
    ep: (c) => {
      if (!allProper(c.args)) return Number.NaN
      const y = gaussianOf(arg(c, 0)).mean
      const loc = c.params['loc'] === undefined ? 0 : numberParam(c.params, 'loc')
      const s = gaussianOf(arg(c, 1))
      return massOf(arg(c, 1)) + tiltedLogGrid(s.mean, s.variance, (v) => logLikelihoodOfLogScale(y, loc, v)).logMass
    },
    vmp: (c) => numericExpectedLog(quadraturePorts(c.args), logScaleAt(c)),
  },
  traits: { readsTarget: (to) => to === 1 },
})

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
export const gaussianNormalGamma: FactorOperator = defineOperator({
  name: 'gaussian-normal-gamma',
  description: 'x ~ N(mu, 1 / (c lambda)) with (mu, lambda) normal-gamma',
  ports: [
    { name: 'x', accepts: ['gaussian'], constant: true },
    { name: 'theta', accepts: ['normal-gamma'] },
  ],
  params: { precision: param.positive({ optional: true }) },
  rules: {
    ep: messageRule((c): OperatorResult => {
      const x = arg(c, 0)
      if (x.value === undefined)
        throw new ParameterError('gaussian-normal-gamma: expectation propagation needs the observation observed')
      if (c.to !== 1) throw new ParameterError('gaussian-normal-gamma: the value is observed, nothing is sent to it')
      const k = numberParam(c.params, 'precision', 1)
      const v = gaussianOf(x).mean
      return { message: [k * v, -k / 2, (-k * v * v) / 2, 0.5] }
    }),
    vmp: messageRule((c): OperatorResult => {
      const x = gaussianOf(arg(c, 0))
      const k = numberParam(c.params, 'precision', 1)
      if (c.to === 1) return { message: [k * x.mean, -k / 2, (-k * (x.variance + x.mean ** 2)) / 2, 0.5] }
      const t = arg(c, 1)
      const mo = t.family.moments(etaOf(t))
      return { message: [mo[0] ?? 0, mo[2] ?? 0] }
    }),
  },
  evidence: {
    ep: (c) => {
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
    vmp: (c) => {
      const x = gaussianOf(arg(c, 0))
      const k = numberParam(c.params, 'precision', 1)
      const t = arg(c, 1)
      const mo = t.family.moments(etaOf(t))
      const quad = (x.variance + x.mean ** 2) * (mo[2] ?? 0) - 2 * x.mean * (mo[0] ?? 0) + (mo[1] ?? 0)
      return 0.5 * ((mo[3] ?? 0) + Math.log(k) - Math.log(2 * Math.PI)) - 0.5 * k * quad
    },
  },
})

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
export const product: FactorOperator = defineOperator({
  name: 'product',
  description: 'y = a b for Gaussian a and b',
  ports: [
    { name: 'y', accepts: ['gaussian'], constant: true },
    { name: 'a', accepts: ['gaussian'], constant: true },
    { name: 'b', accepts: ['gaussian'], constant: true },
  ],
  params: {},
  rules: {
    ep: messageRule((c): OperatorResult => {
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
        const n = na.x.length
        const w = new Float64Array(n)
        const mp = new Float64Array(n)
        const post = new Float64Array(n)
        let mass = 0
        for (let i = 0; i < n; i++) {
          const node = na.x[i] ?? 0
          const s2 = node * node * b.variance
          const center = node * b.mean
          w[i] = (na.w[i] ?? 0) * Math.exp(logNormal(center, y.mean, y.variance + s2))
          post[i] = s2 > 0 ? 1 / (1 / s2 + 1 / y.variance) : 0
          mp[i] = s2 > 0 ? (post[i] as number) * (center / s2 + y.mean / y.variance) : center
          mass += w[i] as number
        }
        // The mixture of the conditional Gaussians over the nodes: the mean of the variances plus the variance of the means.
        const t = centredMoments(w, mass, mp)
        let within = 0
        for (let i = 0; i < n; i++) within += (w[i] as number) * (post[i] as number)
        const mean = t.mean
        const variance = within / mass + t.variance
        return { message: gaussianSiteMessage(incoming, mean, variance) }
      }
      const [self, other] = c.to === 1 ? [a, b] : [b, a]
      const y = gaussianOf(arg(c, 0))
      const tilted = tiltedGrid(self.mean, self.variance, productDensity(other, y))
      return { message: gaussianSiteMessage(etaOf(arg(c, c.to)), tilted.mean, tilted.variance) }
    }, 'numeric'),
    vmp: messageRule((c): OperatorResult => {
      const a = gaussianOf(arg(c, 1))
      const b = gaussianOf(arg(c, 2))
      if (c.to === 0)
        return {
          message: gaussianMessage(
            a.mean * b.mean,
            a.variance * b.variance + a.variance * b.mean ** 2 + b.variance * a.mean ** 2,
          ),
        }
      const out = arg(c, 0).eta
      if (out === undefined)
        throw new OmaiError(
          'not-implemented',
          'a deterministic factor with an observed output has no variational message',
        )
      const other = c.to === 1 ? b : a
      return { message: [(out[0] ?? 0) * other.mean, (out[1] ?? 0) * (other.variance + other.mean ** 2)] }
    }),
  },
  evidence: {
    ep: (c) => {
      // A product whose output has no other factor (nothing observed downstream) has a flat cavity at y: no mass to report.
      if (!allProper(c.args)) return Number.NaN
      const a = gaussianOf(arg(c, 1))
      const b = gaussianOf(arg(c, 2))
      const y = gaussianOf(arg(c, 0))
      const mass = tiltedGrid(a.mean, a.variance, productDensity(b, y)).mass
      return massOf(arg(c, 0)) + massOf(arg(c, 1)) + massOf(arg(c, 2)) + Math.log(mass)
    },
    vmp: () => 0,
  },
  // Deterministic: under the mean field `y` is the pushforward of `a` and `b` (the moments of a product of independent variables are
  // closed form, and are all a Gaussian reader of `y` sees), and the message to `a` is the expected log of the cavity `(h, J)` of
  // `y` over `b`: `[h E b, J E b^2]`.
  traits: { readsTarget: (to) => to !== 0, defines: 0 },
})

/** g(a) = integral of the other factor against the output's incoming message: N(a m_b; m_y, a^2 v_b + v_y). */
function productDensity(
  other: { readonly mean: number; readonly variance: number },
  y: { readonly mean: number; readonly variance: number },
): (a: number) => number {
  return (a) => {
    const v = a * a * other.variance + y.variance
    if (!(v > 0))
      throw new ParameterError('product: the output and one factor are both observed, which determines the other')
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
export const equal: FactorOperator = defineOperator({
  name: 'equal',
  description: 'x = y',
  ports: [
    { name: 'x', accepts: ALL },
    { name: 'y', accepts: ALL },
  ],
  params: {},
  check: (_, families) =>
    families[0]?.kind === families[1]?.kind && families[0]?.size === families[1]?.size
      ? undefined
      : 'equal: both variables must be of one family',
  rules: {
    ep: messageRule((c) => ({ message: etaOf(arg(c, 1 - c.to)) })),
    vmp: messageRule((c) => ({ message: etaOf(arg(c, 1 - c.to)) })),
    bp: messageRule((c) => ({ message: etaOf(arg(c, 1 - c.to)) })),
  },
  evidence: {
    ep: (c) => {
      const a = arg(c, 0)
      return a.family.logPartition(rawSum(etaOf(a), etaOf(arg(c, 1))))
    },
  },
  // A copy defines x from y: under the mean field x is y, so its belief is y's and what x's other factors say goes to y.
  traits: { defines: 0 },
})
