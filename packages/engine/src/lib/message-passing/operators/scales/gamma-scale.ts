/**
 * The factors that make scale hierarchies out of gamma variables: a gamma variable whose rate is a known multiple of another
 * gamma variable (`gamma-rate`: the hierarchy that makes a half-Cauchy and a Student-t, the scale mixtures of Gaussians), and
 * the product of two gamma variables (`gamma-product`: the precision of a Gaussian that is a local precision times a global
 * one, a horseshoe coefficient or a heavy-tailed observation with an unknown scale). Both reduce, against gamma cavities, to a
 * one-dimensional integral over the log of one variable (the other is conditionally a gamma), so expectation propagation is a
 * grid in that one variable and every moment follows in closed form.
 */

import { OmaiError } from '../../base/index.ts'
import { digammaOf, lgammaOf, trigammaOf } from '../_scalar.ts'
import { gamma, gammaShapeRate } from '../families.ts'
import type { Arg, Eta, FactorOperator, OperatorContext, OperatorResult } from '../types.ts'
import { etaOf, ImproperMessageError, numberParam, siteMessage } from './args.ts'

const arg = (c: OperatorContext, i: number): Arg => c.args[i] as Arg

const POINTS = 241
const SPAN = 18

/**
 * The tilted moments of two gamma variables `G` and `H` under a weight `q_G(g) q_H(h) w(g h)`-type factor that, integrated over `h`,
 * leaves `g^kappa (beta0 + c g)^-s` times the cavity of `G`, with `H | G` a gamma of shape `s` and rate `beta0 + c g`. The grid is
 * over `log g`, centred on the cavity of `G`.
 */
function pairTilt(
  cavity: Eta,
  kappa: number,
  beta0: number,
  c: number,
  s: number,
): {
  readonly logZ: number
  readonly logG: number
  readonly g: number
  readonly logH: number
  readonly h: number
  readonly gh: number
} {
  const { shape, rate } = gammaShapeRate(cavity)
  if (!(shape > 0 && rate > 0 && Number.isFinite(shape) && Number.isFinite(rate)))
    throw new ImproperMessageError('a gamma message has no density (its shape and rate are not positive)')
  if (!(s > 0))
    throw new ImproperMessageError(
      'the integral over the dependent gamma variable diverges (its shape is not positive)',
    )
  const centre = digammaOf(shape) - Math.log(rate)
  const half = SPAN * Math.sqrt(trigammaOf(shape))
  const lo = Math.max(centre - half, -60)
  const hi = Math.min(centre + half, 60)
  const step = (hi - lo) / (POINTS - 1)
  const logw = new Float64Array(POINTS)
  let top = Number.NEGATIVE_INFINITY
  for (let i = 0; i < POINTS; i++) {
    const t = lo + i * step
    const rateH = beta0 + c * Math.exp(t)
    if (!(rateH > 0)) throw new ImproperMessageError('gamma-scale: the conditional rate is not positive')
    // The cavity in t = log g has density exp(shape t - rate e^t); the other factors give g^kappa (beta0 + c g)^-s.
    const lw = shape * t - rate * Math.exp(t) + kappa * t - s * Math.log(rateH)
    logw[i] = lw
    if (lw > top) top = lw
  }
  const psi = digammaOf(s)
  let z = 0
  let eLogG = 0
  let eG = 0
  let eLogH = 0
  let eH = 0
  let eGH = 0
  for (let i = 0; i < POINTS; i++) {
    const t = lo + i * step
    const w = Math.exp((logw[i] ?? 0) - top)
    const g = Math.exp(t)
    const rateH = beta0 + c * g
    z += w
    eLogG += w * t
    eG += w * g
    eLogH += w * (psi - Math.log(rateH))
    eH += w * (s / rateH)
    eGH += w * ((g * s) / rateH)
  }
  return {
    logZ: top + Math.log(z * step),
    logG: eLogG / z,
    g: eG / z,
    logH: eLogH / z,
    h: eH / z,
    gh: eGH / z,
  }
}

/** The shape (known) and the rate multiplier of a `gamma-rate` factor. */
function rateParams(c: OperatorContext): { readonly shape: number; readonly scale: number } {
  const shape = numberParam(c.params, 'shape')
  const scale = numberParam(c.params, 'scale', 1)
  if (!(shape > 0 && scale > 0)) throw new OmaiError('value', 'gamma-rate: the shape and the scale are positive')
  return { shape, scale }
}

/**
 * `x ~ Gamma(a, c r)`: a gamma variable `x` whose rate is a known multiple `c` of another gamma variable `r`, the shape `a`
 * known. With `a = 1/2` and `r` itself a gamma this is the Makalic-Schmidt representation of the half-Cauchy scale
 * (`1/lambda^2 | b ~ Gamma(1/2, b)`, `b ~ Gamma(1/2, 1)`), and with `a = nu/2` it is the Student-t's mixing precision. The
 * integral over `x` against its gamma cavity is a power of `beta_x + c r`, so expectation propagation is a grid in `log r`:
 * the message to `r` is the gamma with the tilted moments, and `x | r` is gamma, so the message to `x` is the mixture over
 * the grid. Variational message passing is the conjugate rule.
 *
 * @concept factor-operators
 * @index structure A gamma variable whose rate is a known multiple of another gamma variable: scale-mixture hierarchies.
 * @example
 * ```ts
 * import { gamma, gammaEta, gammaRate, messageArg } from 'omai-engine/message-passing'
 * const args = [messageArg(gamma, gammaEta(3, 2)), messageArg(gamma, gammaEta(2, 1))]
 * gammaRate.ep?.({ mode: 'ep', to: 1, args, params: { shape: 0.5 }, power: 1 }).message.map((v) => v.toFixed(3)) // => ['0.167', '-0.419']
 * ```
 */
export const gammaRate: FactorOperator = {
  name: 'gamma-rate',
  description: 'x ~ Gamma(shape, scale r) with the rate a known multiple of a gamma variable r',
  ports: [
    { name: 'x', accepts: ['gamma'] },
    { name: 'r', accepts: ['gamma'] },
  ],
  support: { ep: 'numeric', vmp: 'exact' },
  readsTarget: (to) => to === 1,
  ep: (c): OperatorResult => {
    const { shape: a, scale } = rateParams(c)
    const x = gammaShapeRate(etaOf(arg(c, 0)))
    if (!(x.shape > 0 && Number.isFinite(x.shape) && x.rate >= 0))
      throw new ImproperMessageError('gamma-rate: the cavity of x has no shape')
    const t = pairTilt(etaOf(arg(c, 1)), a, x.rate, scale, x.shape + a - 1)
    if (c.to === 1) return { message: siteMessage(gamma, etaOf(arg(c, 1)), [t.logG, t.g]) }
    return { message: siteMessage(gamma, etaOf(arg(c, 0)), [t.logH, t.h]) }
  },
  vmp: (c): OperatorResult => {
    const { shape: a, scale } = rateParams(c)
    const other = arg(c, 1 - c.to)
    const mo = other.family.moments(etaOf(other))
    // log f = a log r + (a - 1) log x - c r x + const: linear in (log r, r) and (log x, x).
    if (c.to === 0) return { message: [a - 1, -scale * (mo[1] ?? 0)] }
    return { message: [a, -scale * (mo[1] ?? 0)] }
  },
  expectedLog: (c) => {
    const { shape: a, scale } = rateParams(c)
    const x = arg(c, 0).family.moments(etaOf(arg(c, 0)))
    const r = arg(c, 1).family.moments(etaOf(arg(c, 1)))
    return (
      a * Math.log(scale) - lgammaOf(a) + a * (r[0] ?? 0) + (a - 1) * (x[0] ?? 0) - scale * (r[1] ?? 0) * (x[1] ?? 0)
    )
  },
  logEvidence: (c) => {
    try {
      const { shape: a, scale } = rateParams(c)
      const x = gammaShapeRate(etaOf(arg(c, 0)))
      const s = x.shape + a - 1
      // The integral over x is Gamma(s) / (beta_x + c r)^s, and the grid integrates the rest against the (unnormalised) cavity of r.
      return pairTilt(etaOf(arg(c, 1)), a, x.rate, scale, s).logZ + a * Math.log(scale) - lgammaOf(a) + lgammaOf(s)
    } catch (err) {
      if (err instanceof ImproperMessageError) return Number.NaN
      throw err
    }
  },
}

/**
 * `p = u tau` for three gamma variables: the precision of a Gaussian that is a product of a local and a global precision (a
 * horseshoe coefficient's `a_j g`, a robust observation's mixing precision times the noise precision). The factor is a
 * point mass, so against gamma cavities the variables `u`, `tau`, `p` are all functions of one one-dimensional integral over
 * `log u` (given `u`, the tilted `tau` is gamma): the messages to `u` and `tau` are the gammas with those moments, and the
 * message to `p` has `E log p = E log u + E log tau` and `E p = E[u tau]`, so a flat cavity of `p` gives the plain pushforward.
 * Variational message passing is closed form: `p` is the pushforward (the gamma of those two moments) and the messages to `u` and `tau`
 * are the readers' message to `p` with the other's mean folded in.
 *
 * @concept factor-operators
 * @index structure The product of two gamma variables p = u tau: a local times a global precision.
 * @example
 * ```ts
 * import { gamma, gammaEta, gammaProduct, messageArg } from 'omai-engine/message-passing'
 * const args = [messageArg(gamma, gamma.flat), messageArg(gamma, gammaEta(3, 2)), messageArg(gamma, gammaEta(5, 4))]
 * gammaProduct.ep?.({ mode: 'ep', to: 0, args, params: {}, power: 1 }).message.map((v) => v.toFixed(3)) // => ['0.941', '-1.035']
 * ```
 */
export const gammaProduct: FactorOperator = {
  name: 'gamma-product',
  description: 'p = u tau for gamma variables',
  ports: [
    { name: 'p', accepts: ['gamma'] },
    { name: 'u', accepts: ['gamma'] },
    { name: 'tau', accepts: ['gamma'] },
  ],
  support: { ep: 'numeric', vmp: 'exact' },
  // Deterministic: under the mean field `p` is the pushforward of `u` and `tau`, and the messages to them are those of the
  // substituted model. `E log p = E log u + E log tau` and `E p = E u E tau` are exact, so the gamma of those two moments serves
  // every reader of `p` whose log factor is linear in `(log p, p)` (a gamma or Gaussian precision) exactly.
  defines: 0,
  vmp: (c): OperatorResult => {
    const [lu, mu] = arg(c, 1).family.moments(etaOf(arg(c, 1)))
    const [lt, mt] = arg(c, 2).family.moments(etaOf(arg(c, 2)))
    if (c.to === 0) return { message: gamma.fromMoments([(lu ?? 0) + (lt ?? 0), (mu ?? 0) * (mt ?? 0)]) }
    // The readers' message to p, a log p + b p, becomes a log u + b E[tau] u at u (and symmetrically at tau).
    const cavity = etaOf(arg(c, 0))
    const other = c.to === 1 ? (mt ?? 0) : (mu ?? 0)
    return { message: [cavity[0] ?? 0, (cavity[1] ?? 0) * other] }
  },
  expectedLog: () => 0,
  ep: (c): OperatorResult => {
    const p = gammaShapeRate(etaOf(arg(c, 0)))
    if (!(p.shape > 0 && Number.isFinite(p.shape) && p.rate >= 0))
      throw new ImproperMessageError('gamma-product: the cavity of p has no shape')
    const properGamma = (i: number): boolean => gamma.isProper(etaOf(arg(c, i)))
    // The grid runs over u when its cavity is a density, else over tau (the factor is symmetric in the two).
    const [gi, hi] = properGamma(1) ? [1, 2] : [2, 1]
    const h = gammaShapeRate(etaOf(arg(c, hi)))
    const t = pairTilt(etaOf(arg(c, gi)), p.shape - 1, h.rate, p.rate, h.shape + p.shape - 1)
    const moments = (port: number): Eta =>
      port === 0 ? [t.logG + t.logH, t.gh] : port === gi ? [t.logG, t.g] : [t.logH, t.h]
    return { message: siteMessage(gamma, etaOf(arg(c, c.to)), moments(c.to)) }
  },
  logEvidence: (c) => {
    try {
      const p = gammaShapeRate(etaOf(arg(c, 0)))
      const properGamma = (i: number): boolean => gamma.isProper(etaOf(arg(c, i)))
      const [gi, hi] = properGamma(1) ? [1, 2] : [2, 1]
      const h = gammaShapeRate(etaOf(arg(c, hi)))
      const t = pairTilt(etaOf(arg(c, gi)), p.shape - 1, h.rate, p.rate, h.shape + p.shape - 1)
      return t.logZ + lgammaOf(h.shape + p.shape - 1)
    } catch (err) {
      if (err instanceof ImproperMessageError) return Number.NaN
      throw err
    }
  },
}
