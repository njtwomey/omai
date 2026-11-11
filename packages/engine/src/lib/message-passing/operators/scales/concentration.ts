/**
 * The factor of a beta or Dirichlet variable whose concentration is random: `p ~ Dirichlet(kappa)` with `kappa_k = w_k alpha + o_k`,
 * a known multiple of one positive variable `alpha` plus a known offset (a symmetric Dirichlet with a gamma or log-normal
 * concentration, `Beta(alpha, alpha)`, a beta with a random first shape, a Dirichlet whose total mass is learned). Given
 * `alpha` the cavity of `p` times the factor is a Dirichlet, so the tilted distribution of `alpha` alone is a one-dimensional
 * integral (the ratio of Dirichlet normalisers, a function of `log alpha` against the cavity of `alpha`) that a grid in `log alpha`
 * does, and `p` is the mixture over that grid of the conditional Dirichlets, matched on `E log p_k` (the family's sufficient
 * statistics). The hyperprior of the concentration is the cavity of `alpha`: a gamma, or a Gaussian on `log alpha` (a log-normal).
 * A beta whose two shapes are separate random values (`beta-two-shapes`) is the same integral in two dimensions, on the grid of
 * `_grid.ts`.
 */

import { OmaiError } from '../../../base/index.ts'
import { arg, evidenceOrNaN } from '../../core/_messages.ts'
import {
  allProper,
  etaOf,
  gaussianOf,
  gaussianSiteMessage,
  ImproperMessageError,
  massOf,
  siteMessage,
  vectorParam,
} from '../../core/args.ts'
import { gamma, gammaShapeRate } from '../../core/families.ts'
import type { Arg, Eta, FactorOperator, OperatorContext, OperatorResult } from '../../core/types.ts'
import { axisOf, integratePair } from '../../numerics/_grid.ts'
import { digammaOf, lgammaOf, trigammaOf } from '../../numerics/_scalar.ts'
import { numericExpectedLog, numericMessage, quadraturePorts } from '../kernel/variational-numeric.ts'

const POINTS = 241
const SPAN = 14

/** The concentration `kappa_k = w_k e^t + o_k` and its pieces from the parameters. */
function concentrations(c: OperatorContext): { readonly w: readonly number[]; readonly o: readonly number[] } {
  const w = vectorParam(c.params, 'weights')
  const o = c.params['offsets'] === undefined ? w.map(() => 0) : vectorParam(c.params, 'offsets')
  if (
    w.length !== o.length ||
    w.some((v) => !(v >= 0)) ||
    o.some((v) => !(v >= 0)) ||
    w.every((v, k) => v === 0 && o[k] === 0)
  )
    throw new OmaiError(
      'value',
      'dirichlet-concentration: weights and offsets are non-negative vectors of one length, never both zero',
    )
  return { w, o }
}

/** E alpha under the belief of a concentration: a gamma's mean, or the log-normal mean of a Gaussian on its log. */
function meanShape(a: Arg): number {
  if (a.family.kind === 'gamma') {
    const { shape, rate } = gammaShapeRate(etaOf(a))
    if (!(shape > 0 && rate > 0)) throw new ImproperMessageError('the belief of the concentration has no density')
    return shape / rate
  }
  const g = gaussianOf(a)
  return Math.exp(g.mean + g.variance / 2)
}

/** The belief of the probabilities, checked to have a density, and its `E log p_k`. */
function logMeansOf(a: Arg): readonly number[] {
  const eta = etaOf(a)
  if (!a.family.isProper(eta)) throw new ImproperMessageError('the belief of the probabilities has no density')
  return a.family.moments(eta)
}

/** The concentration `alpha` a coordinate of the quadrature stands for: a gamma's own value, or the exponential of a Gaussian on the log. */
const shapeOf = (a: Arg, coord: number): number => (a.family.kind === 'gamma' ? coord : Math.exp(coord))

/** log of the Dirichlet density's `alpha`-dependent part under `E log p`: `lgamma(sum kappa) - sum lgamma(kappa_k) + sum (kappa_k - 1) E log p_k`. */
function dirichletLog(
  w: readonly number[],
  o: readonly number[],
  logMeans: readonly number[],
): (alpha: number) => number {
  return (alpha) => {
    let total = 0
    let sum = 0
    for (let k = 0; k < w.length; k++) {
      const kappa = (w[k] as number) * alpha + (o[k] as number)
      total += kappa
      sum += (kappa - 1) * (logMeans[k] as number) - lgammaOf(kappa)
    }
    return lgammaOf(total) + sum
  }
}

interface Tilt {
  readonly logZ: number
  /** The tilted moments of the log concentration `t`: its mean and second moment, and the mean concentration `E e^t`. */
  readonly t: number
  readonly t2: number
  readonly alpha: number
  /** `E log p_k` under the mixture of conditional Dirichlets. */
  readonly logMeans: readonly number[]
}

/** The tilted moments of `t = log alpha` and of `log p` for a Dirichlet cavity `eta` (concentrations `eta + 1`) and the cavity of `alpha`. */
function tilt(c: OperatorContext): Tilt {
  const { w, o } = concentrations(c)
  const eta = etaOf(arg(c, 0))
  const logMode = c.params['log'] === true
  const cav = etaOf(arg(c, 1))
  const K = w.length
  if (eta.length !== K || eta.some((v) => !(v > -1) || !Number.isFinite(v)))
    throw new ImproperMessageError('dirichlet-concentration: the cavity of the probabilities has no density')
  let centre: number
  let half: number
  let logCavity: (t: number) => number
  if (logMode) {
    const g = gaussianOf(arg(c, 1))
    centre = g.mean
    half = SPAN * Math.sqrt(g.variance)
    logCavity = (t) => (-0.5 * (t - g.mean) ** 2) / g.variance + (g.mean * g.mean) / (2 * g.variance)
  } else {
    const { shape, rate } = gammaShapeRate(cav)
    if (!(shape > 0 && rate > 0 && Number.isFinite(shape) && Number.isFinite(rate)))
      throw new ImproperMessageError('dirichlet-concentration: the cavity of the concentration has no density')
    centre = digammaOf(shape) - Math.log(rate)
    half = SPAN * Math.sqrt(trigammaOf(shape))
    logCavity = (t) => shape * t - rate * Math.exp(t)
  }
  const lo = Math.max(centre - half, -40)
  const hi = Math.min(centre + half, 40)
  const step = (hi - lo) / (POINTS - 1)
  const logw = new Float64Array(POINTS)
  let top = Number.NEGATIVE_INFINITY
  for (let i = 0; i < POINTS; i++) {
    const t = lo + i * step
    const a = Math.exp(t)
    let total = 0
    let totalPosterior = 0
    let sumLog = 0
    for (let k = 0; k < K; k++) {
      const kappa = (w[k] as number) * a + (o[k] as number)
      const post = kappa + (eta[k] as number)
      total += kappa
      totalPosterior += post
      sumLog += lgammaOf(post) - lgammaOf(kappa)
    }
    const lw = logCavity(t) + lgammaOf(total) + sumLog - lgammaOf(totalPosterior)
    logw[i] = lw
    if (lw > top) top = lw
  }
  let z = 0
  let m1 = 0
  let m2 = 0
  let ea = 0
  const logMeans = new Array<number>(K).fill(0)
  for (let i = 0; i < POINTS; i++) {
    const t = lo + i * step
    const a = Math.exp(t)
    const weight = Math.exp((logw[i] as number) - top)
    z += weight
    m1 += weight * t
    m2 += weight * t * t
    ea += weight * a
    let totalPosterior = 0
    for (let k = 0; k < K; k++) totalPosterior += (w[k] as number) * a + (o[k] as number) + (eta[k] as number)
    const psiTotal = digammaOf(totalPosterior)
    for (let k = 0; k < K; k++)
      logMeans[k] =
        (logMeans[k] as number) +
        weight * (digammaOf((w[k] as number) * a + (o[k] as number) + (eta[k] as number)) - psiTotal)
  }
  return {
    logZ: top + Math.log(z * step),
    t: m1 / z,
    t2: m2 / z,
    alpha: ea / z,
    logMeans: logMeans.map((v) => v / z),
  }
}

/**
 * `p ~ Dirichlet(w alpha + o)` for a beta or Dirichlet variable `p` and a random positive concentration `alpha` (a gamma variable,
 * or with the parameter `log` a Gaussian variable on `log alpha`: a log-normal concentration); `weights` and `offsets` are the
 * known vectors of the concentration (`[1, 1]` and `[0, 0]` for `Beta(alpha, alpha)`, `[1, 0]` and `[0, 2]` for `Beta(alpha, 2)`).
 * Expectation propagation: a grid in `log alpha` for the tilted moments of the concentration (a gamma or a Gaussian site) and
 * of `E log p` (the Dirichlet or beta site, matched on those). Variational message passing: the message to `p` is the Dirichlet with the
 * mean concentration; the one to the concentration is non-conjugate (`lgamma` of it), the natural gradient of the expected log density.
 *
 * @concept factor-operators
 * @index structure A beta or Dirichlet variable with a random concentration: a hyperprior on the prior's own strength.
 * @example
 * ```ts
 * import { beta, betaEta, dirichletConcentration, gamma, gammaEta, messageArg } from 'omai-engine/message-passing'
 * const args = [messageArg(beta, betaEta(3, 2)), messageArg(gamma, gammaEta(2, 1))]
 * const m = dirichletConcentration.ep?.({ mode: 'ep', to: 1, args, params: { weights: [1, 1] }, power: 1 }).message
 * m?.map((v) => v.toFixed(3)) // => ['0.287', '-0.289']
 * ```
 */
export const dirichletConcentration: FactorOperator = {
  name: 'dirichlet-concentration',
  description: 'p ~ Dirichlet(w alpha + o) with the concentration alpha a gamma (or log-normal) variable',
  ports: [
    { name: 'p', accepts: ['beta', 'dirichlet'] },
    { name: 'alpha', accepts: ['gamma', 'gaussian'] },
  ],
  support: { ep: 'numeric', vmp: 'numeric' },
  readsTarget: () => true,
  // The mean field: to the probabilities the Dirichlet with the mean concentration (exact), to the concentration the natural
  // gradient of the expected log of the Dirichlet density, which has `lgamma` of the concentration in it (non-conjugate).
  vmp: (c): OperatorResult => {
    const { w, o } = concentrations(c)
    if (c.to === 0) {
      const mean = meanShape(arg(c, 1))
      return { message: w.map((v, k) => v * mean + (o[k] as number) - 1) }
    }
    const log = dirichletLog(w, o, logMeansOf(arg(c, 0)))
    const a = arg(c, 1)
    return { message: numericMessage(quadraturePorts([a]), 0, (x) => log(shapeOf(a, x[0] as number))) }
  },
  expectedLog: (c) => {
    const { w, o } = concentrations(c)
    const log = dirichletLog(w, o, logMeansOf(arg(c, 0)))
    const a = arg(c, 1)
    return numericExpectedLog(quadraturePorts([a]), (x) => log(shapeOf(a, x[0] as number)))
  },
  check: (params, families) => {
    const log = params['log'] === true
    const kind = families[1]?.spec.kind
    if (log && kind !== 'gaussian') return 'with `log` the concentration is a Gaussian variable on its log'
    if (!log && kind !== 'gamma')
      return 'the concentration is a gamma variable (or a Gaussian on its log with `log: true`)'
    return undefined
  },
  ep: (c): OperatorResult => {
    const t = tilt(c)
    if (c.to === 0) {
      const p = arg(c, 0)
      return { message: siteMessage(p.family, etaOf(p), t.logMeans) }
    }
    if (c.params['log'] === true) {
      const variance = t.t2 - t.t * t.t
      return { message: gaussianSiteMessage(etaOf(arg(c, 1)), t.t, variance) }
    }
    return { message: siteMessage(gamma, etaOf(arg(c, 1)) as Eta, [t.t, t.alpha]) }
  },
  logEvidence: (c) => {
    return evidenceOrNaN(() => {
      return tilt(c).logZ
    })
  },
}

/** The tilted statistics of a beta variable with two random shapes: the 2-d integral over (log a, log b) and `E log p`, `E log (1 - p)`. */
function twoShapes(c: OperatorContext): {
  readonly logMass: number
  readonly r: ReturnType<typeof integratePair>
  readonly logMeans: readonly [number, number]
} {
  const eta = etaOf(arg(c, 0))
  if (!(eta[0] !== undefined && eta[1] !== undefined && eta[0] > -1 && eta[1] > -1))
    throw new ImproperMessageError('beta-two-shapes: the cavity of the probability has no density')
  const [e1, e2] = [eta[0], eta[1]]
  // The cavity of p, p^e1 (1 - p)^e2, against the density Beta(a, b): B(a + e1, b + e2) / B(a, b), a function of the two log shapes.
  const column = (tb: number) => {
    const b = Math.exp(tb)
    const bad = !(b + e2 > 0)
    const cb = lgammaOf(b + e2) - lgammaOf(b)
    return (ta: number): number => {
      const a = Math.exp(ta)
      if (bad || !(a + e1 > 0)) return Number.NEGATIVE_INFINITY
      return c.power * (lgammaOf(a + e1) - lgammaOf(a) + cb - lgammaOf(a + b + e1 + e2) + lgammaOf(a + b))
    }
  }
  const r = integratePair([axisOf(arg(c, 1)), axisOf(arg(c, 2))], column)
  let m1 = 0
  let m2 = 0
  const n1 = r.nodes[1].length
  for (let i = 0; i < r.nodes[0].length; i++) {
    const a = Math.exp(r.nodes[0][i] as number)
    for (let j = 0; j < n1; j++) {
      const w = r.weights[i * n1 + j] as number
      if (!(w > 0)) continue
      const b = Math.exp(r.nodes[1][j] as number)
      const psi = digammaOf(a + b + e1 + e2)
      m1 += w * (digammaOf(a + e1) - psi)
      m2 += w * (digammaOf(b + e2) - psi)
    }
  }
  return { logMass: r.logMass, r, logMeans: [m1, m2] }
}

/** The expected log of the beta density as a function of the two shapes' coordinates, under the belief of `p`. */
function betaLog(c: OperatorContext): (x: readonly number[]) => number {
  const [lp, lq] = logMeansOf(arg(c, 0))
  const a = arg(c, 1)
  const b = arg(c, 2)
  return (x) => {
    const sa = shapeOf(a, x[0] as number)
    const sb = shapeOf(b, x[1] as number)
    return lgammaOf(sa + sb) - lgammaOf(sa) - lgammaOf(sb) + (sa - 1) * (lp as number) + (sb - 1) * (lq as number)
  }
}

/**
 * `p ~ Beta(a, b)` for a beta variable `p` whose two shapes are separate random positive values, each a gamma variable or a
 * Gaussian on its log (a log-normal): the tilted distribution of the two shapes is the integral of `B(a + e1, b + e2) / B(a, b)`
 * against their two cavities, on a two-dimensional grid in `(log a, log b)` (the one-dimensional integral of
 * `dirichlet-concentration`, in two variables); `p` is the mixture over the grid of the conditional betas, matched on
 * `E log p` and `E log (1 - p)`. Variational message passing: the message to `p` is the beta of the mean shapes (exact), the ones to the
 * shapes are non-conjugate (`lgamma` of them), the natural gradient of the expected log density by a two-dimensional rule.
 *
 * @concept factor-operators
 * @index structure A beta variable with two random shapes (gamma or log-normal each): a 2-d grid EP site.
 * @example
 * ```ts
 * import { beta, betaEta, betaTwoShapes, gaussian, gaussianEta, messageArg } from 'omai-engine/message-passing'
 * const args = [messageArg(beta, betaEta(3, 2)), messageArg(gaussian, gaussianEta(0.5, 0.5)), messageArg(gaussian, gaussianEta(0.5, 0.5))]
 * betaTwoShapes.ep?.({ mode: 'ep', to: 1, args, params: {}, power: 1 }).message.map((v) => v.toFixed(3)) // => ['0.558', '0.361']
 * ```
 */
export const betaTwoShapes: FactorOperator = {
  name: 'beta-two-shapes',
  description: 'p ~ Beta(a, b) with both shapes random positive variables (gamma, or Gaussian on the log)',
  ports: [
    { name: 'p', accepts: ['beta'] },
    { name: 'a', accepts: ['gamma', 'gaussian'] },
    { name: 'b', accepts: ['gamma', 'gaussian'] },
  ],
  support: { ep: 'numeric', vmp: 'numeric' },
  readsTarget: () => true,
  power: true,
  vmp: (c): OperatorResult => {
    if (c.to === 0) return { message: [meanShape(arg(c, 1)) - 1, meanShape(arg(c, 2)) - 1] }
    const log = betaLog(c)
    return { message: numericMessage(quadraturePorts([arg(c, 1), arg(c, 2)]), c.to - 1, log) }
  },
  expectedLog: (c) => numericExpectedLog(quadraturePorts([arg(c, 1), arg(c, 2)]), betaLog(c)),
  ep: (c): OperatorResult => {
    const t = twoShapes(c)
    if (c.to === 0) {
      const p = arg(c, 0)
      return { message: siteMessage(p.family, etaOf(p), t.logMeans) }
    }
    const k = c.to - 1
    const target = arg(c, c.to)
    if (target.family.kind === 'gamma')
      return { message: siteMessage(gamma, etaOf(target) as Eta, [t.r.mean[k] as number, t.r.meanExp[k] as number]) }
    return { message: gaussianSiteMessage(etaOf(target), t.r.mean[k] as number, t.r.variance[k] as number) }
  },
  logEvidence: (c) => {
    if (!allProper(c.args)) return Number.NaN
    return evidenceOrNaN(() => {
      return twoShapes(c).logMass + massOf(arg(c, 1)) + massOf(arg(c, 2))
    })
  },
}
