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

import { OmaiError } from '../../base/index.ts'
import { axisOf, integratePair } from '../_grid.ts'
import { digammaOf, lgammaOf, trigammaOf } from '../_scalar.ts'
import { gamma, gammaShapeRate } from '../families.ts'
import type { Arg, Eta, FactorOperator, OperatorContext, OperatorResult } from '../types.ts'
import {
  allProper,
  etaOf,
  gaussianOf,
  gaussianSiteMessage,
  ImproperMessageError,
  massOf,
  siteMessage,
  vectorParam,
} from './args.ts'

const arg = (c: OperatorContext, i: number): Arg => c.args[i] as Arg

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
 * of `E log p` (the Dirichlet or beta site, matched on those). Variational message passing is not provided.
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
  support: { ep: 'numeric' },
  readsTarget: () => true,
  noRule: { vmp: 'the expected log of the Dirichlet normaliser has no conjugate form in the concentration' },
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
    try {
      return tilt(c).logZ
    } catch (e) {
      // At the start of a run the messages have no density yet: no evidence to report.
      if (e instanceof ImproperMessageError) return Number.NaN
      throw e
    }
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

/**
 * `p ~ Beta(a, b)` for a beta variable `p` whose two shapes are separate random positive values, each a gamma variable or a
 * Gaussian on its log (a log-normal): the tilted distribution of the two shapes is the integral of `B(a + e1, b + e2) / B(a, b)`
 * against their two cavities, on a two-dimensional grid in `(log a, log b)` (the one-dimensional integral of
 * `dirichlet-concentration`, in two variables); `p` is the mixture over the grid of the conditional betas, matched on
 * `E log p` and `E log (1 - p)`. Variational message passing is not provided.
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
  support: { ep: 'numeric' },
  readsTarget: () => true,
  power: true,
  noRule: { vmp: 'the expected log of the beta normaliser has no conjugate form in the shapes' },
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
    try {
      return twoShapes(c).logMass + massOf(arg(c, 1)) + massOf(arg(c, 2))
    } catch (e) {
      if (e instanceof ImproperMessageError) return Number.NaN
      throw e
    }
  },
}
