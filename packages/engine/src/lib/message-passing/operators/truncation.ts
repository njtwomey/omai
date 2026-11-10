/**
 * The operators that cut a Gaussian: a probit link, a comparison with a threshold, and an interval test, each from a
 * Gaussian variable to a Bernoulli outcome that may be observed. Moment matching is exact (the moments of a Gaussian
 * times a normal distribution function, or of a truncated Gaussian), so these are the closed forms behind probit
 * regression, TrueSkill and the step likelihoods of ordinal and censored models.
 */

import { logAddExp } from '../../base/index.ts'
import { logPhi, millsRatio, Phi, phi } from '../_scalar.ts'
import { bernoulli } from '../families.ts'
import { truncatedNormalMoments } from '../truncated.ts'
import type { Arg, FactorOperator, OperatorContext, OperatorResult, Port } from '../types.ts'
import { allProper, etaOf, gaussianOf, gaussianSiteMessage, massOf, numberParam, scalarOf } from './args.ts'

const arg = (c: OperatorContext, i: number): Arg => c.args[i] as Arg

const outcomePort: Port = { name: 'b', accepts: ['bernoulli'], constant: true }
const valuePort: Port = { name: 'x', accepts: ['gaussian'] }

/** The probability that the outcome is true under its incoming message, or 0 or 1 for an observed outcome. */
function outcomeProbability(a: Arg): number {
  if (a.value !== undefined) return scalarOf(a) > 0 ? 1 : 0
  return bernoulli.moments(etaOf(a))[0] ?? 0.5
}

/** A region of the line with the weight the factor gives it. */
interface Region {
  readonly lower: number
  readonly upper: number
  readonly weight: number
}

/**
 * The mass and the moments of N(mean, variance) weighted piecewise: the regions' weights times the Gaussian (the
 * tilted distribution of a step factor), combined in the log scale.
 */
function regionMoments(
  mean: number,
  variance: number,
  regions: readonly Region[],
): { readonly logMass: number; readonly mean: number; readonly variance: number } {
  const parts = regions
    .filter((r) => r.weight > 0 && r.lower < r.upper)
    .map((r) => ({ t: truncatedNormalMoments(mean, variance, r.lower, r.upper), log: Math.log(r.weight) }))
    .map((p) => ({ ...p, logMass: p.t.logMass + p.log }))
  const top = Math.max(...parts.map((p) => p.logMass))
  const weights = parts.map((p) => Math.exp(p.logMass - top))
  const total = weights.reduce((a, b) => a + b, 0)
  let m1 = 0
  let m2 = 0
  for (const [i, p] of parts.entries()) {
    const w = (weights[i] ?? 0) / total
    m1 += w * p.t.mean
    m2 += w * (p.t.variance + p.t.mean ** 2)
  }
  return { logMass: top + Math.log(total), mean: m1, variance: Math.max(m2 - m1 * m1, 1e-300) }
}

/** The weight of true outcomes inside a set of regions and of false ones outside: the step factor's weights. */
function stepRegions(
  pi: number,
  inside: readonly (readonly [number, number])[],
  outside: readonly (readonly [number, number])[],
): Region[] {
  return [
    ...inside.map(([lower, upper]) => ({ lower, upper, weight: pi })),
    ...outside.map(([lower, upper]) => ({ lower, upper, weight: 1 - pi })),
  ]
}

/** The EP operator of an outcome that is a step function of a Gaussian: b = [x in inside], false outside. */
function stepOperator(
  name: string,
  description: string,
  regions: (params: OperatorContext['params']) => {
    readonly inside: readonly (readonly [number, number])[]
    readonly outside: readonly (readonly [number, number])[]
  },
): FactorOperator {
  const massOfSet = (set: readonly (readonly [number, number])[], m: number, v: number): number =>
    set.length === 0
      ? Number.NEGATIVE_INFINITY
      : set
          .map(([lo, hi]) => truncatedNormalMoments(m, v, lo, hi).logMass)
          .reduce((a, b) => logAddExp(a, b), Number.NEGATIVE_INFINITY)
  return {
    name,
    description,
    ports: [outcomePort, valuePort],
    support: { ep: 'exact' },
    readsTarget: (to) => to === 1,
    ep: (c): OperatorResult => {
      const x = gaussianOf(arg(c, 1))
      const { inside, outside } = regions(c.params)
      if (c.to === 0) {
        const logIn = massOfSet(inside, x.mean, x.variance)
        const logOut = massOfSet(outside, x.mean, x.variance)
        return { message: [logIn - logOut] }
      }
      const pi = outcomeProbability(arg(c, 0))
      const t = regionMoments(x.mean, x.variance, stepRegions(pi, inside, outside))
      return { message: gaussianSiteMessage(etaOf(arg(c, 1)), t.mean, t.variance) }
    },
    logEvidence: (c) => {
      if (!allProper(c.args)) return Number.NaN
      const x = gaussianOf(arg(c, 1))
      const { inside, outside } = regions(c.params)
      const logIn = massOfSet(inside, x.mean, x.variance)
      const logOut = massOfSet(outside, x.mean, x.variance)
      const b = arg(c, 0)
      if (b.value !== undefined) return massOf(arg(c, 1)) + (scalarOf(b) > 0 ? logIn : logOut)
      const eta = etaOf(b)[0] ?? 0
      // Z = e^{A_x} (p_out + e^eta p_in), with the outcome's unnormalised message exp(eta b).
      return massOf(arg(c, 1)) + logAddExp(logOut, eta + logIn)
    },
  }
}

/**
 * b = [x > threshold] for a Gaussian x: the comparison with a constant, the factor of a win in TrueSkill and of a
 * censored observation. With b observed the message to x is the truncated Gaussian moment matching; with an incoming
 * Bernoulli message it is the matching of the two-region mixture; the message to b is the mass above the threshold.
 *
 * @concept factor-operators
 * @index structure The comparison b = [x > threshold] of a Gaussian: truncated-Gaussian moment matching.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, greaterThan, messageArg, valueArg } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussian, 1), messageArg(gaussian, gaussianEta(0, 1))]
 * greaterThan.ep?.({ mode: 'ep', to: 1, args, params: { threshold: 0 }, power: 1 }).message.map((v) => v.toFixed(4)) // => ['0.7979', '0.6366']
 * ```
 */
export const greaterThan: FactorOperator = stepOperator('greater-than', 'b = [x > threshold]', (params) => {
  const t = numberParam(params, 'threshold', 0)
  return { inside: [[t, Number.POSITIVE_INFINITY]], outside: [[Number.NEGATIVE_INFINITY, t]] }
})

/**
 * b = [lower < x < upper] for a Gaussian x: the interval test (the draw of TrueSkill, an interval-censored
 * observation). An observed true outcome truncates to the interval, an observed false one to the two tails.
 *
 * @concept factor-operators
 * @index structure The interval test b = [lower < x < upper] of a Gaussian.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, isBetween, messageArg, valueArg } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussian, 1), messageArg(gaussian, gaussianEta(0, 1))]
 * isBetween.ep?.({ mode: 'ep', to: 1, args, params: { lower: -1, upper: 1 }, power: 1 }).message.length // => 2
 * ```
 */
export const isBetween: FactorOperator = stepOperator('is-between', 'b = [lower < x < upper]', (params) => {
  const lo = numberParam(params, 'lower')
  const hi = numberParam(params, 'upper')
  return {
    inside: [[lo, hi]],
    outside: [
      [Number.NEGATIVE_INFINITY, lo],
      [hi, Number.POSITIVE_INFINITY],
    ],
  }
})

/** The site message of N(m, v) Phi(x / sigma) for an observed true outcome, in the form that does not subtract nearly equal precisions. */
function probitSite(m: number, v: number, sigma: number): [number, number] {
  const s2 = sigma * sigma + v
  const s = Math.sqrt(s2)
  const z = m / s
  const r = millsRatio(z)
  const k = r * (z + r)
  const rho = (v / s2) * k
  return [((m * k) / s2 + r / s) / (1 - rho), k / s2 / (1 - rho)]
}

/**
 * b | x ~ Bernoulli(Phi(x / noise)): the probit link (`noise` the standard deviation, default 1), the likelihood of
 * probit regression and of Gaussian-process classification. EP is exact in closed form: the mass of a Gaussian under
 * the normal distribution function is `Phi(m / sqrt(noise^2 + v))` and the first two moments follow from the same
 * ratio. An observed outcome gives the usual site update; an incoming Bernoulli message gives the matched mixture.
 *
 * @concept factor-operators
 * @index structure The probit link b | x ~ Bernoulli(Phi(x)): exact EP site updates.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, probit, valueArg } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussian, 1), messageArg(gaussian, gaussianEta(0, 1))]
 * probit.ep?.({ mode: 'ep', to: 1, args, params: {}, power: 1 }).message.map((v) => v.toFixed(4)) // => ['0.5642', '0.3183']
 * ```
 */
export const probit: FactorOperator = {
  name: 'probit',
  description: 'b | x ~ Bernoulli(Phi(x / noise))',
  ports: [outcomePort, valuePort],
  support: { ep: 'exact' },
  readsTarget: (to) => to === 1,
  ep: (c): OperatorResult => {
    const sigma = numberParam(c.params, 'noise', 1)
    const x = gaussianOf(arg(c, 1))
    const s = Math.sqrt(sigma * sigma + x.variance)
    const z = x.mean / s
    if (c.to === 0) return { message: [logPhi(z) - logPhi(-z)] }
    const pi = outcomeProbability(arg(c, 0))
    if (pi === 1) return { message: probitSite(x.mean, x.variance, sigma) }
    if (pi === 0) {
      const [h, j] = probitSite(-x.mean, x.variance, sigma)
      return { message: [-h, j] }
    }
    // The outcome is uncertain: the factor is (1 - pi) + (2 pi - 1) Phi(x / sigma), whose moments are closed forms.
    const w0 = 1 - pi
    const w1 = 2 * pi - 1
    const { mean: m, variance: v } = x
    const cdf = Phi(z)
    const dens = phi(z)
    const m0 = w0 + w1 * cdf
    const m1 = w0 * m + w1 * (m * cdf + (v / s) * dens)
    const m2 = w0 * (m * m + v) + w1 * ((v + m * m) * cdf + ((2 * m * v) / s - (m * v * v) / s ** 3) * dens)
    const mean = m1 / m0
    return { message: gaussianSiteMessage(etaOf(arg(c, 1)), mean, m2 / m0 - mean * mean) }
  },
  logEvidence: (c) => {
    if (!allProper(c.args)) return Number.NaN
    const sigma = numberParam(c.params, 'noise', 1)
    const x = gaussianOf(arg(c, 1))
    const z = x.mean / Math.sqrt(sigma * sigma + x.variance)
    const b = arg(c, 0)
    if (b.value !== undefined) return massOf(arg(c, 1)) + logPhi(scalarOf(b) > 0 ? z : -z)
    const eta = etaOf(b)[0] ?? 0
    return massOf(arg(c, 1)) + logAddExp(logPhi(-z), eta + logPhi(z))
  },
}
