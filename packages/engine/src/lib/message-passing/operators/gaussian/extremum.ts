/**
 * The larger and the smaller of two Gaussian variables: `y = max(a, b)` and `y = min(a, b)`, the factors of a
 * hinge, a rectifier, a best-of-two and a worst-case. The moments of the maximum of two independent Gaussians are
 * Clark's (1961) closed forms; the messages to the arguments, and to the output once it has been told something, are
 * moment matches of a one-dimensional integral whose integrand is a Gaussian times a normal distribution function.
 * Under variational message passing the output is defined by the inputs (the pushforward, Clark's moments), and the messages to the
 * inputs are non-conjugate, by quadrature.
 */

import { arg, isFlat } from '../../core/_messages.ts'
import {
  allProper,
  etaOf,
  gaussianMessage,
  gaussianOf,
  gaussianSiteMessage,
  logNormal,
  massOf,
} from '../../core/args.ts'
import { param } from '../../core/params.ts'
import type { Arg, FactorOperator, OperatorContext, OperatorResult, Port } from '../../core/types.ts'
import { tiltedGrid } from '../../numerics/_quadrature.ts'
import { Phi, phi } from '../../numerics/_scalar.ts'
import { definedFunctionMessage } from '../kernel/variational-numeric.ts'

type Moments = { readonly mean: number; readonly variance: number }

/**
 * The mean and variance of max(X, Y) for independent Gaussians (Clark 1961): with `theta^2 = v_x + v_y` and
 * `alpha = (m_x - m_y) / theta`, `E max = m_x Phi(alpha) + m_y Phi(-alpha) + theta phi(alpha)` and
 * `E max^2 = (m_x^2 + v_x) Phi(alpha) + (m_y^2 + v_y) Phi(-alpha) + (m_x + m_y) theta phi(alpha)`.
 *
 * @param x - Mean and variance of X.
 * @param y - Mean and variance of Y.
 * @returns The mean and variance of the maximum.
 * @concept factor-operators
 * @example
 * ```ts
 * import { clarkMax } from 'omai-engine/message-passing'
 * const m = clarkMax({ mean: 0, variance: 1 }, { mean: 0, variance: 1 })
 * [m.mean.toFixed(4), m.variance.toFixed(4)] // => ['0.5642', '0.6817']
 * ```
 */
export function clarkMax(x: Moments, y: Moments): Moments {
  const theta = Math.sqrt(x.variance + y.variance)
  if (!(theta > 0)) return x.mean >= y.mean ? x : y
  const alpha = (x.mean - y.mean) / theta
  const cdf = Phi(alpha)
  const dens = theta * phi(alpha)
  const mean = x.mean * cdf + y.mean * (1 - cdf) + dens
  const second = (x.mean ** 2 + x.variance) * cdf + (y.mean ** 2 + y.variance) * (1 - cdf) + (x.mean + y.mean) * dens
  return { mean, variance: Math.max(second - mean * mean, 0) }
}

/** The density of the maximum of independent N(a), N(b) at y, up to nothing: N_a(y) Phi_b(y) + N_b(y) Phi_a(y). */
function maxDensity(a: Moments, b: Moments): (y: number) => number {
  return (y) =>
    Math.exp(logNormal(y, a.mean, a.variance)) * Phi((y - b.mean) / Math.sqrt(b.variance)) +
    Math.exp(logNormal(y, b.mean, b.variance)) * Phi((y - a.mean) / Math.sqrt(a.variance))
}

/**
 * The weight a value of `self` carries for the maximum of (self, other) landing under the Gaussian N(y): the integral over
 * `other` of N(other) N_y(max(self, other)), which is `N_y(self) Phi((self - m_o) / s_o)` for other below self plus the
 * mass of the product of the two Gaussians above self.
 */
function maxWeight(other: Moments, y: Moments): (self: number) => number {
  const v = 1 / (1 / other.variance + 1 / y.variance)
  const mu = v * (other.mean / other.variance + y.mean / y.variance)
  const sd = Math.sqrt(v)
  const overlap = Math.exp(logNormal(other.mean, y.mean, other.variance + y.variance))
  return (self) =>
    Math.exp(logNormal(self, y.mean, y.variance)) * Phi((self - other.mean) / Math.sqrt(other.variance)) +
    overlap * (1 - Phi((self - mu) / sd))
}

/** The message of `y = max(a, b)` (ports y, a, b) from the arguments as they are. */
function maxMessage(c: OperatorContext): OperatorResult {
  const a = gaussianOf(arg(c, 1))
  const b = gaussianOf(arg(c, 2))
  const incoming = etaOf(arg(c, 0))
  if (c.to === 0) {
    // With nothing known about y the message is Clark's match of the pushforward; otherwise the pushforward is weighted by
    // the incoming message of y first, as expectation propagation asks.
    if (isFlat(incoming)) {
      const m = clarkMax(a, b)
      return { message: gaussianMessage(m.mean, m.variance) }
    }
    const y = gaussianOf(arg(c, 0))
    const t = tiltedGrid(y.mean, y.variance, maxDensity(a, b))
    return { message: gaussianSiteMessage(incoming, t.mean, t.variance) }
  }
  // Nothing is known about y: the constraint says nothing about a or b.
  if (isFlat(incoming)) return { message: [0, 0] }
  const y = gaussianOf(arg(c, 0))
  const [self, other] = c.to === 1 ? [a, b] : [b, a]
  const t = tiltedGrid(self.mean, self.variance, maxWeight(other, y))
  return { message: gaussianSiteMessage(etaOf(arg(c, c.to)), t.mean, t.variance) }
}

/** The mass of the factor against the incoming messages. */
function maxEvidence(c: OperatorContext): number {
  if (!allProper(c.args)) return Number.NaN
  const a = gaussianOf(arg(c, 1))
  const b = gaussianOf(arg(c, 2))
  const y = gaussianOf(arg(c, 0))
  const mass = tiltedGrid(a.mean, a.variance, maxWeight(b, y)).mass
  return massOf(arg(c, 0)) + massOf(arg(c, 1)) + massOf(arg(c, 2)) + Math.log(mass)
}

const ports: readonly Port[] = [
  { name: 'y', accepts: ['gaussian'] },
  { name: 'a', accepts: ['gaussian'] },
  { name: 'b', accepts: ['gaussian'] },
]

/**
 * y = max(a, b) for Gaussian a and b. The message to y is Clark's moment match of the maximum (exact mean and variance of
 * the pushforward) while y has been told nothing, and the match of the maximum's density weighted by y's cavity after;
 * the messages to a and b match the tilted marginal, a one-dimensional integral. The mean and variance of the match are
 * checked against Clark's closed forms and against integration of the maximum's density.
 *
 * @concept factor-operators
 * @index structure The maximum of two Gaussian variables: Clark's moments, moment matching.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, maximum, messageArg } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussian.flat), messageArg(gaussian, gaussianEta(0, 1)), messageArg(gaussian, gaussianEta(0, 1))]
 * maximum.ep?.({ mode: 'ep', to: 0, args, params: {}, power: 1 }).message.map((v) => v.toFixed(4)) // => ['0.8276', '1.4669']
 * ```
 */
export const maximum: FactorOperator = {
  name: 'max',
  description: 'y = max(a, b) for Gaussians',
  ports,
  params: {},
  support: { ep: 'numeric', vmp: 'numeric' },
  readsTarget: (to) => to !== 0,
  // Deterministic: under the mean field `y` is the pushforward (Clark's moments), and the message to an input is the
  // non-conjugate message of the expected log of the output's cavity over the other input.
  defines: 0,
  vmp: (c): OperatorResult => {
    if (c.to === 0) {
      const m = clarkMax(gaussianOf(arg(c, 1)), gaussianOf(arg(c, 2)))
      return { message: gaussianMessage(m.mean, Math.max(m.variance, 1e-300)) }
    }
    return { message: definedFunctionMessage(c.args, c.to, (x) => Math.max(x[0] as number, x[1] as number)) }
  },
  expectedLog: () => 0,
  ep: maxMessage,
  logEvidence: maxEvidence,
}

const negate = (a: Arg): Arg => (a.eta === undefined ? a : { ...a, eta: [-(a.eta[0] ?? 0), a.eta[1] ?? 0] })

/**
 * y = min(a, b) for Gaussian a and b: the maximum of the negatives, `y = -max(-a, -b)`, so every message is that of
 * {@link maximum} on the negated arguments, negated back.
 *
 * @concept factor-operators
 * @index structure The minimum of two Gaussian variables: the maximum of the negatives.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, minimum } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussian.flat), messageArg(gaussian, gaussianEta(0, 1)), messageArg(gaussian, gaussianEta(0, 1))]
 * minimum.ep?.({ mode: 'ep', to: 0, args, params: {}, power: 1 }).message.map((v) => v.toFixed(4)) // => ['-0.8276', '1.4669']
 * ```
 */
export const minimum: FactorOperator = {
  name: 'min',
  description: 'y = min(a, b) for Gaussians',
  ports,
  params: {},
  support: { ep: 'numeric', vmp: 'numeric' },
  readsTarget: (to) => to !== 0,
  defines: 0,
  vmp: (c): OperatorResult => {
    if (c.to === 0) {
      const a = gaussianOf(arg(c, 1))
      const b = gaussianOf(arg(c, 2))
      const m = clarkMax({ mean: -a.mean, variance: a.variance }, { mean: -b.mean, variance: b.variance })
      return { message: gaussianMessage(-m.mean, Math.max(m.variance, 1e-300)) }
    }
    return { message: definedFunctionMessage(c.args, c.to, (x) => Math.min(x[0] as number, x[1] as number)) }
  },
  expectedLog: () => 0,
  ep: (c): OperatorResult => {
    const { message } = maxMessage({ ...c, args: c.args.map(negate) })
    return { message: [-(message[0] ?? 0), message[1] ?? 0] }
  },
  logEvidence: (c) => maxEvidence({ ...c, args: c.args.map(negate) }),
}
