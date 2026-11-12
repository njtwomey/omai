/**
 * The closed forms of the conjugate and prior setups as reference marginals: a gamma, an inverse gamma, a Student t and a beta
 * (moments, density, the range to draw), the normal-inverse-gamma posterior of a Gaussian's mean and variance with their joint, and
 * the marginal of a conjugate posterior of the library's own table (`normalNormal`, `betaBernoulli`, `gammaPoisson` in `exact.ts`).
 */

import { item } from '../../../tensor/index.ts'
import type { ExactPosterior } from '../exact.ts'
import type { ExactMarginal, RefJoint } from './marginal.ts'
import { lg, normalPdf, sum } from './special.ts'

/**
 * The marginal of the closed-form posterior of a scalar parameter, from the library's conjugate table (`exact.ts`): the
 * posterior distribution's moments and density, zero outside the open support.
 *
 * @param exact - The conjugate posterior (`normalNormal`, `betaBernoulli` or `gammaPoisson` of this folder).
 * @param support - Where the parameter lives: the density is zero outside (0, 1) for `unit` and outside (0, inf) for `positive`.
 * @param range - The interval the curve is drawn on (default the mean plus or minus eight standard deviations).
 * @returns The marginal.
 * @example
 * ```ts
 * import { betaBernoulli, conjugateMarginal } from 'omai-engine/probprog/validation'
 * conjugateMarginal(betaBernoulli({ a: 1, b: 1, y: [1, 1, 0] }), 'unit').mean // => 0.6
 * ```
 */
export function conjugateMarginal(
  exact: ExactPosterior,
  support: 'real' | 'positive' | 'unit',
  range?: readonly [number, number],
): ExactMarginal {
  const d = exact.posterior
  const mean = item(d.mean())
  const sd = item(d.stddev())
  return {
    mean,
    sd,
    pdf: (x) =>
      (support === 'unit' && (x <= 0 || x >= 1)) || (support === 'positive' && x <= 0) ? 0 : item(d.prob(x)),
    ...(range === undefined ? {} : { range }),
  }
}

const gammaPdf = (x: number, shape: number, rate: number): number =>
  x <= 0 ? 0 : Math.exp(shape * Math.log(rate) - lg(shape) + (shape - 1) * Math.log(x) - rate * x)

/**
 * The gamma distribution (shape and rate) as a reference marginal.
 *
 * @param shape - The shape.
 * @param rate - The rate.
 * @returns The marginal.
 * @example
 * ```ts
 * import { gammaMarginal } from 'omai-engine/probprog/validation'
 * gammaMarginal(2, 1).mean // => 2
 * ```
 */
export function gammaMarginal(shape: number, rate: number): ExactMarginal {
  const mean = shape / rate
  const sd = Math.sqrt(shape) / rate
  return { mean, sd, pdf: (x) => gammaPdf(x, shape, rate), range: [Math.max(0, mean - 6 * sd), mean + 8 * sd] }
}

/**
 * The inverse gamma distribution (shape `a`, scale `b`) as a reference marginal.
 *
 * @param a - The shape (above 2, for a variance).
 * @param b - The scale.
 * @returns The marginal.
 * @example
 * ```ts
 * import { inverseGammaMarginal } from 'omai-engine/probprog/validation'
 * inverseGammaMarginal(3, 2).mean // => 1
 * ```
 */
export function inverseGammaMarginal(a: number, b: number): ExactMarginal {
  const mean = b / (a - 1)
  const sd = Math.sqrt((b * b) / ((a - 1) ** 2 * (a - 2)))
  return {
    mean,
    sd,
    pdf: (x) => (x <= 0 ? 0 : Math.exp(a * Math.log(b) - lg(a) - (a + 1) * Math.log(x) - b / x)),
    range: [Math.max(1e-6, mean - 3 * sd), mean + 8 * sd],
  }
}

/**
 * A Student t with `nu` degrees of freedom around `loc` with the squared scale `s2`, as a reference marginal.
 *
 * @param nu - The degrees of freedom (above 2).
 * @param loc - The centre.
 * @param s2 - The squared scale.
 * @returns The marginal.
 * @example
 * ```ts
 * import { studentMarginal } from 'omai-engine/probprog/validation'
 * studentMarginal(5, 1, 2).mean // => 1
 * ```
 */
export function studentMarginal(nu: number, loc: number, s2: number): ExactMarginal {
  const s = Math.sqrt(s2)
  const logNorm = lg((nu + 1) / 2) - lg(nu / 2) - 0.5 * Math.log(nu * Math.PI) - Math.log(s)
  const sd = Math.sqrt((s2 * nu) / (nu - 2))
  return {
    mean: loc,
    sd,
    pdf: (x) => Math.exp(logNorm - ((nu + 1) / 2) * Math.log(1 + ((x - loc) / s) ** 2 / nu)),
    range: [loc - 7 * sd, loc + 7 * sd],
  }
}

/**
 * A beta distribution as a reference marginal.
 *
 * @param a - The first shape.
 * @param b - The second shape.
 * @returns The marginal.
 * @example
 * ```ts
 * import { betaMarginal } from 'omai-engine/probprog/validation'
 * betaMarginal(1, 1).mean // => 0.5
 * ```
 */
export function betaMarginal(a: number, b: number): ExactMarginal {
  const mean = a / (a + b)
  const sd = Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))
  return { mean, sd, pdf: (x) => betaPdf(x, a, b), range: [0, 1] }
}
const betaPdf = (x: number, a: number, b: number): number =>
  x <= 0 || x >= 1 ? 0 : Math.exp(lg(a + b) - lg(a) - lg(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x))

/**
 * The exact posterior of a Gaussian's mean and variance under the conjugate normal-inverse-gamma prior (`sigma2` inverse gamma
 * with shape `a` and scale `b`, `mu | sigma2` normal around `m0` with variance `sigma2 / kappa`): a Student t for the mean, an
 * inverse gamma for the variance and their joint on a grid, as the exact references of an atom.
 *
 * @param y - The observations.
 * @param m0 - The prior mean of the mean.
 * @param kappa - The prior observations behind the mean.
 * @param a - The inverse gamma's shape.
 * @param b - The inverse gamma's scale.
 * @returns The marginals of the mean and the variance, and the joint for contours.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { normalInverseGammaPosterior } from 'omai-engine/probprog/validation'
 * normalInverseGammaPosterior([1, 2, 3], 0, 1, 3, 2).mu.mean // => 1.5
 * ```
 */
export function normalInverseGammaPosterior(
  y: readonly number[],
  m0: number,
  kappa: number,
  a: number,
  b: number,
): { readonly mu: ExactMarginal; readonly sigma2: ExactMarginal; readonly joint: RefJoint } {
  const n = y.length
  const ybar = sum(y) / n
  const ss = sum(y.map((v) => (v - ybar) ** 2))
  const kn = kappa + n
  const mn = (kappa * m0 + n * ybar) / kn
  const an = a + n / 2
  const bn = b + ss / 2 + (kappa * n * (ybar - m0) ** 2) / (2 * kn)
  const mu = studentMarginal(2 * an, mn, bn / (an * kn))
  const sigma2 = inverseGammaMarginal(an, bn)
  // The joint: N(mu; mn, sigma2 / kn) times the inverse gamma, on a grid for the contours.
  const xs = Array.from({ length: 80 }, (_, i) => mu.mean - 5 * mu.sd + (i * 10 * mu.sd) / 79)
  const lo = Math.max(1e-3, sigma2.range?.[0] ?? 0)
  const hi = sigma2.range?.[1] ?? sigma2.mean * 4
  const ys = Array.from({ length: 80 }, (_, i) => lo + (i * (hi - lo)) / 79)
  const joint: RefJoint = {
    a: 'mu',
    b: 'sigma2',
    x: xs,
    y: ys,
    p: xs.map((x) => ys.map((s2) => normalPdf(x, mn, Math.sqrt(s2 / kn)) * sigma2.pdf(s2))),
  }
  return { mu, sigma2, joint }
}
