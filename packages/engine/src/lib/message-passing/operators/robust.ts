/**
 * The robust sites: factors on one Gaussian variable (a regression's linear predictor, or a coefficient) whose tilted
 * distribution against a Gaussian cavity is easy. `mixture-site` is a mixture of Gaussians in the variable, `f(x) = sum
 * pi_i N(x; mu_i, s_i^2)`: the spike and slab prior (`mu = 0`), and a likelihood that is a Gaussian mixture of inliers and
 * outliers (`mu = y`, the clutter problem's observation); a Gaussian cavity times it is a mixture of Gaussians, so the
 * moments are exact and need no quadrature. `student-t-site` and `laplace-site` are heavy-tailed likelihoods of an observed
 * outcome (`f(x) = t_nu(y; x, scale)`, `f(x) = Laplace(y; x, scale)`), whose tilted moments are found on a panelled grid laid
 * where the factor lives (a cavity that is much wider than the likelihood is common in the first sweeps). Each also comes
 * lifted to a regression row on a weight vector (`mixture-row`, `student-t-row`, `laplace-row`) by `glmRow`, so a model
 * `y ~ studentT(df, X w, s)` is one rank-one factor per row on the packed weights.
 */

import { gaussianGridOn } from '../_quadrature.ts'
import { lgammaOf } from '../_scalar.ts'
import type { FactorOperator, OperatorContext, OperatorResult, Params } from '../types.ts'
import { allProper, etaOf, gaussianOf, gaussianSiteMessage, massOf, numberParam, vectorParam } from './args.ts'
import { glmRow } from './rows.ts'

/** The mixture's components from the parameters, with the weights normalised. */
function components(params: Params): {
  readonly pi: number[]
  readonly mu: readonly number[]
  readonly s2: number[]
} {
  const weights = vectorParam(params, 'weights')
  const mu = vectorParam(params, 'means')
  const sds = vectorParam(params, 'sds')
  const total = weights.reduce((a, b) => a + b, 0)
  return { pi: weights.map((w) => w / total), mu, s2: sds.map((s) => s * s) }
}

/** The Gaussian cavity of the variable and the tilted moments and log mass of its product with the mixture. */
function mixtureTilted(c: OperatorContext): { mean: number; variance: number; logMass: number } {
  const x = gaussianOf(c.args[0] as never)
  const { pi, mu, s2 } = components(c.params)
  const logZ = pi.map(
    (p, i) =>
      Math.log(p) -
      0.5 *
        (Math.log(2 * Math.PI * (x.variance + (s2[i] ?? 0))) +
          (x.mean - (mu[i] ?? 0)) ** 2 / (x.variance + (s2[i] ?? 0))),
  )
  const top = Math.max(...logZ)
  const logMass = top + Math.log(logZ.reduce((a, l) => a + Math.exp(l - top), 0))
  let mean = 0
  let second = 0
  for (const [i, l] of logZ.entries()) {
    const r = Math.exp(l - logMass)
    const v = 1 / (1 / x.variance + 1 / (s2[i] ?? 1))
    const m = v * (x.mean / x.variance + (mu[i] ?? 0) / (s2[i] ?? 1))
    mean += r * m
    second += r * (v + m * m)
  }
  return { mean, variance: Math.max(second - mean * mean, 1e-300), logMass }
}

/**
 * A Gaussian mixture in one variable, `f(x) = sum_i pi_i N(x; mu_i, s_i^2)`: the spike and slab prior of a coefficient
 * (`means` zero, a narrow and a wide `sds`) or a likelihood that mixes an inlier and an outlier noise level around the
 * outcome (`means` both `y`). A Gaussian cavity times a mixture of Gaussians is a mixture of Gaussians, so EP is exact: the
 * component responsibilities are the normalised masses `pi_i N(m; mu_i, v + s_i^2)` and the moments those of the
 * mixture of the component posteriors (Hernandez-Lobato, Hernandez-Lobato and Dupont's spike and slab EP).
 *
 * @concept factor-operators
 * @index structure A mixture of Gaussians in one variable (spike and slab, outlier mixture): exact EP.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, mixtureSite } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussianEta(1, 1))]
 * const params = { weights: [0.5, 0.5], means: [0, 0], sds: [3, 0.1] }
 * mixtureSite.ep?.({ mode: 'ep', to: 0, args, params, power: 1 }).message.map((v) => v.toFixed(3)) // => ['-0.365', '1.081']
 * ```
 */
export const mixtureSite: FactorOperator = {
  name: 'mixture-site',
  description: 'f(x) = sum_i pi_i N(x; mu_i, s_i^2): a mixture of Gaussians in one variable',
  ports: [{ name: 'x', accepts: ['gaussian'] }],
  support: { ep: 'exact' },
  readsTarget: true,
  ep: (c): OperatorResult => {
    const t = mixtureTilted(c)
    return { message: gaussianSiteMessage(etaOf(c.args[0] as never), t.mean, t.variance) }
  },
  logEvidence: (c) => (allProper(c.args) ? massOf(c.args[0] as never) + mixtureTilted(c).logMass : Number.NaN),
}

/** How wide a window around the outcome (in scales) a heavy-tailed likelihood is integrated over. */
const REACH = 120

/**
 * The tilted log mass, mean and variance of the Gaussian cavity times a likelihood of the variable that is centred on `y`
 * with the given scale: integrated on a panelled Gauss-Legendre grid over the part of the cavity's window that the
 * likelihood occupies, split at `y` (where a Laplace density has its kink).
 */
function tiltedWindow(
  cavity: { readonly mean: number; readonly variance: number },
  y: number,
  scale: number,
  logf: (x: number) => number,
): { logMass: number; mean: number; variance: number } {
  const sd = Math.sqrt(cavity.variance)
  let lo = cavity.mean - 9 * sd
  let hi = cavity.mean + 9 * sd
  lo = Math.max(lo, y - REACH * scale)
  hi = Math.min(hi, y + REACH * scale)
  if (!(hi > lo)) {
    // The cavity is far from the outcome: take the part of its window nearest to it.
    lo = cavity.mean - 9 * sd
    hi = cavity.mean + 9 * sd
  }
  const pieces: [number, number][] =
    y > lo && y < hi
      ? [
          [lo, y],
          [y, hi],
        ]
      : [[lo, hi]]
  const xs: number[] = []
  const lw: number[] = []
  for (const [a, b] of pieces) {
    const g = gaussianGridOn(cavity.mean, cavity.variance, a, b, 40, 8)
    for (let i = 0; i < g.x.length; i++) {
      xs.push(g.x[i] ?? 0)
      lw.push(Math.log(g.w[i] ?? 0) + logf(g.x[i] ?? 0))
    }
  }
  const top = Math.max(...lw)
  let m0 = 0
  let m1 = 0
  for (const [i, v] of lw.entries()) {
    const e = Math.exp(v - top)
    m0 += e
    m1 += e * (xs[i] ?? 0)
  }
  const mean = m1 / m0
  let m2 = 0
  for (const [i, v] of lw.entries()) m2 += Math.exp(v - top) * ((xs[i] ?? 0) - mean) ** 2
  return { logMass: top + Math.log(m0), mean, variance: m2 / m0 }
}

/** log of the Student-t density with `df` degrees of freedom, centred at `x`, evaluated at the outcome. */
const logStudentT = (y: number, x: number, scale: number, df: number): number =>
  lgammaOf((df + 1) / 2) -
  lgammaOf(df / 2) -
  0.5 * Math.log(df * Math.PI) -
  Math.log(scale) -
  ((df + 1) / 2) * Math.log(1 + ((y - x) / scale) ** 2 / df)

/** log of the Laplace density with scale `b`, centred at `x`, evaluated at the outcome. */
const logLaplace = (y: number, x: number, scale: number): number => -Math.log(2 * scale) - Math.abs(y - x) / scale

/** A heavy-tailed likelihood site of a Gaussian predictor given by its log density. */
function heavySite(
  name: string,
  description: string,
  log: (c: OperatorContext) => (x: number) => number,
  scale: (c: OperatorContext) => number,
): FactorOperator {
  const tilted = (c: OperatorContext): { logMass: number; mean: number; variance: number } => {
    const y = numberParam(c.params, 'y')
    const logf = log(c)
    return tiltedWindow(gaussianOf(c.args[0] as never), y, scale(c), (x) => c.power * logf(x))
  }
  return {
    name,
    description,
    ports: [{ name: 'x', accepts: ['gaussian'] }],
    support: { ep: 'numeric' },
    readsTarget: true,
    power: true,
    ep: (c): OperatorResult => {
      const t = tilted(c)
      const site = gaussianSiteMessage(etaOf(c.args[0] as never), t.mean, t.variance)
      return { message: c.power === 1 ? site : site.map((v) => v / c.power) }
    },
    logEvidence: (c) => (allProper(c.args) ? massOf(c.args[0] as never) + tilted(c).logMass : Number.NaN),
  }
}

/**
 * A Student-t likelihood of a Gaussian predictor, `y | x ~ t_df(x, scale)` with the outcome `y`, the degrees of freedom `df`
 * and the `scale` known (parameters): the robust regression likelihood. EP matches the moments of the Gaussian cavity times
 * the t density by quadrature on a grid laid where the density lives.
 *
 * @concept factor-operators
 * @index structure A Student-t likelihood of a Gaussian predictor: quadrature EP.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, studentTSite } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussianEta(0, 0.25))]
 * studentTSite.ep?.({ mode: 'ep', to: 0, args, params: { y: 1, scale: 1, df: 4 }, power: 1 }).message.map((v) => v.toFixed(2)) // => ['0.93', '0.75']
 * ```
 */
export const studentTSite: FactorOperator = heavySite(
  'student-t-site',
  'y | x ~ Student t(df, x, scale)',
  (c) => {
    const y = numberParam(c.params, 'y')
    const scale = numberParam(c.params, 'scale')
    const df = numberParam(c.params, 'df')
    return (x) => logStudentT(y, x, scale, df)
  },
  (c) => numberParam(c.params, 'scale'),
)

/**
 * A Laplace likelihood of a Gaussian predictor, `y | x ~ Laplace(x, scale)`: the absolute-error (median-like) robust
 * regression likelihood, with the moments of its product with the Gaussian cavity by quadrature (split at the kink).
 *
 * @concept factor-operators
 * @index structure A Laplace likelihood of a Gaussian predictor: quadrature EP.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, laplaceSite, messageArg } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussianEta(0, 1))]
 * laplaceSite.ep?.({ mode: 'ep', to: 0, args, params: { y: 1, scale: 1 }, power: 1 }).message.map((v) => v.toFixed(2)) // => ['0.89', '0.79']
 * ```
 */
export const laplaceSite: FactorOperator = heavySite(
  'laplace-site',
  'y | x ~ Laplace(x, scale)',
  (c) => {
    const y = numberParam(c.params, 'y')
    const scale = numberParam(c.params, 'scale')
    return (x) => logLaplace(y, x, scale)
  },
  (c) => numberParam(c.params, 'scale'),
)

/**
 * One row of a regression with a Gaussian mixture on a weight vector, `a . w ~ sum_i pi_i N(mu_i, s_i^2)`: a coefficient with a
 * spike and slab prior (`a` a unit vector, `means` zero) or a row of data with an inlier and outlier noise level (`means`
 * the outcome).
 *
 * @concept factor-operators
 * @index structure A Gaussian mixture on a linear function of the weights (spike and slab row): exact EP.
 */
export const mixtureRow: FactorOperator = glmRow('mixture-row', mixtureSite, 'a . w ~ a mixture of Gaussians')

/**
 * One row of Student-t regression, `y | a . w ~ t_df(a . w, scale)`, on a Gaussian weight vector.
 *
 * @concept factor-operators
 * @index structure One Student-t regression row on a Gaussian weight vector: quadrature EP.
 */
export const studentTRow: FactorOperator = glmRow(
  'student-t-row',
  studentTSite,
  'y | a . w ~ Student t(df, a . w, scale)',
)

/**
 * One row of Laplace regression, `y | a . w ~ Laplace(a . w, scale)`, on a Gaussian weight vector.
 *
 * @concept factor-operators
 * @index structure One Laplace regression row on a Gaussian weight vector: quadrature EP.
 */
export const laplaceRow: FactorOperator = glmRow('laplace-row', laplaceSite, 'y | a . w ~ Laplace(a . w, scale)')
