/**
 * The ordinal and quantile atoms: an ordered outcome (four categories) through the cumulative probit of a linear predictor with
 * the cut points fixed (the `ordinal-probit` row of the likelihood registry), and quantile regression, a line fitted to the 80th
 * percentile of the outcome with the asymmetric Laplace (the `asymmetric-laplace` row). Each has a basic page on a handful of fixed
 * observations and a twin that learns the intercept and slope from data simulated from stated values.
 */

import * as d from '../probprog/model/dists/index.ts'
import { defineModel } from '../probprog/model/index.ts'
import { foldIn, key, normal, uniform } from '../random/index.ts'
import { dict, real } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import type { Config } from '../trace/index.ts'
import {
  type AtomFigure,
  type AtomProblem,
  type AtomRecoverySpec,
  defineAtom,
  type EpAtom,
  type FigureCurve,
} from './ep-atoms.ts'
import { simulationKnobs } from './ep-atoms-extended-conjugate.ts'
import { exampleNumber } from './example-method.ts'

const CUTS = [-0.6, 0.5, 1.4]
const CURVE_X = Array.from({ length: 41 }, (_, i) => -2 + i / 10)

/** The standard normal distribution function in plain arithmetic (Abramowitz and Stegun 26.2.17). */
function phi(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x))
  const poly = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
  const tail = Math.exp(-0.5 * x * x) * 0.3989422804014327 * poly
  return x >= 0 ? 1 - tail : tail
}

/** The expected category at a predictor value, for the figure. */
function expectedCategory(u: number): number {
  const upper = [...CUTS.map((c) => phi(c - u)), 1]
  return upper.reduce((s, p, k) => s + k * (p - (k === 0 ? 0 : (upper[k - 1] as number))), 0)
}

const recoveryOf = (reason: string, seeds = 20): AtomRecoverySpec => ({
  seeds,
  coverage: 0.7,
  zrms: 1.6,
  reason,
})

const OX = [-1.9, -1.5, -1.1, -0.8, -0.4, -0.1, 0.3, 0.6, 0.9, 1.3, 1.6, 1.9]
const OY = [0, 0, 1, 0, 1, 1, 1, 2, 2, 3, 2, 3]

function ordinalFigure(x: readonly number[], y: readonly number[], truth?: readonly [number, number]): AtomFigure {
  const line = (a: number, b: number, name: string, colour?: string): FigureCurve => ({
    name,
    ...(colour === undefined ? {} : { colour }),
    x: CURVE_X,
    y: CURVE_X.map((v) => expectedCategory(a + b * v)),
  })
  return {
    caption: `The ordered outcomes (0 to 3) at each covariate value, the cut points (${CUTS.join(', ')}) fixed, and the expected category at the mean of the belief${truth === undefined ? '' : ' (the true curve in green)'}.`,
    xLabel: 'x',
    yLabel: 'category',
    xLimits: [-2, 2],
    yLimits: [-0.3, 3.3],
    points: [{ name: 'observations', x: [...x], y: [...y] }],
    ...(truth === undefined ? {} : { curves: [line(truth[0], truth[1], 'true curve', 'tab2')] }),
    answer: (m) => [line(m['a'] ?? Number.NaN, m['b'] ?? Number.NaN, 'fitted curve')],
  }
}

function ordinalProblem(k: Config): AtomProblem {
  const s = exampleNumber(k, 'priorScale')
  // region ordinal-probit
  const definition = defineModel('ordinal-probit', (m) => {
    const row = m.range('row')
    // the intercept and the slope of the predictor, the cut points are known
    const a = m.latent('a', d.normal(0, s))
    const b = m.latent('b', d.normal(0, s))
    const x = m.data('x', { dims: [row], role: 'covariate' })
    const y = m.data('y', { dims: [row], support: 'count' })
    const predictor = a.add(b.mul(x))
    m.observe('y', d.orderedProbit(predictor, CUTS), y)
  })
  // endregion
  return {
    definition,
    data: { x: OX, y: OY },
    equations: [
      `a, b ~ N(0, ${s}²)   (the intercept and the slope)`,
      `y_i ~ OrderedProbit(a + b x_i, cuts = (${CUTS.join(', ')})), i = 1..${OX.length}   (four ordered categories)`,
      `observe x = (${OX.join(', ')}) and y = (${OY.join(', ')})`,
    ],
    quantities: [
      { name: 'a', label: 'a (intercept)', support: 'real' },
      { name: 'b', label: 'b (slope)', support: 'real' },
    ],
    pair: ['a', 'b'],
    grid: { points: 41, maxRefinements: 0 },
    figure: ordinalFigure(OX, OY),
  }
}

/**
 * Twelve ordered outcomes (four categories) along a covariate, a cumulative probit of a linear predictor with the cut points fixed:
 * the `ordinal-probit` site, whose tilted moments are those of a Gaussian times the difference of two normal distribution functions.
 *
 * @index estimator EP atom: an ordinal probit regression with fixed cut points.
 * @example
 * ```ts
 * import { ordinalProbitAtom } from 'omai-engine/bayesian-examples'
 * ordinalProbitAtom.name // => 'ordinal-probit'
 * ```
 */
export const ordinalProbitAtom: EpAtom = defineAtom({
  name: 'ordinal-probit',
  title: 'Ordinal probit',
  what: 'An intercept and a slope with Gaussian priors and twelve ordered outcomes in four categories through a cumulative probit with fixed cut points: the ordinal numeric site on the linear predictor.',
  factors: ['prior (gaussian) x2', 'matmul (linear predictor)', 'numeric likelihood: ordinal probit'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'grid',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.8, 1.2],
    ks: 0.08,
    reason:
      'a Gaussian projection of each tilted distribution (a Gaussian times the mass between two cut points), by quadrature',
  },
  knobs: dict({ priorScale: real(0.5, 6, { default: 2, step: 0.1, label: 'prior sd of the coefficients' }) }),
  problem: ordinalProblem,
})

function ordinalLearnedProblem(k: Config): AtomProblem {
  const [seed, n] = [exampleNumber(k, 'seed'), exampleNumber(k, 'n')]
  const [trueA, trueB] = [exampleNumber(k, 'trueIntercept'), exampleNumber(k, 'trueSlope')]
  const s = exampleNumber(k, 'priorScale')
  // region ordinal-probit-learned
  // the ground truth: a latent score with standard normal noise, cut at the known cut points
  const kk = key(seed)
  const x = Array.from(toFlat(uniform(foldIn(kk, 0), [n])), (v) => 4 * v - 2)
  const noise = toFlat(normal(foldIn(kk, 1), [n]))
  const y = x.map((xi, i) => CUTS.filter((c) => c < trueA + trueB * xi + (noise[i] ?? 0)).length)
  const definition = defineModel('ordinal-probit-learned', (m) => {
    const row = m.range('row')
    // the intercept and the slope of the predictor: both unknown
    const a = m.latent('a', d.normal(0, s))
    const b = m.latent('b', d.normal(0, s))
    const xx = m.data('x', { dims: [row], role: 'covariate' })
    const yy = m.data('y', { dims: [row], support: 'count' })
    const predictor = a.add(b.mul(xx))
    m.observe('y', d.orderedProbit(predictor, CUTS), yy)
  })
  // endregion
  return {
    definition,
    data: { x, y },
    equations: [
      `truth: score ${trueA} + ${trueB} x + N(0, 1) noise cut at (${CUTS.join(', ')}); ${n} points, seed ${seed}`,
      `a, b ~ N(0, ${s}²)   (the unknown intercept and slope)`,
      `y_i ~ OrderedProbit(a + b x_i, cuts), i = 1..${n}`,
    ],
    quantities: [
      { name: 'a', label: 'a (intercept)', support: 'real', truth: trueA },
      { name: 'b', label: 'b (slope)', support: 'real', truth: trueB },
    ],
    pair: ['a', 'b'],
    grid: { points: 41, maxRefinements: 0 },
    figure: ordinalFigure(x, y, [trueA, trueB]),
  }
}

/**
 * The ordinal probit regression with the intercept and slope learned from outcomes simulated from stated values: over many seeds,
 * how often EP's 90% intervals hold them.
 *
 * @index estimator EP atom: an ordinal probit regression with coefficients recovered from simulated outcomes.
 * @example
 * ```ts
 * import { ordinalProbitLearnedAtom } from 'omai-engine/bayesian-examples'
 * ordinalProbitLearnedAtom.extends // => 'ordinal-probit'
 * ```
 */
export const ordinalProbitLearnedAtom: EpAtom = defineAtom({
  name: 'ordinal-probit-learned',
  title: 'Ordinal probit: parameters learned',
  extends: 'ordinal-probit',
  recovery: recoveryOf(
    'a Gaussian projection of each ordinal tilted distribution by quadrature; the categories carry less information per point than a measurement and the prior pulls a little',
  ),
  what: 'The ordinal probit regression of the basic atom with the intercept and the slope unknown, the outcomes simulated from stated values with the same cut points.',
  factors: ['prior (gaussian) x2', 'matmul (linear predictor)', 'numeric likelihood: ordinal probit'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'grid',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.8, 1.2],
    ks: 0.08,
    reason: 'a Gaussian projection of each tilted distribution, by quadrature',
  },
  knobs: dict({
    ...simulationKnobs(30),
    trueIntercept: real(-1.5, 1.5, { default: 0.3, step: 0.1, label: 'true intercept' }),
    trueSlope: real(-2, 2, { default: 1, step: 0.1, label: 'true slope' }),
    priorScale: real(0.5, 6, { default: 2, step: 0.1, label: 'prior sd of the coefficients' }),
  }),
  problem: ordinalLearnedProblem,
})

// ----- quantile regression -----

const QX = [-1.8, -1.4, -1, -0.6, -0.2, 0.2, 0.6, 1, 1.4, 1.8]
const QY = [-1.4, -1.5, -0.6, -0.9, -0.1, 0.2, 0.3, 1.1, 0.9, 1.8]

function quantileFigure(
  what: string,
  x: readonly number[],
  y: readonly number[],
  tau: number,
  truth?: readonly [number, number],
): AtomFigure {
  const line = (a: number, b: number, name: string, colour?: string): FigureCurve => ({
    name,
    ...(colour === undefined ? {} : { colour }),
    x: CURVE_X,
    y: CURVE_X.map((v) => a + b * v),
  })
  return {
    caption: `The ${what} and the ${Math.round(tau * 100)}th percentile line at the mean of the belief${truth === undefined ? '' : ' (the true line in green)'}: about ${Math.round(tau * 100)}% of the points lie below it.`,
    xLabel: 'x',
    yLabel: 'y',
    xLimits: [-2, 2],
    points: [{ name: 'observations', x: [...x], y: [...y] }],
    ...(truth === undefined ? {} : { curves: [line(truth[0], truth[1], 'true line', 'tab2')] }),
    answer: (m) => [line(m['a'] ?? Number.NaN, m['b'] ?? Number.NaN, 'fitted line')],
  }
}

function quantileProblem(k: Config): AtomProblem {
  const [tau, scale, s] = [exampleNumber(k, 'tau'), exampleNumber(k, 'scale'), exampleNumber(k, 'priorScale')]
  // region quantile-regression
  const definition = defineModel('quantile-regression', (m) => {
    const row = m.range('row')
    // the intercept and the slope of the line at the quantile level tau
    const a = m.latent('a', d.normal(0, s))
    const b = m.latent('b', d.normal(0, s))
    const x = m.data('x', { dims: [row], role: 'covariate' })
    const y = m.data('y', { dims: [row], role: 'response' })
    const line = a.add(b.mul(x))
    m.observe('y', d.asymmetricLaplace(line, scale, tau), y)
  })
  // endregion
  return {
    definition,
    data: { x: QX, y: QY },
    equations: [
      `a, b ~ N(0, ${s}²)   (the intercept and the slope of the line at the quantile level)`,
      `y_i ~ AsymmetricLaplace(a + b x_i, scale ${scale}, tau ${tau}), i = 1..${QX.length}   (the check-loss likelihood of quantile regression)`,
      `observe x = (${QX.join(', ')}) and y = (${QY.join(', ')})`,
    ],
    quantities: [
      { name: 'a', label: 'a (intercept)', support: 'real' },
      { name: 'b', label: 'b (slope)', support: 'real' },
    ],
    pair: ['a', 'b'],
    // the kinked log density has no curvature for the library's Laplace box to find: an explicit box around the prior's mass
    grid: {
      points: 81,
      maxRefinements: 0,
      box: [
        [-3, 3],
        [-3, 3],
      ],
    },
    figure: quantileFigure('observations', QX, QY, tau),
  }
}

/**
 * Ten observations and a line fitted to their `tau` quantile, with the asymmetric Laplace (the check loss of quantile regression) as
 * the likelihood: the `asymmetric-laplace` site, a Gaussian times a kinked density, whose tilted moments are found by quadrature
 * split at the observation.
 *
 * @index estimator EP atom: Bayesian quantile regression with the asymmetric Laplace.
 * @example
 * ```ts
 * import { quantileRegressionAtom } from 'omai-engine/bayesian-examples'
 * quantileRegressionAtom.name // => 'quantile-regression'
 * ```
 */
export const quantileRegressionAtom: EpAtom = defineAtom({
  name: 'quantile-regression',
  title: 'Quantile regression',
  what: 'An intercept and a slope with Gaussian priors and ten observations whose 80th percentile line is fitted, the asymmetric Laplace as the likelihood: the numeric quantile site on the linear predictor.',
  factors: ['prior (gaussian) x2', 'matmul (linear predictor)', 'numeric likelihood: asymmetric Laplace'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'grid',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.8, 1.2],
    ks: 0.08,
    reason:
      'a Gaussian projection of each tilted distribution, whose density has a kink at the observation, by split quadrature',
  },
  knobs: dict({
    tau: real(0.1, 0.9, { default: 0.8, step: 0.05, label: 'quantile level tau' }),
    scale: real(0.1, 2, { default: 0.5, step: 0.05, label: 'scale of the check loss' }),
    priorScale: real(0.5, 6, { default: 2, step: 0.1, label: 'prior sd of the coefficients' }),
  }),
  problem: quantileProblem,
})

function quantileLearnedProblem(k: Config): AtomProblem {
  const [seed, n] = [exampleNumber(k, 'seed'), exampleNumber(k, 'n')]
  const [trueA, trueB] = [exampleNumber(k, 'trueIntercept'), exampleNumber(k, 'trueSlope')]
  const [tau, scale, s] = [exampleNumber(k, 'tau'), exampleNumber(k, 'scale'), exampleNumber(k, 'priorScale')]
  // region quantile-regression-learned
  // the ground truth: asymmetric Laplace noise around the line, so its tau quantile is the line
  const kk = key(seed)
  const x = Array.from(toFlat(uniform(foldIn(kk, 0), [n])), (v) => 4 * v - 2)
  const q = toFlat(uniform(foldIn(kk, 1), [n]))
  const below = (p: number): number => Math.log(p / tau) / (1 - tau)
  const above = (p: number): number => -Math.log((1 - p) / (1 - tau)) / tau
  const noise = (p: number): number => (p < tau ? below(p) : above(p))
  const y = x.map((xi, i) => trueA + trueB * xi + scale * noise(q[i] ?? 0.5))
  const definition = defineModel('quantile-regression-learned', (m) => {
    const row = m.range('row')
    // the intercept and the slope of the line at the quantile level: both unknown
    const a = m.latent('a', d.normal(0, s))
    const b = m.latent('b', d.normal(0, s))
    const xx = m.data('x', { dims: [row], role: 'covariate' })
    const yy = m.data('y', { dims: [row], role: 'response' })
    const line = a.add(b.mul(xx))
    m.observe('y', d.asymmetricLaplace(line, scale, tau), yy)
  })
  // endregion
  const half = Math.max(0.6, (10 * scale) / Math.sqrt(n))
  return {
    definition,
    data: { x, y },
    equations: [
      `truth: the ${Math.round(tau * 100)}th percentile line ${trueA} + ${trueB} x with asymmetric Laplace noise of scale ${scale}; ${n} points, seed ${seed}`,
      `a, b ~ N(0, ${s}²)   (the unknown intercept and slope)`,
      `y_i ~ AsymmetricLaplace(a + b x_i, scale ${scale}, tau ${tau}), i = 1..${n}`,
    ],
    quantities: [
      { name: 'a', label: 'a (intercept)', support: 'real', truth: trueA },
      { name: 'b', label: 'b (slope)', support: 'real', truth: trueB },
    ],
    pair: ['a', 'b'],
    // an explicit box around the truth, as wide as the posterior of a median-like fit can be
    grid: {
      points: 81,
      maxRefinements: 0,
      box: [
        [trueA - half, trueA + half],
        [trueB - half, trueB + half],
      ],
    },
    figure: quantileFigure('simulated points', x, y, tau, [trueA, trueB]),
  }
}

/**
 * Quantile regression with the intercept and slope of the percentile line learned from points simulated with asymmetric Laplace
 * noise around a stated line: over many seeds, how often EP's 90% intervals hold the line's coefficients.
 *
 * @index estimator EP atom: Bayesian quantile regression with coefficients recovered from simulated points.
 * @example
 * ```ts
 * import { quantileRegressionLearnedAtom } from 'omai-engine/bayesian-examples'
 * quantileRegressionLearnedAtom.extends // => 'quantile-regression'
 * ```
 */
export const quantileRegressionLearnedAtom: EpAtom = defineAtom({
  name: 'quantile-regression-learned',
  title: 'Quantile regression: parameters learned',
  extends: 'quantile-regression',
  recovery: {
    seeds: 30,
    coverage: 0.65,
    zrms: 1.8,
    reason:
      'the model is the data-generating one (asymmetric Laplace noise) but its posterior is a little overconfident for 40 points (the exact posterior covers the slope 0.85 of 150 simulated sets, EP the same to three digits on 20 seeds), and the line sits at a tail quantile, so a floor of 0.65',
  },
  what: 'The quantile regression of the basic atom with the intercept and the slope unknown, the points simulated from a stated line with asymmetric Laplace noise.',
  factors: ['prior (gaussian) x2', 'matmul (linear predictor)', 'numeric likelihood: asymmetric Laplace'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'grid',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.8, 1.2],
    ks: 0.08,
    reason: 'a Gaussian projection of each tilted distribution, by split quadrature',
  },
  knobs: dict({
    ...simulationKnobs(40),
    trueIntercept: real(-1.5, 1.5, { default: 0.2, step: 0.1, label: 'true intercept' }),
    trueSlope: real(-2, 2, { default: 0.8, step: 0.1, label: 'true slope' }),
    tau: real(0.1, 0.9, { default: 0.8, step: 0.05, label: 'quantile level tau' }),
    scale: real(0.1, 2, { default: 0.5, step: 0.05, label: 'scale of the check loss' }),
    priorScale: real(0.5, 6, { default: 2, step: 0.1, label: 'prior sd of the coefficients' }),
  }),
  problem: quantileLearnedProblem,
})
