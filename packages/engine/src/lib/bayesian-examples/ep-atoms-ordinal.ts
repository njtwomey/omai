/**
 * The ordinal and quantile atoms: an ordered outcome (four categories) through the cumulative probit of a linear predictor with
 * the cut points fixed (the `ordinal-probit` row of the likelihood registry), and quantile regression, a line fitted to the 80th
 * percentile of the outcome with the asymmetric Laplace (the `asymmetric-laplace` row), and a tournament whose games can be drawn, the
 * TrueSkill draw margin as an ordinal probit on the difference of two skills. Each has a basic page on a handful of fixed observations
 * and a twin that learns the unknowns from data simulated from stated values.
 */

import * as d from '../probprog/model/dists/index.ts'
import { defineModel, ops } from '../probprog/model/index.ts'
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
  type ExactMarginal,
  type FigureCurve,
} from './ep-atoms.ts'
import { simulationKnobs } from './ep-atoms-extended-conjugate.ts'
import { gridMarginals } from './ep-atoms-quad.ts'
import { exampleNumber } from './example-method.ts'

const CUTS = [-0.6, 0.5, 1.4]
/** A value computed the first time it is asked for. */
function once<T>(make: () => T): () => T {
  let value: T | undefined
  return () => (value ??= make())
}

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

// ----- a tournament with draws: the TrueSkill draw margin -----

const PLAYERS = ['A', 'B', 'C']
const PAIRS: readonly (readonly [number, number])[] = [
  [0, 1],
  [0, 2],
  [1, 2],
  [0, 1],
  [1, 2],
  [0, 2],
]
/** The result of each game for the first player of its pair: 0 a loss, 1 a draw, 2 a win. */
const RESULTS = [2, 1, 0, 1, 2, 2]

/** The standard normal log density difference `log (Phi(b) - Phi(a))` in plain arithmetic, for the reference grids. */
const logBetween = (a: number, b: number): number => Math.log(Math.max(phi(b) - phi(a), 1e-300))

/** The exact posterior of the three skills (prior N(0, spread^2)) on a grid: games between the pairs with the draw margin. */
function skillGrid(
  pairs: readonly (readonly [number, number])[],
  results: readonly number[],
  margin: number,
  spread: number,
) {
  const half = margin / Math.SQRT2
  return gridMarginals(
    (sk) => {
      let lp = (-0.5 * sk.reduce((a, v) => a + v * v, 0)) / spread ** 2
      for (const [g, [a, b]] of pairs.entries()) {
        const u = ((sk[a] as number) - (sk[b] as number)) / Math.SQRT2
        const r = results[g] as number
        lp += logBetween(
          r === 0 ? -Infinity : r === 1 ? -half - u : half - u,
          r === 0 ? -half - u : r === 1 ? half - u : Infinity,
        )
      }
      return lp
    },
    [
      [-3 * spread, 3 * spread],
      [-3 * spread, 3 * spread],
      [-3 * spread, 3 * spread],
    ],
    33,
  )
}

function drawsFigure(skillTruth?: readonly number[]): AtomFigure {
  const at = [1, 2, 3]
  return {
    caption: `The belief about each player's skill after the games (the bars)${skillTruth === undefined ? '' : ', and the skill the games were played with (green)'}.`,
    xLabel: 'player',
    yLabel: 'skill',
    xLimits: [0.4, 3.6],
    points: [],
    ...(skillTruth === undefined
      ? {}
      : {
          curves: PLAYERS.map((n, i) => ({
            name: `true skill of ${n}`,
            colour: 'tab2',
            x: [(at[i] ?? 0) - 0.3, (at[i] ?? 0) + 0.3],
            y: [skillTruth[i] ?? 0, skillTruth[i] ?? 0],
          })),
        }),
    answer: (m) =>
      PLAYERS.map((n, i) => ({
        name: `player ${n}`,
        x: [(at[i] ?? 0) - 0.3, (at[i] ?? 0) + 0.3],
        y: [m[`skill[${i}]`] ?? Number.NaN, m[`skill[${i}]`] ?? Number.NaN],
      })),
  }
}

function drawsProblem(k: Config): AtomProblem {
  const margin = exampleNumber(k, 'margin')
  const cuts = [-margin / Math.SQRT2, margin / Math.SQRT2]
  // region skill-draws
  const definition = defineModel('skill-draws', (m) => {
    const player = m.range('player', { size: 3 })
    const game = m.range('game')
    // who played whom, and the result for the first player: lost (0), drawn (1) or won (2)
    const a = m.data('a', { dims: [game], support: 'index', indexes: player })
    const b = m.data('b', { dims: [game], support: 'index', indexes: player })
    const result = m.data('result', { dims: [game], support: 'count' })
    // the skill of each player
    const skill = m.latent('skill', d.normal(0, 1), { dims: [player] })
    // the game is lost, drawn or won as the performance gap falls below, inside or above the margin
    const gap = ops.div(skill.at(a).sub(skill.at(b)), Math.SQRT2)
    m.observe('result', d.orderedProbit(gap, cuts), result)
  })
  // endregion
  const grid = once(() => skillGrid(PAIRS, RESULTS, margin, 1))
  const notes = PAIRS.map(([a, b], g) => {
    const r = RESULTS[g] as number
    return `Game ${g + 1}: ${PLAYERS[a]} ${r === 1 ? 'drew with' : r === 2 ? 'beat' : 'lost to'} ${PLAYERS[b]}`
  })
  return {
    definition,
    data: { a: PAIRS.map((p) => p[0]), b: PAIRS.map((p) => p[1]), result: RESULTS },
    equations: [
      'skill_A, skill_B, skill_C ~ N(0, 1)',
      `game between a and b: lost, drawn or won as (skill_a - skill_b) / sqrt 2 + noise falls below -${(margin / Math.SQRT2).toFixed(3)}, inside, or above ${(margin / Math.SQRT2).toFixed(3)} (draw margin ${margin})`,
      ...notes,
    ],
    quantities: PLAYERS.map((n, i) => ({ name: `skill[${i}]`, label: `skill_${n}`, support: 'real' as const })),
    get exact() {
      return Object.fromEntries(PLAYERS.map((_, i) => [`skill[${i}]`, grid().marginals[i] as ExactMarginal]))
    },
    referenceSource: 'the posterior of the three skills on a 33 by 33 by 33 grid',
    figure: { ...drawsFigure(), notes },
  }
}

/**
 * Three players and six games, some drawn: each game is an ordered probit of the difference of two skills with the draw margin as the
 * cut points (the TrueSkill performance model with draws), the `ordinal-probit` site on a difference of two Gaussian variables,
 * against the posterior of the three skills on a grid.
 *
 * @index estimator EP atom: a tournament with draws, an ordinal probit factor on a difference of two skills (the TrueSkill draw margin).
 * @example
 * ```ts
 * import { skillDrawsAtom } from 'omai-engine/bayesian-examples'
 * skillDrawsAtom.name // => 'skill-draws'
 * ```
 */
export const skillDrawsAtom: EpAtom = defineAtom({
  name: 'skill-draws',
  title: 'Ordinal probit on a difference (draws)',
  what: 'Three Gaussian skills and six games that may be drawn, each an ordinal probit of the difference of two skills with the draw margin as cut points (the TrueSkill performance model with draws): every factor joins two variables.',
  factors: ['prior (gaussian) x3', 'ordinal probit on a difference of two variables'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.2,
    sdRatio: [0.75, 1.2],
    ks: 0.1,
    reason:
      'each result is a Gaussian times the mass between two cut points that EP projects onto a Gaussian, and the skills share the games',
  },
  knobs: dict({ margin: real(0.1, 2, { default: 0.7, step: 0.05, label: 'draw margin' }) }),
  problem: drawsProblem,
})

function drawsLearnedProblem(k: Config): AtomProblem {
  const [seed, rounds] = [exampleNumber(k, 'seed'), exampleNumber(k, 'n')]
  const margin = exampleNumber(k, 'margin')
  const truth = [exampleNumber(k, 'trueSkillA'), exampleNumber(k, 'trueSkillB'), exampleNumber(k, 'trueSkillC')]
  const cuts = [-margin / Math.SQRT2, margin / Math.SQRT2]
  // region skill-draws-learned
  // the ground truth: every pair plays in turn, the better performance wins unless the gap is inside the draw margin
  const games: (readonly [number, number])[] = []
  for (let r = 0; r < rounds; r++) {
    for (const [a, b] of PAIRS.slice(0, 3)) games.push(r % 2 ? [b, a] : [a, b])
  }
  const noise = toFlat(normal(foldIn(key(seed), 0), [games.length], { scale: Math.SQRT2 }))
  const skill = (i: number): number => truth[i] as number
  const gaps = games.map(([a, b], g) => skill(a) - skill(b) + (noise[g] as number))
  const results = gaps.map((v) => (v < -margin ? 0 : v > margin ? 2 : 1))
  const definition = defineModel('skill-draws-learned', (m) => {
    const player = m.range('player', { size: 3 })
    const game = m.range('game')
    const a = m.data('a', { dims: [game], support: 'index', indexes: player })
    const b = m.data('b', { dims: [game], support: 'index', indexes: player })
    const result = m.data('result', { dims: [game], support: 'count' })
    // the skill of each player: all unknown
    const skill = m.latent('skill', d.normal(0, 1), { dims: [player] })
    const gap = ops.div(skill.at(a).sub(skill.at(b)), Math.SQRT2)
    m.observe('result', d.orderedProbit(gap, cuts), result)
  })
  // endregion
  const grid = once(() => skillGrid(games, results, margin, 1))
  return {
    definition,
    data: { a: games.map((p) => p[0]), b: games.map((p) => p[1]), result: results },
    equations: [
      `truth: skills (${truth.join(', ')}), draw margin ${margin}; ${games.length} games (each pair ${rounds} times), seed ${seed}`,
      'skill_A, skill_B, skill_C ~ N(0, 1)',
      `game g between a and b: lost, drawn or won by the ordinal probit of (skill_a - skill_b) / sqrt 2 with cut points -${cuts[1]?.toFixed(3)} and ${cuts[1]?.toFixed(3)}`,
    ],
    quantities: PLAYERS.map((n, i) => ({
      name: `skill[${i}]`,
      label: `skill_${n}`,
      support: 'real' as const,
      truth: truth[i] as number,
    })),
    get exact() {
      return Object.fromEntries(PLAYERS.map((_, i) => [`skill[${i}]`, grid().marginals[i] as ExactMarginal]))
    },
    referenceSource: 'the posterior of the three skills on a 33 by 33 by 33 grid',
    figure: drawsFigure(truth),
  }
}

/**
 * The tournament with draws and the skills learned from games simulated from stated skills: over many seeds, how often EP's 90%
 * intervals hold them.
 *
 * @index estimator EP atom: a tournament with draws, skills recovered from simulated games.
 * @example
 * ```ts
 * import { skillDrawsLearnedAtom } from 'omai-engine/bayesian-examples'
 * skillDrawsLearnedAtom.extends // => 'skill-draws'
 * ```
 */
export const skillDrawsLearnedAtom: EpAtom = defineAtom({
  name: 'skill-draws-learned',
  title: 'Ordinal probit on a difference (draws): parameters learned',
  extends: 'skill-draws',
  recovery: recoveryOf(
    'the skills share the games (a loop: players who met have correlated posteriors and EP is a little narrow) and only differences are identified, the prior fixing the level; the truth is drawn from the prior scale, so the floor leaves room',
  ),
  what: 'The tournament with draws of the basic atom with the three skills unknown, the games (each pair several times) simulated from stated skills and the same draw margin.',
  factors: ['prior (gaussian) x3', 'ordinal probit on a difference of two variables'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.25,
    sdRatio: [0.7, 1.2],
    ks: 0.12,
    reason: 'the skills share every game, so one Gaussian each cannot hold their correlations',
  },
  knobs: dict({
    ...simulationKnobs(4),
    margin: real(0.1, 2, { default: 0.7, step: 0.05, label: 'draw margin' }),
    trueSkillA: real(-2, 2, { default: 0.9, step: 0.1, label: 'true skill of A' }),
    trueSkillB: real(-2, 2, { default: -0.2, step: 0.1, label: 'true skill of B' }),
    trueSkillC: real(-2, 2, { default: -0.7, step: 0.1, label: 'true skill of C' }),
  }),
  problem: drawsLearnedProblem,
})
