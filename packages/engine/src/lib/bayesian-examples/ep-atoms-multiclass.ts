/**
 * The multiclass atoms: a categorical outcome among three classes whose probabilities are a softmax of one Gaussian score per
 * class (a multinomial logit classifier with one slope per class on a scalar covariate), the `multiclass-site` factor, against the
 * exact posterior of the three slopes on a grid; and its twin with the slopes learned from data simulated from stated values,
 * with the calibration of the intervals over seeds.
 */

import * as d from '../probprog/model/dists/index.ts'
import { defineModel, type ModelDefinition } from '../probprog/model/index.ts'
import { foldIn, key, uniform } from '../random/index.ts'
import { dict, real } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import type { Config } from '../trace/index.ts'
import {
  type AtomFigure,
  type AtomProblem,
  defineAtom,
  type EpAtom,
  type ExactMarginal,
  type FigureCurve,
} from './ep-atoms.ts'
import { simulationKnobs } from './ep-atoms-extended-conjugate.ts'
import { gridMarginals } from './ep-atoms-quad.ts'
import { exampleNumber } from './example-method.ts'

const CLASSES = 3
const GRID = 37

/** The softmax of the three scores `w_k x`. */
const softmax = (w: readonly number[], x: number): number[] => {
  const z = w.map((v) => v * x)
  const top = Math.max(...z)
  const e = z.map((v) => Math.exp(v - top))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/** A value computed the first time it is asked for. */
function once<T>(make: () => T): () => T {
  let value: T | undefined
  return () => (value ??= make())
}

/** The model both atoms share: three class slopes with Gaussian priors and a categorical outcome with softmax probabilities of `x w_k`. */
function softmaxModel(name: string, priorSd: number): ModelDefinition {
  // region softmax-regression softmax-regression-learned
  const definition = defineModel(name, (m) => {
    const row = m.range('row')
    const cls = m.range('class', { size: CLASSES })
    // one slope per class: the score of class k at x is w_k x
    const w = m.latent('w', d.normal(0, priorSd), { dims: [cls] })
    const x = m.data('x', { dims: [row], role: 'covariate' })
    const y = m.data('y', { dims: [row], support: 'count' })
    // the class scores are a vector over the class dimension, the outcome a softmax of them
    const logits = x.mul(w)
    m.observe('y', d.categorical({ logits }, { over: cls }), y)
  })
  // endregion
  return definition
}

/** The exact posterior of the three slopes on a grid (the box is the prior's plus and minus three standard deviations, capped). */
function slopeGrid(x: readonly number[], y: readonly number[], priorSd: number) {
  const half = Math.min(3 * priorSd, 9)
  return gridMarginals(
    (w) => {
      let lp = (-0.5 * w.reduce((a, v) => a + v * v, 0)) / priorSd ** 2
      for (const [i, xi] of x.entries()) lp += Math.log(softmax(w, xi)[y[i] as number] as number)
      return lp
    },
    [
      [-half, half],
      [-half, half],
      [-half, half],
    ],
    GRID,
  )
}

const CURVE_X = Array.from({ length: 41 }, (_, i) => -2 + i / 10)

/** The figure: the labelled covariates as a rug by class, and the three class probabilities at the mean slopes. */
function figureOf(x: readonly number[], y: readonly number[]): AtomFigure {
  const curve = (w: readonly number[], k: number, name: string): FigureCurve => ({
    name,
    x: CURVE_X,
    y: CURVE_X.map((v) => softmax(w, v)[k] as number),
  })
  return {
    caption:
      'The labelled covariates (a rug, coloured by class) and the probability of each class at the mean of the belief.',
    xLabel: 'x',
    yLabel: 'probability of the class',
    xLimits: [-2, 2],
    yLimits: [0, 1],
    points: [
      {
        name: 'observations',
        x: [...x],
        y: x.map(() => 0.02),
        group: [...y],
        groupNames: { 0: 'class 0', 1: 'class 1', 2: 'class 2' },
      },
    ],
    answer: (m) => {
      const w = [m['w[0]'] ?? Number.NaN, m['w[1]'] ?? Number.NaN, m['w[2]'] ?? Number.NaN]
      return [0, 1, 2].map((k) => curve(w, k, `P(class ${k})`))
    },
  }
}

const X = [-1.4, -0.9, -0.5, -0.1, 0.2, 0.6, 1.0, 1.5, -0.3, 0.8]
const Y = [0, 0, 0, 2, 1, 1, 1, 1, 2, 2]

function softmaxProblem(k: Config): AtomProblem {
  const s = exampleNumber(k, 'priorScale')
  const grid = once(() => slopeGrid(X, Y, s))
  return {
    definition: softmaxModel('softmax-regression', s),
    data: { x: X, y: Y },
    equations: [
      `w_0, w_1, w_2 ~ N(0, ${s}²)   (one slope per class)`,
      'y_i ~ Categorical(softmax(w_0 x_i, w_1 x_i, w_2 x_i)), i = 1..10',
      `observe x = (${X.join(', ')}) and the classes y = (${Y.join(', ')})`,
    ],
    quantities: [0, 1, 2].map((j) => ({
      name: `w[${j}]`,
      label: `w_${j} (slope of class ${j})`,
      support: 'real' as const,
    })),
    pair: ['w[0]', 'w[1]'],
    get exact() {
      return Object.fromEntries([0, 1, 2].map((j) => [`w[${j}]`, grid().marginals[j] as ExactMarginal]))
    },
    get exactJoint() {
      return { a: 'w[0]', b: 'w[1]', ...grid().joint(0, 1) }
    },
    referenceSource: `the posterior of the three slopes on a ${GRID} by ${GRID} by ${GRID} grid`,
    figure: figureOf(X, Y),
  }
}

/**
 * Ten labelled points on a line, three classes whose scores are one slope each times the covariate, and a softmax: the
 * `multiclass-site` factor, whose tilted moments are a three-dimensional integral over the class scores, against the posterior of
 * the three slopes on a grid.
 *
 * @index estimator EP atom: a multinomial logit classifier (softmax of three Gaussian class scores).
 * @example
 * ```ts
 * import { softmaxRegressionAtom } from 'omai-engine/bayesian-examples'
 * softmaxRegressionAtom.name // => 'softmax-regression'
 * ```
 */
export const softmaxRegressionAtom: EpAtom = defineAtom({
  name: 'softmax-regression',
  title: 'Softmax',
  what: 'Three class slopes with Gaussian priors and ten labelled points through a softmax: the multiclass site, an adaptive Gauss-Hermite integral over the three class scores of each point.',
  factors: ['prior (gaussian) x3', 'scale (known covariate times a slope)', 'multiclass-site (softmax)'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.25,
    sdRatio: [0.7, 1.25],
    ks: 0.12,
    reason:
      'the three slopes share every observation, so one Gaussian each cannot hold their correlations and EP is a little narrow; the means are close',
  },
  ep: { damping: 0.5, tol: 1e-4 },
  knobs: dict({ priorScale: real(0.5, 6, { default: 2, step: 0.1, label: 'prior sd of the slopes' }) }),
  problem: softmaxProblem,
})

function softmaxLearnedProblem(k: Config): AtomProblem {
  const [seed, n] = [exampleNumber(k, 'seed'), exampleNumber(k, 'n')]
  const truth = [exampleNumber(k, 'trueSlope0'), exampleNumber(k, 'trueSlope1'), exampleNumber(k, 'trueSlope2')]
  const s = exampleNumber(k, 'priorScale')
  // region softmax-regression-learned
  // the ground truth: covariates on [-2, 2] and a class drawn from the softmax of the true slopes times the covariate
  const kk = key(seed)
  const x = Array.from(toFlat(uniform(foldIn(kk, 0), [n])), (v) => 4 * v - 2)
  const u = toFlat(uniform(foldIn(kk, 1), [n]))
  const y = x.map((xi, i) => {
    const p = softmax(truth, xi)
    const v = u[i] ?? 0.5
    return v < (p[0] as number) ? 0 : v < (p[0] as number) + (p[1] as number) ? 1 : 2
  })
  const definition = softmaxModel('softmax-regression-learned', s)
  // endregion
  const grid = once(() => slopeGrid(x, y, s))
  return {
    definition,
    data: { x, y },
    equations: [
      `truth: slopes (${truth.join(', ')}); ${n} covariates uniform on [-2, 2], classes from the softmax, seed ${seed}`,
      `w_0, w_1, w_2 ~ N(0, ${s}²)   (one slope per class)`,
      `y_i ~ Categorical(softmax(w_0 x_i, w_1 x_i, w_2 x_i)), i = 1..${n}`,
    ],
    quantities: [0, 1, 2].map((j) => ({
      name: `w[${j}]`,
      label: `w_${j} (slope of class ${j})`,
      support: 'real' as const,
      truth: truth[j] as number,
    })),
    pair: ['w[0]', 'w[1]'],
    get exact() {
      return Object.fromEntries([0, 1, 2].map((j) => [`w[${j}]`, grid().marginals[j] as ExactMarginal]))
    },
    get exactJoint() {
      return { a: 'w[0]', b: 'w[1]', ...grid().joint(0, 1) }
    },
    referenceSource: `the posterior of the three slopes on a ${GRID} by ${GRID} by ${GRID} grid`,
    figure: figureOf(x, y),
  }
}

/**
 * The same classifier with the slopes learned from points and classes simulated from stated slopes: over many seeds, how often
 * EP's 90% intervals hold the true slopes, against the exact posterior of the three on a grid.
 *
 * @index estimator EP atom: a multinomial logit classifier with slopes learned from simulated classes.
 * @example
 * ```ts
 * import { softmaxRegressionLearnedAtom } from 'omai-engine/bayesian-examples'
 * softmaxRegressionLearnedAtom.extends // => 'softmax-regression'
 * ```
 */
export const softmaxRegressionLearnedAtom: EpAtom = defineAtom({
  name: 'softmax-regression-learned',
  title: 'Softmax: parameters learned',
  extends: 'softmax-regression',
  recovery: {
    seeds: 16,
    coverage: 0.7,
    zrms: 1.7,
    reason:
      'the three slopes are coupled through every observation (the softmax is invariant to a common shift of the scores, which only the prior breaks), EP carries one Gaussian each and the prior pulls a little; each run integrates three class scores per point, so 16 seeds and a floor of 0.7',
  },
  what: 'The softmax classifier of the basic atom with the three class slopes unknown, the classes simulated from stated slopes.',
  factors: ['prior (gaussian) x3', 'scale (known covariate times a slope)', 'multiclass-site (softmax)'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.3,
    sdRatio: [0.5, 1.3],
    ks: 0.2,
    reason:
      'the slopes are correlated in the posterior (every observation joins the three) and EP carries one Gaussian each, so its spreads are 0.55 to 0.8 of the exact ones and its means within 0.15 of an exact sd',
  },
  ep: { damping: 0.5, tol: 1e-4 },
  knobs: dict({
    ...simulationKnobs(24),
    trueSlope0: real(-3, 3, { default: -1.5, step: 0.1, label: 'true slope of class 0' }),
    trueSlope1: real(-3, 3, { default: 0.3, step: 0.1, label: 'true slope of class 1' }),
    trueSlope2: real(-3, 3, { default: 1.2, step: 0.1, label: 'true slope of class 2' }),
    priorScale: real(0.5, 6, { default: 2, step: 0.1, label: 'prior sd of the slopes' }),
  }),
  problem: softmaxLearnedProblem,
})
