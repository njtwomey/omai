/**
 * The atoms that compose several pieces of the suite in one small model: a softmax over many classes with linear scores, a coefficient for each of two groups drawn around a shared mean with a learned precision, and a gamma prior on that precision. The exact answer is a three-dimensional grid over the two group coefficients and the precision, with the shared mean integrated out in closed form.
 */

import { foldIn, key, uniform } from '../../../random/index.ts'
import { dict, int, real } from '../../../space/index.ts'
import { toFlat } from '../../../tensor/index.ts'
import type { Config } from '../../../trace/index.ts'
import { configNumber } from '../../inference/method.ts'
import * as d from '../../model/dists/index.ts'
import { defineModel, ops } from '../../model/index.ts'
import { type ExactMarginal, posterior3, quadratureMarginal } from '../../validation/index.ts'
import { defineAtom } from '../_define.ts'
import { tolerance } from '../_tolerance.ts'
import { once } from '../_util.ts'
import type { AtomFigure, AtomParts, EpAtom, FigureCurve } from '../atom-types.ts'

/** The prior sd of the shared mean, and the shape and rate of the gamma prior on the precision of the group coefficients. */
const MEAN_SD = 2
const SHAPE = 2
const RATE = 1
/** The shared mean the groups are drawn around, and where the covariates lie. */
const TRUE_MEAN = 1
const SPAN = 2

/** The log probability of the label of each row under the softmax of `x_ik beta`, summed over the rows. */
function logLikelihood(x: readonly (readonly number[])[], y: readonly number[], beta: number): number {
  let total = 0
  for (const [i, row] of x.entries()) {
    const scores = row.map((v) => v * beta)
    const top = Math.max(...scores)
    const norm = Math.log(scores.reduce((s, v) => s + Math.exp(v - top), 0))
    total += (scores[y[i] as number] as number) - top - norm
  }
  return total
}

/** The data of two groups: covariates on [-2, 2] for each class of each row, and a label drawn from the softmax of the true scores. */
function simulate(seed: number, rows: number, classes: number, betas: readonly [number, number]) {
  const kk = key(seed)
  const group = (g: number) => {
    const flat = toFlat(uniform(foldIn(kk, 10 * g), [rows * classes]))
    const u = toFlat(uniform(foldIn(kk, 10 * g + 1), [rows]))
    const x = Array.from({ length: rows }, (_, i) =>
      Array.from({ length: classes }, (_, c) => 2 * SPAN * (flat[i * classes + c] ?? 0.5) - SPAN),
    )
    const y = x.map((row, i) => {
      const scores = row.map((v) => Math.exp(v * (betas[g] as number)))
      const total = scores.reduce((s, v) => s + v, 0)
      let acc = 0
      for (const [c, s] of scores.entries()) {
        acc += s / total
        if ((u[i] ?? 0.5) < acc) return c
      }
      return classes - 1
    })
    return { x, y }
  }
  return { first: group(0), second: group(1) }
}

/** The posterior of the two group coefficients and the log precision with the shared mean integrated out, as a grid that follows the mass. */
function gridOf(first: ReturnType<typeof simulate>['first'], second: ReturnType<typeof simulate>['second']) {
  const caches = [new Map<number, number>(), new Map<number, number>()]
  const likelihood = (g: 0 | 1, beta: number): number => {
    const cache = caches[g] as Map<number, number>
    const hit = cache.get(beta)
    if (hit !== undefined) return hit
    const data = g === 0 ? first : second
    const value = logLikelihood(data.x, data.y, beta)
    cache.set(beta, value)
    return value
  }
  // Where each coefficient lies on its own data, for the first box.
  const box = (g: 0 | 1) => {
    const alone = quadratureMarginal((b) => likelihood(g, b) - (b * b) / 50, -8, 8, 800)
    return { lo: alone.mean - 7 * alone.sd, hi: alone.mean + 7 * alone.sd, min: -20, max: 20 }
  }
  const s2 = MEAN_SD ** 2
  return posterior3([box(0), box(1), { lo: 0.01, hi: 30, min: 1e-6, max: 1e3, warp: 'log' }], (b1, b2, tau) => {
    const v = 1 / tau
    // The two coefficients given the precision, the shared mean integrated out: zero mean, variances s2 + v, covariance s2.
    const det = v * (2 * s2 + v)
    const quad = ((s2 + v) * (b1 * b1 + b2 * b2) - 2 * s2 * b1 * b2) / det
    const logPrior = (SHAPE - 1) * Math.log(tau) - RATE * tau - 0.5 * Math.log(det) - 0.5 * quad
    return logPrior + likelihood(0, b1) + likelihood(1, b2)
  })
}

/** The probability of the first class at covariate value `v`, the other covariates of each row as observed, averaged over the rows. */
function chosenProbability(x: readonly (readonly number[])[], beta: number, v: number): number {
  let total = 0
  for (const row of x) {
    const scores = row.map((u, c) => (c === 0 ? v : u) * beta)
    const top = Math.max(...scores)
    const e = scores.map((s) => Math.exp(s - top))
    total += (e[0] as number) / e.reduce((s, w) => s + w, 0)
  }
  return total / x.length
}

const BINS = 8
const CURVE_X = Array.from({ length: 33 }, (_, i) => -SPAN + (2 * SPAN * i) / 32)

/** The figure: how often a class is chosen at each covariate value, by group, and the curve each group's coefficient makes of it. */
function figureOf(
  data: readonly [ReturnType<typeof simulate>['first'], ReturnType<typeof simulate>['second']],
  classes: number,
): AtomFigure {
  const binned = (g: 0 | 1) => {
    const hits = new Array<number>(BINS).fill(0)
    const seen = new Array<number>(BINS).fill(0)
    const { x, y } = data[g]
    for (const [i, row] of x.entries())
      for (const [c, v] of row.entries()) {
        const bin = Math.min(BINS - 1, Math.floor(((v + SPAN) / (2 * SPAN)) * BINS))
        seen[bin] = (seen[bin] ?? 0) + 1
        if (y[i] === c) hits[bin] = (hits[bin] ?? 0) + 1
      }
    return {
      name: `group ${g + 1}: share of the classes chosen`,
      x: hits.map((_, b) => -SPAN + ((b + 0.5) * 2 * SPAN) / BINS),
      y: hits.map((h, b) => h / Math.max(1, seen[b] ?? 0)),
    }
  }
  const curve = (g: 0 | 1, beta: number): FigureCurve => ({
    name: `group ${g + 1}: P(chosen) with beta_${g + 1} = ${Number(beta.toPrecision(3))}`,
    x: CURVE_X,
    y: CURVE_X.map((v) => chosenProbability(data[g].x, beta, v)),
  })
  return {
    caption: `The covariate of a class against how often that class is the one chosen (${classes} classes, binned; each group a colour), and the probability each group's coefficient gives it.`,
    xLabel: 'covariate x of a class',
    yLabel: 'probability the class is chosen',
    xLimits: [-SPAN, SPAN],
    yLimits: [0, 1],
    points: [binned(0), binned(1)],
    answer: (m) => [curve(0, m['beta_1'] ?? Number.NaN), curve(1, m['beta_2'] ?? Number.NaN)],
  }
}

function twoGroupsProblem(k: Config): AtomParts {
  const [seed, rows, classes] = [configNumber(k, 'seed'), configNumber(k, 'rows'), configNumber(k, 'classes')]
  const difference = configNumber(k, 'difference')
  const truth: [number, number] = [TRUE_MEAN - difference / 2, TRUE_MEAN + difference / 2]
  // region softmax-two-groups
  // the ground truth: covariates on [-2, 2] for each class of each row, labels from the softmax of the true scores
  const { first, second } = simulate(seed, rows, classes, truth)
  const definition = defineModel('softmax-two-groups', (m) => {
    const row1 = m.range('row_1')
    const row2 = m.range('row_2')
    const cls = m.range('class', { size: classes })
    // the precision of the group coefficients and their shared mean
    const tau = m.latent('tau', d.gamma(SHAPE, RATE))
    const mu = m.latent('mu', d.normal(0, MEAN_SD))
    // one coefficient for each group, drawn around the shared mean
    const beta1 = m.latent('beta_1', d.normal(mu, ops.div(1, ops.sqrt(tau))))
    const beta2 = m.latent('beta_2', d.normal(mu, ops.div(1, ops.sqrt(tau))))
    const x1 = m.data('x_1', { dims: [row1, cls], role: 'covariate' })
    const x2 = m.data('x_2', { dims: [row2, cls], role: 'covariate' })
    const y1 = m.data('y_1', { dims: [row1], support: 'count' })
    const y2 = m.data('y_2', { dims: [row2], support: 'count' })
    // the scores of the classes of a row are linear in its covariates, the label a softmax of them
    m.observe('y_1', d.categorical({ logits: x1.mul(beta1) }, { over: cls }), y1)
    m.observe('y_2', d.categorical({ logits: x2.mul(beta2) }, { over: cls }), y2)
  })
  // endregion
  const grid = once(() => gridOf(first, second))
  const tauMarginal = (): ExactMarginal => {
    const g = grid()
    const m = g.marginal(2)
    const mean = g.expect((_a, _b, t) => t)
    const sd = Math.sqrt(g.expect((_a, _b, t) => (t - mean) ** 2))
    return { mean, sd, pdf: m.pdf, ...(m.range === undefined ? {} : { range: m.range }) }
  }
  const muMarginal = (): ExactMarginal =>
    grid().mixture((b1, b2, t) => {
      const precision = 1 / MEAN_SD ** 2 + 2 * t
      return { mean: (t * (b1 + b2)) / precision, variance: 1 / precision }
    })
  return {
    build: {
      definition,
      data: { x_1: first.x, y_1: first.y, x_2: second.x, y_2: second.y },
      quantities: [
        { name: 'beta_1', label: 'beta_1 (coefficient of group 1)', support: 'real', truth: truth[0] },
        { name: 'beta_2', label: 'beta_2 (coefficient of group 2)', support: 'real', truth: truth[1] },
        { name: 'mu', label: 'mu (the shared mean)', support: 'real', truth: TRUE_MEAN },
        { name: 'tau', label: 'tau (precision of the coefficients)', support: 'positive' },
      ],
    },
    reference: {
      get exact() {
        return {
          beta_1: grid().marginal(0),
          beta_2: grid().marginal(1),
          mu: muMarginal(),
          tau: tauMarginal(),
        }
      },
      referenceSource: 'a three-dimensional grid over beta_1, beta_2 and tau, with mu integrated out in closed form',
    },
    present: {
      equations: [
        `truth: beta = (${truth.map((v) => Number(v.toPrecision(3))).join(', ')}) around the mean ${TRUE_MEAN}; ${rows} rows of ${classes} classes in each group, seed ${seed}`,
        `tau ~ Gamma(${SHAPE}, ${RATE}),  mu ~ N(0, ${MEAN_SD}²)`,
        'beta_g ~ N(mu, 1 / tau), g = 1, 2   (one coefficient for each group, around a shared mean)',
        `y_gi ~ Categorical(softmax(x_gi1 beta_g, ..., x_gi${classes} beta_g)), i = 1..${rows}`,
      ],
      pair: ['beta_1', 'beta_2'],
      figure: figureOf([first, second], classes),
    },
  }
}

/**
 * A softmax over many classes with linear scores, in two groups: each group has its own coefficient on the covariates of the
 * classes, drawn around a shared mean with a learned precision. The `multiclass-site` factor (the auxiliary-variable rule beyond six
 * classes) under a Gaussian-gamma hierarchy, against a three-dimensional grid of the two coefficients and the precision.
 */
const softmaxTwoGroupsAtom: EpAtom = defineAtom({
  name: 'softmax-two-groups',
  title: 'A softmax over many classes with coefficients that differ between two groups',
  what: 'A softmax over eight classes with scores linear in a covariate, one coefficient for each of two groups drawn around a shared mean with a gamma-distributed precision: the multiclass site (eight free scores per row) under the Gaussian-gamma hierarchy, with data simulated from stated coefficients.',
  factors: [
    'prior (gamma)',
    'prior (gaussian)',
    'gaussian-gamma-precision (latent value)',
    'scale (known covariate times a coefficient)',
    'multiclass-site (softmax, eight classes)',
  ],
  family: 'composed',
  reference: 'exact',
  tolerance: tolerance(
    'measured',
    'the coefficient of a group is shared by the scores of all its rows, so one Gaussian each cannot hold their correlations and plain EP is narrow (0.55 to 0.9 of the exact spread, the coefficient with the larger posterior spread the narrower; the means 0.2 exact sds off); the precision of two coefficients is hardly identified and skewed, which EP carries as a gamma (0.99 to 1.2 of the exact spread)',
    { meanZ: 0.3, sdRatio: [0.4, 1.3], ks: 0.25 },
  ),
  ep: { damping: 0.5, tol: 1e-4 },
  knobs: dict({
    seed: int(1, 9999, { default: 1, label: 'seed (the simulated data)' }),
    rows: int(5, 60, { default: 12, label: 'rows in each group' }),
    classes: int(4, 12, { default: 8, label: 'classes' }),
    difference: real(0, 2, { default: 1, step: 0.1, label: 'true difference between the groups' }),
  }),
  parts: twoGroupsProblem,
})

/** The atoms of the `composed` file. */
export const composedAtoms: readonly EpAtom[] = [softmaxTwoGroupsAtom]
