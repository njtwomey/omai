/**
 * The setups of expectation propagation with a kernel hyperparameter: a Gaussian process whose length scale is a random
 * value, so its Gram matrix is a function of it that the factor graph cannot see through. The library runs EP inside a grid over
 * the length scale (`epGridMethod`: the length scale is set to each value of a grid, EP runs on the model with a constant
 * covariance, and the EP log evidence of each run weights it), which for a Gaussian likelihood is exact: each node's evidence is
 * the marginal likelihood `N(y; 0, K + sigma^2 I)`, so the reference here, the same quadrature over the length scale with the
 * function values as the mixture of their Gaussian conditionals, is what EP must reproduce.
 */

import { gram, rbfKernel, scaleKernel } from '../../kernels/index.ts'
import { cholesky } from '../../numerics/linalg/index.ts'
import { foldIn, key, normal, uniform } from '../../random/index.ts'
import { dict, real } from '../../space/index.ts'
import { clip, matmul, reshape, tensor, toFlat, type Value } from '../../tensor/index.ts'
import type { Config } from '../../trace/index.ts'
import { configNumber } from '../inference/method.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { gpPosteriorGivenLengthScale } from '../models/index.ts'
import { type ExactMarginal, gaussianMixtureMarginal } from '../validation/index.ts'
import { type AtomProblem, defineAtom, type EpAtom } from './ep-atoms.ts'
import { simulationKnobs } from './ep-atoms-extended-conjugate.ts'

const X6 = [-2.4, -1.5, -0.6, 0.3, 1.2, 2.3]
const Y6 = [-0.8, -1.1, -0.3, 0.7, 0.9, 0.2]
/** The noise level the exact reference reads: it is the one of the library's own Gaussian process regression. */
const NOISE = 0.3
const PRIOR_SD = 0.5

/** The squared-exponential Gram matrix of the inputs at a length scale (the library's kernel, a little jitter on the diagonal). */
function rbfGram(x: Value, ell: Value): Value {
  return gram(rbfKernel({ lengthScale: clip(ell, 0.02, 20) }), reshape(x, [x.shape[0] ?? 0, 1]), { jitter: 1e-6 })
}

/** The density of `log N(0, sd^2)` evaluated at a positive number: the prior of the length scale. */
const logNormalPdf = (v: number, sd: number): number =>
  Math.exp(-0.5 * (Math.log(v) / sd) ** 2) / (v * sd * Math.sqrt(2 * Math.PI))

/**
 * The exact marginals of the length scale and of the function values of a Gaussian process regression with a log-normal prior on
 * the length scale: a quadrature over the length scale, on an even grid of its log, of the marginal likelihood times the prior, and
 * for each function value the mixture over that grid of its Gaussian posterior given the length scale.
 */
function exactMarginals(x: readonly number[], y: readonly number[]): Record<string, ExactMarginal> {
  const points = 801
  const lo = -5
  const hi = 5
  const u = Array.from({ length: points }, (_, i) => lo + ((hi - lo) * i) / (points - 1))
  const given = u.map((t) => gpPosteriorGivenLengthScale(x, y, Math.exp(t)))
  const logw = given.map(
    (g, i) => g.logMarginalLikelihood + Math.log(logNormalPdf(Math.exp(u[i] ?? 0), PRIOR_SD)) + (u[i] ?? 0),
  )
  const top = Math.max(...logw)
  const total = logw.reduce((a, l) => a + Math.exp(l - top), 0)
  const w = logw.map((l) => Math.exp(l - top) / total)
  const step = (hi - lo) / (points - 1)
  const out: Record<string, ExactMarginal> = {}
  const ell = u.map((t) => Math.exp(t))
  const ellMean = ell.reduce((a, v, i) => a + v * (w[i] ?? 0), 0)
  const ellSd = Math.sqrt(ell.reduce((a, v, i) => a + (v - ellMean) ** 2 * (w[i] ?? 0), 0))
  out['ell'] = {
    mean: ellMean,
    sd: ellSd,
    // The density of the length scale itself: the weight of a cell over its width on the length-scale axis.
    pdf: (v) => {
      if (!(v > 0)) return 0
      const t = (Math.log(v) - lo) / step
      const i = Math.floor(t)
      if (i < 0 || i >= points - 1) return 0
      const a = w[i] ?? 0
      const b = w[i + 1] ?? 0
      return (a + (b - a) * (t - i)) / step / v
    },
    range: [Math.max(1e-3, ellMean - 5 * ellSd), ellMean + 8 * ellSd],
  }
  x.forEach((_, j) => {
    out[`f[${j}]`] = gaussianMixtureMarginal(
      given.map((g, i) => ({ weight: w[i] ?? 0, mean: g.mean[j] ?? 0, variance: (g.sd[j] ?? 1) ** 2 })),
      (mean, sd) => [mean - 6 * sd, mean + 6 * sd],
    )
  })
  return out
}

/** The covariance function of the model: the Gram matrix of the inputs at the length scale it is given. */
const rbfGramOf = ([xs, scale]: readonly { readonly value: Value }[]): Value =>
  rbfGram((xs as { value: Value }).value, (scale as { value: Value }).value)

function gpLengthScaleProblem(_k: Config): AtomProblem {
  const zeros = new Array<number>(X6.length).fill(0)
  // region gp-length-scale
  const definition = defineModel('gp-length-scale', (m) => {
    const x = m.data('x', { dims: ['i'], role: 'covariate' })
    const y = m.data('y', { dims: ['i'], role: 'response' })
    // the length scale is random
    const ell = m.latent('ell', d.logNormal(0, 0.5))
    // so the covariance is a custom function of it: the Gram matrix of a squared-exponential kernel
    const cov = m.custom('rbfGram', rbfGramOf, [x, ell], { eventRank: 2 })
    // the function at the six inputs is a dense Gaussian vector
    const prior = d.multivariateNormal(zeros, { covariance: cov })
    const f = m.latent('f', prior, { dims: ['i'] })
    // observed through Gaussian noise
    m.observe('y', d.normal(f, 0.3), y)
  })
  // endregion
  return {
    definition,
    data: { x: X6, y: Y6 },
    exact: exactMarginals(X6, Y6),
    equations: [
      `ell ~ LogNormal(0, ${PRIOR_SD}²)   (the length scale)`,
      'f ~ N(0, K(ell)),  K_ij = exp(-(x_i - x_j)² / (2 ell²))   (a dense Gaussian vector, a custom function of ell)',
      `y_i ~ N(f_i, ${NOISE}²), i = 1..${X6.length}`,
      `observe y = (${Y6.join(', ')}) at x = (${X6.join(', ')})`,
    ],
    quantities: [
      { name: 'ell', label: 'length scale', support: 'positive' },
      ...X6.map((_, i) => ({ name: `f[${i}]`, label: `f(x_${i + 1})`, support: 'real' as const })),
    ],
    figure: {
      caption: 'The six observations and the posterior mean of the function at them (the line joins the means).',
      xLabel: 'x',
      yLabel: 'y',
      points: [{ name: 'observations', x: X6, y: Y6 }],
      answer: (means) => [{ name: 'f (posterior mean)', x: X6, y: X6.map((_, i) => means[`f[${i}]`] ?? Number.NaN) }],
    },
  }
}

/**
 * A Gaussian process regression on six points whose length scale is random: the covariance of the function is a custom
 * function of it, so the graph is built at each value of a grid over the length scale (EP inside a grid, the EP evidence of each
 * run as its weight), with a Gaussian likelihood in which EP is exact at every node.
 *
 * @index estimator EP atom: Gaussian process regression with a learned length scale (EP inside an evidence-weighted grid).
 * @example
 * ```ts
 * import { gpLengthScaleAtom } from 'omai-engine/probprog/ep'
 * gpLengthScaleAtom.name // => 'gp-length-scale'
 * ```
 */
export const gpLengthScaleAtom: EpAtom = defineAtom({
  name: 'gp-length-scale',
  title: 'Gaussian process with a learned length scale',
  what: 'A dense Gaussian prior on six function values whose kernel has a random length scale (a custom function), and a Gaussian observation of each: EP runs inside a grid over the length scale, each node weighted by its EP evidence and the prior.',
  factors: [
    'dense prior (gaussian-mv) at each grid node',
    'gaussian row (exact)',
    'the EP evidence as the weight of a node (ep-grid)',
  ],
  family: 'kernels',
  reference: 'exact',
  tolerance: {
    meanZ: 0.01,
    sdRatio: [0.98, 1.02],
    ks: 0.04,
    reason:
      'EP is exact at every node (a Gaussian likelihood), so the answer is the quadrature over the length scale of the exact conditionals; the gamma drawn for the length scale is a moment-matched summary of a log-normal-like posterior',
  },
  knobs: dict({}),
  problem: gpLengthScaleProblem,
})

// ----- a Gaussian process classifier whose length scale is learned -----

/** The variance of the kernel of the classifier (the amplitude the function varies over). */
const CLASS_VARIANCE = 2

/** The Gram matrix of the classifier's kernel: a variance times a squared-exponential kernel, at a length scale. */
const classGramOf = (x: Value, scale: Value): Value =>
  gram(
    scaleKernel(rbfKernel({ lengthScale: clip(scale, 0.05, 20) }), CLASS_VARIANCE),
    reshape(x, [x.shape[0] ?? 0, 1]),
    { jitter: 1e-6 },
  )

/** The covariance function of the model, as a custom function receives its arguments. */
const classGram = ([xs, scale]: readonly { readonly value: Value }[]): Value =>
  classGramOf((xs as { value: Value }).value, (scale as { value: Value }).value)

const phiOf = (z: number): number => 0.5 * (1 + erfOf(z / Math.SQRT2))

/** The error function (Abramowitz and Stegun 7.1.26, error under 2e-7), in plain arithmetic: the labels are drawn once. */
function erfOf(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x))
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
  const v = 1 - poly * Math.exp(-x * x)
  return x >= 0 ? v : -v
}

function gpClassLearnedProblem(k: Config): AtomProblem {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const trueEll = configNumber(k, 'trueLengthScale')
  // the ground truth: a function drawn from the process at the true length scale, labelled through a probit link
  const inputs = Array.from({ length: n }, (_, i) => -2.4 + (4.8 * i) / Math.max(1, n - 1))
  const kk = key(seed)
  const white = normal(foldIn(kk, 0), [n, 1])
  const f = Array.from(toFlat(matmul(cholesky(classGramOf(tensor(inputs), tensor(trueEll))), white)))
  const u = toFlat(uniform(foldIn(kk, 1), [n]))
  const y = f.map((v, i) => ((u[i] ?? 1) < phiOf(v) ? 1 : 0))
  const zeros = new Array<number>(n).fill(0)
  // region gp-probit-learned
  const definition = defineModel('gp-probit-learned', (m) => {
    const x = m.data('x', { dims: ['i'], role: 'covariate' })
    const yy = m.data('y', { dims: ['i'], support: 'binary', role: 'response' })
    // the length scale is random, and the covariance is a custom function of it
    const ell = m.latent('ell', d.logNormal(0, 0.5))
    const cov = m.custom('rbfGram', classGram, [x, ell], { eventRank: 2 })
    // the function at the inputs, labelled through a probit link
    const prior = d.multivariateNormal(zeros, { covariance: cov })
    const fn = m.latent('f', prior, { dims: ['i'] })
    m.observe('y', d.bernoulli({ probs: ops.probit(fn) }), yy)
  })
  // endregion
  return {
    definition,
    data: { x: inputs, y },
    equations: [
      `truth: a function drawn from the process at the length scale ${trueEll} (kernel variance ${CLASS_VARIANCE}), labelled through a probit link at ${n} inputs; seed ${seed}`,
      'ell ~ LogNormal(0, 0.5²)   (the length scale)',
      'f ~ N(0, K(ell)),  K_ij = 2 exp(-(x_i - x_j)² / (2 ell²))',
      'y_i ~ Bernoulli(Phi(f_i))',
    ],
    quantities: [
      { name: 'ell', label: 'length scale', support: 'positive', truth: trueEll },
      ...inputs.map((_, i) => ({
        name: `f[${i}]`,
        label: `f(x_${i + 1})`,
        support: 'real' as const,
        truth: f[i] as number,
      })),
    ],
    figure: {
      caption:
        'The labels (0 or 1) at the inputs; the green line is the true function, the black one the posterior mean of the function.',
      xLabel: 'x',
      yLabel: 'f',
      points: [{ name: 'labels', x: inputs, y: [...y], group: y, groupNames: { 0: 'label 0', 1: 'label 1' } }],
      curves: [{ name: 'true function', colour: 'tab2', x: inputs, y: f }],
      answer: (means) => [
        { name: 'f (posterior mean)', x: inputs, y: inputs.map((_, i) => means[`f[${i}]`] ?? Number.NaN) },
      ],
    },
  }
}

/**
 * Gaussian process classification with the length scale learned: a function drawn from the process at a stated length scale,
 * labelled through a probit link, and the model's length scale random (so the covariance is a custom function of it, and EP runs
 * inside a grid over it). Does EP recover the length scale and the function from the labels, and how often do its 90% intervals
 * hold them?
 *
 * @index estimator EP atom: a probit Gaussian process classifier with a learned length scale, recovered from simulated labels.
 * @example
 * ```ts
 * import { gpProbitLearnedAtom } from 'omai-engine/probprog/ep'
 * gpProbitLearnedAtom.extends // => 'gp-probit'
 * ```
 */
export const gpProbitLearnedAtom: EpAtom = defineAtom({
  name: 'gp-probit-learned',
  title: 'Gaussian process with a probit likelihood: parameters learned',
  extends: 'gp-probit',
  recovery: {
    seeds: 40,
    coverage: 0.75,
    zrms: 1.5,
    reason:
      "a length scale from eight binary labels is weakly identified, so its posterior stays close to the prior and the interval holds the truth every time (the coverage is 1, not 0.9: the spread is the prior's); the function values are exact up to the probit tilts at each node of the grid, and hold the truth in 83 to 98% of the runs",
  },
  what: 'The probit Gaussian process classifier of the basic atom with the length scale of its kernel unknown (a log-normal prior, the covariance a custom function of it) and the labels simulated from a function drawn at a stated length scale: EP runs inside a grid over the length scale, each node weighted by its EP evidence.',
  factors: [
    'dense prior (gaussian-mv) at each grid node',
    'probit link',
    'the EP evidence as the weight of a node (ep-grid)',
  ],
  family: 'kernels',
  reference: 'truth',
  tolerance: {
    meanZ: 1,
    sdRatio: [0, 10],
    ks: 1,
    reason:
      'there is no reference posterior (a nine-dimensional density); the check is the recovery of the true values',
  },
  knobs: dict({
    ...simulationKnobs(8),
    trueLengthScale: real(0.3, 3, { default: 1, step: 0.1, label: 'true length scale' }),
  }),
  problem: gpClassLearnedProblem,
})
