/**
 * The setups of expectation propagation with a kernel hyperparameter: a Gaussian process whose length scale is a random
 * value, so its Gram matrix is a function of it that the factor graph cannot see through. The library runs EP inside a grid over
 * the length scale (`epGridMethod`: the length scale is set to each value of a grid, EP runs on the model with a constant
 * covariance, and the EP log evidence of each run weights it), which for a Gaussian likelihood is exact: each node's evidence is
 * the marginal likelihood `N(y; 0, K + sigma^2 I)`, so the reference here, the same quadrature over the length scale with the
 * function values as the mixture of their Gaussian conditionals, is what EP must reproduce.
 */

import { gram, rbfKernel } from '../kernels/index.ts'
import * as d from '../probprog/model/dists/index.ts'
import { defineModel } from '../probprog/model/index.ts'
import { gpPosteriorGivenLengthScale } from '../probprog/models/index.ts'
import { dict } from '../space/index.ts'
import { clip, reshape, type Value } from '../tensor/index.ts'
import type { Config } from '../trace/index.ts'
import { type AtomProblem, defineAtom, type EpAtom, type ExactMarginal } from './ep-atoms.ts'

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
    const mean = given.reduce((a, g, i) => a + (g.mean[j] ?? 0) * (w[i] ?? 0), 0)
    const second = given.reduce((a, g, i) => a + ((g.sd[j] ?? 0) ** 2 + (g.mean[j] ?? 0) ** 2) * (w[i] ?? 0), 0)
    const sd = Math.sqrt(second - mean * mean)
    out[`f[${j}]`] = {
      mean,
      sd,
      pdf: (v) =>
        given.reduce((a, g, i) => {
          const s = g.sd[j] ?? 1
          return (
            a + (w[i] ?? 0) * (Math.exp((-0.5 * (v - (g.mean[j] ?? 0)) ** 2) / (s * s)) / (s * Math.sqrt(2 * Math.PI)))
          )
        }, 0),
      range: [mean - 6 * sd, mean + 6 * sd],
    }
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
 * import { gpLengthScaleAtom } from 'omai-engine/bayesian-examples'
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
  expected: 'runs',
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
