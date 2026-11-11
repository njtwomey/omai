/**
 * The atoms of a learned covariance matrix: a multivariate normal whose precision matrix is a Wishart latent (the conjugate case, exact when
 * the mean is known), a bivariate normal with an unknown correlation (an LKJ prior, carried as a beta variable) and unknown standard
 * deviations, and correlated random intercepts and slopes (a Gaussian vector per group with a Wishart precision and a random population
 * mean). Each has a basic form with fixed numbers and a `-learned` twin simulated from a stated truth. The references are exact: the closed-form
 * Wishart posterior, a one-dimensional integral over the correlation, and three-dimensional grids over the entries of the precision matrix (or the
 * two precisions and the correlation) with the means and the random effects integrated out in closed form.
 */

import { foldIn, key, normal } from '../../random/index.ts'
import { dict, int, real } from '../../space/index.ts'
import { toFlat } from '../../tensor/index.ts'
import type { Config } from '../../trace/index.ts'
import { configNumber } from '../inference/method.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { type CoordinateBox, type ExactMarginal, logWeights, logWishart2, posterior3 } from '../validation/index.ts'
import {
  type AtomParts,
  type AtomProblem,
  type AtomRecoverySpec,
  defineAtom,
  type EpAtom,
  type FigureCurve,
} from './ep-atoms.ts'
import { simulationKnobs } from './ep-atoms-extended-conjugate.ts'
import { recovery, tolerance } from './ep-tolerance.ts'

/** The 2 by 2 matrix `[[a, b], [b, c]]` as a nested array. */
const matrix2 = (a: number, b: number, c: number): number[][] => [
  [a, b],
  [b, c],
]

/** The inverse of a symmetric 2 by 2 matrix given by its three entries. */
function inverse2(a: number, b: number, c: number): [number, number, number] {
  const det = a * c - b * b
  return [c / det, -b / det, a / det]
}

/** The points of the two-sd ellipse of a bivariate normal. */
function ellipse(
  name: string,
  colour: string | undefined,
  centre: readonly number[],
  cov: readonly [number, number, number],
): FigureCurve {
  const [a, b, c] = cov
  const l11 = Math.sqrt(Math.max(a, 1e-12))
  const l21 = b / l11
  const l22 = Math.sqrt(Math.max(c - l21 * l21, 1e-12))
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i <= 48; i++) {
    const t = (2 * Math.PI * i) / 48
    const [u, v] = [2 * Math.cos(t), 2 * Math.sin(t)]
    x.push((centre[0] ?? 0) + l11 * u)
    y.push((centre[1] ?? 0) + l21 * u + l22 * v)
  }
  return { name, ...(colour === undefined ? {} : { colour }), x, y }
}

/** `n` draws `mean + L z` of a bivariate normal with sds `(s1, s2)` and correlation `rho`, as rows. */
function simulatePairs(
  seed: number,
  n: number,
  mean: readonly number[],
  s1: number,
  s2: number,
  rho: number,
): number[][] {
  const z = toFlat(normal(foldIn(key(seed), 0), [n, 2]))
  return Array.from({ length: n }, (_, i) => {
    const [u, v] = [z[2 * i] ?? 0, z[2 * i + 1] ?? 0]
    return [(mean[0] ?? 0) + s1 * u, (mean[1] ?? 0) + rho * s2 * u + s2 * Math.sqrt(1 - rho * rho) * v]
  })
}

/** The closed-form marginals of the three entries of a Wishart matrix `W(nu, S)` (the diagonal is a gamma, the off-diagonal a numeric marginal). */
function wishartMarginals(nu: number, scale: readonly [number, number, number, number]): Record<string, ExactMarginal> {
  const [s11, s12, s22] = [scale[0], scale[1], scale[3]]
  const mean = [nu * s11, nu * s12, nu * s22] as const
  const sd = [Math.sqrt(2 * nu) * s11, Math.sqrt(nu * (s12 * s12 + s11 * s22)), Math.sqrt(2 * nu) * s22] as const
  const grid = posterior3(
    [
      { lo: Math.max(1e-6, mean[0] - 6 * sd[0]), hi: mean[0] + 8 * sd[0], min: 1e-9, max: Number.POSITIVE_INFINITY },
      {
        lo: mean[1] - 6 * sd[1],
        hi: mean[1] + 6 * sd[1],
        min: Number.NEGATIVE_INFINITY,
        max: Number.POSITIVE_INFINITY,
      },
      { lo: Math.max(1e-6, mean[2] - 6 * sd[2]), hi: mean[2] + 8 * sd[2], min: 1e-9, max: Number.POSITIVE_INFINITY },
    ],
    logWishart2(nu, scale),
  )
  // the moments are the Wishart's own, the densities the grid's
  const entry = (i: 0 | 1 | 2): ExactMarginal => {
    const g = grid.marginal(i)
    return {
      mean: mean[i],
      sd: sd[i],
      pdf: g.pdf,
      get range() {
        return g.range ?? [mean[i] - 8 * sd[i], mean[i] + 8 * sd[i]]
      },
    }
  }
  return { 'Lambda[0]': entry(0), 'Lambda[1]': entry(1), 'Lambda[3]': entry(2) }
}

const PRECISION_BOX = (mean: number, sd: number): CoordinateBox => ({
  lo: Math.max(1e-6, mean - 5 * sd),
  hi: mean + 9 * sd,
  min: 1e-9,
  max: Number.POSITIVE_INFINITY,
})
const OFFDIAGONAL_BOX = (mean: number, sd: number): CoordinateBox => ({
  lo: mean - 6 * sd,
  hi: mean + 6 * sd,
  min: Number.NEGATIVE_INFINITY,
  max: Number.POSITIVE_INFINITY,
})

/** The starting boxes of the three entries of a precision matrix around an estimate `(l11, l12, l22)` known to within about `sqrt(2 / (dof - 3))` of itself. */
function boxesAround(
  est: readonly [number, number, number],
  dof: number,
): [CoordinateBox, CoordinateBox, CoordinateBox] {
  const rel = Math.sqrt(2 / Math.max(dof - 3, 1))
  const wide = Math.sqrt(est[0] * est[2]) * rel * 1.5
  return [
    {
      lo: Math.max(1e-6, est[0] * (1 - 5 * rel)),
      hi: est[0] * (1 + 8 * rel) + 1e-3,
      min: 1e-9,
      max: Number.POSITIVE_INFINITY,
    },
    { lo: est[1] - 7 * wide, hi: est[1] + 7 * wide, min: Number.NEGATIVE_INFINITY, max: Number.POSITIVE_INFINITY },
    {
      lo: Math.max(1e-6, est[2] * (1 - 5 * rel)),
      hi: est[2] * (1 + 8 * rel) + 1e-3,
      min: 1e-9,
      max: Number.POSITIVE_INFINITY,
    },
  ]
}

/** The three entries of a Wishart prior's boxes from its mean and spread. */
function precisionBoxes(
  nu: number,
  scale: readonly [number, number, number, number],
): [CoordinateBox, CoordinateBox, CoordinateBox] {
  const [s11, s12, s22] = [scale[0], scale[1], scale[3]]
  return [
    PRECISION_BOX(nu * s11, Math.sqrt(2 * nu) * s11),
    OFFDIAGONAL_BOX(nu * s12, Math.sqrt(nu * (s12 * s12 + s11 * s22))),
    PRECISION_BOX(nu * s22, Math.sqrt(2 * nu) * s22),
  ]
}

// ---------------------------------------------------------------------------------------------------------------------
// The multivariate normal with a Wishart precision, mean known: the conjugate case
// ---------------------------------------------------------------------------------------------------------------------

const WISHART_ROWS = [
  [1.1, 0.7],
  [-0.4, -0.5],
  [0.9, 1.2],
  [0.3, 0.5],
  [-1.3, -0.4],
  [0.2, 0.9],
]
const KNOWN_MEAN = [0.1, 0.2]

function mvnWishartProblem(k: Config): AtomParts {
  const nu = configNumber(k, 'df')
  const prior = configNumber(k, 'priorScale')
  const rows = WISHART_ROWS.slice(0, configNumber(k, 'n'))
  const scale = matrix2(prior, 0, prior)
  // region mvn-wishart
  const definition = defineModel('mvn-wishart', (m) => {
    const row = m.range('row')
    // the precision matrix of the observations: a Wishart prior
    const lambda = m.latent('Lambda', d.wishart(nu, scale))
    const y = m.data('y', { dims: [row], eventRank: 1 })
    m.observe('y', d.multivariateNormal(KNOWN_MEAN, { precision: lambda }), y)
  })
  // endregion
  // the exact posterior: nu + n and the inverse scale plus the scatter about the known mean
  const n = rows.length
  let [s11, s12, s22] = [1 / prior, 0, 1 / prior]
  for (const y of rows) {
    const [r1, r2] = [(y[0] as number) - (KNOWN_MEAN[0] as number), (y[1] as number) - (KNOWN_MEAN[1] as number)]
    s11 += r1 * r1
    s12 += r1 * r2
    s22 += r2 * r2
  }
  const [i11, i12, i22] = inverse2(s11, s12, s22)
  const posterior = wishartMarginals(nu + n, [i11, i12, i12, i22])
  const cov = (m: Readonly<Record<string, number>>): [number, number, number] =>
    inverse2(m['Lambda[0]'] ?? 1, m['Lambda[1]'] ?? 0, m['Lambda[3]'] ?? 1)
  return {
    build: {
      definition,
      data: { y: rows },
      quantities: [
        { name: 'Lambda[0]', label: 'Lambda 11', support: 'positive' },
        { name: 'Lambda[1]', label: 'Lambda 12', support: 'real' },
        { name: 'Lambda[3]', label: 'Lambda 22', support: 'positive' },
      ],
    },
    reference: {
      exact: posterior,
      referenceSource: 'the closed-form Wishart posterior (the marginal of the off-diagonal entry by quadrature)',
    },
    present: {
      equations: [
        `Lambda ~ Wishart(${nu}, ${prior} I)   (the precision matrix; prior mean ${nu * prior} I)`,
        `y_i ~ N((${KNOWN_MEAN.join(', ')}), Lambda^-1), i = 1..${n}   (the mean is known)`,
        `posterior: Wishart(${nu + n}, (Lambda0^-1 + sum (y - m)(y - m)')^-1), exactly`,
      ],
      figure: {
        caption:
          'The observed pairs and the covariance ellipse (two sds) the posterior mean precision implies, beside the exact one.',
        xLabel: 'y1',
        yLabel: 'y2',
        points: [{ name: 'observations', x: rows.map((r) => r[0] as number), y: rows.map((r) => r[1] as number) }],
        answer: (mm) => [ellipse('two sds', undefined, KNOWN_MEAN, cov(mm))],
      },
    },
  }
}

/**
 * A multivariate normal with a known mean and a Wishart precision matrix, the conjugate case of a learned covariance: the posterior is the
 * Wishart with the degrees of freedom and the scatter of the rows added, and EP, which sends the exact conjugate message, reaches it in one sweep.
 *
 * @index estimator EP atom: a multivariate normal with a Wishart precision matrix and a known mean (exact).
 * @example
 * ```ts
 * import { mvnWishartAtom } from 'omai-engine/probprog/ep'
 * mvnWishartAtom.expected // => 'runs'
 * ```
 */
export const mvnWishartAtom: EpAtom = defineAtom({
  name: 'mvn-wishart',
  title: 'Multivariate normal with a Wishart precision',
  what: 'Six pairs from a bivariate normal with a known mean, whose precision matrix is a Wishart latent: the conjugate update of the matrix, exact in one sweep, checked against the closed-form posterior.',
  factors: ['prior (wishart)', 'mvn-wishart (observed vector, known mean)'],
  family: 'covariance',
  reference: 'exact',
  tolerance: tolerance(
    'exact-moments',
    'the posterior is exactly Wishart and EP carries it exactly; the curves compare the gamma and normal shapes the page draws for the entries with the exact marginals (the off-diagonal entry is a little skewed), so the distance is small, not zero',
  ),
  knobs: dict({
    df: real(2.2, 12, { default: 4, step: 0.2, label: 'prior degrees of freedom' }),
    priorScale: real(0.2, 2, { default: 0.5, step: 0.05, label: 'prior scale (mean precision is df x this)' }),
    n: int(1, 6, { default: 6, label: 'observations' }),
  }),
  parts: mvnWishartProblem,
})

// ---------------------------------------------------------------------------------------------------------------------
// The same with the mean unknown, learned from simulated data
// ---------------------------------------------------------------------------------------------------------------------

const WISHART_RECOVERY = recovery(
  'calibrated',
  'EP matches the exact posterior here (a Wishart that the prior of 3 degrees of freedom pulls towards the identity), so over seeds drawn from the truth the 90% intervals hold it about 0.85 to 0.95 of the time; 40 seeds leave a binomial spread of about 0.05',
)

const CORRELATION_RECOVERY = recovery(
  'standard',
  'one plated factor: EP carries the marginals of the exact posterior (gammas for the precisions, a beta for the correlation, a Gaussian for the mean), whose intervals hold the truth close to nine times in ten; the correlation, bounded and skewed, has errors a third larger than its spread says; 40 seeds leave a binomial spread of about 0.05',
)

function mvnWishartLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueMean1, trueMean2] = [configNumber(k, 'trueMean1'), configNumber(k, 'trueMean2')]
  const [sd1, sd2, rho] = [configNumber(k, 'trueSd1'), configNumber(k, 'trueSd2'), configNumber(k, 'trueRho')]
  const nu = configNumber(k, 'df')
  const priorSd = configNumber(k, 'priorSd')
  const truth = [trueMean1, trueMean2]
  // the matrix prior has mean the identity: nu times the scale
  const scale = matrix2(1 / nu, 0, 1 / nu)
  const spread = matrix2(priorSd ** 2, 0, priorSd ** 2)
  // region mvn-wishart-learned
  // the ground truth the data are drawn from: the model does not know it
  const rows = simulatePairs(seed, n, truth, sd1, sd2, rho)
  const definition = defineModel('mvn-wishart-learned', (m) => {
    const row = m.range('row')
    const coef = m.range('coef', { size: 2 })
    // the mean and the precision matrix: both unknown
    const vague = d.multivariateNormal([0, 0], { covariance: spread })
    const mu = m.latent('mu', vague, { dims: [coef] })
    const lambda = m.latent('Lambda', d.wishart(nu, scale))
    const y = m.data('y', { dims: [row], eventRank: 1 })
    m.observe('y', d.multivariateNormal(mu, { precision: lambda }, { over: coef }), y)
  })
  // endregion
  // The exact posterior of the precision matrix with the mean integrated out; the mean is a Gaussian mixture over it.
  const ybar = [0, 1].map((c) => rows.reduce((s, r) => s + (r[c] as number), 0) / n)
  let [t11, t12, t22] = [0, 0, 0]
  for (const r of rows) {
    const [a, b] = [(r[0] as number) - (ybar[0] as number), (r[1] as number) - (ybar[1] as number)]
    t11 += a * a
    t12 += a * b
    t22 += b * b
  }
  const prior = logWishart2(nu, [scale[0]?.[0] as number, 0, 0, scale[1]?.[1] as number])
  const c0 = priorSd ** 2
  const logDensity = (a: number, b: number, c: number): number => {
    const lp = prior(a, b, c)
    if (lp === Number.NEGATIVE_INFINITY) return lp
    const det = a * c - b * b
    // ybar ~ N(0, C0 I + Lambda^-1 / n): the mean integrated out
    const [v11, v12, v22] = [c0 + c / (det * n), -b / (det * n), c0 + a / (det * n)]
    const vd = v11 * v22 - v12 * v12
    const q =
      ((ybar[0] as number) ** 2 * v22 -
        2 * (ybar[0] as number) * (ybar[1] as number) * v12 +
        (ybar[1] as number) ** 2 * v11) /
      vd
    return lp + 0.5 * (n - 1) * Math.log(det) - 0.5 * (a * t11 + 2 * b * t12 + c * t22) - 0.5 * Math.log(vd) - 0.5 * q
  }
  // where the box starts: the Wishart posterior mean with the mean integrated out and the prior's inverse scale (nu I) added to the scatter, always positive definite
  const [e11, e12, e22] = inverse2(nu + t11, t12, nu + t22)
  const estimate: [number, number, number] = [(nu + n - 1) * e11, (nu + n - 1) * e12, (nu + n - 1) * e22]
  const grid = posterior3(boxesAround(estimate, nu + n), logDensity)
  const meanMix = (c: 0 | 1): ExactMarginal =>
    grid.mixture((a, b, cc) => {
      // mu | Lambda ~ N(A^-1 n Lambda ybar, A^-1) with A = I / C0 + n Lambda
      const [p11, p12, p22] = [1 / c0 + n * a, n * b, 1 / c0 + n * cc]
      const [c11, c12, c22] = inverse2(p11, p12, p22)
      const [h1, h2] = [
        n * (a * (ybar[0] as number) + b * (ybar[1] as number)),
        n * (b * (ybar[0] as number) + cc * (ybar[1] as number)),
      ]
      return c === 0 ? { mean: c11 * h1 + c12 * h2, variance: c11 } : { mean: c12 * h1 + c22 * h2, variance: c22 }
    })
  const trueCov = [sd1 * sd1, rho * sd1 * sd2, sd2 * sd2] as const
  const trueLambda = inverse2(trueCov[0], trueCov[1], trueCov[2])
  const cov = (mm: Readonly<Record<string, number>>): [number, number, number] =>
    inverse2(mm['Lambda[0]'] ?? 1, mm['Lambda[1]'] ?? 0, mm['Lambda[3]'] ?? 1)
  return {
    build: {
      definition,
      data: { y: rows },
      quantities: [
        { name: 'mu[0]', label: 'mu 1', support: 'real', truth: trueMean1 },
        { name: 'mu[1]', label: 'mu 2', support: 'real', truth: trueMean2 },
        { name: 'Lambda[0]', label: 'Lambda 11', support: 'positive', truth: trueLambda[0] },
        { name: 'Lambda[1]', label: 'Lambda 12', support: 'real', truth: trueLambda[1] },
        { name: 'Lambda[3]', label: 'Lambda 22', support: 'positive', truth: trueLambda[2] },
      ],
    },
    reference: {
      exact: {
        'mu[0]': meanMix(0),
        'mu[1]': meanMix(1),
        'Lambda[0]': grid.marginal(0),
        'Lambda[1]': grid.marginal(1),
        'Lambda[3]': grid.marginal(2),
      },
      referenceSource:
        'quadrature of the three entries of the precision matrix, the mean integrated out in closed form',
    },
    present: {
      equations: [
        `truth: mean (${trueMean1}, ${trueMean2}), sds (${sd1}, ${sd2}), correlation ${rho}; ${n} pairs drawn with seed ${seed}`,
        `mu ~ N(0, ${priorSd}^2 I)   (the unknown mean)`,
        `Lambda ~ Wishart(${nu}, I / ${nu})   (the unknown precision matrix, prior mean the identity)`,
        `y_i ~ N(mu, Lambda^-1), i = 1..${n}`,
      ],
      figure: {
        caption:
          'The simulated pairs, the true covariance ellipse (green) and the one the posterior mean implies (two sds).',
        xLabel: 'y1',
        yLabel: 'y2',
        points: [{ name: 'observations', x: rows.map((r) => r[0] as number), y: rows.map((r) => r[1] as number) }],
        curves: [ellipse('true', 'tab2', truth, trueCov)],
        answer: (mm) => [ellipse('fitted', undefined, [mm['mu[0]'] ?? 0, mm['mu[1]'] ?? 0], cov(mm))],
      },
    },
  }
}

/**
 * A bivariate normal with an unknown mean and an unknown precision matrix (a Wishart), learned from data simulated from a stated mean and
 * covariance: the vector is read through one plated factor, EP matches the Wishart to the moments of its tilted distribution on a Bartlett grid,
 * and the page checks the truth is recovered and how often the 90% intervals hold it.
 *
 * @index estimator EP atom: a bivariate normal with unknown mean and Wishart precision matrix, recovered from simulated data.
 * @example
 * ```ts
 * import { mvnWishartLearnedAtom } from 'omai-engine/probprog/ep'
 * mvnWishartLearnedAtom.extends // => 'mvn-wishart'
 * ```
 */
export const mvnWishartLearnedAtom: EpAtom = defineAtom({
  name: 'mvn-wishart-learned',
  title: 'Multivariate normal with a Wishart precision: parameters learned',
  extends: 'mvn-wishart',
  recovery: WISHART_RECOVERY,
  what: 'The bivariate normal of the basic atom with the mean also unknown (a Gaussian prior) and the data simulated from a stated mean and covariance: the Wishart is no longer conjugate (the mean and the precision are independent in the prior), so EP integrates the mean out and matches the matrix by quadrature.',
  factors: ['prior (gaussian-mv)', 'prior (wishart)', 'mvn-wishart (plate of rows, random mean)'],
  family: 'covariance',
  reference: 'exact',
  tolerance: tolerance(
    'plated',
    'with one plated factor EP matches the exact posterior of the mean and the matrix by its moments (a Gaussian and a Wishart), so the means are exact and the spreads are the Wishart the matrix is closest to: the posterior is very nearly one at these sizes',
  ),
  knobs: dict({
    ...simulationKnobs(40),
    trueMean1: real(-2, 2, { default: 0.8, step: 0.1, label: 'true mean 1' }),
    trueMean2: real(-2, 2, { default: -0.4, step: 0.1, label: 'true mean 2' }),
    trueSd1: real(0.3, 2.5, { default: 1, step: 0.05, label: 'true sd 1' }),
    trueSd2: real(0.3, 2.5, { default: 0.8, step: 0.05, label: 'true sd 2' }),
    trueRho: real(-0.95, 0.95, { default: 0.6, step: 0.05, label: 'true correlation' }),
    df: real(2.2, 12, { default: 3, step: 0.2, label: 'prior degrees of freedom' }),
    priorSd: real(1, 10, { default: 4, step: 0.5, label: 'prior sd of the mean' }),
  }),
  parts: mvnWishartLearnedProblem,
})

// ---------------------------------------------------------------------------------------------------------------------
// A bivariate normal with an unknown correlation
// ---------------------------------------------------------------------------------------------------------------------

const PAIRS = [
  [0.9, 1.1],
  [-0.3, -0.6],
  [1.4, 0.8],
  [0.2, 0.5],
  [-1.1, -0.7],
  [0.6, 1.2],
  [-0.2, 0.1],
  [1.0, 0.4],
]

/** The posterior of a correlation `rho` with known sds and mean zero: `(1 - rho^2)^(eta - 1 - n / 2) exp(-(sxx - 2 rho sxy + syy) / (2 (1 - rho^2)))`, normalised numerically. */
function correlationMarginal(eta: number, n: number, sxx: number, sxy: number, syy: number): ExactMarginal {
  const N = 4000
  const x = Array.from({ length: N }, (_, i) => -1 + (2 * (i + 0.5)) / N)
  const lw = x.map(
    (rho) => (eta - 1 - n / 2) * Math.log(1 - rho * rho) - (sxx - 2 * rho * sxy + syy) / (2 * (1 - rho * rho)),
  )
  const { weights: w } = logWeights(lw)
  const total = w.reduce((s, v) => s + v, 0)
  const h = 2 / N
  const mean = w.reduce((s, v, i) => s + (v * (x[i] as number)) / total, 0)
  const sd = Math.sqrt(w.reduce((s, v, i) => s + (v * ((x[i] as number) - mean) ** 2) / total, 0))
  return {
    mean,
    sd,
    pdf: (r) =>
      r <= -1 || r >= 1 ? 0 : (w[Math.min(N - 1, Math.max(0, Math.floor((r + 1) / h)))] as number) / total / h,
    range: [Math.max(-1, mean - 8 * sd), Math.min(1, mean + 8 * sd)],
  }
}

function bivariateCorrelationProblem(k: Config): AtomParts {
  const eta = configNumber(k, 'eta')
  const rows = PAIRS.slice(0, configNumber(k, 'n'))
  // region bivariate-correlation
  const definition = defineModel('bivariate-correlation', (m) => {
    const row = m.range('row')
    // the correlation: the LKJ prior of a 2 by 2 matrix (uniform at eta = 1)
    const rho = m.latent('rho', d.correlation(eta))
    const y = m.data('y', { dims: [row], eventRank: 1 })
    m.observe('y', d.bivariateNormal([0, 0], { scale1: 1, scale2: 1, rho }), y)
  })
  // endregion
  const n = rows.length
  const sxx = rows.reduce((s, r) => s + (r[0] as number) ** 2, 0)
  const syy = rows.reduce((s, r) => s + (r[1] as number) ** 2, 0)
  const sxy = rows.reduce((s, r) => s + (r[0] as number) * (r[1] as number), 0)
  return {
    build: {
      definition,
      data: { y: rows },
      quantities: [{ name: 'rho', label: 'rho (the correlation)', support: 'real' }],
    },
    reference: {
      exact: { rho: correlationMarginal(eta, n, sxx, sxy, syy) },
      referenceSource: 'the posterior of the correlation, one-dimensional, by quadrature',
    },
    present: {
      equations: [
        `rho ~ LKJ(2, ${eta})   (p(rho) = (1 - rho^2)^${eta - 1} on (-1, 1))`,
        `y_i ~ N(0, [[1, rho], [rho, 1]]), i = 1..${n}   (mean and sds known)`,
        `p(rho | y) = (1 - rho^2)^(${eta - 1} - ${n} / 2) exp(-(sum y1^2 - 2 rho sum y1 y2 + sum y2^2) / (2 (1 - rho^2)))`,
      ],
      figure: {
        caption: 'The observed pairs and the unit-variance ellipse (two sds) the posterior mean correlation implies.',
        xLabel: 'y1',
        yLabel: 'y2',
        points: [{ name: 'observations', x: rows.map((r) => r[0] as number), y: rows.map((r) => r[1] as number) }],
        answer: (mm) => [ellipse('two sds', undefined, [0, 0], [1, mm['rho'] ?? 0, 1])],
      },
    },
  }
}

/**
 * A bivariate normal with known standard deviations and mean and an unknown correlation under the LKJ prior (a beta variable on `(rho + 1) / 2`):
 * the likelihood of the rows is a function of three scatter sums and the correlation, a one-dimensional integral EP matches to a beta.
 *
 * @index estimator EP atom: a bivariate normal with an unknown correlation (LKJ prior).
 * @example
 * ```ts
 * import { bivariateCorrelationAtom } from 'omai-engine/probprog/ep'
 * bivariateCorrelationAtom.expected // => 'runs'
 * ```
 */
export const bivariateCorrelationAtom: EpAtom = defineAtom({
  name: 'bivariate-correlation',
  title: 'Bivariate normal with an unknown correlation',
  what: 'Eight pairs from a bivariate normal with known zero means and unit sds, whose correlation has the LKJ prior of a 2 by 2 matrix (a beta variable on (rho + 1) / 2): a one-dimensional numeric site, checked against the posterior of the correlation by quadrature.',
  factors: ['prior (beta)', 'mvn-sd-corr (plate of rows, known mean and sds)'],
  family: 'covariance',
  reference: 'exact',
  tolerance: tolerance(
    'measured',
    'the posterior of a strong correlation from a handful of pairs is skewed against the boundary: EP carries the beta that matches its two log moments, which has the right mean (0.02 sd) and a spread 17% too small here (0.83 of the exact at eight pairs, 0.92 at three; near 0.99 for the learned twin at a moderate correlation); the curve the page draws is a normal matched to the belief, which adds to the distance',
    { meanZ: 0.1, sdRatio: [0.75, 1.25], ks: 0.25 },
  ),
  knobs: dict({
    eta: real(0.5, 8, { default: 1, step: 0.25, label: 'LKJ concentration eta' }),
    n: int(2, 8, { default: 8, label: 'observations' }),
  }),
  parts: bivariateCorrelationProblem,
})

function bivariateCorrelationLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueMean1, trueMean2] = [configNumber(k, 'trueMean1'), configNumber(k, 'trueMean2')]
  const [sd1, sd2, rho0] = [configNumber(k, 'trueSd1'), configNumber(k, 'trueSd2'), configNumber(k, 'trueRho')]
  const [a, b] = [configNumber(k, 'shape'), configNumber(k, 'rate')]
  const eta = configNumber(k, 'eta')
  const priorSd = configNumber(k, 'priorSd')
  const truth = [trueMean1, trueMean2]
  const spread = matrix2(priorSd ** 2, 0, priorSd ** 2)
  // region bivariate-correlation-learned
  // the ground truth the data are drawn from: the model does not know it
  const rows = simulatePairs(seed, n, truth, sd1, sd2, rho0)
  const definition = defineModel('bivariate-correlation-learned', (m) => {
    const row = m.range('row')
    const coef = m.range('coef', { size: 2 })
    // the two precisions (one over each variance), the correlation and the mean: all unknown
    const tau1 = m.latent('tau1', d.gamma(a, b))
    const tau2 = m.latent('tau2', d.gamma(a, b))
    const rho = m.latent('rho', d.correlation(eta))
    const vague = d.multivariateNormal([0, 0], { covariance: spread })
    const mu = m.latent('mu', vague, { dims: [coef] })
    const sd1 = ops.div(1, ops.sqrt(tau1))
    const sd2 = ops.div(1, ops.sqrt(tau2))
    const y = m.data('y', { dims: [row], eventRank: 1 })
    m.observe('y', d.bivariateNormal(mu, { scale1: sd1, scale2: sd2, rho }, { over: coef }), y)
  })
  // endregion
  const ybar = [0, 1].map((c) => rows.reduce((s, r) => s + (r[c] as number), 0) / n)
  let [t11, t12, t22] = [0, 0, 0]
  for (const r of rows) {
    const [u, v] = [(r[0] as number) - (ybar[0] as number), (r[1] as number) - (ybar[1] as number)]
    t11 += u * u
    t12 += u * v
    t22 += v * v
  }
  const c0 = priorSd ** 2
  const logDensity = (tau1: number, tau2: number, rho: number): number => {
    if (!(tau1 > 0 && tau2 > 0 && rho > -1 && rho < 1)) return Number.NEGATIVE_INFINITY
    const [s11, s22] = [1 / tau1, 1 / tau2]
    const s12 = rho * Math.sqrt(s11 * s22)
    const det = s11 * s22 - s12 * s12
    const [l11, l12, l22] = [s22 / det, -s12 / det, s11 / det]
    // ybar ~ N(0, C0 I + Sigma / n): the mean integrated out
    const [v11, v12, v22] = [c0 + s11 / n, s12 / n, c0 + s22 / n]
    const vd = v11 * v22 - v12 * v12
    const q =
      ((ybar[0] as number) ** 2 * v22 -
        2 * (ybar[0] as number) * (ybar[1] as number) * v12 +
        (ybar[1] as number) ** 2 * v11) /
      vd
    return (
      (a - 1) * Math.log(tau1) -
      b * tau1 +
      (a - 1) * Math.log(tau2) -
      b * tau2 +
      (eta - 1) * Math.log(1 - rho * rho) -
      0.5 * (n - 1) * Math.log(det) -
      0.5 * (l11 * t11 + 2 * l12 * t12 + l22 * t22) -
      0.5 * Math.log(vd) -
      0.5 * q
    )
  }
  const gammaSd = Math.sqrt(a) / b
  const precisionHat = [n / t11, n / t22] as const
  const grid = posterior3(
    [
      { lo: 1e-3, hi: 3 * precisionHat[0] + (a / b) * 3 + 8 * gammaSd, min: 1e-9, max: Number.POSITIVE_INFINITY },
      { lo: 1e-3, hi: 3 * precisionHat[1] + (a / b) * 3 + 8 * gammaSd, min: 1e-9, max: Number.POSITIVE_INFINITY },
      { lo: -0.995, hi: 0.995, min: -0.9999, max: 0.9999 },
    ],
    logDensity,
  )
  const meanMix = (c: 0 | 1): ExactMarginal =>
    grid.mixture((tau1, tau2, rho) => {
      const [s11, s22] = [1 / tau1, 1 / tau2]
      const s12 = rho * Math.sqrt(s11 * s22)
      // mu | Sigma, y ~ N(P^-1 C0^-1... ) with the posterior precision I / C0 + n Sigma^-1: the mean is the shrunk sample mean
      const det = s11 * s22 - s12 * s12
      const [l11, l12, l22] = [(n * s22) / det, (-n * s12) / det, (n * s11) / det]
      const [p11, p12, p22] = [1 / c0 + l11, l12, 1 / c0 + l22]
      const [c11, c12, c22] = inverse2(p11, p12, p22)
      const [h1, h2] = [
        l11 * (ybar[0] as number) + l12 * (ybar[1] as number),
        l12 * (ybar[0] as number) + l22 * (ybar[1] as number),
      ]
      return c === 0 ? { mean: c11 * h1 + c12 * h2, variance: c11 } : { mean: c12 * h1 + c22 * h2, variance: c22 }
    })
  return {
    build: {
      definition,
      data: { y: rows },
      quantities: [
        { name: 'mu[0]', label: 'mu 1', support: 'real', truth: trueMean1 },
        { name: 'mu[1]', label: 'mu 2', support: 'real', truth: trueMean2 },
        { name: 'tau1', label: 'tau 1 (precision of y1)', support: 'positive', truth: 1 / (sd1 * sd1) },
        { name: 'tau2', label: 'tau 2 (precision of y2)', support: 'positive', truth: 1 / (sd2 * sd2) },
        { name: 'rho', label: 'rho (the correlation)', support: 'real', truth: rho0 },
      ],
    },
    reference: {
      exact: {
        'mu[0]': meanMix(0),
        'mu[1]': meanMix(1),
        tau1: grid.marginal(0),
        tau2: grid.marginal(1),
        rho: grid.marginal(2),
      },
      referenceSource: 'quadrature of the two precisions and the correlation, the mean integrated out in closed form',
    },
    present: {
      equations: [
        `truth: mean (${trueMean1}, ${trueMean2}), sds (${sd1}, ${sd2}), correlation ${rho0}; ${n} pairs drawn with seed ${seed}`,
        `tau1, tau2 ~ Gamma(${a}, ${b})   (the precisions: sd = 1 / sqrt(tau))`,
        `rho ~ LKJ(2, ${eta});  mu ~ N(0, ${priorSd}^2 I)`,
        `y_i ~ N(mu, [[1 / tau1, rho / sqrt(tau1 tau2)], [rho / sqrt(tau1 tau2), 1 / tau2]]), i = 1..${n}`,
      ],
      figure: {
        caption:
          'The simulated pairs, the true covariance ellipse (green) and the one the posterior means imply (two sds).',
        xLabel: 'y1',
        yLabel: 'y2',
        points: [{ name: 'observations', x: rows.map((r) => r[0] as number), y: rows.map((r) => r[1] as number) }],
        curves: [ellipse('true', 'tab2', truth, [sd1 * sd1, rho0 * sd1 * sd2, sd2 * sd2])],
        answer: (mm) => {
          const [s1, s2] = [1 / Math.sqrt(mm['tau1'] ?? 1), 1 / Math.sqrt(mm['tau2'] ?? 1)]
          const r = mm['rho'] ?? 0
          return [ellipse('fitted', undefined, [mm['mu[0]'] ?? 0, mm['mu[1]'] ?? 0], [s1 * s1, r * s1 * s2, s2 * s2])]
        },
      },
    },
  }
}

/**
 * A bivariate normal with an unknown mean, unknown standard deviations (gamma precisions) and an unknown correlation (LKJ), learned from data
 * simulated from a stated truth: one plated factor over the two precisions and the correlation, integrated on a zooming three-dimensional grid.
 *
 * @index estimator EP atom: a bivariate normal with unknown mean, standard deviations and correlation, recovered from simulated data.
 * @example
 * ```ts
 * import { bivariateCorrelationLearnedAtom } from 'omai-engine/probprog/ep'
 * bivariateCorrelationLearnedAtom.extends // => 'bivariate-correlation'
 * ```
 */
export const bivariateCorrelationLearnedAtom: EpAtom = defineAtom({
  name: 'bivariate-correlation-learned',
  title: 'Bivariate normal with an unknown correlation: parameters learned',
  extends: 'bivariate-correlation',
  recovery: CORRELATION_RECOVERY,
  what: 'The bivariate normal of the basic atom with the mean and both standard deviations also unknown (a Gaussian prior, gamma precisions) and the data simulated from a stated mean, sds and correlation: three numeric coordinates, matched at once by EP.',
  factors: ['prior (gaussian-mv)', 'prior (gamma)', 'prior (beta)', 'mvn-sd-corr (plate of rows, random mean)'],
  family: 'covariance',
  reference: 'exact',
  tolerance: tolerance(
    'close',
    'one plated factor: EP matches the exact joint posterior of the two precisions, the correlation and the mean by the moments of each marginal (gammas, a beta, a Gaussian), and these are very nearly the exact marginals at 30 pairs; at a strong correlation and few pairs the correlation is skewed and its beta is a little narrower (0.89 of the exact at 10 pairs and rho 0.9)',
  ),
  knobs: dict({
    ...simulationKnobs(30),
    trueMean1: real(-2, 2, { default: 0.8, step: 0.1, label: 'true mean 1' }),
    trueMean2: real(-2, 2, { default: -0.4, step: 0.1, label: 'true mean 2' }),
    trueSd1: real(0.3, 2.5, { default: 1, step: 0.05, label: 'true sd 1' }),
    trueSd2: real(0.3, 2.5, { default: 0.7, step: 0.05, label: 'true sd 2' }),
    trueRho: real(-0.95, 0.95, { default: 0.6, step: 0.05, label: 'true correlation' }),
    shape: real(0.5, 8, { default: 2, step: 0.1, label: 'prior shape of the precisions' }),
    rate: real(0.1, 8, { default: 2, step: 0.1, label: 'prior rate of the precisions' }),
    eta: real(0.5, 8, { default: 1, step: 0.25, label: 'LKJ concentration eta' }),
    priorSd: real(1, 10, { default: 4, step: 0.5, label: 'prior sd of the mean' }),
  }),
  parts: bivariateCorrelationLearnedProblem,
})

// ---------------------------------------------------------------------------------------------------------------------
// Correlated random intercepts and slopes
// ---------------------------------------------------------------------------------------------------------------------

const TIMES = [-1.5, -0.5, 0.5, 1.5]

/** The per-group sums of a regression on `[1, t]`: `X'X`, and for each group the data's `X'y` and `y'y`. */
function groupSums(y: readonly (readonly number[])[]): {
  xtx: [number, number, number]
  xty: [number, number][]
  yty: number[]
} {
  const t = TIMES
  const xtx: [number, number, number] = [t.length, t.reduce((s, v) => s + v, 0), t.reduce((s, v) => s + v * v, 0)]
  return {
    xtx,
    xty: y.map((g) => [g.reduce((s, v) => s + v, 0), g.reduce((s, v, i) => s + v * (t[i] as number), 0)]),
    yty: y.map((g) => g.reduce((s, v) => s + v * v, 0)),
  }
}

/**
 * The log density of the precision matrix of random intercepts and slopes after the effects (and, if the mean is unknown, the mean) are
 * integrated out: each group's data are `N(X mu, X Lambda^-1 X' + sigma^2 I)`.
 */
function randomEffectsLogDensity(
  y: readonly (readonly number[])[],
  noise: number,
  mean: { readonly known: readonly number[] } | { readonly priorSd: number },
) {
  const { xtx, xty, yty } = groupSums(y)
  const s2 = noise * noise
  const g = [xtx[0] / s2, xtx[1] / s2, xtx[2] / s2] as const
  const per = (a: number, b: number, c: number) => {
    // A = Lambda + G; Woodbury: V^-1 = (I - X A^-1 X' / s2) / s2, ln |V| = n ln s2 + ln |A| - ln |Lambda|
    const A = [a + g[0], b + g[1], c + g[2]] as const
    const [i11, i12, i22] = inverse2(A[0], A[1], A[2])
    const logDetA = Math.log(A[0] * A[2] - A[1] * A[1])
    const logDetL = Math.log(a * c - b * b)
    return { i11, i12, i22, logDetV: TIMES.length * Math.log(s2) + logDetA - logDetL }
  }
  const sandwich = (i11: number, i12: number, i22: number, h: readonly number[]): number =>
    i11 * (h[0] as number) ** 2 + 2 * i12 * (h[0] as number) * (h[1] as number) + i22 * (h[1] as number) ** 2
  return {
    per,
    logLikelihood: (a: number, b: number, c: number): number => {
      const { i11, i12, i22, logDetV } = per(a, b, c)
      if ('known' in mean) {
        const mu = mean.known
        // residuals: r = y - X mu, X'r = X'y - X'X mu, r'r = y'y - 2 mu'X'y + mu'X'X mu
        let total = 0
        for (let j = 0; j < y.length; j++) {
          const xy = xty[j] as [number, number]
          const xr = [
            xy[0] - (xtx[0] * (mu[0] as number) + xtx[1] * (mu[1] as number)),
            xy[1] - (xtx[1] * (mu[0] as number) + xtx[2] * (mu[1] as number)),
          ]
          const rr =
            (yty[j] as number) -
            2 * ((mu[0] as number) * xy[0] + (mu[1] as number) * xy[1]) +
            (mu[0] as number) ** 2 * xtx[0] +
            2 * (mu[0] as number) * (mu[1] as number) * xtx[1] +
            (mu[1] as number) ** 2 * xtx[2]
          const h = [(xr[0] as number) / s2, (xr[1] as number) / s2]
          total += -0.5 * (logDetV + rr / s2 - sandwich(i11, i12, i22, h))
        }
        return total
      }
      // the mean unknown: Q_j = X'V^-1X = G - G A^-1 G, q_j = X'V^-1 y = h - G A^-1 h, c_j = y'V^-1 y = y'y / s2 - h'A^-1 h
      let [p11, p12, p22] = [1 / mean.priorSd ** 2, 0, 1 / mean.priorSd ** 2]
      let [b1, b2] = [0, 0]
      let rest = 0
      const gai = (u: readonly number[]): [number, number] => {
        const w = [i11 * (u[0] as number) + i12 * (u[1] as number), i12 * (u[0] as number) + i22 * (u[1] as number)]
        return [g[0] * (w[0] as number) + g[1] * (w[1] as number), g[1] * (w[0] as number) + g[2] * (w[1] as number)]
      }
      const gaigCol = (col: number): [number, number] => gai(col === 0 ? [g[0], g[1]] : [g[1], g[2]])
      const [c0, c1] = [gaigCol(0), gaigCol(1)]
      for (let j = 0; j < y.length; j++) {
        const xy = xty[j] as [number, number]
        const h = [xy[0] / s2, xy[1] / s2]
        p11 += g[0] - c0[0]
        p12 += g[1] - c1[0]
        p22 += g[2] - c1[1]
        const gh = gai(h)
        b1 += (h[0] as number) - gh[0]
        b2 += (h[1] as number) - gh[1]
        rest += -0.5 * (logDetV + (yty[j] as number) / s2 - sandwich(i11, i12, i22, h))
      }
      const [q11, q12, q22] = inverse2(p11, p12, p22)
      return rest + 0.5 * sandwich(q11, q12, q22, [b1, b2]) - 0.5 * Math.log(p11 * p22 - p12 * p12)
    },
    sums: { xtx, xty },
  }
}

interface RandomEffectsData {
  readonly y: number[][]
  readonly effects?: readonly (readonly number[])[]
}

function simulateGroups(
  seed: number,
  groups: number,
  mean: readonly number[],
  s1: number,
  s2: number,
  rho: number,
  noise: number,
): RandomEffectsData {
  const effects = simulatePairs(seed, groups, mean, s1, s2, rho)
  const z = toFlat(normal(foldIn(key(seed), 1), [groups, TIMES.length]))
  return {
    effects,
    y: effects.map((e, j) =>
      TIMES.map((t, i) => (e[0] as number) + (e[1] as number) * t + noise * (z[j * TIMES.length + i] ?? 0)),
    ),
  }
}

/** The group's coefficients' posterior given the matrix (and, for a known mean, the mean): `N(A^-1 (Lambda mu + X'y / s2), A^-1)`. */
function randomEffectsFigure(
  y: readonly (readonly number[])[],
  truth?: readonly (readonly number[])[],
): NonNullable<AtomProblem['figure']> {
  const groups = y.length
  const x = [TIMES[0] as number, TIMES[TIMES.length - 1] as number]
  return {
    caption:
      'The observations of each group over time, the line each group follows by the posterior mean of its intercept and slope (solid) and, where it is known, its true line (green).',
    xLabel: 'time',
    yLabel: 'response',
    points: y.map((g, j) => ({ name: `group ${j + 1}`, x: TIMES, y: g })),
    ...(truth === undefined
      ? {}
      : {
          curves: truth.map(
            (e, j): FigureCurve => ({
              name: `true ${j + 1}`,
              colour: 'tab2',
              x,
              y: x.map((t) => (e[0] as number) + (e[1] as number) * t),
            }),
          ),
        }),
    answer: (mm) =>
      Array.from({ length: groups }, (_, j) => ({
        name: `fit ${j + 1}`,
        x,
        y: x.map((t) => (mm[`theta[${2 * j}]`] ?? 0) + (mm[`theta[${2 * j + 1}]`] ?? 0) * t),
      })),
  }
}

const KNOWN_POPULATION = [0.4, -0.3] as const
const [POPULATION_1, POPULATION_2] = KNOWN_POPULATION

function randomEffectsProblem(k: Config): AtomParts {
  const groups = configNumber(k, 'groups')
  const noise = configNumber(k, 'noise')
  const nu = configNumber(k, 'df')
  const data = simulateGroups(11, groups, KNOWN_POPULATION, 0.8, 0.5, -0.5, noise)
  const scale = matrix2(1 / nu, 0, 1 / nu)
  const design = data.y.flatMap((_, j) => TIMES.map((t) => ({ g: j, t })))
  // region random-intercepts-slopes
  const definition = defineModel('random-intercepts-slopes', (m) => {
    const group = m.range('group', { size: groups })
    const coef = m.range('coef', { size: 2 })
    const obs = m.range('obs')
    // the precision matrix of the (intercept, slope) pairs: a Wishart prior; the population mean is known
    const lambda = m.latent('Lambda', d.wishart(nu, scale))
    const around = d.multivariateNormal(KNOWN_POPULATION, { precision: lambda }, { over: coef })
    const theta = m.latent('theta', around, { dims: [group, coef] })
    const x = m.data('X', { dims: [obs, coef] })
    const g = m.data('g', { dims: [obs], indexes: 'group' })
    const line = ops.sum(ops.mul(theta.at(g), x), coef)
    m.observe('y', d.normal(line, noise), m.data('y', { dims: [obs] }))
  })
  // endregion
  const ll = randomEffectsLogDensity(data.y, noise, { known: KNOWN_POPULATION })
  const prior = logWishart2(nu, [scale[0]?.[0] as number, 0, 0, scale[1]?.[1] as number])
  const grid = posterior3(precisionBoxes(nu, [scale[0]?.[0] as number, 0, 0, scale[1]?.[1] as number]), (a, b, c) => {
    const lp = prior(a, b, c)
    return lp === Number.NEGATIVE_INFINITY ? lp : lp + ll.logLikelihood(a, b, c)
  })
  const s2 = noise * noise
  const { xtx, xty } = ll.sums
  const effect = (j: number, c: 0 | 1): ExactMarginal =>
    grid.mixture((a, b, cc) => {
      const A = [a + xtx[0] / s2, b + xtx[1] / s2, cc + xtx[2] / s2] as const
      const [i11, i12, i22] = inverse2(A[0], A[1], A[2])
      const xy = xty[j] as [number, number]
      const h = [a * POPULATION_1 + b * POPULATION_2 + xy[0] / s2, b * POPULATION_1 + cc * POPULATION_2 + xy[1] / s2]
      return c === 0
        ? { mean: i11 * (h[0] as number) + i12 * (h[1] as number), variance: i11 }
        : { mean: i12 * (h[0] as number) + i22 * (h[1] as number), variance: i22 }
    })
  const exact: Record<string, ExactMarginal> = {
    'Lambda[0]': grid.marginal(0),
    'Lambda[1]': grid.marginal(1),
    'Lambda[3]': grid.marginal(2),
  }
  for (let j = 0; j < groups; j++) {
    exact[`theta[${2 * j}]`] = effect(j, 0)
    exact[`theta[${2 * j + 1}]`] = effect(j, 1)
  }
  return {
    build: {
      definition,
      data: { X: design.map((o) => [1, o.t]), g: design.map((o) => o.g), y: data.y.flat() },
      quantities: [
        { name: 'Lambda[0]', label: 'Lambda 11', support: 'positive' },
        { name: 'Lambda[1]', label: 'Lambda 12', support: 'real' },
        { name: 'Lambda[3]', label: 'Lambda 22', support: 'positive' },
        ...Array.from({ length: groups }, (_, j) => [
          {
            name: `theta[${2 * j}]`,
            label: `intercept ${j + 1}`,
            support: 'real' as const,
            pool: 'intercepts (every group)',
          },
          {
            name: `theta[${2 * j + 1}]`,
            label: `slope ${j + 1}`,
            support: 'real' as const,
            pool: 'slopes (every group)',
          },
        ]).flat(),
      ],
    },
    reference: {
      exact,
      referenceSource:
        'quadrature of the three entries of the precision matrix, the effects integrated out in closed form (each is a Gaussian mixture over it)',
    },
    present: {
      equations: [
        `Lambda ~ Wishart(${nu}, I / ${nu})   (the precision of the (intercept, slope) pairs; prior mean the identity)`,
        `theta_j ~ N((${KNOWN_POPULATION.join(', ')}), Lambda^-1), j = 1..${groups}   (the population mean is known)`,
        `y_jt ~ N(theta_j1 + theta_j2 t, ${noise}^2), t = ${TIMES.join(', ')}`,
      ],
      figure: randomEffectsFigure(data.y),
    },
  }
}

/**
 * Correlated random intercepts and slopes: a line per group whose (intercept, slope) pair is drawn from a bivariate normal with a Wishart precision
 * matrix (a known population mean), observed with known noise. EP carries a Gaussian vector per group and matches the matrix by quadrature
 * against the tilted distribution of each group.
 *
 * @index estimator EP atom: correlated random intercepts and slopes (a Wishart covariance of the effects).
 * @example
 * ```ts
 * import { randomInterceptsSlopesAtom } from 'omai-engine/probprog/ep'
 * randomInterceptsSlopesAtom.expected // => 'runs'
 * ```
 */
export const randomInterceptsSlopesAtom: EpAtom = defineAtom({
  name: 'random-intercepts-slopes',
  title: 'Correlated random intercepts and slopes',
  what: 'A line per group, its intercept and slope drawn together from a bivariate normal with a Wishart precision matrix and a known mean, observed with known noise: a Gaussian vector per group read through its coordinates, the matrix learned from the groups, checked against quadrature of the matrix with the effects integrated out.',
  factors: [
    'prior (wishart)',
    'mvn-wishart (a Gaussian vector per group)',
    'linear-regression-row (the reading of the line)',
  ],
  family: 'covariance',
  reference: 'exact',
  tolerance: tolerance(
    'plated',
    'the effects are Gaussian given the matrix and the matrix a Wishart given the effects, but not together: EP matches a Wishart to the matrix and a Gaussian to each group, and at six groups the spreads are within 1% of the exact (they were never below 0.99 from three to ten groups)',
  ),
  knobs: dict({
    groups: int(3, 10, { default: 6, label: 'groups' }),
    noise: real(0.1, 1, { default: 0.3, step: 0.05, label: 'noise sd (known)' }),
    df: real(2.2, 12, { default: 4, step: 0.2, label: 'prior degrees of freedom' }),
  }),
  parts: randomEffectsProblem,
})

function randomEffectsLearnedProblem(k: Config): AtomParts {
  const [seed, groups] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [m1, m2] = [configNumber(k, 'trueMean1'), configNumber(k, 'trueMean2')]
  const [sd1, sd2, rho] = [configNumber(k, 'trueSd1'), configNumber(k, 'trueSd2'), configNumber(k, 'trueRho')]
  const noise = configNumber(k, 'noise')
  const nu = configNumber(k, 'df')
  const priorSd = configNumber(k, 'priorSd')
  const truth = [m1, m2]
  const scale = matrix2(1 / nu, 0, 1 / nu)
  // the ground truth the data are drawn from: the model does not know it
  const data = simulateGroups(seed, groups, truth, sd1, sd2, rho, noise)
  const design = data.y.flatMap((_, j) => TIMES.map((t) => ({ g: j, t })))
  const spread = matrix2(priorSd ** 2, 0, priorSd ** 2)
  // region random-intercepts-slopes-learned
  const definition = defineModel('random-intercepts-slopes-learned', (m) => {
    const group = m.range('group', { size: groups })
    const coef = m.range('coef', { size: 2 })
    const obs = m.range('obs')
    // the population mean and the precision matrix of the (intercept, slope) pairs: both unknown
    const vague = d.multivariateNormal([0, 0], { covariance: spread })
    const mu = m.latent('mu', vague, { dims: [coef] })
    const lambda = m.latent('Lambda', d.wishart(nu, scale))
    const around = d.multivariateNormal(mu, { precision: lambda }, { over: coef })
    const theta = m.latent('theta', around, { dims: [group, coef] })
    const x = m.data('X', { dims: [obs, coef] })
    const g = m.data('g', { dims: [obs], indexes: 'group' })
    const line = ops.sum(ops.mul(theta.at(g), x), coef)
    m.observe('y', d.normal(line, noise), m.data('y', { dims: [obs] }))
  })
  // endregion
  const ll = randomEffectsLogDensity(data.y, noise, { priorSd })
  const prior = logWishart2(nu, [scale[0]?.[0] as number, 0, 0, scale[1]?.[1] as number])
  const trueCov = [sd1 * sd1, rho * sd1 * sd2, sd2 * sd2] as const
  const trueLambda = inverse2(trueCov[0], trueCov[1], trueCov[2])
  const boxes = precisionBoxes(nu, [scale[0]?.[0] as number, 0, 0, scale[1]?.[1] as number])
  const grid = posterior3(boxes, (a, b, c) => {
    const lp = prior(a, b, c)
    return lp === Number.NEGATIVE_INFINITY ? lp : lp + ll.logLikelihood(a, b, c)
  })
  const s2 = noise * noise
  const { xtx, xty } = ll.sums
  const c0 = priorSd ** 2
  // given the matrix: the mean's posterior N(P^-1 b, P^-1), and each group's coefficients mix over it
  const given = (a: number, b: number, c: number) => {
    const A = [a + xtx[0] / s2, b + xtx[1] / s2, c + xtx[2] / s2] as const
    const [i11, i12, i22] = inverse2(A[0], A[1], A[2])
    const g = [xtx[0] / s2, xtx[1] / s2, xtx[2] / s2] as const
    // Q = G - G A^-1 G
    const gai = (u: readonly number[]): [number, number] => {
      const w = [i11 * (u[0] as number) + i12 * (u[1] as number), i12 * (u[0] as number) + i22 * (u[1] as number)]
      return [g[0] * (w[0] as number) + g[1] * (w[1] as number), g[1] * (w[0] as number) + g[2] * (w[1] as number)]
    }
    const cols = [gai([g[0], g[1]]), gai([g[1], g[2]])]
    let [p11, p12, p22] = [1 / c0, 0, 1 / c0]
    let [b1, b2] = [0, 0]
    for (let j = 0; j < groups; j++) {
      const xy = xty[j] as [number, number]
      const h = [xy[0] / s2, xy[1] / s2]
      p11 += g[0] - (cols[0] as [number, number])[0]
      p12 += g[1] - (cols[1] as [number, number])[0]
      p22 += g[2] - (cols[1] as [number, number])[1]
      const gh = gai(h)
      b1 += (h[0] as number) - gh[0]
      b2 += (h[1] as number) - gh[1]
    }
    const [q11, q12, q22] = inverse2(p11, p12, p22)
    return {
      A,
      i: [i11, i12, i22] as const,
      mu: [q11 * b1 + q12 * b2, q12 * b1 + q22 * b2] as const,
      muCov: [q11, q12, q22] as const,
    }
  }
  const effect = (j: number, c: 0 | 1): ExactMarginal =>
    grid.mixture((a, b, cc) => {
      const { i, mu, muCov } = given(a, b, cc)
      const xy = xty[j] as [number, number]
      // theta_j | mu ~ N(A^-1 (Lambda mu + X'y / s2), A^-1); integrating mu ~ N(mu_hat, muCov) adds A^-1 Lambda muCov Lambda A^-1
      const h0 = [xy[0] / s2, xy[1] / s2]
      const m0 = [
        a * (mu[0] as number) + b * (mu[1] as number) + (h0[0] as number),
        b * (mu[0] as number) + cc * (mu[1] as number) + (h0[1] as number),
      ]
      const mean =
        c === 0
          ? i[0] * (m0[0] as number) + i[1] * (m0[1] as number)
          : i[1] * (m0[0] as number) + i[2] * (m0[1] as number)
      // the row of A^-1 Lambda for the coordinate
      const row = c === 0 ? [i[0] * a + i[1] * b, i[0] * b + i[1] * cc] : [i[1] * a + i[2] * b, i[1] * b + i[2] * cc]
      const extra =
        (row[0] as number) ** 2 * muCov[0] +
        2 * (row[0] as number) * (row[1] as number) * muCov[1] +
        (row[1] as number) ** 2 * muCov[2]
      return { mean, variance: (c === 0 ? i[0] : i[2]) + extra }
    })
  const exact: Record<string, ExactMarginal> = {
    'mu[0]': grid.mixture((a, b, c) => ({ mean: given(a, b, c).mu[0], variance: given(a, b, c).muCov[0] })),
    'mu[1]': grid.mixture((a, b, c) => ({ mean: given(a, b, c).mu[1], variance: given(a, b, c).muCov[2] })),
    'Lambda[0]': grid.marginal(0),
    'Lambda[1]': grid.marginal(1),
    'Lambda[3]': grid.marginal(2),
  }
  for (let j = 0; j < groups; j++) {
    exact[`theta[${2 * j}]`] = effect(j, 0)
    exact[`theta[${2 * j + 1}]`] = effect(j, 1)
  }
  const effects = data.effects ?? []
  return {
    build: {
      definition,
      data: { X: design.map((o) => [1, o.t]), g: design.map((o) => o.g), y: data.y.flat() },
      quantities: [
        { name: 'mu[0]', label: 'mean intercept', support: 'real', truth: m1 },
        { name: 'mu[1]', label: 'mean slope', support: 'real', truth: m2 },
        { name: 'Lambda[0]', label: 'Lambda 11', support: 'positive', truth: trueLambda[0] },
        { name: 'Lambda[1]', label: 'Lambda 12', support: 'real', truth: trueLambda[1] },
        { name: 'Lambda[3]', label: 'Lambda 22', support: 'positive', truth: trueLambda[2] },
        ...Array.from({ length: groups }, (_, j) => [
          {
            name: `theta[${2 * j}]`,
            label: `intercept ${j + 1}`,
            support: 'real' as const,
            truth: effects[j]?.[0] as number,
            pool: 'intercepts (every group)',
          },
          {
            name: `theta[${2 * j + 1}]`,
            label: `slope ${j + 1}`,
            support: 'real' as const,
            truth: effects[j]?.[1] as number,
            pool: 'slopes (every group)',
          },
        ]).flat(),
      ],
    },
    reference: {
      exact,
      referenceSource:
        'quadrature of the three entries of the precision matrix, the mean and the effects integrated out in closed form',
    },
    present: {
      equations: [
        `truth: population mean (${m1}, ${m2}), sds (${sd1}, ${sd2}), correlation ${rho}, noise ${noise}; ${groups} groups of ${TIMES.length}, seed ${seed}`,
        `mu ~ N(0, ${priorSd}^2 I);  Lambda ~ Wishart(${nu}, I / ${nu})`,
        `theta_j ~ N(mu, Lambda^-1), j = 1..${groups}`,
        `y_jt ~ N(theta_j1 + theta_j2 t, ${noise}^2), t = ${TIMES.join(', ')}`,
      ],
      figure: randomEffectsFigure(data.y, effects),
    },
  }
}

/**
 * Correlated random intercepts and slopes with the population mean and the covariance of the effects both unknown, learned from groups simulated from a
 * stated truth: does EP recover the covariance, the mean and each group's line, and how often do its 90% intervals hold them?
 *
 * @index estimator EP atom: correlated random intercepts and slopes with learned mean and covariance, recovered from simulated groups.
 * @example
 * ```ts
 * import { randomInterceptsSlopesLearnedAtom } from 'omai-engine/probprog/ep'
 * randomInterceptsSlopesLearnedAtom.extends // => 'random-intercepts-slopes'
 * ```
 */
export const randomInterceptsSlopesLearnedAtom: EpAtom = defineAtom({
  name: 'random-intercepts-slopes-learned',
  title: 'Correlated random intercepts and slopes: parameters learned',
  extends: 'random-intercepts-slopes',
  recovery: recovery(
    'measured',
    'ten groups identify the covariance of their (intercept, slope) pairs only loosely and its posterior is skewed; EP carries it as a Wishart within a few percent of the exact spread, and the intervals hold the truth about 0.9 of the time over 30 seeds; 16 seeds leave a binomial spread of about 0.08',
    { seeds: 16, coverage: 0.7, zrms: 1.6 },
  ),
  what: 'The random intercepts and slopes of the basic atom with the population mean also unknown (a Gaussian prior) and the groups simulated from a stated mean, covariance and noise: the Wishart matrix, the mean and every group are learned together.',
  factors: [
    'prior (gaussian-mv)',
    'prior (wishart)',
    'mvn-wishart (a Gaussian vector per group, random mean)',
    'linear-regression-row',
  ],
  family: 'covariance',
  reference: 'exact',
  tolerance: tolerance(
    'close',
    'the groups are Gaussian given the matrix and the mean and the matrix a Wishart given them, but not jointly: EP matches each by moments; the population mean (spread 0.93 of the exact at eight groups, 0.77 to 0.87 at three or five) is the part it carries least well, because it is shared by every group',
  ),
  knobs: dict({
    seed: simulationKnobs(10)['seed'] as ReturnType<typeof int>,
    n: int(3, 12, { default: 10, label: 'groups' }),
    trueMean1: real(-2, 2, { default: 0.4, step: 0.1, label: 'true mean intercept' }),
    trueMean2: real(-2, 2, { default: -0.3, step: 0.1, label: 'true mean slope' }),
    trueSd1: real(0.2, 2, { default: 1, step: 0.05, label: 'true sd of the intercepts' }),
    trueSd2: real(0.2, 2, { default: 0.8, step: 0.05, label: 'true sd of the slopes' }),
    trueRho: real(-0.95, 0.95, { default: -0.5, step: 0.05, label: 'true correlation' }),
    noise: real(0.1, 1, { default: 0.3, step: 0.05, label: 'noise sd (known)' }),
    df: real(2.2, 12, { default: 3, step: 0.2, label: 'prior degrees of freedom' }),
    priorSd: real(1, 10, { default: 3, step: 0.5, label: 'prior sd of the mean' }),
  }),
  parts: randomEffectsLearnedProblem,
})
