/**
 * The likelihood setups of expectation propagation, basic and learned: a Student-t, a Laplace, a probit, a logistic and a Poisson-log observation, a Weibull and the degrees of freedom of a Student-t, each with its learned twin (the unknowns made random, the data simulated from a stated truth). The exact answers are closed forms or quadratures of the same model.
 */

import { gram, rbfKernel, scaleKernel } from '../../../kernels/index.ts'
import { foldIn, gamma, key, normal, poisson, studentT, uniform } from '../../../random/index.ts'
import { dict, int, real } from '../../../space/index.ts'
import { item, toFlat } from '../../../tensor/index.ts'
import type { Config } from '../../../trace/index.ts'
import { configNumber } from '../../inference/method.ts'
import * as d from '../../model/dists/index.ts'
import { defineModel, ops } from '../../model/index.ts'
import { probitClassifier, probitData } from '../../models/probit-classification.ts'
import { trueskillModel } from '../../models/ranking.ts'
import { type ExactMarginal, gridMarginals, lg, normalCdf, normalCdfFast, sum } from '../../validation/index.ts'
import { defineAtom } from '../_define.ts'
import { recovery, tolerance } from '../_tolerance.ts'
import { once } from '../_util.ts'
import type { AtomFigure, AtomParts, AtomRecoverySpec, EpAtom, FigureCurve } from '../atom-types.ts'
import { simulationKnobs } from './conjugate.ts'

const HEAVY = [0.2, -0.1, 0.4, 0.0, 4.5]

function heavyProblem(kind: 'student-t' | 'laplace', k: Config): AtomParts {
  const s = configNumber(k, 'scale')
  const df = 3
  // region student-t laplace
  const definition = defineModel(`${kind}-likelihood`, (m) => {
    const mu = m.latent('mu', d.normal(0, 5))
    m.observe(
      'y',
      kind === 'student-t' ? d.studentT(df, mu, s) : d.laplace(mu, s),
      m.data('y', { dims: ['i'], role: 'response' }),
    )
  })
  // endregion
  return {
    build: {
      definition,
      data: { y: HEAVY },
      quantities: [{ name: 'mu', label: 'mu', support: 'real' }],
    },
    present: {
      equations: [
        'mu ~ N(0, 5²)',
        kind === 'student-t' ? `y_i ~ Student-t(df = ${df}, mu, ${s})` : `y_i ~ Laplace(mu, ${s})`,
        `observe y = (${HEAVY.join(', ')})   (the last is far from the rest)`,
      ],
      figure: {
        caption:
          'The five observations; the line is where the belief puts mu. A heavy-tailed likelihood lets the last one be an outlier instead of dragging mu towards it.',
        xLabel: 'observation',
        yLabel: 'y',
        xLimits: [0, HEAVY.length + 1],
        points: [{ name: 'observations', x: HEAVY.map((_, i) => i + 1), y: [...HEAVY] }],
        answer: (m) => [{ name: 'mu', x: [0, HEAVY.length + 1], y: [m['mu'] ?? Number.NaN, m['mu'] ?? Number.NaN] }],
      },
    },
  }
}

/**
 * A mean observed through Student-t noise, one observation far from the rest: the Student-t site (a quadrature), whose
 * influence function redescends so the outlier is not followed.
 */
const studentTAtom: EpAtom = defineAtom({
  name: 'student-t',
  title: 'Student-t likelihood',
  what: 'A mean mu observed five times through Student-t noise with three degrees of freedom, one observation being far from the rest: the Student-t site.',
  factors: ['prior (gaussian)', 'student-t-site (quadrature)'],
  family: 'likelihoods',
  reference: 'grid',
  tolerance: tolerance(
    'skewed',
    'the posterior is bumpy (the outlier may or may not count) and EP matches its mean and variance with one Gaussian',
  ),
  evidence: { nats: 1e-3, reason: 'the Student-t site by quadrature' },
  knobs: dict({ scale: real(0.1, 2, { default: 0.5, step: 0.05, label: 'noise scale' }) }),
  parts: (k) => heavyProblem('student-t', k),
})

/**
 * The same with a Laplace likelihood.
 */
const laplaceAtom: EpAtom = defineAtom({
  name: 'laplace',
  title: 'Laplace likelihood',
  what: 'A mean mu observed five times through Laplace (double exponential) noise, one observation being far from the rest: the Laplace site, an absolute-error likelihood whose posterior is piecewise Gaussian.',
  factors: ['prior (gaussian)', 'laplace-site (quadrature)'],
  family: 'likelihoods',
  reference: 'grid',
  tolerance: tolerance(
    'skewed',
    'the posterior has kinks at the observations and EP matches its mean and variance with one Gaussian',
  ),
  evidence: { nats: 0.05, reason: 'the Laplace site: a kinked density, moment matched' },
  knobs: dict({ scale: real(0.1, 2, { default: 0.5, step: 0.05, label: 'noise scale' }) }),
  parts: (k) => heavyProblem('laplace', k),
})

const POINTS: readonly (readonly [number, number, number])[] = [
  [1.2, 0.9, 1],
  [0.4, 1.6, 1],
  [1.8, -0.2, 1],
  [-0.5, 0.3, 1],
  [-1.3, -0.8, 0],
  [-0.2, -1.5, 0],
  [0.5, -0.9, 0],
  [0.6, 0.7, 0],
]
const LIMIT = 3

function classesProblem(link: 'probit' | 'logistic', k: Config): AtomParts {
  const s = configNumber(k, 'priorScale')
  const design = POINTS.map(([a, b]) => [1, a, b])
  const y = POINTS.map((p) => p[2])
  // region probit logistic
  const definition =
    link === 'probit'
      ? probitClassifier(s)
      : defineModel('logistic-classification', (m) => {
          const row = m.range('row')
          const f = m.range('feature')
          const X = m.data('X', { dims: [row, f], role: 'covariate' })
          const yy = m.data('y', { dims: [row], support: 'binary', role: 'response' })
          const w = m.latent('w', d.normal(0, s), { dims: [f] })
          m.observe('y', d.bernoulli({ probs: ops.sigmoid(X.matmul(w)) }), yy)
        })
  // endregion
  const edge = [-LIMIT, LIMIT]
  return {
    build: {
      definition,
      data: probitData(design, y),
      quantities: [0, 1, 2].map((j) => ({
        name: `w[${j}]`,
        label: ['w_0 (offset)', 'w_1 (slope in x1)', 'w_2 (slope in x2)'][j] ?? `w_${j}`,
        support: 'real' as const,
      })),
    },
    reference: {
      grid: { points: 25, maxRefinements: 0 },
    },
    present: {
      equations: [
        `w_0, w_1, w_2 ~ N(0, ${s}²)   (offset and two slopes)`,
        link === 'probit'
          ? 'y_i ~ Bernoulli(Phi(w_0 + w_1 x_i1 + w_2 x_i2)), i = 1..8'
          : 'y_i ~ Bernoulli(sigmoid(w_0 + w_1 x_i1 + w_2 x_i2)), i = 1..8',
        'observe the eight labelled points drawn in the figure',
      ],
      pair: ['w[1]', 'w[2]'],
      figure: {
        caption: 'The eight labelled points and the boundary w_0 + w_1 x1 + w_2 x2 = 0 at the mean of the belief.',
        xLabel: 'x1',
        yLabel: 'x2',
        xLimits: [-LIMIT, LIMIT],
        yLimits: [-LIMIT, LIMIT],
        points: [
          {
            name: 'points',
            x: POINTS.map((p) => p[0]),
            y: POINTS.map((p) => p[1]),
            group: y,
            groupNames: { 0: 'label 0', 1: 'label 1' },
          },
        ],
        answer: (m) => {
          const a = m['w[0]'] ?? Number.NaN
          const b = m['w[1]'] ?? Number.NaN
          const c = m['w[2]'] ?? Number.NaN
          return [{ name: 'boundary', x: edge, y: edge.map((x) => -(a + b * x) / c) }]
        },
      },
    },
  }
}

/**
 * Eight labelled points and a probit link on a linear predictor: the probit factor, whose tilted moments are those of a
 * truncated Gaussian in closed form.
 */
const probitAtom: EpAtom = defineAtom({
  name: 'probit',
  title: 'Probit',
  what: 'Three weights (an offset and two slopes) with Gaussian priors and eight binary labels through a probit link: the probit factor, a truncated Gaussian with closed-form moments, on a linear predictor.',
  factors: ['prior (gaussian)', 'matmul (linear predictor)', 'probit link'],
  family: 'likelihoods',
  reference: 'grid',
  tolerance: tolerance(
    'truncated',
    'EP matches a Gaussian to each truncated-Gaussian tilted distribution; the posterior is skewed, so it is close, not exact',
  ),
  evidence: { nats: 0.03, reason: 'a probit link: moment matched truncated Gaussians' },
  knobs: dict({ priorScale: real(0.5, 6, { default: 2, step: 0.1, label: 'prior sd of the weights' }) }),
  parts: (k) => classesProblem('probit', k),
})

/**
 * The same eight points through a logistic link: no closed form for the tilted moments, so the rule integrates numerically.
 */
const logisticAtom: EpAtom = defineAtom({
  name: 'logistic',
  title: 'Logistic',
  what: 'The same three weights and eight labels through a logistic link: no closed-form tilted moments, so the rule is a numeric quadrature.',
  factors: ['prior (gaussian)', 'matmul (linear predictor)', 'logistic link (numeric quadrature)'],
  family: 'likelihoods',
  reference: 'grid',
  tolerance: tolerance('tight', 'a Gaussian projection of each logistic tilted distribution, evaluated by quadrature'),
  evidence: { nats: 0.03, reason: 'a logistic link by quadrature, moment matched' },
  knobs: dict({ priorScale: real(0.5, 6, { default: 2, step: 0.1, label: 'prior sd of the weights' }) }),
  parts: (k) => classesProblem('logistic', k),
})

const TOURNAMENT = {
  players: ['A', 'B', 'C'],
  skills: [0, 0, 0],
  pairs: [
    [0, 1],
    [0, 2],
    [1, 2],
    [1, 0],
    [0, 2],
  ] as [number, number][],
  wins: [1, 1, 1, 1, 0],
}

function skillsProblem(): AtomParts {
  // region skill-difference
  const { definition, data } = trueskillModel(TOURNAMENT)
  // endregion
  const names = TOURNAMENT.players
  const notes = TOURNAMENT.pairs.map(([a, b], g) => {
    const aWon = TOURNAMENT.wins[g] === 1
    return `Match ${g + 1}: ${names[aWon ? a : b]} beat ${names[aWon ? b : a]}`
  })
  const at = [1, 2, 3]
  return {
    build: {
      definition,
      data,
      quantities: names.map((n) => ({ name: `skill_${n}`, label: `skill_${n}`, support: 'real' as const })),
    },
    reference: {
      grid: { points: 25, maxRefinements: 0 },
    },
    present: {
      equations: [
        'skill_A, skill_B, skill_C ~ N(0, 1)',
        'match g between a and b: a wins ~ Bernoulli(Phi((skill_a - skill_b) / sqrt 2))',
        ...notes,
      ],
      pair: ['skill_A', 'skill_B'],
      figure: {
        caption: 'The belief about each player’s skill after the five matches (the bars).',
        xLabel: 'player',
        yLabel: 'skill',
        xLimits: [0.4, 3.6],
        yLimits: [-2, 2],
        points: [],
        answer: (m) =>
          names.map((n, i) => ({
            name: `player ${n}`,
            x: [(at[i] ?? 0) - 0.3, (at[i] ?? 0) + 0.3],
            y: [m[`skill_${n}`] ?? Number.NaN, m[`skill_${n}`] ?? Number.NaN],
          })),
      },
    },
  }
}

/**
 * Three players with Gaussian skills and five match results, each a probit factor on a difference of two skills.
 */
const skillDifferenceAtom: EpAtom = defineAtom({
  name: 'skill-difference',
  title: 'Probit on a difference',
  what: 'Three Gaussian skills and five results, each a probit factor on the difference of two skills (the TrueSkill performance model): every factor joins two variables.',
  factors: ['prior (gaussian) x3', 'probit link on a difference of two variables'],
  family: 'likelihoods',
  reference: 'grid',
  tolerance: tolerance(
    'skew-normal',
    'each result is a truncated-Gaussian tilted distribution that EP projects onto a Gaussian: close, not exact',
  ),
  evidence: { nats: 1e-2, reason: 'two skills compared through a probit' },
  knobs: dict({}),
  parts: () => skillsProblem(),
})

/** The inverse of a 3 by 3 matrix by cofactors. */
function invert3(a: readonly (readonly number[])[]): number[][] {
  const g = (i: number, j: number): number => a[i]?.[j] ?? 0
  const c = (i: number, j: number): number => {
    const r = [0, 1, 2].filter((k) => k !== i)
    const s = [0, 1, 2].filter((k) => k !== j)
    const m = g(r[0] ?? 0, s[0] ?? 0) * g(r[1] ?? 0, s[1] ?? 0) - g(r[0] ?? 0, s[1] ?? 0) * g(r[1] ?? 0, s[0] ?? 0)
    return (i + j) % 2 === 0 ? m : -m
  }
  const det = g(0, 0) * c(0, 0) + g(0, 1) * c(0, 1) + g(0, 2) * c(0, 2)
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => c(j, i) / det))
}

const GP_X = [[-1.2], [0.1], [1.4]]
const GP_Y = [0, 1, 1]

function gpProblem(link: 'probit' | 'logistic', k: Config): AtomParts {
  const ls = configNumber(k, 'lengthScale')
  const kernel = scaleKernel(rbfKernel({ lengthScale: ls }), 1.5)
  const n = GP_X.length
  const flat = toFlat(gram(kernel, GP_X, { jitter: 1e-6 }))
  const K = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => flat[i * n + j] ?? 0))
  // region gp-logistic gp-probit
  const definition = defineModel(`gp-${link}`, (m) => {
    const row = m.range('row')
    const prior = d.multivariateNormal(new Array<number>(n).fill(0), { covariance: K })
    const f = m.latent('f', prior, { dims: [row] })
    m.observe(
      'y',
      d.bernoulli({ probs: link === 'probit' ? ops.probit(f) : ops.sigmoid(f) }),
      m.data('y', { dims: [row], support: 'binary' }),
    )
  })
  // endregion
  // The reference is the density written out: a dense Gaussian vector is more coordinates than the program grid takes.
  const inv = invert3(K)
  const logDensity = (f: readonly number[]): number => {
    let q = 0
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) q += (f[i] ?? 0) * (inv[i]?.[j] ?? 0) * (f[j] ?? 0)
    let ll = 0
    for (let i = 0; i < n; i++) {
      const p = link === 'probit' ? normalCdf(f[i] ?? 0) : 1 / (1 + Math.exp(-(f[i] ?? 0)))
      ll += Math.log(Math.max(GP_Y[i] === 1 ? p : 1 - p, 1e-300))
    }
    return -0.5 * q + ll
  }
  const g = gridMarginals(
    logDensity,
    GP_X.map(() => [-5, 5] as [number, number]),
    51,
  )
  const pair = g.joint(0, 1)
  return {
    build: {
      definition,
      data: { y: GP_Y },
      quantities: GP_X.map((_, i) => ({ name: `f[${i}]`, label: `f(x_${i + 1})`, support: 'real' as const })),
    },
    reference: {
      exact: Object.fromEntries(g.marginals.map((m, i) => [`f[${i}]`, m])),
      exactJoint: { a: 'f[0]', b: 'f[1]', ...pair },
    },
    present: {
      equations: [
        `f ~ N(0, K),  K = 1.5 exp(-(x - x')² / (2 · ${ls}²))  at x = (${GP_X.map((r) => r[0]).join(', ')})   (a dense Gaussian vector)`,
        link === 'probit' ? 'y_i ~ Bernoulli(Phi(f_i))' : 'y_i ~ Bernoulli(sigmoid(f_i))',
        `observe y = (${GP_Y.join(', ')})`,
      ],
      pair: ['f[0]', 'f[1]'],
    },
  }
}

/**
 * A Gaussian process prior over three inputs (a dense Gaussian vector) with a logistic likelihood on each: the dense prior
 * message of GP classification.
 */
const gpLogisticAtom: EpAtom = defineAtom({
  name: 'gp-logistic',
  title: 'Gaussian process with a logistic likelihood',
  what: 'A dense Gaussian prior on three function values (an RBF kernel) and a logistic Bernoulli observation of each: the dense Gaussian-vector prior factor of GP classification.',
  factors: ['dense prior (gaussian-mv)', 'logistic link (numeric quadrature)'],
  family: 'likelihoods',
  reference: 'exact',
  tolerance: tolerance(
    'tight',
    'a Gaussian projection of each logistic tilted distribution against a dense Gaussian cavity',
  ),
  knobs: dict({ lengthScale: real(0.3, 3, { default: 1, step: 0.1, label: 'kernel length scale' }) }),
  parts: (k) => gpProblem('logistic', k),
})

/**
 * The same with a probit likelihood.
 */
const gpProbitAtom: EpAtom = defineAtom({
  name: 'gp-probit',
  title: 'Gaussian process with a probit likelihood',
  what: 'A dense Gaussian prior on three function values and a probit Bernoulli observation of each: GP classification with closed-form tilted moments.',
  factors: ['dense prior (gaussian-mv)', 'probit link'],
  family: 'likelihoods',
  reference: 'exact',
  tolerance: tolerance(
    'truncated',
    'a Gaussian projection of each truncated-Gaussian tilted distribution against a dense Gaussian cavity',
  ),
  knobs: dict({ lengthScale: real(0.3, 3, { default: 1, step: 0.1, label: 'kernel length scale' }) }),
  parts: (k) => gpProblem('probit', k),
})

function poissonLogProblem(k: Config): AtomParts {
  const y = configNumber(k, 'y')
  // region poisson-log
  const definition = defineModel('poisson-log-link', (m) => {
    const x = m.latent('x', d.normal(0, 1))
    m.observe('y', d.poisson(ops.exp(x)), m.data('y', { support: 'count' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y },
      quantities: [{ name: 'x', label: 'x', support: 'real' }],
    },
    present: {
      equations: ['x ~ N(0, 1)', 'y ~ Poisson(exp(x))   (a log link)', `observe y = ${y}`],
    },
  }
}

/**
 * A Poisson count with a log link on a Gaussian predictor: the likelihood of a Poisson regression.
 */
const poissonLogAtom: EpAtom = defineAtom({
  name: 'poisson-log',
  title: 'Poisson with a log link',
  what: 'A Gaussian predictor x and a Poisson count with rate exp(x): the numeric likelihood on a linear predictor (the log link of Poisson regression).',
  factors: ['prior (gaussian)', 'numeric likelihood: poisson with an exp link'],
  family: 'likelihoods',
  reference: 'grid',
  tolerance: tolerance('tight', 'a Gaussian projection of the tilted distribution, evaluated by quadrature'),
  evidence: { nats: 1e-3, reason: 'a log-link Poisson by quadrature' },
  knobs: dict({ y: real(0, 20, { default: 5, step: 1, integer: true, label: 'observed count' } as never) }),
  parts: poissonLogProblem,
})

// ----- a Student-t likelihood with an unknown location and scale -----

const T_DF = 3

function studentLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueLoc, trueScale] = [configNumber(k, 'trueLoc'), configNumber(k, 'trueScale')]
  const [a, b] = [configNumber(k, 'shape'), configNumber(k, 'rate')]
  const df = T_DF
  // region student-t-learned
  // the ground truth the data are drawn from: Student-t noise with three degrees of freedom
  const noise = toFlat(studentT(foldIn(key(seed), 0), [n], df))
  const y = Array.from(noise, (e) => trueLoc + trueScale * e)
  const definition = defineModel('student-t-learned', (m) => {
    const row = m.range('row')
    // the location and the precision of the noise: both unknown
    const mu = m.latent('mu', d.normal(0, 5))
    const tau = m.latent('tau', d.gamma(a, b))
    // Student-t noise is Gaussian noise whose precision is scaled by a gamma draw per observation
    const mix = m.latent('mix', d.gamma(df / 2, df / 2), { dims: [row] })
    const precision = mix.mul(tau)
    m.observe('y', d.normal(mu, precision.pow(-0.5)), m.data('y', { dims: [row], role: 'response' }))
  })
  // endregion
  // The exact posterior of (mu, tau) with the mixing precisions integrated out (a Student-t density), on a grid.
  const sorted = [...y].sort((p, q) => p - q)
  const med = sorted[Math.floor(n / 2)] ?? 0
  const mad = [...y.map((v) => Math.abs(v - med))].sort((p, q) => p - q)[Math.floor(n / 2)] ?? 1
  const s0 = Math.max(1.4826 * mad, 0.05)
  const half = (10 * s0) / Math.sqrt(n)
  const tau0 = 1 / s0 ** 2
  const logT = (r2: number, tau: number): number =>
    lg((df + 1) / 2) -
    lg(df / 2) -
    0.5 * Math.log(df * Math.PI) +
    0.5 * Math.log(tau) -
    ((df + 1) / 2) * Math.log(1 + (tau * r2) / df)
  const grid = once(() =>
    gridMarginals(
      ([mu, tau]) => {
        const t = tau as number
        if (t <= 0) return Number.NEGATIVE_INFINITY
        return (
          -0.5 * ((mu as number) / 5) ** 2 +
          (a - 1) * Math.log(t) -
          b * t +
          sum(y.map((v) => logT((v - (mu as number)) ** 2, t)))
        )
      },
      [
        [med - half, med + half],
        [tau0 / 12, tau0 * 6],
      ],
      101,
    ),
  )
  const x = [0, n + 1]
  return {
    build: {
      definition,
      data: { y },
      quantities: [
        { name: 'mu', label: 'mu (the location)', support: 'real', truth: trueLoc },
        { name: 'tau', label: 'tau (the precision of the noise)', support: 'positive', truth: 1 / trueScale ** 2 },
      ],
    },
    reference: {
      get exact() {
        return { mu: grid().marginal(0), tau: grid().marginal(1) }
      },
      get exactJoint() {
        return { a: 'mu', b: 'tau', ...grid().joint(0, 1) }
      },
    },
    present: {
      equations: [
        `truth: location ${trueLoc}, scale ${trueScale} (precision ${Number((1 / trueScale ** 2).toPrecision(3))}), Student-t noise with ${df} degrees of freedom; ${n} draws with seed ${seed}`,
        'mu ~ N(0, 5²)   (the unknown location)',
        `tau ~ Gamma(${a}, ${b})   (the unknown precision of the noise)`,
        `mix_i ~ Gamma(${df}/2, ${df}/2); y_i ~ N(mu, 1 / (mix_i tau))   (the scale mixture: marginally y_i ~ Student-t(${df}, mu, 1/sqrt(tau)))`,
      ],
      pair: ['mu', 'tau'],
      figure: {
        caption:
          'The simulated observations (heavy tails: a few lie far from the rest). Green: the true location; thick and dashed: the fitted one.',
        xLabel: 'observation',
        yLabel: 'y',
        xLimits: x as [number, number],
        points: [{ name: 'observations', x: y.map((_, i) => i + 1), y }],
        curves: [{ name: 'true location', colour: 'tab2', x, y: [trueLoc, trueLoc] }],
        answer: (mm) => [{ name: 'mu', x, y: [mm['mu'] ?? Number.NaN, mm['mu'] ?? Number.NaN] }],
      },
    },
  }
}

/**
 * A location observed through Student-t noise whose scale is unknown too, as a Gaussian scale mixture (a gamma precision per
 * observation times one gamma precision), learned from data simulated from a stated truth: does EP recover the location and the
 * scale of heavy-tailed noise, against the exact posterior with the mixing precisions integrated out?
 */
const studentTLearnedAtom: EpAtom = defineAtom({
  name: 'student-t-learned',
  title: 'Student-t likelihood: parameters learned',
  extends: 'student-t',
  recovery: recovery(
    'standard',
    'EP projects the scale-mixture posterior onto a Gaussian for the location and a gamma for the precision; with heavy-tailed noise and a dozen draws the precision is the harder to pin down',
  ),
  what: 'The Student-t noise of the basic atom (three degrees of freedom, fixed) with the location and the scale both unknown, written as a Gaussian scale mixture; the data are simulated from a stated location and scale.',
  factors: [
    'prior (gaussian)',
    'prior (gamma)',
    'gamma-product (mixing precision times noise precision)',
    'gaussian with a random scale',
  ],
  family: 'likelihoods',
  reference: 'exact',
  tolerance: tolerance(
    'coupled',
    'the posterior of the location and the precision is skewed and coupled through the mixing precisions, and EP carries a Gaussian and a gamma for them',
  ),
  ep: { damping: 0.3 },
  knobs: dict({
    ...simulationKnobs(15),
    trueLoc: real(-3, 3, { default: 1, step: 0.1, label: 'true location' }),
    trueScale: real(0.2, 2, { default: 0.7, step: 0.05, label: 'true scale' }),
    shape: real(0.5, 8, { default: 2, step: 0.1, label: 'prior shape of tau' }),
    rate: real(0.1, 8, { default: 1, step: 0.1, label: 'prior rate of tau' }),
  }),
  parts: studentLearnedProblem,
})

// ----- a Laplace likelihood with an unknown location and scale -----

function laplaceLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueLoc, trueScale] = [configNumber(k, 'trueLoc'), configNumber(k, 'trueScale')]
  // region laplace-learned
  // the ground truth the data are drawn from: Laplace noise (the difference of two exponentials)
  const kk = key(seed)
  const u = toFlat(uniform(foldIn(kk, 0), [n]))
  const tail = (v: number): number => Math.log(1 - 2 * Math.abs(v - 0.5))
  const y = Array.from(u, (v) => trueLoc - trueScale * Math.sign(v - 0.5) * tail(v))
  const definition = defineModel('laplace-learned', (m) => {
    // the location and the scale of the noise: both unknown
    const mu = m.latent('mu', d.normal(0, 5))
    const scale = m.latent('scale', d.halfNormal(2))
    m.observe('y', d.laplace(mu, scale), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  // The reference: the posterior density of (mu, scale) written out, on a grid.
  const sorted = [...y].sort((p, q) => p - q)
  const med = sorted[Math.floor(n / 2)] ?? 0
  const mad = [...y.map((v) => Math.abs(v - med))].sort((p, q) => p - q)[Math.floor(n / 2)] ?? 1
  const s0 = Math.max(1.4826 * mad, 0.05)
  const half = (10 * s0) / Math.sqrt(n)
  const grid = once(() =>
    gridMarginals(
      ([mu, scale]) => {
        const sc = scale as number
        if (sc <= 0) return Number.NEGATIVE_INFINITY
        return (
          -0.5 * ((mu as number) / 5) ** 2 -
          (sc * sc) / 8 -
          n * Math.log(2 * sc) -
          sum(y.map((v) => Math.abs(v - (mu as number)))) / sc
        )
      },
      [
        [med - half, med + half],
        [s0 / 8, s0 * 3.5],
      ],
      121,
    ),
  )
  const x = [0, n + 1]
  return {
    build: {
      definition,
      data: { y },
      quantities: [
        { name: 'mu', label: 'mu (the location)', support: 'real', truth: trueLoc },
        { name: 'scale', label: 'scale (of the noise)', support: 'positive', truth: trueScale },
      ],
    },
    reference: {
      get exact() {
        return { mu: grid().marginal(0), scale: grid().marginal(1) }
      },
      get exactJoint() {
        return { a: 'mu', b: 'scale', ...grid().joint(0, 1) }
      },
    },
    present: {
      equations: [
        `truth: location ${trueLoc}, scale ${trueScale}, Laplace noise; ${n} draws with seed ${seed}`,
        'mu ~ N(0, 5²)   (the unknown location)',
        'scale ~ HalfNormal(2)   (the unknown scale)',
        `y_i ~ Laplace(mu, scale), i = 1..${n}`,
      ],
      pair: ['mu', 'scale'],
      figure: {
        caption: 'The simulated observations. Green: the true location; thick and dashed: the fitted one.',
        xLabel: 'observation',
        yLabel: 'y',
        xLimits: x as [number, number],
        points: [{ name: 'observations', x: y.map((_, i) => i + 1), y }],
        curves: [{ name: 'true location', colour: 'tab2', x, y: [trueLoc, trueLoc] }],
        answer: (mm) => [{ name: 'mu', x, y: [mm['mu'] ?? Number.NaN, mm['mu'] ?? Number.NaN] }],
      },
    },
  }
}

/**
 * A location observed through Laplace noise whose scale is unknown too, with data simulated from a stated truth: the
 * `likelihood-pair` factor (the Laplace density over the location and the log of the scale, a two-variable numeric site), against
 * the grid posterior of the same model.
 */
const laplaceLearnedAtom: EpAtom = defineAtom({
  name: 'laplace-learned',
  title: 'Laplace likelihood: parameters learned',
  extends: 'laplace',
  recovery: recovery(
    'standard',
    'EP carries one Gaussian for the location and one for the log of the scale (the half-normal prior is projected onto the log once), against 15 absolute errors that tell the location well and the scale less so',
  ),
  what: 'The Laplace noise of the basic atom with the location and the scale both unknown, and the data simulated from a stated location and scale.',
  factors: ['prior (gaussian)', 'numeric prior (half-normal)', 'likelihood-pair: laplace with a random scale'],
  family: 'likelihoods',
  reference: 'exact',
  tolerance: tolerance(
    'coupled',
    'the posterior of a location under Laplace noise has kinks at the observations, and the scale is skewed',
  ),
  knobs: dict({
    ...simulationKnobs(15),
    trueLoc: real(-3, 3, { default: 1, step: 0.1, label: 'true location' }),
    trueScale: real(0.1, 2, { default: 0.5, step: 0.05, label: 'true scale' }),
  }),
  parts: laplaceLearnedProblem,
})

// ----- a probit or a logistic classifier whose weights share an unknown prior precision -----

const WEIGHTS = [0.4, 1.6, -1.1]

function classesLearnedProblem(link: 'probit' | 'logistic', k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [w0, w1, w2] = [configNumber(k, 'trueOffset'), configNumber(k, 'trueSlope1'), configNumber(k, 'trueSlope2')]
  const [a, b] = [configNumber(k, 'shape'), configNumber(k, 'rate')]
  const truth = [w0, w1, w2]
  // region probit-learned logistic-learned
  // the ground truth: random points, labelled through the link by the true weights
  const kk = key(seed)
  const pts = toFlat(normal(foldIn(kk, 0), [n, 2]))
  const design = Array.from({ length: n }, (_, i) => [1, pts[2 * i] ?? 0, pts[2 * i + 1] ?? 0])
  const u = toFlat(uniform(foldIn(kk, 1), [n]))
  const score = design.map((row) => w0 * (row[0] ?? 0) + w1 * (row[1] ?? 0) + w2 * (row[2] ?? 0))
  const chance = score.map((z) => (link === 'probit' ? normalCdfFast(z) : 1 / (1 + Math.exp(-z))))
  const y = chance.map((p, i) => ((u[i] ?? 1) < p ? 1 : 0))
  const definition = defineModel(`${link}-learned`, (m) => {
    const row = m.range('row')
    const f = m.range('feature')
    const X = m.data('X', { dims: [row, f], role: 'covariate' })
    const yy = m.data('y', { dims: [row], support: 'binary', role: 'response' })
    // the precision the weights share: how large the weights may be, unknown
    const tau = m.latent('tau', d.gamma(a, b))
    const w = m.latent('w', d.normal(0, ops.div(1, ops.sqrt(tau))), { dims: [f] })
    const probs = link === 'probit' ? ops.probit(X.matmul(w)) : ops.sigmoid(X.matmul(w))
    m.observe('y', d.bernoulli({ probs }), yy)
  })
  // endregion
  // The reference: the density written out, on a tensor grid over (tau, w) around the true weights.
  const logLink = (z: number, label: number): number => {
    const p = link === 'probit' ? normalCdfFast(z) : 1 / (1 + Math.exp(-z))
    return Math.log(Math.max(label === 1 ? p : 1 - p, 1e-300))
  }
  const grid = once(() =>
    gridMarginals(
      ([tau, ...w]) => {
        const t = tau as number
        let lp = (a - 1) * Math.log(t) - b * t + 1.5 * Math.log(t) - 0.5 * t * sum((w as number[]).map((v) => v * v))
        for (let i = 0; i < n; i++) {
          const row = design[i] as number[]
          lp += logLink(sum(row.map((v, j) => v * ((w as number[])[j] ?? 0))), y[i] ?? 0)
        }
        return lp
      },
      [[0.02, (4 * a) / b + 4], ...truth.map((t) => [t - 3.2, t + 3.2] as [number, number])],
      27,
    ),
  )
  const edge = [-LIMIT, LIMIT]
  const boundary = (c: readonly number[], name: string, colour?: string): FigureCurve => ({
    name,
    ...(colour === undefined ? {} : { colour }),
    x: edge,
    y: edge.map((v) => -((c[0] ?? 0) + (c[1] ?? 0) * v) / (c[2] ?? Number.NaN)),
  })
  const px = Array.from({ length: n }, (_, i) => pts[2 * i] ?? 0)
  const py = Array.from({ length: n }, (_, i) => pts[2 * i + 1] ?? 0)
  return {
    build: {
      definition,
      data: { X: design, y },
      quantities: [
        { name: 'tau', label: 'tau (the precision of the weights)', support: 'positive' },
        ...[0, 1, 2].map((j) => ({
          name: `w[${j}]`,
          label: ['w_0 (offset)', 'w_1 (slope in x1)', 'w_2 (slope in x2)'][j] ?? `w_${j}`,
          support: 'real' as const,
          truth: truth[j] as number,
        })),
      ],
    },
    reference: {
      get exact() {
        return Object.fromEntries(grid().marginals.map((m, i) => [i === 0 ? 'tau' : `w[${i - 1}]`, m]))
      },
      get exactJoint() {
        return { a: 'w[1]', b: 'w[2]', ...grid().joint(2, 3) }
      },
    },
    present: {
      equations: [
        `truth: w = (${truth.join(', ')}) (offset and two slopes); ${n} points ~ N(0, I), labelled through the ${link} link; seed ${seed}`,
        `tau ~ Gamma(${a}, ${b})   (the unknown precision the three weights share)`,
        'w_0, w_1, w_2 ~ N(0, 1 / tau)',
        link === 'probit'
          ? 'y_i ~ Bernoulli(Phi(w_0 + w_1 x_i1 + w_2 x_i2))'
          : 'y_i ~ Bernoulli(sigmoid(w_0 + w_1 x_i1 + w_2 x_i2))',
      ],
      pair: ['w[1]', 'w[2]'],
      figure: {
        caption:
          'The labelled points, the true decision boundary (green) and the fitted one at the mean of the belief (thick: the exact; dashed: EP).',
        xLabel: 'x1',
        yLabel: 'x2',
        xLimits: [-LIMIT, LIMIT],
        yLimits: [-LIMIT, LIMIT],
        points: [{ name: 'points', x: px, y: py, group: y, groupNames: { 0: 'label 0', 1: 'label 1' } }],
        curves: [boundary(truth, 'true boundary', 'tab2')],
        answer: (mm) => [
          boundary([mm['w[0]'] ?? Number.NaN, mm['w[1]'] ?? Number.NaN, mm['w[2]'] ?? Number.NaN], 'boundary'),
        ],
      },
    },
  }
}

/**
 * Labelled points and a probit link on a linear predictor whose three weights share an unknown prior precision (a gamma
 * precision on the weights, the penalty of a regularised classifier made random), learned from labels simulated from stated
 * true weights: does EP recover the weights and so the decision boundary?
 */
const probitLearnedAtom: EpAtom = defineAtom({
  name: 'probit-learned',
  title: 'Probit: parameters learned',
  extends: 'probit',
  recovery: recovery(
    'standard',
    'the weights are fixed numbers and the learned shrinkage pulls them a little towards zero, and EP matches a Gaussian to each truncated-Gaussian tilted distribution: the intervals hold the truth a little less than nine times in ten',
  ),
  what: 'The probit classifier of the basic atom with the prior precision of the weights learned (a gamma precision shared by the three weights) and the labels simulated from stated true weights: the weights and the decision boundary are recovered from the data.',
  factors: ['prior (gamma)', 'gaussian-gamma-precision (latent vector)', 'matmul (linear predictor)', 'probit link'],
  family: 'likelihoods',
  reference: 'exact',
  tolerance: tolerance(
    'coupled',
    'the posterior is skewed and coupled with the precision, EP projects each truncated-Gaussian tilted distribution onto a Gaussian, and the reference is a four-dimensional tensor grid of 27 points per axis',
  ),
  knobs: dict({
    ...simulationKnobs(50),
    trueOffset: real(-2, 2, { default: WEIGHTS[0] ?? 0.4, step: 0.1, label: 'true offset w_0' }),
    trueSlope1: real(-3, 3, { default: WEIGHTS[1] ?? 1.6, step: 0.1, label: 'true slope w_1' }),
    trueSlope2: real(-3, 3, { default: WEIGHTS[2] ?? -1.1, step: 0.1, label: 'true slope w_2' }),
    shape: real(0.5, 6, { default: 2, step: 0.1, label: 'prior shape of tau' }),
    rate: real(0.1, 6, { default: 2, step: 0.1, label: 'prior rate of tau' }),
  }),
  parts: (k) => classesLearnedProblem('probit', k),
})

/**
 * The same with a logistic link: no closed form for the tilted moments, so the rule integrates numerically.
 */
const logisticLearnedAtom: EpAtom = defineAtom({
  name: 'logistic-learned',
  title: 'Logistic: parameters learned',
  extends: 'logistic',
  recovery: recovery(
    'standard',
    'the weights are fixed numbers and the learned shrinkage pulls them a little towards zero; a logistic link makes the labels less informative than a probit of the same weights',
  ),
  what: 'The logistic classifier of the basic atom with the prior precision of the weights learned and the labels simulated from stated true weights.',
  factors: [
    'prior (gamma)',
    'gaussian-gamma-precision (latent vector)',
    'matmul (linear predictor)',
    'logistic link (numeric quadrature)',
  ],
  family: 'likelihoods',
  reference: 'exact',
  tolerance: tolerance(
    'coupled',
    'a Gaussian projection of each logistic tilted distribution by quadrature, against a four-dimensional tensor grid',
  ),
  knobs: dict({
    ...simulationKnobs(60),
    trueOffset: real(-2, 2, { default: WEIGHTS[0] ?? 0.4, step: 0.1, label: 'true offset w_0' }),
    trueSlope1: real(-4, 4, { default: 2.2, step: 0.1, label: 'true slope w_1' }),
    trueSlope2: real(-4, 4, { default: -1.6, step: 0.1, label: 'true slope w_2' }),
    shape: real(0.5, 6, { default: 2, step: 0.1, label: 'prior shape of tau' }),
    rate: real(0.1, 6, { default: 2, step: 0.1, label: 'prior rate of tau' }),
  }),
  parts: (k) => classesLearnedProblem('logistic', k),
})

// ----- a regression with a log link: Poisson, negative binomial, Weibull -----

/** The grid of x values a figure draws a fitted curve on. */
const CURVE_X = Array.from({ length: 41 }, (_, i) => -1 + i / 20)

/** The covariate of a regression, drawn uniformly on [-1, 1]. */
const covariate = (seed: number, n: number): number[] =>
  Array.from(toFlat(uniform(foldIn(key(seed), 0), [n])), (v) => 2 * v - 1)

/** The figure of a log-link regression: the observations, the true curve and the fitted one. */
function regressionFigure(
  what: string,
  x: readonly number[],
  y: readonly number[],
  curve: (a: number, b: number) => (v: number) => number,
  truth: readonly [number, number],
): AtomFigure {
  const line = (a: number, b: number, name: string, colour?: string): FigureCurve => ({
    name,
    ...(colour === undefined ? {} : { colour }),
    x: CURVE_X,
    y: CURVE_X.map(curve(a, b)),
  })
  return {
    caption: `The simulated ${what}, the true curve (green) and the fitted one at the mean of the belief (thick: the exact; dashed: EP).`,
    xLabel: 'x',
    yLabel: 'y',
    xLimits: [-1, 1],
    points: [{ name: 'observations', x, y }],
    curves: [line(truth[0], truth[1], 'true curve', 'tab2')],
    answer: (m) => [line(m['a'] ?? Number.NaN, m['b'] ?? Number.NaN, 'fitted curve')],
  }
}

function poissonLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueA, trueB] = [configNumber(k, 'trueIntercept'), configNumber(k, 'trueSlope')]
  const priorSd = configNumber(k, 'priorSd')
  // region poisson-log-learned
  // the ground truth: a count at each x with rate exp(intercept + slope x)
  const x = covariate(seed, n)
  const y = x.map((v, i) => item(poisson(foldIn(key(seed), 1 + i), Math.exp(trueA + trueB * v), [1])))
  const definition = defineModel('poisson-log-learned', (m) => {
    const row = m.range('row')
    // the intercept and the slope of the log rate: both unknown
    const a = m.latent('a', d.normal(0, priorSd))
    const b = m.latent('b', d.normal(0, priorSd))
    const rate = ops.exp(a.add(b.mul(m.data('x', { dims: [row] }))))
    m.observe('y', d.poisson(rate), m.data('y', { dims: [row], support: 'count' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { x, y },
      quantities: [
        { name: 'a', label: 'a (intercept of the log rate)', support: 'real', truth: trueA },
        { name: 'b', label: 'b (slope of the log rate)', support: 'real', truth: trueB },
      ],
    },
    reference: {
      grid: { points: 61, maxRefinements: 0 },
    },
    present: {
      equations: [
        `truth: log rate = ${trueA} + ${trueB} x; ${n} counts at x ~ Uniform(-1, 1), seed ${seed}`,
        `a ~ N(0, ${priorSd}²), b ~ N(0, ${priorSd}²)   (the unknown intercept and slope)`,
        `y_i ~ Poisson(exp(a + b x_i)), i = 1..${n}`,
      ],
      pair: ['a', 'b'],
      figure: regressionFigure('counts', x, y, (a, b) => (v) => Math.exp(a + b * v), [trueA, trueB]),
    },
  }
}

/**
 * A Poisson regression with a log link whose intercept and slope are unknown, learned from counts simulated from stated true
 * coefficients: the numeric Poisson likelihood on a linear predictor, with the recovery of the coefficients checked over many
 * seeds.
 */
const poissonLogLearnedAtom: EpAtom = defineAtom({
  name: 'poisson-log-learned',
  title: 'Poisson with a log link: parameters learned',
  extends: 'poisson-log',
  recovery: recovery(
    'standard',
    'a Gaussian projection of each Poisson tilted distribution by quadrature, with a weakly informative prior on the coefficients: the intervals should hold the truth about nine times in ten',
  ),
  what: 'The Poisson count with a log link of the basic atom, now a regression with an unknown intercept and slope, and the counts simulated from stated true coefficients.',
  factors: ['prior (gaussian) x2', 'matmul (linear predictor)', 'numeric likelihood: poisson with an exp link'],
  family: 'likelihoods',
  reference: 'grid',
  tolerance: tolerance('close', 'a Gaussian projection of each tilted distribution, evaluated by quadrature'),
  knobs: dict({
    ...simulationKnobs(30),
    trueIntercept: real(-1, 2, { default: 0.5, step: 0.1, label: 'true intercept' }),
    trueSlope: real(-2, 2, { default: 1.2, step: 0.1, label: 'true slope' }),
    priorSd: real(0.5, 5, { default: 2, step: 0.1, label: 'prior sd of the coefficients' }),
  }),
  parts: poissonLearnedProblem,
})

function dispersionLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueA, trueB, trueAlpha] = [
    configNumber(k, 'trueIntercept'),
    configNumber(k, 'trueSlope'),
    configNumber(k, 'trueDispersion'),
  ]
  const priorSd = configNumber(k, 'priorSd')
  // region poisson-log-dispersion-learned
  // the ground truth: counts whose rate is jittered by a gamma (overdispersion: variance = mean + alpha mean²)
  const kk = key(seed)
  const x = covariate(seed, n)
  const jitter = toFlat(gamma(foldIn(kk, 7), [n], 1 / trueAlpha))
  const mean = x.map((v) => Math.exp(trueA + trueB * v))
  const rates = mean.map((mu, i) => mu * trueAlpha * (jitter[i] ?? 1))
  const y = rates.map((r, i) => item(poisson(foldIn(kk, 100 + i), r, [1])))
  const definition = defineModel('poisson-log-dispersion-learned', (m) => {
    const row = m.range('row')
    // the intercept, the slope and the dispersion: all unknown
    const a = m.latent('a', d.normal(0, priorSd))
    const b = m.latent('b', d.normal(0, priorSd))
    const alpha = m.latent('alpha', d.halfNormal(1))
    const rate = ops.exp(a.add(b.mul(m.data('x', { dims: [row] }))))
    const nb = d.negativeBinomial(ops.div(1, alpha), ops.div(1, ops.add(1, alpha.mul(rate))))
    m.observe('y', nb, m.data('y', { dims: [row], support: 'count' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { x, y },
      quantities: [
        { name: 'a', label: 'a (intercept of the log mean)', support: 'real', truth: trueA },
        { name: 'b', label: 'b (slope of the log mean)', support: 'real', truth: trueB },
        { name: 'alpha', label: 'alpha (the dispersion)', support: 'positive', truth: trueAlpha },
      ],
    },
    reference: {
      grid: { points: 27, maxRefinements: 0 },
    },
    present: {
      equations: [
        `truth: mean exp(${trueA} + ${trueB} x), dispersion ${trueAlpha} (variance = mean + ${trueAlpha} mean²); ${n} counts, seed ${seed}`,
        `a ~ N(0, ${priorSd}²), b ~ N(0, ${priorSd}²), alpha ~ HalfNormal(1)   (intercept, slope, dispersion)`,
        `y_i ~ NegativeBinomial(mean exp(a + b x_i), dispersion alpha), i = 1..${n}`,
      ],
      pair: ['a', 'b'],
      figure: regressionFigure('counts', x, y, (a, b) => (v) => Math.exp(a + b * v), [trueA, trueB]),
    },
  }
}

/**
 * The same regression for overdispersed counts: a negative binomial with a log link whose dispersion is unknown too (the
 * `negative-binomial-dispersion` operator, a two-variable numeric site), learned from counts simulated with a stated dispersion.
 */
const poissonLogDispersionLearnedAtom: EpAtom = defineAtom({
  name: 'poisson-log-dispersion-learned',
  title: 'Poisson with a log link: parameters learned (overdispersed)',
  extends: 'poisson-log',
  recovery: recovery(
    'measured',
    'the dispersion is the weakest-identified parameter (it needs many counts) and EP carries one Gaussian for the intercept, the slope and the log dispersion at each site; each run costs about a third of a second, so 25 seeds and a floor of 0.7 (a binomial spread of 0.06)',
    { seeds: 25, coverage: 0.7, zrms: 1.6 },
  ),
  what: 'The Poisson regression with a log link whose counts are overdispersed: a negative binomial with an unknown dispersion alongside the unknown intercept and slope, simulated from stated true values.',
  factors: [
    'prior (gaussian) x2',
    'numeric prior (half-normal)',
    'negative-binomial-dispersion (a two-variable numeric site)',
  ],
  family: 'likelihoods',
  reference: 'grid',
  tolerance: tolerance(
    'lumpiest',
    'a Gauss-Hermite product rule over the Gaussian beliefs of the predictor and the log dispersion, against a 27-point grid: close, not exact',
  ),
  ep: { damping: 0.5 },
  knobs: dict({
    ...simulationKnobs(25),
    trueIntercept: real(-1, 2, { default: 0.8, step: 0.1, label: 'true intercept' }),
    trueSlope: real(-2, 2, { default: 1, step: 0.1, label: 'true slope' }),
    trueDispersion: real(0.1, 2, { default: 0.5, step: 0.05, label: 'true dispersion' }),
    priorSd: real(0.5, 5, { default: 2, step: 0.1, label: 'prior sd of the coefficients' }),
  }),
  parts: dispersionLearnedProblem,
})

const WEIBULL_SHAPE = 1.5

function weibullLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueA, trueB] = [configNumber(k, 'trueIntercept'), configNumber(k, 'trueSlope')]
  const priorSd = configNumber(k, 'priorSd')
  const shape = WEIBULL_SHAPE
  // region weibull-learned
  // the ground truth: Weibull times with scale exp(intercept + slope x) and a known shape
  const x = covariate(seed, n)
  const u = toFlat(uniform(foldIn(key(seed), 1), [n]))
  const wait = (i: number): number => (-Math.log(1 - (u[i] ?? 0.5))) ** (1 / shape)
  const y = x.map((v, i) => Math.exp(trueA + trueB * v) * wait(i))
  const definition = defineModel('weibull-learned', (m) => {
    const row = m.range('row')
    // the intercept and the slope of the log scale: both unknown
    const a = m.latent('a', d.normal(0, priorSd))
    const b = m.latent('b', d.normal(0, priorSd))
    const scale = ops.exp(a.add(b.mul(m.data('x', { dims: [row] }))))
    m.observe('y', d.weibull(shape, scale), m.data('y', { dims: [row], support: 'positive' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { x, y },
      quantities: [
        { name: 'a', label: 'a (intercept of the log scale)', support: 'real', truth: trueA },
        { name: 'b', label: 'b (slope of the log scale)', support: 'real', truth: trueB },
      ],
    },
    reference: {
      grid: { points: 61, maxRefinements: 0 },
    },
    present: {
      equations: [
        `truth: Weibull times, shape ${shape} (known), scale exp(${trueA} + ${trueB} x); ${n} times, seed ${seed}`,
        `a ~ N(0, ${priorSd}²), b ~ N(0, ${priorSd}²)   (the unknown intercept and slope of the log scale)`,
        `y_i ~ Weibull(${shape}, exp(a + b x_i)), i = 1..${n}`,
      ],
      pair: ['a', 'b'],
      figure: regressionFigure('times', x, y, (a, b) => (v) => Math.exp(a + b * v), [trueA, trueB]),
    },
  }
}

/**
 * A Weibull regression with a log link on the scale (the shape known), with an unknown intercept and slope, learned from times
 * simulated from stated true coefficients: the numeric Weibull likelihood on a linear predictor. The shape stays fixed: a random
 * shape has no rule in the library yet.
 */
const weibullLearnedAtom: EpAtom = defineAtom({
  name: 'weibull-learned',
  title: 'Prior: normal on a Weibull log scale, normal hyperprior on its centre: parameters learned',
  extends: 'prior-normal-weibull-scale',
  recovery: recovery(
    'standard',
    'a Gaussian projection of each Weibull tilted distribution by quadrature; the times carry less information per observation than a count, so the intervals are wide and the prior pulls a little',
  ),
  what: 'The Weibull time with a normal prior on the log of its scale, as a regression: the intercept and slope of the log scale are unknown (the centre and the effect of a covariate) and the times are simulated from stated true values, with the shape fixed.',
  factors: ['prior (gaussian) x2', 'matmul (linear predictor)', 'numeric likelihood: weibull with a log link'],
  family: 'likelihoods',
  reference: 'grid',
  tolerance: tolerance('close', 'a Gaussian projection of each tilted distribution, evaluated by quadrature'),
  knobs: dict({
    ...simulationKnobs(30),
    trueIntercept: real(-1, 2, { default: 0.5, step: 0.1, label: 'true intercept' }),
    trueSlope: real(-2, 2, { default: -0.8, step: 0.1, label: 'true slope' }),
    priorSd: real(0.5, 5, { default: 2, step: 0.1, label: 'prior sd of the coefficients' }),
  }),
  parts: weibullLearnedProblem,
})

// ----- two-variable sites: a Student-t with learned degrees of freedom, a Weibull with a learned shape -----

/**
 * The marginal of a positive value from the marginal of its log (the coordinate `t = log x` of a grid): the density of the value
 * carries the Jacobian, and its moments are integrated over the grid's range by the midpoint rule.
 */
function positiveFromLog(base: ExactMarginal): ExactMarginal {
  const [lo, hi] = base.range ?? [base.mean - 8 * base.sd, base.mean + 8 * base.sd]
  const n = 4000
  const h = (hi - lo) / n
  let z = 0
  let m1 = 0
  let m2 = 0
  for (let i = 0; i < n; i++) {
    const t = lo + (i + 0.5) * h
    const w = base.pdf(t) * h
    const v = Math.exp(t)
    z += w
    m1 += w * v
    m2 += w * v * v
  }
  const mean = m1 / z
  return {
    mean,
    sd: Math.sqrt(Math.max(m2 / z - mean * mean, 0)),
    pdf: (x) => (x > 0 ? base.pdf(Math.log(x)) / x : 0),
    range: [Math.exp(lo), Math.exp(hi)],
  }
}

function studentDfLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueLoc, trueDf] = [configNumber(k, 'trueLoc'), configNumber(k, 'trueDf')]
  const [a, b] = [configNumber(k, 'shape'), configNumber(k, 'rate')]
  // region student-t-df-learned
  // the ground truth the data are drawn from: Student-t noise with a stated number of degrees of freedom
  const noise = toFlat(studentT(foldIn(key(seed), 0), [n], trueDf))
  const y = Array.from(noise, (e) => trueLoc + e)
  const definition = defineModel('student-t-df-learned', (m) => {
    // the location and the degrees of freedom of the noise (its scale is one): both unknown
    const mu = m.latent('mu', d.normal(0, 5))
    const df = m.latent('df', d.gamma(a, b))
    m.observe('y', d.studentT(df, mu, 1), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  // The exact posterior of (mu, log df) on a grid: the gamma prior of df in the coordinate log df is df^a exp(-b df).
  const sorted = [...y].sort((p, q) => p - q)
  const med = sorted[Math.floor(n / 2)] ?? 0
  const half = 7 / Math.sqrt(n) + 0.5
  const logT = (v: number, mu: number, df: number): number =>
    lg((df + 1) / 2) - lg(df / 2) - 0.5 * Math.log(df * Math.PI) - ((df + 1) / 2) * Math.log1p((v - mu) ** 2 / df)
  const grid = once(() =>
    gridMarginals(
      ([mu, t]) => {
        const df = Math.exp(t as number)
        return (
          -0.5 * ((mu as number) / 5) ** 2 + a * (t as number) - b * df + sum(y.map((v) => logT(v, mu as number, df)))
        )
      },
      [
        [med - half, med + half],
        [Math.log(0.4), Math.log(250)],
      ],
      141,
    ),
  )
  const x = [0, n + 1]
  return {
    build: {
      definition,
      data: { y },
      quantities: [
        { name: 'mu', label: 'mu (the location)', support: 'real', truth: trueLoc },
        { name: 'df', label: 'df (degrees of freedom)', support: 'positive', truth: trueDf },
      ],
    },
    reference: {
      get exact() {
        return { mu: grid().marginal(0), df: positiveFromLog(grid().marginal(1)) }
      },
      referenceSource: 'the posterior of the location and the log of the degrees of freedom on a 141 by 141 grid',
    },
    present: {
      equations: [
        `truth: location ${trueLoc}, scale 1, Student-t noise with ${trueDf} degrees of freedom; ${n} draws with seed ${seed}`,
        'mu ~ N(0, 5²)   (the unknown location)',
        `df ~ Gamma(${a}, ${b})   (the unknown degrees of freedom)`,
        `y_i ~ Student-t(df, mu, 1), i = 1..${n}`,
      ],
      figure: {
        caption:
          'The simulated observations (heavy tails: a few lie far from the rest). Green: the true location; thick and dashed: the fitted one.',
        xLabel: 'observation',
        yLabel: 'y',
        xLimits: x as [number, number],
        points: [{ name: 'observations', x: y.map((_, i) => i + 1), y }],
        curves: [{ name: 'true location', colour: 'tab2', x, y: [trueLoc, trueLoc] }],
        answer: (mm) => [{ name: 'mu', x, y: [mm['mu'] ?? Number.NaN, mm['mu'] ?? Number.NaN] }],
      },
    },
  }
}

/**
 * A location observed through Student-t noise whose degrees of freedom are unknown too (a gamma prior on them), learned from data
 * simulated from a stated truth: the `likelihood-pair` factor reads the location and the degrees of freedom (a gamma variable,
 * integrated in its log) together, against the exact posterior of both on a grid. How heavy the tails are is itself learned.
 */
const studentTDfLearnedAtom: EpAtom = defineAtom({
  name: 'student-t-df-learned',
  title: 'Student-t likelihood: parameters learned (the degrees of freedom too)',
  extends: 'student-t',
  recovery: recovery(
    'quick',
    'the degrees of freedom are weakly identified by a few dozen draws (the posterior is skewed with a long right tail, and EP carries a gamma for it), so the floor leaves room (the exact posterior covers 0.96 of 25 seeds at these defaults, EP the same); each run integrates a grid in the location and the log of the degrees of freedom, so 20 seeds and a floor of 0.7',
  ),
  what: 'The Student-t noise of the basic atom (scale one) with the location and the degrees of freedom both unknown, the data simulated from a stated location and a stated number of degrees of freedom.',
  factors: ['prior (gaussian)', 'prior (gamma)', 'likelihood-pair: student-t with random degrees of freedom'],
  family: 'likelihoods',
  reference: 'exact',
  tolerance: tolerance(
    'coupled',
    'the posterior of the degrees of freedom is skewed with a long right tail and EP carries a gamma for it; the location is nearly Gaussian',
  ),
  evidence: { nats: 0.02, reason: 'a Student-t with a gamma df: the pair site (0.006 measured)' },
  ep: { damping: 0.3 },
  knobs: dict({
    ...simulationKnobs(40),
    trueLoc: real(-3, 3, { default: 1, step: 0.1, label: 'true location' }),
    trueDf: real(1.5, 20, { default: 5, step: 0.5, label: 'true degrees of freedom' }),
    shape: real(0.5, 8, { default: 3, step: 0.1, label: 'prior shape of df' }),
    rate: real(0.02, 2, { default: 0.4, step: 0.02, label: 'prior rate of df' }),
  }),
  parts: studentDfLearnedProblem,
})

function weibullShapeLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueLogScale, trueShape] = [configNumber(k, 'trueLogScale'), configNumber(k, 'trueShape')]
  const [priorSd, shapeSd] = [configNumber(k, 'priorSd'), configNumber(k, 'shapeSd')]
  // region weibull-shape-learned
  // the ground truth: Weibull times with a stated scale and shape
  const u = toFlat(uniform(foldIn(key(seed), 1), [n]))
  const y = Array.from(u, (v) => Math.exp(trueLogScale) * (-Math.log(1 - v)) ** (1 / trueShape))
  const definition = defineModel('weibull-shape-learned', (m) => {
    // the log of the scale and the shape: both unknown
    const logScale = m.latent('logScale', d.normal(0, priorSd))
    const shape = m.latent('shape', d.logNormal(0.4, shapeSd))
    const times = m.data('y', { dims: ['i'], support: 'positive' })
    m.observe('y', d.weibull(shape, ops.exp(logScale)), times)
  })
  // endregion
  const u0 = sum(y.map((v) => Math.log(v))) / n
  const grid = once(() =>
    gridMarginals(
      ([ls, t]) => {
        const shape = Math.exp(t as number)
        return (
          -0.5 * ((ls as number) / priorSd) ** 2 -
          (0.5 * ((t as number) - 0.4) ** 2) / shapeSd ** 2 +
          sum(
            y.map(
              (v) =>
                (t as number) +
                (shape - 1) * Math.log(v) -
                shape * (ls as number) -
                (v * Math.exp(-(ls as number))) ** shape,
            ),
          )
        )
      },
      [
        [u0 - 1.2, u0 + 1.2],
        [Math.log(0.3), Math.log(9)],
      ],
      141,
    ),
  )
  const x = y.map((_, i) => i + 1)
  const median = (ls: number, shape: number): number => Math.exp(ls) * Math.LN2 ** (1 / shape)
  return {
    build: {
      definition,
      data: { y },
      quantities: [
        { name: 'logScale', label: 'logScale (the log of the Weibull scale)', support: 'real', truth: trueLogScale },
        { name: 'shape', label: 'shape (of the Weibull)', support: 'positive', truth: trueShape },
      ],
    },
    reference: {
      get exact() {
        return {
          logScale: grid().marginal(0),
          shape: positiveFromLog(grid().marginal(1)),
        }
      },
      referenceSource: 'the posterior of the log scale and the log shape on a 141 by 141 grid',
    },
    present: {
      equations: [
        `truth: Weibull times, scale exp(${trueLogScale}), shape ${trueShape}; ${n} times, seed ${seed}`,
        `logScale ~ N(0, ${priorSd}²)   (the unknown log of the scale)`,
        `shape ~ LogNormal(0.4, ${shapeSd}²)   (the unknown shape)`,
        `y_i ~ Weibull(shape, exp(logScale)), i = 1..${n}`,
      ],
      figure: {
        caption:
          'The simulated times. Green: the true median; thick and dashed: the median at the fitted scale and shape.',
        xLabel: 'observation',
        yLabel: 'time',
        xLimits: [0, n + 1],
        points: [{ name: 'times', x, y }],
        curves: [
          {
            name: 'true median',
            colour: 'tab2',
            x: [0, n + 1],
            y: [median(trueLogScale, trueShape), median(trueLogScale, trueShape)],
          },
        ],
        answer: (mm) => {
          const v = median(mm['logScale'] ?? Number.NaN, mm['shape'] ?? Number.NaN)
          return [{ name: 'median', x: [0, n + 1], y: [v, v] }]
        },
      },
    },
  }
}

/**
 * A Weibull time with an unknown scale and an unknown shape (a log-normal prior on it), learned from times simulated from a
 * stated truth: the `likelihood-pair` factor reads the log scale and the log shape together, against the exact posterior of both on
 * a grid. The shape of the Weibull (increasing or decreasing hazard) is itself learned.
 */
const weibullShapeLearnedAtom: EpAtom = defineAtom({
  name: 'weibull-shape-learned',
  title: 'Prior: normal on a Weibull log scale, normal hyperprior on its centre: parameters learned (the shape too)',
  extends: 'prior-normal-weibull-scale',
  recovery: recovery(
    'quick',
    'the shape and the log scale are correlated in the posterior and EP carries one Gaussian for each (the log-normal shape is a Gaussian on its log); each run integrates a grid in both, so 20 seeds and a floor of 0.7',
  ),
  what: 'The Weibull time of the basic atom with its log scale and its shape both unknown, the times simulated from a stated scale and shape.',
  factors: ['prior (gaussian)', 'prior (gaussian, on the log shape)', 'likelihood-pair: weibull with a random shape'],
  family: 'likelihoods',
  reference: 'exact',
  tolerance: tolerance(
    'coupled',
    'the scale and the shape are coupled in the posterior (a larger shape moves the scale), and EP carries one Gaussian for each',
  ),
  evidence: { nats: 0.08, reason: 'a Weibull with a log-normal shape: the pair site (0.036 measured)' },
  ep: { damping: 0.3 },
  knobs: dict({
    ...simulationKnobs(30),
    trueLogScale: real(-1, 2, { default: 0.5, step: 0.1, label: 'true log scale' }),
    trueShape: real(0.6, 4, { default: 1.8, step: 0.1, label: 'true shape' }),
    priorSd: real(0.5, 5, { default: 2, step: 0.1, label: 'prior sd of the log scale' }),
    shapeSd: real(0.2, 2, { default: 0.7, step: 0.1, label: 'prior sd of the log shape' }),
  }),
  parts: weibullShapeLearnedProblem,
})

// ----- a tournament in which the spread of the skills is unknown -----

function skillsLearnedProblem(k: Config): AtomParts {
  const [seed, players, rounds] = [configNumber(k, 'seed'), configNumber(k, 'players'), configNumber(k, 'n')]
  const [spread, a, b] = [configNumber(k, 'trueSpread'), configNumber(k, 'shape'), configNumber(k, 'rate')]
  // region skill-difference-learned
  // the ground truth: skills drawn with a spread, a round robin in which the better player usually wins
  const kk = key(seed)
  const skills = Array.from(toFlat(normal(foldIn(kk, 0), [players], { scale: spread })))
  const pairs: [number, number][] = []
  for (let i = 0; i < players; i++) for (let j = i + 1; j < players; j++) pairs.push([i, j])
  const games: [number, number][] = []
  for (let r = 0; r < rounds; r++) for (const [i, j] of pairs) games.push(r % 2 ? [j, i] : [i, j])
  const u = toFlat(uniform(foldIn(kk, 1), [games.length]))
  const gap = games.map(([p, q]) => ((skills[p] ?? 0) - (skills[q] ?? 0)) / Math.SQRT2)
  const win = gap.map((g, i) => ((u[i] ?? 1) < normalCdfFast(g) ? 1 : 0))
  const definition = defineModel('skill-difference-learned', (m) => {
    const player = m.range('player')
    const game = m.range('game')
    const first = m.data('a', { dims: [game], support: 'index', role: 'index', indexes: player })
    const second = m.data('b', { dims: [game], support: 'index', role: 'index', indexes: player })
    // how spread out the players are: unknown
    const tau = m.latent('tau', d.gamma(a, b))
    const skill = m.latent('skill', d.normal(0, ops.div(1, ops.sqrt(tau))), { dims: [player] })
    const diff = ops.div(skill.at(first).sub(skill.at(second)), Math.SQRT2)
    const won = m.data('win', { dims: [game], support: 'binary' })
    m.observe('win', d.bernoulli({ probs: ops.probit(diff) }), won)
  })
  // endregion
  const at = skills.map((_, i) => i + 1)
  return {
    build: {
      definition,
      data: { a: games.map((g) => g[0]), b: games.map((g) => g[1]), win },
      bindOptions: { sizes: { player: players } },
      quantities: [
        { name: 'tau', label: 'tau (the precision of the skills)', support: 'positive', truth: 1 / spread ** 2 },
        ...skills.map((t, j) => ({
          name: `skill[${j}]`,
          label: `skill of player ${j + 1}`,
          support: 'real' as const,
          truth: t,
          pool: 'skill (every player)',
        })),
      ],
    },
    present: {
      equations: [
        `truth: ${players} skills ~ N(0, ${spread}²) (precision ${Number((1 / spread ** 2).toPrecision(3))}); a round robin played ${rounds} times (${games.length} games), seed ${seed}`,
        `tau ~ Gamma(${a}, ${b})   (the unknown precision of the skills: how spread out the players are)`,
        `skill_j ~ N(0, 1 / tau), j = 1..${players}`,
        'game g between a and b: a wins ~ Bernoulli(Phi((skill_a - skill_b) / sqrt 2))',
      ],
      figure: {
        caption:
          'The true skill of each player (open marks) and what EP believes of it (bars) after the tournament: the better players are found, and the spread tau they share is learned from the same games.',
        xLabel: 'player',
        yLabel: 'skill',
        xLimits: [0, players + 1],
        points: [{ name: 'true skill', x: at, y: skills }],
        answer: (mm) =>
          at.map((j) => ({
            name: `player ${j}`,
            x: [j - 0.3, j + 0.3],
            y: [mm[`skill[${j - 1}]`] ?? Number.NaN, mm[`skill[${j - 1}]`] ?? Number.NaN],
          })),
      },
    },
  }
}

/**
 * A round-robin tournament with a probit factor on the difference of two skills, whose skills share an unknown precision (the
 * spread of the players is learned too), simulated from stated true skills: does EP recover the skills and their spread from the
 * results alone? There are too many latents for a grid, so the check is the recovery of the truth.
 */
const skillDifferenceLearnedAtom: EpAtom = defineAtom({
  name: 'skill-difference-learned',
  title: 'Probit on a difference: parameters learned',
  extends: 'skill-difference',
  recovery: recovery(
    'standard',
    'the spread is learned from a handful of players and a few dozen win/lose results, so it is weakly identified; each result is a truncated-Gaussian tilted distribution projected onto a Gaussian',
  ),
  what: 'The TrueSkill probit factor of the basic atom on a round robin of players whose skills share an unknown precision, with the results simulated from stated true skills: the skills and the spread of the players are recovered from the games alone.',
  factors: ['prior (gamma)', 'gaussian-gamma-precision (latent value)', 'probit link on a difference of two variables'],
  family: 'likelihoods',
  reference: 'truth',
  tolerance: tolerance(
    'truth',
    'there is no reference posterior (a gamma precision and eight skills are too many coordinates for a grid): recovery of the truth is the check',
  ),
  knobs: dict({
    seed: simulationKnobs(2)['seed'] as ReturnType<typeof int>,
    players: int(3, 10, { default: 8, label: 'players' }),
    n: int(1, 6, { default: 2, label: 'rounds (each pair plays this many games)' }),
    trueSpread: real(0.3, 2.5, { default: 1, step: 0.1, label: 'true spread (sd) of the skills' }),
    shape: real(0.5, 8, { default: 2, step: 0.1, label: 'prior shape of tau' }),
    rate: real(0.1, 8, { default: 2, step: 0.1, label: 'prior rate of tau' }),
  }),
  parts: skillsLearnedProblem,
})

/** The atoms of the `likelihoods` file, basic and learned twins together. */
export const likelihoodsAtoms: readonly EpAtom[] = [
  studentTAtom,
  laplaceAtom,
  probitAtom,
  logisticAtom,
  skillDifferenceAtom,
  gpLogisticAtom,
  gpProbitAtom,
  poissonLogAtom,
  studentTLearnedAtom,
  laplaceLearnedAtom,
  probitLearnedAtom,
  logisticLearnedAtom,
  poissonLogLearnedAtom,
  poissonLogDispersionLearnedAtom,
  weibullLearnedAtom,
  studentTDfLearnedAtom,
  weibullShapeLearnedAtom,
  skillDifferenceLearnedAtom,
]
