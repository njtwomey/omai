/**
 * The setups of expectation propagation about likelihoods: heavy-tailed noise (a Student-t and a Laplace likelihood on a
 * mean, with one observation far from the rest), a probit and a logistic link on eight labelled points, the probit factor on
 * a difference of skills (three players and five matches), a Poisson count with a log link and a Gaussian process prior with a
 * logistic likelihood. References are grids of the model written as a program.
 */

import { gram, rbfKernel, scaleKernel } from '../kernels/index.ts'
import * as d from '../probprog/model/dists/index.ts'
import { defineModel, ops } from '../probprog/model/index.ts'
import { probitClassifier, probitData } from '../probprog/models/probit-classification.ts'
import { trueskillModel } from '../probprog/models/ranking.ts'
import { dict, real } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import type { Config } from '../trace/index.ts'
import { type AtomProblem, defineAtom, type EpAtom } from './ep-atoms.ts'
import { gridMarginals, normalCdf } from './ep-atoms-quad.ts'
import { exampleNumber } from './example-method.ts'

const HEAVY = [0.2, -0.1, 0.4, 0.0, 4.5]

function heavyProblem(kind: 'student-t' | 'laplace', k: Config): AtomProblem {
  const s = exampleNumber(k, 'scale')
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
    definition,
    data: { y: HEAVY },
    equations: [
      'mu ~ N(0, 5²)',
      kind === 'student-t' ? `y_i ~ Student-t(df = ${df}, mu, ${s})` : `y_i ~ Laplace(mu, ${s})`,
      `observe y = (${HEAVY.join(', ')})   (the last is far from the rest)`,
    ],
    quantities: [{ name: 'mu', label: 'mu', support: 'real' }],
    figure: {
      caption:
        'The five observations; the line is where the belief puts mu. A heavy-tailed likelihood lets the last one be an outlier instead of dragging mu towards it.',
      xLabel: 'observation',
      yLabel: 'y',
      xLimits: [0, HEAVY.length + 1],
      points: [{ name: 'observations', x: HEAVY.map((_, i) => i + 1), y: [...HEAVY] }],
      answer: (m) => [{ name: 'mu', x: [0, HEAVY.length + 1], y: [m['mu'] ?? Number.NaN, m['mu'] ?? Number.NaN] }],
    },
  }
}

/**
 * A mean observed through Student-t noise, one observation far from the rest: the Student-t site (a quadrature), whose
 * influence function redescends so the outlier is not followed.
 *
 * @index estimator EP atom: a Student-t likelihood on a mean (robust to an outlier).
 * @example
 * ```ts
 * import { studentTAtom } from 'omai-engine/bayesian-examples'
 * studentTAtom.name // => 'student-t'
 * ```
 */
export const studentTAtom: EpAtom = defineAtom({
  name: 'student-t',
  title: 'Student-t likelihood',
  what: 'A mean mu observed five times through Student-t noise with three degrees of freedom, one observation being far from the rest: the Student-t site.',
  factors: ['prior (gaussian)', 'student-t-site (quadrature)'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'grid',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.75, 1.25],
    ks: 0.1,
    reason:
      'the posterior is bumpy (the outlier may or may not count) and EP matches its mean and variance with one Gaussian',
  },
  knobs: dict({ scale: real(0.1, 2, { default: 0.5, step: 0.05, label: 'noise scale' }) }),
  problem: (k) => heavyProblem('student-t', k),
})

/**
 * The same with a Laplace likelihood.
 *
 * @index estimator EP atom: a Laplace likelihood on a mean.
 * @example
 * ```ts
 * import { laplaceAtom } from 'omai-engine/bayesian-examples'
 * laplaceAtom.name // => 'laplace'
 * ```
 */
export const laplaceAtom: EpAtom = defineAtom({
  name: 'laplace',
  title: 'Laplace likelihood',
  what: 'A mean mu observed five times through Laplace (double exponential) noise, one observation being far from the rest: the Laplace site, an absolute-error likelihood whose posterior is piecewise Gaussian.',
  factors: ['prior (gaussian)', 'laplace-site (quadrature)'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'grid',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.75, 1.25],
    ks: 0.1,
    reason: 'the posterior has kinks at the observations and EP matches its mean and variance with one Gaussian',
  },
  knobs: dict({ scale: real(0.1, 2, { default: 0.5, step: 0.05, label: 'noise scale' }) }),
  problem: (k) => heavyProblem('laplace', k),
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

function classesProblem(link: 'probit' | 'logistic', k: Config): AtomProblem {
  const s = exampleNumber(k, 'priorScale')
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
    definition,
    data: probitData(design, y),
    equations: [
      `w_0, w_1, w_2 ~ N(0, ${s}²)   (offset and two slopes)`,
      link === 'probit'
        ? 'y_i ~ Bernoulli(Phi(w_0 + w_1 x_i1 + w_2 x_i2)), i = 1..8'
        : 'y_i ~ Bernoulli(sigmoid(w_0 + w_1 x_i1 + w_2 x_i2)), i = 1..8',
      'observe the eight labelled points drawn in the figure',
    ],
    quantities: [0, 1, 2].map((j) => ({
      name: `w[${j}]`,
      label: ['w_0 (offset)', 'w_1 (slope in x1)', 'w_2 (slope in x2)'][j] ?? `w_${j}`,
      support: 'real' as const,
    })),
    pair: ['w[1]', 'w[2]'],
    grid: { points: 25, maxRefinements: 0 },
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
  }
}

/**
 * Eight labelled points and a probit link on a linear predictor: the probit factor, whose tilted moments are those of a
 * truncated Gaussian in closed form.
 *
 * @index estimator EP atom: a probit link on eight labelled points.
 * @example
 * ```ts
 * import { probitAtom } from 'omai-engine/bayesian-examples'
 * probitAtom.name // => 'probit'
 * ```
 */
export const probitAtom: EpAtom = defineAtom({
  name: 'probit',
  title: 'Probit',
  what: 'Three weights (an offset and two slopes) with Gaussian priors and eight binary labels through a probit link: the probit factor, a truncated Gaussian with closed-form moments, on a linear predictor.',
  factors: ['prior (gaussian)', 'matmul (linear predictor)', 'probit link'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'grid',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.75, 1.15],
    ks: 0.1,
    reason:
      'EP matches a Gaussian to each truncated-Gaussian tilted distribution; the posterior is skewed, so it is close, not exact',
  },
  knobs: dict({ priorScale: real(0.5, 6, { default: 2, step: 0.1, label: 'prior sd of the weights' }) }),
  problem: (k) => classesProblem('probit', k),
})

/**
 * The same eight points through a logistic link: no closed form for the tilted moments, so the rule integrates numerically.
 *
 * @index estimator EP atom: a logistic link on eight labelled points (numeric tilted moments).
 * @example
 * ```ts
 * import { logisticAtom } from 'omai-engine/bayesian-examples'
 * logisticAtom.family // => 'likelihoods'
 * ```
 */
export const logisticAtom: EpAtom = defineAtom({
  name: 'logistic',
  title: 'Logistic',
  what: 'The same three weights and eight labels through a logistic link: no closed-form tilted moments, so the rule is a numeric quadrature.',
  factors: ['prior (gaussian)', 'matmul (linear predictor)', 'logistic link (numeric quadrature)'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'grid',
  tolerance: {
    meanZ: 0.1,
    sdRatio: [0.8, 1.15],
    ks: 0.06,
    reason: 'a Gaussian projection of each logistic tilted distribution, evaluated by quadrature',
  },
  knobs: dict({ priorScale: real(0.5, 6, { default: 2, step: 0.1, label: 'prior sd of the weights' }) }),
  problem: (k) => classesProblem('logistic', k),
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

function skillsProblem(): AtomProblem {
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
    definition,
    data,
    equations: [
      'skill_A, skill_B, skill_C ~ N(0, 1)',
      'match g between a and b: a wins ~ Bernoulli(Phi((skill_a - skill_b) / sqrt 2))',
      ...notes,
    ],
    quantities: names.map((n) => ({ name: `skill_${n}`, label: `skill_${n}`, support: 'real' as const })),
    pair: ['skill_A', 'skill_B'],
    grid: { points: 25, maxRefinements: 0 },
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
  }
}

/**
 * Three players with Gaussian skills and five match results, each a probit factor on a difference of two skills.
 *
 * @index estimator EP atom: a probit factor on a difference of two skills (three players, five matches).
 * @example
 * ```ts
 * import { skillDifferenceAtom } from 'omai-engine/bayesian-examples'
 * skillDifferenceAtom.name // => 'skill-difference'
 * ```
 */
export const skillDifferenceAtom: EpAtom = defineAtom({
  name: 'skill-difference',
  title: 'Probit on a difference',
  what: 'Three Gaussian skills and five results, each a probit factor on the difference of two skills (the TrueSkill performance model): every factor joins two variables.',
  factors: ['prior (gaussian) x3', 'probit link on a difference of two variables'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'grid',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.75, 1.2],
    ks: 0.1,
    reason:
      'each result is a truncated-Gaussian tilted distribution that EP projects onto a Gaussian: close, not exact',
  },
  knobs: dict({}),
  problem: () => skillsProblem(),
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

function gpProblem(link: 'probit' | 'logistic', k: Config): AtomProblem {
  const ls = exampleNumber(k, 'lengthScale')
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
    definition,
    data: { y: GP_Y },
    exact: Object.fromEntries(g.marginals.map((m, i) => [`f[${i}]`, m])),
    exactJoint: { a: 'f[0]', b: 'f[1]', ...pair },
    equations: [
      `f ~ N(0, K),  K = 1.5 exp(-(x - x')² / (2 · ${ls}²))  at x = (${GP_X.map((r) => r[0]).join(', ')})   (a dense Gaussian vector)`,
      link === 'probit' ? 'y_i ~ Bernoulli(Phi(f_i))' : 'y_i ~ Bernoulli(sigmoid(f_i))',
      `observe y = (${GP_Y.join(', ')})`,
    ],
    quantities: GP_X.map((_, i) => ({ name: `f[${i}]`, label: `f(x_${i + 1})`, support: 'real' as const })),
    pair: ['f[0]', 'f[1]'],
  }
}

/**
 * A Gaussian process prior over three inputs (a dense Gaussian vector) with a logistic likelihood on each: the dense prior
 * message of GP classification.
 *
 * @index estimator EP atom: a dense Gaussian-process prior with a logistic likelihood (three inputs).
 * @example
 * ```ts
 * import { gpLogisticAtom } from 'omai-engine/bayesian-examples'
 * gpLogisticAtom.name // => 'gp-logistic'
 * ```
 */
export const gpLogisticAtom: EpAtom = defineAtom({
  name: 'gp-logistic',
  title: 'Gaussian process with a logistic likelihood',
  what: 'A dense Gaussian prior on three function values (an RBF kernel) and a logistic Bernoulli observation of each: the dense Gaussian-vector prior factor of GP classification.',
  factors: ['dense prior (gaussian-mv)', 'logistic link (numeric quadrature)'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.1,
    sdRatio: [0.8, 1.15],
    ks: 0.06,
    reason: 'a Gaussian projection of each logistic tilted distribution against a dense Gaussian cavity',
  },
  knobs: dict({ lengthScale: real(0.3, 3, { default: 1, step: 0.1, label: 'kernel length scale' }) }),
  problem: (k) => gpProblem('logistic', k),
})

/**
 * The same with a probit likelihood.
 *
 * @index estimator EP atom: a dense Gaussian-process prior with a probit likelihood (three inputs).
 * @example
 * ```ts
 * import { gpProbitAtom } from 'omai-engine/bayesian-examples'
 * gpProbitAtom.name // => 'gp-probit'
 * ```
 */
export const gpProbitAtom: EpAtom = defineAtom({
  name: 'gp-probit',
  title: 'Gaussian process with a probit likelihood',
  what: 'A dense Gaussian prior on three function values and a probit Bernoulli observation of each: GP classification with closed-form tilted moments.',
  factors: ['dense prior (gaussian-mv)', 'probit link'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.75, 1.15],
    ks: 0.1,
    reason: 'a Gaussian projection of each truncated-Gaussian tilted distribution against a dense Gaussian cavity',
  },
  knobs: dict({ lengthScale: real(0.3, 3, { default: 1, step: 0.1, label: 'kernel length scale' }) }),
  problem: (k) => gpProblem('probit', k),
})

function poissonLogProblem(k: Config): AtomProblem {
  const y = exampleNumber(k, 'y')
  // region poisson-log
  const definition = defineModel('poisson-log-link', (m) => {
    const x = m.latent('x', d.normal(0, 1))
    m.observe('y', d.poisson(ops.exp(x)), m.data('y', { support: 'count' }))
  })
  // endregion
  return {
    definition,
    data: { y },
    equations: ['x ~ N(0, 1)', 'y ~ Poisson(exp(x))   (a log link)', `observe y = ${y}`],
    quantities: [{ name: 'x', label: 'x', support: 'real' }],
  }
}

/**
 * A Poisson count with a log link on a Gaussian predictor: the likelihood of a Poisson regression.
 *
 * @index estimator EP atom: a Poisson count with a log link.
 * @example
 * ```ts
 * import { poissonLogAtom } from 'omai-engine/bayesian-examples'
 * poissonLogAtom.name // => 'poisson-log'
 * ```
 */
export const poissonLogAtom: EpAtom = defineAtom({
  name: 'poisson-log',
  title: 'Poisson with a log link',
  what: 'A Gaussian predictor x and a Poisson count with rate exp(x): the numeric likelihood on a linear predictor (the log link of Poisson regression).',
  factors: ['prior (gaussian)', 'numeric likelihood: poisson with an exp link'],
  family: 'likelihoods',
  expected: 'runs',
  reference: 'grid',
  tolerance: {
    meanZ: 0.1,
    sdRatio: [0.8, 1.15],
    ks: 0.06,
    reason: 'a Gaussian projection of the tilted distribution, evaluated by quadrature',
  },
  knobs: dict({ y: real(0, 20, { default: 5, step: 1, integer: true, label: 'observed count' } as never) }),
  problem: poissonLogProblem,
})
