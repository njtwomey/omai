/**
 * The extended twins of the conjugate atoms: the same setups with the numbers the basic atoms fix made unknown and learned.
 * A Gaussian whose mean and variance are both unknown (the normal-inverse-gamma prior), a population of Bernoulli
 * probabilities whose beta prior has an unknown strength, a population of Poisson rates whose gamma prior has an unknown rate.
 * The data are simulated from a stated truth with a seed, EP runs, and the page checks that the truth is recovered (and, over
 * many seeds, that EP's 90% intervals hold it about nine times in ten).
 */

import { lgamma } from '../numerics/special/index.ts'
import * as d from '../probprog/model/dists/index.ts'
import { defineModel, ops } from '../probprog/model/index.ts'
import { beta, foldIn, gamma, key, normal, poisson, uniform } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { item, toFlat } from '../tensor/index.ts'
import type { Config } from '../trace/index.ts'
import { type AtomProblem, type AtomRecoverySpec, defineAtom, type EpAtom } from './ep-atoms.ts'
import { normalInverseGammaPosterior } from './ep-atoms-priors.ts'
import { type ConditionalMarginal, hyperPosterior } from './ep-atoms-quad.ts'
import { exampleNumber } from './example-method.ts'

/** The knobs every extended atom has: the seed of the simulation and the number of observations. */
export const simulationKnobs = (n: number): Record<string, ReturnType<typeof int>> => ({
  seed: int(1, 9999, { default: 1, label: 'seed (the simulated data)' }),
  n: int(3, 200, { default: n, label: 'observations' }),
})

const GAUSSIAN_RECOVERY: AtomRecoverySpec = {
  seeds: 40,
  coverage: 0.75,
  zrms: 1.5,
  reason:
    'the posterior is exact (the model is conjugate), so over seeds drawn from the truth the intervals hold it close to nine times in ten; 40 seeds leave a binomial spread of about 0.05, and the prior pulls a little',
}

function gaussianLearnedProblem(k: Config): AtomProblem {
  const [seed, n] = [exampleNumber(k, 'seed'), exampleNumber(k, 'n')]
  const [trueMean, trueSd] = [exampleNumber(k, 'trueMean'), exampleNumber(k, 'trueSd')]
  const [m0, kappa, a, b] = [
    exampleNumber(k, 'priorMean'),
    exampleNumber(k, 'kappa'),
    exampleNumber(k, 'shape'),
    exampleNumber(k, 'scale'),
  ]
  // region gaussian-learned
  // the ground truth the data are drawn from: the model does not know it
  const y = Array.from(toFlat(normal(foldIn(key(seed), 0), [n], { loc: trueMean, scale: trueSd })))
  const definition = defineModel('gaussian-learned', (m) => {
    // the variance and, given it, the mean: both unknown
    const sigma2 = m.latent('sigma2', d.inverseGamma(a, b))
    const mu = m.latent('mu', d.normal(m0, ops.sqrt(ops.div(sigma2, kappa))))
    m.observe('y', d.normal(mu, ops.sqrt(sigma2)), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  const { mu, sigma2, joint } = normalInverseGammaPosterior(y, m0, kappa, a, b)
  const x = [0, n + 1]
  return {
    definition,
    data: { y },
    equations: [
      `truth: mean ${trueMean}, sd ${trueSd}; ${n} observations drawn with seed ${seed}`,
      `sigma2 ~ InverseGamma(${a}, ${b})   (the unknown variance; prior mean ${sigmaMean(a, b)})`,
      `mu | sigma2 ~ N(${m0}, sigma2 / ${kappa})   (the unknown mean, worth ${kappa} observations)`,
      `y_i ~ N(mu, sigma2), i = 1..${n}`,
    ],
    quantities: [
      { name: 'mu', label: 'mu (the mean)', support: 'real', truth: trueMean },
      { name: 'sigma2', label: 'sigma2 (the variance)', support: 'positive', truth: trueSd ** 2 },
    ],
    pair: ['mu', 'sigma2'],
    exact: { mu, sigma2 },
    exactJoint: joint,
    figure: {
      caption:
        'The simulated observations. Solid lines: the true mean and its two-sd band; thick and dashed: the fitted ones.',
      xLabel: 'observation',
      yLabel: 'y',
      xLimits: x as [number, number],
      points: [{ name: 'observations', x: y.map((_, i) => i + 1), y }],
      curves: [
        { name: 'true mean', colour: 'tab2', x, y: [trueMean, trueMean] },
        { name: 'true mean + 2 sd', colour: 'tab2', x, y: [trueMean + 2 * trueSd, trueMean + 2 * trueSd] },
        { name: 'true mean - 2 sd', colour: 'tab2', x, y: [trueMean - 2 * trueSd, trueMean - 2 * trueSd] },
      ],
      answer: (mm) => {
        const mean = mm['mu'] ?? Number.NaN
        const w = 2 * Math.sqrt(mm['sigma2'] ?? Number.NaN)
        return [
          { name: 'mu', x, y: [mean, mean] },
          { name: 'mu + 2 sd', x, y: [mean + w, mean + w] },
          { name: 'mu - 2 sd', x, y: [mean - w, mean - w] },
        ]
      },
    },
  }
}

const sigmaMean = (a: number, b: number): string => (a > 1 ? String(Number((b / (a - 1)).toPrecision(3))) : 'undefined')

/**
 * The Gaussian with both its mean and its variance unknown, learned from data simulated from a stated truth: the
 * normal-inverse-gamma pair under EP (one normal-gamma variable and a factor per observation), exact, with the recovery of
 * the truth checked over many seeds.
 *
 * @index estimator EP atom: a Gaussian with unknown mean and variance, recovered from simulated data (exact).
 * @example
 * ```ts
 * import { gaussianLearnedAtom } from 'omai-engine/bayesian-examples'
 * gaussianLearnedAtom.extends // => 'gaussian'
 * ```
 */
export const gaussianLearnedAtom: EpAtom = defineAtom({
  name: 'gaussian-learned',
  title: 'Gaussian: parameters learned',
  extends: 'gaussian',
  recovery: GAUSSIAN_RECOVERY,
  what: 'The Gaussian of the basic atom with its mean and its variance both unknown (a normal-inverse-gamma prior), observed through data simulated from a stated mean and sd: does EP recover them, and how often does its 90% interval hold the truth?',
  factors: ['prior (normal-gamma)', 'gaussian-normal-gamma (observed value)'],
  family: 'conjugate',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 1e-6,
    sdRatio: [0.999999, 1.000001],
    ks: 0.1,
    reason:
      'the posterior is exactly normal-inverse-gamma and EP carries it exactly; the curves compare the Gaussian and gamma shapes the page draws with the exact Student t and inverse gamma, so the distance is small, not zero',
  },
  knobs: dict({
    ...simulationKnobs(12),
    trueMean: real(-3, 3, { default: 1.5, step: 0.1, label: 'true mean' }),
    trueSd: real(0.2, 3, { default: 0.8, step: 0.05, label: 'true sd' }),
    priorMean: real(-2, 2, { default: 0, step: 0.1, label: 'prior mean m' }),
    kappa: real(0.05, 5, { default: 0.2, step: 0.05, label: 'kappa (prior observations for the mean)' }),
    shape: real(1.5, 6, { default: 3, step: 0.1, label: 'shape a of the inverse gamma' }),
    scale: real(0.2, 5, { default: 1.5, step: 0.1, label: 'scale b of the inverse gamma' }),
  }),
  problem: gaussianLearnedProblem,
})

const lg = (x: number): number => item(lgamma(x))
const sum = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0)

/** The group of each of `groups * per` observations, group by group. */
const groupIndex = (groups: number, per: number): number[] =>
  Array.from({ length: groups * per }, (_, i) => Math.floor(i / per))

const FIGURE_GROUPS = (groups: number): number[] => Array.from({ length: groups }, (_, j) => j + 1)

function betaLearnedProblem(k: Config): AtomProblem {
  const [seed, groups, per] = [exampleNumber(k, 'seed'), exampleNumber(k, 'groups'), exampleNumber(k, 'n')]
  const trueStrength = exampleNumber(k, 'trueStrength')
  const [a, b] = [exampleNumber(k, 'shape'), exampleNumber(k, 'rate')]
  // region beta-bernoulli-learned
  // the ground truth: a probability per group drawn from a symmetric beta, then flips with it
  const kk = key(seed)
  const p = Array.from(toFlat(beta(foldIn(kk, 0), [groups], trueStrength, trueStrength)))
  const group = groupIndex(groups, per)
  const u = toFlat(uniform(foldIn(kk, 1), [group.length]))
  const y = group.map((j, i) => ((u[i] ?? 1) < (p[j] ?? 0) ? 1 : 0))
  const definition = defineModel('beta-bernoulli-learned', (m) => {
    const g = m.range('group')
    const r = m.range('flip')
    // how concentrated the group probabilities are around one half: unknown
    const strength = m.latent('strength', d.gamma(a, b))
    const prob = m.latent('p', d.beta(strength, strength), { dims: [g] })
    const which = m.data('group', { dims: [r], support: 'index', role: 'index', indexes: g })
    m.observe('y', d.bernoulli({ probs: prob.at(which) }), m.data('y', { dims: [r], support: 'binary' }))
  })
  // endregion
  const counts = p.map((_, j) => sum(y.filter((__, i) => group[i] === j)))
  const logPrior = (s: number): number => (a - 1) * Math.log(s) - b * s
  const post = hyperPosterior(
    (s) => {
      let ll = logPrior(s) - groups * (2 * lg(s) - lg(2 * s))
      for (const c of counts) ll += lg(s + c) + lg(s + per - c) - lg(2 * s + per)
      return ll
    },
    1e-3,
    80,
  )
  const exactP = counts.map((c) =>
    post.mix((s): ConditionalMarginal => {
      const [pa, pb] = [s + c, s + per - c]
      return {
        mean: pa / (pa + pb),
        variance: (pa * pb) / ((pa + pb) ** 2 * (pa + pb + 1)),
        pdf: (x) =>
          x <= 0 || x >= 1
            ? 0
            : Math.exp(lg(pa + pb) - lg(pa) - lg(pb) + (pa - 1) * Math.log(x) + (pb - 1) * Math.log(1 - x)),
      }
    }),
  )
  const at = FIGURE_GROUPS(groups)
  const share = counts.map((c) => c / per)
  return {
    definition,
    data: { y, group },
    bindOptions: { sizes: { group: groups } },
    equations: [
      `truth: strength ${trueStrength}, so p_j ~ Beta(${trueStrength}, ${trueStrength}); ${groups} groups of ${per} flips, seed ${seed}`,
      `strength ~ Gamma(${a}, ${b})   (unknown: how far from one half the probabilities lie; prior mean ${Number((a / b).toPrecision(3))})`,
      `p_j ~ Beta(strength, strength), j = 1..${groups}`,
      `y_ji ~ Bernoulli(p_j), i = 1..${per}`,
    ],
    quantities: [
      { name: 'strength', label: 'strength (the hyperparameter)', support: 'positive', truth: trueStrength },
      ...p.map((t, j) => ({
        name: `p[${j}]`,
        label: `p_${j + 1}`,
        support: 'unit' as const,
        truth: t,
        pool: 'p (every group)',
      })),
    ],
    exact: { strength: post.hyper, ...Object.fromEntries(exactP.map((e, j) => [`p[${j}]`, e])) },
    figure: {
      caption:
        'The share of heads in each group (dots), the true probability of each group (the open marks) and what EP believes of it (bars): the groups with few heads are pulled up and the others down, by an amount the learned strength sets.',
      xLabel: 'group',
      yLabel: 'probability',
      xLimits: [0, groups + 1],
      yLimits: [0, 1],
      points: [
        { name: 'share of heads', x: at, y: share },
        { name: 'true probability', x: at, y: p },
      ],
      answer: (mm) =>
        at.map((j) => ({
          name: `p_${j}`,
          x: [j - 0.3, j + 0.3],
          y: [mm[`p[${j - 1}]`] ?? Number.NaN, mm[`p[${j - 1}]`] ?? Number.NaN],
        })),
    },
  }
}

/**
 * A population of Bernoulli probabilities whose symmetric beta prior has an unknown, gamma-distributed strength, learned from
 * flips simulated from a stated truth: does EP recover the strength (how far the probabilities spread from one half) and each
 * group's probability?
 *
 * @index estimator EP atom: groups of flips with a beta prior of unknown strength, recovered from simulated data.
 * @example
 * ```ts
 * import { betaBernoulliLearnedAtom } from 'omai-engine/bayesian-examples'
 * betaBernoulliLearnedAtom.extends // => 'beta-bernoulli'
 * ```
 */
export const betaBernoulliLearnedAtom: EpAtom = defineAtom({
  name: 'beta-bernoulli-learned',
  title: 'Beta and Bernoulli: parameters learned',
  extends: 'beta-bernoulli',
  recovery: {
    seeds: 40,
    coverage: 0.75,
    zrms: 1.6,
    reason:
      'the posterior of the strength is skewed and EP carries a gamma for it, and each probability is a beta matched to a mixture over the strength: the intervals hold the truth a little less than nine times in ten',
  },
  what: 'The beta and Bernoulli of the basic atom for a population of groups: each group has its own probability drawn from a symmetric beta whose strength is unknown, so the groups teach each other how spread out the probabilities are.',
  factors: [
    'prior (gamma)',
    'dirichlet-concentration (a beta with a random concentration)',
    'bernoulli likelihood (beta-bernoulli)',
  ],
  family: 'conjugate',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.75, 1.25],
    ks: 0.1,
    reason:
      'the strength is skewed and EP carries one gamma for it; each probability is the matched beta of a mixture over it',
  },
  knobs: dict({
    ...simulationKnobs(10),
    groups: int(2, 20, { default: 8, label: 'groups' }),
    trueStrength: real(0.3, 10, { default: 3, step: 0.1, label: 'true strength' }),
    shape: real(0.5, 8, { default: 2, step: 0.1, label: 'shape of the strength prior' }),
    rate: real(0.1, 4, { default: 0.4, step: 0.1, label: 'rate of the strength prior' }),
  }),
  problem: betaLearnedProblem,
})

function gammaLearnedProblem(k: Config): AtomProblem {
  const [seed, groups, per] = [exampleNumber(k, 'seed'), exampleNumber(k, 'groups'), exampleNumber(k, 'n')]
  const [a, trueRate] = [exampleNumber(k, 'shape'), exampleNumber(k, 'trueRate')]
  const [c, dd] = [exampleNumber(k, 'hyperShape'), exampleNumber(k, 'hyperRate')]
  // region gamma-poisson-learned
  // the ground truth: a rate per group drawn from a gamma, then counts with it
  const kk = key(seed)
  const draw = toFlat(gamma(foldIn(kk, 0), [groups], a))
  const rates = Array.from(draw, (v) => v / trueRate)
  const group = groupIndex(groups, per)
  const y = group.map((j, i) => item(poisson(foldIn(kk, 1 + i), rates[j] ?? 1, [1])))
  const definition = defineModel('gamma-poisson-learned', (m) => {
    const g = m.range('group')
    const r = m.range('count')
    // the rate of the prior every group's rate is drawn from: unknown
    const priorRate = m.latent('priorRate', d.gamma(c, dd))
    const rate = m.latent('rate', d.gamma(a, priorRate), { dims: [g] })
    const which = m.data('group', { dims: [r], support: 'index', role: 'index', indexes: g })
    m.observe('y', d.poisson(rate.at(which)), m.data('y', { dims: [r], support: 'count' }))
  })
  // endregion
  const totals = rates.map((_, j) => sum(y.filter((__, i) => group[i] === j)))
  const post = hyperPosterior(
    (b) =>
      (c - 1) * Math.log(b) - dd * b + groups * a * Math.log(b) - sum(totals.map((t) => (a + t) * Math.log(b + per))),
    1e-4,
    200,
  )
  const exactRate = totals.map((t) =>
    post.mix((b): ConditionalMarginal => {
      const [ga, gb] = [a + t, b + per]
      return {
        mean: ga / gb,
        variance: ga / gb ** 2,
        pdf: (x) => (x <= 0 ? 0 : Math.exp(ga * Math.log(gb) - lg(ga) + (ga - 1) * Math.log(x) - gb * x)),
      }
    }),
  )
  const at = FIGURE_GROUPS(groups)
  return {
    definition,
    data: { y, group },
    bindOptions: { sizes: { group: groups } },
    equations: [
      `truth: priorRate ${trueRate}, so rate_j ~ Gamma(${a}, ${trueRate}); ${groups} groups of ${per} counts, seed ${seed}`,
      `priorRate ~ Gamma(${c}, ${dd})   (unknown: the typical size of a rate is ${a} / priorRate)`,
      `rate_j ~ Gamma(${a}, priorRate), j = 1..${groups}`,
      `y_ji ~ Poisson(rate_j), i = 1..${per}`,
    ],
    quantities: [
      { name: 'priorRate', label: 'priorRate (the hyperparameter)', support: 'positive', truth: trueRate },
      ...rates.map((t, j) => ({
        name: `rate[${j}]`,
        label: `rate_${j + 1}`,
        support: 'positive' as const,
        truth: t,
        pool: 'rate (every group)',
      })),
    ],
    exact: { priorRate: post.hyper, ...Object.fromEntries(exactRate.map((e, j) => [`rate[${j}]`, e])) },
    figure: {
      caption:
        'The mean count of each group (dots), the true rate of each group (the open marks) and what EP believes of it (bars): the learned prior rate sets how hard the groups are pulled together.',
      xLabel: 'group',
      yLabel: 'rate',
      xLimits: [0, groups + 1],
      points: [
        { name: 'mean count', x: at, y: totals.map((t) => t / per) },
        { name: 'true rate', x: at, y: rates },
      ],
      answer: (mm) =>
        at.map((j) => ({
          name: `rate_${j}`,
          x: [j - 0.3, j + 0.3],
          y: [mm[`rate[${j - 1}]`] ?? Number.NaN, mm[`rate[${j - 1}]`] ?? Number.NaN],
        })),
    },
  }
}

/**
 * A population of Poisson rates drawn from a gamma whose rate is unknown (gamma-distributed), learned from counts simulated
 * from a stated truth: does EP recover the prior rate and each group's rate?
 *
 * @index estimator EP atom: groups of counts with a gamma prior of unknown rate, recovered from simulated data.
 * @example
 * ```ts
 * import { gammaPoissonLearnedAtom } from 'omai-engine/bayesian-examples'
 * gammaPoissonLearnedAtom.extends // => 'gamma-poisson'
 * ```
 */
export const gammaPoissonLearnedAtom: EpAtom = defineAtom({
  name: 'gamma-poisson-learned',
  title: 'Gamma and Poisson: parameters learned',
  extends: 'gamma-poisson',
  recovery: {
    seeds: 40,
    coverage: 0.75,
    zrms: 1.6,
    reason:
      'the posterior of the prior rate is skewed and EP carries a gamma for it; each rate is a gamma matched to a mixture over it',
  },
  what: 'The gamma and Poisson of the basic atom for a population of groups: each group has its own rate drawn from a gamma whose rate is unknown, so the groups teach each other where the rates lie.',
  factors: ['prior (gamma)', 'gamma-rate', 'poisson likelihood (gamma-poisson)'],
  family: 'conjugate',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.75, 1.25],
    ks: 0.1,
    reason:
      'the prior rate is skewed and EP carries one gamma for it; each rate is the matched gamma of a mixture over it',
  },
  knobs: dict({
    ...simulationKnobs(4),
    groups: int(2, 20, { default: 8, label: 'groups' }),
    shape: real(0.5, 8, { default: 2, step: 0.1, label: 'shape of each rate prior (known)' }),
    trueRate: real(0.1, 4, { default: 0.5, step: 0.05, label: 'true prior rate' }),
    hyperShape: real(0.5, 6, { default: 2, step: 0.1, label: 'shape of the prior-rate prior' }),
    hyperRate: real(0.2, 4, { default: 1, step: 0.1, label: 'rate of the prior-rate prior' }),
  }),
  problem: gammaLearnedProblem,
})
