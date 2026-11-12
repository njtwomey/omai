/**
 * The conjugate setups of expectation propagation, bare and with their unknowns learned: a Gaussian mean, a beta with Bernoulli flips, a gamma with Poisson counts, and the same three with the numbers the basic atoms fix made unknown (the variance, the strength, the prior rate) and the data simulated from a stated truth. The posterior of each basic atom is in closed form, so these are the sanity setups of the suite: EP is exact on them, up to rounding, and they show the factor graph at its simplest.
 */

import { beta, foldIn, gamma, key, normal, poisson, uniform } from '../../../random/index.ts'
import { type DictSpace, dict, int, real } from '../../../space/index.ts'
import { add, arange, div, floor, item, toFlat } from '../../../tensor/index.ts'
import type { Config } from '../../../trace/index.ts'
import { configNumber } from '../../inference/method.ts'
import * as d from '../../model/dists/index.ts'
import { defineModel, ops } from '../../model/index.ts'
import {
  type ConditionalMarginal,
  hyperPosterior,
  lg,
  normalInverseGammaPosterior,
  normalPdf,
  sum,
} from '../../validation/index.ts'
import { defineAtom } from '../_define.ts'
import { recovery, tolerance } from '../_tolerance.ts'
import type { AtomParts, AtomRecoverySpec, EpAtom } from '../atom-types.ts'
import { beliefDensity } from '../ep-atoms.ts'

const EXACT = tolerance(
  'exact',
  'a conjugate model: EP is exact, so its belief equals the closed-form posterior up to rounding',
)

const GAUSSIAN_Y = [1.2, 0.4, 1.9]

function gaussianProblem(k: Config): AtomParts {
  const ps = configNumber(k, 'priorSd')
  const s = 1
  // region gaussian
  const definition = defineModel('gaussian', (m) => {
    const mu = m.latent('mu', d.normal(0, ps))
    m.observe('y', d.normal(mu, s), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  const prec = 1 / ps ** 2 + GAUSSIAN_Y.length / s ** 2
  const mean = sum(GAUSSIAN_Y) / s ** 2 / prec
  const sd = Math.sqrt(1 / prec)
  return {
    build: {
      definition,
      data: { y: GAUSSIAN_Y },
      quantities: [{ name: 'mu', label: 'mu', support: 'real' }],
    },
    reference: {
      exact: { mu: { mean, sd, pdf: (x) => normalPdf(x, mean, sd) } },
    },
    present: {
      equations: [`mu ~ N(0, ${ps}²)`, `y_i ~ N(mu, 1²), i = 1..3`, `observe y = (${GAUSSIAN_Y.join(', ')})`],
    },
  }
}

/**
 * A Gaussian mean with Gaussian observations: the simplest factor graph, a prior and one likelihood factor.
 */
const gaussianAtom: EpAtom = defineAtom({
  name: 'gaussian',
  title: 'Gaussian',
  what: 'A Gaussian prior on a mean and three Gaussian observations: a prior factor and a linear-Gaussian likelihood factor, closed-form posterior.',
  factors: ['prior (gaussian)', 'gaussian likelihood (linear)'],
  family: 'conjugate',
  reference: 'exact',
  tolerance: EXACT,
  evidence: { nats: 1e-5, reason: 'conjugate: exact' },
  knobs: dict({ priorSd: real(0.3, 10, { default: 2, step: 0.1, label: 'prior sd' }) }),
  parts: gaussianProblem,
})

const FLIPS = [1, 1, 0, 1, 1, 0, 1]

function betaProblem(k: Config): AtomParts {
  const a = configNumber(k, 'a')
  const b = configNumber(k, 'b')
  // region beta-bernoulli
  const definition = defineModel('beta-bernoulli', (m) => {
    const p = m.latent('p', d.beta(a, b))
    const observed = m.data('y', { dims: ['i'], support: 'binary', role: 'response' })
    m.observe('y', d.bernoulli({ probs: p }), observed)
  })
  // endregion
  const heads = sum(FLIPS)
  const A = a + heads
  const B = b + FLIPS.length - heads
  const mean = A / (A + B)
  const sd = Math.sqrt((A * B) / ((A + B) ** 2 * (A + B + 1)))
  return {
    build: {
      definition,
      data: { y: FLIPS },
      quantities: [{ name: 'p', label: 'p', support: 'unit' }],
    },
    reference: {
      exact: { p: { mean, sd, pdf: beliefDensity('unit', mean, sd), range: [0, 1] } },
    },
    present: {
      equations: [`p ~ Beta(${a}, ${b})`, 'y_i ~ Bernoulli(p), i = 1..7', `observe y = (${FLIPS.join(', ')})`],
    },
  }
}

/**
 * A beta prior and Bernoulli observations.
 */
const betaBernoulliAtom: EpAtom = defineAtom({
  name: 'beta-bernoulli',
  title: 'Beta and Bernoulli',
  what: 'A beta prior on a probability and seven Bernoulli observations: the beta-Bernoulli conjugate pair, a beta message from each observation.',
  factors: ['prior (beta)', 'bernoulli likelihood (beta-bernoulli)'],
  family: 'conjugate',
  reference: 'exact',
  tolerance: EXACT,
  evidence: { nats: 1e-5, reason: 'conjugate: exact' },
  knobs: dict({
    a: real(0.2, 20, { default: 2, step: 0.1, label: 'prior a' }),
    b: real(0.2, 20, { default: 2, step: 0.1, label: 'prior b' }),
  }),
  parts: betaProblem,
})

const COUNTS = [3, 5, 2, 4]

function gammaProblem(k: Config): AtomParts {
  const a = configNumber(k, 'shape')
  const b = configNumber(k, 'rate')
  // region gamma-poisson
  const definition = defineModel('gamma-poisson', (m) => {
    const rate = m.latent('rate', d.gamma(a, b))
    const observed = m.data('y', { dims: ['i'], support: 'count', role: 'response' })
    m.observe('y', d.poisson(rate), observed)
  })
  // endregion
  const A = a + sum(COUNTS)
  const B = b + COUNTS.length
  const mean = A / B
  const sd = Math.sqrt(A) / B
  return {
    build: {
      definition,
      data: { y: COUNTS },
      quantities: [{ name: 'rate', label: 'rate', support: 'positive' }],
    },
    reference: {
      exact: {
        rate: {
          mean,
          sd,
          pdf: beliefDensity('positive', mean, sd),
          range: [Math.max(0, mean - 6 * sd), mean + 8 * sd],
        },
      },
    },
    present: {
      equations: [`rate ~ Gamma(${a}, ${b})`, 'y_i ~ Poisson(rate), i = 1..4', `observe y = (${COUNTS.join(', ')})`],
    },
  }
}

/**
 * A gamma prior and Poisson observations.
 */
const gammaPoissonAtom: EpAtom = defineAtom({
  name: 'gamma-poisson',
  title: 'Gamma and Poisson',
  what: 'A gamma prior on a rate and four Poisson observations: the gamma-Poisson conjugate pair, a gamma message from each observation.',
  factors: ['prior (gamma)', 'poisson likelihood (gamma-poisson)'],
  family: 'conjugate',
  reference: 'exact',
  tolerance: EXACT,
  evidence: { nats: 1e-5, reason: 'conjugate: exact' },
  knobs: dict({
    shape: real(0.2, 20, { default: 2, step: 0.1, label: 'prior shape' }),
    rate: real(0.05, 5, { default: 1, step: 0.05, label: 'prior rate' }),
  }) as DictSpace,
  parts: gammaProblem,
})

/** The knobs every extended atom has: the seed of the simulation and the number of observations. */
export const simulationKnobs = (n: number): Record<string, ReturnType<typeof int>> => ({
  seed: int(1, 9999, { default: 1, label: 'seed (the simulated data)' }),
  n: int(3, 200, { default: n, label: 'observations' }),
})

const GAUSSIAN_RECOVERY = recovery(
  'calibrated',
  'the posterior is exact (the model is conjugate), so over seeds drawn from the truth the intervals hold it close to nine times in ten; 40 seeds leave a binomial spread of about 0.05, and the prior pulls a little',
)

function gaussianLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueMean, trueSd] = [configNumber(k, 'trueMean'), configNumber(k, 'trueSd')]
  const [m0, kappa, a, b] = [
    configNumber(k, 'priorMean'),
    configNumber(k, 'kappa'),
    configNumber(k, 'shape'),
    configNumber(k, 'scale'),
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
    build: {
      definition,
      data: { y },
      quantities: [
        { name: 'mu', label: 'mu (the mean)', support: 'real', truth: trueMean },
        { name: 'sigma2', label: 'sigma2 (the variance)', support: 'positive', truth: trueSd ** 2 },
      ],
    },
    reference: {
      exact: { mu, sigma2 },
      exactJoint: joint,
    },
    present: {
      equations: [
        `truth: mean ${trueMean}, sd ${trueSd}; ${n} observations drawn with seed ${seed}`,
        `sigma2 ~ InverseGamma(${a}, ${b})   (the unknown variance; prior mean ${sigmaMean(a, b)})`,
        `mu | sigma2 ~ N(${m0}, sigma2 / ${kappa})   (the unknown mean, worth ${kappa} observations)`,
        `y_i ~ N(mu, sigma2), i = 1..${n}`,
      ],
      pair: ['mu', 'sigma2'],
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
    },
  }
}

const sigmaMean = (a: number, b: number): string => (a > 1 ? String(Number((b / (a - 1)).toPrecision(3))) : 'undefined')

/**
 * The Gaussian with both its mean and its variance unknown, learned from data simulated from a stated truth: the
 * normal-inverse-gamma pair under EP (one normal-gamma variable and a factor per observation), exact, with the recovery of
 * the truth checked over many seeds.
 */
const gaussianLearnedAtom: EpAtom = defineAtom({
  name: 'gaussian-learned',
  title: 'Gaussian: parameters learned',
  extends: 'gaussian',
  recovery: GAUSSIAN_RECOVERY,
  what: 'The Gaussian of the basic atom with its mean and its variance both unknown (a normal-inverse-gamma prior), observed through data simulated from a stated mean and sd: does EP recover them, and how often does its 90% interval hold the truth?',
  factors: ['prior (normal-gamma)', 'gaussian-normal-gamma (observed value)'],
  family: 'conjugate',
  reference: 'exact',
  tolerance: tolerance(
    'exact-heavy-tail',
    'the posterior is exactly normal-inverse-gamma and EP carries it exactly; the curves compare the Gaussian and gamma shapes the page draws with the exact Student t and inverse gamma, so the distance is small, not zero',
  ),
  evidence: { nats: 1e-5, reason: 'a normal-gamma pair: exact' },
  knobs: dict({
    ...simulationKnobs(12),
    trueMean: real(-3, 3, { default: 1.5, step: 0.1, label: 'true mean' }),
    trueSd: real(0.2, 3, { default: 0.8, step: 0.05, label: 'true sd' }),
    priorMean: real(-2, 2, { default: 0, step: 0.1, label: 'prior mean m' }),
    kappa: real(0.05, 5, { default: 0.2, step: 0.05, label: 'kappa (prior observations for the mean)' }),
    shape: real(1.5, 6, { default: 3, step: 0.1, label: 'shape a of the inverse gamma' }),
    scale: real(0.2, 5, { default: 1.5, step: 0.1, label: 'scale b of the inverse gamma' }),
  }),
  parts: gaussianLearnedProblem,
})

/** The group of each of `groups * per` observations, group by group. */
const groupIndex = (groups: number, per: number): number[] => toFlat(floor(div(arange(groups * per), per)))

const FIGURE_GROUPS = (groups: number): number[] => toFlat(add(arange(groups), 1))

function betaLearnedProblem(k: Config): AtomParts {
  const [seed, groups, per] = [configNumber(k, 'seed'), configNumber(k, 'groups'), configNumber(k, 'n')]
  const trueStrength = configNumber(k, 'trueStrength')
  const [a, b] = [configNumber(k, 'shape'), configNumber(k, 'rate')]
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
    const flips = m.data('y', { dims: [r], support: 'binary' })
    m.observe('y', d.bernoulli({ probs: prob.at(which) }), flips)
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
    build: {
      definition,
      data: { y, group },
      bindOptions: { sizes: { group: groups } },
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
    },
    reference: {
      exact: { strength: post.hyper, ...Object.fromEntries(exactP.map((e, j) => [`p[${j}]`, e])) },
    },
    present: {
      equations: [
        `truth: strength ${trueStrength}, so p_j ~ Beta(${trueStrength}, ${trueStrength}); ${groups} groups of ${per} flips, seed ${seed}`,
        `strength ~ Gamma(${a}, ${b})   (unknown: how far from one half the probabilities lie; prior mean ${Number((a / b).toPrecision(3))})`,
        `p_j ~ Beta(strength, strength), j = 1..${groups}`,
        `y_ji ~ Bernoulli(p_j), i = 1..${per}`,
      ],
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
    },
  }
}

/**
 * A population of Bernoulli probabilities whose symmetric beta prior has an unknown, gamma-distributed strength, learned from
 * flips simulated from a stated truth: does EP recover the strength (how far the probabilities spread from one half) and each
 * group's probability?
 */
const betaBernoulliLearnedAtom: EpAtom = defineAtom({
  name: 'beta-bernoulli-learned',
  title: 'Beta and Bernoulli: parameters learned',
  extends: 'beta-bernoulli',
  recovery: recovery(
    'standard',
    'the posterior of the strength is skewed and EP carries a gamma for it, and each probability is a beta matched to a mixture over the strength: the intervals hold the truth a little less than nine times in ten',
  ),
  what: 'The beta and Bernoulli of the basic atom for a population of groups: each group has its own probability drawn from a symmetric beta whose strength is unknown, so the groups teach each other how spread out the probabilities are.',
  factors: [
    'prior (gamma)',
    'dirichlet-concentration (a beta with a random concentration)',
    'bernoulli likelihood (beta-bernoulli)',
  ],
  family: 'conjugate',
  reference: 'exact',
  tolerance: tolerance(
    'skewed',
    'the strength is skewed and EP carries one gamma for it; each probability is the matched beta of a mixture over it',
  ),
  knobs: dict({
    ...simulationKnobs(10),
    groups: int(2, 20, { default: 8, label: 'groups' }),
    trueStrength: real(0.3, 10, { default: 3, step: 0.1, label: 'true strength' }),
    shape: real(0.5, 8, { default: 2, step: 0.1, label: 'shape of the strength prior' }),
    rate: real(0.1, 4, { default: 0.4, step: 0.1, label: 'rate of the strength prior' }),
  }),
  parts: betaLearnedProblem,
})

function gammaLearnedProblem(k: Config): AtomParts {
  const [seed, groups, per] = [configNumber(k, 'seed'), configNumber(k, 'groups'), configNumber(k, 'n')]
  const [a, trueRate] = [configNumber(k, 'shape'), configNumber(k, 'trueRate')]
  const [c, dd] = [configNumber(k, 'hyperShape'), configNumber(k, 'hyperRate')]
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
    build: {
      definition,
      data: { y, group },
      bindOptions: { sizes: { group: groups } },
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
    },
    reference: {
      exact: { priorRate: post.hyper, ...Object.fromEntries(exactRate.map((e, j) => [`rate[${j}]`, e])) },
    },
    present: {
      equations: [
        `truth: priorRate ${trueRate}, so rate_j ~ Gamma(${a}, ${trueRate}); ${groups} groups of ${per} counts, seed ${seed}`,
        `priorRate ~ Gamma(${c}, ${dd})   (unknown: the typical size of a rate is ${a} / priorRate)`,
        `rate_j ~ Gamma(${a}, priorRate), j = 1..${groups}`,
        `y_ji ~ Poisson(rate_j), i = 1..${per}`,
      ],
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
    },
  }
}

/**
 * A population of Poisson rates drawn from a gamma whose rate is unknown (gamma-distributed), learned from counts simulated
 * from a stated truth: does EP recover the prior rate and each group's rate?
 */
const gammaPoissonLearnedAtom: EpAtom = defineAtom({
  name: 'gamma-poisson-learned',
  title: 'Gamma and Poisson: parameters learned',
  extends: 'gamma-poisson',
  recovery: recovery(
    'standard',
    'the posterior of the prior rate is skewed and EP carries a gamma for it; each rate is a gamma matched to a mixture over it',
  ),
  what: 'The gamma and Poisson of the basic atom for a population of groups: each group has its own rate drawn from a gamma whose rate is unknown, so the groups teach each other where the rates lie.',
  factors: ['prior (gamma)', 'gamma-rate', 'poisson likelihood (gamma-poisson)'],
  family: 'conjugate',
  reference: 'exact',
  tolerance: tolerance(
    'skewed',
    'the prior rate is skewed and EP carries one gamma for it; each rate is the matched gamma of a mixture over it',
  ),
  knobs: dict({
    ...simulationKnobs(4),
    groups: int(2, 20, { default: 8, label: 'groups' }),
    shape: real(0.5, 8, { default: 2, step: 0.1, label: 'shape of each rate prior (known)' }),
    trueRate: real(0.1, 4, { default: 0.5, step: 0.05, label: 'true prior rate' }),
    hyperShape: real(0.5, 6, { default: 2, step: 0.1, label: 'shape of the prior-rate prior' }),
    hyperRate: real(0.2, 4, { default: 1, step: 0.1, label: 'rate of the prior-rate prior' }),
  }),
  parts: gammaLearnedProblem,
})

/** The atoms of the `conjugate` file, basic and learned twins together. */
export const conjugateAtoms: readonly EpAtom[] = [
  gaussianAtom,
  betaBernoulliAtom,
  gammaPoissonAtom,
  gaussianLearnedAtom,
  betaBernoulliLearnedAtom,
  gammaPoissonLearnedAtom,
]
