/**
 * The conjugate atoms of expectation propagation: a length measured with a noisy instrument (a Gaussian mean), a coin's
 * bias (beta and Bernoulli), calls per hour (gamma and Poisson), potholes on roads of different lengths (a Poisson with an
 * exposure) and germinating seeds in batches (a binomial). Each has a closed-form posterior, so the check is against the
 * exact answer and EP is expected to be exact up to rounding: these are the sanity atoms of the suite.
 */

import * as d from '../probprog/model/dists/index.ts'
import { defineModel, ops } from '../probprog/model/index.ts'
import { type DictSpace, dict, int, real } from '../space/index.ts'
import type { Config } from '../trace/index.ts'
import { type AtomProblem, beliefDensity, defineAtom, type EpAtom } from './ep-atoms.ts'
import { atomDraws } from './ep-atoms-draws.ts'
import { exampleNumber } from './example-method.ts'

const EXACT = {
  meanZ: 1e-6,
  sdRatio: [0.999999, 1.000001] as const,
  ks: 1e-4,
  reason: 'a conjugate model: EP is exact, so the beliefs equal the closed-form posterior up to rounding',
}

const seedKnob = int(0, 9999, { default: 1, label: 'seed' })
const sum = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0)

/** The knobs of the length atom. */
const lengthKnobs: DictSpace = dict({
  measurements: int(1, 40, { default: 6, label: 'readings' }),
  trueLength: real(5, 20, { default: 12.3, step: 0.1, label: 'true length (cm)' }),
  noiseSd: real(0.1, 3, { default: 0.8, step: 0.05, label: 'instrument noise (cm)' }),
  priorMean: real(0, 20, { default: 10, step: 0.5, label: 'prior guess (cm)' }),
  priorSd: real(0.5, 10, { default: 3, step: 0.5, label: 'prior uncertainty (cm)' }),
  dataSeed: seedKnob,
})

function lengthProblem(k: Config): AtomProblem {
  const n = exampleNumber(k, 'measurements')
  const truth = exampleNumber(k, 'trueLength')
  const s = exampleNumber(k, 'noiseSd')
  const pm = exampleNumber(k, 'priorMean')
  const ps = exampleNumber(k, 'priorSd')
  const z = atomDraws(exampleNumber(k, 'dataSeed')).normals(n)
  const y = z.map((v) => truth + s * v)
  const definition = defineModel('rod-length', (m) => {
    const reading = m.data('y', { dims: ['measurement'], role: 'response', doc: 'each reading of the rod' })
    const mu = m.latent('mu', d.normal(pm, ps), { doc: 'the true length' })
    m.observe('y', d.normal(mu, s), reading)
  })
  const precision = 1 / ps ** 2 + n / s ** 2
  const mean = (pm / ps ** 2 + sum(y) / s ** 2) / precision
  const sd = Math.sqrt(1 / precision)
  const at = Array.from({ length: n }, (_, i) => i + 1)
  return {
    definition,
    data: { y },
    quantities: [{ name: 'mu', label: 'length (cm)', support: 'real', truth }],
    exact: { mu: { mean, sd, pdf: beliefDensity('real', mean, sd) } },
    figure: {
      caption: 'Each dot is one reading of the same rod; the line is the length the belief settles on.',
      xLabel: 'reading number',
      yLabel: 'length (cm)',
      xLimits: [0, n + 1],
      points: [{ name: 'readings', x: at, y }],
      answer: (means) => [{ name: 'length', x: [0, n + 1], y: [means['mu'] ?? Number.NaN, means['mu'] ?? Number.NaN] }],
    },
  }
}

/**
 * Six noisy readings of the length of a rod: a Gaussian mean with a known noise level and a Gaussian prior, the sanity
 * atom of the suite (the posterior is Gaussian and EP is exact).
 *
 * @index estimator EP atom: readings of a length (Gaussian mean, conjugate, exact).
 * @example
 * ```ts
 * import { gaussianMeanAtom } from 'omai-engine/bayesian-examples'
 * [gaussianMeanAtom.name, gaussianMeanAtom.expected] // => ['gaussian-mean', 'runs']
 * ```
 */
export const gaussianMeanAtom: EpAtom = defineAtom({
  name: 'gaussian-mean',
  title: 'Six readings of a length',
  story:
    'A rod is measured several times with an instrument whose noise is known; the belief about its true length starts at a rough prior guess and tightens with every reading.',
  factors: ['prior (gaussian)', 'gaussian likelihood (linear)'],
  family: 'conjugate',
  expected: 'runs',
  reference: 'exact',
  tolerance: EXACT,
  knobs: lengthKnobs,
  problem: lengthProblem,
})

/** The knobs of the coin atom. */
const coinKnobs: DictSpace = dict({
  flips: int(1, 60, { default: 12, label: 'flips' }),
  trueP: real(0.02, 0.98, { default: 0.65, step: 0.01, label: 'true chance of heads' }),
  priorA: real(0.2, 20, { default: 2, step: 0.1, label: 'prior heads (a)' }),
  priorB: real(0.2, 20, { default: 2, step: 0.1, label: 'prior tails (b)' }),
  dataSeed: seedKnob,
})

function coinProblem(k: Config): AtomProblem {
  const n = exampleNumber(k, 'flips')
  const truth = exampleNumber(k, 'trueP')
  const a = exampleNumber(k, 'priorA')
  const b = exampleNumber(k, 'priorB')
  const y = atomDraws(exampleNumber(k, 'dataSeed')).flips(n, truth)
  const heads = sum(y)
  const definition = defineModel('coin-bias', (m) => {
    const flip = m.data('y', { dims: ['flip'], support: 'binary', role: 'response', doc: '1 for heads' })
    const p = m.latent('p', d.beta(a, b), { doc: 'the chance of heads' })
    m.observe('y', d.bernoulli({ probs: p }), flip)
  })
  const A = a + heads
  const B = b + n - heads
  const mean = A / (A + B)
  const sd = Math.sqrt((A * B) / ((A + B) ** 2 * (A + B + 1)))
  const at = Array.from({ length: n }, (_, i) => i + 1)
  let running = 0
  return {
    definition,
    data: { y },
    quantities: [{ name: 'p', label: 'chance of heads', support: 'unit', truth }],
    exact: { p: { mean, sd, pdf: beliefDensity('unit', mean, sd), range: [0, 1] } },
    figure: {
      caption:
        'Each dot is a flip (heads on top); the curve is the share of heads so far and the line the belief about the chance.',
      xLabel: 'flip number',
      yLabel: 'heads (1) or tails (0)',
      xLimits: [0, n + 1],
      yLimits: [-0.08, 1.08],
      points: [
        {
          name: 'flips',
          x: at,
          y,
          group: y,
          groupNames: { 0: 'tails', 1: 'heads' },
        },
      ],
      curves: [
        {
          name: 'share of heads so far',
          x: at,
          y: y.map((v, i) => {
            running += v
            return running / (i + 1)
          }),
        },
      ],
      answer: (means) => [
        { name: 'chance of heads', x: [0, n + 1], y: [means['p'] ?? Number.NaN, means['p'] ?? Number.NaN] },
      ],
    },
  }
}

/**
 * A coin flipped a dozen times: a beta prior on its chance of heads and Bernoulli flips (conjugate, the posterior is a
 * beta and EP is exact).
 *
 * @index estimator EP atom: a coin's bias (beta and Bernoulli, conjugate, exact).
 * @example
 * ```ts
 * import { coinBiasAtom } from 'omai-engine/bayesian-examples'
 * coinBiasAtom.factors.length // => 2
 * ```
 */
export const coinBiasAtom: EpAtom = defineAtom({
  name: 'coin-bias',
  title: 'A coin flipped a dozen times',
  story: 'Is the coin fair? The belief about its chance of heads starts at a mild prior and narrows with every flip.',
  factors: ['prior (beta)', 'bernoulli likelihood (beta-bernoulli)'],
  family: 'conjugate',
  expected: 'runs',
  reference: 'exact',
  tolerance: EXACT,
  knobs: coinKnobs,
  problem: coinProblem,
})

/** The knobs of the help-desk atom. */
const callsKnobs: DictSpace = dict({
  hours: int(1, 48, { default: 12, label: 'hours observed' }),
  trueRate: real(0.5, 20, { default: 4.2, step: 0.1, label: 'true calls per hour' }),
  priorShape: real(0.2, 20, { default: 2, step: 0.1, label: 'prior shape' }),
  priorRate: real(0.05, 5, { default: 0.5, step: 0.05, label: 'prior rate' }),
  dataSeed: seedKnob,
})

function callsProblem(k: Config): AtomProblem {
  const n = exampleNumber(k, 'hours')
  const truth = exampleNumber(k, 'trueRate')
  const a = exampleNumber(k, 'priorShape')
  const b = exampleNumber(k, 'priorRate')
  const y = atomDraws(exampleNumber(k, 'dataSeed')).counts(n, truth)
  const definition = defineModel('calls-per-hour', (m) => {
    const calls = m.data('y', { dims: ['hour'], support: 'count', role: 'response', doc: 'calls in each hour' })
    const rate = m.latent('rate', d.gamma(a, b), { doc: 'calls per hour' })
    m.observe('y', d.poisson(rate), calls)
  })
  const A = a + sum(y)
  const B = b + n
  const mean = A / B
  const sd = Math.sqrt(A) / B
  const at = Array.from({ length: n }, (_, i) => i + 1)
  return {
    definition,
    data: { y },
    quantities: [{ name: 'rate', label: 'calls per hour', support: 'positive', truth }],
    exact: {
      rate: { mean, sd, pdf: beliefDensity('positive', mean, sd), range: [Math.max(0, mean - 6 * sd), mean + 8 * sd] },
    },
    figure: {
      caption:
        'Each dot is the number of calls a help desk took in one hour; the line is the rate the belief settles on.',
      xLabel: 'hour',
      yLabel: 'calls in the hour',
      xLimits: [0, n + 1],
      yLimits: [-0.5, Math.max(...y, 1) + 1.5],
      points: [{ name: 'hourly counts', x: at, y }],
      answer: (means) => [
        { name: 'rate', x: [0, n + 1], y: [means['rate'] ?? Number.NaN, means['rate'] ?? Number.NaN] },
      ],
    },
  }
}

/**
 * Calls per hour at a help desk: a gamma prior on the rate and Poisson counts (conjugate, exact under EP).
 *
 * @index estimator EP atom: calls per hour (gamma and Poisson, conjugate, exact).
 * @example
 * ```ts
 * import { callsPerHourAtom } from 'omai-engine/bayesian-examples'
 * callsPerHourAtom.name // => 'calls-per-hour'
 * ```
 */
export const callsPerHourAtom: EpAtom = defineAtom({
  name: 'calls-per-hour',
  title: 'Calls per hour at a help desk',
  story: 'A dozen hourly counts of calls; how many per hour should the desk expect?',
  factors: ['prior (gamma)', 'poisson likelihood (gamma-poisson)'],
  family: 'conjugate',
  expected: 'runs',
  reference: 'exact',
  tolerance: EXACT,
  knobs: callsKnobs,
  problem: callsProblem,
})

/** The knobs of the pothole atom. */
const potholeKnobs: DictSpace = dict({
  sections: int(1, 30, { default: 8, label: 'road sections' }),
  trueRate: real(0.1, 6, { default: 1.4, step: 0.1, label: 'true potholes per km' }),
  priorShape: real(0.2, 20, { default: 2, step: 0.1, label: 'prior shape' }),
  priorRate: real(0.05, 5, { default: 1, step: 0.05, label: 'prior rate' }),
  dataSeed: seedKnob,
})

function potholeProblem(k: Config): AtomProblem {
  const n = exampleNumber(k, 'sections')
  const truth = exampleNumber(k, 'trueRate')
  const a = exampleNumber(k, 'priorShape')
  const b = exampleNumber(k, 'priorRate')
  const draws = atomDraws(exampleNumber(k, 'dataSeed'))
  const km = draws.uniforms(n, 0.5, 6, 1).map((v) => Math.round(v * 10) / 10)
  const y = draws.counts(
    n,
    km.map((v) => truth * v),
  )
  const definition = defineModel('potholes-per-km', (m) => {
    const length = m.data('km', { dims: ['section'], role: 'covariate', doc: 'the length of the section' })
    const count = m.data('y', { dims: ['section'], support: 'count', role: 'response', doc: 'potholes found' })
    const rate = m.latent('rate', d.gamma(a, b), { doc: 'potholes per km' })
    m.observe('y', d.poisson(ops.mul(length, rate)), count)
  })
  const A = a + sum(y)
  const B = b + sum(km)
  const mean = A / B
  const sd = Math.sqrt(A) / B
  const far = Math.max(...km, 1) * 1.1
  return {
    definition,
    data: { km, y },
    quantities: [{ name: 'rate', label: 'potholes per km', support: 'positive', truth }],
    exact: {
      rate: { mean, sd, pdf: beliefDensity('positive', mean, sd), range: [Math.max(0, mean - 6 * sd), mean + 8 * sd] },
    },
    figure: {
      caption: 'Each dot is a stretch of road: its length against the potholes found; the line is the rate per km.',
      xLabel: 'length of the section (km)',
      yLabel: 'potholes found',
      xLimits: [0, far],
      yLimits: [-0.5, Math.max(...y, 1) + 1.5],
      points: [{ name: 'road sections', x: km, y }],
      answer: (means) => [{ name: 'rate per km', x: [0, far], y: [0, (means['rate'] ?? Number.NaN) * far] }],
    },
  }
}

/**
 * Potholes counted on road sections of different lengths: a Poisson whose rate is a gamma-distributed rate per km times the
 * known length (an exposure).
 *
 * @index estimator EP atom: potholes per km (gamma and Poisson with an exposure, conjugate, exact).
 * @example
 * ```ts
 * import { potholesAtom } from 'omai-engine/bayesian-examples'
 * potholesAtom.family // => 'conjugate'
 * ```
 */
export const potholesAtom: EpAtom = defineAtom({
  name: 'potholes-per-km',
  title: 'Potholes on roads of different lengths',
  story: 'Counts of potholes on sections of road of known length: the rate per km is shared, the exposure differs.',
  factors: ['prior (gamma)', 'poisson with an exposure'],
  family: 'conjugate',
  expected: 'runs',
  reference: 'exact',
  tolerance: EXACT,
  knobs: potholeKnobs,
  problem: potholeProblem,
})

/** The knobs of the germination atom. */
const seedsKnobs: DictSpace = dict({
  batches: int(1, 20, { default: 5, label: 'batches' }),
  seedsPerBatch: int(2, 50, { default: 20, label: 'seeds per batch' }),
  trueP: real(0.05, 0.98, { default: 0.7, step: 0.01, label: 'true germination rate' }),
  priorA: real(0.2, 20, { default: 1, step: 0.1, label: 'prior a' }),
  priorB: real(0.2, 20, { default: 1, step: 0.1, label: 'prior b' }),
  dataSeed: seedKnob,
})

function seedsProblem(k: Config): AtomProblem {
  const n = exampleNumber(k, 'batches')
  const m0 = exampleNumber(k, 'seedsPerBatch')
  const truth = exampleNumber(k, 'trueP')
  const a = exampleNumber(k, 'priorA')
  const b = exampleNumber(k, 'priorB')
  const draws = atomDraws(exampleNumber(k, 'dataSeed'))
  const y = Array.from({ length: n }, (_, j) => sum(draws.flips(m0, truth, j + 1)))
  const definition = defineModel('germination', (m) => {
    const sprouted = m.data('y', { dims: ['batch'], support: 'count', role: 'response', doc: 'seeds that sprouted' })
    const p = m.latent('p', d.beta(a, b), { doc: 'the germination rate' })
    m.observe('y', d.binomial(m0, p), sprouted)
  })
  const A = a + sum(y)
  const B = b + n * m0 - sum(y)
  const mean = A / (A + B)
  const sd = Math.sqrt((A * B) / ((A + B) ** 2 * (A + B + 1)))
  const at = Array.from({ length: n }, (_, i) => i + 1)
  return {
    definition,
    data: { y },
    quantities: [{ name: 'p', label: 'germination rate', support: 'unit', truth }],
    exact: { p: { mean, sd, pdf: beliefDensity('unit', mean, sd), range: [0, 1] } },
    figure: {
      caption: `Each dot is the share of a batch of ${m0} seeds that sprouted; the line is the rate the belief settles on.`,
      xLabel: 'batch',
      yLabel: 'share that sprouted',
      xLimits: [0, n + 1],
      yLimits: [0, 1],
      points: [{ name: 'batches', x: at, y: y.map((v) => v / m0) }],
      answer: (means) => [{ name: 'rate', x: [0, n + 1], y: [means['p'] ?? Number.NaN, means['p'] ?? Number.NaN] }],
    },
  }
}

/**
 * Seeds sown in batches of the same size, counting the ones that sprout: a beta prior and a binomial per batch.
 *
 * @index estimator EP atom: germinating seeds in batches (beta and binomial, conjugate, exact).
 * @example
 * ```ts
 * import { germinationAtom } from 'omai-engine/bayesian-examples'
 * germinationAtom.reference // => 'exact'
 * ```
 */
export const germinationAtom: EpAtom = defineAtom({
  name: 'germination',
  title: 'Seeds germinating in batches',
  story: 'Five batches of twenty seeds, counting those that sprout: how likely is a seed to germinate?',
  factors: ['prior (beta)', 'binomial likelihood (beta-binomial)'],
  family: 'conjugate',
  expected: 'runs',
  reference: 'exact',
  tolerance: EXACT,
  knobs: seedsKnobs,
  problem: seedsProblem,
})
