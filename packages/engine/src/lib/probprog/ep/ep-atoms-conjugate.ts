/**
 * The conjugate setups of expectation propagation, bare: a Gaussian mean, a beta with Bernoulli flips, a gamma with Poisson
 * counts. The posterior of each is in closed form, so these are the sanity setups of the suite: EP is exact on them, up to
 * rounding, and they show the factor graph at its simplest (one latent, one observation factor).
 */

import { type DictSpace, dict, real } from '../../space/index.ts'
import type { Config } from '../../trace/index.ts'
import { configNumber } from '../inference/method.ts'
import * as d from '../model/dists/index.ts'
import { defineModel } from '../model/index.ts'
import { type AtomProblem, beliefDensity, defineAtom, type EpAtom } from './ep-atoms.ts'
import { normalPdf } from './ep-atoms-quad.ts'

const EXACT = {
  meanZ: 1e-6,
  sdRatio: [0.999999, 1.000001] as const,
  ks: 1e-4,
  reason: 'a conjugate model: EP is exact, so its belief equals the closed-form posterior up to rounding',
}
const sum = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0)

const GAUSSIAN_Y = [1.2, 0.4, 1.9]

function gaussianProblem(k: Config): AtomProblem {
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
    definition,
    data: { y: GAUSSIAN_Y },
    equations: [`mu ~ N(0, ${ps}²)`, `y_i ~ N(mu, 1²), i = 1..3`, `observe y = (${GAUSSIAN_Y.join(', ')})`],
    quantities: [{ name: 'mu', label: 'mu', support: 'real' }],
    exact: { mu: { mean, sd, pdf: (x) => normalPdf(x, mean, sd) } },
  }
}

/**
 * A Gaussian mean with Gaussian observations: the simplest factor graph, a prior and one likelihood factor.
 *
 * @index estimator EP atom: Gaussian mean (conjugate, exact).
 * @example
 * ```ts
 * import { gaussianAtom } from 'omai-engine/probprog/ep'
 * gaussianAtom.name // => 'gaussian'
 * ```
 */
export const gaussianAtom: EpAtom = defineAtom({
  name: 'gaussian',
  title: 'Gaussian',
  what: 'A Gaussian prior on a mean and three Gaussian observations: a prior factor and a linear-Gaussian likelihood factor, closed-form posterior.',
  factors: ['prior (gaussian)', 'gaussian likelihood (linear)'],
  family: 'conjugate',
  expected: 'runs',
  reference: 'exact',
  tolerance: EXACT,
  knobs: dict({ priorSd: real(0.3, 10, { default: 2, step: 0.1, label: 'prior sd' }) }),
  problem: gaussianProblem,
})

const FLIPS = [1, 1, 0, 1, 1, 0, 1]

function betaProblem(k: Config): AtomProblem {
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
    definition,
    data: { y: FLIPS },
    equations: [`p ~ Beta(${a}, ${b})`, 'y_i ~ Bernoulli(p), i = 1..7', `observe y = (${FLIPS.join(', ')})`],
    quantities: [{ name: 'p', label: 'p', support: 'unit' }],
    exact: { p: { mean, sd, pdf: beliefDensity('unit', mean, sd), range: [0, 1] } },
  }
}

/**
 * A beta prior and Bernoulli observations.
 *
 * @index estimator EP atom: beta and Bernoulli (conjugate, exact).
 * @example
 * ```ts
 * import { betaBernoulliAtom } from 'omai-engine/probprog/ep'
 * betaBernoulliAtom.family // => 'conjugate'
 * ```
 */
export const betaBernoulliAtom: EpAtom = defineAtom({
  name: 'beta-bernoulli',
  title: 'Beta and Bernoulli',
  what: 'A beta prior on a probability and seven Bernoulli observations: the beta-Bernoulli conjugate pair, a beta message from each observation.',
  factors: ['prior (beta)', 'bernoulli likelihood (beta-bernoulli)'],
  family: 'conjugate',
  expected: 'runs',
  reference: 'exact',
  tolerance: EXACT,
  knobs: dict({
    a: real(0.2, 20, { default: 2, step: 0.1, label: 'prior a' }),
    b: real(0.2, 20, { default: 2, step: 0.1, label: 'prior b' }),
  }),
  problem: betaProblem,
})

const COUNTS = [3, 5, 2, 4]

function gammaProblem(k: Config): AtomProblem {
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
    definition,
    data: { y: COUNTS },
    equations: [`rate ~ Gamma(${a}, ${b})`, 'y_i ~ Poisson(rate), i = 1..4', `observe y = (${COUNTS.join(', ')})`],
    quantities: [{ name: 'rate', label: 'rate', support: 'positive' }],
    exact: {
      rate: { mean, sd, pdf: beliefDensity('positive', mean, sd), range: [Math.max(0, mean - 6 * sd), mean + 8 * sd] },
    },
  }
}

/**
 * A gamma prior and Poisson observations.
 *
 * @index estimator EP atom: gamma and Poisson (conjugate, exact).
 * @example
 * ```ts
 * import { gammaPoissonAtom } from 'omai-engine/probprog/ep'
 * gammaPoissonAtom.reference // => 'exact'
 * ```
 */
export const gammaPoissonAtom: EpAtom = defineAtom({
  name: 'gamma-poisson',
  title: 'Gamma and Poisson',
  what: 'A gamma prior on a rate and four Poisson observations: the gamma-Poisson conjugate pair, a gamma message from each observation.',
  factors: ['prior (gamma)', 'poisson likelihood (gamma-poisson)'],
  family: 'conjugate',
  expected: 'runs',
  reference: 'exact',
  tolerance: EXACT,
  knobs: dict({
    shape: real(0.2, 20, { default: 2, step: 0.1, label: 'prior shape' }),
    rate: real(0.05, 5, { default: 1, step: 0.05, label: 'prior rate' }),
  }) as DictSpace,
  problem: gammaProblem,
})
