/**
 * The conjugate benchmarks, whose posterior and evidence are closed forms: a beta probability with Bernoulli flips, a
 * normal mean with normal observations, and a gamma rate with Poisson counts. They are the models every exact method
 * must solve (conjugate, Gibbs on the conditionals, the Gaussian method) and every sampler and approximation is checked
 * against at no cost; the truth comes from the closed forms of `probprog/validation/exact.ts`. Data are simulated from
 * fixed generating parameters with the library's random keys.
 */

import {
  bernoulli as drawBernoulli,
  normal as drawNormal,
  poisson as drawPoisson,
  foldIn,
  type Key,
} from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel } from '../model/index.ts'
import {
  betaBernoulli as betaBernoulliExact,
  gammaPoisson as gammaPoissonExact,
  normalNormal as normalNormalExact,
} from '../validation/index.ts'
import { type BenchmarkData, type BenchmarkModel, defineBenchmark } from './benchmark.ts'
import { exactPlan, gibbsPlan, importancePlan, laplacePlan, nutsPlan, sviPlan, temperedPlan } from './plans.ts'

const BB_PRIOR = { a: 2, b: 2 }

/**
 * Beta-Bernoulli: a coin of unknown bias `p ~ Beta(2, 2)` and `n` flips (20 at the canonical size, generated at `p = 0.3`).
 * The posterior is `Beta(2 + k, 2 + n - k)` and the evidence `B(2 + k, 2 + n - k) / B(2, 2)`. A one-dimensional latent on the
 * unit interval: the Laplace fit on the logit scale and a mean-field normal guide are visibly off in the skew, while the
 * exact, Gibbs, NUTS and importance methods agree.
 *
 * @concept posterior-validation
 * @index dataset Beta-Bernoulli coin: exact Beta posterior and evidence; the simplest conjugate benchmark.
 * @example
 * ```ts
 * import { betaBernoulli } from 'omai-engine/probprog/models'
 * [betaBernoulli.name, betaBernoulli.bound().sizes, betaBernoulli.truth().kind] // => ['beta-bernoulli', { flip: 20 }, 'exact']
 * ```
 */
export const betaBernoulli: BenchmarkModel = defineBenchmark({
  name: 'beta-bernoulli',
  title: 'Beta-Bernoulli coin',
  description:
    'A coin with a Beta(2, 2) prior on its bias and twenty flips: the conjugate model with a closed-form Beta posterior and evidence.',
  tags: ['conjugate', 'exact', 'discrete-data', 'one-dimensional', 'unit-interval'],
  references: ['Gelman et al., Bayesian Data Analysis, 3rd ed., section 2.4 (the beta-binomial model)'],
  definition: defineModel('beta-bernoulli', (m) => {
    const y = m.data('y', { dims: ['flip'], support: 'binary', role: 'response' })
    const p = m.latent('p', d.beta(BB_PRIOR.a, BB_PRIOR.b), { doc: 'the probability of heads' })
    m.observe('y', d.bernoulli({ probs: p }), y)
  }),
  generate: (key: Key, size = 20): BenchmarkData => {
    const p = 0.3
    const y = Array.from(toFlat(drawBernoulli(foldIn(key, 0), p, [size])))
    return { data: { y }, generating: { p } }
  },
  truth: (data) => {
    const exact = betaBernoulliExact({ ...BB_PRIOR, y: data.data['y'] as number[] })
    const [oracle] = exact.oracles('p')
    return {
      kind: 'exact',
      source: 'the Beta(a + k, b + n - k) posterior and the beta function evidence (probprog/validation/exact.ts)',
      logEvidence: exact.logEvidence,
      marginals: { p: oracle as NonNullable<typeof oracle> },
    }
  },
  plans: (scale) => [
    exactPlan('conjugate'),
    gibbsPlan(scale),
    nutsPlan(scale),
    importancePlan(scale),
    temperedPlan(scale),
    laplacePlan(scale, { note: 'a normal at the mode on the logit scale: misses the skew of a Beta posterior' }),
    sviPlan(scale, {
      note: 'a mean-field normal on the logit scale: close in the mean, symmetric where the posterior is skewed',
    }),
  ],
})

const NN_PRIOR = { priorMean: 0, priorSd: 2, sigma: 1.5 }

/**
 * Normal-normal: the mean `mu ~ N(0, 2²)` of normal observations with known sd 1.5 (twelve at the canonical size,
 * generated at `mu = 0.8`). The posterior is normal in closed form, and the model is both conjugate and linear-Gaussian, so
 * the conjugate and Gaussian methods are exact, Laplace and a mean-field normal guide are exact too (the family is
 * right), and everything else must agree within its Monte Carlo error.
 *
 * @concept posterior-validation
 * @index dataset Normal-normal mean with known noise: exact normal posterior and evidence.
 * @example
 * ```ts
 * import { normalNormal } from 'omai-engine/probprog/models'
 * normalNormal.truth().marginals['mu']?.sd.toFixed(3) // => '0.403'
 * ```
 */
export const normalNormal: BenchmarkModel = defineBenchmark({
  name: 'normal-normal',
  title: 'Normal-normal mean',
  description:
    'The mean of normal observations with known noise sd 1.5 under a N(0, 4) prior: conjugate, linear-Gaussian, a normal posterior.',
  tags: ['conjugate', 'exact', 'gaussian', 'linear-gaussian', 'one-dimensional'],
  references: ['Gelman et al., Bayesian Data Analysis, 3rd ed., section 2.5 (normal model with known variance)'],
  definition: defineModel('normal-normal', (m) => {
    const y = m.data('y', { dims: ['obs'], role: 'response' })
    const mu = m.latent('mu', d.normal(NN_PRIOR.priorMean, NN_PRIOR.priorSd), { doc: 'the mean' })
    m.observe('y', d.normal(mu, NN_PRIOR.sigma), y)
  }),
  generate: (key: Key, size = 12): BenchmarkData => {
    const mu = 0.8
    const z = toFlat(drawNormal(foldIn(key, 0), [size]))
    return { data: { y: Array.from(z, (v) => mu + NN_PRIOR.sigma * v) }, generating: { mu } }
  },
  truth: (data) => {
    const exact = normalNormalExact({ ...NN_PRIOR, y: data.data['y'] as number[] })
    const [oracle] = exact.oracles('mu')
    return {
      kind: 'exact',
      source:
        'the normal posterior of a normal mean with known variance and its closed-form evidence (probprog/validation/exact.ts)',
      logEvidence: exact.logEvidence,
      marginals: { mu: oracle as NonNullable<typeof oracle> },
    }
  },
  plans: (scale) => [
    exactPlan('conjugate'),
    exactPlan('gaussian'),
    gibbsPlan(scale),
    nutsPlan(scale),
    importancePlan(scale),
    temperedPlan(scale),
    laplacePlan(scale, { expect: 'agree', note: 'exact: the posterior is normal and the Laplace fit is a normal' }),
    sviPlan(scale, {
      note: 'a mean-field normal is the right family here; what is left is the noise of the stochastic optimiser',
    }),
  ],
})

const GP_PRIOR = { shape: 2, rate: 0.5 }

/**
 * Gamma-Poisson: a rate `lambda ~ Gamma(2, 0.5)` and fifteen Poisson counts (generated at `lambda = 3.2`). The posterior
 * is `Gamma(2 + sum k, 0.5 + n)`; the rate lives on the positive reals, so the unconstrained methods work on its log.
 *
 * @concept posterior-validation
 * @index dataset Gamma-Poisson rate: exact Gamma posterior and evidence for counts.
 * @example
 * ```ts
 * import { gammaPoisson } from 'omai-engine/probprog/models'
 * gammaPoisson.truth().kind // => 'exact'
 * ```
 */
export const gammaPoisson: BenchmarkModel = defineBenchmark({
  name: 'gamma-poisson',
  title: 'Gamma-Poisson rate',
  description:
    'The rate of Poisson counts under a Gamma(2, 0.5) prior: conjugate, a Gamma posterior on the positive reals.',
  tags: ['conjugate', 'exact', 'counts', 'one-dimensional', 'positive'],
  references: ['Gelman et al., Bayesian Data Analysis, 3rd ed., section 2.6 (Poisson model)'],
  definition: defineModel('gamma-poisson', (m) => {
    const k = m.data('k', { dims: ['obs'], support: 'count', role: 'response' })
    const rate = m.latent('rate', d.gamma(GP_PRIOR.shape, GP_PRIOR.rate), { doc: 'the Poisson rate' })
    m.observe('k', d.poisson(rate), k)
  }),
  generate: (key: Key, size = 15): BenchmarkData => {
    const rate = 3.2
    return { data: { k: Array.from(toFlat(drawPoisson(foldIn(key, 0), rate, [size]))) }, generating: { rate } }
  },
  truth: (data) => {
    const exact = gammaPoissonExact({ ...GP_PRIOR, counts: data.data['k'] as number[] })
    const [oracle] = exact.oracles('rate')
    return {
      kind: 'exact',
      source: 'the Gamma(a + sum k, b + n) posterior and its closed-form evidence (probprog/validation/exact.ts)',
      logEvidence: exact.logEvidence,
      marginals: { rate: oracle as NonNullable<typeof oracle> },
    }
  },
  plans: (scale) => [
    exactPlan('conjugate'),
    gibbsPlan(scale),
    nutsPlan(scale),
    importancePlan(scale),
    temperedPlan(scale),
    laplacePlan(scale, { note: 'a normal at the mode of log rate: close, with a small skew missed' }),
    sviPlan(scale, { note: 'a mean-field normal on log rate: close, with a small skew missed' }),
  ],
})
