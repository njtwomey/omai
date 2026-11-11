/**
 * The prior atlas of expectation propagation, one setup per prior: a prior on one parameter of a distribution, with the
 * hyperprior above it where the prior has parameters of its own that are random (a normal-inverse-gamma prior on a Gaussian's
 * mean and variance, a gamma on a Poisson rate with a gamma hyperprior on the prior's rate, a beta whose strength is gamma or
 * log-normal, a normal on a log scale with a normal hyperprior on its centre). Each page shows the posterior of the parameter
 * and of the hyperparameter, both against the exact answer: a closed form where the model is conjugate (the normal-inverse-gamma
 * pair, the beta and the gamma with their counts, the Dirichlet) or a quadrature over the hyperparameter where it is not, and a
 * grid of the model written as a program for the rest. `docs/probprog/ep-priors.md` is the table the atoms illustrate.
 */

import { dict, real } from '../../../space/index.ts'
import type { Config } from '../../../trace/index.ts'
import { configNumber } from '../../inference/method.ts'
import * as d from '../../model/dists/index.ts'
import { defineModel, ops } from '../../model/index.ts'
import {
  betaMarginal,
  type ExactMarginal,
  gammaMarginal,
  inverseGammaMarginal,
  lg,
  logWeights,
  normalInverseGammaPosterior,
  normalPdf,
  quadratureMarginal,
  studentMarginal,
  sum,
} from '../../validation/index.ts'
import { defineAtom } from '../_define.ts'
import { tolerance } from '../_tolerance.ts'
import type { AtomParts, AtomTolerance, EpAtom } from '../atom-types.ts'

const EXACT = tolerance(
  'exact',
  'a conjugate model: EP is exact, so its belief equals the closed-form posterior up to rounding',
)

/** The tolerance of a hyperprior setup, where EP projects a skewed posterior of the hyperparameter onto a gamma or a Gaussian. */
const HYPER = tolerance(
  'skewed',
  'the posterior of a hyperparameter is skewed and EP carries one gamma or Gaussian for it: the means are close and the spreads within a quarter',
)

// ----- closed-form marginals -----

/**
 * The exact posterior of the strength `alpha` of a symmetric Dirichlet (or beta, for two categories) prior and of each category's
 * probability, by quadrature over `log alpha`: given `alpha` the probabilities are Dirichlet with the prior's concentration plus the
 * counts, so each probability's marginal is the mixture of betas over the posterior of `alpha`.
 */
function strengthExact(
  counts: readonly number[],
  logPrior: (alpha: number) => number,
): { readonly alpha: ExactMarginal; readonly p: readonly ExactMarginal[] } {
  const K = counts.length
  const N = sum(counts)
  const T = 2500
  const lo = -9
  const hi = 6
  const ts = Array.from({ length: T }, (_, i) => lo + ((i + 0.5) * (hi - lo)) / T)
  const logs = ts.map((t) => {
    const a = Math.exp(t)
    const ll = lg(K * a) - lg(K * a + N) + sum(counts.map((c) => lg(a + c) - lg(a)))
    return logPrior(a) + t + ll
  })
  const { weights: w0 } = logWeights(logs)
  const z = sum(w0)
  const w = w0.map((v) => v / z)
  const alphas = ts.map(Math.exp)
  const mean = sum(alphas.map((a, i) => a * (w[i] as number)))
  const sd = Math.sqrt(sum(alphas.map((a, i) => (a - mean) ** 2 * (w[i] as number))))
  const alpha: ExactMarginal = {
    mean,
    sd,
    pdf: (x) => {
      if (x <= 0) return 0
      const t = Math.log(x)
      const i = Math.round(((t - lo) / (hi - lo)) * T - 0.5)
      return i < 0 || i >= T ? 0 : (w[i] as number) / ((hi - lo) / T) / x
    },
    range: [Math.max(0, mean - 4 * sd), mean + 7 * sd],
  }
  // The components that carry weight: the marginal of a probability is their mixture of betas.
  const keep = w.map((_, i) => i).filter((i) => (w[i] as number) > 1e-7 * Math.max(...w))
  const kept = keep.map((i) => w[i] as number)
  const total = sum(kept)
  const p = counts.map((c): ExactMarginal => {
    const comps = keep.map((i, r) => {
      const a = alphas[i] as number
      const [ca, cb] = [a + c, (K - 1) * a + N - c]
      return { weight: (kept[r] as number) / total, a: ca, b: cb, logNorm: lg(ca + cb) - lg(ca) - lg(cb) }
    })
    const m1 = sum(comps.map((q) => (q.weight * q.a) / (q.a + q.b)))
    const m2 = sum(comps.map((q) => (q.weight * (q.a * (q.a + 1))) / ((q.a + q.b) * (q.a + q.b + 1))))
    return {
      mean: m1,
      sd: Math.sqrt(m2 - m1 * m1),
      pdf: (x) =>
        x <= 0 || x >= 1
          ? 0
          : sum(
              comps.map((q) => q.weight * Math.exp(q.logNorm + (q.a - 1) * Math.log(x) + (q.b - 1) * Math.log(1 - x))),
            ),
      range: [0, 1],
    }
  })
  return { alpha, p }
}

const Y6 = [1.2, 0.4, 1.9, 1.1, 0.7, 1.5]
const COUNTS = [3, 5, 2, 4]
const FLIPS = [1, 1, 0, 1, 1, 0, 1]
const CATEGORIES = [0, 1, 1, 2, 1, 0, 1]

// ----- the normal-inverse-gamma prior on a Gaussian mean and variance -----

function nigProblem(k: Config): AtomParts {
  const [m0, kappa, a, b] = [
    configNumber(k, 'priorMean'),
    configNumber(k, 'kappa'),
    configNumber(k, 'shape'),
    configNumber(k, 'scale'),
  ]
  // region prior-normal-inverse-gamma
  const definition = defineModel('normal-inverse-gamma', (m) => {
    const sigma2 = m.latent('sigma2', d.inverseGamma(a, b))
    const mu = m.latent('mu', d.normal(m0, ops.sqrt(ops.div(sigma2, kappa))))
    m.observe('y', d.normal(mu, ops.sqrt(sigma2)), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  const { mu, sigma2, joint } = normalInverseGammaPosterior(Y6, m0, kappa, a, b)
  return {
    build: {
      definition,
      data: { y: Y6 },
      quantities: [
        { name: 'mu', label: 'mu (the mean)', support: 'real' },
        { name: 'sigma2', label: 'sigma2 (the variance: the hyperparameter of the mean)', support: 'positive' },
      ],
    },
    reference: {
      exact: { mu, sigma2 },
      exactJoint: joint,
    },
    present: {
      equations: [
        `sigma2 ~ InverseGamma(${a}, ${b})   (the variance)`,
        `mu | sigma2 ~ N(${m0}, sigma2 / ${kappa})   (the mean's prior spread scales with the variance)`,
        'y_i ~ N(mu, sigma2), i = 1..6',
        `observe y = (${Y6.join(', ')})`,
      ],
      pair: ['mu', 'sigma2'],
    },
  }
}

/**
 * The classic conjugate prior on a Gaussian's mean and variance, written the natural way (an inverse gamma on the variance, a
 * normal on the mean whose variance is that variance over `kappa`): the compiler reads the pair as one normal-gamma variable and
 * the posterior is exact, a Student t for the mean and an inverse gamma for the variance.
 */
const priorNormalInverseGammaAtom: EpAtom = defineAtom({
  name: 'prior-normal-inverse-gamma',
  title: 'Prior: normal-inverse-gamma on a Gaussian mean and variance',
  what: 'An inverse-gamma prior on the variance of a Gaussian and, given it, a normal prior on its mean with variance sigma2 / kappa: the conjugate pair, one normal-gamma variable and a factor per observation.',
  factors: ['prior (normal-gamma)', 'gaussian-normal-gamma (observed value)'],
  family: 'priors',
  reference: 'exact',
  tolerance: tolerance(
    'exact-heavy-tail',
    'the posterior is exactly normal-inverse-gamma and EP carries it exactly (the moments agree to rounding); the curves compare the Gaussian and gamma shapes the page draws with the exact Student t and inverse gamma, so the distance is small, not zero',
  ),
  evidence: { nats: 1e-5, reason: 'normal-gamma: exact' },
  knobs: dict({
    priorMean: real(-2, 2, { default: 0, step: 0.1, label: 'prior mean m' }),
    kappa: real(0.1, 5, { default: 0.5, step: 0.1, label: 'kappa (prior observations for the mean)' }),
    shape: real(1, 6, { default: 3, step: 0.1, label: 'shape a of the inverse gamma' }),
    scale: real(0.5, 5, { default: 2, step: 0.1, label: 'scale b of the inverse gamma' }),
  }),
  parts: nigProblem,
})

// ----- a normal prior on a mean with a normal hyperprior on its centre -----

function hierarchicalNormalProblem(k: Config): AtomParts {
  const [s0, s1] = [configNumber(k, 'hyperSd'), configNumber(k, 'priorSd')]
  // region prior-normal-hyperprior
  const definition = defineModel('normal-hyperprior', (m) => {
    const centre = m.latent('centre', d.normal(0, s0))
    const mu = m.latent('mu', d.normal(centre, s1))
    m.observe('y', d.normal(mu, 1), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  const n = Y6.length
  // The posterior of (centre, mu) is Gaussian with precision [[1/s0² + 1/s1², -1/s1²], [-1/s1², 1/s1² + n]] and linear term [0, sum y].
  const p11 = 1 / s0 ** 2 + 1 / s1 ** 2
  const p12 = -1 / s1 ** 2
  const p22 = 1 / s1 ** 2 + n
  const det = p11 * p22 - p12 * p12
  const cov = [
    [p22 / det, -p12 / det],
    [-p12 / det, p11 / det],
  ] as const
  const h = sum(Y6)
  const means = [cov[0][1] * h, cov[1][1] * h] as const
  const marginal = (i: 0 | 1): ExactMarginal => {
    const sd = Math.sqrt(cov[i][i])
    return { mean: means[i], sd, pdf: (x) => normalPdf(x, means[i], sd) }
  }
  const xs = Array.from(
    { length: 70 },
    (_, i) => means[0] - 5 * Math.sqrt(cov[0][0]) + (i * 10 * Math.sqrt(cov[0][0])) / 69,
  )
  const ys = Array.from(
    { length: 70 },
    (_, i) => means[1] - 5 * Math.sqrt(cov[1][1]) + (i * 10 * Math.sqrt(cov[1][1])) / 69,
  )
  return {
    build: {
      definition,
      data: { y: Y6 },
      quantities: [
        { name: 'mu', label: 'mu (the mean)', support: 'real' },
        { name: 'centre', label: 'centre (the hyperparameter)', support: 'real' },
      ],
    },
    reference: {
      exact: { mu: marginal(1), centre: marginal(0) },
      exactJoint: {
        a: 'centre',
        b: 'mu',
        x: xs,
        y: ys,
        p: xs.map((x) =>
          ys.map((y) => {
            const dx = x - means[0]
            const dy = y - means[1]
            return Math.exp(-0.5 * (p11 * dx * dx + 2 * p12 * dx * dy + p22 * dy * dy))
          }),
        ),
      },
    },
    present: {
      equations: [
        `centre ~ N(0, ${s0}²)   (the hyperprior: where the prior sits)`,
        `mu ~ N(centre, ${s1}²)   (the prior on the mean)`,
        'y_i ~ N(mu, 1²), i = 1..6',
        `observe y = (${Y6.join(', ')})`,
      ],
      pair: ['centre', 'mu'],
    },
  }
}

/**
 * A normal prior on a Gaussian mean whose centre has a normal hyperprior of its own: jointly Gaussian, so EP is exact, and the
 * data inform the centre only through the mean.
 */
const priorNormalHyperAtom: EpAtom = defineAtom({
  name: 'prior-normal-hyperprior',
  title: 'Prior: normal on a Gaussian mean, normal hyperprior on its centre',
  what: 'A normal prior on a Gaussian mean whose centre is itself normal: a chain of three Gaussians, jointly Gaussian, so the posterior of the mean and of the centre are closed forms.',
  factors: ['prior (gaussian)', 'gaussian with a random mean (difference)', 'gaussian likelihood'],
  family: 'priors',
  reference: 'exact',
  tolerance: EXACT,
  evidence: { nats: 1e-3, reason: 'a Gaussian centre with a Gaussian hyperprior' },
  knobs: dict({
    hyperSd: real(0.5, 10, { default: 3, step: 0.1, label: 'sd of the hyperprior' }),
    priorSd: real(0.2, 3, { default: 1, step: 0.1, label: 'sd of the prior given its centre' }),
  }),
  parts: hierarchicalNormalProblem,
})

// ----- an inverse gamma on a variance with a gamma hyperprior on its scale -----

function inverseGammaHyperProblem(k: Config): AtomParts {
  const [a, c, dd] = [configNumber(k, 'shape'), configNumber(k, 'hyperShape'), configNumber(k, 'hyperRate')]
  // region prior-inverse-gamma-hyperprior
  const definition = defineModel('inverse-gamma-hyperprior', (m) => {
    const scale = m.latent('scale', d.gamma(c, dd))
    const sigma2 = m.latent('sigma2', d.inverseGamma(a, scale))
    m.observe('y', d.normal(1, ops.sqrt(sigma2)), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y: Y6 },
      quantities: [
        { name: 'sigma2', label: 'sigma2 (the variance)', support: 'positive' },
        { name: 'scale', label: 'scale (the hyperparameter)', support: 'positive' },
      ],
    },
    reference: {
      grid: { points: 61, maxRefinements: 0 },
    },
    present: {
      equations: [
        `scale ~ Gamma(${c}, ${dd})   (the hyperprior on the inverse gamma's scale)`,
        `sigma2 ~ InverseGamma(${a}, scale)   (the variance)`,
        'y_i ~ N(1, sigma2), i = 1..6   (the mean is known)',
        `observe y = (${Y6.join(', ')})`,
      ],
      pair: ['sigma2', 'scale'],
    },
  }
}

/**
 * A gamma hyperprior on the scale of an inverse-gamma variance: the variance is a gamma precision whose rate is the
 * hyperparameter, the `gamma-rate` hierarchy, and the Gaussian observations are the gamma-precision factor.
 */
const priorInverseGammaHyperAtom: EpAtom = defineAtom({
  name: 'prior-inverse-gamma-hyperprior',
  title: 'Prior: inverse-gamma on a Gaussian variance, gamma hyperprior on its scale',
  what: 'An inverse-gamma prior on a variance whose scale is gamma distributed, six Gaussian observations around a known mean: the variance is a gamma precision with a random rate (the gamma-rate hierarchy).',
  factors: ['prior (gamma)', 'gamma-rate (a gamma with a gamma rate)', 'gaussian-gamma-precision (observed value)'],
  family: 'priors',
  reference: 'grid',
  tolerance: tolerance(
    'measured',
    'the posterior of the variance is heavy-tailed: EP carries a gamma for its reciprocal, which matches the mean and the scale hyperparameter and leaves the variance an inverse gamma whose spread is about a fifth too wide',
    { meanZ: 0.15, sdRatio: [0.75, 1.3], ks: 0.18 },
  ),
  evidence: { nats: 1e-3, reason: 'the gamma precision rule on a grid in log lambda' },
  knobs: dict({
    shape: real(1, 6, { default: 3, step: 0.1, label: 'shape a of the inverse gamma' }),
    hyperShape: real(0.5, 6, { default: 2, step: 0.1, label: 'shape of the hyperprior' }),
    hyperRate: real(0.2, 4, { default: 1, step: 0.1, label: 'rate of the hyperprior' }),
  }),
  parts: inverseGammaHyperProblem,
})

// ----- a gamma on a Poisson rate with a gamma hyperprior on the prior's rate -----

function gammaPoissonHyperProblem(k: Config): AtomParts {
  const [a, c, dd] = [configNumber(k, 'shape'), configNumber(k, 'hyperShape'), configNumber(k, 'hyperRate')]
  // region prior-gamma-poisson-hyperprior
  const definition = defineModel('gamma-poisson-hyperprior', (m) => {
    const prior = m.latent('priorRate', d.gamma(c, dd))
    const rate = m.latent('rate', d.gamma(a, prior))
    m.observe('y', d.poisson(rate), m.data('y', { dims: ['i'], support: 'count' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y: COUNTS },
      quantities: [
        { name: 'rate', label: 'rate (the Poisson rate)', support: 'positive' },
        { name: 'priorRate', label: 'priorRate (the hyperparameter)', support: 'positive' },
      ],
    },
    reference: {
      grid: { points: 61, maxRefinements: 0 },
    },
    present: {
      equations: [
        `priorRate ~ Gamma(${c}, ${dd})   (the hyperprior: how far the rate can be from zero)`,
        `rate ~ Gamma(${a}, priorRate)   (the prior on the Poisson rate)`,
        `y_i ~ Poisson(rate), i = 1..${COUNTS.length}`,
        `observe y = (${COUNTS.join(', ')})`,
      ],
      pair: ['rate', 'priorRate'],
    },
  }
}

/**
 * A gamma prior on a Poisson rate whose own rate has a gamma hyperprior: the `gamma-rate` hierarchy under the conjugate
 * Poisson factor.
 */
const priorGammaPoissonHyperAtom: EpAtom = defineAtom({
  name: 'prior-gamma-poisson-hyperprior',
  title: 'Prior: gamma on a Poisson rate, gamma hyperprior on its rate',
  what: 'A gamma prior on a Poisson rate whose rate parameter is gamma distributed, four counts: the gamma-rate hierarchy under the conjugate Poisson factor.',
  factors: ['prior (gamma)', 'gamma-rate', 'poisson likelihood (gamma-poisson)'],
  family: 'priors',
  reference: 'grid',
  tolerance: HYPER,
  evidence: { nats: 1e-3, reason: 'a gamma rate with a gamma hyperprior' },
  knobs: dict({
    shape: real(0.5, 8, { default: 2, step: 0.1, label: 'shape of the rate prior' }),
    hyperShape: real(0.5, 6, { default: 2, step: 0.1, label: 'shape of the hyperprior' }),
    hyperRate: real(0.2, 4, { default: 2, step: 0.1, label: 'rate of the hyperprior' }),
  }),
  parts: gammaPoissonHyperProblem,
})

// ----- a log-normal prior on a Poisson rate -----

function logNormalPoissonProblem(k: Config): AtomParts {
  const [mu0, s0] = [configNumber(k, 'logMean'), configNumber(k, 'logSd')]
  // region prior-lognormal-poisson
  const definition = defineModel('lognormal-poisson', (m) => {
    const rate = m.latent('rate', d.logNormal(mu0, s0))
    m.observe('y', d.poisson(rate), m.data('y', { dims: ['i'], support: 'count' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y: COUNTS },
      quantities: [{ name: 'rate', label: 'rate (the Poisson rate)', support: 'positive' }],
    },
    reference: {
      grid: { points: 201, maxRefinements: 0 },
    },
    present: {
      equations: [
        `rate ~ LogNormal(${mu0}, ${s0}²)   (log rate ~ N(${mu0}, ${s0}²))`,
        `y_i ~ Poisson(rate), i = 1..${COUNTS.length}`,
        `observe y = (${COUNTS.join(', ')})`,
      ],
    },
  }
}

/**
 * A log-normal prior on a Poisson rate: the prior is a Gaussian on the log rate (projected once, carried on its log) and the
 * count is read through the log link, the numeric likelihood of the registry.
 */
const priorLogNormalPoissonAtom: EpAtom = defineAtom({
  name: 'prior-lognormal-poisson',
  title: 'Prior: log-normal on a Poisson rate',
  what: 'A log-normal prior on a Poisson rate and four counts: a Gaussian on the log rate and the numeric Poisson likelihood with a log link.',
  factors: ['prior (gaussian, on the log rate)', 'numeric likelihood: poisson with an exp link'],
  family: 'priors',
  reference: 'grid',
  tolerance: tolerance('skewed', HYPER.reason, { meanZ: 0.2 }),
  evidence: { nats: 0.01, reason: 'a log-normal rate through exp' },
  knobs: dict({
    logMean: real(-1, 3, { default: 1, step: 0.1, label: 'mean of the log rate' }),
    logSd: real(0.2, 2, { default: 0.8, step: 0.1, label: 'sd of the log rate' }),
  }),
  parts: logNormalPoissonProblem,
})

function logNormalPoissonHyperProblem(k: Config): AtomParts {
  const [s0, s1] = [configNumber(k, 'hyperSd'), configNumber(k, 'logSd')]
  // region prior-lognormal-poisson-hyperprior
  const definition = defineModel('lognormal-poisson-hyperprior', (m) => {
    const centre = m.latent('centre', d.normal(1, s0))
    const logRate = m.latent('logRate', d.normal(centre, s1))
    m.observe('y', d.poisson(ops.exp(logRate)), m.data('y', { dims: ['i'], support: 'count' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y: COUNTS },
      quantities: [
        { name: 'logRate', label: 'logRate (the log of the rate)', support: 'real' },
        { name: 'centre', label: 'centre (the hyperparameter)', support: 'real' },
      ],
    },
    reference: {
      grid: { points: 61, maxRefinements: 0 },
    },
    present: {
      equations: [
        `centre ~ N(1, ${s0}²)   (the hyperprior on the log rate's centre)`,
        `logRate ~ N(centre, ${s1}²)   (so rate = exp(logRate) is log-normal)`,
        `y_i ~ Poisson(exp(logRate)), i = 1..${COUNTS.length}`,
        `observe y = (${COUNTS.join(', ')})`,
      ],
      pair: ['centre', 'logRate'],
    },
  }
}

/**
 * A log-normal prior on a Poisson rate written on the log scale, with a normal hyperprior on the centre of the log rate.
 */
const priorLogNormalPoissonHyperAtom: EpAtom = defineAtom({
  name: 'prior-lognormal-poisson-hyperprior',
  title: 'Prior: log-normal on a Poisson rate, normal hyperprior on its log centre',
  what: 'The log of a Poisson rate is normal around a centre that is itself normal: two Gaussians and the numeric Poisson likelihood with a log link.',
  factors: ['prior (gaussian)', 'gaussian with a random mean', 'numeric likelihood: poisson with an exp link'],
  family: 'priors',
  reference: 'grid',
  tolerance: tolerance('skewed', HYPER.reason, { meanZ: 0.2 }),
  knobs: dict({
    hyperSd: real(0.3, 4, { default: 1, step: 0.1, label: 'sd of the hyperprior' }),
    logSd: real(0.2, 2, { default: 0.5, step: 0.1, label: 'sd of the log rate given its centre' }),
  }),
  parts: logNormalPoissonHyperProblem,
})

// ----- a gamma on the rate of an exponential -----

const WAITS = [0.8, 1.9, 0.3, 1.2, 0.6]

function gammaExponentialProblem(k: Config): AtomParts {
  const [a, b] = [configNumber(k, 'shape'), configNumber(k, 'rate')]
  // region prior-gamma-exponential
  const definition = defineModel('gamma-exponential', (m) => {
    const rate = m.latent('rate', d.gamma(a, b))
    m.observe('y', d.exponential(rate), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y: WAITS },
      quantities: [{ name: 'rate', label: 'rate (of the waiting times)', support: 'positive' }],
    },
    reference: {
      exact: { rate: gammaMarginal(a + WAITS.length, b + sum(WAITS)) },
    },
    present: {
      equations: [
        `rate ~ Gamma(${a}, ${b})`,
        `y_i ~ Exponential(rate), i = 1..${WAITS.length}   (waiting times)`,
        `observe y = (${WAITS.join(', ')})`,
      ],
    },
  }
}

/**
 * A gamma prior on the rate of exponential waiting times: the likelihood is a gamma in the rate, so the posterior is the gamma
 * with the shape plus the count and the rate plus the total time.
 */
const priorGammaExponentialAtom: EpAtom = defineAtom({
  name: 'prior-gamma-exponential',
  title: 'Prior: gamma on an exponential rate',
  what: 'A gamma prior on the rate of exponential waiting times and five observed times: a conjugate gamma message from each observation.',
  factors: ['prior (gamma)', 'gamma-observed (an exponential value with a gamma rate)'],
  family: 'priors',
  reference: 'exact',
  tolerance: tolerance('exact-curve', `${EXACT.reason} (the curve compared is the matched gamma, the exact posterior)`),
  evidence: { nats: 1e-5, reason: 'conjugate: exact' },
  knobs: dict({
    shape: real(0.2, 10, { default: 2, step: 0.1, label: 'prior shape' }),
    rate: real(0.05, 5, { default: 1, step: 0.05, label: 'prior rate' }),
  }),
  parts: gammaExponentialProblem,
})

// ----- a normal prior on the log scale of a Weibull, with a hyperprior on its centre -----

const LIFETIMES = [2.1, 0.7, 1.4, 3.0, 1.1]

function weibullProblem(k: Config): AtomParts {
  const [s0, s1] = [configNumber(k, 'hyperSd'), configNumber(k, 'logSd')]
  const shape = configNumber(k, 'shape')
  // region prior-normal-weibull-scale
  const definition = defineModel('normal-weibull-scale', (m) => {
    const centre = m.latent('centre', d.normal(0, s0))
    const logScale = m.latent('logScale', d.normal(centre, s1))
    const observed = m.data('y', { dims: ['i'], role: 'response' })
    m.observe('y', d.weibull(shape, ops.exp(logScale)), observed)
  })
  // endregion
  return {
    build: {
      definition,
      data: { y: LIFETIMES },
      quantities: [
        { name: 'logScale', label: 'logScale (the log of the Weibull scale)', support: 'real' },
        { name: 'centre', label: 'centre (the hyperparameter)', support: 'real' },
      ],
    },
    reference: {
      grid: { points: 61, maxRefinements: 0 },
    },
    present: {
      equations: [
        `centre ~ N(0, ${s0}²)   (the hyperprior)`,
        `logScale ~ N(centre, ${s1}²)   (the Weibull scale is log-normal)`,
        `y_i ~ Weibull(${shape}, exp(logScale)), i = 1..${LIFETIMES.length}   (lifetimes)`,
        `observe y = (${LIFETIMES.join(', ')})`,
      ],
      pair: ['centre', 'logScale'],
    },
  }
}

/**
 * A normal prior on the log scale of a Weibull lifetime (so a log-normal scale) with a normal hyperprior on its centre: the
 * numeric Weibull likelihood of the registry with a log link.
 */
const priorNormalWeibullAtom: EpAtom = defineAtom({
  name: 'prior-normal-weibull-scale',
  title: 'Prior: normal on a Weibull log scale, normal hyperprior on its centre',
  what: 'The log of a Weibull scale is normal around a normal centre, five observed lifetimes with a known shape: two Gaussians and the numeric Weibull likelihood.',
  factors: ['prior (gaussian)', 'gaussian with a random mean', 'numeric likelihood: weibull with a log link'],
  family: 'priors',
  reference: 'grid',
  tolerance: tolerance('skewed', HYPER.reason, { meanZ: 0.2 }),
  evidence: { nats: 0.02, reason: 'a Weibull with a random scale' },
  knobs: dict({
    shape: real(0.5, 4, { default: 1.5, step: 0.1, label: 'Weibull shape (known)' }),
    hyperSd: real(0.3, 4, { default: 1, step: 0.1, label: 'sd of the hyperprior' }),
    logSd: real(0.2, 2, { default: 0.5, step: 0.1, label: 'sd of the log scale given its centre' }),
  }),
  parts: weibullProblem,
})

// ----- a beta prior on a geometric success probability -----

const TRIALS = [3, 5, 2, 4]

function betaGeometricProblem(k: Config): AtomParts {
  const [a, b] = [configNumber(k, 'a'), configNumber(k, 'b')]
  // region prior-beta-geometric
  const definition = defineModel('beta-geometric', (m) => {
    const p = m.latent('p', d.beta(a, b))
    m.observe('y', d.geometric(p), m.data('y', { dims: ['i'], support: 'count' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y: TRIALS },
      quantities: [{ name: 'p', label: 'p (the success probability)', support: 'unit' }],
    },
    reference: {
      exact: { p: betaMarginal(a + TRIALS.length, b + sum(TRIALS) - TRIALS.length) },
    },
    present: {
      equations: [
        `p ~ Beta(${a}, ${b})`,
        `y_i ~ Geometric(p), i = 1..${TRIALS.length}   (trials up to the first success)`,
        `observe y = (${TRIALS.join(', ')})`,
      ],
    },
  }
}

/**
 * A beta prior on the success probability of a geometric count: the likelihood `p (1 - p)^(y - 1)` is a beta in `p`, exact.
 */
const priorBetaGeometricAtom: EpAtom = defineAtom({
  name: 'prior-beta-geometric',
  title: 'Prior: beta on a geometric success probability',
  what: 'A beta prior on the success probability and four geometric counts (trials to the first success): the likelihood is a beta in the probability.',
  factors: ['prior (beta)', 'negative-binomial-beta (a geometric count with a beta probability)'],
  family: 'priors',
  reference: 'exact',
  tolerance: tolerance('exact-curve', `${EXACT.reason} (the curve compared is the matched beta, the exact posterior)`),
  evidence: { nats: 1e-5, reason: 'conjugate: exact' },
  knobs: dict({
    a: real(0.2, 20, { default: 2, step: 0.1, label: 'prior a' }),
    b: real(0.2, 20, { default: 2, step: 0.1, label: 'prior b' }),
  }),
  parts: betaGeometricProblem,
})

// ----- a beta with a random strength: gamma and log-normal hyperpriors -----

function betaStrengthProblem(kind: 'gamma' | 'lognormal', k: Config): AtomParts {
  const [a, b] = [
    configNumber(k, kind === 'gamma' ? 'shape' : 'logMean'),
    configNumber(k, kind === 'gamma' ? 'rate' : 'logSd'),
  ]
  // region prior-beta-strength-gamma prior-beta-strength-lognormal
  const definition = defineModel(`beta-strength-${kind}`, (m) => {
    const strength = m.latent('strength', kind === 'gamma' ? d.gamma(a, b) : d.logNormal(a, b))
    const p = m.latent('p', d.beta(strength, strength))
    m.observe('y', d.bernoulli({ probs: p }), m.data('y', { dims: ['i'], support: 'binary' }))
  })
  // endregion
  const logPrior =
    kind === 'gamma'
      ? (x: number): number => (a - 1) * Math.log(x) - b * x
      : (x: number): number => -0.5 * ((Math.log(x) - a) / b) ** 2 - Math.log(x)
  const exact = strengthExact([sum(FLIPS), FLIPS.length - sum(FLIPS)], logPrior)
  return {
    build: {
      definition,
      data: { y: FLIPS },
      quantities: [
        { name: 'p', label: 'p (the probability)', support: 'unit' },
        { name: 'strength', label: 'strength (the hyperparameter)', support: 'positive' },
      ],
    },
    reference: {
      exact: { p: exact.p[0] as ExactMarginal, strength: exact.alpha },
    },
    present: {
      equations: [
        kind === 'gamma'
          ? `strength ~ Gamma(${a}, ${b})   (the hyperprior)`
          : `strength ~ LogNormal(${a}, ${b}²)   (the hyperprior)`,
        'p ~ Beta(strength, strength)   (a symmetric beta: its strength is random)',
        `y_i ~ Bernoulli(p), i = 1..${FLIPS.length}`,
        `observe y = (${FLIPS.join(', ')})`,
      ],
    },
  }
}

const STRENGTH_TOLERANCE = tolerance(
  'close',
  'the posterior of the strength is skewed and EP carries a gamma for it (a Gaussian on its log for the log-normal); the probability is the matched beta of the mixture over the strength',
)

/**
 * A symmetric beta prior on a Bernoulli probability whose strength has a gamma hyperprior: the numeric `dirichlet-concentration`
 * factor (a grid over the log strength). The exact answer is a quadrature over the strength.
 */
const priorBetaStrengthGammaAtom: EpAtom = defineAtom({
  name: 'prior-beta-strength-gamma',
  title: 'Prior: beta on a probability, gamma hyperprior on its strength',
  what: 'A symmetric Beta(strength, strength) prior on a Bernoulli probability with a gamma-distributed strength: the strength says whether the probability sits near one half or near the ends.',
  factors: [
    'prior (gamma)',
    'dirichlet-concentration (a beta with a random concentration)',
    'bernoulli likelihood (beta-bernoulli)',
  ],
  family: 'priors',
  reference: 'exact',
  tolerance: STRENGTH_TOLERANCE,
  evidence: { nats: 1e-3, reason: 'a beta with a random concentration' },
  knobs: dict({
    shape: real(0.5, 8, { default: 2, step: 0.1, label: 'shape of the strength prior' }),
    rate: real(0.1, 4, { default: 1, step: 0.1, label: 'rate of the strength prior' }),
  }),
  parts: (k) => betaStrengthProblem('gamma', k),
})

/**
 * The same with a log-normal hyperprior on the strength, carried on its log (a Gaussian site in the factor).
 */
const priorBetaStrengthLogNormalAtom: EpAtom = defineAtom({
  name: 'prior-beta-strength-lognormal',
  title: 'Prior: beta on a probability, log-normal hyperprior on its strength',
  what: 'A symmetric Beta(strength, strength) prior on a Bernoulli probability with a log-normal strength: the factor takes the log strength as a Gaussian variable.',
  factors: [
    'prior (gaussian, on the log strength)',
    'dirichlet-concentration (log mode)',
    'bernoulli likelihood (beta-bernoulli)',
  ],
  family: 'priors',
  reference: 'exact',
  tolerance: tolerance('close', STRENGTH_TOLERANCE.reason, { ks: 0.12 }),
  knobs: dict({
    logMean: real(-1, 3, { default: 0.7, step: 0.1, label: 'mean of the log strength' }),
    logSd: real(0.2, 2, { default: 0.8, step: 0.1, label: 'sd of the log strength' }),
  }),
  parts: (k) => betaStrengthProblem('lognormal', k),
})

// ----- a Dirichlet prior on categorical probabilities -----

function dirichletCategoricalProblem(k: Config): AtomParts {
  const a = configNumber(k, 'concentration')
  // region prior-dirichlet-categorical
  const definition = defineModel('dirichlet-categorical', (m) => {
    const p = m.latent('p', d.dirichlet([a, a, a]))
    m.observe('y', d.categorical({ probs: p }), m.data('y', { dims: ['i'], support: 'count' }))
  })
  // endregion
  const counts = [0, 1, 2].map((c) => CATEGORIES.filter((v) => v === c).length)
  const N = sum(counts)
  return {
    build: {
      definition,
      data: { y: CATEGORIES },
      quantities: counts.map((_, c) => ({ name: `p[${c}]`, label: `p of category ${c}`, support: 'unit' as const })),
    },
    reference: {
      exact: Object.fromEntries(counts.map((n, c) => [`p[${c}]`, betaMarginal(a + n, 2 * a + N - n)])),
    },
    present: {
      equations: [
        `p ~ Dirichlet(${a}, ${a}, ${a})   (probabilities of three categories)`,
        `y_i ~ Categorical(p), i = 1..${CATEGORIES.length}`,
        `observe y = (${CATEGORIES.join(', ')})   (counts ${counts.join(', ')})`,
      ],
    },
  }
}

/**
 * A Dirichlet prior on the probabilities of a categorical variable: each marginal of the posterior is the beta of its count.
 */
const priorDirichletCategoricalAtom: EpAtom = defineAtom({
  name: 'prior-dirichlet-categorical',
  title: 'Prior: Dirichlet on categorical probabilities',
  what: 'A symmetric Dirichlet prior on three category probabilities and seven categorical observations: the Dirichlet-categorical conjugate pair.',
  factors: ['prior (dirichlet)', 'discrete likelihood (discrete-dirichlet)'],
  family: 'priors',
  reference: 'exact',
  tolerance: tolerance('exact-curve', `${EXACT.reason} (the curve compared is the matched beta, the exact marginal)`),
  evidence: { nats: 1e-5, reason: 'conjugate: exact' },
  knobs: dict({ concentration: real(0.2, 10, { default: 1, step: 0.1, label: 'prior concentration' }) }),
  parts: dirichletCategoricalProblem,
})

function dirichletStrengthProblem(k: Config): AtomParts {
  const [a, b] = [configNumber(k, 'shape'), configNumber(k, 'rate')]
  // region prior-dirichlet-strength-gamma
  const definition = defineModel('dirichlet-strength-gamma', (m) => {
    const strength = m.latent('strength', d.gamma(a, b))
    const p = m.latent('p', d.dirichlet(ops.mul(strength, m.const([1, 1, 1]))))
    m.observe('y', d.categorical({ probs: p }), m.data('y', { dims: ['i'], support: 'count' }))
  })
  // endregion
  const counts = [0, 1, 2].map((c) => CATEGORIES.filter((v) => v === c).length)
  const exact = strengthExact(counts, (x) => (a - 1) * Math.log(x) - b * x)
  return {
    build: {
      definition,
      data: { y: CATEGORIES },
      quantities: [
        { name: 'strength', label: 'strength (the hyperparameter)', support: 'positive' },
        ...counts.map((_, c) => ({ name: `p[${c}]`, label: `p of category ${c}`, support: 'unit' as const })),
      ],
    },
    reference: {
      exact: { strength: exact.alpha, ...Object.fromEntries(exact.p.map((m, c) => [`p[${c}]`, m])) },
    },
    present: {
      equations: [
        `strength ~ Gamma(${a}, ${b})   (the hyperprior)`,
        'p ~ Dirichlet(strength, strength, strength)   (a symmetric Dirichlet with a random concentration)',
        `y_i ~ Categorical(p), i = 1..${CATEGORIES.length}`,
        `observe y = (${CATEGORIES.join(', ')})   (counts ${counts.join(', ')})`,
      ],
    },
  }
}

/**
 * A symmetric Dirichlet prior on categorical probabilities whose concentration has a gamma hyperprior: the numeric
 * `dirichlet-concentration` factor with three components.
 */
const priorDirichletStrengthAtom: EpAtom = defineAtom({
  name: 'prior-dirichlet-strength-gamma',
  title: 'Prior: Dirichlet on categorical probabilities, gamma hyperprior on its concentration',
  what: 'A symmetric Dirichlet with a gamma-distributed concentration over three categories: the concentration says how even the probabilities are, and the data inform it through the counts.',
  factors: [
    'prior (gamma)',
    'dirichlet-concentration (a Dirichlet with a random concentration)',
    'discrete likelihood (discrete-dirichlet)',
  ],
  family: 'priors',
  reference: 'exact',
  tolerance: STRENGTH_TOLERANCE,
  knobs: dict({
    shape: real(0.5, 8, { default: 2, step: 0.1, label: 'shape of the concentration prior' }),
    rate: real(0.1, 4, { default: 1, step: 0.1, label: 'rate of the concentration prior' }),
  }),
  parts: dirichletStrengthProblem,
})

// ----- a logit-normal prior on a Bernoulli probability -----

function logitNormalProblem(k: Config): AtomParts {
  const [s0, s1] = [configNumber(k, 'hyperSd'), configNumber(k, 'logitSd')]
  // region prior-logit-normal-bernoulli
  const definition = defineModel('logit-normal-bernoulli', (m) => {
    const centre = m.latent('centre', d.normal(0, s0))
    const logit = m.latent('logit', d.normal(centre, s1))
    m.observe('y', d.bernoulli({ logits: logit }), m.data('y', { dims: ['i'], support: 'binary' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y: FLIPS },
      quantities: [
        { name: 'logit', label: 'logit (the log odds of p)', support: 'real' },
        { name: 'centre', label: 'centre (the hyperparameter)', support: 'real' },
      ],
    },
    reference: {
      grid: { points: 61, maxRefinements: 0 },
    },
    present: {
      equations: [
        `centre ~ N(0, ${s0}²)   (the hyperprior on the logit's centre)`,
        `logit ~ N(centre, ${s1}²)   (p = sigmoid(logit) is logit-normal)`,
        `y_i ~ Bernoulli(sigmoid(logit)), i = 1..${FLIPS.length}`,
        `observe y = (${FLIPS.join(', ')})`,
      ],
      pair: ['centre', 'logit'],
    },
  }
}

/**
 * A logit-normal prior on a Bernoulli probability (a normal on the log odds) with a normal hyperprior on its centre: the logistic
 * factor on a Gaussian predictor.
 */
const priorLogitNormalAtom: EpAtom = defineAtom({
  name: 'prior-logit-normal-bernoulli',
  title: 'Prior: logit-normal on a Bernoulli probability, normal hyperprior on its centre',
  what: 'The log odds of a Bernoulli probability are normal around a normal centre, seven flips: two Gaussians and the logistic factor.',
  factors: ['prior (gaussian)', 'gaussian with a random mean', 'logistic likelihood'],
  family: 'priors',
  reference: 'grid',
  tolerance: tolerance('skewed', HYPER.reason, { meanZ: 0.2 }),
  evidence: { nats: 0.01, reason: 'a logistic-normal probability' },
  knobs: dict({
    hyperSd: real(0.3, 4, { default: 1, step: 0.1, label: 'sd of the hyperprior' }),
    logitSd: real(0.2, 3, { default: 1, step: 0.1, label: 'sd of the log odds given the centre' }),
  }),
  parts: logitNormalProblem,
})

// ----- a uniform prior on a Gaussian mean -----

function uniformProblem(k: Config): AtomParts {
  const [lo, hi] = [configNumber(k, 'low'), configNumber(k, 'high')]
  // region prior-uniform-mean
  const definition = defineModel('uniform-mean', (m) => {
    const mu = m.latent('mu', d.uniform(lo, hi))
    m.observe('y', d.normal(mu, 1), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  const n = Y6.length
  const ybar = sum(Y6) / n
  return {
    build: {
      definition,
      data: { y: Y6 },
      quantities: [{ name: 'mu', label: 'mu (the mean)', support: 'real' }],
    },
    reference: {
      exact: { mu: quadratureMarginal((x) => -0.5 * n * (x - ybar) ** 2, lo, hi) },
    },
    present: {
      equations: [`mu ~ Uniform(${lo}, ${hi})`, 'y_i ~ N(mu, 1²), i = 1..6', `observe y = (${Y6.join(', ')})`],
    },
  }
}

/**
 * A uniform prior on a Gaussian mean: a prior with no message family, kept as the exact numeric `prior-site` (its message is the
 * moment match against the current cavity at every sweep), so the posterior is the truncated Gaussian approximated by a Gaussian.
 */
const priorUniformAtom: EpAtom = defineAtom({
  name: 'prior-uniform-mean',
  title: 'Prior: uniform on a Gaussian mean',
  what: 'A uniform prior on a Gaussian mean with six observations: the density has no message family, so it is carried as an exact numeric site that EP matches against the current cavity at every sweep.',
  factors: ['numeric prior site (uniform, exact)', 'gaussian likelihood'],
  family: 'priors',
  reference: 'exact',
  tolerance: tolerance(
    'coupled',
    'the uniform prior is replaced once by the Gaussian with its mean and variance, so a bound that cuts into the posterior is not respected; where the data sit well inside the bounds the two agree',
  ),
  evidence: {
    nats: 1e-6,
    reason:
      'the uniform prior is the exact numeric site on the value, and a Gaussian likelihood makes the posterior a truncated normal: exact (was 0.09 off when projected once)',
  },
  knobs: dict({
    low: real(-6, 1, { default: -3, step: 0.5, label: 'lower bound' }),
    high: real(1.5, 8, { default: 3, step: 0.5, label: 'upper bound' }),
  }),
  parts: uniformProblem,
})

// ----- a log-normal prior on a Gaussian scale with a hyperprior on its centre -----

const SMALL = [0.5, -1.2, 2.0, 0.3]

function logNormalScaleProblem(k: Config): AtomParts {
  const [s0, s1] = [configNumber(k, 'hyperSd'), configNumber(k, 'logSd')]
  // region prior-lognormal-scale-hyperprior
  const definition = defineModel('lognormal-scale-hyperprior', (m) => {
    const centre = m.latent('centre', d.normal(0, s0))
    const logScale = m.latent('logScale', d.normal(centre, s1))
    m.observe('y', d.normal(0, ops.exp(logScale)), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y: SMALL },
      quantities: [
        { name: 'logScale', label: 'logScale (the log of the sd)', support: 'real' },
        { name: 'centre', label: 'centre (the hyperparameter)', support: 'real' },
      ],
    },
    reference: {
      grid: { points: 61, maxRefinements: 0 },
    },
    present: {
      equations: [
        `centre ~ N(0, ${s0}²)   (the hyperprior on the log scale's centre)`,
        `logScale ~ N(centre, ${s1}²)   (the standard deviation exp(logScale) is log-normal)`,
        `y_i ~ N(0, exp(logScale)²), i = 1..${SMALL.length}`,
        `observe y = (${SMALL.join(', ')})`,
      ],
      pair: ['centre', 'logScale'],
    },
  }
}

/**
 * A log-normal prior on the standard deviation of a Gaussian (a normal on its log) with a normal hyperprior on the centre: the
 * `gaussian-log-scale` factor, a one-dimensional quadrature in the log scale.
 */
const priorLogNormalScaleAtom: EpAtom = defineAtom({
  name: 'prior-lognormal-scale-hyperprior',
  title: 'Prior: log-normal on a Gaussian scale, normal hyperprior on its log centre',
  what: 'The log of a Gaussian standard deviation is normal around a normal centre, four zero-mean observations: two Gaussians and the Gaussian with an exp(h) scale.',
  factors: ['prior (gaussian)', 'gaussian with a random mean', 'gaussian-log-scale'],
  family: 'priors',
  reference: 'grid',
  tolerance: tolerance('skewed', HYPER.reason, { meanZ: 0.2 }),
  knobs: dict({
    hyperSd: real(0.3, 4, { default: 1, step: 0.1, label: 'sd of the hyperprior' }),
    logSd: real(0.2, 2, { default: 0.5, step: 0.1, label: 'sd of the log scale given its centre' }),
  }),
  parts: logNormalScaleProblem,
})

// ----- a beta with two random shapes -----

function betaTwoShapesProblem(k: Config): AtomParts {
  const s = configNumber(k, 'logSd')
  // region prior-beta-two-shapes
  const definition = defineModel('beta-two-shapes', (m) => {
    const a = m.latent('a', d.logNormal(0.5, s))
    const b = m.latent('b', d.logNormal(0.5, s))
    const p = m.latent('p', d.beta(a, b))
    m.observe('y', d.bernoulli({ probs: p }), m.data('y', { dims: ['i'], support: 'binary' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y: FLIPS },
      quantities: [
        { name: 'p', label: 'p (the probability)', support: 'unit' },
        { name: 'a', label: 'a (first shape)', support: 'positive' },
        { name: 'b', label: 'b (second shape)', support: 'positive' },
      ],
    },
    reference: {
      grid: { points: 25, maxRefinements: 0 },
    },
    present: {
      equations: [
        `a ~ LogNormal(0.5, ${s}²), b ~ LogNormal(0.5, ${s}²)   (the two shapes, each with its own hyperprior)`,
        'p ~ Beta(a, b)',
        `y_i ~ Bernoulli(p), i = 1..${FLIPS.length}`,
        `observe y = (${FLIPS.join(', ')})`,
      ],
    },
  }
}

/**
 * A beta prior with two random shapes, each with its own log-normal hyperprior: the tilted distribution is two-dimensional (a
 * grid over both shapes), which the one-dimensional concentration factor does not cover; the `beta-two-shapes` factor does, with a
 * grid in `(log a, log b)` and the conjugate update of `p` at each node.
 */
const priorBetaTwoShapesAtom: EpAtom = defineAtom({
  name: 'prior-beta-two-shapes',
  title: 'Prior: beta with two random shapes',
  what: 'A beta prior on a Bernoulli probability whose two shapes are separate log-normal variables: a three-variable factor with a two-dimensional tilted integral, which a grid in the two log shapes does.',
  factors: [
    'prior (gaussian, on each log shape)',
    'beta-two-shapes (a 2-d grid in the log shapes)',
    'bernoulli likelihood',
  ],
  family: 'priors',
  reference: 'grid',
  tolerance: tolerance('skewed', HYPER.reason, { meanZ: 0.25 }),
  evidence: { nats: 1e-3, reason: 'a beta with two random shapes: the grid in both log shapes (2e-5 measured)' },
  knobs: dict({ logSd: real(0.2, 2, { default: 0.7, step: 0.1, label: 'sd of each log shape' }) }),
  parts: betaTwoShapesProblem,
})

/** The atoms of the `priors` file, basic and learned twins together. */
export const priorsAtoms: readonly EpAtom[] = [
  priorNormalInverseGammaAtom,
  priorNormalHyperAtom,
  priorInverseGammaHyperAtom,
  priorGammaPoissonHyperAtom,
  priorLogNormalPoissonAtom,
  priorLogNormalPoissonHyperAtom,
  priorGammaExponentialAtom,
  priorNormalWeibullAtom,
  priorBetaGeometricAtom,
  priorBetaStrengthGammaAtom,
  priorBetaStrengthLogNormalAtom,
  priorDirichletCategoricalAtom,
  priorDirichletStrengthAtom,
  priorLogitNormalAtom,
  priorUniformAtom,
  priorLogNormalScaleAtom,
  priorBetaTwoShapesAtom,
]
