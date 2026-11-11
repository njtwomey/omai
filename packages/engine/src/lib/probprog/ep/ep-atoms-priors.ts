/**
 * The prior atlas of expectation propagation, one setup per prior: a prior on one parameter of a distribution, with the
 * hyperprior above it where the prior has parameters of its own that are random (a normal-inverse-gamma prior on a Gaussian's
 * mean and variance, a gamma on a Poisson rate with a gamma hyperprior on the prior's rate, a beta whose strength is gamma or
 * log-normal, a normal on a log scale with a normal hyperprior on its centre). Each page shows the posterior of the parameter
 * and of the hyperparameter, both against the exact answer: a closed form where the model is conjugate (the normal-inverse-gamma
 * pair, the beta and the gamma with their counts, the Dirichlet) or a quadrature over the hyperparameter where it is not, and a
 * grid of the model written as a program for the rest. `docs/probprog/ep-priors.md` is the table the atoms illustrate.
 */

import { dict, real } from '../../space/index.ts'
import type { Config } from '../../trace/index.ts'
import { configNumber } from '../inference/method.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { type ExactMarginal, lg, normalPdf, quadratureMarginal } from '../validation/index.ts'
import { sum } from './_util.ts'
import { type AtomProblem, type AtomTolerance, defineAtom, type EpAtom, type RefJoint } from './ep-atoms.ts'

const EXACT: AtomTolerance = {
  meanZ: 1e-6,
  sdRatio: [0.999999, 1.000001],
  ks: 1e-4,
  reason: 'a conjugate model: EP is exact, so its belief equals the closed-form posterior up to rounding',
}

/** The tolerance of a hyperprior setup, where EP projects a skewed posterior of the hyperparameter onto a gamma or a Gaussian. */
const HYPER: AtomTolerance = {
  meanZ: 0.15,
  sdRatio: [0.75, 1.25],
  ks: 0.1,
  reason:
    'the posterior of a hyperparameter is skewed and EP carries one gamma or Gaussian for it: the means are close and the spreads within a quarter',
}

// ----- closed-form marginals -----

const gammaPdf = (x: number, shape: number, rate: number): number =>
  x <= 0 ? 0 : Math.exp(shape * Math.log(rate) - lg(shape) + (shape - 1) * Math.log(x) - rate * x)

/** The gamma distribution as an exact marginal. */
function gammaExact(shape: number, rate: number): ExactMarginal {
  const mean = shape / rate
  const sd = Math.sqrt(shape) / rate
  return { mean, sd, pdf: (x) => gammaPdf(x, shape, rate), range: [Math.max(0, mean - 6 * sd), mean + 8 * sd] }
}

/** The inverse gamma distribution (shape `a`, scale `b`) as an exact marginal. */
function inverseGammaExact(a: number, b: number): ExactMarginal {
  const mean = b / (a - 1)
  const sd = Math.sqrt((b * b) / ((a - 1) ** 2 * (a - 2)))
  return {
    mean,
    sd,
    pdf: (x) => (x <= 0 ? 0 : Math.exp(a * Math.log(b) - lg(a) - (a + 1) * Math.log(x) - b / x)),
    range: [Math.max(1e-6, mean - 3 * sd), mean + 8 * sd],
  }
}

/** A Student t with `nu` degrees of freedom around `loc` with the squared scale `s2`, as an exact marginal. */
function studentExact(nu: number, loc: number, s2: number): ExactMarginal {
  const s = Math.sqrt(s2)
  const logNorm = lg((nu + 1) / 2) - lg(nu / 2) - 0.5 * Math.log(nu * Math.PI) - Math.log(s)
  const sd = Math.sqrt((s2 * nu) / (nu - 2))
  return {
    mean: loc,
    sd,
    pdf: (x) => Math.exp(logNorm - ((nu + 1) / 2) * Math.log(1 + ((x - loc) / s) ** 2 / nu)),
    range: [loc - 7 * sd, loc + 7 * sd],
  }
}

/** A beta distribution as an exact marginal. */
function betaExact(a: number, b: number): ExactMarginal {
  const mean = a / (a + b)
  const sd = Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))
  return { mean, sd, pdf: (x) => betaPdf(x, a, b), range: [0, 1] }
}
const betaPdf = (x: number, a: number, b: number): number =>
  x <= 0 || x >= 1 ? 0 : Math.exp(lg(a + b) - lg(a) - lg(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x))

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
  const top = Math.max(...logs)
  const w0 = logs.map((l) => Math.exp(l - top))
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

/**
 * The exact posterior of a Gaussian's mean and variance under the conjugate normal-inverse-gamma prior (`sigma2` inverse gamma
 * with shape `a` and scale `b`, `mu | sigma2` normal around `m0` with variance `sigma2 / kappa`): a Student t for the mean, an
 * inverse gamma for the variance and their joint on a grid, as the exact references of an atom.
 *
 * @param y - The observations.
 * @param m0 - The prior mean of the mean.
 * @param kappa - The prior observations behind the mean.
 * @param a - The inverse gamma's shape.
 * @param b - The inverse gamma's scale.
 * @returns The marginals of the mean and the variance, and the joint for contours.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { normalInverseGammaPosterior } from 'omai-engine/probprog/ep'
 * normalInverseGammaPosterior([1, 2, 3], 0, 1, 3, 2).mu.mean // => 1.5
 * ```
 */
export function normalInverseGammaPosterior(
  y: readonly number[],
  m0: number,
  kappa: number,
  a: number,
  b: number,
): { readonly mu: ExactMarginal; readonly sigma2: ExactMarginal; readonly joint: RefJoint } {
  const n = y.length
  const ybar = sum(y) / n
  const ss = sum(y.map((v) => (v - ybar) ** 2))
  const kn = kappa + n
  const mn = (kappa * m0 + n * ybar) / kn
  const an = a + n / 2
  const bn = b + ss / 2 + (kappa * n * (ybar - m0) ** 2) / (2 * kn)
  const mu = studentExact(2 * an, mn, bn / (an * kn))
  const sigma2 = inverseGammaExact(an, bn)
  // The joint: N(mu; mn, sigma2 / kn) times the inverse gamma, on a grid for the contours.
  const xs = Array.from({ length: 80 }, (_, i) => mu.mean - 5 * mu.sd + (i * 10 * mu.sd) / 79)
  const lo = Math.max(1e-3, sigma2.range?.[0] ?? 0)
  const hi = sigma2.range?.[1] ?? sigma2.mean * 4
  const ys = Array.from({ length: 80 }, (_, i) => lo + (i * (hi - lo)) / 79)
  const joint: RefJoint = {
    a: 'mu',
    b: 'sigma2',
    x: xs,
    y: ys,
    p: xs.map((x) => ys.map((s2) => normalPdf(x, mn, Math.sqrt(s2 / kn)) * sigma2.pdf(s2))),
  }
  return { mu, sigma2, joint }
}

function nigProblem(k: Config): AtomProblem {
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
    definition,
    data: { y: Y6 },
    equations: [
      `sigma2 ~ InverseGamma(${a}, ${b})   (the variance)`,
      `mu | sigma2 ~ N(${m0}, sigma2 / ${kappa})   (the mean's prior spread scales with the variance)`,
      'y_i ~ N(mu, sigma2), i = 1..6',
      `observe y = (${Y6.join(', ')})`,
    ],
    quantities: [
      { name: 'mu', label: 'mu (the mean)', support: 'real' },
      { name: 'sigma2', label: 'sigma2 (the variance: the hyperparameter of the mean)', support: 'positive' },
    ],
    pair: ['mu', 'sigma2'],
    exact: { mu, sigma2 },
    exactJoint: joint,
  }
}

/**
 * The classic conjugate prior on a Gaussian's mean and variance, written the natural way (an inverse gamma on the variance, a
 * normal on the mean whose variance is that variance over `kappa`): the compiler reads the pair as one normal-gamma variable and
 * the posterior is exact, a Student t for the mean and an inverse gamma for the variance.
 *
 * @index estimator EP atom: a normal-inverse-gamma prior on a Gaussian mean and variance (exact).
 * @example
 * ```ts
 * import { priorNormalInverseGammaAtom } from 'omai-engine/probprog/ep'
 * priorNormalInverseGammaAtom.name // => 'prior-normal-inverse-gamma'
 * ```
 */
export const priorNormalInverseGammaAtom: EpAtom = defineAtom({
  name: 'prior-normal-inverse-gamma',
  title: 'Prior: normal-inverse-gamma on a Gaussian mean and variance',
  what: 'An inverse-gamma prior on the variance of a Gaussian and, given it, a normal prior on its mean with variance sigma2 / kappa: the conjugate pair, one normal-gamma variable and a factor per observation.',
  factors: ['prior (normal-gamma)', 'gaussian-normal-gamma (observed value)'],
  family: 'priors',
  reference: 'exact',
  tolerance: {
    meanZ: 1e-6,
    sdRatio: [0.999999, 1.000001],
    ks: 0.1,
    reason:
      'the posterior is exactly normal-inverse-gamma and EP carries it exactly (the moments agree to rounding); the curves compare the Gaussian and gamma shapes the page draws with the exact Student t and inverse gamma, so the distance is small, not zero',
  },
  knobs: dict({
    priorMean: real(-2, 2, { default: 0, step: 0.1, label: 'prior mean m' }),
    kappa: real(0.1, 5, { default: 0.5, step: 0.1, label: 'kappa (prior observations for the mean)' }),
    shape: real(1, 6, { default: 3, step: 0.1, label: 'shape a of the inverse gamma' }),
    scale: real(0.5, 5, { default: 2, step: 0.1, label: 'scale b of the inverse gamma' }),
  }),
  problem: nigProblem,
})

// ----- a normal prior on a mean with a normal hyperprior on its centre -----

function hierarchicalNormalProblem(k: Config): AtomProblem {
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
    definition,
    data: { y: Y6 },
    equations: [
      `centre ~ N(0, ${s0}²)   (the hyperprior: where the prior sits)`,
      `mu ~ N(centre, ${s1}²)   (the prior on the mean)`,
      'y_i ~ N(mu, 1²), i = 1..6',
      `observe y = (${Y6.join(', ')})`,
    ],
    quantities: [
      { name: 'mu', label: 'mu (the mean)', support: 'real' },
      { name: 'centre', label: 'centre (the hyperparameter)', support: 'real' },
    ],
    pair: ['centre', 'mu'],
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
  }
}

/**
 * A normal prior on a Gaussian mean whose centre has a normal hyperprior of its own: jointly Gaussian, so EP is exact, and the
 * data inform the centre only through the mean.
 *
 * @index estimator EP atom: a normal prior on a mean with a normal hyperprior on its centre (exact).
 * @example
 * ```ts
 * import { priorNormalHyperAtom } from 'omai-engine/probprog/ep'
 * priorNormalHyperAtom.name // => 'prior-normal-hyperprior'
 * ```
 */
export const priorNormalHyperAtom: EpAtom = defineAtom({
  name: 'prior-normal-hyperprior',
  title: 'Prior: normal on a Gaussian mean, normal hyperprior on its centre',
  what: 'A normal prior on a Gaussian mean whose centre is itself normal: a chain of three Gaussians, jointly Gaussian, so the posterior of the mean and of the centre are closed forms.',
  factors: ['prior (gaussian)', 'gaussian with a random mean (difference)', 'gaussian likelihood'],
  family: 'priors',
  reference: 'exact',
  tolerance: EXACT,
  knobs: dict({
    hyperSd: real(0.5, 10, { default: 3, step: 0.1, label: 'sd of the hyperprior' }),
    priorSd: real(0.2, 3, { default: 1, step: 0.1, label: 'sd of the prior given its centre' }),
  }),
  problem: hierarchicalNormalProblem,
})

// ----- an inverse gamma on a variance with a gamma hyperprior on its scale -----

function inverseGammaHyperProblem(k: Config): AtomProblem {
  const [a, c, dd] = [configNumber(k, 'shape'), configNumber(k, 'hyperShape'), configNumber(k, 'hyperRate')]
  // region prior-inverse-gamma-hyperprior
  const definition = defineModel('inverse-gamma-hyperprior', (m) => {
    const scale = m.latent('scale', d.gamma(c, dd))
    const sigma2 = m.latent('sigma2', d.inverseGamma(a, scale))
    m.observe('y', d.normal(1, ops.sqrt(sigma2)), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  return {
    definition,
    data: { y: Y6 },
    equations: [
      `scale ~ Gamma(${c}, ${dd})   (the hyperprior on the inverse gamma's scale)`,
      `sigma2 ~ InverseGamma(${a}, scale)   (the variance)`,
      'y_i ~ N(1, sigma2), i = 1..6   (the mean is known)',
      `observe y = (${Y6.join(', ')})`,
    ],
    quantities: [
      { name: 'sigma2', label: 'sigma2 (the variance)', support: 'positive' },
      { name: 'scale', label: 'scale (the hyperparameter)', support: 'positive' },
    ],
    pair: ['sigma2', 'scale'],
    grid: { points: 61, maxRefinements: 0 },
  }
}

/**
 * A gamma hyperprior on the scale of an inverse-gamma variance: the variance is a gamma precision whose rate is the
 * hyperparameter, the `gamma-rate` hierarchy, and the Gaussian observations are the gamma-precision factor.
 *
 * @index estimator EP atom: an inverse-gamma prior on a variance with a gamma hyperprior on its scale.
 * @example
 * ```ts
 * import { priorInverseGammaHyperAtom } from 'omai-engine/probprog/ep'
 * priorInverseGammaHyperAtom.family // => 'priors'
 * ```
 */
export const priorInverseGammaHyperAtom: EpAtom = defineAtom({
  name: 'prior-inverse-gamma-hyperprior',
  title: 'Prior: inverse-gamma on a Gaussian variance, gamma hyperprior on its scale',
  what: 'An inverse-gamma prior on a variance whose scale is gamma distributed, six Gaussian observations around a known mean: the variance is a gamma precision with a random rate (the gamma-rate hierarchy).',
  factors: ['prior (gamma)', 'gamma-rate (a gamma with a gamma rate)', 'gaussian-gamma-precision (observed value)'],
  family: 'priors',
  reference: 'grid',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.75, 1.3],
    ks: 0.18,
    reason:
      'the posterior of the variance is heavy-tailed: EP carries a gamma for its reciprocal, which matches the mean and the scale hyperparameter and leaves the variance an inverse gamma whose spread is about a fifth too wide',
  },
  knobs: dict({
    shape: real(1, 6, { default: 3, step: 0.1, label: 'shape a of the inverse gamma' }),
    hyperShape: real(0.5, 6, { default: 2, step: 0.1, label: 'shape of the hyperprior' }),
    hyperRate: real(0.2, 4, { default: 1, step: 0.1, label: 'rate of the hyperprior' }),
  }),
  problem: inverseGammaHyperProblem,
})

// ----- a gamma on a Poisson rate with a gamma hyperprior on the prior's rate -----

function gammaPoissonHyperProblem(k: Config): AtomProblem {
  const [a, c, dd] = [configNumber(k, 'shape'), configNumber(k, 'hyperShape'), configNumber(k, 'hyperRate')]
  // region prior-gamma-poisson-hyperprior
  const definition = defineModel('gamma-poisson-hyperprior', (m) => {
    const prior = m.latent('priorRate', d.gamma(c, dd))
    const rate = m.latent('rate', d.gamma(a, prior))
    m.observe('y', d.poisson(rate), m.data('y', { dims: ['i'], support: 'count' }))
  })
  // endregion
  return {
    definition,
    data: { y: COUNTS },
    equations: [
      `priorRate ~ Gamma(${c}, ${dd})   (the hyperprior: how far the rate can be from zero)`,
      `rate ~ Gamma(${a}, priorRate)   (the prior on the Poisson rate)`,
      `y_i ~ Poisson(rate), i = 1..${COUNTS.length}`,
      `observe y = (${COUNTS.join(', ')})`,
    ],
    quantities: [
      { name: 'rate', label: 'rate (the Poisson rate)', support: 'positive' },
      { name: 'priorRate', label: 'priorRate (the hyperparameter)', support: 'positive' },
    ],
    pair: ['rate', 'priorRate'],
    grid: { points: 61, maxRefinements: 0 },
  }
}

/**
 * A gamma prior on a Poisson rate whose own rate has a gamma hyperprior: the `gamma-rate` hierarchy under the conjugate
 * Poisson factor.
 *
 * @index estimator EP atom: a gamma prior on a Poisson rate with a gamma hyperprior on the prior's rate.
 * @example
 * ```ts
 * import { priorGammaPoissonHyperAtom } from 'omai-engine/probprog/ep'
 * priorGammaPoissonHyperAtom.expected // => 'runs'
 * ```
 */
export const priorGammaPoissonHyperAtom: EpAtom = defineAtom({
  name: 'prior-gamma-poisson-hyperprior',
  title: 'Prior: gamma on a Poisson rate, gamma hyperprior on its rate',
  what: 'A gamma prior on a Poisson rate whose rate parameter is gamma distributed, four counts: the gamma-rate hierarchy under the conjugate Poisson factor.',
  factors: ['prior (gamma)', 'gamma-rate', 'poisson likelihood (gamma-poisson)'],
  family: 'priors',
  reference: 'grid',
  tolerance: HYPER,
  knobs: dict({
    shape: real(0.5, 8, { default: 2, step: 0.1, label: 'shape of the rate prior' }),
    hyperShape: real(0.5, 6, { default: 2, step: 0.1, label: 'shape of the hyperprior' }),
    hyperRate: real(0.2, 4, { default: 2, step: 0.1, label: 'rate of the hyperprior' }),
  }),
  problem: gammaPoissonHyperProblem,
})

// ----- a log-normal prior on a Poisson rate -----

function logNormalPoissonProblem(k: Config): AtomProblem {
  const [mu0, s0] = [configNumber(k, 'logMean'), configNumber(k, 'logSd')]
  // region prior-lognormal-poisson
  const definition = defineModel('lognormal-poisson', (m) => {
    const rate = m.latent('rate', d.logNormal(mu0, s0))
    m.observe('y', d.poisson(rate), m.data('y', { dims: ['i'], support: 'count' }))
  })
  // endregion
  return {
    definition,
    data: { y: COUNTS },
    equations: [
      `rate ~ LogNormal(${mu0}, ${s0}²)   (log rate ~ N(${mu0}, ${s0}²))`,
      `y_i ~ Poisson(rate), i = 1..${COUNTS.length}`,
      `observe y = (${COUNTS.join(', ')})`,
    ],
    quantities: [{ name: 'rate', label: 'rate (the Poisson rate)', support: 'positive' }],
    grid: { points: 201, maxRefinements: 0 },
  }
}

/**
 * A log-normal prior on a Poisson rate: the prior is a Gaussian on the log rate (projected once, carried on its log) and the
 * count is read through the log link, the numeric likelihood of the registry.
 *
 * @index estimator EP atom: a log-normal prior on a Poisson rate.
 * @example
 * ```ts
 * import { priorLogNormalPoissonAtom } from 'omai-engine/probprog/ep'
 * priorLogNormalPoissonAtom.name // => 'prior-lognormal-poisson'
 * ```
 */
export const priorLogNormalPoissonAtom: EpAtom = defineAtom({
  name: 'prior-lognormal-poisson',
  title: 'Prior: log-normal on a Poisson rate',
  what: 'A log-normal prior on a Poisson rate and four counts: a Gaussian on the log rate and the numeric Poisson likelihood with a log link.',
  factors: ['prior (gaussian, on the log rate)', 'numeric likelihood: poisson with an exp link'],
  family: 'priors',
  reference: 'grid',
  tolerance: { ...HYPER, meanZ: 0.2 },
  knobs: dict({
    logMean: real(-1, 3, { default: 1, step: 0.1, label: 'mean of the log rate' }),
    logSd: real(0.2, 2, { default: 0.8, step: 0.1, label: 'sd of the log rate' }),
  }),
  problem: logNormalPoissonProblem,
})

function logNormalPoissonHyperProblem(k: Config): AtomProblem {
  const [s0, s1] = [configNumber(k, 'hyperSd'), configNumber(k, 'logSd')]
  // region prior-lognormal-poisson-hyperprior
  const definition = defineModel('lognormal-poisson-hyperprior', (m) => {
    const centre = m.latent('centre', d.normal(1, s0))
    const logRate = m.latent('logRate', d.normal(centre, s1))
    m.observe('y', d.poisson(ops.exp(logRate)), m.data('y', { dims: ['i'], support: 'count' }))
  })
  // endregion
  return {
    definition,
    data: { y: COUNTS },
    equations: [
      `centre ~ N(1, ${s0}²)   (the hyperprior on the log rate's centre)`,
      `logRate ~ N(centre, ${s1}²)   (so rate = exp(logRate) is log-normal)`,
      `y_i ~ Poisson(exp(logRate)), i = 1..${COUNTS.length}`,
      `observe y = (${COUNTS.join(', ')})`,
    ],
    quantities: [
      { name: 'logRate', label: 'logRate (the log of the rate)', support: 'real' },
      { name: 'centre', label: 'centre (the hyperparameter)', support: 'real' },
    ],
    pair: ['centre', 'logRate'],
    grid: { points: 61, maxRefinements: 0 },
  }
}

/**
 * A log-normal prior on a Poisson rate written on the log scale, with a normal hyperprior on the centre of the log rate.
 *
 * @index estimator EP atom: a log-normal prior on a Poisson rate with a normal hyperprior on its log centre.
 * @example
 * ```ts
 * import { priorLogNormalPoissonHyperAtom } from 'omai-engine/probprog/ep'
 * priorLogNormalPoissonHyperAtom.family // => 'priors'
 * ```
 */
export const priorLogNormalPoissonHyperAtom: EpAtom = defineAtom({
  name: 'prior-lognormal-poisson-hyperprior',
  title: 'Prior: log-normal on a Poisson rate, normal hyperprior on its log centre',
  what: 'The log of a Poisson rate is normal around a centre that is itself normal: two Gaussians and the numeric Poisson likelihood with a log link.',
  factors: ['prior (gaussian)', 'gaussian with a random mean', 'numeric likelihood: poisson with an exp link'],
  family: 'priors',
  reference: 'grid',
  tolerance: { ...HYPER, meanZ: 0.2 },
  knobs: dict({
    hyperSd: real(0.3, 4, { default: 1, step: 0.1, label: 'sd of the hyperprior' }),
    logSd: real(0.2, 2, { default: 0.5, step: 0.1, label: 'sd of the log rate given its centre' }),
  }),
  problem: logNormalPoissonHyperProblem,
})

// ----- a gamma on the rate of an exponential -----

const WAITS = [0.8, 1.9, 0.3, 1.2, 0.6]

function gammaExponentialProblem(k: Config): AtomProblem {
  const [a, b] = [configNumber(k, 'shape'), configNumber(k, 'rate')]
  // region prior-gamma-exponential
  const definition = defineModel('gamma-exponential', (m) => {
    const rate = m.latent('rate', d.gamma(a, b))
    m.observe('y', d.exponential(rate), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  return {
    definition,
    data: { y: WAITS },
    equations: [
      `rate ~ Gamma(${a}, ${b})`,
      `y_i ~ Exponential(rate), i = 1..${WAITS.length}   (waiting times)`,
      `observe y = (${WAITS.join(', ')})`,
    ],
    quantities: [{ name: 'rate', label: 'rate (of the waiting times)', support: 'positive' }],
    exact: { rate: gammaExact(a + WAITS.length, b + sum(WAITS)) },
  }
}

/**
 * A gamma prior on the rate of exponential waiting times: the likelihood is a gamma in the rate, so the posterior is the gamma
 * with the shape plus the count and the rate plus the total time.
 *
 * @index estimator EP atom: a gamma prior on an exponential rate (conjugate, exact).
 * @example
 * ```ts
 * import { priorGammaExponentialAtom } from 'omai-engine/probprog/ep'
 * priorGammaExponentialAtom.reference // => 'exact'
 * ```
 */
export const priorGammaExponentialAtom: EpAtom = defineAtom({
  name: 'prior-gamma-exponential',
  title: 'Prior: gamma on an exponential rate',
  what: 'A gamma prior on the rate of exponential waiting times and five observed times: a conjugate gamma message from each observation.',
  factors: ['prior (gamma)', 'gamma-observed (an exponential value with a gamma rate)'],
  family: 'priors',
  reference: 'exact',
  tolerance: {
    ...EXACT,
    ks: 5e-3,
    reason: `${EXACT.reason} (the curve compared is the matched gamma, the exact posterior)`,
  },
  knobs: dict({
    shape: real(0.2, 10, { default: 2, step: 0.1, label: 'prior shape' }),
    rate: real(0.05, 5, { default: 1, step: 0.05, label: 'prior rate' }),
  }),
  problem: gammaExponentialProblem,
})

// ----- a normal prior on the log scale of a Weibull, with a hyperprior on its centre -----

const LIFETIMES = [2.1, 0.7, 1.4, 3.0, 1.1]

function weibullProblem(k: Config): AtomProblem {
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
    definition,
    data: { y: LIFETIMES },
    equations: [
      `centre ~ N(0, ${s0}²)   (the hyperprior)`,
      `logScale ~ N(centre, ${s1}²)   (the Weibull scale is log-normal)`,
      `y_i ~ Weibull(${shape}, exp(logScale)), i = 1..${LIFETIMES.length}   (lifetimes)`,
      `observe y = (${LIFETIMES.join(', ')})`,
    ],
    quantities: [
      { name: 'logScale', label: 'logScale (the log of the Weibull scale)', support: 'real' },
      { name: 'centre', label: 'centre (the hyperparameter)', support: 'real' },
    ],
    pair: ['centre', 'logScale'],
    grid: { points: 61, maxRefinements: 0 },
  }
}

/**
 * A normal prior on the log scale of a Weibull lifetime (so a log-normal scale) with a normal hyperprior on its centre: the
 * numeric Weibull likelihood of the registry with a log link.
 *
 * @index estimator EP atom: a normal prior on a Weibull log scale with a normal hyperprior on its centre.
 * @example
 * ```ts
 * import { priorNormalWeibullAtom } from 'omai-engine/probprog/ep'
 * priorNormalWeibullAtom.name // => 'prior-normal-weibull-scale'
 * ```
 */
export const priorNormalWeibullAtom: EpAtom = defineAtom({
  name: 'prior-normal-weibull-scale',
  title: 'Prior: normal on a Weibull log scale, normal hyperprior on its centre',
  what: 'The log of a Weibull scale is normal around a normal centre, five observed lifetimes with a known shape: two Gaussians and the numeric Weibull likelihood.',
  factors: ['prior (gaussian)', 'gaussian with a random mean', 'numeric likelihood: weibull with a log link'],
  family: 'priors',
  reference: 'grid',
  tolerance: { ...HYPER, meanZ: 0.2 },
  knobs: dict({
    shape: real(0.5, 4, { default: 1.5, step: 0.1, label: 'Weibull shape (known)' }),
    hyperSd: real(0.3, 4, { default: 1, step: 0.1, label: 'sd of the hyperprior' }),
    logSd: real(0.2, 2, { default: 0.5, step: 0.1, label: 'sd of the log scale given its centre' }),
  }),
  problem: weibullProblem,
})

// ----- a beta prior on a geometric success probability -----

const TRIALS = [3, 5, 2, 4]

function betaGeometricProblem(k: Config): AtomProblem {
  const [a, b] = [configNumber(k, 'a'), configNumber(k, 'b')]
  // region prior-beta-geometric
  const definition = defineModel('beta-geometric', (m) => {
    const p = m.latent('p', d.beta(a, b))
    m.observe('y', d.geometric(p), m.data('y', { dims: ['i'], support: 'count' }))
  })
  // endregion
  return {
    definition,
    data: { y: TRIALS },
    equations: [
      `p ~ Beta(${a}, ${b})`,
      `y_i ~ Geometric(p), i = 1..${TRIALS.length}   (trials up to the first success)`,
      `observe y = (${TRIALS.join(', ')})`,
    ],
    quantities: [{ name: 'p', label: 'p (the success probability)', support: 'unit' }],
    exact: { p: betaExact(a + TRIALS.length, b + sum(TRIALS) - TRIALS.length) },
  }
}

/**
 * A beta prior on the success probability of a geometric count: the likelihood `p (1 - p)^(y - 1)` is a beta in `p`, exact.
 *
 * @index estimator EP atom: a beta prior on a geometric success probability (conjugate, exact).
 * @example
 * ```ts
 * import { priorBetaGeometricAtom } from 'omai-engine/probprog/ep'
 * priorBetaGeometricAtom.name // => 'prior-beta-geometric'
 * ```
 */
export const priorBetaGeometricAtom: EpAtom = defineAtom({
  name: 'prior-beta-geometric',
  title: 'Prior: beta on a geometric success probability',
  what: 'A beta prior on the success probability and four geometric counts (trials to the first success): the likelihood is a beta in the probability.',
  factors: ['prior (beta)', 'negative-binomial-beta (a geometric count with a beta probability)'],
  family: 'priors',
  reference: 'exact',
  tolerance: {
    ...EXACT,
    ks: 5e-3,
    reason: `${EXACT.reason} (the curve compared is the matched beta, the exact posterior)`,
  },
  knobs: dict({
    a: real(0.2, 20, { default: 2, step: 0.1, label: 'prior a' }),
    b: real(0.2, 20, { default: 2, step: 0.1, label: 'prior b' }),
  }),
  problem: betaGeometricProblem,
})

// ----- a beta with a random strength: gamma and log-normal hyperpriors -----

function betaStrengthProblem(kind: 'gamma' | 'lognormal', k: Config): AtomProblem {
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
    definition,
    data: { y: FLIPS },
    equations: [
      kind === 'gamma'
        ? `strength ~ Gamma(${a}, ${b})   (the hyperprior)`
        : `strength ~ LogNormal(${a}, ${b}²)   (the hyperprior)`,
      'p ~ Beta(strength, strength)   (a symmetric beta: its strength is random)',
      `y_i ~ Bernoulli(p), i = 1..${FLIPS.length}`,
      `observe y = (${FLIPS.join(', ')})`,
    ],
    quantities: [
      { name: 'p', label: 'p (the probability)', support: 'unit' },
      { name: 'strength', label: 'strength (the hyperparameter)', support: 'positive' },
    ],
    exact: { p: exact.p[0] as ExactMarginal, strength: exact.alpha },
  }
}

const STRENGTH_TOLERANCE: AtomTolerance = {
  meanZ: 0.1,
  sdRatio: [0.8, 1.2],
  ks: 0.08,
  reason:
    'the posterior of the strength is skewed and EP carries a gamma for it (a Gaussian on its log for the log-normal); the probability is the matched beta of the mixture over the strength',
}

/**
 * A symmetric beta prior on a Bernoulli probability whose strength has a gamma hyperprior: the numeric `dirichlet-concentration`
 * factor (a grid over the log strength). The exact answer is a quadrature over the strength.
 *
 * @index estimator EP atom: a beta prior on a probability with a gamma hyperprior on its strength.
 * @example
 * ```ts
 * import { priorBetaStrengthGammaAtom } from 'omai-engine/probprog/ep'
 * priorBetaStrengthGammaAtom.family // => 'priors'
 * ```
 */
export const priorBetaStrengthGammaAtom: EpAtom = defineAtom({
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
  knobs: dict({
    shape: real(0.5, 8, { default: 2, step: 0.1, label: 'shape of the strength prior' }),
    rate: real(0.1, 4, { default: 1, step: 0.1, label: 'rate of the strength prior' }),
  }),
  problem: (k) => betaStrengthProblem('gamma', k),
})

/**
 * The same with a log-normal hyperprior on the strength, carried on its log (a Gaussian site in the factor).
 *
 * @index estimator EP atom: a beta prior on a probability with a log-normal hyperprior on its strength.
 * @example
 * ```ts
 * import { priorBetaStrengthLogNormalAtom } from 'omai-engine/probprog/ep'
 * priorBetaStrengthLogNormalAtom.family // => 'priors'
 * ```
 */
export const priorBetaStrengthLogNormalAtom: EpAtom = defineAtom({
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
  tolerance: { ...STRENGTH_TOLERANCE, ks: 0.12 },
  knobs: dict({
    logMean: real(-1, 3, { default: 0.7, step: 0.1, label: 'mean of the log strength' }),
    logSd: real(0.2, 2, { default: 0.8, step: 0.1, label: 'sd of the log strength' }),
  }),
  problem: (k) => betaStrengthProblem('lognormal', k),
})

// ----- a Dirichlet prior on categorical probabilities -----

function dirichletCategoricalProblem(k: Config): AtomProblem {
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
    definition,
    data: { y: CATEGORIES },
    equations: [
      `p ~ Dirichlet(${a}, ${a}, ${a})   (probabilities of three categories)`,
      `y_i ~ Categorical(p), i = 1..${CATEGORIES.length}`,
      `observe y = (${CATEGORIES.join(', ')})   (counts ${counts.join(', ')})`,
    ],
    quantities: counts.map((_, c) => ({ name: `p[${c}]`, label: `p of category ${c}`, support: 'unit' as const })),
    exact: Object.fromEntries(counts.map((n, c) => [`p[${c}]`, betaExact(a + n, 2 * a + N - n)])),
  }
}

/**
 * A Dirichlet prior on the probabilities of a categorical variable: each marginal of the posterior is the beta of its count.
 *
 * @index estimator EP atom: a Dirichlet prior on categorical probabilities (conjugate, exact).
 * @example
 * ```ts
 * import { priorDirichletCategoricalAtom } from 'omai-engine/probprog/ep'
 * priorDirichletCategoricalAtom.name // => 'prior-dirichlet-categorical'
 * ```
 */
export const priorDirichletCategoricalAtom: EpAtom = defineAtom({
  name: 'prior-dirichlet-categorical',
  title: 'Prior: Dirichlet on categorical probabilities',
  what: 'A symmetric Dirichlet prior on three category probabilities and seven categorical observations: the Dirichlet-categorical conjugate pair.',
  factors: ['prior (dirichlet)', 'discrete likelihood (discrete-dirichlet)'],
  family: 'priors',
  reference: 'exact',
  tolerance: {
    ...EXACT,
    ks: 5e-3,
    reason: `${EXACT.reason} (the curve compared is the matched beta, the exact marginal)`,
  },
  knobs: dict({ concentration: real(0.2, 10, { default: 1, step: 0.1, label: 'prior concentration' }) }),
  problem: dirichletCategoricalProblem,
})

function dirichletStrengthProblem(k: Config): AtomProblem {
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
    definition,
    data: { y: CATEGORIES },
    equations: [
      `strength ~ Gamma(${a}, ${b})   (the hyperprior)`,
      'p ~ Dirichlet(strength, strength, strength)   (a symmetric Dirichlet with a random concentration)',
      `y_i ~ Categorical(p), i = 1..${CATEGORIES.length}`,
      `observe y = (${CATEGORIES.join(', ')})   (counts ${counts.join(', ')})`,
    ],
    quantities: [
      { name: 'strength', label: 'strength (the hyperparameter)', support: 'positive' },
      ...counts.map((_, c) => ({ name: `p[${c}]`, label: `p of category ${c}`, support: 'unit' as const })),
    ],
    exact: { strength: exact.alpha, ...Object.fromEntries(exact.p.map((m, c) => [`p[${c}]`, m])) },
  }
}

/**
 * A symmetric Dirichlet prior on categorical probabilities whose concentration has a gamma hyperprior: the numeric
 * `dirichlet-concentration` factor with three components.
 *
 * @index estimator EP atom: a Dirichlet prior on categorical probabilities with a gamma hyperprior on its concentration.
 * @example
 * ```ts
 * import { priorDirichletStrengthAtom } from 'omai-engine/probprog/ep'
 * priorDirichletStrengthAtom.family // => 'priors'
 * ```
 */
export const priorDirichletStrengthAtom: EpAtom = defineAtom({
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
  problem: dirichletStrengthProblem,
})

// ----- a logit-normal prior on a Bernoulli probability -----

function logitNormalProblem(k: Config): AtomProblem {
  const [s0, s1] = [configNumber(k, 'hyperSd'), configNumber(k, 'logitSd')]
  // region prior-logit-normal-bernoulli
  const definition = defineModel('logit-normal-bernoulli', (m) => {
    const centre = m.latent('centre', d.normal(0, s0))
    const logit = m.latent('logit', d.normal(centre, s1))
    m.observe('y', d.bernoulli({ logits: logit }), m.data('y', { dims: ['i'], support: 'binary' }))
  })
  // endregion
  return {
    definition,
    data: { y: FLIPS },
    equations: [
      `centre ~ N(0, ${s0}²)   (the hyperprior on the logit's centre)`,
      `logit ~ N(centre, ${s1}²)   (p = sigmoid(logit) is logit-normal)`,
      `y_i ~ Bernoulli(sigmoid(logit)), i = 1..${FLIPS.length}`,
      `observe y = (${FLIPS.join(', ')})`,
    ],
    quantities: [
      { name: 'logit', label: 'logit (the log odds of p)', support: 'real' },
      { name: 'centre', label: 'centre (the hyperparameter)', support: 'real' },
    ],
    pair: ['centre', 'logit'],
    grid: { points: 61, maxRefinements: 0 },
  }
}

/**
 * A logit-normal prior on a Bernoulli probability (a normal on the log odds) with a normal hyperprior on its centre: the logistic
 * factor on a Gaussian predictor.
 *
 * @index estimator EP atom: a logit-normal prior on a Bernoulli probability with a normal hyperprior on its centre.
 * @example
 * ```ts
 * import { priorLogitNormalAtom } from 'omai-engine/probprog/ep'
 * priorLogitNormalAtom.family // => 'priors'
 * ```
 */
export const priorLogitNormalAtom: EpAtom = defineAtom({
  name: 'prior-logit-normal-bernoulli',
  title: 'Prior: logit-normal on a Bernoulli probability, normal hyperprior on its centre',
  what: 'The log odds of a Bernoulli probability are normal around a normal centre, seven flips: two Gaussians and the logistic factor.',
  factors: ['prior (gaussian)', 'gaussian with a random mean', 'logistic likelihood'],
  family: 'priors',
  reference: 'grid',
  tolerance: { ...HYPER, meanZ: 0.2 },
  knobs: dict({
    hyperSd: real(0.3, 4, { default: 1, step: 0.1, label: 'sd of the hyperprior' }),
    logitSd: real(0.2, 3, { default: 1, step: 0.1, label: 'sd of the log odds given the centre' }),
  }),
  problem: logitNormalProblem,
})

// ----- a uniform prior on a Gaussian mean -----

function uniformProblem(k: Config): AtomProblem {
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
    definition,
    data: { y: Y6 },
    equations: [`mu ~ Uniform(${lo}, ${hi})`, 'y_i ~ N(mu, 1²), i = 1..6', `observe y = (${Y6.join(', ')})`],
    quantities: [{ name: 'mu', label: 'mu (the mean)', support: 'real' }],
    exact: { mu: quadratureMarginal((x) => -0.5 * n * (x - ybar) ** 2, lo, hi) },
  }
}

/**
 * A uniform prior on a Gaussian mean: a prior with no message family, kept as the exact numeric `prior-site` (its message is the
 * moment match against the current cavity at every sweep), so the posterior is the truncated Gaussian approximated by a Gaussian.
 *
 * @index estimator EP atom: a uniform prior on a Gaussian mean (exact numeric site).
 * @example
 * ```ts
 * import { priorUniformAtom } from 'omai-engine/probprog/ep'
 * priorUniformAtom.name // => 'prior-uniform-mean'
 * ```
 */
export const priorUniformAtom: EpAtom = defineAtom({
  name: 'prior-uniform-mean',
  title: 'Prior: uniform on a Gaussian mean',
  what: 'A uniform prior on a Gaussian mean with six observations: the density has no message family, so it is carried as an exact numeric site that EP matches against the current cavity at every sweep.',
  factors: ['numeric prior site (uniform, exact)', 'gaussian likelihood'],
  family: 'priors',
  reference: 'exact',
  tolerance: {
    meanZ: 0.25,
    sdRatio: [0.7, 1.3],
    ks: 0.15,
    reason:
      'the uniform prior is replaced once by the Gaussian with its mean and variance, so a bound that cuts into the posterior is not respected; where the data sit well inside the bounds the two agree',
  },
  knobs: dict({
    low: real(-6, 1, { default: -3, step: 0.5, label: 'lower bound' }),
    high: real(1.5, 8, { default: 3, step: 0.5, label: 'upper bound' }),
  }),
  problem: uniformProblem,
})

// ----- a log-normal prior on a Gaussian scale with a hyperprior on its centre -----

const SMALL = [0.5, -1.2, 2.0, 0.3]

function logNormalScaleProblem(k: Config): AtomProblem {
  const [s0, s1] = [configNumber(k, 'hyperSd'), configNumber(k, 'logSd')]
  // region prior-lognormal-scale-hyperprior
  const definition = defineModel('lognormal-scale-hyperprior', (m) => {
    const centre = m.latent('centre', d.normal(0, s0))
    const logScale = m.latent('logScale', d.normal(centre, s1))
    m.observe('y', d.normal(0, ops.exp(logScale)), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  return {
    definition,
    data: { y: SMALL },
    equations: [
      `centre ~ N(0, ${s0}²)   (the hyperprior on the log scale's centre)`,
      `logScale ~ N(centre, ${s1}²)   (the standard deviation exp(logScale) is log-normal)`,
      `y_i ~ N(0, exp(logScale)²), i = 1..${SMALL.length}`,
      `observe y = (${SMALL.join(', ')})`,
    ],
    quantities: [
      { name: 'logScale', label: 'logScale (the log of the sd)', support: 'real' },
      { name: 'centre', label: 'centre (the hyperparameter)', support: 'real' },
    ],
    pair: ['centre', 'logScale'],
    grid: { points: 61, maxRefinements: 0 },
  }
}

/**
 * A log-normal prior on the standard deviation of a Gaussian (a normal on its log) with a normal hyperprior on the centre: the
 * `gaussian-log-scale` factor, a one-dimensional quadrature in the log scale.
 *
 * @index estimator EP atom: a log-normal prior on a Gaussian scale with a normal hyperprior on its log centre.
 * @example
 * ```ts
 * import { priorLogNormalScaleAtom } from 'omai-engine/probprog/ep'
 * priorLogNormalScaleAtom.family // => 'priors'
 * ```
 */
export const priorLogNormalScaleAtom: EpAtom = defineAtom({
  name: 'prior-lognormal-scale-hyperprior',
  title: 'Prior: log-normal on a Gaussian scale, normal hyperprior on its log centre',
  what: 'The log of a Gaussian standard deviation is normal around a normal centre, four zero-mean observations: two Gaussians and the Gaussian with an exp(h) scale.',
  factors: ['prior (gaussian)', 'gaussian with a random mean', 'gaussian-log-scale'],
  family: 'priors',
  reference: 'grid',
  tolerance: { ...HYPER, meanZ: 0.2 },
  knobs: dict({
    hyperSd: real(0.3, 4, { default: 1, step: 0.1, label: 'sd of the hyperprior' }),
    logSd: real(0.2, 2, { default: 0.5, step: 0.1, label: 'sd of the log scale given its centre' }),
  }),
  problem: logNormalScaleProblem,
})

// ----- a beta with two random shapes -----

function betaTwoShapesProblem(k: Config): AtomProblem {
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
    definition,
    data: { y: FLIPS },
    equations: [
      `a ~ LogNormal(0.5, ${s}²), b ~ LogNormal(0.5, ${s}²)   (the two shapes, each with its own hyperprior)`,
      'p ~ Beta(a, b)',
      `y_i ~ Bernoulli(p), i = 1..${FLIPS.length}`,
      `observe y = (${FLIPS.join(', ')})`,
    ],
    quantities: [
      { name: 'p', label: 'p (the probability)', support: 'unit' },
      { name: 'a', label: 'a (first shape)', support: 'positive' },
      { name: 'b', label: 'b (second shape)', support: 'positive' },
    ],
    grid: { points: 25, maxRefinements: 0 },
  }
}

/**
 * A beta prior with two random shapes, each with its own log-normal hyperprior: the tilted distribution is two-dimensional (a
 * grid over both shapes), which the one-dimensional concentration factor does not cover; the `beta-two-shapes` factor does, with a
 * grid in `(log a, log b)` and the conjugate update of `p` at each node.
 *
 * @index estimator EP atom: a beta prior on a probability with two random shapes.
 * @example
 * ```ts
 * import { priorBetaTwoShapesAtom } from 'omai-engine/probprog/ep'
 * priorBetaTwoShapesAtom.expected // => 'runs'
 * ```
 */
export const priorBetaTwoShapesAtom: EpAtom = defineAtom({
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
  tolerance: { ...HYPER, meanZ: 0.25 },
  knobs: dict({ logSd: real(0.2, 2, { default: 0.7, step: 0.1, label: 'sd of each log shape' }) }),
  problem: betaTwoShapesProblem,
})
