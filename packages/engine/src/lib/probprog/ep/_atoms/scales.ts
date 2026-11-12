/**
 * The setups of expectation propagation with a random scale: a Gaussian with an unknown precision, a shared precision over a population, a precision multiplied by a known factor, half-Cauchy and half-normal scales, and the same with the unknowns learned from simulated data. The exact answers are quadratures over the precision, the effects being Gaussian given it.
 */

import { foldIn, key, normal } from '../../../random/index.ts'
import { dict, int, real } from '../../../space/index.ts'
import { toFlat } from '../../../tensor/index.ts'
import type { Config } from '../../../trace/index.ts'
import { configNumber } from '../../inference/method.ts'
import * as d from '../../model/dists/index.ts'
import { defineModel, ops } from '../../model/index.ts'
import {
  type ConditionalMarginal,
  type ExactMarginal,
  gaussianMixtureMarginal,
  hyperPosterior,
  logWeights,
  normalPdf,
} from '../../validation/index.ts'
import { defineAtom } from '../_define.ts'
import { recovery, tolerance } from '../_tolerance.ts'
import type { AtomParts, EpAtom } from '../atom-types.ts'
import { simulationKnobs } from './conjugate.ts'

const Y6 = [1.2, 0.4, 1.9, 1.1, 0.7, 1.5]

function precisionProblem(k: Config): AtomParts {
  const a = configNumber(k, 'shape')
  const b = configNumber(k, 'rate')
  // region gamma-precision
  const definition = defineModel('gamma-precision', (m) => {
    const tau = m.latent('tau', d.gamma(a, b))
    const mu = m.latent('mu', d.normal(0, 10))
    const observed = m.data('y', { dims: ['i'], role: 'response' })
    m.observe('y', d.normal(mu, ops.div(1, ops.sqrt(tau))), observed)
  })
  // endregion
  return {
    build: {
      definition,
      data: { y: Y6 },
      quantities: [
        { name: 'mu', label: 'mu', support: 'real' },
        { name: 'tau', label: 'tau (precision)', support: 'positive' },
      ],
    },
    reference: {
      grid: { points: 61, maxRefinements: 0 },
    },
    present: {
      equations: [
        `tau ~ Gamma(${a}, ${b})   (a precision)`,
        'mu ~ N(0, 10²)',
        'y_i ~ N(mu, 1 / tau), i = 1..6',
        `observe y = (${Y6.join(', ')})`,
      ],
      pair: ['mu', 'tau'],
      figure: {
        caption:
          'The six observations; the line is mu and the band two standard deviations, 2 / sqrt(tau), either side.',
        xLabel: 'observation',
        yLabel: 'y',
        xLimits: [0, Y6.length + 1],
        points: [{ name: 'observations', x: Y6.map((_, i) => i + 1), y: [...Y6] }],
        answer: (m) => {
          const mu = m['mu'] ?? Number.NaN
          const w = 2 / Math.sqrt(m['tau'] ?? Number.NaN)
          const x = [0, Y6.length + 1]
          return [
            { name: 'mu', x, y: [mu, mu] },
            { name: 'mu + 2 sd', x, y: [mu + w, mu + w] },
            { name: 'mu - 2 sd', x, y: [mu - w, mu - w] },
          ]
        },
      },
    },
  }
}

/**
 * A Gaussian with a gamma prior on its precision, observed six times: the gamma-precision factor on an observed value,
 * with a skewed, correlated posterior of the mean and the precision.
 */
const gammaPrecisionAtom: EpAtom = defineAtom({
  name: 'gamma-precision',
  title: 'Gamma precision',
  what: 'A Gaussian mean mu and a gamma-distributed precision tau, six observations y_i ~ N(mu, 1/tau): the gamma-precision factor with an observed value, a Student-t-like joint posterior.',
  factors: ['prior (gaussian)', 'prior (gamma)', 'gaussian-gamma-precision (observed value)'],
  family: 'scales',
  reference: 'grid',
  tolerance: tolerance(
    'tight',
    'the factor is not Gaussian in mu and tau together, so EP projects onto a Gaussian and a gamma and the beliefs are close, not exact',
  ),
  evidence: {
    nats: 0.05,
    reason: 'a Gaussian with a gamma precision: the gamma is projected from a Student-t shaped tilt',
  },
  knobs: dict({
    shape: real(0.5, 10, { default: 2, step: 0.1, label: 'prior shape of tau' }),
    rate: real(0.1, 10, { default: 1, step: 0.1, label: 'prior rate of tau' }),
  }),
  parts: precisionProblem,
})

/** The exact posterior of the precision and of each effect, by quadrature over the precision (the effects are Gaussian given it). */
function scaledExact(
  y: readonly number[],
  mult: readonly number[],
  s: number,
  a: number,
  b: number,
): { tau: ExactMarginal; theta: ExactMarginal[] } {
  const top = (a + 12 * Math.sqrt(a) + 20) / b
  const M = 6000
  const dt = top / M
  const taus = Array.from({ length: M }, (_, i) => (i + 0.5) * dt)
  const logPost = (t: number): number =>
    (a - 1) * Math.log(t) -
    b * t +
    y.reduce((acc, v, j) => {
      const var0 = s * s + 1 / (t * (mult[j] ?? 1))
      return acc - 0.5 * Math.log(var0) - (v * v) / (2 * var0)
    }, 0)
  const logs = taus.map(logPost)
  const { weights: w, top: peak } = logWeights(logs)
  const z = w.reduce((p, q) => p + q, 0)
  const pw = w.map((v) => v / z)
  const tauMean = taus.reduce((acc, t, i) => acc + (pw[i] ?? 0) * t, 0)
  const tauSd = Math.sqrt(taus.reduce((acc, t, i) => acc + (pw[i] ?? 0) * (t - tauMean) ** 2, 0))
  const theta = y.map((yj, j): ExactMarginal => {
    const m = mult[j] ?? 1
    return gaussianMixtureMarginal(
      taus.map((t, i) => {
        const prec = t * m + 1 / (s * s)
        return { weight: pw[i] ?? 0, mean: yj / (s * s) / prec, variance: 1 / prec }
      }),
    )
  })
  return {
    tau: {
      mean: tauMean,
      sd: tauSd,
      pdf: (x) => (x <= 0 ? 0 : Math.exp(logPost(x) - peak) / (z * dt)),
      range: [Math.max(0, tauMean - 4 * tauSd), tauMean + 7 * tauSd],
    },
    theta,
  }
}

const Y_POP = [1.4, -0.6, 2.1, 0.3, -1.2, 0.9]

function populationProblem(mult: readonly number[], name: string, k: Config): AtomParts {
  const a = configNumber(k, 'shape')
  const b = configNumber(k, 'rate')
  const J = Y_POP.length
  const scaled = mult.some((v) => v !== 1)
  // region shared-precision scaled-precision
  const definition = defineModel(name, (m) => {
    const j = m.range('j')
    const tau = m.latent('tau', d.gamma(a, b))
    const multiplier = m.data('mult', { dims: [j] })
    const scale = scaled ? ops.div(1, ops.sqrt(ops.mul(tau, multiplier))) : ops.div(1, ops.sqrt(tau))
    const theta = m.latent('theta', d.normal(0, scale), { dims: [j] })
    m.observe('y', d.normal(theta, 1), m.data('y', { dims: [j], role: 'response' }))
  })
  // endregion
  const exact = scaledExact(Y_POP, mult, 1, a, b)
  const at = Y_POP.map((_, i) => i + 1)
  return {
    build: {
      definition,
      data: { y: Y_POP, mult: [...mult] },
      quantities: [
        { name: 'tau', label: 'tau (precision)', support: 'positive' },
        ...at.map((i) => ({ name: `theta[${i - 1}]`, label: `theta_${i}`, support: 'real' as const })),
      ],
    },
    reference: {
      exact: { tau: exact.tau, ...Object.fromEntries(exact.theta.map((e, i) => [`theta[${i}]`, e])) },
    },
    present: {
      equations: [
        `tau ~ Gamma(${a}, ${b})`,
        scaled
          ? `theta_j ~ N(0, 1 / (tau c_j)), c = (${mult.join(', ')})   (a known multiplier on the precision of each)`
          : 'theta_j ~ N(0, 1 / tau), j = 1..6   (one precision shared by all)',
        'y_j ~ N(theta_j, 1²)',
        `observe y = (${Y_POP.join(', ')})`,
      ],
      figure: {
        caption: `The observed y_j (dots) and the belief about theta_j (bars): each is pulled towards zero by an amount that depends on the precision the data suggest${scaled ? ' and on its multiplier' : ''}.`,
        xLabel: 'j',
        yLabel: 'value',
        xLimits: [0, J + 1],
        points: [{ name: 'observed y', x: at, y: [...Y_POP] }],
        answer: (m) =>
          at.map((j) => ({
            name: `theta_${j}`,
            x: [j - 0.3, j + 0.3],
            y: [m[`theta[${j - 1}]`] ?? Number.NaN, m[`theta[${j - 1}]`] ?? Number.NaN],
          })),
      },
    },
  }
}

const POP_KNOBS = dict({
  shape: real(0.5, 10, { default: 2, step: 0.1, label: 'prior shape of tau' }),
  rate: real(0.1, 10, { default: 1, step: 0.1, label: 'prior rate of tau' }),
})

/**
 * Six Gaussian values that share one gamma precision, each observed once (the latent form of the gamma-precision rule): the
 * shrinkage of a hierarchical model.
 */
const sharedPrecisionAtom: EpAtom = defineAtom({
  name: 'shared-precision',
  title: 'Shared precision',
  what: 'Six latent values theta_j ~ N(0, 1/tau) with one gamma-distributed precision tau, each observed once with unit noise: the gamma-precision factor on latent values, six factors on one gamma variable (the hierarchy).',
  factors: ['prior (gamma)', 'gaussian-gamma-precision (latent value)', 'gaussian likelihood'],
  family: 'scales',
  reference: 'exact',
  tolerance: tolerance(
    'gamma-projected',
    'the posterior of tau is skewed and EP projects it onto a gamma; each theta is a scale mixture that EP matches with a Gaussian',
  ),
  knobs: POP_KNOBS,
  parts: (k) =>
    populationProblem(
      Y_POP.map(() => 1),
      'shared-precision',
      k,
    ),
})

/**
 * The same with a known multiplier on each value's precision (the penalty weights of a spline basis): the scaled precision.
 */
const scaledPrecisionAtom: EpAtom = defineAtom({
  name: 'scaled-precision',
  title: 'Scaled precision (spline penalty)',
  what: 'Six latent values theta_j ~ N(0, 1/(tau c_j)) with known multipliers c_j and one gamma-distributed tau, each observed once: the penalised-spline coefficient prior, one random smoothing parameter on a vector.',
  factors: ['prior (gamma)', 'gaussian-gamma-precision with a known multiplier', 'gaussian likelihood'],
  family: 'scales',
  reference: 'exact',
  tolerance: tolerance(
    'coupled',
    'as for the shared precision: a gamma projection of tau and a Gaussian for each value',
  ),
  knobs: POP_KNOBS,
  parts: (k) => populationProblem([1, 2, 4, 8, 16, 32], 'scaled-precision', k),
})

function halfProblem(kind: 'cauchy' | 'normal', k: Config): AtomParts {
  const y = [0.5, -1.2, 2.0, 0.3]
  const scale = configNumber(k, 'priorScale')
  // region half-cauchy half-normal
  const definition = defineModel(`half-${kind}-scale`, (m) => {
    const sigma = m.latent('sigma', kind === 'cauchy' ? d.halfCauchy(scale) : d.halfNormal(scale))
    m.observe('y', d.normal(0, sigma), m.data('y', { dims: ['i'], role: 'response' }))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y },
      quantities: [{ name: 'sigma', label: 'sigma', support: 'positive' }],
    },
    present: {
      equations: [
        kind === 'cauchy' ? `sigma ~ HalfCauchy(${scale})` : `sigma ~ HalfNormal(${scale})`,
        'y_i ~ N(0, sigma²), i = 1..4',
        `observe y = (${y.join(', ')})`,
      ],
    },
  }
}

/**
 * A half-Cauchy prior on a scale, observed through four Gaussian draws: a prior with no message family.
 */
const halfCauchyAtom: EpAtom = defineAtom({
  name: 'half-cauchy',
  title: 'Half-Cauchy scale',
  what: 'A half-Cauchy prior on a standard deviation sigma and four zero-mean Gaussian observations with that scale: a heavy-tailed scale prior that has no message family of its own.',
  factors: ['numeric prior (half-Cauchy)', 'gaussian with a random scale'],
  family: 'scales',
  reference: 'grid',
  tolerance: tolerance('lumpy', 'the posterior of the scale is skewed and heavy-tailed; EP projects it onto a gamma'),
  evidence: {
    nats: 0.03,
    reason:
      'the half-Cauchy prior is the exact numeric site on a log scale; the Gaussian belief of the log is the only approximation (0.015 measured, was 0.17 more when the prior was projected once)',
  },
  knobs: dict({ priorScale: real(0.2, 5, { default: 1, step: 0.1, label: 'prior scale' }) }),
  parts: (k) => halfProblem('cauchy', k),
})

/**
 * A half-normal prior on a scale, the same observations.
 */
const halfNormalAtom: EpAtom = defineAtom({
  name: 'half-normal',
  title: 'Half-normal scale',
  what: 'A half-normal prior on a standard deviation sigma and four zero-mean Gaussian observations with that scale.',
  factors: ['numeric prior (half-normal)', 'gaussian with a random scale'],
  family: 'scales',
  reference: 'grid',
  tolerance: tolerance('lumpy', 'the posterior of the scale is skewed; EP projects it onto a gamma'),
  evidence: {
    nats: 0.01,
    reason:
      'the half-normal prior is the exact numeric site on a log scale (0.004 measured, was 0.42 more when the prior was projected once)',
  },
  knobs: dict({ priorScale: real(0.2, 5, { default: 1, step: 0.1, label: 'prior scale' }) }),
  parts: (k) => halfProblem('normal', k),
})

function precisionLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueMean, trueSd] = [configNumber(k, 'trueMean'), configNumber(k, 'trueSd')]
  const [a, b] = [configNumber(k, 'shape'), configNumber(k, 'rate')]
  const priorSd = configNumber(k, 'priorSd')
  // region gamma-precision-learned
  // the ground truth the data are drawn from: the model does not know it
  const y = Array.from(toFlat(normal(foldIn(key(seed), 0), [n], { loc: trueMean, scale: trueSd })))
  const definition = defineModel('gamma-precision-learned', (m) => {
    // the precision (one over the variance) and the mean: both unknown
    const tau = m.latent('tau', d.gamma(a, b))
    const mu = m.latent('mu', d.normal(0, priorSd))
    const observed = m.data('y', { dims: ['i'], role: 'response' })
    m.observe('y', d.normal(mu, ops.div(1, ops.sqrt(tau))), observed)
  })
  // endregion
  const x = [0, n + 1]
  return {
    build: {
      definition,
      data: { y },
      quantities: [
        { name: 'mu', label: 'mu (the mean)', support: 'real', truth: trueMean },
        { name: 'tau', label: 'tau (the precision)', support: 'positive', truth: 1 / trueSd ** 2 },
      ],
    },
    reference: {
      grid: { points: 61, maxRefinements: 0 },
    },
    present: {
      equations: [
        `truth: mean ${trueMean}, sd ${trueSd} (precision ${Number((1 / trueSd ** 2).toPrecision(3))}); ${n} observations drawn with seed ${seed}`,
        `tau ~ Gamma(${a}, ${b})   (the unknown precision; prior mean ${Number((a / b).toPrecision(3))})`,
        `mu ~ N(0, ${priorSd}²)   (the unknown mean)`,
        `y_i ~ N(mu, 1 / tau), i = 1..${n}`,
      ],
      pair: ['mu', 'tau'],
      figure: {
        caption:
          'The simulated observations. Green lines: the true mean and its two-sd band; thick and dashed: the fitted ones.',
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
          const w = 2 / Math.sqrt(mm['tau'] ?? Number.NaN)
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

/**
 * A Gaussian with an unknown mean and a gamma-distributed precision (independent priors, so the joint posterior has no closed
 * form), learned from data simulated from a stated truth: does EP recover the mean and the precision?
 */
const gammaPrecisionLearnedAtom: EpAtom = defineAtom({
  name: 'gamma-precision-learned',
  title: 'Gamma precision: parameters learned',
  extends: 'gamma-precision',
  recovery: recovery(
    'standard',
    'EP projects the skewed joint of the mean and the precision onto a Gaussian and a gamma, so its intervals are close to the exact ones but not equal; 40 seeds leave a binomial spread of about 0.05',
  ),
  what: 'The Gaussian with a gamma precision of the basic atom, with the mean and the precision both unknown and the data simulated from a stated mean and sd: the recovery of both, against the grid posterior of the same model.',
  factors: ['prior (gaussian)', 'prior (gamma)', 'gaussian-gamma-precision (observed value)'],
  family: 'scales',
  reference: 'grid',
  tolerance: tolerance(
    'close',
    'the factor is not Gaussian in the mean and the precision together, so EP projects onto a Gaussian and a gamma and the beliefs are close, not exact',
  ),
  evidence: { nats: 0.05, reason: 'the same with a learned mean' },
  knobs: dict({
    ...simulationKnobs(12),
    trueMean: real(-3, 3, { default: 1.5, step: 0.1, label: 'true mean' }),
    trueSd: real(0.2, 3, { default: 0.8, step: 0.05, label: 'true sd' }),
    priorSd: real(1, 20, { default: 5, step: 0.5, label: 'prior sd of the mean' }),
    shape: real(0.5, 10, { default: 1.5, step: 0.1, label: 'prior shape of tau' }),
    rate: real(0.1, 10, { default: 1, step: 0.1, label: 'prior rate of tau' }),
  }),
  parts: precisionLearnedProblem,
})

/** The known noise sd of each of the eight observed values (the standard errors of the eight schools' estimates, in spirit). */
const NOISE = [1, 1.5, 0.7, 2, 1.2, 0.9, 1.8, 1.1]

function populationLearnedProblem(k: Config): AtomParts {
  const [seed, groups] = [configNumber(k, 'seed'), configNumber(k, 'groups')]
  const [trueMean, trueSpread] = [configNumber(k, 'trueMean'), configNumber(k, 'trueSpread')]
  const [a, b, priorSd] = [configNumber(k, 'shape'), configNumber(k, 'rate'), configNumber(k, 'priorSd')]
  const noise = Array.from({ length: groups }, (_, j) => NOISE[j % NOISE.length] ?? 1)
  // region shared-precision-learned
  // the ground truth: effects drawn around a mean with a spread, then each observed with its own noise
  const kk = key(seed)
  const around = { loc: trueMean, scale: trueSpread }
  const effects = Array.from(toFlat(normal(foldIn(kk, 0), [groups], around)))
  const draw = Array.from(toFlat(normal(foldIn(kk, 1), [groups])))
  const y = effects.map((v, j) => v + (noise[j] ?? 1) * (draw[j] ?? 0))
  const definition = defineModel('shared-precision-learned', (m) => {
    const j = m.range('group')
    // the precision of the effects (one over their spread) and their mean: both unknown
    const tau = m.latent('tau', d.gamma(a, b))
    const mu = m.latent('mu', d.normal(0, priorSd))
    const theta = m.latent('theta', d.normal(mu, ops.div(1, ops.sqrt(tau))), { dims: [j] })
    const noise = m.data('sd', { dims: [j] })
    const observed = m.data('y', { dims: [j], role: 'response' })
    m.observe('y', d.normal(theta, noise), observed)
  })
  // endregion
  // The exact posterior: the precision by quadrature with the mean integrated out, each of the others Gaussian given it.
  const [m0, v0] = [0, priorSd ** 2]
  const stats = (tau: number): { prec: number; mean: number; w: number[] } => {
    const w = noise.map((s) => 1 / (s * s + 1 / tau))
    const prec = 1 / v0 + w.reduce((p, q) => p + q, 0)
    const mean = (m0 / v0 + w.reduce((p, q, j) => p + q * (y[j] ?? 0), 0)) / prec
    return { prec, mean, w }
  }
  const logMarginal = (tau: number): number => {
    const { prec, w } = stats(tau)
    const lin = m0 / v0 + w.reduce((p, q, j) => p + q * (y[j] ?? 0), 0)
    const quad = w.reduce((p, q, j) => p + q * (y[j] ?? 0) ** 2, 0) + m0 ** 2 / v0
    return (
      (a - 1) * Math.log(tau) -
      b * tau +
      0.5 * w.reduce((p, q) => p + Math.log(q), 0) -
      0.5 * quad +
      (0.5 * lin * lin) / prec -
      0.5 * Math.log(prec * v0)
    )
  }
  const post = hyperPosterior(logMarginal, 1e-4, 200)
  const gauss = (mean: number, variance: number): ConditionalMarginal => ({
    mean,
    variance,
    pdf: (x) => normalPdf(x, mean, Math.sqrt(variance)),
  })
  const exactMu = post.mix((tau) => {
    const s = stats(tau)
    return gauss(s.mean, 1 / s.prec)
  })
  const exactTheta = effects.map((_, j) =>
    post.mix((tau) => {
      const s = stats(tau)
      const prec = 1 / (noise[j] ?? 1) ** 2
      const shrink = prec / (tau + prec)
      return gauss(shrink * (y[j] ?? 0) + (1 - shrink) * s.mean, 1 / (tau + prec) + (1 - shrink) ** 2 / s.prec)
    }),
  )
  const at = effects.map((_, j) => j + 1)
  return {
    build: {
      definition,
      data: { y, sd: noise },
      quantities: [
        { name: 'tau', label: 'tau (the precision of the effects)', support: 'positive', truth: 1 / trueSpread ** 2 },
        { name: 'mu', label: 'mu (the mean effect)', support: 'real', truth: trueMean },
        ...effects.map((t, j) => ({
          name: `theta[${j}]`,
          label: `theta_${j + 1}`,
          support: 'real' as const,
          truth: t,
          pool: 'theta (every group)',
        })),
      ],
    },
    reference: {
      exact: { tau: post.hyper, mu: exactMu, ...Object.fromEntries(exactTheta.map((e, j) => [`theta[${j}]`, e])) },
    },
    present: {
      equations: [
        `truth: effects around ${trueMean} with sd ${trueSpread} (precision ${Number((1 / trueSpread ** 2).toPrecision(3))}); ${groups} groups, seed ${seed}`,
        `tau ~ Gamma(${a}, ${b})   (the unknown precision of the effects)`,
        `mu ~ N(0, ${priorSd}²)   (the unknown mean of the effects)`,
        `theta_j ~ N(mu, 1 / tau), j = 1..${groups}`,
        `y_j ~ N(theta_j, s_j²), s = (${noise.join(', ')})   (known noise)`,
      ],
      figure: {
        caption:
          'The observed value of each group (dots), its true effect (open marks) and what EP believes of the effect (bars): each is pulled towards the mean by how much the learned spread allows.',
        xLabel: 'group',
        yLabel: 'value',
        xLimits: [0, groups + 1],
        points: [
          { name: 'observed y', x: at, y },
          { name: 'true effect', x: at, y: effects },
        ],
        answer: (mm) =>
          at.map((j) => ({
            name: `theta_${j}`,
            x: [j - 0.3, j + 0.3],
            y: [mm[`theta[${j - 1}]`] ?? Number.NaN, mm[`theta[${j - 1}]`] ?? Number.NaN],
          })),
      },
    },
  }
}

/**
 * Latent values around an unknown mean with an unknown shared precision, each observed once with its own known noise (the eight
 * schools shape), learned from data simulated from a stated truth: does EP recover the spread of the population, its mean and
 * each value?
 */
const sharedPrecisionLearnedAtom: EpAtom = defineAtom({
  name: 'shared-precision-learned',
  title: 'Shared precision: parameters learned',
  extends: 'shared-precision',
  recovery: recovery(
    'standard',
    'a spread learned from a handful of groups is poorly identified and its posterior is skewed, which EP carries as a gamma; each value is a Gaussian matched to a scale mixture',
  ),
  what: 'The shared precision of the basic atom with the population mean also unknown and each value observed with its own known noise: a hierarchical model in the shape of the eight schools, simulated from a stated mean and spread.',
  factors: ['prior (gamma)', 'prior (gaussian)', 'gaussian-gamma-precision (latent value)', 'gaussian likelihood'],
  family: 'scales',
  reference: 'exact',
  tolerance: tolerance(
    'coupled',
    'the posterior of the precision is skewed and EP projects it onto a gamma; each effect is a scale mixture that EP matches with a Gaussian',
  ),
  knobs: dict({
    seed: simulationKnobs(8)['seed'] as ReturnType<typeof int>,
    groups: int(3, 8, { default: 8, label: 'groups' }),
    trueMean: real(-3, 3, { default: 1, step: 0.1, label: 'true mean effect' }),
    trueSpread: real(0.2, 4, { default: 1.5, step: 0.1, label: 'true spread (sd) of the effects' }),
    priorSd: real(1, 20, { default: 5, step: 0.5, label: 'prior sd of the mean' }),
    shape: real(0.5, 8, { default: 1.5, step: 0.1, label: 'prior shape of tau' }),
    rate: real(0.1, 8, { default: 1, step: 0.1, label: 'prior rate of tau' }),
  }),
  parts: populationLearnedProblem,
})

/** The atoms of the `scales` file, basic and learned twins together. */
export const scalesAtoms: readonly EpAtom[] = [
  gammaPrecisionAtom,
  sharedPrecisionAtom,
  scaledPrecisionAtom,
  halfCauchyAtom,
  halfNormalAtom,
  gammaPrecisionLearnedAtom,
  sharedPrecisionLearnedAtom,
]
