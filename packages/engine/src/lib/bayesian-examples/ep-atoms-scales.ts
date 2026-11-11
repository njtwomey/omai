/**
 * The setups of expectation propagation with an unknown scale: a Gaussian with a gamma-distributed precision (the
 * gamma-precision rule, observed and latent), the same with a known multiplier on the precision of each effect (the penalty
 * weights of a spline), and half-Cauchy and half-normal scale priors. The references are a grid over the mean and the
 * precision, and a quadrature over the precision with the effects Gaussian given it.
 */

import * as d from '../probprog/model/dists/index.ts'
import { defineModel, ops } from '../probprog/model/index.ts'
import { dict, real } from '../space/index.ts'
import type { Config } from '../trace/index.ts'
import { type AtomProblem, defineAtom, type EpAtom, type ExactMarginal } from './ep-atoms.ts'
import { exampleNumber } from './example-method.ts'

const Y6 = [1.2, 0.4, 1.9, 1.1, 0.7, 1.5]

function precisionProblem(k: Config): AtomProblem {
  const a = exampleNumber(k, 'shape')
  const b = exampleNumber(k, 'rate')
  const definition = defineModel('gamma-precision', (m) => {
    const tau = m.latent('tau', d.gamma(a, b))
    const mu = m.latent('mu', d.normal(0, 10))
    m.observe('y', d.normal(mu, ops.div(1, ops.sqrt(tau))), m.data('y', { dims: ['i'], role: 'response' }))
  })
  return {
    definition,
    data: { y: Y6 },
    equations: [
      `tau ~ Gamma(${a}, ${b})   (a precision)`,
      'mu ~ N(0, 10²)',
      'y_i ~ N(mu, 1 / tau), i = 1..6',
      `observe y = (${Y6.join(', ')})`,
    ],
    quantities: [
      { name: 'mu', label: 'mu', support: 'real' },
      { name: 'tau', label: 'tau (precision)', support: 'positive' },
    ],
    pair: ['mu', 'tau'],
    grid: { points: 61, maxRefinements: 0 },
    figure: {
      caption: 'The six observations; the line is mu and the band two standard deviations, 2 / sqrt(tau), either side.',
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
  }
}

/**
 * A Gaussian with a gamma prior on its precision, observed six times: the gamma-precision factor on an observed value,
 * with a skewed, correlated posterior of the mean and the precision.
 *
 * @index estimator EP atom: a Gaussian with an unknown precision (gamma-precision rule, observed value).
 * @example
 * ```ts
 * import { gammaPrecisionAtom } from 'omai-engine/bayesian-examples'
 * gammaPrecisionAtom.name // => 'gamma-precision'
 * ```
 */
export const gammaPrecisionAtom: EpAtom = defineAtom({
  name: 'gamma-precision',
  title: 'Gamma precision',
  what: 'A Gaussian mean mu and a gamma-distributed precision tau, six observations y_i ~ N(mu, 1/tau): the gamma-precision factor with an observed value, a Student-t-like joint posterior.',
  factors: ['prior (gaussian)', 'prior (gamma)', 'gaussian-gamma-precision (observed value)'],
  family: 'scales',
  expected: 'runs',
  reference: 'grid',
  tolerance: {
    meanZ: 0.1,
    sdRatio: [0.85, 1.15],
    ks: 0.05,
    reason:
      'the factor is not Gaussian in mu and tau together, so EP projects onto a Gaussian and a gamma and the beliefs are close, not exact',
  },
  knobs: dict({
    shape: real(0.5, 10, { default: 2, step: 0.1, label: 'prior shape of tau' }),
    rate: real(0.1, 10, { default: 1, step: 0.1, label: 'prior rate of tau' }),
  }),
  problem: precisionProblem,
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
  const peak = Math.max(...logs)
  const w = logs.map((l) => Math.exp(l - peak))
  const z = w.reduce((p, q) => p + q, 0)
  const pw = w.map((v) => v / z)
  const tauMean = taus.reduce((acc, t, i) => acc + (pw[i] ?? 0) * t, 0)
  const tauSd = Math.sqrt(taus.reduce((acc, t, i) => acc + (pw[i] ?? 0) * (t - tauMean) ** 2, 0))
  const theta = y.map((yj, j): ExactMarginal => {
    const m = mult[j] ?? 1
    const comp = (t: number): { mean: number; v: number } => {
      const prec = t * m + 1 / (s * s)
      return { mean: yj / (s * s) / prec, v: 1 / prec }
    }
    let m1 = 0
    let m2 = 0
    taus.forEach((t, i) => {
      const c = comp(t)
      m1 += (pw[i] ?? 0) * c.mean
      m2 += (pw[i] ?? 0) * (c.v + c.mean ** 2)
    })
    const pdf = (x: number): number =>
      taus.reduce((acc, t, i) => {
        const c = comp(t)
        return acc + ((pw[i] ?? 0) * Math.exp((-0.5 * (x - c.mean) ** 2) / c.v)) / Math.sqrt(2 * Math.PI * c.v)
      }, 0)
    return { mean: m1, sd: Math.sqrt(Math.max(m2 - m1 * m1, 0)), pdf }
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

function populationProblem(mult: readonly number[], name: string, k: Config): AtomProblem {
  const a = exampleNumber(k, 'shape')
  const b = exampleNumber(k, 'rate')
  const J = Y_POP.length
  const scaled = mult.some((v) => v !== 1)
  const definition = defineModel(name, (m) => {
    const j = m.range('j')
    const tau = m.latent('tau', d.gamma(a, b))
    const multiplier = m.data('mult', { dims: [j] })
    const scale = scaled ? ops.div(1, ops.sqrt(ops.mul(tau, multiplier))) : ops.div(1, ops.sqrt(tau))
    const theta = m.latent('theta', d.normal(0, scale), { dims: [j] })
    m.observe('y', d.normal(theta, 1), m.data('y', { dims: [j], role: 'response' }))
  })
  const exact = scaledExact(Y_POP, mult, 1, a, b)
  const at = Y_POP.map((_, i) => i + 1)
  return {
    definition,
    data: { y: Y_POP, mult: [...mult] },
    equations: [
      `tau ~ Gamma(${a}, ${b})`,
      scaled
        ? `theta_j ~ N(0, 1 / (tau c_j)), c = (${mult.join(', ')})   (a known multiplier on the precision of each)`
        : 'theta_j ~ N(0, 1 / tau), j = 1..6   (one precision shared by all)',
      'y_j ~ N(theta_j, 1²)',
      `observe y = (${Y_POP.join(', ')})`,
    ],
    quantities: [
      { name: 'tau', label: 'tau (precision)', support: 'positive' },
      ...at.map((i) => ({ name: `theta[${i - 1}]`, label: `theta_${i}`, support: 'real' as const })),
    ],
    exact: { tau: exact.tau, ...Object.fromEntries(exact.theta.map((e, i) => [`theta[${i}]`, e])) },
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
  }
}

const POP_KNOBS = dict({
  shape: real(0.5, 10, { default: 2, step: 0.1, label: 'prior shape of tau' }),
  rate: real(0.1, 10, { default: 1, step: 0.1, label: 'prior rate of tau' }),
})

/**
 * Six Gaussian values that share one gamma precision, each observed once (the latent form of the gamma-precision rule): the
 * shrinkage of a hierarchical model.
 *
 * @index estimator EP atom: latent values with a shared unknown precision (gamma-precision rule, latent form).
 * @example
 * ```ts
 * import { sharedPrecisionAtom } from 'omai-engine/bayesian-examples'
 * sharedPrecisionAtom.family // => 'scales'
 * ```
 */
export const sharedPrecisionAtom: EpAtom = defineAtom({
  name: 'shared-precision',
  title: 'Shared precision',
  what: 'Six latent values theta_j ~ N(0, 1/tau) with one gamma-distributed precision tau, each observed once with unit noise: the gamma-precision factor on latent values, six factors on one gamma variable (the hierarchy).',
  factors: ['prior (gamma)', 'gaussian-gamma-precision (latent value)', 'gaussian likelihood'],
  family: 'scales',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.1,
    sdRatio: [0.7, 1.15],
    ks: 0.1,
    reason:
      'the posterior of tau is skewed and EP projects it onto a gamma; each theta is a scale mixture that EP matches with a Gaussian',
  },
  knobs: POP_KNOBS,
  problem: (k) =>
    populationProblem(
      Y_POP.map(() => 1),
      'shared-precision',
      k,
    ),
})

/**
 * The same with a known multiplier on each value's precision (the penalty weights of a spline basis): the scaled precision.
 *
 * @index estimator EP atom: latent values whose precisions are a known multiple of one unknown precision (scaled precision).
 * @example
 * ```ts
 * import { scaledPrecisionAtom } from 'omai-engine/bayesian-examples'
 * scaledPrecisionAtom.name // => 'scaled-precision'
 * ```
 */
export const scaledPrecisionAtom: EpAtom = defineAtom({
  name: 'scaled-precision',
  title: 'Scaled precision (spline penalty)',
  what: 'Six latent values theta_j ~ N(0, 1/(tau c_j)) with known multipliers c_j and one gamma-distributed tau, each observed once: the penalised-spline coefficient prior, one random smoothing parameter on a vector.',
  factors: ['prior (gamma)', 'gaussian-gamma-precision with a known multiplier', 'gaussian likelihood'],
  family: 'scales',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.1,
    sdRatio: [0.7, 1.15],
    ks: 0.1,
    reason: 'as for the shared precision: a gamma projection of tau and a Gaussian for each value',
  },
  knobs: POP_KNOBS,
  problem: (k) => populationProblem([1, 2, 4, 8, 16, 32], 'scaled-precision', k),
})

function halfProblem(kind: 'cauchy' | 'normal', k: Config): AtomProblem {
  const y = [0.5, -1.2, 2.0, 0.3]
  const scale = exampleNumber(k, 'priorScale')
  const definition = defineModel(`half-${kind}-scale`, (m) => {
    const sigma = m.latent('sigma', kind === 'cauchy' ? d.halfCauchy(scale) : d.halfNormal(scale))
    m.observe('y', d.normal(0, sigma), m.data('y', { dims: ['i'], role: 'response' }))
  })
  return {
    definition,
    data: { y },
    equations: [
      kind === 'cauchy' ? `sigma ~ HalfCauchy(${scale})` : `sigma ~ HalfNormal(${scale})`,
      'y_i ~ N(0, sigma²), i = 1..4',
      `observe y = (${y.join(', ')})`,
    ],
    quantities: [{ name: 'sigma', label: 'sigma', support: 'positive' }],
  }
}

/**
 * A half-Cauchy prior on a scale, observed through four Gaussian draws: a prior with no message family.
 *
 * @index estimator EP atom: a half-Cauchy prior on a scale.
 * @example
 * ```ts
 * import { halfCauchyAtom } from 'omai-engine/bayesian-examples'
 * halfCauchyAtom.name // => 'half-cauchy'
 * ```
 */
export const halfCauchyAtom: EpAtom = defineAtom({
  name: 'half-cauchy',
  title: 'Half-Cauchy scale',
  what: 'A half-Cauchy prior on a standard deviation sigma and four zero-mean Gaussian observations with that scale: a heavy-tailed scale prior that has no message family of its own.',
  factors: ['numeric prior (half-Cauchy)', 'gaussian with a random scale'],
  family: 'scales',
  expected: 'planned',
  plan: 'ep-plan.md section 2, D (numeric priors: half-normal, half-Cauchy, log-normal, uniform), with a random scale outside the precision idiom; in progress on another branch.',
  reference: 'grid',
  tolerance: {
    meanZ: 0.2,
    sdRatio: [0.6, 1.4],
    ks: 0.15,
    reason: 'the posterior of the scale is skewed and heavy-tailed; EP projects it onto a gamma',
  },
  knobs: dict({ priorScale: real(0.2, 5, { default: 1, step: 0.1, label: 'prior scale' }) }),
  problem: (k) => halfProblem('cauchy', k),
})

/**
 * A half-normal prior on a scale, the same observations.
 *
 * @index estimator EP atom: a half-normal prior on a scale.
 * @example
 * ```ts
 * import { halfNormalAtom } from 'omai-engine/bayesian-examples'
 * halfNormalAtom.name // => 'half-normal'
 * ```
 */
export const halfNormalAtom: EpAtom = defineAtom({
  name: 'half-normal',
  title: 'Half-normal scale',
  what: 'A half-normal prior on a standard deviation sigma and four zero-mean Gaussian observations with that scale.',
  factors: ['numeric prior (half-normal)', 'gaussian with a random scale'],
  family: 'scales',
  expected: 'planned',
  plan: 'ep-plan.md section 2, D (numeric priors) with a random scale outside the precision idiom; in progress on another branch.',
  reference: 'grid',
  tolerance: {
    meanZ: 0.2,
    sdRatio: [0.6, 1.4],
    ks: 0.15,
    reason: 'the posterior of the scale is skewed; EP projects it onto a gamma',
  },
  knobs: dict({ priorScale: real(0.2, 5, { default: 1, step: 0.1, label: 'prior scale' }) }),
  problem: (k) => halfProblem('normal', k),
})
