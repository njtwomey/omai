/**
 * The extended twins of the atoms with an unknown scale: a Gaussian with an unknown mean and a gamma precision (independent
 * priors, so no closed form: the reference is a grid) and a population of values with an unknown mean and an unknown spread (the
 * eight schools shape: each value observed with its own known noise). The data are simulated from a stated truth and the page
 * checks that EP recovers the mean, the precision and the values, and how often its 90% intervals hold them.
 */

import { foldIn, key, normal } from '../../random/index.ts'
import { dict, int, real } from '../../space/index.ts'
import { toFlat } from '../../tensor/index.ts'
import type { Config } from '../../trace/index.ts'
import { configNumber } from '../inference/method.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { type ConditionalMarginal, hyperPosterior, normalPdf } from '../validation/index.ts'
import { type AtomParts, defineAtom, type EpAtom } from './ep-atoms.ts'
import { simulationKnobs } from './ep-atoms-extended-conjugate.ts'
import { recovery, tolerance } from './ep-tolerance.ts'

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
 *
 * @index estimator EP atom: a Gaussian with unknown mean and gamma precision, recovered from simulated data.
 * @example
 * ```ts
 * import { gammaPrecisionLearnedAtom } from 'omai-engine/probprog/ep'
 * gammaPrecisionLearnedAtom.extends // => 'gamma-precision'
 * ```
 */
export const gammaPrecisionLearnedAtom: EpAtom = defineAtom({
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
 *
 * @index estimator EP atom: a population with unknown mean and spread, each value observed with known noise, recovered from simulated data.
 * @example
 * ```ts
 * import { sharedPrecisionLearnedAtom } from 'omai-engine/probprog/ep'
 * sharedPrecisionLearnedAtom.extends // => 'shared-precision'
 * ```
 */
export const sharedPrecisionLearnedAtom: EpAtom = defineAtom({
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
