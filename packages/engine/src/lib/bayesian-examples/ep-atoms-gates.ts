/**
 * The setups of expectation propagation with a discrete switch or a mixture: a gate that selects between two components
 * (known values, then fresh Gaussian variables), the spike and slab prior, an observation that is a mixture of an inlier and
 * an outlier (Minka's clutter) and an ordered pair (a hard comparison, a truncation). The exact answers are closed forms:
 * a switch's probability from the evidence of each branch, a mixture of Gaussians for what a branch leaves, a truncated
 * Gaussian for the ordered pair.
 */

import * as d from '../probprog/model/dists/index.ts'
import { defineModel, ops } from '../probprog/model/index.ts'
import { dict, real } from '../space/index.ts'
import type { Config } from '../trace/index.ts'
import { type AtomProblem, defineAtom, type EpAtom, type RefJoint } from './ep-atoms.ts'
import { gridMarginals, mixtureMarginal, normalCdf, normalPdf, quadratureMarginal } from './ep-atoms-quad.ts'
import { exampleNumber } from './example-method.ts'

function knownGateProblem(k: Config): AtomProblem {
  const pi = exampleNumber(k, 'pi')
  const y = exampleNumber(k, 'y')
  const definition = defineModel('gate-known', (m) => {
    const z = m.latent('z', d.bernoulli({ probs: pi }))
    m.observe('y', d.normal(ops.where(z, 3, 0), 1), m.data('y'))
  })
  const e1 = pi * normalPdf(y, 3, 1)
  const e0 = (1 - pi) * normalPdf(y, 0, 1)
  const p = e1 / (e1 + e0)
  return {
    definition,
    data: { y },
    equations: [
      `z ~ Bernoulli(${pi})`,
      'g = z ? 3 : 0   (the gate: a select between two known values)',
      'y ~ N(g, 1²)',
      `observe y = ${y}`,
    ],
    quantities: [{ name: 'z', label: 'P(z = 1)', support: 'probability' }],
    exact: { z: { mean: p, sd: Math.sqrt(p * (1 - p)), pdf: () => 0 } },
  }
}

/**
 * A switch that selects one of two known values, observed through Gaussian noise: the select gate with known components.
 * The message to the switch is the ratio of the two branches' evidence.
 *
 * @index estimator EP atom: a gate (select) between two known components.
 * @example
 * ```ts
 * import { gateKnownAtom } from 'omai-engine/bayesian-examples'
 * gateKnownAtom.name // => 'gate'
 * ```
 */
export const gateKnownAtom: EpAtom = defineAtom({
  name: 'gate',
  title: 'Gate',
  what: 'A Bernoulli switch z selects between the known values 0 and 3, and y is that value plus Gaussian noise: the select factor, whose message to z is the ratio of the two branches’ evidence.',
  factors: ['prior (bernoulli)', 'select gate (known components)', 'gaussian likelihood'],
  family: 'gates',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 1e-6,
    sdRatio: [0.999999, 1.000001],
    ks: 1e-4,
    reason: 'a single discrete switch: the message is exact',
  },
  knobs: dict({
    pi: real(0.02, 0.98, { default: 0.3, step: 0.01, label: 'P(z = 1) before the observation' }),
    y: real(-3, 6, { default: 1.5, step: 0.1, label: 'observed y' }),
  }),
  problem: knownGateProblem,
})

function randomGateProblem(k: Config): AtomProblem {
  const pi = exampleNumber(k, 'pi')
  const y = exampleNumber(k, 'y')
  const s = 0.5
  const [ma, sa, mb, sb] = [-2, 1, 2, 1]
  const definition = defineModel('gate-random', (m) => {
    const z = m.latent('z', d.bernoulli({ probs: pi }))
    const a = m.latent('a', d.normal(ma, sa))
    const b = m.latent('b', d.normal(mb, sb))
    m.observe('y', d.normal(ops.where(z, b, a), s), m.data('y'))
  })
  // z = 1 selects b: y | z = 1 ~ N(mb, sb² + s²) and b | y, z = 1 is the conjugate update; a keeps its prior (and the other way for z = 0).
  const e1 = pi * normalPdf(y, mb, Math.sqrt(sb ** 2 + s ** 2))
  const e0 = (1 - pi) * normalPdf(y, ma, Math.sqrt(sa ** 2 + s ** 2))
  const p1 = e1 / (e1 + e0)
  const update = (m0: number, s0: number): { mean: number; sd: number } => {
    const prec = 1 / s0 ** 2 + 1 / s ** 2
    return { mean: (m0 / s0 ** 2 + y / s ** 2) / prec, sd: Math.sqrt(1 / prec) }
  }
  const ub = update(mb, sb)
  const ua = update(ma, sa)
  return {
    definition,
    data: { y },
    equations: [
      `z ~ Bernoulli(${pi})`,
      `a ~ N(${ma}, ${sa}²),  b ~ N(${mb}, ${sb}²)`,
      'g = z ? b : a   (the gate: a select between two random variables)',
      `y ~ N(g, ${s}²)`,
      `observe y = ${y}`,
    ],
    quantities: [
      { name: 'z', label: 'P(z = 1)', support: 'probability' },
      { name: 'a', label: 'a', support: 'real' },
      { name: 'b', label: 'b', support: 'real' },
    ],
    exact: {
      z: { mean: p1, sd: Math.sqrt(p1 * (1 - p1)), pdf: () => 0 },
      a: mixtureMarginal([
        { weight: p1, mean: ma, sd: sa },
        { weight: 1 - p1, mean: ua.mean, sd: ua.sd },
      ]),
      b: mixtureMarginal([
        { weight: p1, mean: ub.mean, sd: ub.sd },
        { weight: 1 - p1, mean: mb, sd: sb },
      ]),
    },
  }
}

/**
 * A switch that selects between two random variables: the one that is selected is pulled towards the observation, the
 * other keeps its prior, so the exact marginals are two-component mixtures and EP's one Gaussian per variable covers both.
 *
 * @index estimator EP atom: a gate between two random components (bimodal exact marginals).
 * @example
 * ```ts
 * import { gateRandomAtom } from 'omai-engine/bayesian-examples'
 * gateRandomAtom.family // => 'gates'
 * ```
 */
export const gateRandomAtom: EpAtom = defineAtom({
  name: 'gate-random',
  title: 'Gate between two random values',
  what: 'A Bernoulli switch z selects one of two Gaussian variables a and b, and y is the selected one plus noise. The exact marginal of each of a and b is a mixture of the updated and the prior Gaussian, with weights that are the posterior of z.',
  factors: ['prior (bernoulli)', 'prior (gaussian) x2', 'select gate (fresh random components)', 'gaussian likelihood'],
  family: 'gates',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.3,
    sdRatio: [0.6, 1.6],
    ks: 0.25,
    reason:
      'the exact marginals are two-component mixtures; EP keeps one Gaussian for each of a and b, so it matches their mean and spread and cannot show the two humps',
  },
  knobs: dict({
    pi: real(0.02, 0.98, { default: 0.5, step: 0.01, label: 'P(z = 1) before the observation' }),
    y: real(-4, 4, { default: 0.5, step: 0.1, label: 'observed y' }),
  }),
  problem: randomGateProblem,
})

function spikeSlabProblem(k: Config): AtomProblem {
  const pi = exampleNumber(k, 'pi')
  const y = exampleNumber(k, 'y')
  const [slab, spike, s] = [2, 0.1, 0.5]
  const definition = defineModel('spike-and-slab', (m) => {
    const w = m.latent('w', d.normalSpikeSlab(pi, slab, spike))
    m.observe('y', d.normal(w, s), m.data('y'))
  })
  const branch = (weight: number, sd0: number): { weight: number; mean: number; sd: number } => {
    const v = sd0 ** 2 + s ** 2
    return {
      weight: weight * normalPdf(y, 0, Math.sqrt(v)),
      mean: (sd0 ** 2 * y) / v,
      sd: Math.sqrt((sd0 ** 2 * s ** 2) / v),
    }
  }
  return {
    definition,
    data: { y },
    equations: [
      `w ~ ${pi} N(0, ${slab}²) + ${(1 - pi).toFixed(2)} N(0, ${spike}²)   (spike and slab)`,
      `y ~ N(w, ${s}²)`,
      `observe y = ${y}`,
    ],
    quantities: [{ name: 'w', label: 'w', support: 'real' }],
    exact: { w: mixtureMarginal([branch(pi, slab), branch(1 - pi, spike)]) },
  }
}

/**
 * A coefficient with a spike and slab prior (a two-component Gaussian mixture centred at zero) observed through noise: the
 * mixture site, which has a closed form because a Gaussian times a Gaussian mixture is a Gaussian mixture.
 *
 * @index estimator EP atom: spike and slab prior on a coefficient (mixture site).
 * @example
 * ```ts
 * import { spikeSlabAtom } from 'omai-engine/bayesian-examples'
 * spikeSlabAtom.name // => 'spike-and-slab'
 * ```
 */
export const spikeSlabAtom: EpAtom = defineAtom({
  name: 'spike-and-slab',
  title: 'Spike and slab',
  what: 'A coefficient w with a two-component zero-centred Gaussian mixture prior, a narrow spike and a wide slab, observed through Gaussian noise: the mixture site, whose exact marginal is a two-component mixture.',
  factors: ['mixture-site (two-component prior)', 'gaussian likelihood'],
  family: 'gates',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.3,
    sdRatio: [0.6, 1.6],
    ks: 0.25,
    reason: 'the exact posterior has a spike and a slab hump; EP matches its mean and variance with one Gaussian',
  },
  knobs: dict({
    pi: real(0.05, 0.95, { default: 0.5, step: 0.01, label: 'slab probability' }),
    y: real(-4, 4, { default: 0.7, step: 0.1, label: 'observed y' }),
  }),
  problem: spikeSlabProblem,
})

const CLUTTER_Y = [1.0, 1.4, 0.9, 1.2, 1.6, -4.2, 1.1, 9.5]

function clutterProblem(k: Config): AtomProblem {
  const share = exampleNumber(k, 'share')
  const [s, width, priorSd] = [0.5, 10, 5]
  const definition = defineModel('clutter', (m) => {
    const mu = m.latent('mu', d.normal(0, priorSd))
    m.observe('y', d.normalOutliers(mu, s, share, width), m.data('y', { dims: ['i'], role: 'response' }))
  })
  const logPost = (x: number): number =>
    Math.log(normalPdf(x, 0, priorSd)) +
    CLUTTER_Y.reduce((a, v) => a + Math.log((1 - share) * normalPdf(v, x, s) + share * normalPdf(v, x, width * s)), 0)
  return {
    definition,
    data: { y: CLUTTER_Y },
    equations: [
      `mu ~ N(0, ${priorSd}²)`,
      `y_i ~ ${(1 - share).toFixed(2)} N(mu, ${s}²) + ${share} N(mu, ${width * s}²)   (inlier or outlier)`,
      `observe y = (${CLUTTER_Y.join(', ')})`,
    ],
    quantities: [{ name: 'mu', label: 'mu', support: 'real' }],
    exact: { mu: quadratureMarginal(logPost, -20, 20) },
    figure: {
      caption: 'The eight observations on a line: most agree near 1, two are far off.',
      xLabel: 'observation number',
      yLabel: 'value',
      xLimits: [0, CLUTTER_Y.length + 1],
      points: [{ name: 'observations', x: CLUTTER_Y.map((_, i) => i + 1), y: [...CLUTTER_Y] }],
      answer: (m) => [
        {
          name: 'mu',
          x: [0, CLUTTER_Y.length + 1],
          y: [m['mu'] ?? Number.NaN, m['mu'] ?? Number.NaN],
        },
      ],
    },
  }
}

/**
 * Minka's clutter problem: an unknown centre seen through observations that are each a narrow Gaussian or, with some
 * probability, a wide one (the mixture site). Two observations are far from the rest.
 *
 * @index estimator EP atom: clutter, an inlier/outlier mixture likelihood (mixture site).
 * @example
 * ```ts
 * import { clutterAtom } from 'omai-engine/bayesian-examples'
 * clutterAtom.family // => 'gates'
 * ```
 */
export const clutterAtom: EpAtom = defineAtom({
  name: 'clutter',
  title: 'Clutter (inlier and outlier mixture)',
  what: 'An unknown mean mu observed eight times through a mixture of a narrow and a wide Gaussian around it, two observations being far from the rest: the mixture site as a likelihood.',
  factors: ['prior (gaussian)', 'mixture-site (inlier and outlier noise)'],
  family: 'gates',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.3,
    sdRatio: [0.6, 1.5],
    ks: 0.2,
    reason: 'the posterior is lumpy and EP matches only its mean and variance with one Gaussian',
  },
  knobs: dict({ share: real(0.02, 0.6, { default: 0.25, step: 0.01, label: 'outlier share' }) }),
  problem: clutterProblem,
})

function orderedProblem(k: Config): AtomProblem {
  const gap = exampleNumber(k, 'gap')
  const [ma, mb, s] = [0, gap, 1]
  const definition = defineModel('ordered-pair', (m) => {
    const a = m.latent('a', d.normal(ma, s))
    const b = m.latent('b', d.normal(mb, s))
    m.observe('ordered', d.bernoulli({ probs: ops.gt(ops.sub(a, b), 0) }), m.data('ordered', { support: 'binary' }))
  })
  // a > b is observed: the density is the product of the two priors on the half plane a > b.
  const za = 1 - normalCdf((mb - ma) / Math.SQRT2 / s)
  const logA = (x: number): number =>
    Math.log(normalPdf(x, ma, s)) + Math.log(Math.max(normalCdf((x - mb) / s), 1e-300))
  const logB = (x: number): number =>
    Math.log(normalPdf(x, mb, s)) + Math.log(Math.max(1 - normalCdf((x - ma) / s), 1e-300))
  const N = 70
  const lo = -4.5 + Math.min(ma, mb)
  const hi = 4.5 + Math.max(ma, mb)
  const xs = Array.from({ length: N }, (_, i) => lo + ((hi - lo) * i) / (N - 1))
  const joint: RefJoint = {
    a: 'a',
    b: 'b',
    x: xs,
    y: xs,
    p: xs.map((x) => xs.map((y) => (x > y ? (normalPdf(x, ma, s) * normalPdf(y, mb, s)) / za : 0))),
  }
  return {
    definition,
    data: { ordered: 1 },
    equations: [`a ~ N(${ma}, ${s}²)`, `b ~ N(${mb}, ${s}²)`, 'observe a > b   (a hard comparison: a truncation)'],
    quantities: [
      { name: 'a', label: 'a', support: 'real' },
      { name: 'b', label: 'b', support: 'real' },
    ],
    pair: ['a', 'b'],
    exact: { a: quadratureMarginal(logA, lo, hi, 4000), b: quadratureMarginal(logB, lo, hi, 4000) },
    exactJoint: joint,
  }
}

/**
 * Two Gaussians and the observation that the first is larger: a hard comparison, whose tilted moments are those of a
 * truncated Gaussian in closed form (the greater-than factor).
 *
 * @index estimator EP atom: an ordered pair, a hard comparison (greater-than, truncation).
 * @example
 * ```ts
 * import { orderedPairAtom } from 'omai-engine/bayesian-examples'
 * orderedPairAtom.name // => 'ordered-pair'
 * ```
 */
export const orderedPairAtom: EpAtom = defineAtom({
  name: 'ordered-pair',
  title: 'Ordered pair (truncation)',
  what: 'Two independent Gaussians a and b with the observation that a > b: the greater-than factor, a truncation of their difference.',
  factors: ['prior (gaussian) x2', 'greater-than (truncation)'],
  family: 'gates',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.7, 1.2],
    ks: 0.12,
    reason:
      'the exact beliefs are skew-normal; EP matches their mean and variance with Gaussians, and the joint leans where the two beliefs cannot',
  },
  knobs: dict({ gap: real(-2, 2, { default: 0.5, step: 0.1, label: 'mean of b (a has mean 0)' }) }),
  problem: orderedProblem,
})

function orderedMeansProblem(k: Config): AtomProblem {
  const priorSd = exampleNumber(k, 'priorSd')
  const y1 = [0.8, 1.2]
  const y2 = [0.3, 0.6]
  const definition = defineModel('ordered-means', (m) => {
    const group = m.range('group', { size: 2 })
    const obs = m.range('obs')
    const mu = m.latent('mu', d.normal(0, priorSd), { dims: [group], constraint: { kind: 'ordered' } })
    const g = m.data('g', { dims: [obs], support: 'index', indexes: group })
    m.observe('y', d.normal(mu.at(g), 1), m.data('y', { dims: [obs], role: 'response' }))
  })
  const logDensity = (x: readonly number[]): number => {
    const [a, b] = [x[0] ?? 0, x[1] ?? 0]
    if (!(a < b)) return Number.NEGATIVE_INFINITY
    const lp = (v: number): number => -0.5 * (v / priorSd) ** 2
    const ll = (m0: number, ys: readonly number[]): number => ys.reduce((acc, v) => acc - 0.5 * (v - m0) ** 2, 0)
    return lp(a) + lp(b) + ll(a, y1) + ll(b, y2)
  }
  const g = gridMarginals(
    logDensity,
    [
      [-4, 4],
      [-4, 4],
    ],
    161,
  )
  return {
    definition,
    data: { g: [0, 0, 1, 1], y: [...y1, ...y2] },
    bindOptions: { sizes: { group: 2 } },
    equations: [
      `mu_1, mu_2 ~ N(0, ${priorSd}²), constrained to mu_1 < mu_2   (an ordered vector)`,
      'y ~ N(mu_group, 1²), two observations per group',
      `observe group 1: y = (${y1.join(', ')}),  group 2: y = (${y2.join(', ')})   (the data disagree with the order)`,
    ],
    quantities: [
      { name: 'mu[0]', label: 'mu_1', support: 'real' },
      { name: 'mu[1]', label: 'mu_2', support: 'real' },
    ],
    pair: ['mu[0]', 'mu[1]'],
    exact: { 'mu[0]': g.marginals[0] as never, 'mu[1]': g.marginals[1] as never },
    exactJoint: { a: 'mu[0]', b: 'mu[1]', ...g.joint(0, 1) },
  }
}

/**
 * Two means constrained to be ordered, whose data say the opposite: a constraint written as an ordered vector, which is a
 * chain of greater-than factors, so the posterior is the unconstrained one cut along the diagonal.
 *
 * @index estimator EP atom: two means with an ordered constraint, against data that disagree.
 * @example
 * ```ts
 * import { orderedMeansAtom } from 'omai-engine/bayesian-examples'
 * orderedMeansAtom.name // => 'ordered-means'
 * ```
 */
export const orderedMeansAtom: EpAtom = defineAtom({
  name: 'ordered-means',
  title: 'Ordered means',
  what: 'Two means with an ordered-vector constraint mu_1 < mu_2 and data that prefer the opposite order: the constraint is a greater-than factor between neighbours, and the posterior is cut along the diagonal.',
  factors: ['prior (gaussian) x2', 'ordered constraint (greater-than between neighbours)', 'gaussian likelihood'],
  family: 'gates',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.7, 1.2],
    ks: 0.12,
    reason: 'the truncated posterior is skewed; EP matches the mean and variance of each margin with a Gaussian',
  },
  knobs: dict({ priorSd: real(0.5, 6, { default: 3, step: 0.1, label: 'prior sd of the means' }) }),
  problem: orderedMeansProblem,
})

function componentsProblem(k: Config): AtomProblem {
  const y = exampleNumber(k, 'y')
  const [sd0, s] = [3, 1]
  const definition = defineModel('mixture-components', (m) => {
    const comp = m.range('component', { size: 2 })
    const z = m.latent('z', d.categorical({ probs: [0.5, 0.5] }))
    const mu = m.latent('mu', d.normal(0, sd0), { dims: [comp] })
    m.observe('y', d.normal(mu.at(z), s), m.data('y'))
  })
  const e = normalPdf(y, 0, Math.sqrt(sd0 ** 2 + s ** 2))
  const post = { mean: (sd0 ** 2 * y) / (sd0 ** 2 + s ** 2), sd: Math.sqrt((sd0 ** 2 * s ** 2) / (sd0 ** 2 + s ** 2)) }
  // The two components are exchangeable: whichever one z selects is updated, the other keeps its prior.
  const marginal = mixtureMarginal([
    { weight: 0.5 * e, mean: post.mean, sd: post.sd },
    { weight: 0.5 * e, mean: 0, sd: sd0 },
  ])
  return {
    definition,
    data: { y },
    bindOptions: { sizes: { component: 2 } },
    equations: [
      'z ~ Categorical(1/2, 1/2)   (which component)',
      `mu_1, mu_2 ~ N(0, ${sd0}²)   (the components' means, random)`,
      `y ~ N(mu_z, ${s}²)   (the component picked by a random index)`,
      `observe y = ${y}`,
    ],
    quantities: [
      { name: 'mu[0]', label: 'mu_1', support: 'real' },
      { name: 'mu[1]', label: 'mu_2', support: 'real' },
    ],
    exact: { 'mu[0]': marginal, 'mu[1]': marginal },
  }
}

/**
 * A component chosen by a random index among components that are themselves random: the gate written with an index
 * (`mu[z]`) instead of a select.
 *
 * @index estimator EP atom: a random index into random components (mu[z]).
 * @example
 * ```ts
 * import { mixtureComponentsAtom } from 'omai-engine/bayesian-examples'
 * mixtureComponentsAtom.name // => 'mixture-components'
 * ```
 */
export const mixtureComponentsAtom: EpAtom = defineAtom({
  name: 'mixture-components',
  title: 'Random index into random components',
  what: 'Two random component means mu_1 and mu_2 and a random index z that picks one of them for the observation y: the gate written as an index, mu[z], whose exact marginals are two-component mixtures.',
  factors: [
    'prior (categorical)',
    'prior (gaussian) x2',
    'index gate (random index, random components)',
    'gaussian likelihood',
  ],
  family: 'gates',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.3,
    sdRatio: [0.6, 1.6],
    ks: 0.25,
    reason: 'the exact marginals are two-component mixtures; EP keeps one Gaussian for each',
  },
  knobs: dict({ y: real(-4, 4, { default: 1.4, step: 0.1, label: 'observed y' }) }),
  problem: componentsProblem,
})
