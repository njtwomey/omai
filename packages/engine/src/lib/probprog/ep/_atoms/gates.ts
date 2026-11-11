/**
 * The setups of expectation propagation that are a switch between components, each written as the gate it is, with an explicit switch so that EP reports the probability of each branch and not only a blended belief: a switch between known and random values, the spike and slab prior (with the switch explicit or summed out), Minka's clutter problem with a switch per observation, a mixture of regressions, and an ordered pair, each with its learned twin where the unknowns are random and the data simulated. The exact answers are closed forms, quadratures or a Gibbs sampler (`validation/references`); EP's single Gaussian for a mixed quantity is drawn as what it is, the moment-matched projection of that mixture.
 */

import { foldIn, key, normal, uniform } from '../../../random/index.ts'
import { dict, int, real } from '../../../space/index.ts'
import { toFlat } from '../../../tensor/index.ts'
import type { Config } from '../../../trace/index.ts'
import { configNumber } from '../../inference/method.ts'
import * as d from '../../model/dists/index.ts'
import { defineModel, ops } from '../../model/index.ts'
import {
  clutterPosterior,
  type ExactMarginal,
  gridMarginals,
  logWeights,
  mixtureGibbs,
  mixtureMarginal,
  mixtureOfRegressions,
  momentMatch,
  type NormalComponent,
  normalCdf,
  normalPdf,
  outlierMixturePosterior,
  probabilityMarginal,
  quadratureMarginal,
  type RefJoint,
  samplesMarginal,
  selectedBranch,
  spikeSlabPosterior,
  type TwoComponentGate,
  twoComponentGate,
  weightedMarginal,
} from '../../validation/index.ts'
import { defineAtom } from '../_define.ts'
import { recovery, tolerance } from '../_tolerance.ts'
import type { AtomParts, AtomProblem, AtomSupport, Belief, EpAtom, MixtureBranch, MixtureView } from '../atom-types.ts'
import { simulationKnobs } from './conjugate.ts'

function orderedProblem(k: Config): AtomParts {
  const gap = configNumber(k, 'gap')
  const [ma, mb, s] = [0, gap, 1]
  // region ordered-pair
  const definition = defineModel('ordered-pair', (m) => {
    const a = m.latent('a', d.normal(ma, s))
    const b = m.latent('b', d.normal(mb, s))
    const observed = m.data('ordered', { support: 'binary' })
    m.observe('ordered', d.bernoulli({ probs: ops.gt(ops.sub(a, b), 0) }), observed)
  })
  // endregion
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
    build: {
      definition,
      data: { ordered: 1 },
      quantities: [
        { name: 'a', label: 'a', support: 'real' },
        { name: 'b', label: 'b', support: 'real' },
      ],
    },
    reference: {
      exact: { a: quadratureMarginal(logA, lo, hi, 4000), b: quadratureMarginal(logB, lo, hi, 4000) },
      exactJoint: joint,
    },
    present: {
      equations: [`a ~ N(${ma}, ${s}²)`, `b ~ N(${mb}, ${s}²)`, 'observe a > b   (a hard comparison: a truncation)'],
      pair: ['a', 'b'],
    },
  }
}

/**
 * Two Gaussians and the observation that the first is larger: a hard comparison, whose tilted moments are those of a
 * truncated Gaussian in closed form (the greater-than factor).
 */
const orderedPairAtom: EpAtom = defineAtom({
  name: 'ordered-pair',
  title: 'Ordered pair (truncation)',
  what: 'Two independent Gaussians a and b with the observation that a > b: the greater-than factor, a truncation of their difference.',
  factors: ['prior (gaussian) x2', 'greater-than (truncation)'],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance(
    'skew-normal',
    'the exact beliefs are skew-normal; EP matches their mean and variance with Gaussians, and the joint leans where the two beliefs cannot',
  ),
  knobs: dict({ gap: real(-2, 2, { default: 0.5, step: 0.1, label: 'mean of b (a has mean 0)' }) }),
  parts: orderedProblem,
})

function orderedMeansProblem(k: Config): AtomParts {
  const priorSd = configNumber(k, 'priorSd')
  const y1 = [0.8, 1.2]
  const y2 = [0.3, 0.6]
  // region ordered-means
  const definition = defineModel('ordered-means', (m) => {
    const group = m.range('group', { size: 2 })
    const obs = m.range('obs')
    const mu = m.latent('mu', d.normal(0, priorSd), {
      dims: [group],
      constraint: { kind: 'ordered' },
    })
    const g = m.data('g', { dims: [obs], support: 'index', indexes: group })
    m.observe('y', d.normal(mu.at(g), 1), m.data('y', { dims: [obs], role: 'response' }))
  })
  // endregion
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
    build: {
      definition,
      data: { g: [0, 0, 1, 1], y: [...y1, ...y2] },
      bindOptions: { sizes: { group: 2 } },
      quantities: [
        { name: 'mu[0]', label: 'mu_1', support: 'real' },
        { name: 'mu[1]', label: 'mu_2', support: 'real' },
      ],
    },
    reference: {
      exact: { 'mu[0]': g.marginal(0), 'mu[1]': g.marginal(1) },
      exactJoint: { a: 'mu[0]', b: 'mu[1]', ...g.joint(0, 1) },
    },
    present: {
      equations: [
        `mu_1, mu_2 ~ N(0, ${priorSd}²), constrained to mu_1 < mu_2   (an ordered vector)`,
        'y ~ N(mu_group, 1²), two observations per group',
        `observe group 1: y = (${y1.join(', ')}),  group 2: y = (${y2.join(', ')})   (the data disagree with the order)`,
      ],
      pair: ['mu[0]', 'mu[1]'],
    },
  }
}

/**
 * Two means constrained to be ordered, whose data say the opposite: a constraint written as an ordered vector, which is a
 * chain of greater-than factors, so the posterior is the unconstrained one cut along the diagonal.
 */
const orderedMeansAtom: EpAtom = defineAtom({
  name: 'ordered-means',
  title: 'Ordered means',
  what: 'Two means with an ordered-vector constraint mu_1 < mu_2 and data that prefer the opposite order: the constraint is a greater-than factor between neighbours, and the posterior is cut along the diagonal.',
  factors: ['prior (gaussian) x2', 'ordered constraint (greater-than between neighbours)', 'gaussian likelihood'],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance(
    'skew-normal',
    'the truncated posterior is skewed; EP matches the mean and variance of each margin with a Gaussian',
  ),
  evidence: { nats: 1e-3, reason: 'an ordering constraint on two Gaussians' },
  knobs: dict({ priorSd: real(0.5, 6, { default: 3, step: 0.1, label: 'prior sd of the means' }) }),
  parts: orderedMeansProblem,
})

/** EP's belief that is not a belief of the model but of a function of its beliefs, proper when the pieces are. */
const belief = (m: { mean: number; sd: number } | undefined): Belief =>
  m === undefined ? { mean: Number.NaN, sd: Number.NaN, proper: false } : { ...m, proper: true }

/**
 * The branches of a gate's selected value as EP's beliefs imply them: `names[0]` is component 0 (selected with probability
 * 1 - P(z = 1)) and `names[1]` component 1, each recovered from the component's marginal belief and the switch's.
 */
function epBranches(
  b: Readonly<Record<string, Belief>>,
  names: readonly [string, string],
  sites: readonly [string, string],
  priors: readonly [NormalComponent, NormalComponent],
  z: string,
): readonly [MixtureBranch, MixtureBranch] | undefined {
  const p1 = b[z]?.mean
  const c0 = b[sites[0]]
  const c1 = b[sites[1]]
  if (p1 === undefined || c0 === undefined || c1 === undefined || !c0.proper || !c1.proper) return undefined
  const s0 = selectedBranch(c0, 1 - p1, priors[0])
  const s1 = selectedBranch(c1, p1, priors[1])
  return s0 === undefined || s1 === undefined
    ? undefined
    : [
        { label: names[0], weight: 1 - p1, ...s0 },
        { label: names[1], weight: p1, ...s1 },
      ]
}

/** The mean and sd of what a gate selects, from EP's beliefs: the moment match of its branches. */
function epSelected(
  b: Readonly<Record<string, Belief>>,
  sites: readonly [string, string],
  priors: readonly [NormalComponent, NormalComponent],
  z: string,
): Belief {
  const br = epBranches(b, ['0', '1'], sites, priors, z)
  return br === undefined ? belief(undefined) : belief(momentMatch(br))
}

/** The view of a quantity that is a component of a gate: selected (its update) or not (its prior). */
function componentView(
  quantity: string,
  label: string,
  index: 0 | 1,
  gate: TwoComponentGate,
  prior: NormalComponent,
  z: string,
): MixtureView {
  const pSel = index === 1 ? gate.p1 : 1 - gate.p1
  return {
    title: `${label}: updated by y if the switch selects it, its prior if not`,
    quantity,
    label,
    exact: [
      { label: 'not selected: prior', weight: 1 - pSel, ...prior },
      { label: 'selected: updated', weight: pSel, ...gate.updated[index] },
    ],
    ep: (b) => {
      const p1 = b[z]?.mean
      const c = b[quantity]
      if (p1 === undefined || c === undefined || !c.proper) return undefined
      const p = index === 1 ? p1 : 1 - p1
      const sel = selectedBranch(c, p, prior)
      return sel === undefined
        ? undefined
        : [
            { label: 'not selected: prior', weight: 1 - p, ...prior },
            { label: 'selected: updated', weight: p, ...sel },
          ]
    },
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// A switch between two known values.

function knownGateProblem(k: Config): AtomParts {
  const pi = configNumber(k, 'pi')
  const y = configNumber(k, 'y')
  // region gate
  const definition = defineModel('gate-known', (m) => {
    const z = m.latent('z', d.bernoulli({ probs: pi }))
    m.observe('y', d.normal(ops.where(z, 3, 0), 1), m.data('y'))
  })
  // endregion
  const w1 = pi * normalPdf(y, 3, 1)
  const w0 = (1 - pi) * normalPdf(y, 0, 1)
  const p = w1 / (w1 + w0)
  const ys = Array.from({ length: 141 }, (_, i) => -4 + (i * 11) / 140)
  return {
    build: {
      definition,
      data: { y },
      quantities: [{ name: 'z', label: 'P(z = 1)', support: 'probability' }],
    },
    reference: {
      exact: { z: probabilityMarginal(p) },
    },
    present: {
      equations: [
        `z ~ Bernoulli(${pi})`,
        'g = z ? 3 : 0   (the gate: a select between two known values)',
        'y ~ N(g, 1²)',
        `observe y = ${y}`,
      ],
      figure: {
        caption:
          'Each branch of the gate explains the observation with its own weight: the prior probability of the branch times the density of y under it. The posterior of z is the share of the taller one.',
        xLabel: 'y',
        yLabel: 'prior weight x density of y',
        xLimits: [-4, 7],
        points: [{ name: 'observed y', x: [y], y: [0] }],
        curves: [
          {
            name: `z = 0: ${(1 - pi).toFixed(2)} N(y; 0, 1)`,
            colour: 'tab1',
            x: ys,
            y: ys.map((v) => (1 - pi) * normalPdf(v, 0, 1)),
          },
          { name: `z = 1: ${pi} N(y; 3, 1)`, colour: 'tab2', x: ys, y: ys.map((v) => pi * normalPdf(v, 3, 1)) },
        ],
      },
    },
  }
}

/**
 * A switch that selects one of two known values, observed through Gaussian noise: the select gate with known components.
 * The message to the switch is the ratio of the two branches' evidence.
 */
const gateKnownAtom: EpAtom = defineAtom({
  name: 'gate',
  title: 'Gate',
  what: 'A Bernoulli switch z selects between the known values 0 and 3, and y is that value plus Gaussian noise: the select factor, whose message to z is the ratio of the two branches’ evidence.',
  factors: ['prior (bernoulli)', 'select gate (known components)', 'gaussian likelihood'],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance('exact', 'a single discrete switch: the message is exact'),
  knobs: dict({
    pi: real(0.02, 0.98, { default: 0.3, step: 0.01, label: 'P(z = 1) before the observation' }),
    y: real(-3, 6, { default: 2.5, step: 0.1, label: 'observed y' }),
  }),
  parts: knownGateProblem,
})

// ---------------------------------------------------------------------------------------------------------------------
// A switch between two random values, as a select and as an index.

function randomGateProblem(k: Config, form: 'select' | 'index'): AtomParts {
  const pi = configNumber(k, 'pi')
  const y = configNumber(k, 'y')
  const s = 0.5
  const priors: readonly [NormalComponent, NormalComponent] = [
    { mean: -2, sd: 1 },
    { mean: 2, sd: 1 },
  ]
  const sites: readonly [string, string] = form === 'select' ? ['a', 'b'] : ['mu[0]', 'mu[1]']
  // region gate-random mixture-components
  const definition =
    form === 'select'
      ? defineModel('gate-random', (m) => {
          const z = m.latent('z', d.bernoulli({ probs: pi }))
          const a = m.latent('a', d.normal(priors[0].mean, priors[0].sd))
          const b = m.latent('b', d.normal(priors[1].mean, priors[1].sd))
          m.observe('y', d.normal(ops.where(z, b, a), s), m.data('y'))
        })
      : defineModel('mixture-components', (m) => {
          const comp = m.range('component', { size: 2 })
          const loc = m.data('loc', { dims: [comp] })
          const z = m.latent('z', d.categorical({ probs: [1 - pi, pi] }))
          const mu = m.latent('mu', d.normal(loc, 1), { dims: [comp] })
          m.observe('y', d.normal(mu.at(z), s), m.data('y'))
        })
  // endregion
  const gate = twoComponentGate(pi, priors, y, s)
  const names: readonly [string, string] = form === 'select' ? ['a', 'b'] : ['mu_1', 'mu_2']
  const selectedView: MixtureView = {
    title: `The value the switch selects (${form === 'select' ? 'g' : 'mu[z]'}): the branch of each component, weighted by the posterior of z`,
    quantity: 'g',
    label: form === 'select' ? 'g' : 'mu[z]',
    exact: [
      { label: `z = 0: ${names[0]}`, weight: 1 - gate.p1, ...gate.updated[0] },
      { label: `z = 1: ${names[1]}`, weight: gate.p1, ...gate.updated[1] },
    ],
    ep: (b) => epBranches(b, [`z = 0: ${names[0]}`, `z = 1: ${names[1]}`], sites, priors, 'z'),
  }
  return {
    build: {
      definition,
      data: form === 'select' ? { y } : { y, loc: [priors[0].mean, priors[1].mean] },
      ...(form === 'select' ? {} : { bindOptions: { sizes: { component: 2 } } }),
      quantities: [
        { name: 'z', label: 'P(z = 1)', support: 'probability' },
        { name: sites[0], label: names[0], support: 'real' },
        { name: sites[1], label: names[1], support: 'real' },
        {
          name: 'g',
          label: form === 'select' ? 'g (the selected value)' : 'mu[z] (the selected value)',
          support: 'real',
          derive: (b) => epSelected(b, sites, priors, 'z'),
        },
      ],
    },
    reference: {
      exact: {
        z: probabilityMarginal(gate.p1),
        [sites[0]]: gate.marginals[0],
        [sites[1]]: gate.marginals[1],
        g: gate.selected,
      },
    },
    present: {
      equations:
        form === 'select'
          ? [
              `z ~ Bernoulli(${pi})`,
              `a ~ N(${priors[0].mean}, ${priors[0].sd}²),  b ~ N(${priors[1].mean}, ${priors[1].sd}²)`,
              'g = z ? b : a   (the gate: a select between two random variables)',
              `y ~ N(g, ${s}²)`,
              `observe y = ${y}`,
            ]
          : [
              `z ~ Categorical(${(1 - pi).toFixed(2)}, ${pi})   (which component)`,
              `mu_1 ~ N(${priors[0].mean}, ${priors[0].sd}²),  mu_2 ~ N(${priors[1].mean}, ${priors[1].sd}²)`,
              `y ~ N(mu[z], ${s}²)   (the component picked by a random index)`,
              `observe y = ${y}`,
            ],
      mixtures: [
        selectedView,
        componentView(sites[0], names[0], 0, gate, priors[0], 'z'),
        componentView(sites[1], names[1], 1, gate, priors[1], 'z'),
      ],
    },
  }
}

const gateKnobs = dict({
  pi: real(0.02, 0.98, { default: 0.5, step: 0.01, label: 'P(z = 1) before the observation' }),
  y: real(-4, 4, { default: 0.5, step: 0.1, label: 'observed y' }),
})
const GATE_REASON =
  'the means, spreads and the probability of the switch are exact (moment matching through a gate); the Kolmogorov distance is large because each exact marginal is a mixture and EP draws one Gaussian for it'

/**
 * A switch that selects one of two random variables, written as a select, `where(z, b, a)`: the selected one is pulled
 * towards the observation, the other keeps its prior.
 */
const gateRandomAtom: EpAtom = defineAtom({
  name: 'gate-random',
  title: 'Gate between two random values',
  what: 'A Bernoulli switch z selects one of two Gaussian variables a and b, and y is the selected one plus noise. EP reports the probability of z; the exact posterior of what is selected is the mixture of the two updated components, weighted by it.',
  factors: ['prior (bernoulli)', 'prior (gaussian) x2', 'select gate (fresh random components)', 'gaussian likelihood'],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance('exact-moments', GATE_REASON),
  evidence: { nats: 1e-3, reason: 'the gate: a mixture of branch evidences' },
  knobs: gateKnobs,
  parts: (k) => randomGateProblem(k, 'select'),
})

/**
 * The same gate written as an index, `mu[z]`, with a categorical switch.
 */
const mixtureComponentsAtom: EpAtom = defineAtom({
  name: 'mixture-components',
  title: 'Random index into random components',
  what: 'The same switch written as an index: a categorical z picks component mu[z] of two random component means for the observation y. EP reports the probability of each component; the exact posterior of the selected value is the mixture of the updated components.',
  factors: [
    'prior (categorical)',
    'prior (gaussian) x2',
    'index gate (random index, random components)',
    'gaussian likelihood',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance('exact-moments', GATE_REASON),
  evidence: { nats: 1e-3, reason: 'the gate again, written with components' },
  knobs: gateKnobs,
  parts: (k) => randomGateProblem(k, 'index'),
})

// ---------------------------------------------------------------------------------------------------------------------
// Spike and slab.

const SPIKE_NOISE = 0.5

const spikeKnobs = dict({
  pi: real(0.05, 0.95, { default: 0.5, step: 0.01, label: 'prior probability of the slab' }),
  y: real(-5, 6, { default: 0.8, step: 0.1, label: 'observed y' }),
  slab: real(1, 6, { default: 2, step: 0.1, label: 'slab width (sd)' }),
  spike: real(0.01, 0.5, { default: 0.1, step: 0.01, label: 'spike width (sd)' }),
})

function spikeSlabGateProblem(k: Config): AtomParts {
  const pi = configNumber(k, 'pi')
  const y = configNumber(k, 'y')
  const slab = configNumber(k, 'slab')
  const spike = configNumber(k, 'spike')
  const s = SPIKE_NOISE
  const priors: readonly [NormalComponent, NormalComponent] = [
    { mean: 0, sd: spike },
    { mean: 0, sd: slab },
  ]
  // region spike-and-slab
  const definition = defineModel('spike-and-slab', (m) => {
    const z = m.latent('z', d.bernoulli({ probs: pi }))
    const narrow = m.latent('spike', d.normal(0, spike))
    const wide = m.latent('slab', d.normal(0, slab))
    m.observe('y', d.normal(ops.where(z, wide, narrow), s), m.data('y'))
  })
  // endregion
  const gate = spikeSlabPosterior(pi, slab, spike, y, s)
  const sites: readonly [string, string] = ['spike', 'slab']
  return {
    build: {
      definition,
      data: { y },
      quantities: [
        { name: 'z', label: 'P(slab)', support: 'probability' },
        { name: 'spike', label: 'spike component', support: 'real', hidden: true },
        { name: 'slab', label: 'slab component', support: 'real', hidden: true },
        { name: 'x', label: 'x (the coefficient)', support: 'real', derive: (b) => epSelected(b, sites, priors, 'z') },
      ],
    },
    reference: {
      exact: { z: probabilityMarginal(gate.p1), spike: gate.marginals[0], slab: gate.marginals[1], x: gate.selected },
    },
    present: {
      equations: [
        `z ~ Bernoulli(${pi})   (z = 1: the coefficient is in the slab)`,
        `spike ~ N(0, ${spike}²),  slab ~ N(0, ${slab}²)   (a narrow and a wide component)`,
        'x = z ? slab : spike   (the gate: the coefficient is the component z selects)',
        `y ~ N(x, ${s}²)`,
        `observe y = ${y}`,
      ],
      mixtures: [
        {
          title: 'The coefficient x: the spike branch and the slab branch, each weighted by its posterior probability',
          quantity: 'x',
          label: 'x',
          exact: [
            { label: 'spike (z = 0)', weight: 1 - gate.p1, ...gate.updated[0] },
            { label: 'slab (z = 1)', weight: gate.p1, ...gate.updated[1] },
          ],
          ep: (b) => epBranches(b, ['spike (z = 0)', 'slab (z = 1)'], sites, priors, 'z'),
        },
      ],
    },
  }
}

/**
 * A coefficient that is a switch between a very narrow Gaussian (the spike) and a wide one (the slab): the gate with
 * explicit `z`, so EP reports the posterior probability of the slab and the two branches, not only one blended belief.
 */
const spikeSlabAtom: EpAtom = defineAtom({
  name: 'spike-and-slab',
  title: 'Spike and slab',
  what: 'A coefficient that is a gate between a very narrow Gaussian (the spike) and a wide one (the slab), chosen by a Bernoulli switch z, observed through noise. EP reports P(slab) and each branch; the exact posterior of the coefficient is the two branches weighted by it.',
  factors: [
    'prior (bernoulli)',
    'prior (gaussian) x2',
    'select gate (spike and slab components)',
    'gaussian likelihood',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance('exact-moments', GATE_REASON),
  evidence: { nats: 1e-3, reason: 'a gate between a narrow and a wide Gaussian' },
  knobs: spikeKnobs,
  parts: spikeSlabGateProblem,
})

function spikeSlabMarginalProblem(k: Config): AtomParts {
  const pi = configNumber(k, 'pi')
  const y = configNumber(k, 'y')
  const slab = configNumber(k, 'slab')
  const spike = configNumber(k, 'spike')
  const s = SPIKE_NOISE
  // region spike-and-slab-marginal
  const definition = defineModel('spike-and-slab-marginal', (m) => {
    const w = m.latent('w', d.normalSpikeSlab(pi, slab, spike))
    m.observe('y', d.normal(w, s), m.data('y'))
  })
  // endregion
  const gate = spikeSlabPosterior(pi, slab, spike, y, s)
  return {
    build: {
      definition,
      data: { y },
      quantities: [{ name: 'w', label: 'w (the coefficient)', support: 'real' }],
    },
    reference: {
      exact: { w: gate.selected },
    },
    present: {
      equations: [
        `w ~ ${pi} N(0, ${slab}²) + ${(1 - pi).toFixed(2)} N(0, ${spike}²)   (spike and slab, the switch summed out)`,
        `y ~ N(w, ${s}²)`,
        `observe y = ${y}`,
      ],
      mixtures: [
        {
          title:
            'The coefficient w as the gate it is: the exact spike and slab branches. This form has no switch variable, so EP reports only its single Gaussian',
          quantity: 'w',
          label: 'w',
          exact: [
            { label: 'spike (z = 0)', weight: 1 - gate.p1, ...gate.updated[0] },
            { label: 'slab (z = 1)', weight: gate.p1, ...gate.updated[1] },
          ],
          ep: () => undefined,
        },
      ],
    },
  }
}

/**
 * The same prior with the switch summed out, as the library's spike and slab family: a mixture site on one variable, whose
 * moments are exact but which has no variable for the probability of the slab.
 */
const spikeSlabMarginalAtom: EpAtom = defineAtom({
  name: 'spike-and-slab-marginal',
  title: 'Spike and slab, switch summed out',
  what: 'The same prior with the switch summed out (the mixture-site form): one variable with a two-component zero-centred prior. EP matches the mean and variance of the exact two-branch posterior with one Gaussian and has no variable for the probability of the slab.',
  factors: ['mixture-site (two-component prior)', 'gaussian likelihood'],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance(
    'measured',
    'the mean and variance are exact; the exact posterior is a spike hump and a slab hump and EP draws one Gaussian',
    { meanZ: 0.001, sdRatio: [0.999, 1.001], ks: 0.3 },
  ),
  evidence: { nats: 1e-3, reason: 'the mixture site, switch summed out: exact' },
  knobs: spikeKnobs,
  parts: spikeSlabMarginalProblem,
})

// ---------------------------------------------------------------------------------------------------------------------
// Clutter.

const CLUTTER_Y = [1.0, 1.4, 0.9, 1.2, 1.6, -4.2, 1.1, 9.5]
const INLIER_SD = 0.5
const CLUTTER_SD = 5

function clutterGateProblem(k: Config): AtomParts {
  const w = configNumber(k, 'share')
  const priorSd = 5
  // region clutter
  const definition = defineModel('clutter', (m) => {
    const row = m.range('i')
    const theta = m.latent('theta', d.normal(0, priorSd))
    const z = m.latent('z', d.bernoulli({ probs: w }), { dims: [row] })
    const clutter = z.gt(0.5)
    m.observe(
      'y',
      d.normal(ops.where(clutter, 0, theta), ops.where(clutter, CLUTTER_SD, INLIER_SD)),
      m.data('y', { dims: [row], role: 'response' }),
    )
  })
  // endregion
  const logPost = (t: number): number =>
    Math.log(normalPdf(t, 0, priorSd)) +
    CLUTTER_Y.reduce((a, v) => a + Math.log((1 - w) * normalPdf(v, t, INLIER_SD) + w * normalPdf(v, 0, CLUTTER_SD)), 0)
  const marginal = quadratureMarginal(logPost, -20, 20)
  // P(observation i is clutter) = the posterior of the location averaged over the branch weight of that observation.
  const M = 8000
  const ts = Array.from({ length: M }, (_, i) => -20 + ((i + 0.5) * 40) / M)
  const lw = ts.map(logPost)
  const { weights: wt } = logWeights(lw)
  const total = wt.reduce((a, b) => a + b, 0)
  const pClutter = CLUTTER_Y.map((v) =>
    ts.reduce((acc, t, j) => {
      const c = w * normalPdf(v, 0, CLUTTER_SD)
      return acc + ((wt[j] ?? 0) / total) * (c / (c + (1 - w) * normalPdf(v, t, INLIER_SD)))
    }, 0),
  )
  const at = CLUTTER_Y.map((_, i) => i + 1)
  const n = CLUTTER_Y.length
  return {
    build: {
      definition,
      data: { y: CLUTTER_Y },
      quantities: [
        { name: 'theta', label: 'theta (the centre)', support: 'real' },
        ...at.map((i) => ({
          name: `z[${i - 1}]`,
          label: `observation ${i} is clutter`,
          support: 'probability' as const,
        })),
      ],
    },
    reference: {
      exact: { theta: marginal, ...Object.fromEntries(pClutter.map((p, i) => [`z[${i}]`, probabilityMarginal(p)])) },
    },
    present: {
      equations: [
        `theta ~ N(0, ${priorSd}²)   (the hidden centre)`,
        `z_i ~ Bernoulli(${w})   (z_i = 1: observation i is clutter)`,
        `y_i ~ z_i ? N(0, ${CLUTTER_SD}²) : N(theta, ${INLIER_SD}²)   (a gate per observation: clutter or the centre)`,
        `observe y = (${CLUTTER_Y.join(', ')})`,
      ],
      figure: {
        caption:
          'The eight observations, coloured by whether EP believes each is clutter (a draw from the wide component around 0) or a detection of the centre (the line). The two bands are the two branches of every gate.',
        xLabel: 'observation',
        yLabel: 'value',
        xLimits: [0, n + 1],
        points: [
          {
            name: 'observations',
            x: at,
            y: [...CLUTTER_Y],
            group: pClutter.map((p) => (p > 0.5 ? 1 : 0)),
            groupNames: { 0: 'centre', 1: 'clutter' },
            groupFrom: (m) => at.map((i) => ((m[`z[${i - 1}]`] ?? 0) > 0.5 ? 1 : 0)),
          },
        ],
        curves: [
          { name: 'clutter branch: 0 ± 2 sd', colour: 'tab1', x: [0, n + 1], y: [2 * CLUTTER_SD, 2 * CLUTTER_SD] },
          { name: 'clutter branch: 0 ± 2 sd', colour: 'tab1', x: [0, n + 1], y: [-2 * CLUTTER_SD, -2 * CLUTTER_SD] },
        ],
        answer: (m) => [{ name: 'theta', x: [0, n + 1], y: [m['theta'] ?? Number.NaN, m['theta'] ?? Number.NaN] }],
      },
    },
  }
}

/**
 * Minka's clutter problem with an explicit switch per observation: each observation is a draw from a wide clutter component
 * around zero or a detection of the hidden centre, so EP reports the probability that each is clutter as well as the belief
 * about the centre.
 */
const clutterAtom: EpAtom = defineAtom({
  name: 'clutter',
  title: 'Clutter',
  what: 'A hidden centre theta observed eight times, each observation a gate between a detection of the centre and wide clutter around zero (a switch z_i per observation, two of which look like clutter). EP reports the probability that each is clutter and the belief about the centre.',
  factors: [
    'prior (gaussian)',
    'prior (bernoulli) per observation',
    'select gate per observation (random centre, known clutter)',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance(
    'lumpiest',
    'the exact posterior of the centre is lumpy (a clutter reading can be explained two ways) and EP matches only its mean and variance, observation by observation',
  ),
  evidence: { nats: 1e-3, reason: 'a switch per observation: exact moment matching' },
  knobs: dict({ share: real(0.02, 0.6, { default: 0.25, step: 0.01, label: 'prior probability of clutter' }) }),
  ep: { damping: 0.5 },
  parts: clutterGateProblem,
})

function clutterMixtureProblem(k: Config): AtomParts {
  const share = configNumber(k, 'share')
  const [s, width, priorSd] = [0.5, 10, 5]
  // region clutter-mixture
  const definition = defineModel('clutter-mixture', (m) => {
    const mu = m.latent('mu', d.normal(0, priorSd))
    const observed = m.data('y', { dims: ['i'], role: 'response' })
    m.observe('y', d.normalOutliers(mu, s, share, width), observed)
  })
  // endregion
  const logPost = (x: number): number =>
    Math.log(normalPdf(x, 0, priorSd)) +
    CLUTTER_Y.reduce((a, v) => a + Math.log((1 - share) * normalPdf(v, x, s) + share * normalPdf(v, x, width * s)), 0)
  return {
    build: {
      definition,
      data: { y: CLUTTER_Y },
      quantities: [{ name: 'mu', label: 'mu', support: 'real' }],
    },
    reference: {
      exact: { mu: quadratureMarginal(logPost, -20, 20) },
    },
    present: {
      equations: [
        `mu ~ N(0, ${priorSd}²)`,
        `y_i ~ ${(1 - share).toFixed(2)} N(mu, ${s}²) + ${share} N(mu, ${width * s}²)   (inlier or outlier, the switch summed out)`,
        `observe y = (${CLUTTER_Y.join(', ')})`,
      ],
    },
  }
}

/**
 * The inlier and outlier mixture likelihood with the switches summed out (the library's `normalOutliers` family): a mixture
 * site per observation around the centre, which has no variable for the probability that an observation is an outlier.
 */
const clutterMixtureAtom: EpAtom = defineAtom({
  name: 'clutter-mixture',
  title: 'Inlier and outlier mixture, switches summed out',
  what: 'An unknown mean mu observed eight times through a mixture of a narrow and a wide Gaussian around it, two observations being far from the rest: the mixture site as a likelihood. The switch per observation is summed out, so EP gives a belief about mu and no probability that an observation is an outlier.',
  factors: ['prior (gaussian)', 'mixture-site (inlier and outlier noise)'],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance('lumpy', 'the posterior is lumpy and EP matches only its mean and variance with one Gaussian'),
  knobs: dict({ share: real(0.02, 0.6, { default: 0.25, step: 0.01, label: 'outlier share' }) }),
  parts: clutterMixtureProblem,
})

// Two lines, y = 1 + 2 x and y = -1 - x, eight rows whose classes are not told.
const XS = [-1.5, -0.5, 0.2, 1.1, -0.9, 0.7, 1.6, -1.2]
const YS = [-2.1, 0.2, 1.4, 3.3, -0.1, -1.6, -2.4, -0.3]
// The prior means (an intercept and a slope of each class) tell the classes apart softly, which fixes the labels.
const PRIOR_MEAN = [
  [1, 1.5],
  [-1, -1.5],
]

function mixtureRegressionProblem(k: Config): AtomParts {
  const noise = configNumber(k, 'noise')
  const priorSd = configNumber(k, 'priorSd')
  const design = XS.map((x) => [1, x])
  // region mixture-regression
  const definition = defineModel('mixture-regression', (m) => {
    const row = m.range('row')
    const feature = m.range('feature', { size: 2 })
    const component = m.range('component', { size: 2 })
    const x = m.data('X', { dims: [row, feature] })
    const y = m.data('y', { dims: [row], role: 'response' })
    // an intercept and a slope for each class, with priors that tell the classes apart
    const prior = m.const(
      [
        [1, -1],
        [1.5, -1.5],
      ],
      [feature, component],
    )
    const b = m.latent('B', d.normal(prior, priorSd), { dims: [feature, component] })
    // the class of each row, and the prediction of that class for the row
    const z = m.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: [row] })
    const mean = ops.pick(ops.matmul(x, b), z)
    m.observe('y', d.normal(mean, noise), y)
  })
  // endregion
  const post = mixtureOfRegressions({ X: design, y: YS, priorMean: PRIOR_MEAN, priorSd, noiseSd: noise })
  const names = ['intercept of class 1', 'intercept of class 2', 'slope of class 1', 'slope of class 2']
  return {
    build: {
      definition,
      data: { X: design, y: YS },
      quantities: [0, 1, 2, 3].map((i) => ({
        name: `B[${i}]`,
        label: names[i] ?? `B_${i}`,
        support: 'real' as const,
      })),
    },
    reference: {
      exact: Object.fromEntries(
        [0, 1, 2, 3].map((i) => {
          const feature = Math.floor(i / 2)
          const klass = i % 2
          return [`B[${i}]`, mixtureMarginal(((post.coefficients[klass] ?? [])[feature]?.components ?? []).slice())]
        }),
      ),
      referenceSource: 'the 256 assignments of the eight rows to the two classes, each class conjugate given its rows',
    },
    present: {
      equations: [
        `B_class,0 and B_class,1 (an intercept and a slope) ~ N(prior mean, ${priorSd}²), prior means (1, 1.5) and (-1, -1.5)`,
        `z_i ~ Categorical(1/2, 1/2),  y_i ~ N(B_z[i],0 + B_z[i],1 x_i, ${noise}²)`,
        `observe the eight points drawn in the figure`,
      ],
      figure: {
        caption: 'The eight points and the two lines at the means of the beliefs.',
        xLabel: 'x',
        yLabel: 'y',
        xLimits: [-2, 2],
        yLimits: [-3.5, 4],
        points: [{ name: 'rows', x: XS, y: YS }],
        answer: (mean) =>
          [0, 1].map((c) => ({
            name: `class ${c + 1}`,
            x: [-2, 2],
            y: [-2, 2].map((x) => (mean[`B[${c}]`] ?? Number.NaN) + (mean[`B[${2 + c}]`] ?? Number.NaN) * x),
          })),
      },
    },
  }
}

/**
 * A mixture of two linear regressions on a shared design: eight points on two lines, a class per row chosen by a categorical
 * latent, the row's prediction picked from its class's line. The exact posterior enumerates the class assignments.
 */
const mixtureRegressionAtom: EpAtom = defineAtom({
  name: 'mixture-regression',
  title: 'Mixture of regressions (a class picks the line)',
  what: 'Eight points on two lines with a categorical class per row and a coefficient vector per class: the class picks the prediction of its branch, one gate of Gaussian likelihoods per row, against the enumeration of the assignments.',
  factors: [
    'prior (gaussian) x4',
    'dot-constant (the prediction of each class)',
    'gate (the class picks the branch of Gaussian likelihoods)',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance(
    'measured',
    'the coefficients are shared by every row, so the classes of the rows are coupled through them and EP matches one Gaussian per coefficient to a mixture of Gaussians over the assignments: measured within 0.14 reference sds, spreads down to 0.72 of the reference',
    { meanZ: 0.2, sdRatio: [0.65, 1.2], ks: 0.1 },
  ),
  knobs: dict({
    noise: real(0.3, 1.5, { default: 0.6, step: 0.05, label: 'noise sd of each row' }),
    priorSd: real(0.5, 4, { default: 1.5, step: 0.1, label: 'prior sd of each coefficient' }),
  }),
  parts: mixtureRegressionProblem,
})

/** The knobs every extended atom has (the seed and the number of observations), with this atom's most observations. */
const learnedKnobs = (n: number, max: number): Record<string, ReturnType<typeof int>> => ({
  ...simulationKnobs(n),
  n: int(8, max, { default: n, label: 'observations' }),
})

const sig = (v: number, digits = 3): number => Number(v.toPrecision(digits))

// ---------------------------------------------------------------------------------------------------------------------
// Clutter with the parameters of the clutter and of the inliers unknown.

/** The priors of the extended clutter problem, stated on the page and in the code. */
const CLUTTER_PRIORS = {
  centre: 5,
  inlier: { shape: 2, rate: 1 },
  clutter: { shape: 2, rate: 20 },
  weight: { a: 2, b: 5 },
} as const

function clutterLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueTheta, trueInlierSd] = [configNumber(k, 'trueTheta'), configNumber(k, 'trueInlierSd')]
  const [trueClutterSd, trueWeight] = [configNumber(k, 'trueClutterSd'), configNumber(k, 'trueWeight')]
  // region clutter-learned
  // the ground truth the data are drawn from: the model does not know it
  const u = Array.from(toFlat(uniform(foldIn(key(seed), 0), [n])))
  const e = Array.from(toFlat(normal(foldIn(key(seed), 1), [n])))
  const isClutter = u.map((v) => v < trueWeight)
  const y = e.map((v, i) => (isClutter[i] ? trueClutterSd * v : trueTheta + trueInlierSd * v))
  const definition = defineModel('clutter-learned', (m) => {
    const row = m.range('i')
    // the centre, the precision of the inliers and of the clutter, the share of clutter
    const theta = m.latent('theta', d.normal(0, 5))
    const inlier = m.latent('inlier', d.gamma(2, 1))
    const clutter = m.latent('clutter', d.gamma(2, 20))
    const share = m.latent('share', d.beta(2, 5))
    // is observation i clutter?
    const z = m.latent('z', d.bernoulli({ probs: share }), { dims: [row] })
    // each branch of the gate has its own centre and its own precision
    const centre = ops.where(z, 0, theta)
    const precision = ops.where(z, clutter, inlier)
    const sd = ops.div(1, ops.sqrt(precision))
    m.observe('y', d.normal(centre, sd), m.data('y', { dims: [row], role: 'response' }))
  })
  // endregion
  const post = clutterPosterior(y, CLUTTER_PRIORS)
  const at = y.map((_, i) => i + 1)
  const band = (c: number, w: number): number[] => [c + w, c + w]
  const x = [0, n + 1]
  const [inSd, outSd] = [trueInlierSd, trueClutterSd]
  return {
    build: {
      definition,
      data: { y },
      quantities: [
        { name: 'theta', label: 'theta (the centre)', support: 'real', truth: trueTheta },
        { name: 'inlier', label: 'inlier precision', support: 'positive', truth: 1 / inSd ** 2 },
        { name: 'clutter', label: 'clutter precision', support: 'positive', truth: 1 / outSd ** 2 },
        { name: 'share', label: 'share of clutter', support: 'unit', truth: trueWeight },
        ...at.map((i) => ({
          name: `z[${i - 1}]`,
          label: `observation ${i} is clutter`,
          support: 'probability' as const,
          hidden: true,
        })),
      ],
    },
    reference: {
      exact: {
        theta: post.centre,
        inlier: post.inlier,
        clutter: post.clutter,
        share: post.weight,
        ...Object.fromEntries(post.clutterProbability.map((p, i) => [`z[${i}]`, probabilityMarginal(p)])),
      },
      referenceSource: 'quadrature over the centre, the two precisions and the share, the switches summed out',
    },
    present: {
      equations: [
        `truth: centre ${trueTheta}, inlier sd ${trueInlierSd} (precision ${sig(1 / inSd ** 2)}), clutter sd ${trueClutterSd} (precision ${sig(1 / outSd ** 2)}), clutter share ${trueWeight}; ${n} observations, seed ${seed}`,
        'theta ~ N(0, 5²)   (the centre)',
        'inlier ~ Gamma(2, 1)   (precision of the inliers: mean 2, sd 1.4)',
        'clutter ~ Gamma(2, 20)   (precision of the clutter: mean 0.1, a standard deviation of about 3: wider than the inliers)',
        'share ~ Beta(2, 5)   (the share of clutter: a minority, mean 0.29)',
        'z_i ~ Bernoulli(share)   (z_i = 1: observation i is clutter)',
        'y_i ~ z_i ? N(0, 1 / clutter) : N(theta, 1 / inlier)   (a gate per observation; each branch has its own precision)',
      ],
      figure: {
        caption:
          'The simulated observations, coloured by what EP believes of each (clutter, a draw from the wide component around 0, or an inlier, a detection of the centre). Green: the truth (the centre, the inlier band and the clutter band at two sds); thick and dashed: what EP learned.',
        xLabel: 'observation',
        yLabel: 'value',
        xLimits: x as [number, number],
        points: [
          {
            name: 'observations',
            x: at,
            y,
            group: isClutter.map((c) => (c ? 1 : 0)),
            groupNames: { 0: 'inlier', 1: 'clutter' },
            groupFrom: (m) => at.map((i) => ((m[`z[${i - 1}]`] ?? 0) > 0.5 ? 1 : 0)),
          },
        ],
        curves: [
          { name: 'true centre', colour: 'tab2', x, y: band(trueTheta, 0) },
          { name: 'true inlier band', colour: 'tab2', x, y: band(trueTheta, 2 * inSd) },
          { name: 'true inlier band', colour: 'tab2', x, y: band(trueTheta, -2 * inSd) },
          { name: 'true clutter band', colour: 'tab1', x, y: band(0, 2 * outSd) },
          { name: 'true clutter band', colour: 'tab1', x, y: band(0, -2 * outSd) },
        ],
        answer: (m) => {
          const centre = m['theta'] ?? Number.NaN
          const w = 2 / Math.sqrt(m['inlier'] ?? Number.NaN)
          const o = 2 / Math.sqrt(m['clutter'] ?? Number.NaN)
          return [
            { name: 'theta', x, y: band(centre, 0) },
            { name: 'theta + 2 inlier sd', x, y: band(centre, w) },
            { name: 'theta - 2 inlier sd', x, y: band(centre, -w) },
            { name: 'clutter + 2 sd', x, y: band(0, o) },
            { name: 'clutter - 2 sd', x, y: band(0, -o) },
          ]
        },
      },
    },
  }
}

/**
 * Minka's clutter problem with everything unknown: the centre, the precision of the inliers, the precision of the clutter (a gamma
 * that keeps it wider than the inliers) and the share of clutter (a beta), learned from data simulated from a stated truth. Each
 * observation is a gate between a Gaussian at the centre with the inlier precision and a Gaussian at zero with the clutter
 * precision, so EP reports the probability that each observation is clutter as well as the parameters.
 */
const clutterLearnedAtom: EpAtom = defineAtom({
  name: 'clutter-learned',
  title: 'Clutter: parameters learned',
  extends: 'clutter',
  recovery: recovery(
    'measured',
    'EP matches the exact posterior closely, so over seeds drawn from the truth its intervals hold it about nine times in ten (40 seeds: 0.88 to 1.0 per parameter); 12 seeds, to keep the test quick, leave a binomial spread of about 0.1',
    { seeds: 12, coverage: 0.6, zrms: 1.8 },
  ),
  what: 'The clutter problem of the basic atom with the numbers it fixes made unknown: the width of the inliers, the width of the clutter and the share of clutter all have priors (the clutter kept wider than the inliers), and the data are simulated from a stated truth. EP, through a gate per observation whose branches each have a gamma precision, recovers all four parameters and says which observations are clutter.',
  factors: [
    'prior (gaussian, gamma, beta)',
    'bernoulli (a switch with a beta probability)',
    'select gate per observation (gaussian-gamma-precision branches)',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance(
    'measured',
    'every observation is a two-way gate and EP matches the moments of each, carrying one Gaussian, two gammas and a beta for the four unknowns; against the exact posterior by quadrature the parameters agree to a few hundredths of a posterior sd (spreads within 2%) and the responsibilities to 0.01 (the spread of a nearly certain responsibility, a coin, differs in the third decimal, which the lower bound allows)',
    { meanZ: 0.15, sdRatio: [0.65, 1.2], ks: 0.12 },
  ),
  knobs: dict({
    ...learnedKnobs(50, 80),
    trueTheta: real(-3, 3, { default: 1.5, step: 0.1, label: 'true centre' }),
    trueInlierSd: real(0.2, 1.5, { default: 0.5, step: 0.05, label: 'true inlier sd' }),
    trueClutterSd: real(1.5, 8, { default: 4, step: 0.5, label: 'true clutter sd' }),
    trueWeight: real(0.05, 0.5, { default: 0.25, step: 0.01, label: 'true share of clutter' }),
  }),
  ep: { damping: 0.5, tol: 1e-4 },
  parts: clutterLearnedProblem,
})

// ---------------------------------------------------------------------------------------------------------------------
// Mixtures with unknown components.

/** The true mixture the data are drawn from: weights, means and sds of each component. */
interface TrueMixture {
  readonly weights: readonly number[]
  readonly means: readonly number[]
  readonly sds: readonly number[]
}

const mixtureDensity =
  (t: TrueMixture) =>
  (x: number): number =>
    t.weights.reduce((a, w, k) => a + w * normalPdf(x, t.means[k] as number, t.sds[k] as number), 0)

/** Draws from a mixture: which component, then the value. */
function simulateMixture(seed: number, n: number, t: TrueMixture): { y: number[]; from: number[] } {
  const u = Array.from(toFlat(uniform(foldIn(key(seed), 0), [n])))
  const e = Array.from(toFlat(normal(foldIn(key(seed), 1), [n])))
  const from = u.map((v) => {
    let acc = 0
    for (let k = 0; k < t.weights.length - 1; k++) {
      acc += t.weights[k] as number
      if (v < acc) return k
    }
    return t.weights.length - 1
  })
  return { y: e.map((v, i) => (t.means[from[i] as number] as number) + (t.sds[from[i] as number] as number) * v), from }
}

/** A figure of mixture data: the points along the axis (coloured by component), the true density and the fitted one. */
function mixtureFigure(
  y: readonly number[],
  from: readonly number[],
  truth: TrueMixture,
  fitted: (m: Readonly<Record<string, number>>) => TrueMixture,
  names: Record<number, string>,
  groupFrom?: (m: Readonly<Record<string, number>>) => readonly number[],
): NonNullable<AtomProblem['figure']> {
  const lo = Math.min(...y, ...truth.means.map((m, k) => m - 3 * (truth.sds[k] as number)))
  const hi = Math.max(...y, ...truth.means.map((m, k) => m + 3 * (truth.sds[k] as number)))
  const xs = Array.from({ length: 160 }, (_, i) => lo + ((hi - lo) * i) / 159)
  const density = mixtureDensity(truth)
  const top = Math.max(...xs.map(density))
  return {
    caption:
      'The simulated observations along the axis (coloured by the component each was drawn from), the true mixture density (green) and the mixture EP learned from the means of its beliefs (thick, dashed). The densities are drawn in the units of the vertical axis, scaled to the picture.',
    xLabel: 'value',
    yLabel: 'density',
    xLimits: [lo, hi],
    yLimits: [-0.12 * top, 1.15 * top],
    points: [
      {
        name: 'observations',
        x: [...y],
        y: y.map((_, i) => -0.04 * top - 0.06 * top * ((i * 7) % 5) * 0.2),
        group: [...from],
        groupNames: names,
        ...(groupFrom === undefined ? {} : { groupFrom }),
      },
    ],
    curves: [{ name: 'true density', colour: 'tab2', x: xs, y: xs.map(density) }],
    answer: (m) => {
      const f = fitted(m)
      const g = mixtureDensity(f)
      return [{ name: 'EP mixture', x: xs, y: xs.map(g) }]
    },
  }
}

const sdOf = (precision: number | undefined): number => 1 / Math.sqrt(precision ?? Number.NaN)

/** The truths of a mixture's quantities, in the model's names (`mu[k]`, `tau[k]`, `weights[k]`). */
function mixtureQuantities(
  truth: TrueMixture,
  weightName: (k: number) => string,
  meanName: (k: number) => string,
  precisionName: (k: number) => string,
  label: (k: number) => string,
): AtomProblem['quantities'] {
  const K = truth.weights.length
  return [
    ...Array.from({ length: K }, (_, k) => ({
      name: meanName(k),
      label: `mean of ${label(k)}`,
      support: 'real' as AtomSupport,
      truth: truth.means[k] as number,
    })),
    ...Array.from({ length: K }, (_, k) => ({
      name: precisionName(k),
      label: `precision of ${label(k)}`,
      support: 'positive' as AtomSupport,
      truth: 1 / (truth.sds[k] as number) ** 2,
    })),
    ...Array.from({ length: K }, (_, k) => ({
      name: weightName(k),
      label: `weight of ${label(k)}`,
      support: 'unit' as AtomSupport,
      truth: truth.weights[k] as number,
    })),
  ]
}

/** The mixture a set of beliefs' means stand for. */
function fittedMixture(
  m: Readonly<Record<string, number>>,
  K: number,
  weightName: (k: number) => string,
  meanName: (k: number) => string,
  precisionName: (k: number) => string,
): TrueMixture {
  const ks = Array.from({ length: K }, (_, k) => k)
  return {
    weights: ks.map((k) => m[weightName(k)] ?? Number.NaN),
    means: ks.map((k) => m[meanName(k)] ?? Number.NaN),
    sds: ks.map((k) => sdOf(m[precisionName(k)])),
  }
}

function mixtureLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const K = configNumber(k, 'components')
  const gap = configNumber(k, 'separation')
  const truth: TrueMixture =
    K === 2
      ? { weights: [0.6, 0.4], means: [-gap / 2, gap / 2], sds: [0.8, 1.2] }
      : { weights: [0.4, 0.35, 0.25], means: [-gap, 0, gap], sds: [0.8, 1, 1.2] }
  const { y, from } = simulateMixture(seed, n, truth)
  const alpha = Array.from({ length: K }, () => 2)
  // region mixture-components-learned
  const definition = defineModel('mixture-components-learned', (m) => {
    const row = m.range('i')
    const comp = m.range('component', { size: K })
    // the means (in increasing order, which names the components), their precisions, the weights
    const mu = m.latent('mu', d.normal(0, 4), { dims: [comp], constraint: { kind: 'ordered' } })
    const tau = m.latent('tau', d.gamma(2, 2), { dims: [comp] })
    const weights = m.latent('weights', d.dirichlet(alpha))
    // which component each observation comes from
    const z = m.latent('z', d.categorical({ probs: weights }), { dims: [row] })
    const sd = ops.div(1, ops.sqrt(tau.at(z)))
    m.observe('y', d.normal(mu.at(z), sd), m.data('y', { dims: [row], role: 'response' }))
  })
  // endregion
  const gibbs = mixtureGibbs({
    x: y,
    means: Array.from({ length: K }, () => ({ mean: 0, sd: 4 })),
    precision: { shape: 2, rate: 2 },
    weights: alpha,
    seed: seed + 1000,
    burn: 300,
    draws: 1500,
    ordered: true,
  })
  const names = (kk: number): string => `component ${kk + 1}`
  const [wn, mn, pn] = [(c: number) => `weights[${c}]`, (c: number) => `mu[${c}]`, (c: number) => `tau[${c}]`]
  const exact: Record<string, ExactMarginal> = {}
  for (let c = 0; c < K; c++) {
    exact[mn(c)] = samplesMarginal(gibbs.mean[c] as number[])
    exact[pn(c)] = samplesMarginal(gibbs.precision[c] as number[])
    exact[wn(c)] = samplesMarginal(gibbs.weight[c] as number[])
  }
  return {
    build: {
      definition,
      data: { y },
      bindOptions: { sizes: { component: K } },
      quantities: mixtureQuantities(truth, wn, mn, pn, names),
    },
    reference: {
      exact,
      referenceSource: `a Gibbs sampler of the same model (1500 draws after 300), the means relabelled in increasing order`,
    },
    present: {
      equations: [
        `truth: weights ${truth.weights.join(', ')}, means ${truth.means.map((v) => sig(v)).join(', ')}, sds ${truth.sds.join(', ')}; ${n} observations, seed ${seed}`,
        `mu_1 < ... < mu_${K}, each ~ N(0, 4²)   (an ordered vector: the first has the smallest mean)`,
        'tau_k ~ Gamma(2, 2)   (the precision of each component: mean 1)',
        `weights ~ Dirichlet(${alpha.join(', ')})`,
        'z_i ~ Categorical(weights)   (which component)',
        'y_i ~ N(mu[z_i], 1 / tau[z_i])   (the mean and the precision both picked by the index)',
      ],
      figure: mixtureFigure(
        y,
        from,
        truth,
        (m) => fittedMixture(m, K, wn, mn, pn),
        Object.fromEntries(Array.from({ length: K }, (_, c) => [c, names(c)])),
      ),
    },
  }
}

/**
 * A mixture of two or three Gaussians whose means, precisions and weights are all unknown (ordered means, gamma precisions, a
 * Dirichlet on the weights), learned from data simulated from a stated truth: the index gate carries the mean and the precision of
 * the chosen component together. The separation knob shows when EP recovers the components and when it settles on one reading of
 * overlapping ones.
 */
const mixtureComponentsLearnedAtom: EpAtom = defineAtom({
  name: 'mixture-components-learned',
  title: 'Random index into random components: parameters learned',
  extends: 'mixture-components',
  recovery: recovery(
    'measured',
    'at the default separation of 4 the intervals hold the truth 0.75 to 1.0 of the time per parameter (20 seeds); the floor leaves the spread of 10 seeds, and the page says honestly that closer components (a separation of 2.5 or less) are covered less often',
    { seeds: 10, coverage: 0.6, zrms: 1.8 },
  ),
  what: 'The index gate of the basic atom, mu[z], with the means, the widths (gamma precisions) and the weights (a Dirichlet) all unknown, the means ordered so the components are named, and the data simulated from a stated truth. The separation knob moves the components together: EP recovers them when they are apart, and reports a confident reading of the overlap when they are not.',
  factors: [
    'prior (gaussian, gamma, dirichlet)',
    'ordered constraint (greater-than between neighbours)',
    'discrete (a switch with a Dirichlet probability)',
    'index gate (gaussian-gamma-precision branches)',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance(
    'lumpy',
    'for well-separated components EP is within a fifth of a posterior sd of a Gibbs sampler of the same model, a little overconfident (spreads about 0.9 of the sampler); where the components overlap the sampler is wider (spreads 0.5 to 0.7) because EP settles on one reading of the labels',
  ),
  knobs: dict({
    ...learnedKnobs(45, 120),
    components: int(2, 3, { default: 2, label: 'components' }),
    separation: real(1, 6, { default: 4, step: 0.25, label: 'distance between the true means' }),
  }),
  ep: { damping: 0.5, tol: 1e-4 },
  parts: mixtureLearnedProblem,
})

function gateRandomLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const gap = configNumber(k, 'separation')
  const truth: TrueMixture = { weights: [0.65, 0.35], means: [-gap / 2, gap / 2], sds: [0.8, 1.2] }
  const { y, from } = simulateMixture(seed, n, truth)
  // region gate-random-learned
  const definition = defineModel('gate-random-learned', (m) => {
    const row = m.range('i')
    // the two components: where each is centred (a prior on each) and how wide (a precision)
    const low = m.latent('low', d.normal(-2, 2))
    const high = m.latent('high', d.normal(2, 2))
    const lowPrecision = m.latent('lowPrecision', d.gamma(2, 2))
    const highPrecision = m.latent('highPrecision', d.gamma(2, 2))
    // the weight of the second component, and the switch of each observation
    const share = m.latent('share', d.beta(2, 2))
    const z = m.latent('z', d.bernoulli({ probs: share }), { dims: [row] })
    // the switch selects the centre and the precision of its component together
    const centre = ops.where(z, high, low)
    const precision = ops.where(z, highPrecision, lowPrecision)
    const sd = ops.div(1, ops.sqrt(precision))
    m.observe('y', d.normal(centre, sd), m.data('y', { dims: [row], role: 'response' }))
  })
  // endregion
  const gibbs = mixtureGibbs({
    x: y,
    means: [
      { mean: -2, sd: 2 },
      { mean: 2, sd: 2 },
    ],
    precision: { shape: 2, rate: 2 },
    weights: [2, 2],
    seed: seed + 1000,
    burn: 300,
    draws: 1500,
    ordered: false,
  })
  const at = y.map((_, i) => i + 1)
  const exact: Record<string, ExactMarginal> = {
    low: samplesMarginal(gibbs.mean[0] as number[]),
    high: samplesMarginal(gibbs.mean[1] as number[]),
    lowPrecision: samplesMarginal(gibbs.precision[0] as number[]),
    highPrecision: samplesMarginal(gibbs.precision[1] as number[]),
    share: samplesMarginal(gibbs.weight[1] as number[]),
    ...Object.fromEntries(gibbs.responsibility.map((r, i) => [`z[${i}]`, probabilityMarginal(r[1] as number)])),
  }
  const figure = mixtureFigure(
    y,
    from,
    truth,
    (m) => ({
      weights: [1 - (m['share'] ?? Number.NaN), m['share'] ?? Number.NaN],
      means: [m['low'] ?? Number.NaN, m['high'] ?? Number.NaN],
      sds: [sdOf(m['lowPrecision']), sdOf(m['highPrecision'])],
    }),
    { 0: 'low component', 1: 'high component' },
    (m) => at.map((i) => ((m[`z[${i - 1}]`] ?? 0) > 0.5 ? 1 : 0)),
  )
  return {
    build: {
      definition,
      data: { y },
      quantities: [
        { name: 'low', label: 'mean of the low component', support: 'real', truth: truth.means[0] as number },
        { name: 'high', label: 'mean of the high component', support: 'real', truth: truth.means[1] as number },
        {
          name: 'lowPrecision',
          label: 'precision of the low component',
          support: 'positive',
          truth: 1 / (truth.sds[0] as number) ** 2,
        },
        {
          name: 'highPrecision',
          label: 'precision of the high component',
          support: 'positive',
          truth: 1 / (truth.sds[1] as number) ** 2,
        },
        { name: 'share', label: 'weight of the high component', support: 'unit', truth: truth.weights[1] as number },
        ...at.map((i) => ({
          name: `z[${i - 1}]`,
          label: `observation ${i} is from the high component`,
          support: 'probability' as const,
          hidden: true,
        })),
      ],
    },
    reference: {
      exact,
      referenceSource: 'a Gibbs sampler of the same model (1500 draws after 300)',
    },
    present: {
      equations: [
        `truth: weights ${truth.weights.join(', ')}, means ${truth.means.map((v) => sig(v)).join(', ')}, sds ${truth.sds.join(', ')}; ${n} observations, seed ${seed}`,
        'low ~ N(-2, 2²),  high ~ N(2, 2²)   (the two components, told apart by their priors)',
        'lowPrecision, highPrecision ~ Gamma(2, 2)   (their widths: mean 1)',
        'share ~ Beta(2, 2)   (the weight of the high component)',
        'z_i ~ Bernoulli(share)',
        'y_i ~ z_i ? N(high, 1 / highPrecision) : N(low, 1 / lowPrecision)   (a select gate: centre and precision together)',
      ],
      figure,
    },
  }
}

/**
 * A switch between two random Gaussians, written as a select, with everything about them unknown: the two means, their two
 * widths (gamma precisions) and the weight of the switch (a beta), learned from data simulated from a stated truth. The twin
 * of the gate between two random values and of the gate-weight atoms, with the components' widths learned too.
 */
const gateRandomLearnedAtom: EpAtom = defineAtom({
  name: 'gate-random-learned',
  title: 'Gate between two random values: parameters learned',
  extends: 'gate-random',
  recovery: recovery(
    'measured',
    'at the default separation of 4 the intervals hold the truth 0.75 to 1.0 of the time per parameter (20 seeds); the floor leaves the spread of 10 seeds, and the page says honestly that closer components are covered less often',
    { seeds: 10, coverage: 0.6, zrms: 1.8 },
  ),
  what: 'The gate between two random values of the basic atom, with the two means, the two widths and the weight of the switch all unknown (a prior on each), and the data simulated from a stated truth: the switch picks the centre and the precision of its component together, and EP also says which component each observation came from.',
  factors: [
    'prior (gaussian, gamma, beta)',
    'bernoulli (a switch with a beta probability)',
    'select gate per observation (gaussian-gamma-precision branches)',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance(
    'measured',
    'for well-separated components EP is within a fifth of a posterior sd of a Gibbs sampler of the same model, a little overconfident (the parameters within 12% of its spread, the responsibilities, which are coins, within 0.03); where the components overlap the sampler is wider because EP settles on one reading',
    { meanZ: 0.3, sdRatio: [0.6, 1.2], ks: 0.15 },
  ),
  knobs: dict({
    ...learnedKnobs(40, 120),
    separation: real(1, 6, { default: 4, step: 0.25, label: 'distance between the true means' }),
  }),
  ep: { damping: 0.5, tol: 1e-4 },
  parts: gateRandomLearnedProblem,
})

// ---------------------------------------------------------------------------------------------------------------------
// Spike and slab with the inclusion probability and the slab width unknown.

/** The coefficients that are really active in the simulated sequence, and their values. */
const ACTIVE: readonly (readonly [number, number])[] = [
  [1, 3.2],
  [4, -2.4],
  [6, 4.1],
]
const SPIKE_SD = 0.1

function spikeSlabLearnedProblem(k: Config): AtomParts {
  const [seed, p] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const noise = configNumber(k, 'noise')
  // region spike-and-slab-learned
  // the ground truth: three coefficients are active, the rest are zero
  const values = Array.from({ length: p }, (_, j) => ACTIVE.find(([at]) => at === j)?.[1] ?? 0)
  const e = Array.from(toFlat(normal(foldIn(key(seed), 0), [p])))
  const y = values.map((v, j) => v + noise * (e[j] ?? 0))
  const definition = defineModel('spike-and-slab-learned', (m) => {
    const j = m.range('j')
    // how often a coefficient is active, and the precision of the active ones
    const share = m.latent('share', d.beta(1.5, 3.5))
    const slabPrecision = m.latent('slabPrecision', d.gamma(2, 8))
    // is coefficient j in the slab?
    const z = m.latent('z', d.bernoulli({ probs: share }), { dims: [j] })
    const slabSd = ops.div(1, ops.sqrt(slabPrecision))
    const slab = m.latent('slab', d.normal(0, slabSd), { dims: [j] })
    const spike = m.latent('spike', d.normal(0, 0.1), { dims: [j] })
    // each coefficient is whichever component its switch selects
    m.observe('y', d.normal(ops.where(z, slab, spike), noise), m.data('y', { dims: [j] }))
  })
  // endregion
  const active = values.map((v) => v !== 0)
  const trueShare = active.filter(Boolean).length / p
  const rms = Math.sqrt(values.filter((v) => v !== 0).reduce((a, v) => a + v * v, 0) / active.filter(Boolean).length)
  // The exact posterior: given the share and the slab precision the coefficients are independent two-component mixtures.
  const [aS, bS, aT, bT] = [1.5, 3.5, 2, 8]
  const spikeVar = noise ** 2 + SPIKE_SD ** 2
  const ps = Array.from({ length: 300 }, (_, i) => (i + 0.5) / 300)
  const lts = Array.from(
    { length: 300 },
    (_, i) => Math.log(0.002) + ((Math.log(4) - Math.log(0.002)) * (i + 0.5)) / 300,
  )
  const ts = lts.map(Math.exp)
  const logs: number[] = []
  const pips: number[][] = []
  let top = Number.NEGATIVE_INFINITY
  for (const t of ts) {
    const slabVar = noise ** 2 + 1 / t
    for (const q of ps) {
      let lp = (aS - 1) * Math.log(q) + (bS - 1) * Math.log(1 - q) + aT * Math.log(t) - bT * t
      const pip: number[] = []
      for (const v of y) {
        const s1 = q * normalPdf(v, 0, Math.sqrt(slabVar))
        const s0 = (1 - q) * normalPdf(v, 0, Math.sqrt(spikeVar))
        lp += Math.log(s0 + s1)
        pip.push(s1 / (s0 + s1))
      }
      logs.push(lp)
      pips.push(pip)
      if (lp > top) top = lp
    }
  }
  const mass = logs.map((l) => Math.exp(l - top))
  const norm = mass.reduce((a, b) => a + b, 0)
  const shareMass = ps.map((_, i) => ts.reduce((a, __, jj) => a + (mass[jj * ps.length + i] as number), 0))
  const precisionMass = ts.map((_, jj) => ps.reduce((a, __, i) => a + (mass[jj * ps.length + i] as number), 0))
  const pipMean = y.map((_, j) => mass.reduce((a, m, c) => a + (m * ((pips[c] as number[])[j] as number)) / norm, 0))
  const shareMarginal = weightedMarginal(ps, shareMass, 'nearest')
  const precisionExact = weightedMarginal(ts, precisionMass, 'nearest')
  const at = y.map((_, j) => j + 1)
  const x = [0, p + 1]
  return {
    build: {
      definition,
      data: { y },
      quantities: [
        { name: 'share', label: 'share of active coefficients', support: 'unit', truth: trueShare },
        {
          name: 'slabPrecision',
          label: 'slab precision (1 / rms of the active coefficients squared)',
          support: 'positive',
          truth: 1 / rms ** 2,
        },
        ...at.map((i) => ({
          name: `z[${i - 1}]`,
          label: `coefficient ${i} is active`,
          support: 'probability' as const,
        })),
      ],
    },
    reference: {
      exact: {
        share: shareMarginal,
        slabPrecision: precisionExact,
        ...Object.fromEntries(pipMean.map((v, j) => [`z[${j}]`, probabilityMarginal(v)])),
      },
      referenceSource: 'the posterior of the share and the slab precision on a grid, the coefficients summed out',
    },
    present: {
      equations: [
        `truth: coefficients ${ACTIVE.map(([j, v]) => `${j + 1}: ${v}`).join(', ')} active (share ${sig(trueShare)}), the other ${p - ACTIVE.length} zero; noise sd ${noise} known; seed ${seed}`,
        'share ~ Beta(1.5, 3.5)   (the inclusion probability: mean 0.3)',
        'slabPrecision ~ Gamma(2, 8)   (the precision of the active coefficients: mean 0.25, a slab sd of about 2)',
        'z_j ~ Bernoulli(share)   (z_j = 1: coefficient j is in the slab)',
        'slab_j ~ N(0, 1 / slabPrecision),  spike_j ~ N(0, 0.1²)   (a wide and a narrow component)',
        `y_j ~ N(z_j ? slab_j : spike_j, ${noise}²), j = 1..${p}`,
      ],
      figure: {
        caption:
          'The observed value of each coefficient, coloured by whether it is truly active (marks) and by what EP believes (P(slab) above one half). The green lines are the truth at zero and the true values of the active ones.',
        xLabel: 'coefficient',
        yLabel: 'observed value',
        xLimits: x as [number, number],
        points: [
          {
            name: 'observed y',
            x: at,
            y: [...y],
            group: active.map((a) => (a ? 1 : 0)),
            groupNames: { 0: 'inactive', 1: 'active' },
            groupFrom: (m) => at.map((i) => ((m[`z[${i - 1}]`] ?? 0) > 0.5 ? 1 : 0)),
          },
          { name: 'true value', x: at, y: values },
        ],
        notes: [`The truly active coefficients are ${ACTIVE.map(([j]) => j + 1).join(', ')}.`],
      },
    },
  }
}

/**
 * A sparse sequence of coefficients, each a switch between a narrow spike and a wide slab, with the inclusion probability (a
 * beta) and the precision of the slab (a gamma) unknown, learned from data simulated from a stated set of active coefficients:
 * EP recovers which coefficients are active, how common they are and how large.
 */
const spikeSlabLearnedAtom: EpAtom = defineAtom({
  name: 'spike-and-slab-learned',
  title: 'Spike and slab: parameters learned',
  extends: 'spike-and-slab',
  recovery: recovery(
    'measured',
    'with ten coefficients and three active, the share and the slab precision are learned from a handful of values and their posteriors are skewed, which EP carries as a beta and a gamma; 30 seeds leave a spread of about 0.06',
    { seeds: 30, coverage: 0.7, zrms: 1.6 },
  ),
  what: 'The spike and slab gate of the basic atom on a sequence of coefficients, with the inclusion probability (a beta) and the precision of the slab (a gamma) unknown, and the data simulated from a stated set of active coefficients: EP recovers which are active, how common they are and how wide the slab is.',
  factors: [
    'prior (beta, gamma)',
    'bernoulli (a switch with a beta probability)',
    'gaussian-gamma-precision (the slab)',
    'select gate (spike and slab components)',
    'gaussian likelihood',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance(
    'wide',
    'the share and the slab precision are tied through the switches, which EP carries as a beta and a gamma; the posterior of the precision is skewed and EP is a little narrower than the exact one (0.89 of its sd)',
  ),
  knobs: dict({
    seed: simulationKnobs(10)['seed'] as ReturnType<typeof int>,
    n: int(7, 14, { default: 10, label: 'coefficients' }),
    noise: real(0.2, 1.2, { default: 0.5, step: 0.05, label: 'noise sd (known)' }),
  }),
  ep: { damping: 0.5, tol: 1e-4 },
  parts: spikeSlabLearnedProblem,
})

function spikeSlabMarginalLearnedProblem(k: Config): AtomParts {
  const base = spikeSlabLearnedProblem(k)
  const [seed, p] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const noise = configNumber(k, 'noise')
  // region spike-and-slab-marginal-learned
  // the ground truth: three coefficients are active, the rest are zero
  const coefficients = Array.from({ length: p }, (_, j) => ACTIVE.find(([at]) => at === j)?.[1] ?? 0)
  const draw = Array.from(toFlat(normal(foldIn(key(seed), 0), [p])))
  const observed = coefficients.map((v, j) => v + noise * (draw[j] ?? 0))
  const definition = defineModel('spike-and-slab-marginal-learned', (m) => {
    const j = m.range('j')
    // how often a coefficient is active, and the precision of the active ones
    const share = m.latent('share', d.beta(1.5, 3.5))
    const slabPrecision = m.latent('slabPrecision', d.gamma(2, 8))
    const slabSd = ops.div(1, ops.sqrt(slabPrecision))
    // each coefficient has the two-component prior, its switch summed out
    const w = m.latent('w', d.normalSpikeSlab(share, slabSd, 0.1), { dims: [j] })
    m.observe('y', d.normal(w, noise), m.data('y', { dims: [j] }))
  })
  // endregion
  return {
    build: {
      ...base.build,
      definition,
      data: { y: observed },
      quantities: base.build.quantities.filter((q) => q.support !== 'probability'),
    },
    reference: {
      ...base.reference,
      exact: Object.fromEntries(Object.entries(base.reference?.exact ?? {}).filter(([name]) => !name.startsWith('z['))),
    },
    present: {
      ...base.present,
      equations: base.present.equations
        .map((line) =>
          line.startsWith('z_j ~')
            ? 'w_j ~ share N(0, 1 / slabPrecision) + (1 - share) N(0, 0.1²)   (the switch summed out)'
            : line,
        )
        .filter((line) => !line.startsWith('slab_j ~')),
    },
  }
}

/**
 * The same sparse sequence with each coefficient's switch summed out (the library's spike and slab family) and the inclusion
 * probability and the slab precision unknown: the compiler gives each coefficient the gate of its spike and slab over a switch
 * that the posterior does not report, the same graph as the explicit-switch twin, so EP agrees with the exact posterior of the
 * share and the slab precision (the coefficients summed out) about as closely as the twin does.
 */
const spikeSlabMarginalLearnedAtom: EpAtom = defineAtom({
  name: 'spike-and-slab-marginal-learned',
  title: 'Spike and slab, switch summed out: parameters learned',
  extends: 'spike-and-slab-marginal',
  recovery: recovery(
    'measured',
    'the same posterior and the same floor as the explicit-switch twin (spike and slab: parameters learned): the share and the slab precision are learned from a handful of values',
    { seeds: 30, coverage: 0.7, zrms: 1.6 },
  ),
  what: 'The spike and slab prior with the switch summed out (the mixture form, no variable for the probability of the slab) on a sequence of coefficients, with the inclusion probability and the slab precision unknown, and the data simulated from a stated set of active coefficients: EP recovers how common the active coefficients are and how wide the slab is.',
  factors: [
    'prior (beta, gamma)',
    'bernoulli (a hidden switch with a beta probability)',
    'gate of spike and slab components (gaussian-gamma-precision slab)',
    'gaussian likelihood',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance(
    'wide',
    'the same posterior as the explicit-switch twin: EP carries a beta and a gamma (measured: 0.006 and 0.003 posterior sds, spreads 0.98)',
  ),
  knobs: dict({
    seed: simulationKnobs(10)['seed'] as ReturnType<typeof int>,
    n: int(7, 14, { default: 10, label: 'coefficients' }),
    noise: real(0.2, 1.2, { default: 0.5, step: 0.05, label: 'noise sd (known)' }),
  }),
  ep: { damping: 0.5, tol: 1e-4 },
  parts: spikeSlabMarginalLearnedProblem,
})

function clutterMixtureLearnedProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueTheta, trueSd, trueWeight] = [
    configNumber(k, 'trueTheta'),
    configNumber(k, 'trueSd'),
    configNumber(k, 'trueWeight'),
  ]
  const width = 10
  // region clutter-mixture-learned
  // the ground truth: outliers are draws around the same centre, ten times wider
  const u = Array.from(toFlat(uniform(foldIn(key(seed), 0), [n])))
  const e = Array.from(toFlat(normal(foldIn(key(seed), 1), [n])))
  const isOutlier = u.map((v) => v < trueWeight)
  const y = e.map((v, i) => trueTheta + trueSd * v * (isOutlier[i] ? width : 1))
  const definition = defineModel('clutter-mixture-learned', (m) => {
    const row = m.range('i')
    // the centre, the precision of the inliers, the share of outliers
    const mu = m.latent('mu', d.normal(0, 5))
    const precision = m.latent('precision', d.gamma(2, 1))
    const share = m.latent('share', d.beta(2, 5))
    // each observation is an inlier or, ten times wider, an outlier: the switch summed out
    const sd = ops.div(1, ops.sqrt(precision))
    const observed = m.data('y', { dims: [row], role: 'response' })
    m.observe('y', d.normalOutliers(mu, sd, share, 10), observed)
  })
  // endregion
  const priors = { centre: 5, precision: { shape: 2, rate: 1 }, share: { a: 2, b: 5 }, width } as const
  const post = outlierMixturePosterior(y, priors)
  const at = y.map((_, i) => i + 1)
  const x = [0, n + 1]
  const band = (c: number, w: number): number[] => [c + w, c + w]
  return {
    build: {
      definition,
      data: { y },
      quantities: [
        { name: 'mu', label: 'mu (the centre)', support: 'real', truth: trueTheta },
        { name: 'precision', label: 'inlier precision', support: 'positive', truth: 1 / trueSd ** 2 },
        { name: 'share', label: 'share of outliers', support: 'unit', truth: trueWeight },
      ],
    },
    reference: {
      exact: { mu: post.centre, precision: post.precision, share: post.share },
      referenceSource: 'quadrature over the centre, the precision and the share, the switches summed out',
    },
    present: {
      equations: [
        `truth: centre ${trueTheta}, inlier sd ${trueSd} (precision ${sig(1 / trueSd ** 2)}), outlier share ${trueWeight}, outliers ${width} times wider; ${n} observations, seed ${seed}`,
        'mu ~ N(0, 5²)   (the centre)',
        'precision ~ Gamma(2, 1)   (of the inliers)',
        'share ~ Beta(2, 5)   (the share of outliers)',
        `y_i ~ (1 - share) N(mu, 1 / precision) + share N(mu, ${width}² / precision)   (the switch summed out)`,
      ],
      figure: {
        caption:
          'The simulated observations (outliers drawn ten times wider around the same centre, marked) and the centre with the inlier band at two sds: green, the truth; thick and dashed, what EP believes.',
        xLabel: 'observation',
        yLabel: 'value',
        xLimits: x as [number, number],
        points: [
          {
            name: 'observations',
            x: at,
            y,
            group: isOutlier.map((o) => (o ? 1 : 0)),
            groupNames: { 0: 'inlier', 1: 'outlier' },
          },
        ],
        curves: [
          { name: 'true centre', colour: 'tab2', x, y: band(trueTheta, 0) },
          { name: 'true inlier band', colour: 'tab2', x, y: band(trueTheta, 2 * trueSd) },
          { name: 'true inlier band', colour: 'tab2', x, y: band(trueTheta, -2 * trueSd) },
        ],
        answer: (m) => {
          const centre = m['mu'] ?? Number.NaN
          const w = 2 / Math.sqrt(m['precision'] ?? Number.NaN)
          return [
            { name: 'mu', x, y: band(centre, 0) },
            { name: 'mu + 2 inlier sd', x, y: band(centre, w) },
            { name: 'mu - 2 inlier sd', x, y: band(centre, -w) },
          ]
        },
      },
    },
  }
}

/**
 * The inlier and outlier mixture with the switches summed out (the library's `normalOutliers`) and the centre, the width of the
 * inliers and the share of outliers all unknown, learned from data simulated from a stated truth. The compiler reads each
 * observation as the gate of an inlier and an outlier branch over a switch the posterior does not report (the graph of the
 * explicit-switch twin, clutter: parameters learned), and EP agrees with the exact posterior (quadrature over the centre, the
 * precision and the share, the switches summed out).
 */
const clutterMixtureLearnedAtom: EpAtom = defineAtom({
  name: 'clutter-mixture-learned',
  title: 'Inlier and outlier mixture, switches summed out: parameters learned',
  extends: 'clutter-mixture',
  recovery: recovery('quick', 'the same floor as the explicit-switch twin (clutter: parameters learned)'),
  what: 'The inlier and outlier mixture likelihood of the basic atom with the centre, the width of the inliers and the share of outliers all unknown (priors), and the data simulated from a stated truth. The switches are summed out, so EP gives beliefs about the centre, the precision and the share, and no probability that an observation is an outlier.',
  factors: [
    'prior (gaussian, gamma, beta)',
    'bernoulli (a hidden switch per observation with a beta probability)',
    'gate of inlier and outlier branches (gaussian-gamma-precision)',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance(
    'lumpiest',
    'the posterior can be lumpy and EP matches only its mean and variance with a Gaussian, a gamma and a beta (measured on the default data: within 0.02 posterior sds, spreads 0.96 to 1.0)',
  ),
  evidence: {
    nats: 0.05,
    reason:
      'the summed-out outlier mixture with a learned scale and share: a hidden switch per observation under a gate (0.02 measured)',
  },
  knobs: dict({
    ...learnedKnobs(50, 80),
    trueTheta: real(-3, 3, { default: 1.5, step: 0.1, label: 'true centre' }),
    trueSd: real(0.2, 1.5, { default: 0.5, step: 0.05, label: 'true inlier sd' }),
    trueWeight: real(0.05, 0.5, { default: 0.25, step: 0.01, label: 'true share of outliers' }),
  }),
  ep: { damping: 0.5, tol: 1e-4 },
  parts: clutterMixtureLearnedProblem,
})

/** The atoms of the `gates` file, basic and learned twins together. */
export const gatesAtoms: readonly EpAtom[] = [
  orderedPairAtom,
  orderedMeansAtom,
  gateKnownAtom,
  gateRandomAtom,
  mixtureComponentsAtom,
  spikeSlabAtom,
  spikeSlabMarginalAtom,
  clutterAtom,
  clutterMixtureAtom,
  mixtureRegressionAtom,
  clutterLearnedAtom,
  mixtureComponentsLearnedAtom,
  gateRandomLearnedAtom,
  spikeSlabLearnedAtom,
  spikeSlabMarginalLearnedAtom,
  clutterMixtureLearnedAtom,
]
