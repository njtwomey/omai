/**
 * The setups of expectation propagation that are a switch between components, each written as the gate it is, with an
 * explicit switch `z` so that EP reports the probability of each branch and not only a blended belief: a switch between two
 * known values, between two random values (written as a select and as an index), the spike and slab prior as a switch between
 * a narrow and a wide Gaussian (and, as a second setup, with the switch summed out: the mixture site), and Minka's clutter
 * problem with a switch per observation. The exact answers are closed forms: the posterior of a switch is the ratio of the
 * branches' evidence, each branch is the conjugate update of its component, and what is selected is the mixture of the
 * branches weighted by that posterior. EP's own single Gaussian for a mixed quantity is drawn as what it is, the moment-matched
 * projection of that mixture.
 */

import * as d from '../probprog/model/dists/index.ts'
import { defineModel, ops } from '../probprog/model/index.ts'
import { dict, real } from '../space/index.ts'
import type { Config } from '../trace/index.ts'
import {
  type AtomProblem,
  type Belief,
  defineAtom,
  type EpAtom,
  type ExactMarginal,
  type MixtureBranch,
  type MixtureView,
} from './ep-atoms.ts'
import { mixtureMarginal, momentMatch, normalPdf, quadratureMarginal, selectedBranch } from './ep-atoms-quad.ts'
import { exampleNumber } from './example-method.ts'

/** A Gaussian: its mean and sd. */
interface Normal {
  readonly mean: number
  readonly sd: number
}

/** The conjugate update of a Gaussian component observed through Gaussian noise `s` at `y`. */
function update(prior: Normal, y: number, s: number): Normal {
  const prec = 1 / prior.sd ** 2 + 1 / s ** 2
  return { mean: (prior.mean / prior.sd ** 2 + y / s ** 2) / prec, sd: Math.sqrt(1 / prec) }
}

/** What a switch between two Gaussian components, observed through noise, has as its exact posterior. */
export interface TwoComponentGate {
  /** P(z = 1): the switch selects component 1. */
  readonly p1: number
  /** The components' conjugate updates (what each is if it is the selected one). */
  readonly updated: readonly [Normal, Normal]
  /** What is selected: the mixture of the two updates weighted by the switch. */
  readonly selected: ExactMarginal
  /** Each component's marginal: its update if selected, its prior otherwise. */
  readonly marginals: readonly [ExactMarginal, ExactMarginal]
}

/**
 * The exact posterior of a switch `z` (P(z = 1) = `pi`) that selects one of two independent Gaussian components, with `y`
 * observed through Gaussian noise `s`: the posterior of the switch is the ratio of the branches' evidence, each branch is the
 * conjugate update of its component, the selected value is their mixture and each component's marginal is its update if it was
 * selected and its prior otherwise.
 *
 * @param pi - The prior probability of component 1.
 * @param components - The two components' priors.
 * @param y - The observation.
 * @param s - The observation noise.
 * @returns The posterior.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { twoComponentGate } from 'omai-engine/bayesian-examples'
 * const g = twoComponentGate(0.5, [{ mean: 0, sd: 1 }, { mean: 0, sd: 1 }], 0.3, 1)
 * g.p1 // => 0.5
 * ```
 */
export function twoComponentGate(
  pi: number,
  components: readonly [Normal, Normal],
  y: number,
  s: number,
): TwoComponentGate {
  const e = components.map((c) => normalPdf(y, c.mean, Math.sqrt(c.sd ** 2 + s ** 2)))
  const e1 = pi * (e[1] ?? 0)
  const e0 = (1 - pi) * (e[0] ?? 0)
  const p1 = e1 / (e1 + e0)
  const updated: [Normal, Normal] = [update(components[0], y, s), update(components[1], y, s)]
  return {
    p1,
    updated,
    selected: mixtureMarginal([
      { weight: 1 - p1, ...updated[0] },
      { weight: p1, ...updated[1] },
    ]),
    marginals: [
      mixtureMarginal([
        { weight: 1 - p1, ...updated[0] },
        { weight: p1, ...components[0] },
      ]),
      mixtureMarginal([
        { weight: p1, ...updated[1] },
        { weight: 1 - p1, ...components[1] },
      ]),
    ],
  }
}

const probability = (p: number): ExactMarginal => ({ mean: p, sd: Math.sqrt(p * (1 - p)), pdf: () => 0 })

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
  priors: readonly [Normal, Normal],
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
  priors: readonly [Normal, Normal],
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
  prior: Normal,
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

function knownGateProblem(k: Config): AtomProblem {
  const pi = exampleNumber(k, 'pi')
  const y = exampleNumber(k, 'y')
  const definition = defineModel('gate-known', (m) => {
    const z = m.latent('z', d.bernoulli({ probs: pi }))
    m.observe('y', d.normal(ops.where(z, 3, 0), 1), m.data('y'))
  })
  const w1 = pi * normalPdf(y, 3, 1)
  const w0 = (1 - pi) * normalPdf(y, 0, 1)
  const p = w1 / (w1 + w0)
  const ys = Array.from({ length: 141 }, (_, i) => -4 + (i * 11) / 140)
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
    exact: { z: probability(p) },
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
    y: real(-3, 6, { default: 2.5, step: 0.1, label: 'observed y' }),
  }),
  problem: knownGateProblem,
})

// ---------------------------------------------------------------------------------------------------------------------
// A switch between two random values, as a select and as an index.

function randomGateProblem(k: Config, form: 'select' | 'index'): AtomProblem {
  const pi = exampleNumber(k, 'pi')
  const y = exampleNumber(k, 'y')
  const s = 0.5
  const priors: readonly [Normal, Normal] = [
    { mean: -2, sd: 1 },
    { mean: 2, sd: 1 },
  ]
  const sites: readonly [string, string] = form === 'select' ? ['a', 'b'] : ['mu[0]', 'mu[1]']
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
    definition,
    data: form === 'select' ? { y } : { y, loc: [priors[0].mean, priors[1].mean] },
    ...(form === 'select' ? {} : { bindOptions: { sizes: { component: 2 } } }),
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
    exact: { z: probability(gate.p1), [sites[0]]: gate.marginals[0], [sites[1]]: gate.marginals[1], g: gate.selected },
    mixtures: [
      selectedView,
      componentView(sites[0], names[0], 0, gate, priors[0], 'z'),
      componentView(sites[1], names[1], 1, gate, priors[1], 'z'),
    ],
  }
}

const gateKnobs = dict({
  pi: real(0.02, 0.98, { default: 0.5, step: 0.01, label: 'P(z = 1) before the observation' }),
  y: real(-4, 4, { default: 0.5, step: 0.1, label: 'observed y' }),
})
const gateTolerance = {
  meanZ: 1e-3,
  sdRatio: [0.999, 1.001] as const,
  ks: 0.25,
  reason:
    'the means, spreads and the probability of the switch are exact (moment matching through a gate); the Kolmogorov distance is large because each exact marginal is a mixture and EP draws one Gaussian for it',
}

/**
 * A switch that selects one of two random variables, written as a select, `where(z, b, a)`: the selected one is pulled
 * towards the observation, the other keeps its prior.
 *
 * @index estimator EP atom: a gate between two random components (select), drawn as its branches.
 * @example
 * ```ts
 * import { gateRandomAtom } from 'omai-engine/bayesian-examples'
 * gateRandomAtom.family // => 'gates'
 * ```
 */
export const gateRandomAtom: EpAtom = defineAtom({
  name: 'gate-random',
  title: 'Gate between two random values',
  what: 'A Bernoulli switch z selects one of two Gaussian variables a and b, and y is the selected one plus noise. EP reports the probability of z; the exact posterior of what is selected is the mixture of the two updated components, weighted by it.',
  factors: ['prior (bernoulli)', 'prior (gaussian) x2', 'select gate (fresh random components)', 'gaussian likelihood'],
  family: 'gates',
  expected: 'runs',
  reference: 'exact',
  tolerance: gateTolerance,
  knobs: gateKnobs,
  problem: (k) => randomGateProblem(k, 'select'),
})

/**
 * The same gate written as an index, `mu[z]`, with a categorical switch.
 *
 * @index estimator EP atom: a random index into random components (mu[z]), drawn as its branches.
 * @example
 * ```ts
 * import { mixtureComponentsAtom } from 'omai-engine/bayesian-examples'
 * mixtureComponentsAtom.name // => 'mixture-components'
 * ```
 */
export const mixtureComponentsAtom: EpAtom = defineAtom({
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
  expected: 'runs',
  reference: 'exact',
  tolerance: gateTolerance,
  knobs: gateKnobs,
  problem: (k) => randomGateProblem(k, 'index'),
})

// ---------------------------------------------------------------------------------------------------------------------
// Spike and slab.

/**
 * The exact posterior of a coefficient with a spike and slab prior (inclusion probability `pi`, slab sd, spike sd) observed
 * through noise: a switch between the spike and the slab, so the same closed form as a two-component gate with zero means.
 *
 * @param pi - The prior probability of the slab.
 * @param slab - The slab's sd.
 * @param spike - The spike's sd.
 * @param y - The observation.
 * @param s - The noise sd.
 * @returns The posterior: `p1` is the posterior probability of the slab, `updated[0]` the spike branch and `updated[1]` the slab branch.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { spikeSlabPosterior } from 'omai-engine/bayesian-examples'
 * spikeSlabPosterior(0.5, 2, 0.1, 0, 0.5).p1 < 0.2 // => true
 * ```
 */
export function spikeSlabPosterior(pi: number, slab: number, spike: number, y: number, s: number): TwoComponentGate {
  return twoComponentGate(
    pi,
    [
      { mean: 0, sd: spike },
      { mean: 0, sd: slab },
    ],
    y,
    s,
  )
}

const SPIKE_NOISE = 0.5

const spikeKnobs = dict({
  pi: real(0.05, 0.95, { default: 0.5, step: 0.01, label: 'prior probability of the slab' }),
  y: real(-5, 6, { default: 0.8, step: 0.1, label: 'observed y' }),
  slab: real(1, 6, { default: 2, step: 0.1, label: 'slab width (sd)' }),
  spike: real(0.01, 0.5, { default: 0.1, step: 0.01, label: 'spike width (sd)' }),
})

function spikeSlabGateProblem(k: Config): AtomProblem {
  const pi = exampleNumber(k, 'pi')
  const y = exampleNumber(k, 'y')
  const slab = exampleNumber(k, 'slab')
  const spike = exampleNumber(k, 'spike')
  const s = SPIKE_NOISE
  const priors: readonly [Normal, Normal] = [
    { mean: 0, sd: spike },
    { mean: 0, sd: slab },
  ]
  const definition = defineModel('spike-and-slab', (m) => {
    const z = m.latent('z', d.bernoulli({ probs: pi }))
    const narrow = m.latent('spike', d.normal(0, spike))
    const wide = m.latent('slab', d.normal(0, slab))
    m.observe('y', d.normal(ops.where(z, wide, narrow), s), m.data('y'))
  })
  const gate = spikeSlabPosterior(pi, slab, spike, y, s)
  const sites: readonly [string, string] = ['spike', 'slab']
  return {
    definition,
    data: { y },
    equations: [
      `z ~ Bernoulli(${pi})   (z = 1: the coefficient is in the slab)`,
      `spike ~ N(0, ${spike}²),  slab ~ N(0, ${slab}²)   (a narrow and a wide component)`,
      'x = z ? slab : spike   (the gate: the coefficient is the component z selects)',
      `y ~ N(x, ${s}²)`,
      `observe y = ${y}`,
    ],
    quantities: [
      { name: 'z', label: 'P(slab)', support: 'probability' },
      { name: 'spike', label: 'spike component', support: 'real', hidden: true },
      { name: 'slab', label: 'slab component', support: 'real', hidden: true },
      { name: 'x', label: 'x (the coefficient)', support: 'real', derive: (b) => epSelected(b, sites, priors, 'z') },
    ],
    exact: { z: probability(gate.p1), spike: gate.marginals[0], slab: gate.marginals[1], x: gate.selected },
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
  }
}

/**
 * A coefficient that is a switch between a very narrow Gaussian (the spike) and a wide one (the slab): the gate with
 * explicit `z`, so EP reports the posterior probability of the slab and the two branches, not only one blended belief.
 *
 * @index estimator EP atom: spike and slab as a gate between a narrow and a wide Gaussian.
 * @example
 * ```ts
 * import { spikeSlabAtom } from 'omai-engine/bayesian-examples'
 * spikeSlabAtom.name // => 'spike-and-slab'
 * ```
 */
export const spikeSlabAtom: EpAtom = defineAtom({
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
  expected: 'runs',
  reference: 'exact',
  tolerance: gateTolerance,
  knobs: spikeKnobs,
  problem: spikeSlabGateProblem,
})

function spikeSlabMarginalProblem(k: Config): AtomProblem {
  const pi = exampleNumber(k, 'pi')
  const y = exampleNumber(k, 'y')
  const slab = exampleNumber(k, 'slab')
  const spike = exampleNumber(k, 'spike')
  const s = SPIKE_NOISE
  const definition = defineModel('spike-and-slab-marginal', (m) => {
    const w = m.latent('w', d.normalSpikeSlab(pi, slab, spike))
    m.observe('y', d.normal(w, s), m.data('y'))
  })
  const gate = spikeSlabPosterior(pi, slab, spike, y, s)
  return {
    definition,
    data: { y },
    equations: [
      `w ~ ${pi} N(0, ${slab}²) + ${(1 - pi).toFixed(2)} N(0, ${spike}²)   (spike and slab, the switch summed out)`,
      `y ~ N(w, ${s}²)`,
      `observe y = ${y}`,
    ],
    quantities: [{ name: 'w', label: 'w (the coefficient)', support: 'real' }],
    exact: { w: gate.selected },
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
  }
}

/**
 * The same prior with the switch summed out, as the library's spike and slab family: a mixture site on one variable, whose
 * moments are exact but which has no variable for the probability of the slab.
 *
 * @index estimator EP atom: spike and slab with the switch summed out (mixture site, no P(slab)).
 * @example
 * ```ts
 * import { spikeSlabMarginalAtom } from 'omai-engine/bayesian-examples'
 * spikeSlabMarginalAtom.name // => 'spike-and-slab-marginal'
 * ```
 */
export const spikeSlabMarginalAtom: EpAtom = defineAtom({
  name: 'spike-and-slab-marginal',
  title: 'Spike and slab, switch summed out',
  what: 'The same prior with the switch summed out (the mixture-site form): one variable with a two-component zero-centred prior. EP matches the mean and variance of the exact two-branch posterior with one Gaussian and has no variable for the probability of the slab.',
  factors: ['mixture-site (two-component prior)', 'gaussian likelihood'],
  family: 'gates',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 1e-3,
    sdRatio: [0.999, 1.001],
    ks: 0.3,
    reason:
      'the mean and variance are exact; the exact posterior is a spike hump and a slab hump and EP draws one Gaussian',
  },
  knobs: spikeKnobs,
  problem: spikeSlabMarginalProblem,
})

// ---------------------------------------------------------------------------------------------------------------------
// Clutter.

const CLUTTER_Y = [1.0, 1.4, 0.9, 1.2, 1.6, -4.2, 1.1, 9.5]
const INLIER_SD = 0.5
const CLUTTER_SD = 5

function clutterGateProblem(k: Config): AtomProblem {
  const w = exampleNumber(k, 'share')
  const priorSd = 5
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
  const logPost = (t: number): number =>
    Math.log(normalPdf(t, 0, priorSd)) +
    CLUTTER_Y.reduce((a, v) => a + Math.log((1 - w) * normalPdf(v, t, INLIER_SD) + w * normalPdf(v, 0, CLUTTER_SD)), 0)
  const marginal = quadratureMarginal(logPost, -20, 20)
  // P(observation i is clutter) = the posterior of the location averaged over the branch weight of that observation.
  const M = 8000
  const ts = Array.from({ length: M }, (_, i) => -20 + ((i + 0.5) * 40) / M)
  const lw = ts.map(logPost)
  const top = Math.max(...lw)
  const wt = lw.map((l) => Math.exp(l - top))
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
    definition,
    data: { y: CLUTTER_Y },
    equations: [
      `theta ~ N(0, ${priorSd}²)   (the hidden centre)`,
      `z_i ~ Bernoulli(${w})   (z_i = 1: observation i is clutter)`,
      `y_i ~ z_i ? N(0, ${CLUTTER_SD}²) : N(theta, ${INLIER_SD}²)   (a gate per observation: clutter or the centre)`,
      `observe y = (${CLUTTER_Y.join(', ')})`,
    ],
    quantities: [
      { name: 'theta', label: 'theta (the centre)', support: 'real' },
      ...at.map((i) => ({
        name: `z[${i - 1}]`,
        label: `observation ${i} is clutter`,
        support: 'probability' as const,
      })),
    ],
    exact: { theta: marginal, ...Object.fromEntries(pClutter.map((p, i) => [`z[${i}]`, probability(p)])) },
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
  }
}

/**
 * Minka's clutter problem with an explicit switch per observation: each observation is a draw from a wide clutter component
 * around zero or a detection of the hidden centre, so EP reports the probability that each is clutter as well as the belief
 * about the centre.
 *
 * @index estimator EP atom: clutter, a gate per observation between the centre and wide clutter.
 * @example
 * ```ts
 * import { clutterAtom } from 'omai-engine/bayesian-examples'
 * clutterAtom.family // => 'gates'
 * ```
 */
export const clutterAtom: EpAtom = defineAtom({
  name: 'clutter',
  title: 'Clutter',
  what: 'A hidden centre theta observed eight times, each observation a gate between a detection of the centre and wide clutter around zero (a switch z_i per observation, two of which look like clutter). EP reports the probability that each is clutter and the belief about the centre.',
  factors: [
    'prior (gaussian)',
    'prior (bernoulli) per observation',
    'select gate per observation (random centre, known clutter)',
  ],
  family: 'gates',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.5,
    sdRatio: [0.5, 1.6],
    ks: 0.3,
    reason:
      'the exact posterior of the centre is lumpy (a clutter reading can be explained two ways) and EP matches only its mean and variance, observation by observation',
  },
  knobs: dict({ share: real(0.02, 0.6, { default: 0.25, step: 0.01, label: 'prior probability of clutter' }) }),
  ep: { damping: 0.5 },
  problem: clutterGateProblem,
})

function clutterMixtureProblem(k: Config): AtomProblem {
  const share = exampleNumber(k, 'share')
  const [s, width, priorSd] = [0.5, 10, 5]
  const definition = defineModel('clutter-mixture', (m) => {
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
      `y_i ~ ${(1 - share).toFixed(2)} N(mu, ${s}²) + ${share} N(mu, ${width * s}²)   (inlier or outlier, the switch summed out)`,
      `observe y = (${CLUTTER_Y.join(', ')})`,
    ],
    quantities: [{ name: 'mu', label: 'mu', support: 'real' }],
    exact: { mu: quadratureMarginal(logPost, -20, 20) },
  }
}

/**
 * The inlier and outlier mixture likelihood with the switches summed out (the library's `normalOutliers` family): a mixture
 * site per observation around the centre, which has no variable for the probability that an observation is an outlier.
 *
 * @index estimator EP atom: an inlier and outlier mixture likelihood with the switches summed out (mixture site).
 * @example
 * ```ts
 * import { clutterMixtureAtom } from 'omai-engine/bayesian-examples'
 * clutterMixtureAtom.name // => 'clutter-mixture'
 * ```
 */
export const clutterMixtureAtom: EpAtom = defineAtom({
  name: 'clutter-mixture',
  title: 'Inlier and outlier mixture, switches summed out',
  what: 'An unknown mean mu observed eight times through a mixture of a narrow and a wide Gaussian around it, two observations being far from the rest: the mixture site as a likelihood. The switch per observation is summed out, so EP gives a belief about mu and no probability that an observation is an outlier.',
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
  problem: clutterMixtureProblem,
})
