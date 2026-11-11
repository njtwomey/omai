/**
 * The setups of expectation propagation where the gate itself has parameters to learn: a mixture of known components whose
 * mixing weight has a prior (a beta for two components, a Dirichlet for three), with the switch of each observation kept as a
 * variable (so EP reports each observation's responsibility) or summed out (one factor on the weight per observation, the same
 * message), and a hyperprior one level up on how strong the prior on the weight is. The exact answers are sums over the
 * switches done analytically, so the posterior of the weight is a one- or two-dimensional quadrature and each responsibility is
 * its average over that posterior: EP carries one beta (or Dirichlet) for the weight, and the pages show where that single
 * distribution differs from the exact one.
 */

import * as d from '../probprog/model/dists/index.ts'
import { defineModel, ops } from '../probprog/model/index.ts'
import { dict, real } from '../space/index.ts'
import type { Config } from '../trace/index.ts'
import { type AtomProblem, type AtomTolerance, defineAtom, type EpAtom, type ExactMarginal } from './ep-atoms.ts'
import { exampleNumber } from './example-method.ts'

const BASE = [-2.1, 1.9, 0.3, 2.2, -0.4, 1.7, 0.6, -1.8, 0.9, 2.3]
const SD = 0.5
const MEANS = [-2, 2]

const normalDensity = (x: number, mean: number, sd: number): number =>
  Math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))
const lgammaStirling = (x: number): number => {
  let s = 0
  let z = x
  while (z < 10) {
    s -= Math.log(z)
    z += 1
  }
  return (
    s +
    (z - 0.5) * Math.log(z) -
    z +
    0.5 * Math.log(2 * Math.PI) +
    1 / (12 * z) -
    1 / (360 * z ** 3) +
    1 / (1260 * z ** 5)
  )
}

/** The observations the knob `spread` makes: the base set scaled towards zero, where the two components overlap. */
const observations = (k: Config): number[] => BASE.map((v) => v * exampleNumber(k, 'spread'))

/** A probability quantity: the exact probability that a switch is on, and its spread as a coin. */
const responsibility = (p: number): ExactMarginal => ({ mean: p, sd: Math.sqrt(p * (1 - p)), pdf: () => 0 })

/** The marginal of a quantity on a grid of midpoints with weights, as an exact marginal with a piecewise-constant density. */
function gridMarginal(points: readonly number[], weights: readonly number[]): ExactMarginal {
  const z = weights.reduce((a, b) => a + b, 0)
  const mean = points.reduce((a, x, i) => a + (x * (weights[i] as number)) / z, 0)
  const sd = Math.sqrt(points.reduce((a, x, i) => a + ((x - mean) ** 2 * (weights[i] as number)) / z, 0))
  const step = (points[1] as number) - (points[0] as number)
  const top = Math.max(...weights)
  const live = points.filter((_, i) => (weights[i] as number) > 1e-6 * top)
  return {
    mean,
    sd,
    pdf: (x) => {
      const i = Math.round((x - (points[0] as number)) / step)
      return i < 0 || i >= points.length ? 0 : (weights[i] as number) / z / step
    },
    range: [Math.min(...live) - step, Math.max(...live) + step],
  }
}

/**
 * The exact posterior of the weight of the second component (a beta prior) and each observation's responsibility: the switches
 * summed out one by one, so the posterior of the weight is its prior times the product of the observations' mixture likelihoods, a
 * one-dimensional integral, and a responsibility is the average of the conditional responsibility over it.
 */
function twoComponentExact(
  x: readonly number[],
  logPrior: (p: number) => number,
): { readonly weight: ExactMarginal; readonly responsibilities: readonly number[] } {
  const T = 4000
  const ps = Array.from({ length: T }, (_, i) => (i + 0.5) / T)
  const l1 = x.map((v) => normalDensity(v, MEANS[1] as number, SD))
  const l0 = x.map((v) => normalDensity(v, MEANS[0] as number, SD))
  const logs = ps.map(
    (p) => logPrior(p) + x.reduce((a, _, i) => a + Math.log(p * (l1[i] as number) + (1 - p) * (l0[i] as number)), 0),
  )
  const top = Math.max(...logs)
  const w = logs.map((v) => Math.exp(v - top))
  const z = w.reduce((a, b) => a + b, 0)
  return {
    weight: gridMarginal(ps, w),
    responsibilities: x.map((_, i) =>
      ps.reduce(
        (a, p, j) =>
          a +
          ((w[j] as number) / z) * ((p * (l1[i] as number)) / (p * (l1[i] as number) + (1 - p) * (l0[i] as number))),
        0,
      ),
    ),
  }
}

const GATE_TOLERANCE: AtomTolerance = {
  meanZ: 0.15,
  sdRatio: [0.85, 1.15],
  ks: 0.08,
  reason:
    'the exact posterior of the weight is a product of mixtures of two terms each, skewed or two-humped where observations are ambiguous; EP carries one beta for it, matched on its log moments through each switch',
}

const priorKnobs = {
  a: real(0.3, 8, { default: 1, step: 0.1, label: 'prior a of the weight' }),
  b: real(0.3, 8, { default: 1, step: 0.1, label: 'prior b of the weight' }),
  spread: real(0.1, 1.2, {
    default: 0.3,
    step: 0.05,
    label: 'spread of the observations (small: the components overlap)',
  }),
}

function weightQuantities(n: number): AtomProblem['quantities'] {
  return [
    { name: 'pi', label: 'pi (the mixing weight of the + component)', support: 'unit' },
    ...Array.from({ length: n }, (_, i) => ({
      name: `z[${i}]`,
      label: `P(observation ${i + 1} is from the + component)`,
      support: 'probability' as const,
    })),
  ]
}

const equationsOf = (a: number, b: number, x: readonly number[], summed: boolean): string[] => [
  `pi ~ Beta(${a}, ${b})   (the weight of the + component)`,
  summed
    ? `x_i ~ (1 - pi) N(${MEANS[0]}, ${SD}²) + pi N(${MEANS[1]}, ${SD}²)   (the switch summed out)`
    : `z_i ~ Bernoulli(pi),  z_i = 0: x_i ~ N(${MEANS[0]}, ${SD}²),  z_i = 1: x_i ~ N(${MEANS[1]}, ${SD}²)`,
  `observe x = (${x.map((v) => v.toFixed(2)).join(', ')})`,
]

function explicitProblem(k: Config): AtomProblem {
  const [a, b] = [exampleNumber(k, 'a'), exampleNumber(k, 'b')]
  const x = observations(k)
  // region gate-weight
  const definition = defineModel('gate-weight', (m) => {
    const i = m.range('i')
    const pi = m.latent('pi', d.beta(a, b))
    const z = m.latent('z', d.bernoulli({ probs: pi }), { dims: [i] })
    const observed = m.data('x', { dims: [i], role: 'response' })
    m.observe('x', d.normal(ops.sub(ops.mul(z, 4), 2), SD), observed)
  })
  // endregion
  const exact = twoComponentExact(x, (p) => (a - 1) * Math.log(p) + (b - 1) * Math.log(1 - p))
  return {
    definition,
    data: { x },
    equations: equationsOf(a, b, x, false),
    quantities: weightQuantities(x.length),
    exact: {
      pi: exact.weight,
      ...Object.fromEntries(exact.responsibilities.map((p, i) => [`z[${i}]`, responsibility(p)])),
    },
  }
}

/**
 * A mixture of two known Gaussians whose mixing weight has a beta prior, the switch of each observation kept as a variable: the
 * conjugate Bernoulli-beta factor ties every switch to the weight, and EP reports the weight as one beta and each observation's
 * responsibility. The exact answer sums the switches out analytically.
 *
 * @index estimator EP atom: a gate with an unknown mixing weight (a beta prior on the Bernoulli switch probability).
 * @example
 * ```ts
 * import { gateWeightAtom } from 'omai-engine/bayesian-examples'
 * gateWeightAtom.name // => 'gate-weight'
 * ```
 */
export const gateWeightAtom: EpAtom = defineAtom({
  name: 'gate-weight',
  title: 'Gate with an unknown mixing weight (beta prior)',
  what: 'A mixture of two known narrow Gaussians with ten observations, each with its own switch z_i ~ Bernoulli(pi) and a beta prior on pi: the posterior of the weight and each switch from EP, against the switches summed out exactly.',
  factors: [
    'prior (beta)',
    'bernoulli (a switch with a beta probability)',
    'table (the component mean chosen by the switch)',
  ],
  family: 'gate weights',
  expected: 'runs',
  reference: 'exact',
  tolerance: GATE_TOLERANCE,
  knobs: dict(priorKnobs),
  problem: explicitProblem,
})

function summedOutProblem(k: Config): AtomProblem {
  const [a, b] = [exampleNumber(k, 'a'), exampleNumber(k, 'b')]
  const x = observations(k)
  // region gate-weight-summed-out
  const definition = defineModel('gate-weight-summed-out', (m) => {
    const i = m.range('i')
    const pi = m.latent('pi', d.beta(a, b))
    const observed = m.data('x', { dims: [i], role: 'response' })
    m.observe('x', d.normalPair(pi, MEANS[0] as number, MEANS[1] as number, SD), observed)
  })
  // endregion
  const exact = twoComponentExact(x, (p) => (a - 1) * Math.log(p) + (b - 1) * Math.log(1 - p))
  return {
    definition,
    data: { x },
    equations: equationsOf(a, b, x, true),
    quantities: [{ name: 'pi', label: 'pi (the mixing weight of the + component)', support: 'unit' }],
    exact: { pi: exact.weight },
  }
}

/**
 * The same model with the switches summed out: each observation is one factor on the weight, `l1 pi + l0 (1 - pi)`. EP's message
 * from it equals the one the explicit gate sends, so the two routes give the same posterior of the weight; this one has no
 * variable for the switches and so no responsibilities.
 *
 * @index estimator EP atom: a mixture with an unknown weight, the switches summed out (mixture-weight factor).
 * @example
 * ```ts
 * import { gateWeightSummedOutAtom } from 'omai-engine/bayesian-examples'
 * gateWeightSummedOutAtom.name // => 'gate-weight-summed-out'
 * ```
 */
export const gateWeightSummedOutAtom: EpAtom = defineAtom({
  name: 'gate-weight-summed-out',
  title: 'Mixture with an unknown weight, switches summed out',
  what: 'The same mixture and beta prior with the switches summed out: one mixture-weight factor per observation on the weight, so EP reports the weight but not each observation’s responsibility.',
  factors: ['prior (beta)', 'mixture-weight (a two-component mixture with a beta weight)'],
  family: 'gate weights',
  expected: 'runs',
  reference: 'exact',
  tolerance: GATE_TOLERANCE,
  knobs: dict(priorKnobs),
  problem: summedOutProblem,
})

// ----- three components and a Dirichlet on the weights -----

const THREE = [-3, 0, 3]
const BASE3 = [-2.9, 3.1, 0.2, 2.8, -0.3, -3.2, 0.5, 1.4, -1.6, 2.9]

function dirichletWeightProblem(k: Config): AtomProblem {
  const a = exampleNumber(k, 'concentration')
  const x = BASE3.map((v) => v * exampleNumber(k, 'spread'))
  // region gate-weight-dirichlet
  const definition = defineModel('gate-weight-dirichlet', (m) => {
    const i = m.range('i')
    const pi = m.latent('pi', d.dirichlet([a, a, a]))
    const z = m.latent('z', d.categorical({ probs: pi }), { dims: [i] })
    const means = m.data('means', { dims: ['k'] })
    const observed = m.data('x', { dims: [i], role: 'response' })
    m.observe('x', d.normal(ops.at(means, z), SD), observed)
  })
  // endregion
  // The posterior of the weights on the simplex, summed over the switches analytically: a grid over (pi0, pi1).
  const G = 220
  const l = x.map((v) => THREE.map((c) => normalDensity(v, c, SD)))
  const marg = [new Array<number>(G).fill(0), new Array<number>(G).fill(0), new Array<number>(G).fill(0)] as [
    number[],
    number[],
    number[],
  ]
  const logs: { i: number; j: number; v: number }[] = []
  let top = Number.NEGATIVE_INFINITY
  for (let i = 0; i < G; i++)
    for (let j = 0; j < G; j++) {
      const p0 = (i + 0.5) / G
      const p1 = (j + 0.5) / G
      const p2 = 1 - p0 - p1
      if (p2 <= 0) continue
      const v =
        (a - 1) * (Math.log(p0) + Math.log(p1) + Math.log(p2)) +
        l.reduce(
          (s, row) => s + Math.log(p0 * (row[0] as number) + p1 * (row[1] as number) + p2 * (row[2] as number)),
          0,
        )
      logs.push({ i, j, v })
      if (v > top) top = v
    }
  for (const { i, j, v } of logs) {
    const w = Math.exp(v - top)
    ;(marg[0][i] as number) += w
    ;(marg[1][j] as number) += w
    // pi2 = 1 - pi0 - pi1 falls in the cell of the grid nearest to it.
    const k2 = Math.min(G - 1, Math.max(0, Math.round((1 - (i + 0.5) / G - (j + 0.5) / G) * G - 0.5)))
    ;(marg[2][k2] as number) += w
  }
  const points = Array.from({ length: G }, (_, i) => (i + 0.5) / G)
  const exact = Object.fromEntries(marg.map((w, c) => [`pi[${c}]`, gridMarginal(points, w)]))
  return {
    definition,
    data: { x, means: THREE },
    equations: [
      `pi ~ Dirichlet(${a}, ${a}, ${a})   (the weights of the three components)`,
      `z_i ~ Categorical(pi),  x_i ~ N(mean[z_i], ${SD}²), means (${THREE.join(', ')})`,
      `observe x = (${x.map((v) => v.toFixed(2)).join(', ')})`,
    ],
    quantities: THREE.map((c, i) => ({
      name: `pi[${i}]`,
      label: `pi of the component at ${c}`,
      support: 'unit' as const,
    })),
    exact,
  }
}

/**
 * A mixture of three known Gaussians with a Dirichlet prior on the weights and a categorical switch per observation: the
 * conjugate discrete-Dirichlet factor. The exact answer is the posterior on the simplex, summed over the switches analytically.
 *
 * @index estimator EP atom: a gate with unknown weights of three components (a Dirichlet prior on the switch probabilities).
 * @example
 * ```ts
 * import { gateWeightDirichletAtom } from 'omai-engine/bayesian-examples'
 * gateWeightDirichletAtom.name // => 'gate-weight-dirichlet'
 * ```
 */
export const gateWeightDirichletAtom: EpAtom = defineAtom({
  name: 'gate-weight-dirichlet',
  title: 'Gate with unknown mixing weights (Dirichlet prior, three components)',
  what: 'A mixture of three known narrow Gaussians with ten observations, a categorical switch per observation and a Dirichlet prior on the three weights: the posterior of each weight from EP, against the exact one on the simplex.',
  factors: [
    'prior (dirichlet)',
    'discrete (a switch with a Dirichlet probability vector)',
    'table (the component mean chosen by the switch)',
  ],
  family: 'gate weights',
  expected: 'runs',
  reference: 'exact',
  tolerance: { ...GATE_TOLERANCE, meanZ: 0.2, sdRatio: [0.8, 1.2], ks: 0.1 },
  knobs: dict({
    concentration: real(0.3, 6, { default: 1, step: 0.1, label: 'prior concentration of each weight' }),
    spread: real(0.1, 1.2, {
      default: 0.3,
      step: 0.05,
      label: 'spread of the observations (small: the components overlap)',
    }),
  }),
  problem: dirichletWeightProblem,
})

// ----- a hyperprior on the strength of the prior on the weight -----

function hyperWeightProblem(k: Config): AtomProblem {
  const [shape, rate] = [exampleNumber(k, 'shape'), exampleNumber(k, 'rate')]
  const x = observations(k)
  // region gate-weight-hyperprior
  const definition = defineModel('gate-weight-hyperprior', (m) => {
    const i = m.range('i')
    const strength = m.latent('strength', d.gamma(shape, rate))
    const pi = m.latent('pi', d.beta(strength, strength))
    const z = m.latent('z', d.bernoulli({ probs: pi }), { dims: [i] })
    const observed = m.data('x', { dims: [i], role: 'response' })
    m.observe('x', d.normal(ops.sub(ops.mul(z, 4), 2), SD), observed)
  })
  // endregion
  const l1 = x.map((v) => normalDensity(v, MEANS[1] as number, SD))
  const l0 = x.map((v) => normalDensity(v, MEANS[0] as number, SD))
  const P = 260
  const S = 260
  const ps = Array.from({ length: P }, (_, i) => (i + 0.5) / P)
  const lo = -5
  const hi = 4
  const ts = Array.from({ length: S }, (_, j) => lo + ((j + 0.5) * (hi - lo)) / S)
  const logLik = ps.map((p) =>
    x.reduce((a, _, i) => a + Math.log(p * (l1[i] as number) + (1 - p) * (l0[i] as number)), 0),
  )
  const w: number[][] = []
  let top = Number.NEGATIVE_INFINITY
  const raw: number[][] = ps.map((p, i) =>
    ts.map((t) => {
      const s = Math.exp(t)
      const v =
        (shape - 1) * t -
        rate * s +
        t + // the gamma prior of the strength, in log s
        2 * lgammaStirling(s) * -1 +
        lgammaStirling(2 * s) +
        (s - 1) * (Math.log(p) + Math.log(1 - p)) +
        (logLik[i] as number)
      if (v > top) top = v
      return v
    }),
  )
  for (const row of raw) w.push(row.map((v) => Math.exp(v - top)))
  const z = w.reduce((a, row) => a + row.reduce((b, v) => b + v, 0), 0)
  const piW = w.map((row) => row.reduce((a, v) => a + v, 0))
  const sW = ts.map((_, j) => w.reduce((a, row) => a + (row[j] as number), 0))
  const exact = {
    pi: gridMarginal(ps, piW),
    strength: ((): ExactMarginal => {
      const alphas = ts.map(Math.exp)
      const mean = alphas.reduce((a, s, j) => a + (s * (sW[j] as number)) / z, 0)
      const sd = Math.sqrt(alphas.reduce((a, s, j) => a + ((s - mean) ** 2 * (sW[j] as number)) / z, 0))
      return {
        mean,
        sd,
        pdf: (s) => {
          if (s <= 0) return 0
          const j = Math.round(((Math.log(s) - lo) / (hi - lo)) * S - 0.5)
          return j < 0 || j >= S ? 0 : (sW[j] as number) / z / (((hi - lo) / S) * s)
        },
        range: [Math.max(0, mean - 3 * sd), mean + 7 * sd],
      }
    })(),
  }
  const responsibilities = x.map((_, i) =>
    ps.reduce(
      (a, p, pi) =>
        a +
        ((piW[pi] as number) / z) * ((p * (l1[i] as number)) / (p * (l1[i] as number) + (1 - p) * (l0[i] as number))),
      0,
    ),
  )
  return {
    definition,
    data: { x },
    equations: [
      `strength ~ Gamma(${shape}, ${rate})   (the hyperprior: how concentrated the weight prior is)`,
      'pi ~ Beta(strength, strength)   (the weight of the + component)',
      'z_i ~ Bernoulli(pi),  z_i = 0: x_i ~ N(-2, 0.5²), z_i = 1: x_i ~ N(+2, 0.5²)',
      `observe x = (${x.map((v) => v.toFixed(2)).join(', ')})`,
    ],
    quantities: [
      ...weightQuantities(x.length),
      { name: 'strength', label: 'strength (the hyperparameter)', support: 'positive' },
    ],
    exact: {
      pi: exact.pi,
      strength: exact.strength,
      ...Object.fromEntries(responsibilities.map((p, i) => [`z[${i}]`, responsibility(p)])),
    },
  }
}

/**
 * The mixing weight's prior is a symmetric beta whose strength has a gamma hyperprior: priors and hyperpriors going up through a
 * gate. The strength factor is the numeric `dirichlet-concentration`; the exact answer is a two-dimensional quadrature over the
 * weight and the strength with the switches summed out.
 *
 * @index estimator EP atom: a gate whose weight has a beta prior with a gamma hyperprior on its strength.
 * @example
 * ```ts
 * import { gateWeightHyperAtom } from 'omai-engine/bayesian-examples'
 * gateWeightHyperAtom.family // => 'gate weights'
 * ```
 */
export const gateWeightHyperAtom: EpAtom = defineAtom({
  name: 'gate-weight-hyperprior',
  title: 'Gate with an unknown mixing weight, gamma hyperprior on its prior strength',
  what: 'The beta prior on the mixing weight is symmetric with a gamma-distributed strength: a hyperprior one level above the gate, so the data inform the strength through how unbalanced the switches are.',
  factors: [
    'prior (gamma)',
    'dirichlet-concentration (a beta with a random strength)',
    'bernoulli (a switch with a beta probability)',
    'table',
  ],
  family: 'gate weights',
  expected: 'runs',
  reference: 'exact',
  tolerance: { ...GATE_TOLERANCE, meanZ: 0.2, sdRatio: [0.8, 1.2], ks: 0.1 },
  knobs: dict({
    shape: real(0.5, 8, { default: 2, step: 0.1, label: 'shape of the strength prior' }),
    rate: real(0.1, 4, { default: 1, step: 0.1, label: 'rate of the strength prior' }),
    spread: real(0.1, 1.2, {
      default: 0.3,
      step: 0.05,
      label: 'spread of the observations (small: the components overlap)',
    }),
  }),
  problem: hyperWeightProblem,
})
