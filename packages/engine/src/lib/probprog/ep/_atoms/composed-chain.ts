/**
 * The composed setups of expectation propagation: small models that put several pieces together, each checked against a
 * brute-force reference. The first is a Gaussian chain of hidden states with an unknown level, an unknown persistence on a grid and a
 * gamma precision of its steps, read through a non-linear function of the state. The reference is a point-mass filter for the
 * likelihood of each (level, persistence, precision) on a three-dimensional grid, made when it is first read.
 */

import { foldIn, key, normal } from '../../../random/index.ts'
import { dict, int, real } from '../../../space/index.ts'
import { add, arange, div, exp, mul, neg, sub, toFlat, zeros } from '../../../tensor/index.ts'
import type { Config } from '../../../trace/index.ts'
import { configNumber } from '../../inference/method.ts'
import * as d from '../../model/dists/index.ts'
import { defineModel, ops } from '../../model/index.ts'
import { type ExactMarginal, weightedMarginal } from '../../validation/index.ts'
import { defineAtom } from '../_define.ts'
import { tolerance } from '../_tolerance.ts'
import { once } from '../_util.ts'
import type { AtomParts, EpAtom } from '../atom-types.ts'

/** The priors of the three parameters and of the first state, which the model and the reference both read. */
const PRIOR = {
  level: { mean: -1, sd: 1 },
  persistence: { mean: 0.8, sd: 0.3 },
  precision: { shape: 3, rate: 0.3 },
  start: { mean: -1, sd: 1 },
}
const TRUE_LEVEL = -1

/** The chain and what it is read through, from a seed: the states, the observations and the scale the states set. */
function simulateChain(seed: number, n: number, persistence: number, precision: number) {
  const k = key(seed)
  const draws = Array.from(toFlat(normal(foldIn(k, 1), [2 * n])))
  const step = precision ** -0.5
  const h: number[] = []
  for (let t = 0; t < n; t++) {
    const previous = h[t - 1]
    const noise = draws[t] ?? 0
    h.push(
      previous === undefined
        ? PRIOR.start.mean + PRIOR.start.sd * noise
        : TRUE_LEVEL + persistence * (previous - TRUE_LEVEL) + step * noise,
    )
  }
  const y = h.map((v, t) => Math.exp(v / 2) * (draws[n + t] ?? 0))
  return { h, y }
}

const STATES = 80
const STATE_LO = -6.5
const STATE_HI = 4.5
const LOG_2PI = Math.log(2 * Math.PI)

/**
 * The marginals of the level, the persistence and the precision by a point-mass filter: for each triple on a grid the likelihood of
 * the observations integrates the chain's states out on an 80-point grid (the transition kernel is banded, six sds), and the
 * likelihood times the prior weighs the triple. Independent of EP, and of the grid method of the run: nothing in it reads the model.
 */
function chainMarginals(y: readonly number[]): Record<string, ExactMarginal> {
  const dh = (STATE_HI - STATE_LO) / (STATES - 1)
  const stateGrid = add(mul(arange(STATES), dh), STATE_LO)
  const states = toFlat(stateGrid)
  // exp(log density of each observation at each state): the same for every triple
  const read = y.map((v) =>
    toFlat(exp(sub(sub(-0.5 * LOG_2PI, mul(stateGrid, 0.5)), mul(exp(neg(stateGrid)), 0.5 * v * v)))),
  )
  const z0 = div(sub(stateGrid, PRIOR.start.mean), PRIOR.start.sd)
  const start = toFlat(div(mul(exp(mul(mul(z0, -0.5), z0)), dh), PRIOR.start.sd * Math.sqrt(2 * Math.PI)))
  const levels = toFlat(add(div(mul(add(arange(18), 0.5), 4.4), 18), -3.2))
  const persistences = toFlat(add(div(mul(add(arange(40), 0.5), 1.55), 40), -0.1))
  const logLo = Math.log(1.5)
  const logHi = Math.log(60)
  const logPrecisions = toFlat(add(div(mul(add(arange(22), 0.5), logHi - logLo), 22), logLo))
  const width = 2 * Math.ceil(6 / (dh * Math.sqrt(1.5))) + 1
  // The five buffers of the filter below are the working memory of its triple loop (15840 triples, each a sweep of a banded
  // kernel over every observation), rewritten in place; a tensor op per step would allocate in the innermost loop.
  // banned-ok: reference: the forward-backward pass of the atom written out in plain arithmetic
  const kernel = new Float64Array(STATES * width)
  const first = new Int32Array(STATES)
  const last = new Int32Array(STATES)
  const alpha = new Float64Array(STATES)
  const next = new Float64Array(STATES)
  const logLikelihood = (mu: number, phi: number, tau: number): number => {
    const sd = tau ** -0.5
    const scale = dh / (sd * Math.sqrt(2 * Math.PI))
    for (let i = 0; i < STATES; i++) {
      const centre = mu + phi * ((states[i] as number) - mu)
      const lo = Math.max(0, Math.ceil((centre - 6 * sd - STATE_LO) / dh))
      const hi = Math.min(STATES - 1, Math.floor((centre + 6 * sd - STATE_LO) / dh))
      first[i] = lo
      last[i] = hi
      for (let j = lo; j <= hi; j++) {
        const z = ((states[j] as number) - centre) / sd
        kernel[i * width + (j - lo)] = scale * Math.exp(-0.5 * z * z)
      }
    }
    let total = 0
    for (let t = 0; t < y.length; t++) {
      const likelihood = read[t] as number[]
      if (t === 0) next.set(start)
      else {
        // banned-ok: reference: the forward-backward pass of the atom written out in plain arithmetic
        next.fill(0)
        for (let i = 0; i < STATES; i++) {
          const a = alpha[i] as number
          if (a === 0) continue
          const lo = first[i] as number
          const row = i * width - lo
          for (let j = lo; j <= (last[i] as number); j++)
            next[j] = (next[j] as number) + a * (kernel[row + j] as number)
        }
      }
      let sum = 0
      for (let j = 0; j < STATES; j++) {
        const v = (next[j] as number) * (likelihood[j] as number)
        next[j] = v
        sum += v
      }
      total += Math.log(sum)
      for (let j = 0; j < STATES; j++) alpha[j] = (next[j] as number) / sum
    }
    return total
  }
  const logPrior = (mu: number, phi: number, tau: number): number =>
    -0.5 * ((mu - PRIOR.level.mean) / PRIOR.level.sd) ** 2 -
    0.5 * ((phi - PRIOR.persistence.mean) / PRIOR.persistence.sd) ** 2 +
    (PRIOR.precision.shape - 1) * Math.log(tau) -
    PRIOR.precision.rate * tau
  const logWeight = toFlat(zeros([levels.length * persistences.length * logPrecisions.length]))
  let top = Number.NEGATIVE_INFINITY
  let at = 0
  for (const mu of levels)
    for (const phi of persistences)
      for (const logTau of logPrecisions) {
        const tau = Math.exp(logTau)
        // the grid is even in log tau, so the density of log tau has the Jacobian tau
        const value = logLikelihood(mu, phi, tau) + logPrior(mu, phi, tau) + logTau
        logWeight[at++] = value
        top = Math.max(top, value)
      }
  const mass = [levels.length, persistences.length, logPrecisions.length].map((n) => toFlat(zeros([n])))
  at = 0
  for (let a = 0; a < levels.length; a++)
    for (let b = 0; b < persistences.length; b++)
      for (let c = 0; c < logPrecisions.length; c++) {
        const w = Math.exp((logWeight[at++] as number) - top)
        for (const [axis, index] of [
          [0, a],
          [1, b],
          [2, c],
        ] as const) {
          const m = mass[axis] as number[]
          m[index] = (m[index] as number) + w
        }
      }
  return {
    mu: weightedMarginal(levels, mass[0] as number[], 'linear'),
    phi: weightedMarginal(persistences, mass[1] as number[], 'linear'),
    tau: weightedMarginal(
      logPrecisions.map((l) => Math.exp(l)),
      mass[2] as number[],
      'linear',
    ),
  }
}

function chainPersistenceGridProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [truePersistence, truePrecision] = [configNumber(k, 'truePersistence'), configNumber(k, 'truePrecision')]
  const sim = simulateChain(seed, n, truePersistence, truePrecision)
  // region chain-persistence-grid
  const definition = defineModel('chain-persistence-grid', (m) => {
    const y = m.data('y', { dims: ['t'], role: 'response' })
    // a level, a persistence and a precision, each unknown
    const mu = m.latent('mu', d.normal(-1, 1))
    const phi = m.latent('phi', d.normal(0.8, 0.3))
    const tau = m.latent('tau', d.gamma(3, 0.3))
    // a chain of hidden states: each is Gaussian around the level plus the persistence times the last one's distance from it
    m.scan('h', {
      over: 't',
      init: d.normal(-1, 1),
      step: ({ prev }) => d.normal(prev.sub(mu).mul(phi).add(mu), tau.pow(-0.5)),
      // and each is read through a non-linear function: a Gaussian whose sd is exp(h / 2)
      emit: ({ state, observe }) => observe('y', d.normal(0, ops.exp(state.mul(0.5))), y),
    })
  })
  // endregion
  const reference = once(() => chainMarginals(sim.y))
  return {
    build: {
      definition,
      data: { y: sim.y },
      // the persistence multiplies a state: run EP inside a grid over it, a walk from the middle outwards
      hyperGrid: { over: ['phi'], knobs: { points: 21, order: 'walk', window: 7, warm: true, maxNodes: 40 } },
      quantities: [
        { name: 'mu', label: 'level mu', support: 'real', truth: TRUE_LEVEL },
        { name: 'phi', label: 'persistence phi', support: 'real', truth: truePersistence },
        { name: 'tau', label: 'step precision tau', support: 'positive', truth: truePrecision },
      ],
    },
    reference: {
      get exact() {
        return reference()
      },
      referenceSource:
        'a point-mass filter for the likelihood of each (mu, phi, tau) on an 18 by 40 by 22 grid, with the prior',
    },
    present: {
      equations: [
        `mu ~ N(${PRIOR.level.mean}, ${PRIOR.level.sd}²)   phi ~ N(${PRIOR.persistence.mean}, ${PRIOR.persistence.sd}²)   tau ~ Gamma(${PRIOR.precision.shape}, rate ${PRIOR.precision.rate})`,
        `h_1 ~ N(${PRIOR.start.mean}, ${PRIOR.start.sd}²),   h_t ~ N(mu + phi (h_(t-1) - mu), 1/tau)   (a Gaussian chain, one factor given phi)`,
        `y_t ~ N(0, exp(h_t / 2)²), t = 1..${n}   (read through exp of the state)`,
        `simulated with mu = ${TRUE_LEVEL}, phi = ${truePersistence}, tau = ${truePrecision}, seed ${seed}; observe y`,
      ],
      figure: {
        caption:
          'The observations and the scale exp(h_t / 2) the simulated chain gives them (the lines are plus and minus that scale: the truth, not what EP found).',
        xLabel: 't',
        yLabel: 'y',
        points: [{ name: 'y', x: sim.y.map((_, t) => t + 1), y: sim.y }],
        curves: [
          {
            name: '+exp(h/2) (simulated)',
            colour: 'tab2',
            x: sim.h.map((_, t) => t + 1),
            y: sim.h.map((v) => Math.exp(v / 2)),
          },
          {
            name: '-exp(h/2) (simulated)',
            colour: 'tab2',
            x: sim.h.map((_, t) => t + 1),
            y: sim.h.map((v) => -Math.exp(v / 2)),
          },
        ],
      },
      pair: ['phi', 'tau'],
    },
  }
}

/**
 * A chain of hidden Gaussian states with an unknown level, an unknown persistence and a gamma precision of its steps, each state read
 * through a non-linear function (a Gaussian whose sd is exp of half the state). The persistence multiplies the state, so a single EP
 * run gives it one narrow Gaussian; EP inside a grid over it (a walk from the middle outwards, each node started from its neighbour's
 * messages, the steps of the chain one `gaussian-chain` factor at each node) weighs the nodes by their EP evidence.
 */
const chainPersistenceGridAtom: EpAtom = defineAtom({
  name: 'chain-persistence-grid',
  title: 'Chain with a grid over its persistence',
  what: 'A Gaussian chain of hidden states h with an unknown level mu, a gamma precision tau of its steps and a persistence phi that multiplies the state, each state read through y ~ N(0, exp(h / 2)): EP inside a grid over phi, each node one gaussian-chain factor with mu and tau shared, the nodes weighted by their EP evidence.',
  factors: [
    'prior (gaussian) x2, prior (gamma)',
    'gaussian-chain (the steps, at each grid node)',
    'unary function (exp of half the state) with a Gaussian read',
    'the EP evidence as the weight of a node (ep-grid, a walk over one axis)',
  ],
  family: 'composed',
  reference: 'exact',
  tolerance: tolerance(
    'measured',
    "each node is exact in the chain, so what is left is the Gaussian projection of the read exp(h / 2), the weak prior the chain factor puts on a state whose cavity says nothing, the 21-point grid over phi and the reference's own coarse grid (18 by 40 by 22); over nine simulated sets (20 to 120 steps, phi 0.4 to 0.97) the means are within 0.1 reference sds, the spreads 0.95 to 1.10 of the reference's and the Kolmogorov distance up to 0.13",
    { meanZ: 0.2, sdRatio: [0.88, 1.15], ks: 0.18 },
  ),
  ep: { tol: 1e-4, damping: 0, maxSweeps: 60 },
  knobs: dict({
    seed: int(1, 9999, { default: 1, label: 'seed (the simulated data)' }),
    n: int(20, 120, { default: 50, label: 'length of the chain' }),
    truePersistence: real(0.3, 0.97, { default: 0.9, step: 0.01, label: 'true persistence' }),
    truePrecision: real(4, 30, { default: 11, step: 0.5, label: 'true precision of the steps' }),
  }),
  parts: chainPersistenceGridProblem,
})

/** The atoms of the `composed` file. */
export const composedChainAtoms: readonly EpAtom[] = [chainPersistenceGridAtom]
