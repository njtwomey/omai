/**
 * Things that spread over a network: an epidemic (SIR, SIS or SEIR) simulated exactly in continuous time by Gillespie's
 * method, one event per step, and the two classical cascades of adoption, the independent cascade (every newly active
 * node gets one chance to activate each neighbour) and the threshold cascade (a node adopts once enough of its
 * neighbours have), one wave per step. Beside the simulations stand the results known about them: the final size of an
 * SIR epidemic from the degree distribution and the transmissibility, the epidemic thresholds of a graph, and the
 * condition for Watts's global cascades.
 */

import { OmaiError } from '../base/index.ts'
import { eigsh } from '../numerics/sparse/index.ts'
import { foldIn, type Key, subsample, uniform, uniformStream } from '../random/index.ts'
import { type Algorithm, type Contract, contractSchema, defineContract, event, scalar, tensor } from '../trace/index.ts'
import { checkNumber, codec, resolveSeeds } from './_run.ts'
import { type Graph, type NodeId, nodeIndex } from './graph.ts'
import { adjacencyCsr } from './matrices.ts'

/** The compartments of a network epidemic: SIR (recovery is immunity), SIS (recovery is susceptibility again), SEIR (an exposed stage first). */
export type EpidemicModel = 'SIR' | 'SIS' | 'SEIR'

/** Options of {@link networkEpidemicAlgorithm}. */
export interface NetworkEpidemicOptions {
  /** The compartments (default `'SIR'`). */
  readonly model?: EpidemicModel
  /** The infection rate across each edge from an infectious to a susceptible node, per unit time (default 1). */
  readonly beta?: number
  /** The recovery rate (default 1): an infectious period is exponential with mean 1 / gamma. */
  readonly gamma?: number
  /** The rate at which exposed nodes become infectious, for SEIR (default 1). */
  readonly sigma?: number
  /** The infectious nodes at the start: labels, or a count to draw at random (default 1). */
  readonly seeds?: readonly NodeId[] | number
  /** Nodes immune from the start, vaccinated: they can neither catch nor pass the disease. */
  readonly immune?: readonly NodeId[]
  /** Stop at this time (default: run until no event can happen). */
  readonly horizon?: number
}

/** State of {@link networkEpidemicAlgorithm}. */
export interface NetworkEpidemicState {
  /** 0 susceptible, 1 exposed, 2 infectious, 3 recovered (or immune). */
  readonly status: Uint8Array
  /** For each node, how many of its in-neighbours are infectious. */
  readonly pressure: Int32Array
  /** The number of nodes in each state: susceptible, exposed, infectious, recovered. */
  readonly counts: readonly number[]
  /** The time of the last event. */
  readonly t: number
  /** Events so far. */
  readonly events: number
  /** Nodes that have ever left the susceptible state (seeds included). */
  readonly ever: number
  /** The most nodes infectious at once, and when. */
  readonly peak: number
  readonly peakTime: number
  /** True when nothing more can happen or the horizon was reached. */
  readonly finished: boolean
}

/** The paths of {@link networkEpidemicAlgorithm}. */
export type NetworkEpidemicPath =
  | 'epidemic/t'
  | 'epidemic/counts'
  | 'epidemic/states'
  | 'epidemic/infected'
  | 'epidemic/attackRate'
  | 'epidemic/peak'
  | 'epidemic/over'

/** The log contract of {@link networkEpidemicAlgorithm}. */
export const networkEpidemicContract: Contract<NetworkEpidemicState, NetworkEpidemicPath> = defineContract({
  'epidemic/t': scalar({ description: 'the time of the last event', unit: 'time' }),
  'epidemic/counts': tensor({
    role: 'state',
    shape: [4],
    description: 'nodes susceptible, exposed, infectious and recovered',
  }),
  'epidemic/states': tensor({ role: 'state', shape: ['nodes'], description: '0 S, 1 E, 2 I, 3 R per node' }),
  'epidemic/infected': scalar({ description: 'the share of nodes infectious now' }),
  'epidemic/attackRate': scalar({ description: 'the share of nodes that have ever been infected' }),
  'epidemic/peak': scalar({ description: 'the largest share infectious so far' }),
  'epidemic/over': event({ context: { t: 'the time at which no event could happen any more' } }),
})

const SUSCEPTIBLE = 0
const EXPOSED = 1
const INFECTIOUS = 2
const RECOVERED = 3

/**
 * An epidemic on a contact network by Gillespie's exact stochastic simulation, one event per step. Every susceptible
 * node is infected at rate `beta` times its number of infectious in-neighbours, every exposed node becomes infectious
 * at rate `sigma`, every infectious node recovers at rate `gamma`; the time to the next event is exponential in the
 * total rate and the event falls on a node in proportion to its rate. The run ends when no event is possible
 * (an SIR or SEIR epidemic burns out, an SIS one dies out) or at the horizon. The infection is counted per edge, so
 * edge weights are ignored; a directed graph passes the disease along its edges' direction. The state serialises.
 *
 * Known results to test it against: an infectious period is exponential, so an edge transmits with probability
 * `beta / (beta + gamma)` ({@link sirTransmissibility}), and the share finally infected follows from that and the
 * degree distribution ({@link epidemicFinalSize}).
 *
 * @param g - The contact network.
 * @param options - Model, rates, seeds, vaccinated nodes and horizon.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a negative rate or an unknown model.
 * @concept network-epidemics
 * @example
 * ```ts
 * import { networkEpidemicAlgorithm, completeGraph } from 'omai-engine/graph'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(networkEpidemicAlgorithm(completeGraph(5), { gamma: 1, beta: 0 }), { seed: 0, maxSteps: 50 })
 * [run.state.counts, run.status] // => [[4, 0, 0, 1], 'converged']
 * ```
 */
export function networkEpidemicAlgorithm(
  g: Graph,
  options: NetworkEpidemicOptions = {},
): Algorithm<NetworkEpidemicState, NetworkEpidemicPath> {
  const model = options.model ?? 'SIR'
  const beta = options.beta ?? 1
  const gamma = options.gamma ?? 1
  const sigma = options.sigma ?? 1
  const horizon = options.horizon ?? Number.POSITIVE_INFINITY
  if (model !== 'SIR' && model !== 'SIS' && model !== 'SEIR')
    throw new OmaiError('value', `networkEpidemicAlgorithm: unknown model '${String(model)}'`)
  checkNumber('networkEpidemicAlgorithm', 'beta', beta, 0)
  checkNumber('networkEpidemicAlgorithm', 'gamma', gamma, 0)
  checkNumber('networkEpidemicAlgorithm', 'sigma', sigma, 0)
  checkNumber('networkEpidemicAlgorithm', 'horizon', horizon, 0)
  const n = g.n
  const immune = new Set((options.immune ?? []).map((v) => nodeIndex(g, v)))
  const rate = (status: number, pressure: number): number =>
    status === SUSCEPTIBLE ? beta * pressure : status === EXPOSED ? sigma : status === INFECTIOUS ? gamma : 0
  const touch = (pressure: Int32Array, v: number, by: 1 | -1): void => {
    for (let k = g.out.indptr[v] ?? 0; k < (g.out.indptr[v + 1] ?? 0); k++) {
      const w = g.out.indices[k] ?? 0
      pressure[w] = (pressure[w] ?? 0) + by
    }
  }
  return {
    name: 'graph/network-epidemic',
    config: { model, beta, gamma, sigma, nodes: n, horizon: Number.isFinite(horizon) ? horizon : null },
    schema: contractSchema(networkEpidemicContract),
    init: (key) => {
      const seeds = resolveSeeds('networkEpidemicAlgorithm', g, options.seeds ?? 1, key, immune)
      const status = new Uint8Array(n)
      for (const v of immune) status[v] = RECOVERED
      const pressure = new Int32Array(n)
      for (const v of seeds) {
        if (immune.has(v)) throw new OmaiError('value', `networkEpidemicAlgorithm: seed ${v} is immune`)
        status[v] = INFECTIOUS
        touch(pressure, v, 1)
      }
      const counts = [0, 0, 0, 0]
      for (let v = 0; v < n; v++) counts[status[v] ?? 0] = (counts[status[v] ?? 0] ?? 0) + 1
      return {
        status,
        pressure,
        counts,
        t: 0,
        events: 0,
        ever: seeds.length,
        peak: seeds.length,
        peakTime: 0,
        finished: false,
      }
    },
    step: (s, ctx) => {
      const status = Uint8Array.from(s.status)
      const pressure = Int32Array.from(s.pressure)
      let total = 0
      for (let v = 0; v < n; v++) total += rate(status[v] ?? 0, pressure[v] ?? 0)
      if (!(total > 0)) {
        ctx.emit('epidemic/over', { t: s.t })
        return { ...s, finished: true }
      }
      const next = uniformStream(ctx.key, 4)
      const t = s.t + -Math.log(1 - next()) / total
      if (t > horizon) return { ...s, t: horizon, finished: true }
      // The event falls on a node in proportion to its rate.
      const target = next() * total
      let acc = 0
      let v = 0
      for (; v < n; v++) {
        acc += rate(status[v] ?? 0, pressure[v] ?? 0)
        if (acc > target) break
      }
      v = Math.min(v, n - 1)
      while (rate(status[v] ?? 0, pressure[v] ?? 0) === 0 && v > 0) v--
      const counts = [...s.counts]
      const from = status[v] ?? 0
      let to: number
      let ever = s.ever
      if (from === SUSCEPTIBLE) {
        to = model === 'SEIR' ? EXPOSED : INFECTIOUS
        ever += 1
      } else if (from === EXPOSED) to = INFECTIOUS
      else to = model === 'SIS' ? SUSCEPTIBLE : RECOVERED
      status[v] = to
      counts[from] = (counts[from] ?? 0) - 1
      counts[to] = (counts[to] ?? 0) + 1
      if (to === INFECTIOUS) touch(pressure, v, 1)
      if (from === INFECTIOUS) touch(pressure, v, -1)
      const infectious = counts[INFECTIOUS] ?? 0
      const better = infectious > s.peak
      return {
        status,
        pressure,
        counts,
        t,
        events: s.events + 1,
        ever,
        peak: better ? infectious : s.peak,
        peakTime: better ? t : s.peakTime,
        finished: false,
      }
    },
    done: (s) => (s.finished ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        'epidemic/t': s.t,
        'epidemic/infected': (s.counts[INFECTIOUS] ?? 0) / Math.max(n, 1),
        'epidemic/attackRate': s.ever / Math.max(n, 1),
        'epidemic/peak': s.peak / Math.max(n, 1),
      },
      tensors: {
        'epidemic/counts': { role: 'state', value: s.counts, shape: [4] },
        'epidemic/states': { role: 'state', value: s.status, shape: [n] },
      },
    }),
    ...codec<NetworkEpidemicState>(),
  }
}

/**
 * The probability that an infectious node passes the disease along an edge before it recovers, when infection across
 * the edge happens at rate beta and the infectious period is exponential with rate gamma: beta / (beta + gamma). An SIR
 * epidemic is bond percolation on the contact network with this occupation probability.
 *
 * @param beta - The infection rate across an edge.
 * @param gamma - The recovery rate.
 * @returns The transmissibility, in [0, 1].
 * @throws OmaiError with code `value` when both rates are zero or either is negative.
 * @concept network-epidemics
 * @example
 * ```ts
 * import { sirTransmissibility } from 'omai-engine/graph'
 * sirTransmissibility(1, 3) // => 0.25
 * ```
 */
export function sirTransmissibility(beta: number, gamma: number): number {
  if (!(beta >= 0 && gamma >= 0 && beta + gamma > 0))
    throw new OmaiError('value', 'sirTransmissibility: need non-negative rates, not both zero')
  return beta / (beta + gamma)
}

/** The final size of an SIR epidemic on a random network, from {@link epidemicFinalSize}. */
export interface EpidemicFinalSize {
  /** The share of the network finally infected by a large outbreak. */
  readonly finalSize: number
  /** The probability that following an edge reaches a node that is not infected (the root of the generating-function equation). */
  readonly u: number
  /** The transmissibility used. */
  readonly transmissibility: number
  /** The basic reproduction number T (<k^2> - <k>) / <k>: a large outbreak is possible when it exceeds 1. */
  readonly r0: number
  /** The critical transmissibility <k> / (<k^2> - <k>). */
  readonly threshold: number
}

/**
 * The final size of an SIR epidemic on a configuration-model network with a given degree sequence and transmissibility
 * T (Newman 2002): with G0 and G1 the generating functions of the degree and of the excess degree, the probability u
 * that an edge does not lead to an infected node is the smallest root in [0, 1] of u = 1 - T + T G1(u), and the share
 * infected is 1 - G0(u). On an Erdos-Renyi graph with mean degree c this is the classical relation
 * z = 1 - exp(-R0 z) with R0 = cT; a heavy-tailed degree law raises R0 and lowers the threshold, vanishing in the limit.
 * The result is the size of a large outbreak (seeds of a vanishing share); outbreaks that die out early are not
 * included.
 *
 * @param degrees - The degree of each node.
 * @param transmissibility - T in [0, 1], for example {@link sirTransmissibility}.
 * @returns The final size, the fixed point, R0 and the threshold.
 * @throws OmaiError with code `value` for an empty sequence or a transmissibility outside [0, 1].
 * @concept network-epidemics
 * @example
 * ```ts
 * import { epidemicFinalSize } from 'omai-engine/graph'
 * epidemicFinalSize([3, 3, 3, 3], 0).finalSize // => 0
 * epidemicFinalSize([3, 3, 3, 3], 1).finalSize // => 1
 * ```
 */
export function epidemicFinalSize(degrees: ArrayLike<number>, transmissibility: number): EpidemicFinalSize {
  const k = Array.from(degrees)
  if (k.length === 0) throw new OmaiError('value', 'epidemicFinalSize: no degrees')
  if (!(transmissibility >= 0 && transmissibility <= 1))
    throw new OmaiError('value', 'epidemicFinalSize: transmissibility must be in [0, 1]')
  const n = k.length
  const mean = k.reduce((a, d) => a + d, 0) / n
  const second = k.reduce((a, d) => a + d * d, 0) / n
  const g0 = (x: number): number => k.reduce((a, d) => a + x ** d, 0) / n
  const g1 = (x: number): number =>
    mean === 0 ? 1 : k.reduce((a, d) => a + (d === 0 ? 0 : d * x ** (d - 1)), 0) / (n * mean)
  // Iterating from 0 climbs monotonically to the smallest fixed point.
  let u = 0
  for (let i = 0; i < 200000; i++) {
    const nu = 1 - transmissibility + transmissibility * g1(u)
    if (Math.abs(nu - u) < 1e-15) {
      u = nu
      break
    }
    u = nu
  }
  const r0 = mean === 0 ? 0 : (transmissibility * (second - mean)) / mean
  return {
    finalSize: 1 - g0(u),
    u,
    transmissibility,
    r0,
    threshold: second - mean > 0 ? mean / (second - mean) : Number.POSITIVE_INFINITY,
  }
}

/**
 * The share finally infected in the homogeneous-mixing (mean-field) SIR epidemic, the positive root z of
 * z = 1 - exp(-R0 z), zero when R0 <= 1. It is also the giant-component share of an Erdos-Renyi graph with mean degree
 * R0 ({@link erdosRenyiGiantFraction}).
 *
 * @param r0 - The basic reproduction number.
 * @returns The final size.
 * @throws OmaiError with code `value` for a negative R0.
 * @concept network-epidemics
 * @example
 * ```ts
 * import { homogeneousFinalSize } from 'omai-engine/graph'
 * Math.round(homogeneousFinalSize(2) * 10000) / 10000 // => 0.7968
 * homogeneousFinalSize(0.9) // => 0
 * ```
 */
export function homogeneousFinalSize(r0: number): number {
  if (!(r0 >= 0)) throw new OmaiError('value', 'homogeneousFinalSize: R0 must be non-negative')
  if (r0 <= 1) return 0
  let z = 1
  for (let i = 0; i < 100000; i++) {
    const nz = 1 - Math.exp(-r0 * z)
    if (Math.abs(nz - z) < 1e-15) return nz
    z = nz
  }
  return z
}

/** The thresholds of a network for the spread of a disease. */
export interface EpidemicThresholds {
  /** The largest eigenvalue of the adjacency matrix. */
  readonly lambdaMax: number
  /** 1 / lambdaMax: the SIS threshold beta / gamma of the quenched mean field. */
  readonly spectral: number
  /** <k> / <k^2>: the SIS threshold of the heterogeneous mean field. */
  readonly heterogeneous: number
  /** <k> / (<k^2> - <k>): the critical transmissibility of SIR, as bond percolation. */
  readonly percolation: number
}

/**
 * The epidemic thresholds of a network: the spectral one (SIS dies out below beta / gamma = 1 / lambda_max of the
 * adjacency matrix), the degree-based mean-field one <k> / <k^2>, and the SIR transmissibility threshold
 * <k> / (<k^2> - <k>) of percolation. Hubs raise lambda_max and lower all three.
 *
 * @param g - The network.
 * @returns The four numbers.
 * @concept network-epidemics
 * @example
 * ```ts
 * import { completeGraph, epidemicThresholds } from 'omai-engine/graph'
 * const t = epidemicThresholds(completeGraph(5))
 * [Math.round(t.lambdaMax * 1e6) / 1e6, t.spectral, t.percolation] // => [4, 0.25, 0.3333333333333333]
 * ```
 */
export function epidemicThresholds(g: Graph): EpidemicThresholds {
  const n = g.n
  const d = Array.from({ length: n }, (_, v) => (g.out.indptr[v + 1] ?? 0) - (g.out.indptr[v] ?? 0))
  const mean = d.reduce((a, x) => a + x, 0) / Math.max(n, 1)
  const second = d.reduce((a, x) => a + x * x, 0) / Math.max(n, 1)
  let lambdaMax = 0
  if (g.edges.length > 0) {
    // eigsh needs k < n; a two-node graph has lambda_max equal to its edge weight.
    lambdaMax = n > 2 ? (eigsh(adjacencyCsr(g, { weighted: false }), 1, { which: 'LA' }).values.data[0] ?? 0) : 1
  }
  return {
    lambdaMax,
    spectral: lambdaMax > 0 ? 1 / lambdaMax : Number.POSITIVE_INFINITY,
    heterogeneous: second > 0 ? mean / second : Number.POSITIVE_INFINITY,
    percolation: second - mean > 0 ? mean / (second - mean) : Number.POSITIVE_INFINITY,
  }
}

/** Whom to vaccinate: at random, the highest-degree nodes (targeted), or a random neighbour of a random node (the acquaintance immunisation of Cohen, Havlin and ben-Avraham). */
export type VaccinationStrategy = 'random' | 'degree' | 'acquaintance'

/**
 * Choose `count` nodes to make immune ({@link NetworkEpidemicOptions.immune}). Targeting the hubs protects a heavy-tailed
 * network with far fewer doses than vaccinating at random, because the hubs carry most of the edges; acquaintance
 * immunisation gets much of that benefit without knowing the degrees, since a random neighbour of a random node is
 * likely to be a hub.
 *
 * @param g - The contact network.
 * @param count - How many nodes.
 * @param strategy - `'random'`, `'degree'` (highest degree first, ties by index) or `'acquaintance'`.
 * @param key - The random key (unused by `'degree'`).
 * @returns The nodes, ascending.
 * @throws OmaiError with code `value` for a count beyond the number of nodes or an unknown strategy.
 * @concept network-epidemics
 * @example
 * ```ts
 * import { starGraph, vaccinationSet } from 'omai-engine/graph'
 * import { key } from 'omai-engine/random'
 * vaccinationSet(starGraph(6), 1, 'degree', key(0)) // => [0]
 * ```
 */
export function vaccinationSet(g: Graph, count: number, strategy: VaccinationStrategy, key: Key): number[] {
  if (!(Number.isInteger(count) && count >= 0 && count <= g.n))
    throw new OmaiError('value', `vaccinationSet: count must be an integer in [0, ${g.n}]`)
  const degree = (v: number): number => (g.out.indptr[v + 1] ?? 0) - (g.out.indptr[v] ?? 0)
  if (strategy === 'random') return Array.from(subsample(key, g.n, count).data).sort((a, b) => a - b)
  if (strategy === 'degree')
    return Array.from({ length: g.n }, (_, v) => v)
      .sort((a, b) => degree(b) - degree(a) || a - b)
      .slice(0, count)
      .sort((a, b) => a - b)
  if (strategy !== 'acquaintance')
    throw new OmaiError('value', `vaccinationSet: unknown strategy '${String(strategy)}'`)
  const chosen = new Set<number>()
  const next = uniformStream(key, 64)
  for (let guard = 0; chosen.size < count && guard < 200 * Math.max(count, 1); guard++) {
    const v = Math.min(g.n - 1, Math.floor(next() * g.n))
    const lo = g.out.indptr[v] ?? 0
    const d = degree(v)
    if (d > 0) chosen.add(g.out.indices[lo + Math.min(d - 1, Math.floor(next() * d))] ?? 0)
  }
  // A graph with too few nodes of positive degree is topped up with the highest-degree nodes left.
  for (const v of vaccinationSet(g, g.n, 'degree', key).sort((a, b) => degree(b) - degree(a) || a - b)) {
    if (chosen.size >= count) break
    chosen.add(v)
  }
  return [...chosen].sort((a, b) => a - b)
}

/**
 * The herd-immunity threshold of the homogeneous-mixing epidemic: vaccinating the share 1 - 1/R0 of the population at
 * random leaves an effective reproduction number of one, so the epidemic cannot take off.
 *
 * @param r0 - The basic reproduction number.
 * @returns The share to immunise, in [0, 1).
 * @throws OmaiError with code `value` for a negative R0.
 * @concept network-epidemics
 * @example
 * ```ts
 * import { herdImmunityThreshold } from 'omai-engine/graph'
 * herdImmunityThreshold(4) // => 0.75
 * ```
 */
export function herdImmunityThreshold(r0: number): number {
  if (!(r0 >= 0)) throw new OmaiError('value', 'herdImmunityThreshold: R0 must be non-negative')
  return r0 <= 1 ? 0 : 1 - 1 / r0
}

/** Options shared by the cascade algorithms. */
export interface CascadeOptions {
  /** Use edge weights (the activation probability of the independent cascade; the influence of the threshold cascade) instead of treating every edge alike. */
  readonly weighted?: boolean
}

/** State of the cascade algorithms. */
export interface CascadeState {
  /** 0 inactive, 1 activated in the last wave, 2 activated earlier. */
  readonly status: Uint8Array
  /** The waves so far. */
  readonly wave: number
  /** Nodes active. */
  readonly adopted: number
  /** Nodes activated by the last wave. */
  readonly fresh: number
  /** The size of each wave, the seeds first. */
  readonly sizes: readonly number[]
  /** Per node, the weight of active in-neighbours (threshold cascades) or unused zeros. */
  readonly pressure: Float64Array
  /** Per node, its threshold (threshold cascades) or unused zeros. */
  readonly thresholds: Float64Array
}

/** The paths of the cascade algorithms. */
export type CascadePath = 'cascade/wave' | 'cascade/adopted' | 'cascade/new' | 'cascade/states' | 'cascade/over'

/** The log contract of the cascade algorithms. */
export const cascadeContract: Contract<CascadeState, CascadePath> = defineContract({
  'cascade/wave': scalar({ description: 'the wave just finished (0 is the seeds)' }),
  'cascade/adopted': scalar({ description: 'nodes active now' }),
  'cascade/new': scalar({ description: 'nodes activated by the wave' }),
  'cascade/states': tensor({ role: 'state', shape: ['nodes'], description: '0 inactive, 1 newly active, 2 active' }),
  'cascade/over': event({ context: { wave: 'the wave after which nothing more activated' } }),
})

function cascadeStart(g: Graph, seeds: readonly number[]): CascadeState {
  const status = new Uint8Array(g.n)
  for (const v of seeds) status[v] = 1
  return {
    status,
    wave: 0,
    adopted: seeds.length,
    fresh: seeds.length,
    sizes: [seeds.length],
    pressure: new Float64Array(g.n),
    thresholds: new Float64Array(g.n),
  }
}

function cascadeObserve(s: CascadeState): {
  metrics: Record<string, number>
  tensors: Record<string, { role: 'state'; value: Uint8Array; shape: number[] }>
} {
  return {
    metrics: { 'cascade/wave': s.wave, 'cascade/adopted': s.adopted, 'cascade/new': s.fresh },
    tensors: { 'cascade/states': { role: 'state', value: s.status, shape: [s.status.length] } },
  }
}

/**
 * The independent cascade model of diffusion, one wave per step: every node activated in the last wave gets a single
 * chance to activate each inactive out-neighbour, succeeding independently with probability `p` (or the edge's weight).
 * The cascade ends when a wave activates no one. The expected size of the cascade from a seed set is the quantity
 * influence maximisation maximises ({@link influenceMaximisationAlgorithm}). The state serialises.
 *
 * @param g - The network.
 * @param seeds - The initially active nodes: labels, or a count to draw at random.
 * @param p - The activation probability (default 0.1); with `weighted` the edge weights are used instead.
 * @param options - `weighted`.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a probability outside [0, 1].
 * @concept information-cascades
 * @example
 * ```ts
 * import { independentCascadeAlgorithm, pathGraph } from 'omai-engine/graph'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(independentCascadeAlgorithm(pathGraph(5), [0], 1), { seed: 0, maxSteps: 10 })
 * [run.state.adopted, run.steps] // => [5, 5]
 * ```
 */
export function independentCascadeAlgorithm(
  g: Graph,
  seeds: readonly NodeId[] | number,
  p = 0.1,
  options: CascadeOptions = {},
): Algorithm<CascadeState, CascadePath> {
  checkNumber('independentCascadeAlgorithm', 'p', p, 0, 1)
  const weighted = options.weighted ?? false
  return {
    name: 'graph/independent-cascade',
    config: { p, weighted, nodes: g.n },
    schema: contractSchema(cascadeContract),
    init: (key) => cascadeStart(g, resolveSeeds('independentCascadeAlgorithm', g, seeds, key)),
    step: (s, ctx) => {
      const frontier: number[] = []
      for (let v = 0; v < g.n; v++) if (s.status[v] === 1) frontier.push(v)
      let attempts = 0
      for (const u of frontier) attempts += (g.out.indptr[u + 1] ?? 0) - (g.out.indptr[u] ?? 0)
      const draw = uniform(ctx.key, [Math.max(attempts, 1)]).data
      const status = Uint8Array.from(s.status)
      for (let v = 0; v < g.n; v++) if (status[v] === 1) status[v] = 2
      let k = 0
      let fresh = 0
      for (const u of frontier) {
        for (let e = g.out.indptr[u] ?? 0; e < (g.out.indptr[u + 1] ?? 0); e++) {
          const v = g.out.indices[e] ?? 0
          const chance = weighted ? (g.out.weights[e] ?? 0) : p
          if (status[v] === 0 && (draw[k] ?? 1) < chance) {
            status[v] = 3
            fresh += 1
          }
          k += 1
        }
      }
      for (let v = 0; v < g.n; v++) if (status[v] === 3) status[v] = 1
      const wave = s.wave + 1
      if (fresh === 0) ctx.emit('cascade/over', { wave })
      return {
        ...s,
        status,
        wave,
        adopted: s.adopted + fresh,
        fresh,
        sizes: [...s.sizes, fresh],
      }
    },
    done: (s) => (s.wave > 0 && s.fresh === 0 ? 'converged' : undefined),
    observe: cascadeObserve,
    ...codec<CascadeState>(),
  }
}

/** Options of {@link linearThresholdAlgorithm}. */
export interface LinearThresholdOptions extends CascadeOptions {
  /**
   * The threshold of every node: a number in [0, 1] (Watts's model: a node adopts when at least this share of its
   * in-neighbours have), or `'random'` for thresholds drawn uniformly from [0, 1] (Kempe, Kleinberg and Tardos; default).
   */
  readonly threshold?: number | 'random'
}

/**
 * The linear threshold model of diffusion, one synchronous wave per step: each node has a threshold, and each active
 * in-neighbour contributes its edge's share of the node's total in-weight (1 / in-degree when unweighted); an inactive
 * node with at least one active in-neighbour becomes active when the contributions reach its threshold. The cascade
 * ends when a wave activates no one. With a fixed threshold this is Watts's model of global cascades
 * ({@link wattsCascadeCondition}). The state serialises.
 *
 * @param g - The network.
 * @param seeds - The initially active nodes: labels, or a count to draw at random.
 * @param options - `threshold` (a number or `'random'`, default `'random'`) and `weighted`.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a threshold outside [0, 1].
 * @concept information-cascades
 * @example
 * ```ts
 * import { linearThresholdAlgorithm, pathGraph } from 'omai-engine/graph'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // On a path every inner node has two neighbours: one active neighbour is half its influence.
 * runAlgorithm(linearThresholdAlgorithm(pathGraph(6), [0], { threshold: 0.5 }), { seed: 0, maxSteps: 20 }).state.adopted // => 6
 * ```
 */
export function linearThresholdAlgorithm(
  g: Graph,
  seeds: readonly NodeId[] | number,
  options: LinearThresholdOptions = {},
): Algorithm<CascadeState, CascadePath> {
  const threshold = options.threshold ?? 'random'
  const weighted = options.weighted ?? false
  if (threshold !== 'random') checkNumber('linearThresholdAlgorithm', 'threshold', threshold, 0, 1)
  // The influence of an in-neighbour on a node is its edge's share of the node's total in-weight.
  const inTotal = new Float64Array(g.n)
  for (let v = 0; v < g.n; v++) {
    for (let e = g.in.indptr[v] ?? 0; e < (g.in.indptr[v + 1] ?? 0); e++)
      inTotal[v] = (inTotal[v] ?? 0) + (weighted ? (g.in.weights[e] ?? 0) : 1)
  }
  return {
    name: 'graph/linear-threshold',
    config: { threshold, weighted, nodes: g.n },
    schema: contractSchema(cascadeContract),
    init: (key) => {
      const start = cascadeStart(g, resolveSeeds('linearThresholdAlgorithm', g, seeds, foldIn(key, 1)))
      const thresholds =
        threshold === 'random'
          ? Float64Array.from(uniform(foldIn(key, 2), [Math.max(g.n, 1)]).data.subarray(0, g.n))
          : new Float64Array(g.n).fill(threshold)
      return { ...start, thresholds }
    },
    step: (s, ctx) => {
      const status = Uint8Array.from(s.status)
      const pressure = Float64Array.from(s.pressure)
      const touched = new Set<number>()
      for (let u = 0; u < g.n; u++) {
        if (s.status[u] !== 1) continue
        status[u] = 2
        for (let e = g.out.indptr[u] ?? 0; e < (g.out.indptr[u + 1] ?? 0); e++) {
          const v = g.out.indices[e] ?? 0
          pressure[v] = (pressure[v] ?? 0) + (weighted ? (g.out.weights[e] ?? 0) : 1)
          touched.add(v)
        }
      }
      let fresh = 0
      const joined: number[] = []
      for (const v of touched) {
        if (status[v] !== 0) continue
        const share = (pressure[v] ?? 0) / (inTotal[v] ?? 1)
        if (share > 0 && share >= (s.thresholds[v] ?? 1) - 1e-12) joined.push(v)
      }
      for (const v of joined) {
        status[v] = 1
        fresh += 1
      }
      const wave = s.wave + 1
      if (fresh === 0) ctx.emit('cascade/over', { wave })
      return { ...s, status, pressure, wave, adopted: s.adopted + fresh, fresh, sizes: [...s.sizes, fresh] }
    },
    done: (s) => (s.wave > 0 && s.fresh === 0 ? 'converged' : undefined),
    observe: cascadeObserve,
    ...codec<CascadeState>(),
  }
}

/**
 * Watts's condition for a global cascade in the threshold model (Watts 2002): with every node adopting when a share
 * phi of its neighbours has, a node of degree k can be tipped by a single neighbour only if k <= 1/phi (it is
 * "vulnerable"). A single seed can start a cascade across the network when the vulnerable nodes percolate, that is when
 * the sum over k <= 1/phi of k (k - 1) p_k exceeds the mean degree <k>. The value returned is that sum divided by
 * <k>: above 1 the network is inside the cascade window. Too sparse a network does not connect the vulnerable nodes;
 * too dense a one has no vulnerable nodes.
 *
 * @param degrees - The degree of each node.
 * @param threshold - The common threshold phi in (0, 1].
 * @returns The ratio; a global cascade is possible above 1.
 * @throws OmaiError with code `value` for a threshold outside (0, 1] or an empty sequence.
 * @concept information-cascades
 * @example
 * ```ts
 * import { wattsCascadeCondition } from 'omai-engine/graph'
 * wattsCascadeCondition([3, 3, 3, 3], 0.2) // => 2
 * wattsCascadeCondition([3, 3, 3, 3], 0.5) // => 0
 * ```
 */
export function wattsCascadeCondition(degrees: ArrayLike<number>, threshold: number): number {
  const k = Array.from(degrees)
  if (k.length === 0) throw new OmaiError('value', 'wattsCascadeCondition: no degrees')
  if (!(threshold > 0 && threshold <= 1))
    throw new OmaiError('value', 'wattsCascadeCondition: threshold must be in (0, 1]')
  const mean = k.reduce((a, d) => a + d, 0) / k.length
  if (mean === 0) return 0
  const limit = Math.floor(1 / threshold + 1e-12)
  return k.reduce((a, d) => a + (d <= limit ? d * (d - 1) : 0), 0) / k.length / mean
}
