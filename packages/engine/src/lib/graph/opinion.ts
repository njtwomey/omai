/**
 * Opinion dynamics on a network: the voter model (a node copies a random neighbour, fixation probability known
 * exactly), DeGroot averaging and its Friedkin-Johnsen extension with stubborn agents (consensus and equilibrium in
 * closed form), and bounded confidence (only opinions within a tolerance are listened to; consensus splits into
 * clusters). Each is an algorithm with an observable for the disagreement.
 */

import { MdmlError } from '../base/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { permutation, uniformStream } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { type Algorithm, type Contract, contractSchema, defineContract, event, scalar, tensor } from '../trace/index.ts'
import { checkNumber, codec } from './_run.ts'
import type { Graph } from './graph.ts'

function degreesOf(g: Graph): Float64Array {
  const d = new Float64Array(g.n)
  for (let v = 0; v < g.n; v++) d[v] = (g.out.indptr[v + 1] ?? 0) - (g.out.indptr[v] ?? 0)
  return d
}

/** State of {@link voterModelAlgorithm}. */
export interface VoterState {
  /** The opinion of each node, a small integer. */
  readonly opinions: Int32Array
  /** Single-node updates so far. */
  readonly updates: number
  /** True once every node holds the same opinion. */
  readonly consensus: boolean
}

/** The paths of {@link voterModelAlgorithm}. */
export type VoterPath = 'voter/share' | 'voter/interface' | 'voter/updates' | 'voter/opinions' | 'voter/consensus'

/** The log contract of {@link voterModelAlgorithm}. */
export const voterContract: Contract<VoterState, VoterPath> = defineContract({
  'voter/share': scalar({ description: 'the share of nodes holding opinion 1' }),
  'voter/interface': scalar({ description: 'the share of edges whose ends disagree', direction: 'lower' }),
  'voter/updates': scalar({ description: 'single-node updates so far' }),
  'voter/opinions': tensor({ role: 'state', shape: ['nodes'], description: 'the opinion of each node' }),
  'voter/consensus': event({ context: { updates: 'the updates it took to reach consensus' } }),
})

/** Options of {@link voterModelAlgorithm}. */
export interface VoterOptions {
  /**
   * The start: a share of nodes holding opinion 1, the others 0 (exactly round(share n) nodes, chosen at random), or
   * one integer opinion per node.
   */
  readonly initial?: number | ArrayLike<number>
  /** Single-node updates per step (default 1). */
  readonly updatesPerStep?: number
}

/**
 * The voter model: at each update a node chosen uniformly at random copies the opinion of a neighbour chosen uniformly
 * at random. On a connected graph it ends in consensus; the probability that opinion 1 wins is the degree-weighted
 * share of its initial supporters ({@link voterFixationProbability}), because that share is the martingale. The
 * interface density (the share of edges that disagree) decays as consensus nears. A node with no neighbour keeps its
 * opinion. Steps are `updatesPerStep` single updates (default 1); the state serialises.
 *
 * @param g - The network.
 * @param options - The initial opinions and the updates per step.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a share outside [0, 1] or opinions of the wrong length.
 * @concept voter-model
 * @example
 * ```ts
 * import { completeGraph, voterModelAlgorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(voterModelAlgorithm(completeGraph(6), { initial: 0.5 }), { seed: 0, maxSteps: 10000 })
 * [run.status, new Set(run.state.opinions).size] // => ['converged', 1]
 * ```
 */
export function voterModelAlgorithm(g: Graph, options: VoterOptions = {}): Algorithm<VoterState, VoterPath> {
  const initial = options.initial ?? 0.5
  const per = options.updatesPerStep ?? 1
  if (!(Number.isInteger(per) && per >= 1))
    throw new MdmlError('value', 'voterModelAlgorithm: updatesPerStep must be a positive integer')
  if (typeof initial === 'number') checkNumber('voterModelAlgorithm', 'initial share', initial, 0, 1)
  else if (initial.length !== g.n) throw new MdmlError('shape', 'voterModelAlgorithm: one initial opinion per node')
  const agree = (o: Int32Array): boolean => o.every((x) => x === o[0])
  return {
    name: 'graph/voter-model',
    config: { nodes: g.n, updatesPerStep: per, initial: typeof initial === 'number' ? initial : 'explicit' },
    schema: contractSchema(voterContract),
    init: (key) => {
      const opinions = new Int32Array(g.n)
      if (typeof initial === 'number') {
        const order = permutation(key, g.n).data
        for (let i = 0; i < Math.round(initial * g.n); i++) opinions[order[i] ?? 0] = 1
      } else for (let v = 0; v < g.n; v++) opinions[v] = Math.round(initial[v] ?? 0)
      return { opinions, updates: 0, consensus: agree(opinions) }
    },
    step: (s, ctx) => {
      const opinions = Int32Array.from(s.opinions)
      const next = uniformStream(ctx.key, 64)
      for (let i = 0; i < per; i++) {
        const v = Math.min(g.n - 1, Math.floor(next() * g.n))
        const lo = g.out.indptr[v] ?? 0
        const deg = (g.out.indptr[v + 1] ?? 0) - lo
        if (deg === 0) {
          next()
          continue
        }
        opinions[v] = opinions[g.out.indices[lo + Math.min(deg - 1, Math.floor(next() * deg))] ?? 0] ?? 0
      }
      const consensus = agree(opinions)
      const updates = s.updates + per
      if (consensus && !s.consensus) ctx.emit('voter/consensus', { updates })
      return { opinions, updates, consensus }
    },
    done: (s) => (s.consensus ? 'converged' : undefined),
    observe: (s) => {
      let ones = 0
      for (const x of s.opinions) if (x === 1) ones += 1
      let split = 0
      for (const e of g.edges) if (s.opinions[e.source] !== s.opinions[e.target]) split += 1
      return {
        metrics: {
          'voter/share': ones / Math.max(g.n, 1),
          'voter/interface': split / Math.max(g.edges.length, 1),
          'voter/updates': s.updates,
        },
        tensors: { 'voter/opinions': { role: 'state', value: s.opinions, shape: [g.n] } },
      }
    },
    ...codec<VoterState>(),
  }
}

/**
 * The probability that opinion 1 takes over in the voter model on a connected undirected graph: the initial supporters'
 * share of the total degree, sum(d_i x_i) / sum(d_i), since a node's chance of being copied is proportional to its
 * degree. On a regular graph it is the plain share.
 *
 * @param g - A connected undirected graph.
 * @param initial - The initial opinion of each node (0 or 1).
 * @returns The fixation probability of opinion 1.
 * @throws MdmlError with code `value` for a directed graph or one without edges, `shape` for the wrong length.
 * @concept voter-model
 * @example
 * ```ts
 * import { starGraph, voterFixationProbability } from 'mdml-engine/graph'
 * voterFixationProbability(starGraph(4), [1, 0, 0, 0, 0]) // => 0.5
 * ```
 */
export function voterFixationProbability(g: Graph, initial: ArrayLike<number>): number {
  if (g.directed) throw new MdmlError('value', 'voterFixationProbability: undirected graphs only')
  if (initial.length !== g.n) throw new MdmlError('shape', 'voterFixationProbability: one opinion per node')
  const d = degreesOf(g)
  const total = d.reduce((a, x) => a + x, 0)
  if (total === 0) throw new MdmlError('value', 'voterFixationProbability: the graph has no edges')
  let mass = 0
  for (let v = 0; v < g.n; v++) if (initial[v] === 1) mass += d[v] ?? 0
  return mass / total
}

/** State of the averaging algorithms ({@link degrootAlgorithm}, {@link boundedConfidenceAlgorithm}). */
export interface OpinionState {
  /** The opinions now. */
  readonly x: Float64Array
  /** The opinions at the start (the stubborn agents' anchor). */
  readonly x0: Float64Array
  /** The largest change of any opinion in the last round. */
  readonly change: number
  /** Rounds so far. */
  readonly round: number
}

/** The paths of the averaging algorithms. */
export type OpinionPath =
  | 'opinion/values'
  | 'opinion/mean'
  | 'opinion/spread'
  | 'opinion/disagreement'
  | 'opinion/change'
  | 'opinion/settled'

/** The log contract of the averaging algorithms. */
export const opinionContract: Contract<OpinionState, OpinionPath> = defineContract({
  'opinion/values': tensor({ role: 'state', shape: ['nodes'], description: 'the opinion of each node' }),
  'opinion/mean': scalar({ description: 'the mean opinion' }),
  'opinion/spread': scalar({ description: 'the largest opinion minus the smallest', direction: 'lower' }),
  'opinion/disagreement': scalar({ description: 'the mean squared difference across the edges', direction: 'lower' }),
  'opinion/change': scalar({ description: 'the largest change of any opinion in the round', direction: 'lower' }),
  'opinion/settled': event({ context: { round: 'the round in which the opinions stopped changing' } }),
})

function opinionObserve(
  g: Graph,
  s: OpinionState,
): {
  metrics: Record<string, number>
  tensors: Record<string, { role: 'state'; value: Float64Array; shape: number[] }>
} {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  let mean = 0
  for (const v of s.x) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
    mean += v
  }
  let gap = 0
  for (const e of g.edges) gap += ((s.x[e.source] ?? 0) - (s.x[e.target] ?? 0)) ** 2
  return {
    metrics: {
      'opinion/mean': mean / Math.max(s.x.length, 1),
      'opinion/spread': s.x.length === 0 ? 0 : hi - lo,
      'opinion/disagreement': gap / Math.max(g.edges.length, 1),
      'opinion/change': s.change,
    },
    tensors: { 'opinion/values': { role: 'state', value: s.x, shape: [s.x.length] } },
  }
}

/** Options of {@link degrootAlgorithm}. */
export interface DeGrootOptions {
  /**
   * Friedkin-Johnsen stubbornness: the weight a node keeps on its own initial opinion, a number for all nodes or one
   * per node (default 0, plain DeGroot).
   */
  readonly stubbornness?: number | ArrayLike<number>
  /** The weight a node gives its own current opinion among its neighbours' (default 0); a positive value lets averaging settle on a bipartite graph. */
  readonly selfWeight?: number
  /** Use edge weights as trust (default: equal trust in every neighbour). */
  readonly weighted?: boolean
  /** Stop when no opinion changes by more than this (default 1e-10). */
  readonly tolerance?: number
}

function averagingWeights(g: Graph, weighted: boolean, selfWeight: number): (v: number) => [number, number][] {
  return (v) => {
    const lo = g.out.indptr[v] ?? 0
    const hi = g.out.indptr[v + 1] ?? 0
    let total = 0
    for (let e = lo; e < hi; e++) total += weighted ? (g.out.weights[e] ?? 0) : 1
    if (total === 0) return [[v, 1]]
    const row: [number, number][] = []
    for (let e = lo; e < hi; e++)
      row.push([g.out.indices[e] ?? 0, ((weighted ? (g.out.weights[e] ?? 0) : 1) / total) * (1 - selfWeight)])
    if (selfWeight > 0) row.push([v, selfWeight])
    return row
  }
}

/**
 * DeGroot learning and its Friedkin-Johnsen extension, one synchronous round per step: every node replaces its opinion
 * by the average of its neighbours' (its trust matrix W is the row-normalised adjacency), and a stubborn node
 * (stubbornness lambda_i) keeps weight lambda_i on its initial opinion: x(t + 1) = Lambda x(0) + (I - Lambda) W x(t).
 * Without stubbornness the opinions of a connected, non-bipartite graph converge to the degree-weighted mean of the
 * initial ones ({@link degrootConsensus}); with it they converge to the equilibrium
 * ({@link friedkinJohnsenEquilibrium}) and persistent disagreement. The run stops when the largest change falls below
 * the tolerance. The state serialises.
 *
 * @param g - The network.
 * @param initial - The opinion of each node at the start.
 * @param options - Stubbornness, self weight, trust weights and tolerance.
 * @returns The algorithm.
 * @throws MdmlError with code `shape` for the wrong number of opinions or stubbornness values, `value` outside [0, 1].
 * @concept opinion-dynamics
 * @example
 * ```ts
 * import { degrootAlgorithm, pathGraph } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(degrootAlgorithm(pathGraph(3), [0, 0, 3], { selfWeight: 0.5 }), { seed: 0, maxSteps: 400 })
 * Array.from(run.state.x, (v) => Math.round(v * 1e6) / 1e6) // => [0.75, 0.75, 0.75]
 * ```
 */
export function degrootAlgorithm(
  g: Graph,
  initial: ArrayLike<number>,
  options: DeGrootOptions = {},
): Algorithm<OpinionState, OpinionPath> {
  const stub = stubbornnessOf(g, initial, options.stubbornness)
  const selfWeight = options.selfWeight ?? 0
  checkNumber('degrootAlgorithm', 'selfWeight', selfWeight, 0, 1)
  const tolerance = options.tolerance ?? 1e-10
  const row = averagingWeights(g, options.weighted ?? false, selfWeight)
  return {
    name: 'graph/degroot',
    config: { nodes: g.n, selfWeight, weighted: options.weighted ?? false, tolerance },
    schema: contractSchema(opinionContract),
    init: () => ({
      x: Float64Array.from(initial),
      x0: Float64Array.from(initial),
      change: Number.POSITIVE_INFINITY,
      round: 0,
    }),
    step: (s, ctx) => {
      const x = new Float64Array(g.n)
      let change = 0
      for (let v = 0; v < g.n; v++) {
        let avg = 0
        for (const [u, w] of row(v)) avg += w * (s.x[u] ?? 0)
        x[v] = (stub[v] ?? 0) * (s.x0[v] ?? 0) + (1 - (stub[v] ?? 0)) * avg
        change = Math.max(change, Math.abs((x[v] ?? 0) - (s.x[v] ?? 0)))
      }
      if (change <= tolerance) ctx.emit('opinion/settled', { round: s.round + 1 })
      return { ...s, x, change, round: s.round + 1 }
    },
    done: (s) => (s.change <= tolerance ? 'converged' : undefined),
    observe: (s) => opinionObserve(g, s),
    ...codec<OpinionState>(),
  }
}

function stubbornnessOf(
  g: Graph,
  initial: ArrayLike<number>,
  stubbornness: number | ArrayLike<number> | undefined,
): Float64Array {
  if (initial.length !== g.n) throw new MdmlError('shape', 'degrootAlgorithm: one initial opinion per node')
  const raw = stubbornness ?? 0
  const out = new Float64Array(g.n)
  if (typeof raw === 'number') out.fill(raw)
  else if (raw.length !== g.n) throw new MdmlError('shape', 'degrootAlgorithm: one stubbornness value per node')
  else for (let v = 0; v < g.n; v++) out[v] = raw[v] ?? 0
  for (const x of out) checkNumber('degrootAlgorithm', 'stubbornness', x, 0, 1)
  return out
}

/**
 * The consensus of plain DeGroot averaging on a connected undirected graph with a positive self weight (or a
 * non-bipartite graph): the degree-weighted mean of the initial opinions, sum(d_i x_i) / sum(d_i), because the degrees
 * are the stationary weights of the averaging matrix.
 *
 * @param g - A connected undirected graph.
 * @param initial - The initial opinions.
 * @returns The consensus value.
 * @throws MdmlError with code `value` for a directed graph or one without edges, `shape` for the wrong length.
 * @concept opinion-dynamics
 * @example
 * ```ts
 * import { degrootConsensus, starGraph } from 'mdml-engine/graph'
 * degrootConsensus(starGraph(4), [4, 0, 0, 0, 0]) // => 2
 * ```
 */
export function degrootConsensus(g: Graph, initial: ArrayLike<number>): number {
  if (g.directed) throw new MdmlError('value', 'degrootConsensus: undirected graphs only')
  if (initial.length !== g.n) throw new MdmlError('shape', 'degrootConsensus: one opinion per node')
  const d = degreesOf(g)
  const total = d.reduce((a, x) => a + x, 0)
  if (total === 0) throw new MdmlError('value', 'degrootConsensus: the graph has no edges')
  let mass = 0
  for (let v = 0; v < g.n; v++) mass += (d[v] ?? 0) * (initial[v] ?? 0)
  return mass / total
}

/**
 * The equilibrium of Friedkin-Johnsen opinion dynamics, the fixed point of x = Lambda x0 + (I - Lambda) W x, solved
 * as the linear system (I - (I - Lambda) W) x = Lambda x0. It exists whenever some node is stubborn in every
 * connected component.
 *
 * @param g - The network.
 * @param initial - The initial (anchor) opinions.
 * @param stubbornness - The weight on the own initial opinion, one number or one per node.
 * @param options - `selfWeight` and `weighted`, as for {@link degrootAlgorithm}.
 * @returns The equilibrium opinion of each node.
 * @throws MdmlError with code `shape` for wrong lengths, `value` outside [0, 1].
 * @concept opinion-dynamics
 * @example
 * ```ts
 * import { friedkinJohnsenEquilibrium, pathGraph } from 'mdml-engine/graph'
 * Array.from(friedkinJohnsenEquilibrium(pathGraph(2), [0, 1], [0.5, 0.5]), (v) => Math.round(v * 1e6) / 1e6) // => [0.333333, 0.666667]
 * ```
 */
export function friedkinJohnsenEquilibrium(
  g: Graph,
  initial: ArrayLike<number>,
  stubbornness: number | ArrayLike<number>,
  options: Pick<DeGrootOptions, 'selfWeight' | 'weighted'> = {},
): Float64Array {
  const stub = stubbornnessOf(g, initial, stubbornness)
  const row = averagingWeights(g, options.weighted ?? false, options.selfWeight ?? 0)
  const a = new Float64Array(g.n * g.n)
  const b = new Float64Array(g.n)
  for (let v = 0; v < g.n; v++) {
    a[v * g.n + v] = 1
    for (const [u, w] of row(v)) a[v * g.n + u] = (a[v * g.n + u] ?? 0) - (1 - (stub[v] ?? 0)) * w
    b[v] = (stub[v] ?? 0) * (initial[v] ?? 0)
  }
  return Float64Array.from(
    toFlat(
      solve(
        Array.from({ length: g.n }, (_, i) => Array.from(a.subarray(i * g.n, (i + 1) * g.n))),
        Array.from(b),
      ),
    ),
  )
}

/**
 * Bounded-confidence opinion dynamics (Hegselmann and Krause), one synchronous round per step: every node replaces its
 * opinion by the mean of its own and those neighbours' opinions that lie within `confidence` of its own. With a
 * large tolerance the network reaches consensus; with a small one it freezes into several clusters of opinion, roughly
 * the range divided by twice the tolerance on a complete graph.
 *
 * @param g - The network (a complete graph gives the original model).
 * @param initial - The initial opinions.
 * @param confidence - The tolerance: the largest opinion difference a node still listens to.
 * @param tolerance - Stop when no opinion changes by more than this (default 1e-10).
 * @returns The algorithm.
 * @throws MdmlError with code `shape` for the wrong number of opinions, `value` for a negative confidence.
 * @concept opinion-dynamics
 * @example
 * ```ts
 * import { boundedConfidenceAlgorithm, completeGraph } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(boundedConfidenceAlgorithm(completeGraph(3), [0, 0.5, 1], 1), { seed: 0, maxSteps: 5 })
 * Array.from(run.state.x) // => [0.5, 0.5, 0.5]
 * ```
 */
export function boundedConfidenceAlgorithm(
  g: Graph,
  initial: ArrayLike<number>,
  confidence: number,
  tolerance = 1e-10,
): Algorithm<OpinionState, OpinionPath> {
  if (initial.length !== g.n) throw new MdmlError('shape', 'boundedConfidenceAlgorithm: one initial opinion per node')
  checkNumber('boundedConfidenceAlgorithm', 'confidence', confidence, 0)
  return {
    name: 'graph/bounded-confidence',
    config: { nodes: g.n, confidence, tolerance },
    schema: contractSchema(opinionContract),
    init: () => ({
      x: Float64Array.from(initial),
      x0: Float64Array.from(initial),
      change: Number.POSITIVE_INFINITY,
      round: 0,
    }),
    step: (s, ctx) => {
      const x = new Float64Array(g.n)
      let change = 0
      for (let v = 0; v < g.n; v++) {
        let sum = s.x[v] ?? 0
        let count = 1
        for (let e = g.out.indptr[v] ?? 0; e < (g.out.indptr[v + 1] ?? 0); e++) {
          const o = s.x[g.out.indices[e] ?? 0] ?? 0
          if (Math.abs(o - (s.x[v] ?? 0)) <= confidence) {
            sum += o
            count += 1
          }
        }
        x[v] = sum / count
        change = Math.max(change, Math.abs((x[v] ?? 0) - (s.x[v] ?? 0)))
      }
      if (change <= tolerance) ctx.emit('opinion/settled', { round: s.round + 1 })
      return { ...s, x, change, round: s.round + 1 }
    },
    done: (s) => (s.change <= tolerance ? 'converged' : undefined),
    observe: (s) => opinionObserve(g, s),
    ...codec<OpinionState>(),
  }
}
