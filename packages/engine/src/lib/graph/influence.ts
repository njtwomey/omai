/**
 * Influence maximisation: choose k seed nodes whose cascade is expected to reach the most of a network. The spread of
 * a seed set under the independent cascade or the linear threshold model equals the expected reach in a random "live
 * edge" graph (Kempe, Kleinberg and Tardos), so a fixed set of sampled live-edge graphs gives one spread function that
 * is monotone and exactly submodular: greedy selection then guarantees 1 - 1/e of the optimum, and CELF (lazy
 * greedy) returns the same seeds with far fewer evaluations. Here the sampling, the spread, and greedy, CELF and
 * highest-degree selection as an algorithm that adds one seed per step, with the marginal gains on the tape.
 */

import { MdmlError } from '../base/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { type Algorithm, type Contract, contractSchema, defineContract, message, scalar } from '../trace/index.ts'
import { checkNumber, codec } from './_run.ts'
import { type Graph, type NodeId, nodeIndex } from './graph.ts'

/** The two diffusion models whose spread is a reachability in a random graph: independent cascade and linear threshold. */
export type DiffusionModel = 'ic' | 'lt'

/** A live-edge graph in compressed sparse row form: the successors of node `u` are `indices[indptr[u] .. indptr[u + 1]]`. */
export interface LiveEdgeSample {
  readonly indptr: Int32Array
  readonly indices: Int32Array
}

/** Options of {@link liveEdgeSamples}. */
export interface LiveEdgeOptions {
  /** The activation probability of every edge, for the independent cascade (default 0.1). */
  readonly p?: number
  /** Use edge weights as activation probabilities (`ic`) or as influence (`lt`) instead of `p` and equal shares. */
  readonly weighted?: boolean
}

function csr(n: number, pairs: readonly (readonly [number, number])[]): LiveEdgeSample {
  const indptr = new Int32Array(n + 1)
  for (const [u] of pairs) indptr[u + 1] = (indptr[u + 1] ?? 0) + 1
  for (let u = 0; u < n; u++) indptr[u + 1] = (indptr[u + 1] ?? 0) + (indptr[u] ?? 0)
  const fill = Int32Array.from(indptr)
  const indices = new Int32Array(pairs.length)
  for (const [u, v] of pairs) {
    indices[fill[u] ?? 0] = v
    fill[u] = (fill[u] ?? 0) + 1
  }
  return { indptr, indices }
}

/**
 * Sample live-edge graphs. Under the independent cascade every edge is live independently with its activation
 * probability; under the linear threshold model every node keeps at most one of its in-edges, chosen with probability
 * equal to the edge's share of the node's in-weight (and none with the remaining probability, which is zero for the
 * default equal shares). The nodes a seed set reaches in a live-edge graph are distributed as the nodes it activates in
 * the cascade, so the mean reach over many samples estimates the expected spread.
 *
 * @param g - The network.
 * @param model - `'ic'` or `'lt'`.
 * @param key - The random key.
 * @param samples - How many live-edge graphs.
 * @param options - `p` and `weighted`.
 * @returns The samples.
 * @throws MdmlError with code `value` for a bad model, probability or sample count.
 * @concept influence-maximisation
 * @example
 * ```ts
 * import { liveEdgeSamples, pathGraph } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * liveEdgeSamples(pathGraph(4), 'ic', key(0), 2, { p: 1 })[0]?.indices.length // => 6
 * ```
 */
export function liveEdgeSamples(
  g: Graph,
  model: DiffusionModel,
  key: Key,
  samples: number,
  options: LiveEdgeOptions = {},
): LiveEdgeSample[] {
  const p = options.p ?? 0.1
  const weighted = options.weighted ?? false
  if (model !== 'ic' && model !== 'lt')
    throw new MdmlError('value', `liveEdgeSamples: unknown model '${String(model)}'`)
  checkNumber('liveEdgeSamples', 'p', p, 0, 1)
  if (!(Number.isInteger(samples) && samples >= 1))
    throw new MdmlError('value', 'liveEdgeSamples: samples must be a positive integer')
  const out: LiveEdgeSample[] = []
  for (let r = 0; r < samples; r++) {
    const pairs: [number, number][] = []
    if (model === 'ic') {
      const u = uniform(foldIn(key, r), [Math.max(g.out.indices.length, 1)]).data
      for (let src = 0; src < g.n; src++) {
        for (let e = g.out.indptr[src] ?? 0; e < (g.out.indptr[src + 1] ?? 0); e++) {
          if ((u[e] ?? 1) < (weighted ? (g.out.weights[e] ?? 0) : p)) pairs.push([src, g.out.indices[e] ?? 0])
        }
      }
    } else {
      const u = uniform(foldIn(key, r), [Math.max(g.n, 1)]).data
      for (let v = 0; v < g.n; v++) {
        const lo = g.in.indptr[v] ?? 0
        const hi = g.in.indptr[v + 1] ?? 0
        let total = 0
        for (let e = lo; e < hi; e++) total += weighted ? (g.in.weights[e] ?? 0) : 1
        let acc = 0
        for (let e = lo; e < hi; e++) {
          acc += (weighted ? (g.in.weights[e] ?? 0) : 1) / (total || 1)
          if ((u[v] ?? 1) < acc) {
            pairs.push([g.in.indices[e] ?? 0, v])
            break
          }
        }
      }
    }
    out.push(csr(g.n, pairs))
  }
  return out
}

/**
 * The nodes reached from a seed set in a live-edge graph.
 *
 * @param sample - A live-edge graph.
 * @param seeds - The seed nodes.
 * @returns The number of nodes reached, seeds included.
 * @concept influence-maximisation
 * @example
 * ```ts
 * import { liveEdgeReach } from 'mdml-engine/graph'
 * const chain = { indptr: Int32Array.of(0, 1, 2, 2), indices: Int32Array.of(1, 2) }
 * liveEdgeReach(chain, [0]) // => 3
 * liveEdgeReach(chain, [2]) // => 1
 * ```
 */
export function liveEdgeReach(sample: LiveEdgeSample, seeds: readonly number[]): number {
  const n = sample.indptr.length - 1
  const seen = new Uint8Array(n)
  const stack: number[] = []
  for (const s of seeds) {
    if (seen[s] === 0) {
      seen[s] = 1
      stack.push(s)
    }
  }
  let count = stack.length
  while (stack.length > 0) {
    const u = stack.pop() as number
    for (let e = sample.indptr[u] ?? 0; e < (sample.indptr[u + 1] ?? 0); e++) {
      const v = sample.indices[e] ?? 0
      if (seen[v] === 0) {
        seen[v] = 1
        count += 1
        stack.push(v)
      }
    }
  }
  return count
}

/**
 * The expected spread of a seed set, estimated as the mean reach over sampled live-edge graphs: the expected number of
 * nodes active when the cascade ends, seeds included.
 *
 * @param samples - The live-edge graphs ({@link liveEdgeSamples}).
 * @param seeds - The seed nodes.
 * @returns The mean reach.
 * @concept influence-maximisation
 * @example
 * ```ts
 * import { estimateSpread, liveEdgeSamples, pathGraph } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * estimateSpread(liveEdgeSamples(pathGraph(5), 'ic', key(0), 4, { p: 1 }), [0]) // => 5
 * ```
 */
export function estimateSpread(samples: readonly LiveEdgeSample[], seeds: readonly number[]): number {
  let total = 0
  for (const s of samples) total += liveEdgeReach(s, seeds)
  return total / samples.length
}

/**
 * The exact expected spread of a seed set under the independent cascade with one probability on every edge, by
 * enumerating all 2^m live-edge subgraphs of the m directed edges (an undirected edge counts as two). The truth the
 * Monte Carlo estimate converges to, for small graphs only.
 *
 * @param g - The network, with at most 22 directed edges.
 * @param seeds - The seed nodes.
 * @param p - The activation probability of each edge.
 * @returns The expected number of active nodes.
 * @throws MdmlError with code `value` for more than 22 directed edges.
 * @concept influence-maximisation
 * @example
 * ```ts
 * import { exactIndependentCascadeSpread, pathGraph } from 'mdml-engine/graph'
 * exactIndependentCascadeSpread(pathGraph(3), [0], 0.5) // => 1.75
 * ```
 */
export function exactIndependentCascadeSpread(g: Graph, seeds: readonly NodeId[], p: number): number {
  const m = g.out.indices.length
  if (m > 22)
    throw new MdmlError('value', `exactIndependentCascadeSpread: ${m} directed edges are too many to enumerate`)
  checkNumber('exactIndependentCascadeSpread', 'p', p, 0, 1)
  const sources = new Int32Array(m)
  for (let u = 0; u < g.n; u++) sources.fill(u, g.out.indptr[u] ?? 0, g.out.indptr[u + 1] ?? 0)
  const start = seeds.map((v) => nodeIndex(g, v))
  let expected = 0
  for (let mask = 0; mask < 2 ** m; mask++) {
    let weight = 1
    const pairs: [number, number][] = []
    for (let e = 0; e < m; e++) {
      if ((mask >> e) & 1) {
        weight *= p
        pairs.push([sources[e] ?? 0, g.out.indices[e] ?? 0])
      } else weight *= 1 - p
    }
    if (weight > 0) expected += weight * liveEdgeReach(csr(g.n, pairs), start)
  }
  return expected
}

/** How seeds are chosen by {@link influenceMaximisationAlgorithm}. */
export type InfluenceMethod = 'greedy' | 'celf' | 'degree'

/** Options of {@link influenceMaximisationAlgorithm}. */
export interface InfluenceOptions extends LiveEdgeOptions {
  /** The diffusion model (default `'ic'`). */
  readonly model?: DiffusionModel
  /** The selection rule (default `'celf'`): `'greedy'` re-evaluates every node each round, `'celf'` lazily, `'degree'` takes the highest out-degrees. */
  readonly method?: InfluenceMethod
  /** How many live-edge graphs estimate the spread (default 200). */
  readonly samples?: number
}

/** State of {@link influenceMaximisationAlgorithm}. */
export interface InfluenceState {
  /** The sampled live-edge graphs, fixed for the run. */
  readonly sampled: readonly LiveEdgeSample[]
  /** The seeds chosen so far, in order. */
  readonly seeds: readonly number[]
  /** The summed reach of the seeds over the samples. */
  readonly total: number
  /** The marginal gain of each chosen seed in expected nodes. */
  readonly gains: readonly number[]
  /** Spread evaluations so far (each a pass over every sample). */
  readonly evaluations: number
  /** CELF's upper bounds on each node's marginal gain, as summed reach, and the round in which each was computed. */
  readonly bounds: Float64Array
  readonly computed: Int32Array
}

/** The paths of {@link influenceMaximisationAlgorithm}. */
export type InfluencePath = 'seeding/seed' | 'seeding/gain' | 'seeding/spread' | 'seeding/evaluations'

/** The log contract of {@link influenceMaximisationAlgorithm}. */
export const influenceContract: Contract<InfluenceState, InfluencePath> = defineContract({
  'seeding/seed': message({ description: 'the seed chosen', payload: { node: 'the node', gain: 'its marginal gain' } }),
  'seeding/gain': scalar({
    description: 'the marginal gain of the seed just chosen, in expected nodes',
    direction: 'lower',
  }),
  'seeding/spread': scalar({ description: 'the expected spread of the seeds so far' }),
  'seeding/evaluations': scalar({
    description: 'spread evaluations so far (greedy needs about n per seed, CELF far fewer)',
  }),
})

/**
 * Influence maximisation, one seed per step. The spread of a set is estimated over a fixed collection of sampled
 * live-edge graphs ({@link liveEdgeSamples}), which makes it exactly monotone and submodular, so each seed's marginal
 * gain is no larger than the one before (diminishing returns) and greedy selection reaches at least 1 - 1/e of the best
 * possible spread. `greedy` evaluates every candidate each round; `celf` keeps upper bounds on the gains and
 * re-evaluates only the top candidate until it is current, returning the same seeds (ties go to the lowest index) with
 * far fewer evaluations; `degree` takes the highest out-degrees, the baseline both beat. The state serialises.
 *
 * @param g - The network.
 * @param k - How many seeds.
 * @param options - Model, activation probability, selection rule and number of samples.
 * @returns The algorithm.
 * @throws MdmlError with code `value` when k is not in [1, n] or the options are bad.
 * @concept influence-maximisation
 * @example
 * ```ts
 * import { influenceMaximisationAlgorithm, starGraph } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(influenceMaximisationAlgorithm(starGraph(6), 1, { p: 1, samples: 3 }), { seed: 0, maxSteps: 1 }).state.seeds // => [0]
 * ```
 */
export function influenceMaximisationAlgorithm(
  g: Graph,
  k: number,
  options: InfluenceOptions = {},
): Algorithm<InfluenceState, InfluencePath> {
  const model = options.model ?? 'ic'
  const method = options.method ?? 'celf'
  const samples = options.samples ?? 200
  if (!(Number.isInteger(k) && k >= 1 && k <= g.n))
    throw new MdmlError('value', `influenceMaximisationAlgorithm: k must be in [1, ${g.n}]`)
  if (method !== 'greedy' && method !== 'celf' && method !== 'degree')
    throw new MdmlError('value', `influenceMaximisationAlgorithm: unknown method '${String(method)}'`)
  const live: LiveEdgeOptions = {
    ...(options.p === undefined ? {} : { p: options.p }),
    ...(options.weighted === undefined ? {} : { weighted: options.weighted }),
  }
  const reach = (sampled: readonly LiveEdgeSample[], seeds: readonly number[]): number => {
    let total = 0
    for (const s of sampled) total += liveEdgeReach(s, seeds)
    return total
  }
  const outDegree = (v: number): number => (g.out.indptr[v + 1] ?? 0) - (g.out.indptr[v] ?? 0)
  return {
    name: 'graph/influence-maximisation',
    config: { model, method, samples, k, p: options.p ?? 0.1, weighted: options.weighted ?? false, nodes: g.n },
    schema: contractSchema(influenceContract),
    init: (key) => ({
      sampled: liveEdgeSamples(g, model, key, samples, live),
      seeds: [],
      total: 0,
      gains: [],
      evaluations: 0,
      bounds: new Float64Array(g.n).fill(Number.POSITIVE_INFINITY),
      computed: new Int32Array(g.n).fill(-1),
    }),
    step: (s) => {
      const round = s.seeds.length
      const chosen = new Set(s.seeds)
      const bounds = Float64Array.from(s.bounds)
      const computed = Int32Array.from(s.computed)
      let evaluations = s.evaluations
      const gainOf = (v: number): number => {
        evaluations += 1
        return reach(s.sampled, [...s.seeds, v]) - s.total
      }
      let pick = -1
      let gain = 0
      if (method === 'degree') {
        for (let v = 0; v < g.n; v++) if (!chosen.has(v) && (pick < 0 || outDegree(v) > outDegree(pick))) pick = v
        gain = gainOf(pick)
      } else if (method === 'greedy') {
        let best = Number.NEGATIVE_INFINITY
        for (let v = 0; v < g.n; v++) {
          if (chosen.has(v)) continue
          const gv = gainOf(v)
          if (gv > best) {
            best = gv
            pick = v
          }
        }
        gain = best
      } else {
        for (;;) {
          let top = -1
          for (let v = 0; v < g.n; v++) {
            if (chosen.has(v)) continue
            if (top < 0 || (bounds[v] ?? 0) > (bounds[top] ?? 0)) top = v
          }
          if (computed[top] === round) {
            pick = top
            gain = bounds[top] ?? 0
            break
          }
          bounds[top] = gainOf(top)
          computed[top] = round
        }
      }
      const total = s.total + gain
      return {
        ...s,
        seeds: [...s.seeds, pick],
        total,
        gains: [...s.gains, gain / s.sampled.length],
        evaluations,
        bounds,
        computed,
      }
    },
    done: (s) => (s.seeds.length >= k ? 'completed' : undefined),
    observe: (s) => {
      const node = s.seeds[s.seeds.length - 1]
      return {
        metrics: {
          'seeding/gain': s.gains[s.gains.length - 1] ?? 0,
          'seeding/spread': s.total / s.sampled.length,
          'seeding/evaluations': s.evaluations,
        },
        ...(node === undefined
          ? {}
          : { messages: [{ kind: 'seeding/seed', payload: { node, gain: s.gains[s.gains.length - 1] ?? 0 } }] }),
      }
    },
    ...codec<InfluenceState>(),
  }
}
