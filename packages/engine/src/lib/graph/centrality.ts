/**
 * Centrality measures, which score how important each node is to a graph's structure, as in NetworkX: degree and
 * closeness, betweenness by Brandes' algorithm (and edge betweenness), and the spectral scores eigenvector centrality,
 * Katz centrality, PageRank and HITS hubs and authorities, the iterative ones also available as traced power
 * iterations.
 */

import { type Json, kahanSum, maxOf, OmaiError } from '../base/index.ts'
import { dasum, daxpy, dnrm2 } from '../numerics/linalg/index.ts'
import { type Key, subsample } from '../random/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  runAlgorithm,
  scalar,
  tensor,
} from '../trace/index.ts'
import { codec } from './_run.ts'
import { degree, type Graph, type GraphRunOptions, inDegree, type NodeId, nodeIndex, outDegree } from './graph.ts'
import { IndexedQueue } from './structures.ts'

/**
 * Degree centrality (NetworkX degree_centrality and its in- and out-degree variants): the degree divided by n - 1.
 *
 * @param g - The graph.
 * @param options - `direction`: total degree (`'both'`, default), `'in'` or `'out'`.
 * @returns The centrality of each node.
 * @concept centrality
 * @example
 * ```ts
 * import { degreeCentrality, starGraph } from 'omai-engine/graph'
 * Array.from(degreeCentrality(starGraph(3))) // => [1, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
 * ```
 */
export function degreeCentrality(g: Graph, options: { readonly direction?: 'both' | 'in' | 'out' } = {}): Float64Array {
  const d = options.direction === 'in' ? inDegree(g) : options.direction === 'out' ? outDegree(g) : degree(g)
  const s = g.n > 1 ? 1 / (g.n - 1) : 1
  return Float64Array.from(d, (x) => x * s)
}

/** Distances (hops, or weights) from a source along out- or in-edges, with the BFS/Dijkstra settle order, path counts
 * and predecessors: the single-source stage of Brandes' algorithm. */
function singleSource(
  g: Graph,
  s: number,
  weighted: boolean,
  incoming: boolean,
): { order: number[]; sigma: Float64Array; preds: number[][]; dist: Float64Array; predEdges: number[][] } {
  const a = incoming ? g.in : g.out
  const dist = new Float64Array(g.n).fill(Number.POSITIVE_INFINITY)
  const sigma = new Float64Array(g.n)
  const preds: number[][] = Array.from({ length: g.n }, () => [])
  const predEdges: number[][] = Array.from({ length: g.n }, () => [])
  const order: number[] = []
  dist[s] = 0
  sigma[s] = 1
  if (!weighted) {
    const queue = [s]
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head] ?? 0
      order.push(u)
      for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
        const v = a.indices[k] ?? 0
        if (v === u) continue
        if (dist[v] === Number.POSITIVE_INFINITY) {
          dist[v] = (dist[u] ?? 0) + 1
          queue.push(v)
        }
        if (dist[v] === (dist[u] ?? 0) + 1) {
          sigma[v] = (sigma[v] ?? 0) + (sigma[u] ?? 0)
          preds[v]?.push(u)
          predEdges[v]?.push(a.edges[k] ?? 0)
        }
      }
    }
    return { order, sigma, preds, dist, predEdges }
  }
  const done = new Uint8Array(g.n)
  const queue = new IndexedQueue(g.n)
  queue.push(s, 0)
  for (let u = queue.pop(); u !== undefined; u = queue.pop()) {
    done[u] = 1
    order.push(u)
    for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
      const v = a.indices[k] ?? 0
      if (done[v] === 1 || v === u) continue
      const w = a.weights[k] ?? 0
      if (w < 0) throw new OmaiError('domain', 'centrality: negative weights are not allowed')
      const alt = (dist[u] ?? 0) + w
      const dv = dist[v] ?? Number.POSITIVE_INFINITY
      if (alt < dv) {
        dist[v] = alt
        sigma[v] = sigma[u] ?? 0
        preds[v] = [u]
        predEdges[v] = [a.edges[k] ?? 0]
        queue.push(v, alt)
      } else if (alt === dv) {
        sigma[v] = (sigma[v] ?? 0) + (sigma[u] ?? 0)
        preds[v]?.push(u)
        predEdges[v]?.push(a.edges[k] ?? 0)
      }
    }
  }
  return { order, sigma, preds, dist, predEdges }
}

/**
 * Closeness centrality (NetworkX closeness_centrality): the reciprocal of the mean distance to a node from the nodes
 * that reach it, scaled by the fraction of nodes that do (the Wasserman-Faust improvement, on by default). For a
 * directed graph distances run into the node, as in NetworkX.
 *
 * @param g - The graph.
 * @param options - `weighted` (default false: hop counts) and `wfImproved` (default true).
 * @returns The centrality of each node.
 * @concept centrality
 * @example
 * ```ts
 * import { closenessCentrality, pathGraph } from 'omai-engine/graph'
 * Array.from(closenessCentrality(pathGraph(3))) // => [0.6666666666666666, 1, 0.6666666666666666]
 * ```
 */
export function closenessCentrality(
  g: Graph,
  options: { readonly weighted?: boolean; readonly wfImproved?: boolean } = {},
): Float64Array {
  const weighted = options.weighted ?? false
  const improved = options.wfImproved ?? true
  return Float64Array.from({ length: g.n }, (_, u) => {
    const { dist } = singleSource(g, u, weighted, g.directed)
    const finite = Array.from(dist).filter((d) => d !== Number.POSITIVE_INFINITY)
    const total = kahanSum(finite)
    const reached = finite.length
    if (total <= 0 || g.n <= 1) return 0
    const c = (reached - 1) / total
    return improved ? c * ((reached - 1) / (g.n - 1)) : c
  })
}

/** Options of {@link betweennessCentrality}. */
export interface BetweennessOptions {
  /** Divide by the number of pairs (default true), as NetworkX. */
  readonly normalized?: boolean
  /** Use edge weights as lengths (default false). */
  readonly weighted?: boolean
  /** Count the path ends too (default false). */
  readonly endpoints?: boolean
}

/** The paths the betweenness algorithms log, as a type. */
export type BetweennessPath = 'between/source' | 'between/score' | 'between/delta'

/**
 * What the betweenness algorithms log, by path: the source searched at this step, the scores so far (each node's, or
 * each edge's, scaled as the final result and extrapolated to all sources, so they approach the answer) and the
 * dependencies the source's shortest paths put on each node.
 */
export const betweennessContract: Contract<unknown, BetweennessPath> = defineContract({
  'between/source': scalar({ description: 'the source node whose shortest paths were searched at this step' }),
  'between/score': tensor({
    role: 'state',
    shape: ['scores'],
    description: 'the betweenness of each node (or edge) so far, extrapolated from the sources searched',
  }),
  'between/delta': tensor({
    role: 'state',
    shape: ['nodes'],
    description: "the dependency of the step's source on each node: the share of its shortest paths through it",
  }),
})

/** Options of {@link betweennessAlgorithm} and {@link edgeBetweennessAlgorithm}: sampling the sources. */
export interface BetweennessSampling {
  /** Search only from these nodes and extrapolate by n over their number (default: every node). */
  readonly sources?: readonly NodeId[]
  /** Search from this many nodes drawn at random with the `key` (instead of `sources`). */
  readonly samples?: number
  /** The key for `samples`. */
  readonly key?: Key
}

/** The state of the betweenness algorithms. */
export interface BetweennessState {
  /** The unscaled sum of dependencies over the sources searched, per node (or per edge). */
  readonly raw: Float64Array
  /** The scores so far, scaled as the final result and extrapolated to all sources. */
  readonly score: Float64Array
  /** The dependencies of the last source on each node. */
  readonly delta: Float64Array
  /** The sources to search, in order. */
  readonly sources: readonly number[]
  /** How many have been searched. */
  readonly cursor: number
  /** The last source searched (-1 before the first). */
  readonly source: number
}

function brandesSources(g: Graph, sampling: BetweennessSampling): number[] {
  if (sampling.sources !== undefined) return [...new Set(sampling.sources.map((v) => nodeIndex(g, v)))]
  if (sampling.samples === undefined) return Array.from({ length: g.n }, (_, v) => v)
  if (sampling.key === undefined) throw new OmaiError('value', 'betweenness: sampling sources needs a key')
  if (!(Number.isInteger(sampling.samples) && sampling.samples >= 1 && sampling.samples <= g.n))
    throw new OmaiError('value', `betweenness: cannot draw ${sampling.samples} sources from ${g.n} nodes`)
  return Array.from(subsample(sampling.key, g.n, sampling.samples).data).sort((a, b) => a - b)
}

/** The one-source stage of Brandes' algorithm for nodes (`edges` false) or edges, as a steppable algorithm. */
function brandes(
  name: string,
  g: Graph,
  edges: boolean,
  options: BetweennessOptions & BetweennessSampling,
): Algorithm<BetweennessState, BetweennessPath> {
  const normalized = options.normalized ?? true
  const endpoints = !edges && (options.endpoints ?? false)
  const weighted = options.weighted ?? false
  const sources = brandesSources(g, options)
  const n = g.n
  const size = edges ? g.edges.length : n
  let scale: number | undefined
  if (edges) scale = normalized ? (n <= 1 ? undefined : 1 / (n * (n - 1))) : g.directed ? undefined : 0.5
  else if (normalized) {
    if (endpoints) scale = n < 2 ? undefined : 1 / (n * (n - 1))
    else scale = n <= 2 ? undefined : 1 / ((n - 1) * (n - 2))
  } else scale = g.directed ? undefined : 0.5
  /** Scale as the final result, extrapolating a partial or sampled run by n over the sources searched. */
  const scaled = (raw: Float64Array, searched: number): Float64Array => {
    const factor = searched === 0 || (searched === n && sources.length === n) ? 1 : n / searched
    const total = (scale ?? 1) * factor
    return scale === undefined && factor === 1 ? raw : Float64Array.from(raw, (x) => x * total)
  }
  return {
    name,
    config: { nodes: n, edges, normalized, weighted, endpoints, sources: sources.length },
    init: () => ({
      raw: new Float64Array(size),
      score: new Float64Array(size),
      delta: new Float64Array(n),
      sources,
      cursor: 0,
      source: -1,
    }),
    step: (st, ctx) => {
      const s = st.sources[st.cursor]
      if (s === undefined) return st
      const { order, sigma, preds, predEdges } = singleSource(g, s, weighted, false)
      const raw = Float64Array.from(st.raw)
      const delta = new Float64Array(n)
      if (endpoints) raw[s] = (raw[s] ?? 0) + order.length - 1
      for (let i = order.length - 1; i >= 0; i--) {
        const w = order[i] ?? 0
        const coeff = (1 + (delta[w] ?? 0)) / (sigma[w] ?? 1)
        const ps = preds[w] ?? []
        const es = predEdges[w] ?? []
        for (let j = 0; j < ps.length; j++) {
          const v = ps[j] ?? 0
          const c = (sigma[v] ?? 0) * coeff
          delta[v] = (delta[v] ?? 0) + c
          if (edges) {
            const e = es[j] ?? 0
            raw[e] = (raw[e] ?? 0) + c
          }
        }
        if (!edges && w !== s) raw[w] = (raw[w] ?? 0) + (delta[w] ?? 0) + (endpoints ? 1 : 0)
      }
      const cursor = st.cursor + 1
      const score = scaled(raw, cursor)
      if (ctx.wants('between/source')) ctx.log('between/source', s)
      if (ctx.wants('between/score')) ctx.log('between/score', Float64Array.from(score))
      if (ctx.wants('between/delta')) ctx.log('between/delta', Float64Array.from(delta))
      return { raw, score, delta, sources: st.sources, cursor, source: s }
    },
    done: (st) => (st.cursor >= st.sources.length ? 'converged' : undefined),
    ...codec<BetweennessState>(),
  }
}

/**
 * Brandes' betweenness centrality as a steppable algorithm: each step searches the shortest paths from one source
 * (breadth-first, or Dijkstra when weighted), sweeps back accumulating each node's dependency, and adds it to the
 * scores. Sources can be restricted or sampled; the scores are then extrapolated by n over the number searched.
 *
 * @param g - The graph.
 * @param options - `normalized` (default true), `weighted`, `endpoints`, and `sources` or `samples` with a `key`.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for `samples` without a key or out of range.
 * @concept centrality
 * @reference U. Brandes, A faster algorithm for betweenness centrality, J. Math. Sociology 25 (2001).
 * @index algorithm Betweenness centrality, one source node per step.
 * @example
 * ```ts
 * import { betweennessAlgorithm, pathGraph } from 'omai-engine/graph'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(betweennessAlgorithm(pathGraph(3)), { maxSteps: 3 })
 * [run.steps, Array.from(run.state.score)] // => [3, [0, 1, 0]]
 * ```
 */
export function betweennessAlgorithm(
  g: Graph,
  options: BetweennessOptions & BetweennessSampling = {},
): Algorithm<BetweennessState, BetweennessPath> {
  return brandes('graph/betweenness', g, false, options)
}

/**
 * Brandes' edge betweenness as a steppable algorithm: as {@link betweennessAlgorithm}, scoring each edge by the shortest
 * paths through it (the quantity Girvan-Newman removes greedily).
 *
 * @param g - The graph.
 * @param options - `normalized` (default true), `weighted`, and `sources` or `samples` with a `key`.
 * @returns The algorithm; its scores are per edge, in edge order.
 * @concept centrality
 * @example
 * ```ts
 * import { edgeBetweennessAlgorithm, pathGraph } from 'omai-engine/graph'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = edgeBetweennessAlgorithm(pathGraph(3), { normalized: false })
 * Array.from(runAlgorithm(alg, { maxSteps: 3 }).state.score) // => [2, 2]
 * ```
 */
export function edgeBetweennessAlgorithm(
  g: Graph,
  options: Omit<BetweennessOptions, 'endpoints'> & BetweennessSampling = {},
): Algorithm<BetweennessState, BetweennessPath> {
  return brandes('graph/edge-betweenness', g, true, options)
}

/**
 * Betweenness centrality by Brandes' algorithm (NetworkX betweenness_centrality): the number of shortest paths
 * between other pairs that pass through a node, each pair's paths sharing one unit. One breadth-first (or Dijkstra)
 * search per source and a reverse sweep accumulating dependencies, in O(n m) (O(n m + n^2 log n) weighted).
 *
 * @param g - The graph.
 * @param options - `normalized` (default true), `weighted` (default false) and `endpoints` (default false).
 * @returns The centrality of each node.
 * @concept centrality
 * @reference U. Brandes, A faster algorithm for betweenness centrality, J. Math. Sociology 25 (2001).
 * @example
 * ```ts
 * import { betweennessCentrality, pathGraph } from 'omai-engine/graph'
 * Array.from(betweennessCentrality(pathGraph(3))) // => [0, 1, 0]
 * Array.from(betweennessCentrality(pathGraph(3), { normalized: false })) // => [0, 1, 0]
 * ```
 */
export function betweennessCentrality(g: Graph, options: BetweennessOptions = {}): Float64Array {
  return runAlgorithm(betweennessAlgorithm(g, options), { maxSteps: g.n + 1 }).state.score
}

/**
 * Edge betweenness centrality (NetworkX edge_betweenness_centrality): the shortest paths through each edge, the
 * quantity Girvan-Newman removes greedily.
 *
 * @param g - The graph.
 * @param options - `normalized` (default true, dividing by n (n - 1)) and `weighted` (default false).
 * @returns The centrality of each edge, in edge order.
 * @concept centrality
 * @example
 * ```ts
 * import { edgeBetweennessCentrality, pathGraph } from 'omai-engine/graph'
 * Array.from(edgeBetweennessCentrality(pathGraph(3), { normalized: false })) // => [2, 2]
 * ```
 */
export function edgeBetweennessCentrality(
  g: Graph,
  options: { readonly normalized?: boolean; readonly weighted?: boolean } = {},
): Float64Array {
  return runAlgorithm(edgeBetweennessAlgorithm(g, options), { maxSteps: g.n + 1 }).state.score
}

/** The state of a power iteration: the current vector and the l1 change of the last step. */
export interface PowerState {
  readonly x: Float64Array
  readonly error: number
  /** For HITS: the authority scores alongside the hubs in `x`. */
  readonly authorities?: Float64Array
}

const POWER_SCHEMA = channels({
  metrics: { error: { description: 'l1 change of the vector in the last step', direction: 'lower' } },
  tensors: { x: { role: 'state', description: 'the current scores' } },
})

function observePower(s: PowerState) {
  return { metrics: { error: s.error }, tensors: { x: { role: 'state' as const, value: s.x } } }
}

function runPower(algorithm: Algorithm<PowerState>, maxIter: number, options: GraphRunOptions): PowerState {
  const result = runAlgorithm(algorithm, { ...options, maxSteps: maxIter })
  if (result.status !== 'converged') {
    throw new OmaiError('convergence', `${algorithm.name}: no convergence in ${maxIter} iterations`)
  }
  return result.state
}

/** y = A^T x over the out-adjacency (each node passes its value to its successors), optionally weighted. */
function pushForward(g: Graph, x: Float64Array, weights: Float64Array | null): Float64Array {
  const a = g.out
  const y = new Float64Array(g.n)
  for (let u = 0; u < g.n; u++) {
    const xu = x[u] ?? 0
    if (xu === 0) continue
    for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
      const v = a.indices[k] ?? 0
      y[v] = (y[v] ?? 0) + xu * (weights === null ? 1 : (weights[k] ?? 0))
    }
  }
  return y
}

/** The l1 distance |a - b|_1, the convergence measure of NetworkX's power iterations. */
function l1(a: Float64Array, b: Float64Array): number {
  return dasum(daxpy(-1, b, a))
}

/** Options of the power-iteration centralities. */
export interface PowerOptions extends GraphRunOptions {
  readonly maxIter?: number
  readonly tol?: number
  readonly weighted?: boolean
}

/**
 * Eigenvector centrality as a traced power iteration (NetworkX eigenvector_centrality): iterate x <- (A + I)^T x,
 * normalised in l2, until the l1 change falls below n tol. The shift by I keeps bipartite graphs from oscillating.
 *
 * @param g - The graph.
 * @param options - `tol` (default 1e-6), `weighted` (default false) and a start vector.
 * @returns The algorithm.
 * @concept centrality
 * @example
 * ```ts
 * import { cycleGraph, eigenvectorCentralityAlgorithm } from 'omai-engine/graph'
 * eigenvectorCentralityAlgorithm(cycleGraph(4)).name // => 'eigenvector-centrality'
 * ```
 */
export function eigenvectorCentralityAlgorithm(
  g: Graph,
  options: { readonly tol?: number; readonly weighted?: boolean; readonly start?: ArrayLike<number> } = {},
): Algorithm<PowerState> {
  const tol = options.tol ?? 1e-6
  const weights = options.weighted === true ? g.out.weights : null
  return {
    name: 'eigenvector-centrality',
    config: { tol, weighted: options.weighted ?? false } as Json,
    schema: POWER_SCHEMA,
    init: () => {
      const x = Float64Array.from({ length: g.n }, (_, i) => options.start?.[i] ?? 1)
      const total = kahanSum(x)
      if (total === 0) throw new OmaiError('value', 'eigenvectorCentrality: the start vector sums to zero')
      return { x: Float64Array.from(x, (v) => v / total), error: Number.POSITIVE_INFINITY }
    },
    step: (s) => {
      const y = pushForward(g, s.x, weights)
      for (let i = 0; i < g.n; i++) y[i] = (y[i] ?? 0) + (s.x[i] ?? 0)
      const norm = dnrm2(y) || 1
      const x = Float64Array.from(y, (v) => v / norm)
      return { x, error: l1(x, s.x) }
    },
    done: (s) => (s.error < g.n * tol ? 'converged' : undefined),
    observe: observePower,
  }
}

/**
 * Eigenvector centrality (NetworkX eigenvector_centrality): the leading eigenvector of A^T, unit l2 norm.
 *
 * @param g - The graph.
 * @param options - `maxIter` (default 100), `tol` (default 1e-6), `weighted` (default false) and tracing.
 * @returns The centrality of each node.
 * @throws OmaiError with code `convergence` when the iteration does not converge.
 * @concept centrality
 * @example
 * ```ts
 * import { completeGraph, eigenvectorCentrality } from 'omai-engine/graph'
 * Array.from(eigenvectorCentrality(completeGraph(4)), (v) => v.toFixed(4)) // => ['0.5000', '0.5000', '0.5000', '0.5000']
 * ```
 */
export function eigenvectorCentrality(
  g: Graph,
  options: PowerOptions & { readonly start?: ArrayLike<number> } = {},
): Float64Array {
  const { maxIter = 100, tol, weighted, start, ...run } = options
  const algorithm = eigenvectorCentralityAlgorithm(g, {
    ...(tol === undefined ? {} : { tol }),
    ...(weighted === undefined ? {} : { weighted }),
    ...(start === undefined ? {} : { start }),
  })
  return runPower(algorithm, maxIter, run).x
}

/** Options of {@link katzCentrality}. */
export interface KatzOptions extends PowerOptions {
  /** The attenuation (default 0.1); converges when below 1 / lambda_max. */
  readonly alpha?: number
  /** The exogenous term, one value or one per node (default 1). */
  readonly beta?: number | ArrayLike<number>
  /** Scale to unit l2 norm (default true). */
  readonly normalized?: boolean
}

/**
 * Katz centrality as a traced fixed-point iteration (NetworkX katz_centrality): x <- alpha A^T x + beta until the l1
 * change falls below n tol.
 *
 * @param g - The graph.
 * @param options - `alpha` (default 0.1), `beta` (default 1), `tol` (default 1e-6) and `weighted` (default false).
 * @returns The algorithm (the state's vector is not yet normalised).
 * @concept centrality
 * @example
 * ```ts
 * import { katzCentralityAlgorithm, pathGraph } from 'omai-engine/graph'
 * katzCentralityAlgorithm(pathGraph(3)).name // => 'katz-centrality'
 * ```
 */
export function katzCentralityAlgorithm(
  g: Graph,
  options: Omit<KatzOptions, 'maxIter' | 'normalized'> = {},
): Algorithm<PowerState> {
  const alpha = options.alpha ?? 0.1
  const tol = options.tol ?? 1e-6
  const beta = options.beta ?? 1
  const b = Float64Array.from({ length: g.n }, (_, i) => (typeof beta === 'number' ? beta : (beta[i] ?? 0)))
  const weights = options.weighted === true ? g.out.weights : null
  return {
    name: 'katz-centrality',
    config: { alpha, tol, weighted: options.weighted ?? false } as Json,
    schema: POWER_SCHEMA,
    init: () => ({ x: new Float64Array(g.n), error: Number.POSITIVE_INFINITY }),
    step: (s) => {
      const y = pushForward(g, s.x, weights)
      const x = Float64Array.from(y, (v, i) => alpha * v + (b[i] ?? 0))
      return { x, error: l1(x, s.x) }
    },
    done: (s) => (s.error < g.n * tol ? 'converged' : undefined),
    observe: observePower,
  }
}

/**
 * Katz centrality (NetworkX katz_centrality): the solution of x = alpha A^T x + beta, the weighted count of walks
 * ending at each node with longer walks attenuated by powers of alpha.
 *
 * @param g - The graph.
 * @param options - `alpha` (default 0.1), `beta` (default 1), `normalized` (default true), `maxIter` (default 1000),
 *   `tol` (default 1e-6), `weighted` (default false) and tracing.
 * @returns The centrality of each node.
 * @throws OmaiError with code `convergence` when alpha is too large for the iteration to converge.
 * @concept centrality
 * @example
 * ```ts
 * import { katzCentrality, starGraph } from 'omai-engine/graph'
 * const x = katzCentrality(starGraph(4))
 * (x[0] ?? 0) > (x[1] ?? 0) // => true
 * ```
 */
export function katzCentrality(g: Graph, options: KatzOptions = {}): Float64Array {
  const { maxIter = 1000, normalized = true, alpha, beta, tol, weighted, ...run } = options
  const algorithm = katzCentralityAlgorithm(g, {
    ...(alpha === undefined ? {} : { alpha }),
    ...(beta === undefined ? {} : { beta }),
    ...(tol === undefined ? {} : { tol }),
    ...(weighted === undefined ? {} : { weighted }),
  })
  const { x } = runPower(algorithm, maxIter, run)
  if (!normalized) return x
  const norm = dnrm2(x) || 1
  return Float64Array.from(x, (v) => v / norm)
}

/** Options of {@link pagerank}. */
export interface PagerankOptions extends PowerOptions {
  /** The damping factor (default 0.85). */
  readonly alpha?: number
  /** The teleport distribution (default uniform); normalised to sum 1. */
  readonly personalization?: ArrayLike<number>
  /** Where dangling nodes (no out-edges) send their mass (default: the personalization vector). */
  readonly dangling?: ArrayLike<number>
  readonly start?: ArrayLike<number>
}

function distribution(n: number, v: ArrayLike<number> | undefined, name: string): Float64Array {
  if (v === undefined) return new Float64Array(n).fill(1 / n)
  const x = Float64Array.from({ length: n }, (_, i) => v[i] ?? 0)
  const total = kahanSum(x)
  if (!(total > 0)) throw new OmaiError('value', `pagerank: the ${name} vector must have a positive sum`)
  return Float64Array.from(x, (p) => p / total)
}

/**
 * PageRank as a traced power iteration (NetworkX pagerank): x <- alpha (x P + (mass on dangling nodes) d) + (1 - alpha)
 * p, with P the weighted transition matrix, until the l1 change falls below n tol.
 *
 * @param g - The graph (undirected edges count both ways).
 * @param options - `alpha`, personalization, dangling distribution, start vector, `tol` (default 1e-6) and `weighted`
 *   (default true).
 * @returns The algorithm.
 * @concept pagerank
 * @example
 * ```ts
 * import { cycleGraph, pagerankAlgorithm } from 'omai-engine/graph'
 * pagerankAlgorithm(cycleGraph(3)).name // => 'pagerank'
 * ```
 */
export function pagerankAlgorithm(g: Graph, options: Omit<PagerankOptions, 'maxIter'> = {}): Algorithm<PowerState> {
  const alpha = options.alpha ?? 0.85
  const tol = options.tol ?? 1e-6
  const weighted = options.weighted ?? true
  const n = g.n
  const p = distribution(n, options.personalization, 'personalization')
  const d = options.dangling === undefined ? p : distribution(n, options.dangling, 'dangling')
  const a = g.out
  const strength = new Float64Array(n)
  for (let u = 0; u < n; u++) {
    for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
      strength[u] = (strength[u] ?? 0) + (weighted ? (a.weights[k] ?? 0) : 1)
    }
  }
  const dangling = Array.from({ length: n }, (_, u) => u).filter((u) => strength[u] === 0)
  const w = Float64Array.from(a.weights, (x, k) => (weighted ? x : 1) / (strength[sourceRow(g, k)] || 1))
  return {
    name: 'pagerank',
    config: { alpha, tol, weighted } as Json,
    schema: POWER_SCHEMA,
    init: () => ({ x: distribution(n, options.start, 'start'), error: Number.POSITIVE_INFINITY }),
    step: (s) => {
      const y = pushForward(g, s.x, w)
      let lost = 0
      for (const u of dangling) lost += s.x[u] ?? 0
      const x = Float64Array.from(y, (v, i) => alpha * (v + lost * (d[i] ?? 0)) + (1 - alpha) * (p[i] ?? 0))
      return { x, error: l1(x, s.x) }
    },
    done: (s) => (s.error < n * tol ? 'converged' : undefined),
    observe: observePower,
  }
}

/** The row of adjacency entry k. */
function sourceRow(g: Graph, k: number): number {
  const indptr = g.out.indptr
  let lo = 0
  let hi = g.n - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if ((indptr[mid] ?? 0) <= k) lo = mid
    else hi = mid - 1
  }
  return lo
}

/**
 * PageRank (NetworkX pagerank): the stationary distribution of a random surfer who follows an out-edge (chosen by
 * weight) with probability alpha and teleports by the personalization vector otherwise.
 *
 * @param g - The graph.
 * @param options - `alpha` (default 0.85), personalization, dangling, start, `maxIter` (default 100), `tol` (default
 *   1e-6), `weighted` (default true) and tracing.
 * @returns The PageRank of each node (sums to 1).
 * @throws OmaiError with code `convergence` when the iteration does not converge.
 * @concept pagerank
 * @reference L. Page, S. Brin, R. Motwani and T. Winograd, The PageRank citation ranking (1999).
 * @index function PageRank centrality.
 * @example
 * ```ts
 * import { cycleGraph, pagerank } from 'omai-engine/graph'
 * Array.from(pagerank(cycleGraph(4)), (v) => v.toFixed(3)) // => ['0.250', '0.250', '0.250', '0.250']
 * ```
 */
export function pagerank(g: Graph, options: PagerankOptions = {}): Float64Array {
  const { maxIter = 100, tracer, policy, parent, runId, seed, key, ...rest } = options
  const run: GraphRunOptions = {
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
    ...(seed === undefined ? {} : { seed }),
    ...(key === undefined ? {} : { key }),
  }
  return runPower(pagerankAlgorithm(g, rest), maxIter, run).x
}

/** Hub and authority scores. */
export interface HitsResult {
  readonly hubs: Float64Array
  readonly authorities: Float64Array
}

/**
 * HITS as a traced power iteration (NetworkX hits): authorities a <- A^T h and hubs h <- A a, the hubs scaled to
 * maximum 1, until the l1 change of the hubs falls below tol.
 *
 * @param g - The graph.
 * @param options - `tol` (default 1e-8), `weighted` (default true) and a start vector for the hubs.
 * @returns The algorithm (the state's `x` holds the hubs, `authorities` the authorities).
 * @concept hits
 * @example
 * ```ts
 * import { digraph, hitsAlgorithm } from 'omai-engine/graph'
 * hitsAlgorithm(digraph([[0, 1]])).name // => 'hits'
 * ```
 */
export function hitsAlgorithm(
  g: Graph,
  options: { readonly tol?: number; readonly weighted?: boolean; readonly start?: ArrayLike<number> } = {},
): Algorithm<PowerState> {
  const tol = options.tol ?? 1e-8
  const weights = (options.weighted ?? true) ? g.out.weights : null
  const pull = (a: Float64Array): Float64Array => {
    // h = A a: each node sums the scores of its successors.
    const h = new Float64Array(g.n)
    for (let u = 0; u < g.n; u++) {
      let s = 0
      for (let k = g.out.indptr[u] ?? 0; k < (g.out.indptr[u + 1] ?? 0); k++) {
        s += (a[g.out.indices[k] ?? 0] ?? 0) * (weights === null ? 1 : (weights[k] ?? 0))
      }
      h[u] = s
    }
    return h
  }
  return {
    name: 'hits',
    config: { tol, weighted: options.weighted ?? true } as Json,
    schema: POWER_SCHEMA,
    init: () => ({ x: distribution(g.n, options.start, 'start'), error: Number.POSITIVE_INFINITY }),
    step: (s) => {
      const authorities = pushForward(g, s.x, weights)
      const y = pull(authorities)
      const top = Math.max(0, maxOf(y))
      const x = Float64Array.from(y, (v) => v / (top || 1))
      return { x, error: l1(x, s.x), authorities }
    },
    done: (s) => (s.error < tol ? 'converged' : undefined),
    observe: observePower,
  }
}

/**
 * Hubs and authorities by Kleinberg's HITS (NetworkX hits): good hubs point to good authorities and good authorities
 * are pointed to by good hubs; the hubs are the leading eigenvector of A A^T and the authorities of A^T A.
 *
 * @param g - The graph.
 * @param options - `maxIter` (default 100), `tol` (default 1e-8), `normalized` (default true: each sums to 1;
 *   otherwise each has maximum 1), `weighted` (default true), a start vector and tracing.
 * @returns Hub and authority scores.
 * @throws OmaiError with code `convergence` when the iteration does not converge.
 * @concept hits
 * @reference J. Kleinberg, Authoritative sources in a hyperlinked environment, J. ACM 46 (1999).
 * @example
 * ```ts
 * import { digraph, hits } from 'omai-engine/graph'
 * const r = hits(digraph([[0, 2], [1, 2]]))
 * [Array.from(r.hubs), Array.from(r.authorities)] // => [[0.5, 0.5, 0], [0, 0, 1]]
 * ```
 */
export function hits(
  g: Graph,
  options: PowerOptions & { readonly normalized?: boolean; readonly start?: ArrayLike<number> } = {},
): HitsResult {
  const { maxIter = 100, normalized = true, tol, weighted, start, ...run } = options
  const state = runPower(
    hitsAlgorithm(g, {
      ...(tol === undefined ? {} : { tol }),
      ...(weighted === undefined ? {} : { weighted }),
      ...(start === undefined ? {} : { start }),
    }),
    maxIter,
    run,
  )
  const hubs = state.x
  const authorities = state.authorities ?? new Float64Array(g.n)
  const scaled = (v: Float64Array, by: 'sum' | 'max'): Float64Array => {
    const s = by === 'sum' ? kahanSum(v) : Math.max(0, maxOf(v))
    return Float64Array.from(v, (x) => x / (s || 1))
  }
  return normalized
    ? { hubs: scaled(hubs, 'sum'), authorities: scaled(authorities, 'sum') }
    : { hubs, authorities: scaled(authorities, 'max') }
}

/**
 * The single-source shortest-path DAG used by the centralities: settle order, path counts and predecessors.
 *
 * @param g - The graph.
 * @param source - The source.
 * @param options - `weighted` (default false).
 * @returns The order nodes were settled in, the number of shortest paths to each node, and their predecessors.
 * @example
 * ```ts
 * import { cycleGraph, shortestPathCounts } from 'omai-engine/graph'
 * Array.from(shortestPathCounts(cycleGraph(4), 0).sigma) // => [1, 1, 2, 1]
 * ```
 */
export function shortestPathCounts(
  g: Graph,
  source: NodeId,
  options: { readonly weighted?: boolean } = {},
): {
  readonly order: readonly number[]
  readonly sigma: Float64Array
  readonly predecessors: readonly (readonly number[])[]
} {
  const r = singleSource(g, nodeIndex(g, source), options.weighted ?? false, false)
  return { order: r.order, sigma: r.sigma, predecessors: r.preds }
}
