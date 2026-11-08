/**
 * Sampling from a graph, for scale: GraphSAGE's fan-out sampling of a few neighbours per node and hop (with the weights
 * that keep the aggregate unbiased and the variance it costs), layer-wise importance sampling of nodes shared by a
 * whole layer (FastGCN and LADIES), uniform random walks, node2vec's second-order biased walks with their exact
 * transition probabilities, and the random-walk subgraph samplers of GraphSAINT.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, type Key, subsample, uniform, uniformStream } from '../random/index.ts'
import { type Graph, graph, hasEdge, type NodeId, nodeIndex, subgraph } from './graph.ts'

/** Which neighbours a node draws from: those that point to it (`'in'`, the messages it receives), those it points to, or both. */
export type SampleDirection = 'in' | 'out' | 'both'

function neighbourList(g: Graph, v: number, direction: SampleDirection): number[] {
  const out: number[] = []
  const take = (a: Graph['out']): void => {
    for (let e = a.indptr[v] ?? 0; e < (a.indptr[v + 1] ?? 0); e++) out.push(a.indices[e] ?? 0)
  }
  if (!g.directed) take(g.out)
  else if (direction === 'in') take(g.in)
  else if (direction === 'out') take(g.out)
  else {
    take(g.in)
    take(g.out)
  }
  return out
}

/** The edges sampled at one hop: messages from `src` to `dst`, each with the weight that makes the sum an unbiased estimate. */
export interface SampledHop {
  readonly dst: readonly number[]
  readonly src: readonly number[]
  /** degree / (number sampled): multiplying each neighbour's contribution by it keeps the sum over the sample unbiased for the sum over all neighbours. */
  readonly weight: readonly number[]
}

/** The result of {@link neighbourSample}. */
export interface NeighbourSample {
  /** The nodes the batch is for. */
  readonly targets: readonly number[]
  /** The nodes needed after each hop (cumulative): `layers[0]` is the targets, the last is the receptive field. */
  readonly layers: readonly (readonly number[])[]
  /** The sampled edges of each hop. Hop h resamples every node in `layers[h]`, as in GraphSAGE. */
  readonly hops: readonly SampledHop[]
  /** Every node of the receptive field, the targets first. */
  readonly nodes: readonly number[]
  /** The sampled edges as a graph over those nodes, labelled by their ids in the full graph. */
  readonly subgraph: Graph
}

/** Options of {@link neighbourSample}. */
export interface NeighbourSampleOptions {
  /** Which neighbours to draw from (default `'in'`; irrelevant for undirected graphs). */
  readonly direction?: SampleDirection
  /** Draw with replacement (default false). */
  readonly replace?: boolean
}

/**
 * GraphSAGE's neighbour sampling (Hamilton, Ying and Leskovec): from a batch of target nodes, hop by hop, every node
 * known so far draws `fanouts[h]` of its neighbours (all of them when it has no more, or when the fan-out is negative),
 * so the receptive field grows by at most the product of the fan-outs instead of exploding with the degree. Each sampled
 * edge carries the weight degree / sampled, which makes the weighted neighbour sum unbiased for the full sum
 * ({@link neighbourSampleVariance} gives its variance). The draws for a node at a hop depend only on the key, the hop and
 * the node.
 *
 * @param g - The graph.
 * @param targets - The batch of target nodes.
 * @param fanouts - How many neighbours to draw at each hop; negative means all.
 * @param key - The random key.
 * @param options - Direction and whether to draw with replacement.
 * @returns The layers, the sampled edges per hop, the receptive field and the sampled subgraph.
 * @throws OmaiError with code `value` for a non-integer fan-out.
 * @concept neighbour-sampling
 * @example
 * ```ts
 * import { neighbourSample, starGraph } from 'omai-engine/graph'
 * import { key } from 'omai-engine/random'
 * const s = neighbourSample(starGraph(10), [0], [3], key(0))
 * [s.hops[0]?.src.length, s.hops[0]?.weight[0]] // => [3, 3.3333333333333335]
 * ```
 */
export function neighbourSample(
  g: Graph,
  targets: readonly NodeId[],
  fanouts: readonly number[],
  key: Key,
  options: NeighbourSampleOptions = {},
): NeighbourSample {
  const direction = options.direction ?? 'in'
  const replace = options.replace ?? false
  if (fanouts.some((f) => !Number.isInteger(f)))
    throw new OmaiError('value', 'neighbourSample: fan-outs must be integers')
  const known: number[] = [...new Set(targets.map((v) => nodeIndex(g, v)))]
  const seen = new Set(known)
  const layers: number[][] = [[...known]]
  const hops: SampledHop[] = []
  const pairs = new Map<string, [number, number]>()
  fanouts.forEach((fan, h) => {
    const dst: number[] = []
    const src: number[] = []
    const weight: number[] = []
    for (const v of [...known]) {
      const nb = neighbourList(g, v, direction)
      const d = nb.length
      if (d === 0) continue
      const k = fan < 0 ? d : fan
      const at = foldIn(foldIn(key, h), v)
      let picks: number[]
      if (replace && fan >= 0) {
        const u = uniform(at, [Math.max(k, 1)]).data
        picks = Array.from({ length: k }, (_, i) => nb[Math.min(d - 1, Math.floor((u[i] ?? 0) * d))] ?? 0)
      } else if (k >= d) picks = nb
      else picks = Array.from(subsample(at, d, k).data, (i) => nb[i] ?? 0)
      for (const u of picks) {
        dst.push(v)
        src.push(u)
        weight.push(picks.length === d && !(replace && fan >= 0) ? 1 : d / picks.length)
        pairs.set(`${u},${v}`, [u, v])
        if (!seen.has(u)) {
          seen.add(u)
          known.push(u)
        }
      }
    }
    hops.push({ dst, src, weight })
    layers.push([...known])
  })
  const sampled = graph(
    [...pairs.values()].map(([a, b]): [string, string] => [String(a), String(b)]),
    { nodes: known.map(String), directed: g.directed },
  )
  return { targets: layers[0] ?? [], layers, hops, nodes: [...known], subgraph: sampled }
}

/**
 * The variance of the unbiased neighbour-sum estimate that a fan-out of k gives: with population values x over a node's d
 * neighbours and sigma^2 their variance (divisor d), drawing k without replacement and scaling by d / k gives
 * d^2 sigma^2 / k times the finite-population factor (d - k) / (d - 1); with replacement the factor is 1. It falls to zero
 * as k approaches d.
 *
 * @param values - The value at each neighbour.
 * @param k - The fan-out.
 * @param replace - Whether the draws are with replacement (default false).
 * @returns The variance of the estimate of the sum.
 * @throws OmaiError with code `value` unless 1 <= k (and k <= d without replacement).
 * @concept neighbour-sampling
 * @example
 * ```ts
 * import { neighbourSampleVariance } from 'omai-engine/graph'
 * Math.round(neighbourSampleVariance([1, 2, 3, 4], 2) * 1e6) / 1e6 // => 6.666667
 * neighbourSampleVariance([1, 2, 3, 4], 4) // => 0
 * ```
 */
export function neighbourSampleVariance(values: ArrayLike<number>, k: number, replace = false): number {
  const d = values.length
  if (!(Number.isInteger(k) && k >= 1 && (replace || k <= d)))
    throw new OmaiError(
      'value',
      'neighbourSampleVariance: need an integer fan-out in [1, d] (any k >= 1 with replacement)',
    )
  const mean = Array.from(values).reduce((a, b) => a + b, 0) / d
  const sigma2 = Array.from(values).reduce((a, b) => a + (b - mean) ** 2, 0) / d
  const factor = replace || d === 1 ? 1 : (d - k) / (d - 1)
  return ((d * d * sigma2) / k) * factor
}

/** Options of {@link randomWalks} and {@link node2vecWalks}. */
export interface WalkSampleOptions {
  /** Choose steps in proportion to edge weights (default false: uniformly). */
  readonly weighted?: boolean
  /** Walks started from each start node (default 1). */
  readonly walksPerNode?: number
}

function pickByWeight(weights: readonly number[], u: number): number {
  let total = 0
  for (const w of weights) total += w
  let acc = 0
  for (let i = 0; i < weights.length; i++) {
    acc += (weights[i] ?? 0) / total
    if (u < acc) return i
  }
  return weights.length - 1
}

function steps(g: Graph, v: number, weighted: boolean): { nodes: number[]; weights: number[] } {
  const nodes: number[] = []
  const weights: number[] = []
  for (let e = g.out.indptr[v] ?? 0; e < (g.out.indptr[v + 1] ?? 0); e++) {
    nodes.push(g.out.indices[e] ?? 0)
    weights.push(weighted ? (g.out.weights[e] ?? 0) : 1)
  }
  return { nodes, weights }
}

/**
 * Uniform random walks (DeepWalk's sampler): from each start node, `walksPerNode` walks of `length` steps, each step to a
 * uniformly random out-neighbour (or by edge weight). A walk that reaches a node without out-edges ends there. The
 * walks are in start order, then walk index.
 *
 * @param g - The graph.
 * @param starts - The start nodes.
 * @param length - Steps per walk.
 * @param key - The random key.
 * @param options - `weighted` and `walksPerNode`.
 * @returns The walks, each `[start, ...]` with at most `length + 1` nodes.
 * @throws OmaiError with code `value` for a negative length.
 * @concept neighbour-sampling
 * @example
 * ```ts
 * import { pathGraph, randomWalks } from 'omai-engine/graph'
 * import { key } from 'omai-engine/random'
 * randomWalks(pathGraph(2), [0], 3, key(0))[0] // => [0, 1, 0, 1]
 * ```
 */
export function randomWalks(
  g: Graph,
  starts: readonly NodeId[],
  length: number,
  key: Key,
  options: WalkSampleOptions = {},
): number[][] {
  return walk(g, starts, length, key, options, () => 1)
}

function walk(
  g: Graph,
  starts: readonly NodeId[],
  length: number,
  key: Key,
  options: WalkSampleOptions,
  bias: (previous: number, x: number) => number,
): number[][] {
  if (!(Number.isInteger(length) && length >= 0))
    throw new OmaiError('value', 'walk: length must be a non-negative integer')
  const weighted = options.weighted ?? false
  const per = options.walksPerNode ?? 1
  const out: number[][] = []
  let index = 0
  for (const s of starts) {
    for (let w = 0; w < per; w++) {
      const next = uniformStream(foldIn(key, index++), 32)
      const path = [nodeIndex(g, s)]
      for (let t = 0; t < length; t++) {
        const here = path[path.length - 1] as number
        const { nodes, weights } = steps(g, here, weighted)
        if (nodes.length === 0) break
        const previous = path.length >= 2 ? (path[path.length - 2] as number) : -1
        const biased = previous < 0 ? weights : weights.map((x, i) => x * bias(previous, nodes[i] as number))
        path.push(nodes[pickByWeight(biased, next())] as number)
      }
      out.push(path)
    }
  }
  return out
}

/** Options of node2vec's walks. */
export interface Node2VecOptions extends WalkSampleOptions {
  /** The return parameter: a step back to the previous node has weight 1 / p (default 1). Small p keeps the walk local, like breadth-first search. */
  readonly p?: number
  /** The in-out parameter: a step to a node not adjacent to the previous one has weight 1 / q (default 1). Small q pushes the walk outwards, like depth-first search. */
  readonly q?: number
}

function node2vecBias(g: Graph, p: number, q: number): (previous: number, x: number) => number {
  if (!(p > 0 && q > 0)) throw new OmaiError('value', 'node2vec: p and q must be positive')
  return (previous, x) => (x === previous ? 1 / p : hasEdge(g, previous, x) ? 1 : 1 / q)
}

/**
 * node2vec's biased second-order random walks (Grover and Leskovec): having come from `t` to `v`, the next node `x` is
 * chosen in proportion to the edge weight times 1 / p if `x` is `t`, 1 if `x` is a neighbour of `t`, and 1 / q otherwise.
 * The first step is uniform. See {@link node2vecTransition} for the exact probabilities.
 *
 * @param g - The graph.
 * @param starts - The start nodes.
 * @param length - Steps per walk.
 * @param key - The random key.
 * @param options - `p`, `q`, `weighted` and `walksPerNode`.
 * @returns The walks.
 * @throws OmaiError with code `value` for a non-positive p or q or a negative length.
 * @concept neighbour-sampling
 * @example
 * ```ts
 * import { node2vecWalks, pathGraph } from 'omai-engine/graph'
 * import { key } from 'omai-engine/random'
 * // On a path with a tiny return parameter p the walk bounces straight back.
 * node2vecWalks(pathGraph(3), [1], 4, key(0), { p: 1e-9 })[0]?.slice(2) // => [1, 0, 1]
 * ```
 */
export function node2vecWalks(
  g: Graph,
  starts: readonly NodeId[],
  length: number,
  key: Key,
  options: Node2VecOptions = {},
): number[][] {
  return walk(g, starts, length, key, options, node2vecBias(g, options.p ?? 1, options.q ?? 1))
}

/**
 * The exact distribution of node2vec's next step after going from `previous` to `current`: the neighbours of
 * `current` and their probabilities, in proportion to weight times 1 / p (back to `previous`), 1 (a common neighbour of
 * both) or 1 / q (further out).
 *
 * @param g - The graph.
 * @param previous - The node the walk came from.
 * @param current - Where it is.
 * @param options - `p`, `q` and `weighted`.
 * @returns The candidate next nodes and their probabilities.
 * @throws OmaiError with code `value` for a non-positive p or q.
 * @concept neighbour-sampling
 * @example
 * ```ts
 * import { graph, node2vecTransition } from 'omai-engine/graph'
 * // Came from 0 to 1: back to 0 (1 / p = 0.5), to 2 which is adjacent to 0 (1), to 3 further out (1 / q = 4).
 * const g = graph([[0, 1], [1, 2], [0, 2], [1, 3]])
 * node2vecTransition(g, 0, 1, { p: 2, q: 0.25 }).probabilities // => [0.09090909090909091, 0.18181818181818182, 0.7272727272727273]
 * ```
 */
export function node2vecTransition(
  g: Graph,
  previous: NodeId,
  current: NodeId,
  options: Node2VecOptions = {},
): { readonly nodes: readonly number[]; readonly probabilities: readonly number[] } {
  const bias = node2vecBias(g, options.p ?? 1, options.q ?? 1)
  const t = nodeIndex(g, previous)
  const { nodes, weights } = steps(g, nodeIndex(g, current), options.weighted ?? false)
  const raw = weights.map((w, i) => w * bias(t, nodes[i] as number))
  const total = raw.reduce((a, b) => a + b, 0)
  return { nodes, probabilities: raw.map((x) => x / total) }
}

/** A subgraph sampled by random walks. */
export interface WalkSubgraph {
  /** The nodes visited, ascending. */
  readonly nodes: readonly number[]
  /** The subgraph they induce, renumbered `0 ..`, with the original ids as labels. */
  readonly subgraph: Graph
}

/**
 * GraphSAINT's random-walk sampler: start walks of a given length from random root nodes and keep the subgraph induced
 * by every node they visit, a minibatch that is connected and dense where the walks went.
 *
 * @param g - The graph.
 * @param roots - The number of root nodes to draw, or the roots.
 * @param length - Steps per walk.
 * @param key - The random key.
 * @returns The visited nodes and the induced subgraph.
 * @throws OmaiError with code `value` for more roots than nodes.
 * @concept neighbour-sampling
 * @example
 * ```ts
 * import { completeGraph, randomWalkSubgraph } from 'omai-engine/graph'
 * import { key } from 'omai-engine/random'
 * randomWalkSubgraph(completeGraph(6), [0], 0, key(0)).nodes // => [0]
 * ```
 */
export function randomWalkSubgraph(
  g: Graph,
  roots: number | readonly NodeId[],
  length: number,
  key: Key,
): WalkSubgraph {
  let starts: readonly NodeId[]
  if (typeof roots === 'number') {
    if (!(Number.isInteger(roots) && roots >= 0 && roots <= g.n))
      throw new OmaiError('value', 'randomWalkSubgraph: bad number of roots')
    starts = Array.from(subsample(foldIn(key, 0), g.n, roots).data)
  } else starts = roots
  const nodes = [...new Set(randomWalks(g, starts, length, foldIn(key, 1)).flat())].sort((a, b) => a - b)
  return { nodes, subgraph: subgraph(g, nodes) }
}

/** The nodes sampled for one layer by {@link layerwiseImportanceSample}. */
export interface LayerSample {
  /** The distinct nodes drawn. */
  readonly nodes: readonly number[]
  /** The probability each was drawn with at each of the layer's draws. */
  readonly probabilities: readonly number[]
  /** count / (draws x probability): the weights that make the layer's aggregate an unbiased estimate. */
  readonly weights: readonly number[]
}

/** Options of {@link layerwiseImportanceSample}. */
export interface LayerwiseOptions {
  /** `'importance'` (default) draws a node in proportion to the squared mass it sends to the layer above, `'uniform'` uniformly among the candidates. */
  readonly mode?: 'importance' | 'uniform'
}

/**
 * Layer-wise importance sampling (FastGCN, LADIES): instead of each node drawing its own neighbours, a whole layer draws
 * `sizes[l]` nodes (with replacement) from the neighbours of the layer above it, so the layers have a fixed width. A
 * candidate u is drawn with probability proportional to sum over v in the layer above of P[v, u]^2, where P = D^-1 A is
 * the mean-aggregation matrix (the mass u sends to the layer: FastGCN's choice, which bounds the variance without
 * looking at the features; the variance-optimal probability also depends on them), and the weights
 * count / (draws x probability) make the estimate sum over drawn u of weight_u P[v, u] x_u unbiased
 * ({@link sampledAggregate}). Its variance for one node v is (sum over u of (P[v, u] x_u)^2 / p_u - (P x)_v^2) / draws.
 *
 * @param g - The graph.
 * @param targets - The top layer (the batch).
 * @param sizes - How many draws for each layer, outwards.
 * @param key - The random key.
 * @param options - `mode`.
 * @returns One sample per layer.
 * @throws OmaiError with code `value` for a non-positive size, or when a layer has no candidates.
 * @concept neighbour-sampling
 * @example
 * ```ts
 * import { layerwiseImportanceSample, starGraph } from 'omai-engine/graph'
 * import { key } from 'omai-engine/random'
 * layerwiseImportanceSample(starGraph(5), [1], [4], key(0))[0]?.nodes // => [0]
 * ```
 */
export function layerwiseImportanceSample(
  g: Graph,
  targets: readonly NodeId[],
  sizes: readonly number[],
  key: Key,
  options: LayerwiseOptions = {},
): LayerSample[] {
  const mode = options.mode ?? 'importance'
  let above = [...new Set(targets.map((v) => nodeIndex(g, v)))]
  const layers: LayerSample[] = []
  sizes.forEach((draws, l) => {
    if (!(Number.isInteger(draws) && draws >= 1))
      throw new OmaiError('value', 'layerwiseImportanceSample: sizes must be positive integers')
    const score = new Map<number, number>()
    for (const v of above) {
      const nb = neighbourList(g, v, 'in')
      for (const u of nb) score.set(u, (score.get(u) ?? 0) + (mode === 'importance' ? 1 / nb.length ** 2 : 1))
    }
    const candidates = [...score.keys()].sort((a, b) => a - b)
    if (candidates.length === 0)
      throw new OmaiError('value', 'layerwiseImportanceSample: the layer has no neighbours to draw from')
    const total = [...score.values()].reduce((a, b) => a + b, 0)
    const p = candidates.map((u) => (score.get(u) ?? 0) / total)
    const next = uniformStream(foldIn(key, l), 64)
    const counts = new Map<number, number>()
    for (let i = 0; i < draws; i++) {
      const c = candidates[pickByWeight(p, next())] as number
      counts.set(c, (counts.get(c) ?? 0) + 1)
    }
    const nodes = [...counts.keys()].sort((a, b) => a - b)
    const probabilities = nodes.map((u) => p[candidates.indexOf(u)] ?? 0)
    layers.push({
      nodes,
      probabilities,
      weights: nodes.map((u, i) => (counts.get(u) ?? 0) / (draws * (probabilities[i] ?? 1))),
    })
    above = nodes
  })
  return layers
}

/**
 * The estimate of the mean-aggregated features P x of the nodes of one layer from the nodes sampled for the layer below:
 * for each node v, the sum over drawn u that are its neighbours of weight_u x_u / degree(v).
 *
 * @param g - The graph.
 * @param layer - The nodes whose aggregate is wanted.
 * @param sample - The nodes drawn below them.
 * @param x - The feature of every node of the graph.
 * @returns The estimate for each node of `layer`.
 * @concept neighbour-sampling
 * @example
 * ```ts
 * import { sampledAggregate, starGraph } from 'omai-engine/graph'
 * // The hub of a star averages its leaves; one drawn leaf with weight 4 stands in for all four.
 * sampledAggregate(starGraph(4), [0], { nodes: [1], probabilities: [0.25], weights: [4] }, [0, 1, 2, 3, 4]) // => [1]
 * ```
 */
export function sampledAggregate(
  g: Graph,
  layer: readonly NodeId[],
  sample: LayerSample,
  x: ArrayLike<number>,
): number[] {
  return layer.map((raw) => {
    const v = nodeIndex(g, raw)
    const nb = new Set(neighbourList(g, v, 'in'))
    let total = 0
    sample.nodes.forEach((u, i) => {
      if (nb.has(u)) total += ((sample.weights[i] ?? 0) * (x[u] ?? 0)) / nb.size
    })
    return total
  })
}
