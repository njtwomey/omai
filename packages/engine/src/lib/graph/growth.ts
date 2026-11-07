/**
 * Networks grown a step at a time, with what is known about them: Erdos-Renyi graphs edge by edge (the giant component
 * appearing as the mean degree passes 1, connectivity near (n / 2) ln n edges), preferential attachment node by node
 * (the degree law 2m(m + 1) / (k (k + 1) (k + 2)) against the exponential law of uniform attachment), the
 * Watts-Strogatz ring rewired edge by edge (clustering staying high while the paths shorten), a block model grown a node
 * at a time, and the configuration model that realises a given degree sequence. The growth processes are traced
 * algorithms whose tape holds the size, the giant component, the degrees and the clustering.
 */

import { MdmlError, maxOf } from '../base/index.ts'
import { foldIn, type Key, permutation, subsample, uniformStream } from '../random/index.ts'
import { type Algorithm, type Contract, contractSchema, defineContract, scalar, tensor } from '../trace/index.ts'
import { checkNumber, codec } from './_run.ts'
import { connectedComponents } from './components.ts'
import { wattsStrogatzGraph } from './generators.ts'
import { type Graph, graph } from './graph.ts'
import { averageClustering, averageShortestPathLength } from './measures.ts'

/** The paths of the growth algorithms. */
export type GrowthPath =
  | 'graph/nodes'
  | 'graph/edges'
  | 'graph/giantShare'
  | 'graph/components'
  | 'graph/meanDegree'
  | 'graph/maxDegree'
  | 'graph/clustering'
  | 'graph/pathLength'
  | 'graph/degrees'
  | 'graph/blocks'

/** The log contract of the growth algorithms (each logs the paths it has something to say about). */
export const growthContract: Contract<unknown, GrowthPath> = defineContract({
  'graph/nodes': scalar({ description: 'nodes so far' }),
  'graph/edges': scalar({ description: 'edges so far' }),
  'graph/giantShare': scalar({ description: 'the largest component as a share of the nodes' }),
  'graph/components': scalar({ description: 'connected components' }),
  'graph/meanDegree': scalar({ description: 'the mean degree' }),
  'graph/maxDegree': scalar({ description: 'the largest degree (the biggest hub)' }),
  'graph/clustering': scalar({ description: 'the average clustering coefficient' }),
  'graph/pathLength': scalar({ description: 'the average shortest path length over connected pairs' }),
  'graph/degrees': tensor({ role: 'state', shape: ['nodes'], description: 'the degree of each node' }),
  'graph/blocks': tensor({ role: 'state', shape: ['nodes'], description: 'the block of each node' }),
})

function pairs(flat: Int32Array): [number, number][] {
  const out: [number, number][] = []
  for (let i = 0; i < flat.length; i += 2) out.push([flat[i] ?? 0, flat[i + 1] ?? 0])
  return out
}

/**
 * The graph with `n` nodes and the edges stored as flat pairs in a growth state.
 *
 * @param n - The number of nodes.
 * @param flat - The edge ends, `[u0, v0, u1, v1, ...]`.
 * @returns The graph.
 * @concept growing-networks
 * @example
 * ```ts
 * import { growthGraph } from 'mdml-engine/graph'
 * growthGraph(3, Int32Array.of(0, 1, 1, 2)).edges.length // => 2
 * ```
 */
export function growthGraph(n: number, flat: Int32Array): Graph {
  return graph(pairs(flat), { nodes: n })
}

function observeGrowth(
  n: number,
  flat: Int32Array,
  extra: { readonly clustering?: boolean; readonly pathLength?: boolean } = {},
): { metrics: Record<string, number>; tensors: Record<string, { role: 'state'; value: Int32Array; shape: number[] }> } {
  const g = growthGraph(n, flat)
  const degrees = new Int32Array(n)
  for (let v = 0; v < n; v++) degrees[v] = (g.out.indptr[v + 1] ?? 0) - (g.out.indptr[v] ?? 0)
  const comps = connectedComponents(g)
  const metrics: Record<string, number> = {
    'graph/nodes': n,
    'graph/edges': g.edges.length,
    'graph/giantShare': n === 0 ? 0 : maxOf(comps.components.map((c) => c.length)) / n,
    'graph/components': comps.count,
    'graph/meanDegree': n === 0 ? 0 : (2 * g.edges.length) / n,
    'graph/maxDegree': n === 0 ? 0 : maxOf(degrees),
  }
  if (extra.clustering === true) metrics['graph/clustering'] = averageClustering(g)
  if (extra.pathLength === true) metrics['graph/pathLength'] = averageShortestPathLength(g)
  return { metrics, tensors: { 'graph/degrees': { role: 'state', value: degrees, shape: [n] } } }
}

/**
 * The pair of nodes with index `r` in the order (0,1), (0,2), ..., (0,n-1), (1,2), ...: the inverse of the ranking of
 * the pairs of an n-node graph.
 *
 * @param n - The number of nodes.
 * @param r - The pair's index, in [0, n (n - 1) / 2).
 * @returns The pair `[i, j]` with `i < j`.
 * @concept growing-networks
 * @example
 * ```ts
 * import { unrankPair } from 'mdml-engine/graph'
 * [unrankPair(4, 0), unrankPair(4, 3), unrankPair(4, 5)] // => [[0, 1], [1, 2], [2, 3]]
 * ```
 */
export function unrankPair(n: number, r: number): [number, number] {
  const i = n - 2 - Math.floor(Math.sqrt(-8 * r + 4 * n * (n - 1) - 7) / 2 - 0.5)
  const j = r + i + 1 - (n * (n - 1)) / 2 + ((n - i) * (n - i - 1)) / 2
  return [i, j]
}

/** State of {@link edgeByEdgeAlgorithm}. */
export interface EdgeByEdgeState {
  /** A random order of all node pairs, fixed at the start. */
  readonly order: Int32Array
  /** Edges added so far. */
  readonly k: number
  /** The edges, as flat pairs. */
  readonly edges: Int32Array
}

/**
 * The Erdos-Renyi graph grown one edge per step: the next pair of nodes in a uniformly random order of all pairs is
 * joined, so after k steps the graph is uniform among those with k edges (G(n, k)). The giant component appears when the
 * mean degree 2k / n passes 1 ({@link erdosRenyiGiantFraction}) and the graph becomes connected near
 * (n / 2) ln n edges ({@link erdosRenyiConnectivityEdges}). The order of all n (n - 1) / 2 pairs is drawn at the
 * start, so keep n to a few thousand. The state serialises.
 *
 * @param n - The number of nodes.
 * @param maxEdges - Stop after this many edges (default all pairs).
 * @returns The algorithm.
 * @throws MdmlError with code `value` for fewer than two nodes or `maxEdges` beyond the number of pairs.
 * @concept growing-networks
 * @example
 * ```ts
 * import { edgeByEdgeAlgorithm, growthGraph } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(edgeByEdgeAlgorithm(10, 12), { seed: 0, maxSteps: 100 })
 * [run.steps, growthGraph(10, run.state.edges).edges.length] // => [12, 12]
 * ```
 */
export function edgeByEdgeAlgorithm(n: number, maxEdges?: number): Algorithm<EdgeByEdgeState, GrowthPath> {
  if (!(Number.isInteger(n) && n >= 2)) throw new MdmlError('value', 'edgeByEdgeAlgorithm: need at least two nodes')
  const total = (n * (n - 1)) / 2
  const limit = maxEdges ?? total
  if (!(Number.isInteger(limit) && limit >= 0 && limit <= total))
    throw new MdmlError('value', `edgeByEdgeAlgorithm: maxEdges must be an integer in [0, ${total}]`)
  return {
    name: 'graph/edge-by-edge',
    config: { nodes: n, maxEdges: limit },
    schema: contractSchema(growthContract),
    init: (key) => ({ order: Int32Array.from(permutation(key, total).data), k: 0, edges: new Int32Array(0) }),
    step: (s) => {
      const [i, j] = unrankPair(n, s.order[s.k] ?? 0)
      const edges = new Int32Array(s.edges.length + 2)
      edges.set(s.edges)
      edges[s.edges.length] = i
      edges[s.edges.length + 1] = j
      return { order: s.order, k: s.k + 1, edges }
    },
    done: (s) => (s.k >= limit ? 'completed' : undefined),
    observe: (s) => observeGrowth(n, s.edges),
    ...codec<EdgeByEdgeState>(),
  }
}

/**
 * The number of edges at which an Erdos-Renyi graph on n nodes becomes connected, to first order: (n / 2) ln n, a mean
 * degree of ln n. Below it isolated nodes remain; the last of them vanishes within a few multiples of n / 2 edges.
 *
 * @param n - The number of nodes.
 * @returns The edge count.
 * @concept growing-networks
 * @example
 * ```ts
 * import { erdosRenyiConnectivityEdges } from 'mdml-engine/graph'
 * erdosRenyiConnectivityEdges(1) // => 0
 * ```
 */
export function erdosRenyiConnectivityEdges(n: number): number {
  return (n * Math.log(n)) / 2
}

/** How a new node picks the nodes it attaches to. */
export type AttachmentRule = 'preferential' | 'uniform'

/** State of {@link preferentialAttachmentAlgorithm}. */
export interface AttachmentState {
  /** Nodes so far. */
  readonly nodes: number
  /** The edges, as flat pairs; the endpoints are the "stubs" a preferential choice samples from. */
  readonly edges: Int32Array
}

/**
 * Preferential attachment (Barabasi and Albert) one node per step: the new node joins `m` distinct existing nodes, each
 * chosen with probability proportional to its degree (`'preferential'`: pick a uniformly random end of a uniformly
 * random edge) or uniformly (`'uniform'`, the control). It starts from a star on m + 1 nodes. Preferential attachment
 * grows hubs and a power-law degree distribution, P(k) = 2m(m + 1) / (k (k + 1) (k + 2)) ({@link barabasiAlbertDegreeLaw});
 * uniform attachment gives an exponential tail ({@link uniformAttachmentDegreeLaw}). The state serialises.
 *
 * @param n - The final number of nodes.
 * @param m - Edges per new node, 1 <= m < n.
 * @param rule - How targets are chosen (default `'preferential'`).
 * @returns The algorithm; it completes when the graph has n nodes.
 * @throws MdmlError with code `value` unless 1 <= m < n, or for an unknown rule.
 * @concept growing-networks
 * @example
 * ```ts
 * import { growthGraph, preferentialAttachmentAlgorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(preferentialAttachmentAlgorithm(20, 2), { seed: 0, maxSteps: 100 })
 * [run.state.nodes, growthGraph(20, run.state.edges).edges.length] // => [20, 36]
 * ```
 */
export function preferentialAttachmentAlgorithm(
  n: number,
  m: number,
  rule: AttachmentRule = 'preferential',
): Algorithm<AttachmentState, GrowthPath> {
  if (!(Number.isInteger(m) && m >= 1 && Number.isInteger(n) && m < n))
    throw new MdmlError('value', 'preferentialAttachmentAlgorithm: need integers 1 <= m < n')
  if (rule !== 'preferential' && rule !== 'uniform')
    throw new MdmlError('value', `preferentialAttachmentAlgorithm: unknown rule '${String(rule)}'`)
  return {
    name: `graph/attachment-${rule}`,
    config: { nodes: n, m, rule },
    schema: contractSchema(growthContract),
    init: () => {
      const edges = new Int32Array(2 * m)
      for (let i = 1; i <= m; i++) {
        edges[2 * (i - 1)] = 0
        edges[2 * (i - 1) + 1] = i
      }
      return { nodes: m + 1, edges }
    },
    step: (s, ctx) => {
      const targets = new Set<number>()
      if (rule === 'uniform') for (const t of subsample(ctx.key, s.nodes, m).data) targets.add(t)
      else {
        const next = uniformStream(ctx.key, 16)
        while (targets.size < m) targets.add(s.edges[Math.floor(next() * s.edges.length)] ?? 0)
      }
      const edges = new Int32Array(s.edges.length + 2 * m)
      edges.set(s.edges)
      let at = s.edges.length
      for (const t of [...targets].sort((a, b) => a - b)) {
        edges[at++] = t
        edges[at++] = s.nodes
      }
      return { nodes: s.nodes + 1, edges }
    },
    done: (s) => (s.nodes >= n ? 'completed' : undefined),
    observe: (s) => observeGrowth(s.nodes, s.edges),
    ...codec<AttachmentState>(),
  }
}

/**
 * The degree law of preferential attachment, in the limit of many nodes: P(k) = 2m(m + 1) / (k (k + 1) (k + 2)) for
 * k >= m (zero below), a power law with exponent 3.
 *
 * @param k - A degree.
 * @param m - The edges added per node.
 * @returns The probability that a node has degree k.
 * @concept growing-networks
 * @example
 * ```ts
 * import { barabasiAlbertDegreeLaw } from 'mdml-engine/graph'
 * barabasiAlbertDegreeLaw(3, 3) // => 0.4
 * ```
 */
export function barabasiAlbertDegreeLaw(k: number, m: number): number {
  return k < m ? 0 : (2 * m * (m + 1)) / (k * (k + 1) * (k + 2))
}

/**
 * The degree law of uniform attachment, in the limit of many nodes: P(k) = (m / (m + 1))^(k - m) / (m + 1) for k >= m,
 * geometric, so hubs do not form.
 *
 * @param k - A degree.
 * @param m - The edges added per node.
 * @returns The probability that a node has degree k.
 * @concept growing-networks
 * @example
 * ```ts
 * import { uniformAttachmentDegreeLaw } from 'mdml-engine/graph'
 * uniformAttachmentDegreeLaw(2, 2) // => 0.3333333333333333
 * ```
 */
export function uniformAttachmentDegreeLaw(k: number, m: number): number {
  return k < m ? 0 : (m / (m + 1)) ** (k - m) / (m + 1)
}

/** State of {@link rewiringAlgorithm}. */
export interface RewiringState {
  /** The edges as flat pairs, in the order of the ring lattice's edges: neighbour distance 1 first, node by node. */
  readonly edges: Int32Array
  /** Lattice edges considered so far. */
  readonly considered: number
  /** Edges that were rewired. */
  readonly rewired: number
}

/**
 * The Watts-Strogatz small world rewired one edge per step: start from the ring where each node joins its k nearest
 * neighbours, then go through the lattice edges (u, u + j), distance j = 1 first and node by node, rewiring each with
 * probability p to (u, w) for a uniformly random w that is neither u nor already a neighbour of u. Clustering stays high
 * ({@link wattsStrogatzClustering}) while the mean path length collapses from its lattice value
 * ({@link ringLatticePathLength}) after a few percent of the edges are rewired. The state serialises.
 *
 * @param n - Nodes on the ring.
 * @param k - Each node's lattice degree (even).
 * @param p - The rewiring probability.
 * @returns The algorithm; it completes after all n k / 2 lattice edges are considered.
 * @throws MdmlError with code `value` unless 2 <= k < n with k even and p in [0, 1].
 * @concept small-world-networks
 * @example
 * ```ts
 * import { rewiringAlgorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(rewiringAlgorithm(10, 4, 0), { seed: 0, maxSteps: 100 }).steps // => 20
 * ```
 */
export function rewiringAlgorithm(n: number, k: number, p: number): Algorithm<RewiringState, GrowthPath> {
  if (!(Number.isInteger(n) && Number.isInteger(k) && k >= 2 && k % 2 === 0 && k < n))
    throw new MdmlError('value', 'rewiringAlgorithm: need an even integer k with 2 <= k < n')
  checkNumber('rewiringAlgorithm', 'p', p, 0, 1)
  const half = k / 2
  const total = n * half
  return {
    name: 'graph/rewiring',
    config: { nodes: n, k, p },
    schema: contractSchema(growthContract),
    init: () => {
      const edges = new Int32Array(2 * total)
      for (let j = 1; j <= half; j++) {
        for (let u = 0; u < n; u++) {
          const e = (j - 1) * n + u
          edges[2 * e] = u
          edges[2 * e + 1] = (u + j) % n
        }
      }
      return { edges, considered: 0, rewired: 0 }
    },
    step: (s, ctx) => {
      const e = s.considered
      const next = uniformStream(ctx.key, 4)
      if (next() >= p) return { ...s, considered: e + 1 }
      const u = s.edges[2 * e] ?? 0
      const taken = new Set<number>([u])
      for (let i = 0; i < s.edges.length; i += 2) {
        if (s.edges[i] === u) taken.add(s.edges[i + 1] ?? 0)
        else if (s.edges[i + 1] === u) taken.add(s.edges[i] ?? 0)
      }
      if (taken.size >= n) return { ...s, considered: e + 1 }
      let w = Math.floor(next() * n)
      while (taken.has(w)) w = Math.floor(next() * n)
      const edges = Int32Array.from(s.edges)
      edges[2 * e + 1] = w
      return { edges, considered: e + 1, rewired: s.rewired + 1 }
    },
    done: (s) => (s.considered >= total ? 'completed' : undefined),
    observe: (s) => observeGrowth(n, s.edges, { clustering: true, pathLength: true }),
    ...codec<RewiringState>(),
  }
}

/**
 * The clustering coefficient of a Watts-Strogatz graph after rewiring with probability p, to first order:
 * 3 (k - 2) / (4 (k - 1)) (1 - p)^3 (Barrat and Weigt). The ring lattice's value is the p = 0 case.
 *
 * @param k - The lattice degree.
 * @param p - The rewiring probability.
 * @returns The expected clustering coefficient.
 * @concept small-world-networks
 * @example
 * ```ts
 * import { wattsStrogatzClustering } from 'mdml-engine/graph'
 * wattsStrogatzClustering(6, 0) // => 0.6
 * ```
 */
export function wattsStrogatzClustering(k: number, p: number): number {
  return ((3 * (k - 2)) / (4 * (k - 1))) * (1 - p) ** 3
}

/**
 * The exact mean shortest path length of the ring lattice on n nodes with k (even) nearest neighbours: a node at ring
 * distance d is ceil(d / (k / 2)) hops away.
 *
 * @param n - Nodes on the ring.
 * @param k - The lattice degree (even).
 * @returns The mean over pairs of distinct nodes.
 * @concept small-world-networks
 * @example
 * ```ts
 * import { ringLatticePathLength } from 'mdml-engine/graph'
 * ringLatticePathLength(6, 2) // => 1.8
 * ```
 */
export function ringLatticePathLength(n: number, k: number): number {
  const half = k / 2
  let total = 0
  for (let d = 1; d < n; d++) total += Math.ceil(Math.min(d, n - d) / half)
  return total / (n - 1)
}

/** One point of a {@link smallWorldSweep}. */
export interface SmallWorldPoint {
  readonly p: number
  /** Clustering divided by the lattice's. */
  readonly clustering: number
  /** Mean path length divided by the lattice's. */
  readonly pathLength: number
}

/**
 * The Watts-Strogatz picture: for each rewiring probability, the clustering and the mean path length of random
 * Watts-Strogatz graphs (averaged over trials), each relative to the ring lattice's. In the small-world window the path
 * length has collapsed while the clustering has hardly moved.
 *
 * @param key - The random key.
 * @param n - Nodes on the ring.
 * @param k - The lattice degree (even).
 * @param ps - The rewiring probabilities.
 * @param trials - Graphs per probability (default 5).
 * @returns One point per probability.
 * @concept small-world-networks
 * @example
 * ```ts
 * import { smallWorldSweep } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * smallWorldSweep(key(0), 40, 4, [0], 1) // => [{ p: 0, clustering: 1, pathLength: 1 }]
 * ```
 */
export function smallWorldSweep(key: Key, n: number, k: number, ps: readonly number[], trials = 5): SmallWorldPoint[] {
  const c0 = averageClustering(wattsStrogatzGraph(key, n, k, 0))
  const l0 = ringLatticePathLength(n, k)
  return ps.map((p, a) => {
    let c = 0
    let l = 0
    for (let t = 0; t < trials; t++) {
      const g = wattsStrogatzGraph(foldIn(foldIn(key, a), t), n, k, p)
      c += averageClustering(g)
      l += averageShortestPathLength(g)
    }
    return { p, clustering: c / trials / c0, pathLength: l / trials / l0 }
  })
}

/** State of {@link blockGrowthAlgorithm}. */
export interface BlockGrowthState {
  /** The block of each node so far. */
  readonly blocks: Int32Array
  /** The edges, as flat pairs. */
  readonly edges: Int32Array
}

/**
 * A stochastic block model grown one node per step: each new node joins block b with probability `shares[b]` and is
 * linked to every earlier node in block c independently with probability `probs[b][c]`. The finished graph is a
 * stochastic block model with random block sizes, whose expected edge count is {@link blockModelExpectedEdges}; the
 * communities show up in the adjacency as the graph fills in. The state serialises.
 *
 * @param n - The final number of nodes.
 * @param shares - The probability of each block (summing to 1).
 * @param probs - The symmetric matrix of edge probabilities between blocks.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for shares that do not sum to 1 or a matrix of the wrong shape or range.
 * @concept growing-networks
 * @example
 * ```ts
 * import { blockGrowthAlgorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(blockGrowthAlgorithm(6, [1], [[1]]), { seed: 0, maxSteps: 10 })
 * run.state.edges.length / 2 // => 15
 * ```
 */
export function blockGrowthAlgorithm(
  n: number,
  shares: readonly number[],
  probs: readonly (readonly number[])[],
): Algorithm<BlockGrowthState, GrowthPath> {
  checkBlocks(shares, probs)
  if (!(Number.isInteger(n) && n >= 1)) throw new MdmlError('value', 'blockGrowthAlgorithm: need at least one node')
  const draw = (u: number): number => {
    let acc = 0
    for (let b = 0; b < shares.length; b++) {
      acc += shares[b] ?? 0
      if (u < acc) return b
    }
    return shares.length - 1
  }
  return {
    name: 'graph/block-growth',
    config: { nodes: n, shares: [...shares], probs: probs.map((r) => [...r]) },
    schema: contractSchema(growthContract),
    init: (key) => ({ blocks: Int32Array.of(draw(uniformStream(key, 1)())), edges: new Int32Array(0) }),
    step: (s, ctx) => {
      const next = uniformStream(ctx.key, 64)
      const b = draw(next())
      const fresh: number[] = []
      for (let v = 0; v < s.blocks.length; v++) if (next() < (probs[b]?.[s.blocks[v] ?? 0] ?? 0)) fresh.push(v)
      const edges = new Int32Array(s.edges.length + 2 * fresh.length)
      edges.set(s.edges)
      fresh.forEach((v, i) => {
        edges[s.edges.length + 2 * i] = v
        edges[s.edges.length + 2 * i + 1] = s.blocks.length
      })
      const blocks = new Int32Array(s.blocks.length + 1)
      blocks.set(s.blocks)
      blocks[s.blocks.length] = b
      return { blocks, edges }
    },
    done: (s) => (s.blocks.length >= n ? 'completed' : undefined),
    observe: (s) => {
      const o = observeGrowth(s.blocks.length, s.edges)
      return {
        metrics: o.metrics,
        tensors: { ...o.tensors, 'graph/blocks': { role: 'state', value: s.blocks, shape: [s.blocks.length] } },
      }
    },
    ...codec<BlockGrowthState>(),
  }
}

function checkBlocks(shares: readonly number[], probs: readonly (readonly number[])[]): void {
  const k = shares.length
  if (k === 0 || shares.some((x) => !(x >= 0)) || Math.abs(shares.reduce((a, b) => a + b, 0) - 1) > 1e-9)
    throw new MdmlError('value', 'a block model needs shares that sum to 1')
  if (probs.length !== k || probs.some((r) => r.length !== k || r.some((x) => !(x >= 0 && x <= 1))))
    throw new MdmlError('value', 'a block model needs a square matrix of probabilities, one row per block')
}

/**
 * The expected number of edges of a block model grown to n nodes with random blocks: n (n - 1) / 2 times the chance
 * that two nodes are linked, sum over b and c of shares[b] shares[c] probs[b][c].
 *
 * @param n - The number of nodes.
 * @param shares - The probability of each block.
 * @param probs - The edge probabilities between blocks.
 * @returns The expected number of edges.
 * @throws MdmlError with code `value` for shares that do not sum to 1 or a bad matrix.
 * @concept growing-networks
 * @example
 * ```ts
 * import { blockModelExpectedEdges } from 'mdml-engine/graph'
 * blockModelExpectedEdges(10, [0.5, 0.5], [[1, 0], [0, 1]]) // => 22.5
 * ```
 */
export function blockModelExpectedEdges(
  n: number,
  shares: readonly number[],
  probs: readonly (readonly number[])[],
): number {
  checkBlocks(shares, probs)
  let link = 0
  for (let b = 0; b < shares.length; b++)
    for (let c = 0; c < shares.length; c++) link += (shares[b] ?? 0) * (shares[c] ?? 0) * (probs[b]?.[c] ?? 0)
  return ((n * (n - 1)) / 2) * link
}

/**
 * The configuration model: a random graph with a given degree sequence. Each node gets as many stubs as its degree, the
 * stubs are paired uniformly at random, and the self-loops and repeated edges the pairing produces are erased (a
 * `Graph` holds at most one edge per pair), so realised degrees fall short of the targets by the erased ends, which
 * are counted in the graph's attributes: `targetDegrees`, `selfLoops` and `multiEdges`. For sequences with a small
 * second moment the erasure is a vanishing share.
 *
 * @param key - The random key.
 * @param degrees - The target degree of each node (an even total).
 * @returns The graph.
 * @throws MdmlError with code `value` for a negative or non-integer degree or an odd total.
 * @concept growing-networks
 * @example
 * ```ts
 * import { configurationModelGraph } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * const g = configurationModelGraph(key(0), [1, 1])
 * g.edges.length // => 1
 * ```
 */
export function configurationModelGraph(key: Key, degrees: readonly number[]): Graph {
  if (degrees.some((d) => !(Number.isInteger(d) && d >= 0)))
    throw new MdmlError('value', 'configurationModelGraph: degrees must be non-negative integers')
  const stubs: number[] = []
  degrees.forEach((d, v) => {
    for (let i = 0; i < d; i++) stubs.push(v)
  })
  if (stubs.length % 2 !== 0)
    throw new MdmlError('value', 'configurationModelGraph: the degrees must sum to an even number')
  const order = permutation(key, stubs.length).data
  const seen = new Set<string>()
  const edges: [number, number][] = []
  let selfLoops = 0
  let multiEdges = 0
  for (let i = 0; i < stubs.length; i += 2) {
    const a = stubs[order[i] ?? 0] ?? 0
    const b = stubs[order[i + 1] ?? 0] ?? 0
    if (a === b) {
      selfLoops += 1
      continue
    }
    const id = a < b ? `${a},${b}` : `${b},${a}`
    if (seen.has(id)) {
      multiEdges += 1
      continue
    }
    seen.add(id)
    edges.push([a, b])
  }
  return graph(edges, {
    nodes: degrees.length,
    attributes: { targetDegrees: [...degrees], selfLoops, multiEdges },
  })
}
