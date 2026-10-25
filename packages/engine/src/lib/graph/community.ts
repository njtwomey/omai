/**
 * Community detection: partitioning a graph's nodes into groups that are densely linked inside and sparsely linked
 * between. Newman's modularity, the Louvain and Leiden multilevel modularity optimisers, asynchronous label
 * propagation, the Girvan-Newman hierarchy by edge-betweenness removal and spectral bisection by the Fiedler vector,
 * the iterative ones as traced algorithms (as in NetworkX).
 */

import { type Json, MdmlError, maxOf } from '../base/index.ts'
import { foldIn, type Key, permutation, uniform } from '../random/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { partitionFromLabels } from './_build.ts'
import {
  aggregate,
  compact,
  fastMoves,
  type LevelGraph,
  levelGraph,
  louvainMoves,
  quality,
  refine,
  totalWeight,
} from './_levels.ts'
import { edgeBetweennessCentrality } from './centrality.ts'
import { type Components, connectedComponents } from './components.ts'
import { type Graph, type GraphRunOptions, graph, toUndirected } from './graph.ts'
import { fiedlerVector } from './spectral.ts'

/** A partition of the nodes into communities, in the canonical form of {@link Components}. */
export type Partition = Components

/** Communities as lists of nodes, or a community label per node. */
export type CommunitiesLike = readonly (readonly number[])[] | ArrayLike<number>

function labelsOf(n: number, communities: CommunitiesLike): Int32Array {
  const first = (communities as ArrayLike<unknown>)[0]
  if (Array.isArray(first) || (communities as ArrayLike<unknown>).length === 0) {
    const labels = new Int32Array(n).fill(-1)
    for (const [c, members] of (communities as readonly (readonly number[])[]).entries()) {
      for (const v of members) {
        if (labels[v] !== -1) throw new MdmlError('value', `communities: node ${v} is in two communities`)
        labels[v] = c
      }
    }
    if (labels.includes(-1)) throw new MdmlError('value', 'communities: every node needs a community')
    return labels
  }
  if ((communities as ArrayLike<number>).length !== n) {
    throw new MdmlError('value', 'communities: give one label per node')
  }
  return Int32Array.from(communities as ArrayLike<number>)
}

/**
 * The partition with the given community labels, in canonical form (communities numbered by their smallest node).
 *
 * @param labels - A community label per node, or the communities as lists.
 * @param n - The number of nodes (needed for lists).
 * @returns The canonical partition.
 * @example
 * ```ts
 * import { partition } from 'mdml-engine/graph'
 * partition([5, 5, 2, 5]).components // => [[0, 1, 3], [2]]
 * ```
 */
export function partition(labels: CommunitiesLike, n?: number): Partition {
  const length =
    n ??
    (Array.isArray((labels as ArrayLike<unknown>)[0])
      ? (labels as readonly (readonly number[])[]).flat().length
      : labels.length)
  return partitionFromLabels(labelsOf(length, labels))
}

/**
 * Newman's modularity of a partition (NetworkX community.modularity): for an undirected graph
 * Q = sum over communities of L_c / m - resolution (K_c / 2m)^2, with L_c the weight inside the community, K_c its total
 * degree and m the total weight; for a directed graph Q = sum of L_c / m - resolution K_c^out K_c^in / m^2.
 *
 * @param g - The graph.
 * @param communities - Communities as node lists, or a label per node.
 * @param options - `resolution` (default 1) and `weighted` (default true).
 * @returns The modularity, in [-1/2, 1].
 * @throws MdmlError with code `value` when the communities do not partition the nodes.
 * @concept modularity
 * @reference M. E. J. Newman, Modularity and community structure in networks, PNAS 103 (2006).
 * @example
 * ```ts
 * import { barbellGraph, modularity } from 'mdml-engine/graph'
 * modularity(barbellGraph(3, 0), [[0, 1, 2], [3, 4, 5]]).toFixed(4) // => '0.3571'
 * ```
 */
export function modularity(
  g: Graph,
  communities: CommunitiesLike,
  options: { readonly resolution?: number; readonly weighted?: boolean } = {},
): number {
  const labels = labelsOf(g.n, communities)
  const resolution = options.resolution ?? 1
  const weighted = options.weighted ?? true
  const count = Math.max(0, maxOf(labels) + 1)
  const inside = new Float64Array(count)
  const out = new Float64Array(count)
  const into = new Float64Array(count)
  let m = 0
  for (const e of g.edges) {
    const w = weighted ? e.weight : 1
    m += w
    const a = labels[e.source] ?? 0
    const b = labels[e.target] ?? 0
    if (a === b) inside[a] = (inside[a] ?? 0) + w
    out[a] = (out[a] ?? 0) + w
    into[b] = (into[b] ?? 0) + w
  }
  if (m === 0) return 0
  let q = 0
  for (let c = 0; c < count; c++) {
    const kOut = out[c] ?? 0
    const kIn = into[c] ?? 0
    q += g.directed
      ? (inside[c] ?? 0) / m - (resolution * kOut * kIn) / (m * m)
      : (inside[c] ?? 0) / m - resolution * ((kOut + kIn) / (2 * m)) ** 2
  }
  return q
}

/** The state of a multilevel community search (Louvain or Leiden). */
export interface MultilevelState {
  /** The community of each original node. */
  readonly labels: Int32Array
  /** The current level graph (the communities contracted to nodes). */
  readonly level: LevelGraph
  /** The community of each level node (Leiden keeps it across aggregation). */
  readonly levelLabels: Int32Array
  readonly quality: number
  readonly levels: number
  readonly converged: boolean
}

/** Options of {@link louvain} and {@link leiden}. */
export interface MultilevelOptions extends GraphRunOptions {
  /** The resolution (default 1); larger values give smaller communities. */
  readonly resolution?: number
  /** Use edge weights (default true). */
  readonly weighted?: boolean
  /** Stop when a level improves the quality by at most this (default 1e-7). */
  readonly threshold?: number
  /** The most levels to run (default: until converged). */
  readonly maxLevel?: number
}

const MULTILEVEL_SCHEMA = channels({
  metrics: {
    quality: { description: 'modularity (or CPM quality) of the current partition', direction: 'higher' },
    communities: { description: 'number of communities' },
  },
  tensors: { labels: { role: 'state', description: 'community of each node' } },
})

function order(key: Key, n: number): number[] {
  return Array.from(permutation(key, n).data)
}

function mapThrough(labels: Int32Array, levelLabels: Int32Array): Int32Array {
  return Int32Array.from(labels, (c) => levelLabels[c] ?? c)
}

function qualityOf(
  g: Graph,
  labels: Int32Array,
  kind: 'modularity' | 'cpm',
  resolution: number,
  weighted: boolean,
): number {
  if (kind === 'modularity') return modularity(g, labels, { resolution, weighted })
  // CPM: sum over communities of internal weight - resolution * (n_c choose 2).
  const sizes = new Map<number, number>()
  for (const c of labels) sizes.set(c, (sizes.get(c) ?? 0) + 1)
  let h = 0
  for (const e of g.edges) if (labels[e.source] === labels[e.target]) h += weighted ? e.weight : 1
  for (const s of sizes.values()) h -= (resolution * s * (s - 1)) / 2
  return h
}

/**
 * The Louvain method as a traced algorithm (NetworkX louvain_communities): each step is one level, local moving of
 * nodes (in a random order from the step key) to the neighbouring community of largest modularity gain until nothing
 * moves, then contraction of the communities into nodes. It stops when a level improves modularity by at most the
 * threshold.
 *
 * @param input - The graph (directed graphs are read as undirected, reciprocal weights summed).
 * @param options - `resolution`, `weighted`, `threshold`.
 * @returns The algorithm.
 * @concept community-detection
 * @example
 * ```ts
 * import { karateClubGraph, louvainAlgorithm } from 'mdml-engine/graph'
 * louvainAlgorithm(karateClubGraph()).name // => 'louvain'
 * ```
 */
export function louvainAlgorithm(
  input: Graph,
  options: Omit<MultilevelOptions, 'maxLevel' | keyof GraphRunOptions> = {},
): Algorithm<MultilevelState> {
  const g = toUndirected(input, { combine: 'sum' })
  const resolution = options.resolution ?? 1
  const weighted = options.weighted ?? true
  const threshold = options.threshold ?? 1e-7
  const base = levelGraph(g, weighted)
  const m = totalWeight(base)
  return {
    name: 'louvain',
    config: { resolution, weighted, threshold } as Json,
    schema: MULTILEVEL_SCHEMA,
    init: () => {
      const labels = Int32Array.from({ length: g.n }, (_, i) => i)
      return {
        labels,
        level: base,
        levelLabels: Int32Array.from(labels),
        quality: qualityOf(g, labels, 'modularity', resolution, weighted),
        levels: 0,
        converged: g.n === 0 || m === 0,
      }
    },
    step: (s, ctx) => {
      const level = s.level
      const raw = Int32Array.from({ length: level.n }, (_, i) => i)
      const moved = louvainMoves(level, raw, quality(level, 'modularity', resolution, m), order(ctx.key, level.n))
      const { labels: local, count } = compact(raw)
      const labels = mapThrough(s.labels, local)
      const q = qualityOf(g, labels, 'modularity', resolution, weighted)
      const converged = !moved || q - s.quality <= threshold
      return {
        labels,
        level: aggregate(level, local, count),
        levelLabels: Int32Array.from({ length: count }, (_, i) => i),
        quality: q,
        levels: s.levels + 1,
        converged,
      }
    },
    done: (s) => (s.converged ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { quality: s.quality, communities: s.level.n },
      tensors: { labels: { role: 'state', value: Float64Array.from(s.labels) } },
    }),
  }
}

/**
 * Communities by the Louvain method (NetworkX louvain_communities): greedy multilevel modularity optimisation.
 * Randomised through the node order; the seed (or key) makes it reproducible.
 *
 * @param g - The graph.
 * @param options - `resolution` (default 1), `weighted` (default true), `threshold` (default 1e-7), `maxLevel`, and the
 *   seed, key and tracing.
 * @returns The partition.
 * @concept community-detection
 * @reference V. D. Blondel, J.-L. Guillaume, R. Lambiotte and E. Lefebvre, Fast unfolding of communities in large
 *   networks, J. Stat. Mech. (2008).
 * @example
 * ```ts
 * import { barbellGraph, louvain } from 'mdml-engine/graph'
 * louvain(barbellGraph(4, 0), { seed: 1 }).components // => [[0, 1, 2, 3], [4, 5, 6, 7]]
 * ```
 */
export function louvain(g: Graph, options: MultilevelOptions = {}): Partition {
  const { maxLevel, resolution, weighted, threshold, ...run } = options
  const algorithm = louvainAlgorithm(g, {
    ...(resolution === undefined ? {} : { resolution }),
    ...(weighted === undefined ? {} : { weighted }),
    ...(threshold === undefined ? {} : { threshold }),
  })
  return partitionFromLabels(runAlgorithm(algorithm, { ...run, maxSteps: maxLevel ?? g.n + 1 }).state.labels)
}

/** Options of {@link leiden} (it runs until no level merges anything, so it has no threshold). */
export interface LeidenOptions extends Omit<MultilevelOptions, 'threshold'> {
  /** The quality function: modularity (default) or the constant Potts model (NetworkX's default). */
  readonly quality?: 'modularity' | 'cpm'
  /** The randomness of the refinement (default 0.01). */
  readonly theta?: number
}

/**
 * The Leiden algorithm as a traced algorithm (NetworkX leiden_communities): each step runs fast local moving (a
 * queue of nodes revisited only when a neighbour moves), refines each community from singletons by randomised merges
 * of well-connected sub-communities, and contracts the refined partition while keeping the unrefined one as the
 * starting point of the next level. Unlike Louvain, every community it returns is connected.
 *
 * @param input - The graph (directed graphs are read as undirected, reciprocal weights summed).
 * @param options - `quality`, `resolution`, `theta` and `weighted`.
 * @returns The algorithm.
 * @concept community-detection
 * @reference V. A. Traag, L. Waltman and N. J. van Eck, From Louvain to Leiden: guaranteeing well-connected
 *   communities, Scientific Reports 9 (2019).
 * @example
 * ```ts
 * import { karateClubGraph, leidenAlgorithm } from 'mdml-engine/graph'
 * leidenAlgorithm(karateClubGraph()).name // => 'leiden'
 * ```
 */
export function leidenAlgorithm(
  input: Graph,
  options: Omit<LeidenOptions, 'maxLevel' | keyof GraphRunOptions> = {},
): Algorithm<MultilevelState> {
  const g = toUndirected(input, { combine: 'sum' })
  const kind = options.quality ?? 'modularity'
  const resolution = options.resolution ?? 1
  const weighted = options.weighted ?? true
  const theta = options.theta ?? 0.01
  const base = levelGraph(g, weighted)
  const m = totalWeight(base)
  return {
    name: 'leiden',
    config: { quality: kind, resolution, weighted, theta } as Json,
    schema: MULTILEVEL_SCHEMA,
    init: () => {
      const labels = Int32Array.from({ length: g.n }, (_, i) => i)
      return {
        labels,
        level: base,
        levelLabels: Int32Array.from(labels),
        quality: qualityOf(g, labels, kind, resolution, weighted),
        levels: 0,
        converged: g.n === 0 || (kind === 'modularity' && m === 0),
      }
    },
    step: (s, ctx) => {
      const level = s.level
      const q = quality(level, kind, resolution, m)
      const current = Int32Array.from(s.levelLabels)
      let next = level.n + current.length
      const moved = fastMoves(level, current, q, order(foldIn(ctx.key, 0), level.n), () => next++)
      const { labels: moving, count } = compact(current)
      const labels = mapThrough(s.labels, moving)
      const value = qualityOf(g, labels, kind, resolution, weighted)
      if (count === level.n || (!moved && s.levels > 0)) {
        return { ...s, levelLabels: moving, quality: value, levels: s.levels + 1, converged: true }
      }
      // Refine, contract the refined communities, and start the next level from the unrefined partition.
      let draw = 0
      const random = (): number => uniform(foldIn(foldIn(ctx.key, 2), draw++), [1]).data[0] ?? 0
      const refined = compact(refine(level, moving, q, order(foldIn(ctx.key, 1), level.n), theta, random))
      const nextLevel = aggregate(level, refined.labels, refined.count)
      const levelLabels = new Int32Array(refined.count)
      for (let v = 0; v < level.n; v++) levelLabels[refined.labels[v] ?? 0] = moving[v] ?? 0
      const stalled = refined.count === level.n
      return {
        labels: mapThrough(s.labels, refined.labels),
        level: nextLevel,
        levelLabels,
        quality: value,
        levels: s.levels + 1,
        converged: stalled,
      }
    },
    done: (s) => (s.converged ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { quality: s.quality, communities: s.level.n },
      tensors: { labels: { role: 'state', value: Float64Array.from(mapThrough(s.labels, s.levelLabels)) } },
    }),
  }
}

/**
 * Communities by the Leiden algorithm (NetworkX leiden_communities): Louvain's multilevel optimisation with a
 * refinement phase that guarantees connected communities. Randomised; the seed (or key) makes it reproducible.
 *
 * @param g - The graph.
 * @param options - `quality` (default modularity), `resolution` (default 1), `theta` (default 0.01), `weighted`
 *   (default true), `maxLevel`, and the seed, key and tracing.
 * @returns The partition.
 * @concept community-detection
 * @example
 * ```ts
 * import { karateClubGraph, leiden, modularity } from 'mdml-engine/graph'
 * const g = karateClubGraph()
 * modularity(g, leiden(g, { seed: 0 }).labels) > 0.41 // => true
 * ```
 */
export function leiden(g: Graph, options: LeidenOptions = {}): Partition {
  const { maxLevel, quality: kind, resolution, theta, weighted, ...run } = options
  const algorithm = leidenAlgorithm(g, {
    ...(kind === undefined ? {} : { quality: kind }),
    ...(resolution === undefined ? {} : { resolution }),
    ...(theta === undefined ? {} : { theta }),
    ...(weighted === undefined ? {} : { weighted }),
  })
  const { state } = runAlgorithm(algorithm, { ...run, maxSteps: maxLevel ?? 2 * g.n + 2 })
  return partitionFromLabels(mapThrough(state.labels, state.levelLabels))
}

/** The state of label propagation: the labels and how many changed in the last sweep. */
export interface LabelPropagationState {
  readonly labels: Int32Array
  readonly changed: number
}

/**
 * Asynchronous label propagation as a traced algorithm (NetworkX asyn_lpa_communities): every node starts with its own
 * label; each step sweeps the nodes in a random order, and a node whose label is not among the most frequent (by
 * weight) in its neighbourhood takes one of those at random. It stops after a sweep that changes nothing.
 *
 * @param g - The graph (edge directions are followed from each node to its successors).
 * @param options - `weighted` (default false).
 * @returns The algorithm.
 * @concept community-detection
 * @reference U. N. Raghavan, R. Albert and S. Kumara, Near linear time algorithm to detect community structures in
 *   large-scale networks, Phys. Rev. E 76 (2007).
 * @example
 * ```ts
 * import { karateClubGraph, labelPropagationAlgorithm } from 'mdml-engine/graph'
 * labelPropagationAlgorithm(karateClubGraph()).name // => 'label-propagation'
 * ```
 */
export function labelPropagationAlgorithm(
  g: Graph,
  options: { readonly weighted?: boolean } = {},
): Algorithm<LabelPropagationState> {
  const weighted = options.weighted ?? false
  const a = g.out
  return {
    name: 'label-propagation',
    config: { weighted } as Json,
    schema: channels({
      metrics: { changed: { description: 'labels changed in the last sweep', direction: 'lower' } },
      tensors: { labels: { role: 'state', description: 'label of each node' } },
    }),
    init: () => ({ labels: Int32Array.from({ length: g.n }, (_, i) => i), changed: g.n }),
    step: (s, ctx) => {
      const labels = Int32Array.from(s.labels)
      let changed = 0
      let draw = 0
      for (const u of order(foldIn(ctx.key, 0), g.n)) {
        const lo = a.indptr[u] ?? 0
        const hi = a.indptr[u + 1] ?? 0
        if (lo === hi) continue
        const freq = new Map<number, number>()
        for (let k = lo; k < hi; k++) {
          const c = labels[a.indices[k] ?? 0] ?? 0
          freq.set(c, (freq.get(c) ?? 0) + (weighted ? (a.weights[k] ?? 0) : 1))
        }
        const top = maxOf([...freq.values()])
        const best = [...freq.entries()].filter(([, f]) => f === top).map(([c]) => c)
        if (best.includes(labels[u] ?? -1)) continue
        const r = uniform(foldIn(foldIn(ctx.key, 1), draw++), [1]).data[0] ?? 0
        labels[u] = best[Math.floor(r * best.length)] ?? best[0] ?? 0
        changed++
      }
      return { labels, changed }
    },
    done: (s) => (s.changed === 0 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { changed: s.changed },
      tensors: { labels: { role: 'state', value: Float64Array.from(s.labels) } },
    }),
  }
}

/**
 * Communities by asynchronous label propagation (NetworkX asyn_lpa_communities): fast and parameter-free, but the
 * result depends on the random order (the seed or key makes it reproducible).
 *
 * @param g - The graph.
 * @param options - `weighted` (default false), `maxIter` (default 100 sweeps), and the seed, key and tracing.
 * @returns The partition.
 * @concept community-detection
 * @example
 * ```ts
 * import { barbellGraph, labelPropagation } from 'mdml-engine/graph'
 * labelPropagation(barbellGraph(5, 0), { seed: 0 }).count // => 2
 * ```
 */
export function labelPropagation(
  g: Graph,
  options: { readonly weighted?: boolean; readonly maxIter?: number } & GraphRunOptions = {},
): Partition {
  const { weighted, maxIter = 100, ...run } = options
  const algorithm = labelPropagationAlgorithm(g, weighted === undefined ? {} : { weighted })
  return partitionFromLabels(runAlgorithm(algorithm, { ...run, maxSteps: maxIter }).state.labels)
}

/** The state of Girvan-Newman: the edges still present and the partitions found so far. */
export interface GirvanNewmanState {
  /** Original edge indices still in the graph. */
  readonly remaining: readonly number[]
  readonly components: number
  readonly partitions: readonly Partition[]
  /** The edge removed by the last step. */
  readonly removed?: number
}

/**
 * The Girvan-Newman method as a traced algorithm (NetworkX girvan_newman): each step removes the edge of highest edge
 * betweenness (recomputed every time; ties go to the first edge in edge order), and whenever the number of components
 * grows the new partition is recorded.
 *
 * @param g - The graph.
 * @param options - `weighted` (default false: betweenness by hop counts).
 * @returns The algorithm.
 * @concept community-detection
 * @reference M. Girvan and M. E. J. Newman, Community structure in social and biological networks, PNAS 99 (2002).
 * @example
 * ```ts
 * import { girvanNewmanAlgorithm, karateClubGraph } from 'mdml-engine/graph'
 * girvanNewmanAlgorithm(karateClubGraph()).name // => 'girvan-newman'
 * ```
 */
export function girvanNewmanAlgorithm(
  g: Graph,
  options: { readonly weighted?: boolean } = {},
): Algorithm<GirvanNewmanState> {
  const weighted = options.weighted ?? false
  const rebuild = (remaining: readonly number[]): Graph =>
    graph(
      remaining.map((k) => {
        const e = g.edges[k] ?? { source: 0, target: 0, weight: 1 }
        return [e.source, e.target, e.weight] as const
      }),
      { nodes: g.n, directed: g.directed },
    )
  return {
    name: 'girvan-newman',
    config: { weighted } as Json,
    schema: channels({
      metrics: { components: { description: 'number of components' }, edges: { description: 'edges left' } },
      messages: { removed: { description: 'the edge just removed', payload: { source: 'node', target: 'node' } } },
    }),
    init: () => ({
      remaining: g.edges.map((_, k) => k).filter((k) => g.edges[k]?.source !== g.edges[k]?.target),
      components: connectedComponents(g).count,
      partitions: [],
    }),
    step: (s) => {
      const h = rebuild(s.remaining)
      const eb = edgeBetweennessCentrality(h, { weighted })
      const top = maxOf(eb)
      const at = eb.findIndex((x) => x >= top - 1e-9 * Math.max(1, Math.abs(top)))
      // Edges of the rebuilt graph are in the order of `remaining`.
      const removed = s.remaining[at] ?? -1
      const remaining = s.remaining.filter((_, i) => i !== at)
      const parts = connectedComponents(rebuild(remaining))
      const partitions = parts.count > s.components ? [...s.partitions, parts] : s.partitions
      return { remaining, components: parts.count, partitions, removed }
    },
    done: (s) => (s.remaining.length === 0 ? 'converged' : undefined),
    observe: (s) => {
      const e = s.removed === undefined ? undefined : g.edges[s.removed]
      return {
        metrics: { components: s.components, edges: s.remaining.length },
        ...(e === undefined
          ? {}
          : { messages: [{ kind: 'removed', payload: { source: e.source, target: e.target } }] }),
      }
    },
  }
}

/**
 * The Girvan-Newman hierarchy (NetworkX girvan_newman): the successive partitions obtained by removing edges of highest
 * betweenness, one per increase in the number of components.
 *
 * @param g - The graph.
 * @param options - `levels` (how many partitions to return; default all), `weighted` (default false), and tracing.
 * @returns The partitions, coarsest first.
 * @concept community-detection
 * @example
 * ```ts
 * import { barbellGraph, girvanNewman } from 'mdml-engine/graph'
 * girvanNewman(barbellGraph(3, 0), { levels: 1 })[0]?.components // => [[0, 1, 2], [3, 4, 5]]
 * ```
 */
export function girvanNewman(
  g: Graph,
  options: { readonly levels?: number; readonly weighted?: boolean } & GraphRunOptions = {},
): Partition[] {
  const { levels, weighted, ...run } = options
  const algorithm = girvanNewmanAlgorithm(g, weighted === undefined ? {} : { weighted })
  const limit = levels ?? Number.POSITIVE_INFINITY
  const stop: Algorithm<GirvanNewmanState> = {
    ...algorithm,
    done: (s, t) => (s.partitions.length >= limit ? 'converged' : algorithm.done?.(s, t)),
  }
  return [...runAlgorithm(stop, { ...run, maxSteps: g.edges.length + 1 }).state.partitions]
}

/**
 * Spectral bisection (NetworkX spectral_bisection): split the nodes by the sign of the Fiedler vector, the eigenvector
 * of the second-smallest Laplacian eigenvalue, which relaxes the minimum ratio cut.
 *
 * @param g - A connected graph.
 * @param options - `normalised` (default false: the combinatorial Laplacian) and `weighted` (default true).
 * @returns The two sides: nodes with a negative entry, then the rest.
 * @concept spectral-clustering
 * @example
 * ```ts
 * import { barbellGraph, spectralBisection } from 'mdml-engine/graph'
 * spectralBisection(barbellGraph(4, 0)).map((side) => side.length) // => [4, 4]
 * ```
 */
export function spectralBisection(
  g: Graph,
  options: { readonly normalised?: boolean; readonly weighted?: boolean } = {},
): [number[], number[]] {
  const v = fiedlerVector(g, options)
  const negative: number[] = []
  const rest: number[] = []
  for (let i = 0; i < g.n; i++) ((v[i] ?? 0) < 0 ? negative : rest).push(i)
  return [negative, rest]
}
