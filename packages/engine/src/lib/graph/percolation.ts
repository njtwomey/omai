/**
 * Percolation: sites or bonds of a graph switched on one at a time in a random order, the clusters merged by
 * union-find as they join, the share of the largest cluster jumping near a critical occupation probability and a
 * cluster spanning from one side to the other appearing there. The process as a traced algorithm (the step is the
 * occupation probability), the critical probabilities known for the common lattices, the giant component of an
 * Erdos-Renyi graph in closed form, and sweeps that collect the spanning thresholds of many random lattices to show the
 * transition sharpening with size.
 */

import { MdmlError, maxOf } from '../base/index.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  event,
  iterate,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { connectedComponents } from './components.ts'
import { gridGraph } from './generators.ts'
import type { Graph } from './graph.ts'
import { type NodeId, nodeIndex } from './graph.ts'
import { UnionFind } from './structures.ts'

/** The lattices whose critical probabilities are tabulated in {@link PERCOLATION_THRESHOLDS}. */
export type PercolationLattice =
  | 'square-site'
  | 'square-bond'
  | 'triangular-site'
  | 'triangular-bond'
  | 'honeycomb-site'
  | 'honeycomb-bond'
  | 'cubic-site'
  | 'cubic-bond'

/** What is switched on: the nodes (site percolation) or the edges (bond percolation). */
export type PercolationMode = 'site' | 'bond'

/** Options of {@link percolationAlgorithm}. */
export interface PercolationOptions {
  /** Site or bond percolation (default `'site'`). */
  readonly mode?: PercolationMode
  /** The nodes on one side; a cluster touching both `top` and `bottom` spans (default: no spanning). */
  readonly top?: readonly NodeId[]
  /** The nodes on the opposite side. */
  readonly bottom?: readonly NodeId[]
  /** Stop at the first step in which a cluster spans (default false). */
  readonly stopAtSpanning?: boolean
  /** A fixed order in which to switch things on (a permutation of the nodes, or of the edges in bond mode), instead of a random one. */
  readonly order?: ArrayLike<number>
}

/** State of {@link percolationAlgorithm}. */
export interface PercolationState {
  /** The order in which nodes (site mode) or edges (bond mode) are switched on. */
  readonly order: Int32Array
  /** How many are on. */
  readonly added: number
  /** The clusters. */
  readonly uf: UnionFind
  /** 1 for each node that is on (every node in bond mode). */
  readonly occupied: Uint8Array
  /** For each cluster root, 1 if it touches the top side and 2 if it touches the bottom. */
  readonly flags: Uint8Array
  /** The size of the largest cluster. */
  readonly largest: number
  /** The number of clusters (of occupied sites in site mode). */
  readonly clusters: number
  /** Whether a cluster spans. */
  readonly spans: boolean
  /** The step at which a cluster first spanned, or -1. */
  readonly spannedAt: number
}

/** The paths of {@link percolationAlgorithm}. */
export type PercolationPath =
  | 'lattice/p'
  | 'lattice/largestCluster'
  | 'lattice/clusters'
  | 'lattice/spans'
  | 'lattice/board'

/** The log contract of {@link percolationAlgorithm}. */
export const percolationContract: Contract<PercolationState, PercolationPath> = defineContract({
  'lattice/p': scalar({ description: 'the share of sites (or bonds) switched on' }),
  'lattice/largestCluster': scalar({ description: 'the largest cluster as a share of the nodes' }),
  'lattice/clusters': scalar({ description: 'the number of clusters' }),
  'lattice/spans': event({ context: { p: 'the occupation probability at which a cluster first spans' } }),
  'lattice/board': tensor({
    role: 'state',
    shape: ['nodes'],
    description: 'the cluster (its root node) of each node, -1 for a node that is off',
  }),
})

/**
 * Percolation on any graph as a traced algorithm, one site or bond switched on per step in a uniformly random order
 * drawn at the start. A cluster is a connected set of nodes that are on (site mode: an edge joins two nodes that are
 * both on) or of all nodes joined by edges that are on (bond mode). The occupation probability is the step over the
 * number of sites or bonds; the largest cluster's share rises sharply near the critical probability
 * ({@link PERCOLATION_THRESHOLDS}) and, with `top` and `bottom` given, `lattice/spans` fires once at the step at which
 * a single cluster first touches both. The state serialises (the union-find as plain data).
 *
 * @param g - The graph, for example {@link gridGraph}.
 * @param options - Site or bond mode, the two sides, and whether to stop at spanning.
 * @returns The algorithm; it completes when everything is on.
 * @throws MdmlError with code `value` for an unknown mode, `shape` for a fixed order of the wrong length.
 * @concept percolation
 * @example
 * ```ts
 * import { gridGraph, percolationAlgorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(percolationAlgorithm(gridGraph(4, 4), { mode: 'bond' }), { seed: 0, maxSteps: 100 })
 * [run.state.largest, run.state.clusters, run.status] // => [16, 1, 'completed']
 * ```
 */
export function percolationAlgorithm(
  g: Graph,
  options: PercolationOptions = {},
): Algorithm<PercolationState, PercolationPath> {
  const mode = options.mode ?? 'site'
  if (mode !== 'site' && mode !== 'bond')
    throw new MdmlError('value', `percolationAlgorithm: unknown mode '${String(mode)}'`)
  const top = new Set((options.top ?? []).map((v) => nodeIndex(g, v)))
  const bottom = new Set((options.bottom ?? []).map((v) => nodeIndex(g, v)))
  const sides = top.size > 0 && bottom.size > 0
  const total = mode === 'site' ? g.n : g.edges.length
  if (options.order !== undefined && options.order.length !== total)
    throw new MdmlError(
      'shape',
      `percolationAlgorithm: the order must list all ${total} ${mode === 'site' ? 'nodes' : 'edges'}`,
    )
  const sideFlags = (v: number): number => (top.has(v) ? 1 : 0) | (bottom.has(v) ? 2 : 0)
  const merge = (uf: UnionFind, flags: Uint8Array, a: number, b: number): boolean => {
    const fa = flags[uf.find(a)] ?? 0
    const fb = flags[uf.find(b)] ?? 0
    if (!uf.union(a, b)) return false
    flags[uf.find(a)] = fa | fb
    return true
  }
  return {
    name: `graph/percolation-${mode}`,
    config: {
      mode,
      nodes: g.n,
      edges: g.edges.length,
      spanning: sides,
      stopAtSpanning: options.stopAtSpanning ?? false,
    },
    schema: contractSchema(percolationContract),
    init: (key) => {
      const flags = new Uint8Array(g.n)
      const occupied = new Uint8Array(g.n)
      if (mode === 'bond') {
        occupied.fill(1)
        for (let v = 0; v < g.n; v++) flags[v] = sideFlags(v)
      }
      return {
        order:
          options.order === undefined ? Int32Array.from(permutation(key, total).data) : Int32Array.from(options.order),
        added: 0,
        uf: new UnionFind(g.n),
        occupied,
        flags,
        largest: mode === 'bond' && g.n > 0 ? 1 : 0,
        clusters: mode === 'bond' ? g.n : 0,
        spans: false,
        spannedAt: -1,
      }
    },
    step: (s, ctx) => {
      const uf = s.uf.clone()
      const occupied = Uint8Array.from(s.occupied)
      const flags = Uint8Array.from(s.flags)
      let clusters = s.clusters
      let touched: number
      if (mode === 'site') {
        const v = s.order[s.added] ?? 0
        occupied[v] = 1
        flags[v] = sideFlags(v)
        clusters += 1
        for (let e = g.out.indptr[v] ?? 0; e < (g.out.indptr[v + 1] ?? 0); e++) {
          const w = g.out.indices[e] ?? 0
          if (occupied[w] === 1 && merge(uf, flags, v, w)) clusters -= 1
        }
        // A directed graph's in-neighbours join the cluster as well.
        if (g.directed) {
          for (let e = g.in.indptr[v] ?? 0; e < (g.in.indptr[v + 1] ?? 0); e++) {
            const w = g.in.indices[e] ?? 0
            if (occupied[w] === 1 && merge(uf, flags, v, w)) clusters -= 1
          }
        }
        touched = v
      } else {
        const edge = g.edges[s.order[s.added] ?? 0]
        if (edge === undefined) throw new MdmlError('value', 'percolationAlgorithm: ran past the last bond')
        if (merge(uf, flags, edge.source, edge.target)) clusters -= 1
        touched = edge.source
      }
      const added = s.added + 1
      const largest = Math.max(s.largest, uf.setSize(touched))
      const spansNow = sides && (flags[uf.find(touched)] ?? 0) === 3
      const spans = s.spans || spansNow
      if (spans && !s.spans) ctx.emit('lattice/spans', { p: added / Math.max(total, 1) })
      return {
        order: s.order,
        added,
        uf,
        occupied,
        flags,
        largest,
        clusters,
        spans,
        spannedAt: spans && !s.spans ? added : s.spannedAt,
      }
    },
    done: (s) =>
      s.added >= total || (options.stopAtSpanning === true && s.spans)
        ? s.added >= total
          ? 'completed'
          : 'converged'
        : undefined,
    observe: (s) => {
      const board = new Int32Array(g.n)
      for (let v = 0; v < g.n; v++) board[v] = s.occupied[v] === 1 ? s.uf.find(v) : -1
      return {
        metrics: {
          'lattice/p': s.added / Math.max(total, 1),
          'lattice/largestCluster': s.largest / Math.max(g.n, 1),
          'lattice/clusters': s.clusters,
        },
        tensors: { 'lattice/board': { role: 'state', value: board, shape: [g.n] } },
      }
    },
    serialise: (s) => stateToJson({ ...s, uf: s.uf.toJSON() }),
    deserialise: (j) => {
      const o = stateFromJson(j) as Omit<PercolationState, 'uf'> & { uf: ReturnType<UnionFind['toJSON']> }
      return { ...o, uf: UnionFind.fromJSON(o.uf) }
    },
  }
}

/** A square lattice with its top and bottom rows, ready for {@link percolationAlgorithm}. */
export interface Lattice {
  readonly graph: Graph
  readonly top: readonly number[]
  readonly bottom: readonly number[]
}

/**
 * A `width` by `height` square lattice (open boundaries) with the nodes of its first and last rows as the two sides.
 *
 * @param width - Columns.
 * @param height - Rows.
 * @returns The graph and the sides.
 * @concept percolation
 * @example
 * ```ts
 * import { latticeSides } from 'mdml-engine/graph'
 * const l = latticeSides(3, 2)
 * [l.graph.n, l.top, l.bottom] // => [6, [0, 1, 2], [3, 4, 5]]
 * ```
 */
export function latticeSides(width: number, height: number): Lattice {
  return {
    graph: gridGraph(height, width),
    top: Array.from({ length: width }, (_, j) => j),
    bottom: Array.from({ length: width }, (_, j) => (height - 1) * width + j),
  }
}

/**
 * Site percolation on a square lattice, sites switched on one per step, spanning top to bottom
 * ({@link percolationAlgorithm} on {@link latticeSides}). The critical probability is 0.5927 in the limit of a large
 * lattice.
 *
 * @param width - Columns.
 * @param height - Rows (default `width`).
 * @param stopAtSpanning - Stop at the first spanning cluster (default false).
 * @returns The algorithm.
 * @concept percolation
 * @example
 * ```ts
 * import { sitePercolationAlgorithm } from 'mdml-engine/graph'
 * sitePercolationAlgorithm(10).name // => 'graph/percolation-site'
 * ```
 */
export function sitePercolationAlgorithm(
  width: number,
  height: number = width,
  stopAtSpanning = false,
): Algorithm<PercolationState, PercolationPath> {
  const l = latticeSides(width, height)
  return percolationAlgorithm(l.graph, { mode: 'site', top: l.top, bottom: l.bottom, stopAtSpanning })
}

/**
 * Bond percolation on a square lattice, bonds switched on one per step, spanning top to bottom. The critical
 * probability is exactly 1/2 in the limit of a large lattice.
 *
 * @param width - Columns.
 * @param height - Rows (default `width`).
 * @param stopAtSpanning - Stop at the first spanning cluster (default false).
 * @returns The algorithm.
 * @concept percolation
 * @example
 * ```ts
 * import { bondPercolationAlgorithm } from 'mdml-engine/graph'
 * bondPercolationAlgorithm(10).name // => 'graph/percolation-bond'
 * ```
 */
export function bondPercolationAlgorithm(
  width: number,
  height: number = width,
  stopAtSpanning = false,
): Algorithm<PercolationState, PercolationPath> {
  const l = latticeSides(width, height)
  return percolationAlgorithm(l.graph, { mode: 'bond', top: l.top, bottom: l.bottom, stopAtSpanning })
}

/**
 * The nodes of the spanning cluster of a state: those in a cluster that touches both sides, or an empty list when none
 * does.
 *
 * @param state - A state of {@link percolationAlgorithm} run with sides.
 * @returns The node indices, ascending.
 * @concept percolation
 * @example
 * ```ts
 * import { bondPercolationAlgorithm, spanningCluster } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * spanningCluster(runAlgorithm(bondPercolationAlgorithm(3, 3), { seed: 0, maxSteps: 100 }).state).length // => 9
 * ```
 */
export function spanningCluster(state: PercolationState): number[] {
  const out: number[] = []
  for (let v = 0; v < state.occupied.length; v++) {
    if (state.occupied[v] === 1 && (state.flags[state.uf.find(v)] ?? 0) === 3) out.push(v)
  }
  return out
}

/**
 * The critical occupation probabilities of the common lattices in the limit of infinite size: the square lattice's site
 * threshold (numerical, 0.592746) and its exactly known bond threshold 1/2, the triangular lattice's exact 1/2 (sites)
 * and 2 sin(pi / 18) (bonds), the honeycomb lattice's duals of those, and the simple cubic lattice's.
 *
 * @concept percolation
 * @example
 * ```ts
 * import { PERCOLATION_THRESHOLDS } from 'mdml-engine/graph'
 * [PERCOLATION_THRESHOLDS['square-bond'], PERCOLATION_THRESHOLDS['triangular-site']] // => [0.5, 0.5]
 * ```
 */
export const PERCOLATION_THRESHOLDS: Readonly<Record<PercolationLattice, number>> = {
  'square-site': 0.5927460507921,
  'square-bond': 0.5,
  'triangular-site': 0.5,
  'triangular-bond': 2 * Math.sin(Math.PI / 18),
  'honeycomb-site': 0.697043,
  'honeycomb-bond': 1 - 2 * Math.sin(Math.PI / 18),
  'cubic-site': 0.3116077,
  'cubic-bond': 0.2488126,
}

/**
 * The share of the nodes in the giant component of an Erdos-Renyi graph with mean degree c in the limit of many nodes:
 * zero for c <= 1, otherwise the positive root S of S = 1 - exp(-c S). The transition is at c = 1, the mean degree at
 * which the average node has one neighbour to pass a cascade on.
 *
 * @param c - The mean degree.
 * @returns The giant component's share of the nodes.
 * @throws MdmlError with code `value` for a negative mean degree.
 * @concept percolation
 * @example
 * ```ts
 * import { erdosRenyiGiantFraction } from 'mdml-engine/graph'
 * erdosRenyiGiantFraction(0.5) // => 0
 * Math.round(erdosRenyiGiantFraction(2) * 1e6) / 1e6 // => 0.796812
 * ```
 */
export function erdosRenyiGiantFraction(c: number): number {
  if (!(c >= 0)) throw new MdmlError('value', 'erdosRenyiGiantFraction: the mean degree must be non-negative')
  if (c <= 1) return 0
  let s = 1
  for (let i = 0; i < 200000; i++) {
    const next = 1 - Math.exp(-c * s)
    if (Math.abs(next - s) < 1e-15) return next
    s = next
  }
  return s
}

/**
 * The share of the nodes in the largest connected component of a graph.
 *
 * @param g - The graph.
 * @returns The share, in (0, 1] (0 for the empty graph).
 * @concept percolation
 * @example
 * ```ts
 * import { giantComponentFraction, graph } from 'mdml-engine/graph'
 * giantComponentFraction(graph([[0, 1], [1, 2]], { nodes: 5 })) // => 0.6
 * ```
 */
export function giantComponentFraction(g: Graph): number {
  if (g.n === 0) return 0
  return maxOf(connectedComponents(g).components.map((c) => c.length)) / g.n
}

/** The curves of one run of {@link percolationSweep}. */
export interface PercolationSweep {
  /** The occupation probability after each step, from 0. */
  readonly p: Float64Array
  /** The share of the nodes in the largest cluster. */
  readonly largest: Float64Array
  /** The number of clusters. */
  readonly clusters: Int32Array
  /** The occupation probability at which a cluster first spanned, or `null` if none did. */
  readonly spanningP: number | null
}

/**
 * One whole percolation run as curves: the occupation probability, the largest cluster's share and the number of
 * clusters after every step.
 *
 * @param g - The graph.
 * @param key - The random key.
 * @param options - As for {@link percolationAlgorithm}.
 * @returns The curves.
 * @concept percolation
 * @example
 * ```ts
 * import { gridGraph, percolationSweep } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * const s = percolationSweep(gridGraph(5, 5), key(0), { mode: 'bond' })
 * [s.p.length, s.largest[s.largest.length - 1]] // => [41, 1]
 * ```
 */
export function percolationSweep(g: Graph, key: Key, options: PercolationOptions = {}): PercolationSweep {
  const alg = percolationAlgorithm(g, options)
  const total = (options.mode ?? 'site') === 'site' ? g.n : g.edges.length
  const p: number[] = []
  const largest: number[] = []
  const clusters: number[] = []
  let spanningP: number | null = null
  for (const { state } of iterate(alg, { key, maxSteps: total })) {
    p.push(state.added / Math.max(total, 1))
    largest.push(state.largest / Math.max(g.n, 1))
    clusters.push(state.clusters)
    if (spanningP === null && state.spans) spanningP = state.added / Math.max(total, 1)
  }
  return {
    p: Float64Array.from(p),
    largest: Float64Array.from(largest),
    clusters: Int32Array.from(clusters),
    spanningP,
  }
}

/**
 * The occupation probability at which a cluster first spans a `width` by `height` lattice from top to bottom, for each
 * of `trials` random lattices. Their mean tends to the critical probability and their spread shrinks as the lattice
 * grows: the transition sharpening (finite-size scaling).
 *
 * @param width - Columns.
 * @param height - Rows.
 * @param mode - Site or bond percolation.
 * @param trials - How many lattices.
 * @param key - The random key.
 * @returns One spanning probability per trial.
 * @throws MdmlError with code `value` for fewer than one trial.
 * @concept percolation
 * @example
 * ```ts
 * import { spanningThresholds } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * spanningThresholds(6, 6, 'bond', 3, key(0)).length // => 3
 * ```
 */
export function spanningThresholds(
  width: number,
  height: number,
  mode: PercolationMode,
  trials: number,
  key: Key,
): number[] {
  if (!(Number.isInteger(trials) && trials >= 1))
    throw new MdmlError('value', 'spanningThresholds: trials must be a positive integer')
  const l = latticeSides(width, height)
  const alg = percolationAlgorithm(l.graph, { mode, top: l.top, bottom: l.bottom, stopAtSpanning: true })
  const total = mode === 'site' ? l.graph.n : l.graph.edges.length
  const out: number[] = []
  for (let i = 0; i < trials; i++) {
    const run = runAlgorithm(alg, { key: foldIn(key, i), maxSteps: total })
    out.push(run.state.added / total)
  }
  return out
}
