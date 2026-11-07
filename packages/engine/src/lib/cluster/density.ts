/**
 * Density-based clustering as scikit-learn does it: DBSCAN (clusters grown through dense neighbourhoods, other points
 * noise, one cluster per step), OPTICS (a reachability ordering built one point per step, from which clusters at any
 * radius are read), HDBSCAN (a hierarchy over the mutual reachability distance, condensed and cut by excess of mass,
 * with the condensed tree and the stability of its clusters) and mean shift (hill climbing on a kernel density to its
 * modes, every seed moving each step).
 */

import { MdmlError, maxOf, toRows } from '../base/index.ts'
import {
  type DistanceAlias,
  type DistanceName,
  type DistanceOptions,
  distanceMatrix,
  type MetricSlot,
  pairwiseDistances,
} from '../distances/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { choice, dict, int, real, type SpaceValue } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { kthInRow, labelledPredict, rowsAt } from './_support.ts'
import { type ClusterSelectionRule, hdbscanSelections } from './density-options.ts'
import type { Merge } from './hierarchical.ts'

/** A point's role in a density clustering. */
export type DensityRole = 'core' | 'border' | 'noise'

/** The verbs of a fitted density clustering: new points join the cluster of a nearby labelled point, else noise. */
export interface DensityVerbs {
  readonly predict: (x: TensorLike) => Tensor
}

const rolesOf = (labels: readonly number[], core: readonly boolean[]): DensityRole[] =>
  labels.map((l, i) => (core[i] === true ? 'core' : l >= 0 ? 'border' : 'noise'))

// ---------------------------------------------------------------------------------------------------------------
// DBSCAN

/** The state of DBSCAN's cluster expansion: the labels so far (-1: unlabelled or noise) and the next seed row. */
export interface DbscanState {
  readonly labels: readonly number[]
  /** Whether each row is a core point (it has `minSamples` rows within `eps`, itself included). */
  readonly core: readonly boolean[]
  /** The next row to try as the seed of a new cluster. */
  readonly next: number
  readonly clusters: number
}

/** Options of DBSCAN. */
export interface DbscanOptions {
  /** The neighbourhood radius (default 0.5). */
  readonly eps?: number
  /** Rows within `eps` (itself included) that make a core point (default 5). */
  readonly minSamples?: number
  /** The distance (default euclidean), or `'precomputed'`. */
  readonly metric?: MetricSlot
  readonly distanceOptions?: DistanceOptions
}

/**
 * DBSCAN's expansion (Ester et al. 1996; sklearn.cluster.DBSCAN) as a traced {@link Algorithm}: each step takes the
 * next unlabelled core point and grows its cluster through every density-reachable point, so the cluster appearing
 * each step is on the tape; what is left unlabelled at the end is noise.
 *
 * @param x - Points `[n, d]`, or with `metric: 'precomputed'` the distances.
 * @param options - Radius, core threshold and metric.
 * @returns The algorithm; at most n steps.
 * @example
 * ```ts
 * import { dbscanAlgorithm } from 'mdml-engine/cluster'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0], [0.1], [0.2], [5], [5.1], [5.2], [9]]
 * runAlgorithm(dbscanAlgorithm(x, { eps: 0.3, minSamples: 2 }), { maxSteps: 7 }).state.labels // => [0, 0, 0, 1, 1, 1, -1]
 * ```
 */
export function dbscanAlgorithm(x: TensorLike, options: DbscanOptions = {}): Algorithm<DbscanState> {
  const eps = options.eps ?? 0.5
  const minSamples = options.minSamples ?? 5
  const { n, data } = distanceMatrix(x, options.metric ?? 'euclidean', options.distanceOptions)
  const neighbours = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => j).filter((j) => (data[i * n + j] ?? 0) <= eps),
  )
  const core = neighbours.map((nb) => nb.length >= minSamples)
  const nextCore = (labels: readonly number[], from: number): number => {
    let i = from
    while (i < n && !(core[i] === true && (labels[i] ?? -1) < 0)) i++
    return i
  }
  return {
    name: 'dbscan',
    config: { eps, minSamples, n },
    schema: channels({
      metrics: {
        clusters: { description: 'the number of clusters found' },
        unlabelled: { description: 'the rows not yet in a cluster' },
      },
      tensors: { labels: { role: 'state', description: 'the cluster of each row (-1: none)', shape: ['n'] } },
    }),
    init: () => {
      const labels = new Array<number>(n).fill(-1)
      return { labels, core, next: nextCore(labels, 0), clusters: 0 }
    },
    step: (s) => {
      const labels = [...s.labels]
      const c = s.clusters
      const queue = [s.next]
      labels[s.next] = c
      for (let head = 0; head < queue.length; head++) {
        const p = queue[head] as number
        if (core[p] !== true) continue
        for (const q of neighbours[p] ?? []) {
          if ((labels[q] ?? -1) >= 0) continue
          labels[q] = c
          queue.push(q)
        }
      }
      return { labels, core, next: nextCore(labels, s.next + 1), clusters: c + 1 }
    },
    done: (s) => (s.next >= n ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { clusters: s.clusters, unlabelled: s.labels.filter((l) => l < 0).length },
      tensors: { labels: { role: 'state', value: s.labels, shape: [n] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as DbscanState,
  }
}

/** Hyperparameters of {@link dbscan}. */
export interface DbscanHyper {
  readonly eps: number
  readonly minSamples: number
  readonly metric: DistanceName | DistanceAlias
}

/** The fitted parameters of {@link dbscan}. */
export type DbscanParams = {
  readonly labels: number[]
  readonly roles: DensityRole[]
  readonly coreSampleIndices: number[]
  readonly components: number[][]
  readonly eps: number
  readonly metric: DistanceName | DistanceAlias
}

/**
 * DBSCAN on the iterative base: one cluster grown per step. New rows join the cluster of the nearest core point within
 * `eps`, else are noise.
 *
 * @index estimator DBSCAN density clustering.
 * @example
 * ```ts
 * import { dbscan } from 'mdml-engine/cluster'
 * dbscan.with({ eps: 0.3, minSamples: 2 }).fit({ x: [[0], [0.1], [5], [5.1], [9]] }).params.labels // => [0, 0, 1, 1, -1]
 * ```
 */
export const dbscan: IterativeEstimator<DbscanHyper, DbscanState, DbscanParams, DensityVerbs> = iterative<
  DbscanHyper,
  DbscanState,
  DbscanParams,
  DensityVerbs
>({
  name: 'cluster/dbscan',
  kind: 'clusterer',
  defaults: { eps: 0.5, minSamples: 5, metric: 'euclidean' },
  hyperSpace: (h) =>
    dict({
      eps: real(0.01, 5, { log: true, default: h.eps, label: 'radius' }),
      minSamples: int(1, 50, { default: h.minSamples, label: 'core threshold' }),
    }),
  algorithm: (data, h) => dbscanAlgorithm(data.x, { eps: h.eps, minSamples: h.minSamples, metric: h.metric }),
  maxSteps: (_, data) => matrixView(data.x, 'dbscan').n + 1,
  params: (s, data, h) => {
    const m = matrixView(data.x, 'dbscan')
    const labels = [...s.labels]
    const coreSampleIndices = s.core.flatMap((c, i) => (c ? [i] : []))
    return {
      labels,
      roles: rolesOf(labels, s.core),
      coreSampleIndices,
      components: rowsAt(m.data, m.d, coreSampleIndices),
      eps: h.eps,
      metric: h.metric,
    }
  },
  verbs: (p, h) => ({
    predict: labelledPredict(
      p.components,
      p.coreSampleIndices.map((i) => p.labels[i] ?? -1),
      h.eps,
      h.metric,
      p.components[0]?.length ?? 0,
    ),
  }),
})

// ---------------------------------------------------------------------------------------------------------------
// OPTICS

/** The state of OPTICS: reachability (-1: not reached yet), predecessors, the processing order so far. */
export interface OpticsState {
  readonly reachability: readonly number[]
  readonly predecessor: readonly number[]
  readonly ordering: readonly number[]
  readonly processed: readonly boolean[]
}

/** Options of OPTICS. */
export interface OpticsOptions {
  /** Rows (itself included) that make a core point (default 5). */
  readonly minSamples?: number
  /** The largest neighbourhood radius considered (default infinity). */
  readonly maxEps?: number
  readonly metric?: MetricSlot
  readonly distanceOptions?: DistanceOptions
}

/** The core distances of the rows: the distance to the `minSamples`-th nearest row (itself included), infinite past `maxEps`. */
function coreDistancesOf(dist: ArrayLike<number>, n: number, minSamples: number, maxEps: number): number[] {
  return Array.from({ length: n }, (_, i) => {
    if (minSamples > n) return Number.POSITIVE_INFINITY
    const c = kthInRow(dist, n, i, minSamples)
    return c > maxEps ? Number.POSITIVE_INFINITY : c
  })
}

/**
 * OPTICS's ordering (Ankerst et al. 1999; sklearn's `compute_optics_graph`) as a traced {@link Algorithm}: each step
 * emits the unprocessed point of smallest reachability (the first on ties) and lowers the reachability of its
 * neighbours within `maxEps`. The reachability plot grows one bar per step. It observes `reachability` (-1 where not
 * yet reached), `ordering` and `predecessor`.
 *
 * @param x - Points `[n, d]`, or the distances with `metric: 'precomputed'`.
 * @param options - Core threshold, the largest radius and the metric.
 * @returns The algorithm; n steps.
 * @example
 * ```ts
 * import { opticsAlgorithm } from 'mdml-engine/cluster'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const out = runAlgorithm(opticsAlgorithm([[0], [1], [5]], { minSamples: 2 }), { maxSteps: 3 })
 * out.state.ordering // => [0, 1, 2]
 * ```
 */
export function opticsAlgorithm(x: TensorLike, options: OpticsOptions = {}): Algorithm<OpticsState> {
  const minSamples = options.minSamples ?? 5
  const maxEps = options.maxEps ?? Number.POSITIVE_INFINITY
  const { n, data } = distanceMatrix(x, options.metric ?? 'euclidean', options.distanceOptions)
  if (!(Number.isInteger(minSamples) && minSamples >= 1 && minSamples <= n))
    throw new MdmlError('value', `opticsAlgorithm: minSamples must be in 1..${n}, got ${minSamples}`)
  const core = coreDistancesOf(data, n, minSamples, maxEps)
  const reach = (s: OpticsState, i: number): number =>
    (s.reachability[i] ?? -1) < 0 ? Number.POSITIVE_INFINITY : (s.reachability[i] as number)
  return {
    name: 'optics',
    config: { minSamples, maxEps: Number.isFinite(maxEps) ? maxEps : null, n },
    schema: channels({
      metrics: { reachability: { description: 'the reachability of the point just emitted (-1: none)' } },
      tensors: {
        reachability: { role: 'state', description: 'reachability of each row so far (-1: not reached)', shape: ['n'] },
        ordering: { role: 'state', description: 'the rows in the order emitted', shape: ['t'] },
      },
    }),
    init: () => ({
      reachability: new Array<number>(n).fill(-1),
      predecessor: new Array<number>(n).fill(-1),
      ordering: [],
      processed: new Array<boolean>(n).fill(false),
    }),
    step: (s) => {
      let p = -1
      for (let i = 0; i < n; i++) if (s.processed[i] !== true && (p < 0 || reach(s, i) < reach(s, p))) p = i
      const reachability = [...s.reachability]
      const predecessor = [...s.predecessor]
      const processed = [...s.processed]
      processed[p] = true
      const cd = core[p] ?? Number.POSITIVE_INFINITY
      if (Number.isFinite(cd)) {
        for (let q = 0; q < n; q++) {
          const dpq = data[p * n + q] ?? 0
          if (processed[q] === true || dpq > maxEps) continue
          const r = Math.max(cd, dpq)
          if (r < reach({ ...s, reachability }, q)) {
            reachability[q] = r
            predecessor[q] = p
          }
        }
      }
      return { reachability, predecessor, ordering: [...s.ordering, p], processed }
    },
    done: (s) => (s.ordering.length >= n ? 'completed' : undefined),
    observe: (s) => {
      const last = s.ordering.at(-1)
      return {
        metrics: { reachability: last === undefined ? -1 : (s.reachability[last] ?? -1) },
        tensors: {
          reachability: { role: 'state', value: s.reachability, shape: [n] },
          ordering: { role: 'state', value: s.ordering, shape: [s.ordering.length] },
        },
      }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as OpticsState,
  }
}

/** The OPTICS ordering: reachability and core distances by row (infinite where undefined), the order and predecessors. */
export interface OpticsOrdering {
  readonly reachability: number[]
  readonly coreDistances: number[]
  readonly ordering: number[]
  readonly predecessor: number[]
}

/**
 * The OPTICS ordering run to the end (sklearn's `compute_optics_graph`), with the core distances.
 *
 * @param x - Points `[n, d]`.
 * @param options - Core threshold, the largest radius and the metric.
 * @returns The ordering, with infinite reachability where a point was never reached.
 * @example
 * ```ts
 * import { opticsOrdering } from 'mdml-engine/cluster'
 * opticsOrdering([[0], [1], [5]], { minSamples: 2 }).reachability // => [Infinity, 1, 4]
 * ```
 */
export function opticsOrdering(x: TensorLike, options: OpticsOptions = {}): OpticsOrdering {
  const { n, data } = distanceMatrix(x, options.metric ?? 'euclidean', options.distanceOptions)
  const s = runAlgorithm(opticsAlgorithm(x, options), { maxSteps: n }).state
  const coreDistances = coreDistancesOf(data, n, options.minSamples ?? 5, options.maxEps ?? Number.POSITIVE_INFINITY)
  return {
    reachability: s.reachability.map((r) => (r < 0 ? Number.POSITIVE_INFINITY : r)),
    coreDistances,
    ordering: [...s.ordering],
    predecessor: [...s.predecessor],
  }
}

/**
 * DBSCAN-equivalent clusters from an OPTICS ordering at radius eps (sklearn's `cluster_optics_dbscan`).
 *
 * @param o - The ordering.
 * @param eps - The radius (at most the ordering's maxEps).
 * @returns A label per row (-1 for noise).
 * @example
 * ```ts
 * import { opticsClustersAt, opticsOrdering } from 'mdml-engine/cluster'
 * opticsClustersAt(opticsOrdering([[0], [0.5], [9]], { minSamples: 2 }), 1) // => [0, 0, -1]
 * ```
 */
export function opticsClustersAt(o: OpticsOrdering, eps: number): number[] {
  const n = o.ordering.length
  const labels = new Array<number>(n).fill(0)
  let c = -1
  for (const p of o.ordering) {
    const far = (o.reachability[p] ?? 0) > eps
    const near = (o.coreDistances[p] ?? 0) <= eps
    if (far && near) c++
    labels[p] = c
  }
  for (let p = 0; p < n; p++) if ((o.reachability[p] ?? 0) > eps && !((o.coreDistances[p] ?? 0) <= eps)) labels[p] = -1
  return labels
}

/** Hyperparameters of {@link optics}. */
export interface OpticsHyper {
  readonly minSamples: number
  /** The largest radius considered (`null`: unbounded). */
  readonly maxEps: number | null
  /** The radius of the DBSCAN-style extraction. */
  readonly eps: number
  readonly metric: DistanceName | DistanceAlias
}

/** The fitted parameters of {@link optics}. */
export type OpticsParams = {
  readonly labels: number[]
  /** Reachability in row order (-1 where never reached). */
  readonly reachability: number[]
  readonly coreDistances: number[]
  readonly ordering: number[]
  readonly components: number[][]
  readonly componentLabels: number[]
  readonly eps: number
  readonly metric: DistanceName | DistanceAlias
}

const finiteOr = (v: number): number => (Number.isFinite(v) ? v : -1)

/**
 * OPTICS (sklearn.cluster.OPTICS with the DBSCAN extraction) on the iterative base: one point emitted per step; the
 * fitted value extracts clusters at `eps` from the ordering so far. TODO: the xi extraction.
 *
 * @example
 * ```ts
 * import { optics } from 'mdml-engine/cluster'
 * optics.with({ minSamples: 2, eps: 1 }).fit({ x: [[0], [0.5], [9]] }).params.labels // => [0, 0, -1]
 * ```
 */
export const optics: IterativeEstimator<OpticsHyper, OpticsState, OpticsParams, DensityVerbs> = iterative<
  OpticsHyper,
  OpticsState,
  OpticsParams,
  DensityVerbs
>({
  name: 'cluster/optics',
  kind: 'clusterer',
  defaults: { minSamples: 5, maxEps: null, eps: 0.5, metric: 'euclidean' },
  hyperSpace: (h) =>
    dict({
      minSamples: int(2, 50, { default: h.minSamples, label: 'core threshold' }),
      eps: real(0.01, 5, { log: true, default: h.eps, label: 'extraction radius' }),
    }),
  algorithm: (data, h) =>
    opticsAlgorithm(data.x, {
      minSamples: h.minSamples,
      metric: h.metric,
      ...(h.maxEps === null ? {} : { maxEps: h.maxEps }),
    }),
  maxSteps: (_, data) => matrixView(data.x, 'optics').n,
  params: (s, data, h) => {
    const m = matrixView(data.x, 'optics')
    const { data: dist, n } = distanceMatrix(data.x, h.metric)
    const maxEps = h.maxEps ?? Number.POSITIVE_INFINITY
    const coreDistances = coreDistancesOf(dist, n, h.minSamples, maxEps)
    // Rows not yet emitted count as unreachable and unprocessed: the extraction sees the ordering so far.
    const ordering = [...s.ordering]
    const reachability = s.reachability.map((r) => (r < 0 ? Number.POSITIVE_INFINITY : r))
    const labels = opticsClustersAt({ reachability, coreDistances, ordering, predecessor: [...s.predecessor] }, h.eps)
    const emitted = new Set(ordering)
    for (let i = 0; i < n; i++) if (!emitted.has(i)) labels[i] = -1
    const cores = coreDistances.flatMap((c, i) => (c <= h.eps && emitted.has(i) ? [i] : []))
    return {
      labels,
      reachability: reachability.map(finiteOr),
      coreDistances: coreDistances.map(finiteOr),
      ordering,
      components: rowsAt(m.data, m.d, cores),
      componentLabels: cores.map((i) => labels[i] ?? -1),
      eps: h.eps,
      metric: h.metric,
    }
  },
  verbs: (p) => ({
    predict: labelledPredict(p.components, p.componentLabels, p.eps, p.metric, p.components[0]?.length ?? 0),
  }),
})

// ---------------------------------------------------------------------------------------------------------------
// HDBSCAN

/** One row of HDBSCAN's condensed tree: a parent cluster, a child (a point below n, a cluster above), lambda, size. */
export type CondensedRow = [parent: number, child: number, lambda: number, size: number]

/** The edges of the complete graph on n points weighted by mutual reachability, in the order Kruskal takes them. */
interface MutualReachabilityEdges {
  readonly a: Int32Array
  readonly b: Int32Array
  readonly weight: Float64Array
  /** Edge indices by weight, ties in input order. */
  readonly order: Int32Array
}

function mutualReachabilityEdges(
  dist: ArrayLike<number>,
  core: readonly number[],
  alpha: number,
): MutualReachabilityEdges {
  const n = core.length
  const m = (n * (n - 1)) / 2
  const a = new Int32Array(m)
  const b = new Int32Array(m)
  const weight = new Float64Array(m)
  let k = 0
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      a[k] = i
      b[k] = j
      weight[k] = Math.max(core[i] ?? 0, core[j] ?? 0, (dist[i * n + j] ?? 0) / alpha)
      k++
    }
  }
  const order = Int32Array.from({ length: m }, (_, i) => i).sort((p, q) => (weight[p] ?? 0) - (weight[q] ?? 0) || p - q)
  return { a, b, weight, order }
}

/** What the single-linkage phase carries: the edge cursor, the tree node each point is in now, and the merges so far. */
interface LinkageProgress {
  readonly cursor: number
  readonly component: readonly number[]
  readonly merges: readonly Merge[]
}

/** Take edges in order until one joins two components; that merge is the next row of the linkage. */
function linkageNext(p: LinkageProgress, e: MutualReachabilityEdges, n: number): LinkageProgress {
  const sizeOf = (id: number): number => (id < n ? 1 : (p.merges[id - n]?.[3] ?? 1))
  let cursor = p.cursor
  while (cursor < e.order.length) {
    const k = e.order[cursor] as number
    cursor++
    const ia = p.component[e.a[k] as number] ?? 0
    const ib = p.component[e.b[k] as number] ?? 0
    if (ia === ib) continue
    const id = n + p.merges.length
    return {
      cursor,
      component: p.component.map((c) => (c === ia || c === ib ? id : c)),
      merges: [...p.merges, [Math.min(ia, ib), Math.max(ia, ib), e.weight[k] ?? 0, sizeOf(ia) + sizeOf(ib)]],
    }
  }
  return { ...p, cursor }
}

/**
 * The single-linkage tree of the mutual reachability distance max(core(a), core(b), d(a, b) / alpha), as scipy linkage
 * rows.
 *
 * @param dist - Pairwise distances `[n, n]`, flat row-major.
 * @param core - Core distances.
 * @param alpha - The distance scaling (default 1).
 * @returns The n - 1 merges.
 * @example
 * ```ts
 * import { mutualReachabilityTree } from 'mdml-engine/cluster'
 * mutualReachabilityTree([0, 1, 1, 0], [0, 2]) // => [[0, 1, 2, 2]]
 * ```
 */
export function mutualReachabilityTree(dist: ArrayLike<number>, core: readonly number[], alpha = 1): Merge[] {
  const n = core.length
  const edges = mutualReachabilityEdges(dist, core, alpha)
  let p: LinkageProgress = { cursor: 0, component: Array.from({ length: n }, (_, i) => i), merges: [] }
  while (p.merges.length < n - 1 && p.cursor < edges.order.length) p = linkageNext(p, edges, n)
  return [...p.merges]
}

/** What the condensing phase carries: the queue of tree nodes to visit, the cluster id of each, and the rows so far. */
interface CondenseProgress {
  readonly queue: readonly number[]
  readonly head: number
  /** The condensed cluster id of a tree node (-1: none). */
  readonly relabel: readonly number[]
  readonly nextId: number
  readonly rows: readonly CondensedRow[]
}

function condenseStart(n: number): CondenseProgress {
  const relabel = new Array<number>(Math.max(2 * n - 1, 1)).fill(-1)
  const root = 2 * n - 2
  relabel[root] = n
  return { queue: [root], head: 0, relabel, nextId: n + 1, rows: [] }
}

/** The index in the queue of the next internal node to visit (past the leaves), or the queue's length when none is left. */
function condenseHead(p: CondenseProgress, n: number): number {
  let h = p.head
  while (h < p.queue.length && (p.queue[h] as number) < n) h++
  return h
}

/** Visit one internal node: the walk down from the root, one split or one shrinking per step. */
function condenseNode(
  p: CondenseProgress,
  merges: readonly Merge[],
  n: number,
  minClusterSize: number,
): { readonly progress: CondenseProgress; readonly split: boolean; readonly lambda: number } {
  const node = p.queue[condenseHead(p, n)] as number
  const head = condenseHead(p, n) + 1
  const sizeOf = (id: number): number => (id < n ? 1 : (merges[id - n]?.[3] ?? 1))
  const leaves = (id: number): number[] => {
    const out: number[] = []
    const stack = [id]
    while (stack.length > 0) {
      const v = stack.pop() as number
      if (v < n) out.push(v)
      else {
        const m = merges[v - n] as Merge
        stack.push(m[1], m[0])
      }
    }
    return out
  }
  const [left, right, d] = merges[node - n] as Merge
  const lambda = d > 0 ? 1 / d : Number.POSITIVE_INFINITY
  const parent = p.relabel[node] as number
  const big = (c: number): boolean => sizeOf(c) >= minClusterSize
  const relabel = [...p.relabel]
  const queue = [...p.queue]
  const rows = [...p.rows]
  let nextId = p.nextId
  const split = big(left) && big(right)
  if (split) {
    for (const c of [left, right]) {
      relabel[c] = nextId
      rows.push([parent, nextId, lambda, sizeOf(c)])
      nextId++
      queue.push(c)
    }
  } else {
    for (const c of [left, right]) {
      if (big(c)) {
        relabel[c] = parent
        queue.push(c)
      } else for (const q of leaves(c)) rows.push([parent, q, lambda, 1])
    }
  }
  return { progress: { queue, head, relabel, nextId, rows }, split, lambda }
}

/**
 * Condense a single-linkage tree (Campello et al. 2013; hdbscan's `condense_tree`): walking down from the root, a split
 * where both sides have at least `minClusterSize` points makes two new clusters; smaller sides fall out as points at
 * lambda = 1 / distance.
 *
 * @param merges - The linkage rows.
 * @param minClusterSize - The smallest cluster.
 * @returns The condensed tree rows `[parent, child, lambda, size]` (cluster ids from n, the root being n).
 * @example
 * ```ts
 * import { condenseTree } from 'mdml-engine/cluster'
 * condenseTree([[0, 1, 1, 2], [2, 3, 2, 3]], 2).length // => 3
 * ```
 */
export function condenseTree(merges: readonly Merge[], minClusterSize: number): CondensedRow[] {
  const n = merges.length + 1
  let p = condenseStart(n)
  while (condenseHead(p, n) < p.queue.length) p = condenseNode(p, merges, n, minClusterSize).progress
  return [...p.rows]
}

/** What the excess-of-mass selection of clusters from a condensed tree gives. */
export interface ClusterSelection {
  /** A cluster index per point (-1: noise). */
  readonly labels: number[]
  /** Membership strength in `[0, 1]` (0 for noise). */
  readonly probabilities: number[]
  /** The ids of the selected clusters in the condensed tree. */
  readonly selected: number[]
  /** The stability of every cluster of the condensed tree, by id. */
  readonly stability: Map<number, number>
}

/** What the selection phase carries: per cluster (index = id - n) its stability, children, running total and flag. */
interface SelectProgress {
  readonly stability: readonly number[]
  readonly children: readonly (readonly number[])[]
  readonly total: readonly number[]
  /** 1 for a cluster currently selected. */
  readonly chosen: readonly number[]
  /** How many clusters have been visited (they are taken from the largest id down). */
  readonly visited: number
}

function selectStart(rows: readonly CondensedRow[], n: number): SelectProgress {
  const count = 1 + rows.reduce((k, r) => k + (r[1] >= n ? 1 : 0), 0)
  const birth = new Array<number>(count).fill(0)
  const children: number[][] = Array.from({ length: count }, () => [])
  for (const [p, c, l] of rows) {
    if (c >= n) {
      birth[c - n] = l
      children[p - n]?.push(c)
    }
  }
  const stability = new Array<number>(count).fill(0)
  for (const [p, , l, s] of rows) {
    const b = birth[p - n] ?? 0
    const lam = Number.isFinite(l) ? l : b
    stability[p - n] = (stability[p - n] ?? 0) + (lam - b) * s
  }
  return {
    stability,
    children,
    total: new Array<number>(count).fill(0),
    chosen: new Array<number>(count).fill(0),
    visited: 0,
  }
}

/** Visit the next cluster from the largest id down: keep it or its descendants (excess of mass), or keep leaves. */
function selectNode(
  p: SelectProgress,
  n: number,
  rule: ClusterSelectionRule,
  allowSingleCluster: boolean,
): { readonly progress: SelectProgress; readonly cluster: number } {
  const idx = p.stability.length - 1 - p.visited
  const c = n + idx
  const kids = p.children[idx] ?? []
  const own = p.stability[idx] ?? 0
  const total = [...p.total]
  const chosen = [...p.chosen]
  if (rule === 'leaf') {
    if (kids.length === 0 && (c !== n || allowSingleCluster)) chosen[idx] = 1
  } else {
    const sub = kids.reduce((a, k) => a + (total[k - n] ?? 0), 0)
    if (kids.length > 0 && sub > own) total[idx] = sub
    else if (c === n && !allowSingleCluster) total[idx] = sub
    else {
      total[idx] = own
      chosen[idx] = 1
      const stack = [...kids]
      while (stack.length > 0) {
        const k = stack.pop() as number
        chosen[k - n] = 0
        stack.push(...(p.children[k - n] ?? []))
      }
    }
  }
  return { progress: { ...p, total, chosen, visited: p.visited + 1 }, cluster: c }
}

/** The labels, strengths and stabilities of a selection: points take the selected cluster above them, if any. */
function selectFinal(rows: readonly CondensedRow[], n: number, p: SelectProgress): ClusterSelection {
  const stability = new Map<number, number>()
  for (const [parent] of rows) if (!stability.has(parent)) stability.set(parent, p.stability[parent - n] ?? 0)
  const parentOf = new Map<number, number>()
  const lambdaOf = new Map<number, number>()
  for (const [parent, c, l] of rows) {
    parentOf.set(c, parent)
    lambdaOf.set(c, l)
  }
  const chosen = p.chosen.flatMap((flag, i) => (flag === 1 ? [n + i] : []))
  const index = new Map(chosen.map((c, i) => [c, i]))
  const clusterOf = (i: number): number => {
    let c = parentOf.get(i)
    while (c !== undefined) {
      const l = index.get(c)
      if (l !== undefined) return l
      c = parentOf.get(c)
    }
    return -1
  }
  const labels = Array.from({ length: n }, (_, i) => clusterOf(i))
  // The largest finite lambda among the points of each selected cluster scales the strength of membership.
  const maxLambda = new Array<number>(chosen.length).fill(0)
  for (let i = 0; i < n; i++) {
    const l = lambdaOf.get(i) ?? 0
    if (labels[i] !== undefined && (labels[i] as number) >= 0 && Number.isFinite(l))
      maxLambda[labels[i] as number] = Math.max(maxLambda[labels[i] as number] ?? 0, l)
  }
  const probabilities = labels.map((c, i) => {
    if (c < 0) return 0
    const top = maxLambda[c] ?? 0
    const l = lambdaOf.get(i) ?? 0
    return top > 0 ? Math.min(Number.isFinite(l) ? l : top, top) / top : 1
  })
  return { labels, probabilities, selected: chosen, stability }
}

/**
 * The selection of flat clusters from a condensed tree, with the labels, membership strengths and stabilities it gives:
 * by excess of mass (the default) or as the leaves of the tree.
 *
 * @param rows - The condensed tree.
 * @param n - The number of points.
 * @param allowSingleCluster - Whether the root may be selected (default false).
 * @param rule - `'eom'` (default) or `'leaf'`.
 * @returns Labels (-1 for noise), probabilities, the selected cluster ids and the stability of each cluster.
 * @example
 * ```ts
 * import { condenseTree, selectClusters } from 'mdml-engine/cluster'
 * const rows = condenseTree([[0, 1, 1, 2], [2, 3, 1, 2], [4, 5, 9, 4]], 2)
 * selectClusters(rows, 4).labels // => [0, 0, 1, 1]
 * ```
 */
export function selectClusters(
  rows: readonly CondensedRow[],
  n: number,
  allowSingleCluster = false,
  rule: ClusterSelectionRule = 'eom',
): ClusterSelection {
  let p = selectStart(rows, n)
  while (p.visited < p.stability.length) p = selectNode(p, n, rule, allowSingleCluster).progress
  return selectFinal(rows, n, p)
}

/** Options of {@link hdbscan}. */
export interface HdbscanOptions {
  /** The smallest cluster (default 5). */
  readonly minClusterSize?: number
  /** The core-distance neighbour count, itself included (default `minClusterSize`). */
  readonly minSamples?: number
  readonly allowSingleCluster?: boolean
  readonly metric?: MetricSlot
  readonly distanceOptions?: DistanceOptions
  /** How clusters are chosen from the condensed tree, a value of `hdbscanSelections.space` (default excess of mass). */
  readonly selection?: SpaceValue
  /** Distances are divided by this before mutual reachability (default 1; sklearn's `alpha`). */
  readonly alpha?: number
}

/** The result of {@link hdbscan}: the clusters, and the hierarchy they were cut from. */
export interface HdbscanResult {
  readonly labels: number[]
  readonly probabilities: number[]
  readonly coreDistances: number[]
  /** The single-linkage tree of the mutual reachability distance (scipy linkage rows). */
  readonly linkage: Merge[]
  /** The condensed tree: `[parent, child, lambda, size]`. */
  readonly condensedTree: CondensedRow[]
  /** The ids of the clusters selected from the condensed tree. */
  readonly selected: number[]
  /** The stability of each cluster of the condensed tree as `[id, stability]`. */
  readonly stability: [number, number][]
}

/** The phases of {@link hdbscanAlgorithm}, in order. */
export type HdbscanPhase = 'tree' | 'condense' | 'select' | 'label' | 'done'

/**
 * The state of {@link hdbscanAlgorithm}: the phase and what each phase has built so far. The single-linkage tree grows
 * one merge per step; the condensed tree one visited tree node per step; the selection one cluster per step; a last
 * step labels the points. A state is never changed in place.
 */
export interface HdbscanState {
  readonly phase: HdbscanPhase
  /** The core distance of every point. */
  readonly core: readonly number[]
  /** The edges of the mutual reachability graph taken so far (the cursor into the sorted edge list). */
  readonly cursor: number
  /** The tree node each point is in now. */
  readonly component: readonly number[]
  /** The linkage rows so far. */
  readonly merges: readonly Merge[]
  readonly queue: readonly number[]
  readonly head: number
  readonly relabel: readonly number[]
  readonly nextId: number
  /** The condensed tree so far. */
  readonly rows: readonly CondensedRow[]
  /** Per condensed cluster (index = id - n): its stability, once the selection has started (else empty). */
  readonly stability: readonly number[]
  readonly children: readonly (readonly number[])[]
  readonly total: readonly number[]
  readonly chosen: readonly number[]
  readonly visited: number
  /** The cluster of each point (-1: noise) and its membership strength, once labelled (else empty). */
  readonly labels: readonly number[]
  readonly probabilities: readonly number[]
  /** The step's own bookkeeping for the log: the last merge's height, whether the last node split, its lambda. */
  readonly lastHeight: number
  readonly lastSplit: boolean
  readonly lastLambda: number
}

/** The paths {@link hdbscanAlgorithm} logs. */
export type HdbscanPath =
  | 'hdbscan/phase'
  | 'tree/merge'
  | 'tree/height'
  | 'tree/size'
  | 'condense/split'
  | 'condense/lambda'
  | 'condense/clusters'
  | 'select/cluster'
  | 'select/stability'
  | 'select/selected'
  | 'labels/labels'
  | 'labels/strength'
  | 'labels/clusters'
  | 'labels/noise'

/** What a stepped HDBSCAN logs, by phase: the tree growing, the condensing, the selection and the labels. */
export const hdbscanContract: Contract<unknown, HdbscanPath> = defineContract({
  'hdbscan/phase': scalar({ description: 'the phase: 0 tree, 1 condense, 2 select, 3 label, 4 done' }),
  'tree/merge': scalar({ description: 'merges of the single-linkage tree so far' }),
  'tree/height': scalar({ description: 'the mutual reachability distance of the merge just made' }),
  'tree/size': scalar({ description: 'the size of the cluster the merge just made' }),
  'condense/split': scalar({ description: '1 when the tree node just visited split into two clusters, else 0' }),
  'condense/lambda': scalar({ description: '1 / distance at the tree node just visited (-1 when the distance is 0)' }),
  'condense/clusters': scalar({ description: 'clusters of the condensed tree so far' }),
  'select/cluster': scalar({ description: 'the id of the condensed cluster just visited' }),
  'select/stability': scalar({ description: 'the stability of that cluster' }),
  'select/selected': scalar({ description: 'clusters selected so far' }),
  'labels/labels': tensor({ role: 'output', shape: ['n'], description: 'the cluster of each point (-1: noise)' }),
  'labels/strength': tensor({ role: 'output', shape: ['n'], description: 'the membership strength of each point' }),
  'labels/clusters': scalar({ description: 'the number of clusters' }),
  'labels/noise': scalar({ description: 'the number of noise points' }),
})

const PHASE_INDEX: Readonly<Record<HdbscanPhase, number>> = { tree: 0, condense: 1, select: 2, label: 3, done: 4 }

const encodeLambda = (l: number): number | string => (Number.isFinite(l) ? l : 'Infinity')

/**
 * HDBSCAN (Campello, Moulavi and Sander 2013; sklearn.cluster.HDBSCAN) as a stepped algorithm. First the
 * single-linkage tree of the mutual reachability distance grows one merge per step; then the tree is condensed one
 * visited node per step; then the clusters are chosen one per step (from the largest id down) by the selection rule;
 * a last step labels the points with their membership strengths. The state serialises, and `hdbscan` is this run to
 * the end.
 *
 * @param x - Points `[n, d]`, or the distances with `metric: 'precomputed'`.
 * @param options - Smallest cluster, core neighbour count, metric, selection rule, single-cluster flag and alpha.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a smallest cluster below 2 or a non-positive alpha.
 * @concept density-clustering
 * @example
 * ```ts
 * import { hdbscanAlgorithm } from 'mdml-engine/cluster'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0], [0.1], [0.2], [5], [5.1], [5.2]]
 * const run = runAlgorithm(hdbscanAlgorithm(x, { minClusterSize: 3 }), { maxSteps: 40 })
 * [run.status, run.state.labels] // => ['converged', [0, 0, 0, 1, 1, 1]]
 * ```
 */
export function hdbscanAlgorithm(x: TensorLike, options: HdbscanOptions = {}): Algorithm<HdbscanState, HdbscanPath> {
  const { n, data } = distanceMatrix(x, options.metric ?? 'euclidean', options.distanceOptions)
  const mcs = options.minClusterSize ?? 5
  if (!(Number.isInteger(mcs) && mcs >= 2))
    throw new MdmlError('value', `hdbscan: minClusterSize must be >= 2, got ${mcs}`)
  const alpha = options.alpha ?? 1
  if (!(alpha > 0)) throw new MdmlError('value', `hdbscan: alpha must be positive, got ${alpha}`)
  const k = Math.min(options.minSamples ?? mcs, n)
  const rule = hdbscanSelections.make(options.selection)
  const single = options.allowSingleCluster ?? false
  const core = Array.from({ length: n }, (_, i) => kthInRow(data, n, i, k))
  const edges = mutualReachabilityEdges(data, core, alpha)
  const base: HdbscanState = {
    phase: n > 1 ? 'tree' : 'condense',
    core,
    cursor: 0,
    component: Array.from({ length: n }, (_, i) => i),
    merges: [],
    ...condenseStart(n),
    stability: [],
    children: [],
    total: [],
    chosen: [],
    visited: 0,
    labels: [],
    probabilities: [],
    lastHeight: 0,
    lastSplit: false,
    lastLambda: 0,
  }
  /** Move to the next phase for as long as the current one has nothing left to do. */
  const settle = (s: HdbscanState): HdbscanState => {
    let t = s
    for (;;) {
      if (t.phase === 'tree' && t.merges.length >= n - 1) t = { ...t, phase: 'condense' }
      else if (t.phase === 'condense' && condenseHead(t, n) >= t.queue.length) {
        const sel = selectStart(t.rows, n)
        t = { ...t, ...sel, phase: 'select' }
      } else if (t.phase === 'select' && t.visited >= t.stability.length) t = { ...t, phase: 'label' }
      else return t
    }
  }
  const labelled = (s: HdbscanState): ClusterSelection =>
    selectFinal(s.rows, n, {
      stability: s.stability,
      children: s.children,
      total: s.total,
      chosen: s.chosen,
      visited: s.visited,
    })
  return {
    name: 'cluster/hdbscan',
    config: { n, minClusterSize: mcs, minSamples: k, alpha, selection: rule, allowSingleCluster: single },
    schema: channels({
      metrics: {
        merges: { description: 'merges of the single-linkage tree' },
        clusters: { description: 'clusters of the condensed tree' },
        selected: { description: 'clusters selected so far' },
      },
    }),
    init: () => settle(base),
    step: (s, ctx) => {
      let t: HdbscanState = s
      if (s.phase === 'tree') {
        const next = linkageNext({ cursor: s.cursor, component: s.component, merges: s.merges }, edges, n)
        const last = next.merges.at(-1)
        t = { ...s, cursor: next.cursor, component: next.component, merges: next.merges, lastHeight: last?.[2] ?? 0 }
        ctx.log('tree/merge', next.merges.length)
        ctx.log('tree/height', last?.[2] ?? 0)
        ctx.log('tree/size', last?.[3] ?? 0)
      } else if (s.phase === 'condense') {
        const r = condenseNode(s, s.merges, n, mcs)
        t = { ...s, ...r.progress, lastSplit: r.split, lastLambda: r.lambda }
        ctx.log('condense/split', r.split ? 1 : 0)
        ctx.log('condense/lambda', Number.isFinite(r.lambda) ? r.lambda : -1)
        ctx.log('condense/clusters', r.progress.nextId - n)
      } else if (s.phase === 'select') {
        const r = selectNode(s, n, rule, single)
        t = { ...s, ...r.progress }
        ctx.log('select/cluster', r.cluster)
        ctx.log('select/stability', s.stability[r.cluster - n] ?? 0)
        ctx.log(
          'select/selected',
          r.progress.chosen.reduce((a, f) => a + f, 0),
        )
      } else if (s.phase === 'label') {
        const sel = labelled(s)
        t = { ...s, labels: sel.labels, probabilities: sel.probabilities, phase: 'done' }
        const noise = sel.labels.filter((l) => l < 0).length
        if (ctx.wants('labels/labels')) ctx.log('labels/labels', Int32Array.from(sel.labels))
        if (ctx.wants('labels/strength')) ctx.log('labels/strength', Float64Array.from(sel.probabilities))
        ctx.log('labels/clusters', sel.selected.length)
        ctx.log('labels/noise', noise)
      }
      ctx.log('hdbscan/phase', PHASE_INDEX[t.phase])
      return settle(t)
    },
    done: (s) => (s.phase === 'done' ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        merges: s.merges.length,
        clusters: s.nextId - n,
        selected: s.chosen.reduce((a, f) => a + f, 0),
      },
    }),
    serialise: (s) => stateToJson({ ...s, rows: s.rows.map((r) => [r[0], r[1], encodeLambda(r[2]), r[3]]) }),
    deserialise: (j) => {
      const s = stateFromJson(j) as Omit<HdbscanState, 'rows'> & {
        readonly rows: readonly [number, number, number | string, number][]
      }
      return { ...s, rows: s.rows.map((r): CondensedRow => [r[0], r[1], Number(r[2]), r[3]]) }
    },
  }
}

/**
 * HDBSCAN (Campello, Moulavi and Sander 2013; sklearn.cluster.HDBSCAN): core distances, the single-linkage tree of the
 * mutual reachability distance, its condensed tree and the selection of clusters (excess of mass by default). This is
 * {@link hdbscanAlgorithm} run to the end.
 *
 * @param x - Points `[n, d]`, or the distances with `metric: 'precomputed'`.
 * @param options - The smallest cluster, the core neighbour count, whether one cluster may be returned, the metric,
 *   the selection rule and alpha.
 * @returns The labels (-1: noise), membership strengths, the tree, the condensed tree and the stabilities.
 * @example
 * ```ts
 * import { hdbscan } from 'mdml-engine/cluster'
 * const x = [[0], [0.1], [0.2], [0.3], [5], [5.1], [5.2], [5.3]]
 * hdbscan(x, { minClusterSize: 3 }).labels // => [0, 0, 0, 0, 1, 1, 1, 1]
 * ```
 */
export function hdbscan(x: TensorLike, options: HdbscanOptions = {}): HdbscanResult {
  const alg = hdbscanAlgorithm(x, options)
  const n = (alg.config as { readonly n: number }).n
  const s = runAlgorithm(alg, { maxSteps: 3 * n + 5 }).state
  const sel = selectFinal(s.rows, n, {
    stability: s.stability,
    children: s.children,
    total: s.total,
    chosen: s.chosen,
    visited: s.visited,
  })
  return {
    labels: [...s.labels],
    probabilities: [...s.probabilities],
    coreDistances: [...s.core],
    linkage: s.merges.map((m): Merge => [...m]),
    condensedTree: s.rows.map(([p, c, l, size]): CondensedRow => [p, c, Number.isFinite(l) ? l : -1, size]),
    selected: sel.selected,
    stability: [...sel.stability.entries()],
  }
}

/** Hyperparameters of {@link hdbscanClustering}. */
export interface HdbscanHyper {
  readonly minClusterSize: number
  /** The core neighbour count (`null`: `minClusterSize`). */
  readonly minSamples: number | null
  readonly allowSingleCluster: boolean
  readonly metric: DistanceName | DistanceAlias
  /** How clusters are chosen from the condensed tree: `'eom'` or `'leaf'`. */
  readonly selection: ClusterSelectionRule
  /** Distances are divided by this before mutual reachability. */
  readonly alpha: number
}

/** The fitted parameters of {@link hdbscanClustering}. */
export type HdbscanParams = {
  readonly labels: number[]
  readonly probabilities: number[]
  readonly coreDistances: number[]
  readonly linkage: Merge[]
  readonly condensedTree: CondensedRow[]
  readonly components: number[][]
  readonly componentLabels: number[]
  readonly reach: number
  readonly metric: DistanceName | DistanceAlias
}

/**
 * HDBSCAN on the iterative base: the phases of {@link hdbscanAlgorithm} are the steps, so the fitted value at step `t`
 * is the clustering with a partial tree (no labels until the last step). New rows join the cluster of the nearest member
 * within the largest core distance of the clustered rows (else noise).
 *
 * @example
 * ```ts
 * import { hdbscanEstimator } from 'mdml-engine/cluster'
 * const x = [[0], [0.1], [0.2], [5], [5.1], [5.2]]
 * hdbscanEstimator.with({ minClusterSize: 3 }).fit({ x }).params.labels // => [0, 0, 0, 1, 1, 1]
 * ```
 */
export const hdbscanEstimator: IterativeEstimator<
  HdbscanHyper,
  HdbscanState,
  HdbscanParams,
  DensityVerbs,
  HdbscanPath
> = iterative<HdbscanHyper, HdbscanState, HdbscanParams, DensityVerbs, HdbscanPath>({
  name: 'cluster/hdbscan',
  kind: 'clusterer',
  defaults: {
    minClusterSize: 5,
    minSamples: null,
    allowSingleCluster: false,
    metric: 'euclidean',
    selection: 'eom',
    alpha: 1,
  },
  hyperSpace: (h) =>
    dict({
      minClusterSize: int(2, 100, { default: h.minClusterSize, label: 'smallest cluster' }),
      ...(h.minSamples === null
        ? {}
        : { minSamples: int(1, 100, { default: h.minSamples, label: 'core neighbours' }) }),
      selection: choice(hdbscanSelections.names as ClusterSelectionRule[], {
        default: h.selection,
        label: 'cluster selection',
      }),
      alpha: real(0.1, 5, { log: true, default: h.alpha, label: 'distance scaling' }),
    }),
  algorithm: (data, h) =>
    hdbscanAlgorithm(data.x, {
      minClusterSize: h.minClusterSize,
      ...(h.minSamples === null ? {} : { minSamples: h.minSamples }),
      allowSingleCluster: h.allowSingleCluster,
      metric: h.metric,
      selection: { kind: h.selection },
      alpha: h.alpha,
    }),
  maxSteps: (_, data) => 3 * matrixView(data.x, 'hdbscan').n + 5,
  params: (s, data, h) => {
    const m = matrixView(data.x, 'hdbscan', { minSamples: 2 })
    const labels = s.labels.length === m.n ? [...s.labels] : new Array<number>(m.n).fill(-1)
    const probabilities = s.labels.length === m.n ? [...s.probabilities] : new Array<number>(m.n).fill(0)
    const members = labels.flatMap((l, i) => (l >= 0 ? [i] : []))
    return {
      labels,
      probabilities,
      coreDistances: [...s.core],
      linkage: s.merges.map((r): Merge => [...r]),
      condensedTree: s.rows.map(([p, c, l, size]): CondensedRow => [p, c, Number.isFinite(l) ? l : -1, size]),
      components: rowsAt(m.data, m.d, members),
      componentLabels: members.map((i) => labels[i] ?? -1),
      reach: Math.max(0, maxOf(members.map((i) => s.core[i] ?? 0))),
      metric: h.metric,
    }
  },
  verbs: (p) => ({
    predict: labelledPredict(p.components, p.componentLabels, p.reach, p.metric, p.components[0]?.length ?? 0),
  }),
  contract: () => hdbscanContract as Contract<HdbscanState, HdbscanPath>,
})

/**
 * HDBSCAN as an estimator: {@link hdbscanEstimator} with the given hyperparameters.
 *
 * @param hyper - The hyperparameters.
 * @returns The estimator.
 * @index estimator HDBSCAN with its condensed tree.
 * @example
 * ```ts
 * import { hdbscanClustering } from 'mdml-engine/cluster'
 * const x = [[0], [0.1], [0.2], [5], [5.1], [5.2]]
 * hdbscanClustering({ minClusterSize: 3 }).fit({ x }).params.labels // => [0, 0, 0, 1, 1, 1]
 * ```
 */
export function hdbscanClustering(
  hyper: Partial<HdbscanHyper> = {},
): IterativeEstimator<HdbscanHyper, HdbscanState, HdbscanParams, DensityVerbs, HdbscanPath> {
  return hdbscanEstimator.with(hyper)
}

// ---------------------------------------------------------------------------------------------------------------
// Mean shift

/**
 * sklearn's `estimate_bandwidth`: the mean over rows of the distance to the k-th nearest row, k = quantile * n.
 *
 * @param x - Points `[n, d]`.
 * @param quantile - The neighbour fraction (default 0.3).
 * @returns The bandwidth.
 * @example
 * ```ts
 * import { estimateBandwidth } from 'mdml-engine/cluster'
 * estimateBandwidth([[0], [1], [2], [3]], 0.5) // => 1
 * ```
 */
export function estimateBandwidth(x: TensorLike, quantile = 0.3): number {
  const { n, data } = distanceMatrix(x)
  const k = Math.max(1, Math.floor(n * quantile))
  let total = 0
  for (let i = 0; i < n; i++) total += kthInRow(data, n, i, k)
  return total / n
}

/** The state of mean shift: every seed's position, whether it is still moving, and how many rows it last averaged. */
export interface MeanShiftState {
  readonly seeds: number[][]
  readonly moving: boolean[]
  readonly counts: number[]
}

/**
 * Mean shift (Comaniciu and Meer 2002; sklearn.cluster.MeanShift) as a traced {@link Algorithm}: each step moves every
 * still-moving seed to the flat (rows within the bandwidth) or Gaussian kernel-weighted mean of the rows around it; a
 * seed stops when it moves less than 1e-3 bandwidths. It observes the seeds' positions (`seeds`, their trajectories
 * over the tape) and how many are still moving.
 *
 * @param x - Points `[n, d]`.
 * @param seeds - Starting positions `[s, d]` (default every row).
 * @param bandwidth - The kernel radius (flat) or standard deviation (Gaussian).
 * @param kernel - 'flat' (sklearn) or 'gaussian'.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { meanShiftAlgorithm } from 'mdml-engine/cluster'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const out = runAlgorithm(meanShiftAlgorithm([[0], [1], [10]], undefined, 2), { maxSteps: 50 })
 * out.state.seeds.map((s) => s[0]) // => [0.5, 0.5, 10]
 * ```
 */
export function meanShiftAlgorithm(
  x: TensorLike,
  seeds: readonly (readonly number[])[] | undefined,
  bandwidth: number,
  kernel: 'flat' | 'gaussian' = 'flat',
): Algorithm<MeanShiftState> {
  const m = matrixView(x, 'meanShiftAlgorithm')
  const start = seeds ?? toRows(m.data, m.n, m.d)
  return {
    name: 'mean-shift',
    config: { bandwidth, kernel },
    schema: channels({
      metrics: { moving: { description: 'the seeds still moving' } },
      tensors: { seeds: { role: 'state', description: 'the position of every seed', shape: ['s', 'd'] } },
    }),
    init: () => ({ seeds: start.map((r) => [...r]), moving: start.map(() => true), counts: start.map(() => 0) }),
    step: (s) => {
      const dist = toFlat(
        pairwiseDistances(
          fromStorage(Float64Array.from(s.seeds.flat()), [s.seeds.length, m.d], 'float64'),
          fromStorage(m.data, [m.n, m.d], 'float64'),
          'euclidean',
        ),
      )
      const moving = [...s.moving]
      const counts = [...s.counts]
      const next = s.seeds.map((seed, a) => {
        if (moving[a] !== true) return seed
        const mean = new Array<number>(m.d).fill(0)
        let w = 0
        let within = 0
        for (let i = 0; i < m.n; i++) {
          const dd = dist[a * m.n + i] ?? 0
          const inside = dd <= bandwidth
          const wi = kernel === 'flat' ? (inside ? 1 : 0) : Math.exp(-0.5 * (dd / bandwidth) ** 2)
          if (inside) within++
          if (wi === 0) continue
          w += wi
          for (let q = 0; q < m.d; q++) mean[q] = (mean[q] ?? 0) + wi * (m.data[i * m.d + q] ?? 0)
        }
        counts[a] = within
        if (w === 0) {
          moving[a] = false
          return seed
        }
        const out = mean.map((v) => v / w)
        if (Math.hypot(...out.map((v, q) => v - (seed[q] ?? 0))) < 1e-3 * bandwidth) moving[a] = false
        return out
      })
      return { seeds: next, moving, counts }
    },
    done: (s) => (s.moving.some((v) => v) ? undefined : 'converged'),
    observe: (s) => ({
      metrics: { moving: s.moving.filter((v) => v).length },
      tensors: { seeds: { role: 'state', value: s.seeds.flat(), shape: [s.seeds.length, m.d] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MeanShiftState,
  }
}

/**
 * The cluster centres mean shift keeps: modes ordered by how many rows they averaged (ties by position, largest
 * first, as scikit-learn), each dropping the modes within a bandwidth of it.
 *
 * @param state - The seeds' state.
 * @param bandwidth - The bandwidth.
 * @returns The centres.
 * @example
 * ```ts
 * import { meanShiftCentres } from 'mdml-engine/cluster'
 * meanShiftCentres({ seeds: [[0], [0.1], [9]], moving: [false, false, false], counts: [2, 2, 1] }, 1) // => [[0.1], [9]]
 * ```
 */
export function meanShiftCentres(state: MeanShiftState, bandwidth: number): number[][] {
  const unique = new Map<string, { centre: number[]; count: number }>()
  for (const [a, c] of state.seeds.entries()) unique.set(c.join(','), { centre: c, count: state.counts[a] ?? 0 })
  const order = [...unique.values()].sort((u, v) => {
    if (u.count !== v.count) return v.count - u.count
    for (let q = 0; q < u.centre.length; q++)
      if (u.centre[q] !== v.centre[q]) return (v.centre[q] ?? 0) - (u.centre[q] ?? 0)
    return 0
  })
  const keep: number[][] = []
  for (const { centre } of order)
    if (keep.every((k) => Math.hypot(...k.map((v, q) => v - (centre[q] ?? 0))) > bandwidth)) keep.push(centre)
  return keep
}

/** Hyperparameters of {@link meanShift}. */
export interface MeanShiftHyper {
  /** The kernel bandwidth (`null`: {@link estimateBandwidth}). */
  readonly bandwidth: number | null
  readonly kernel: 'flat' | 'gaussian'
  readonly maxIter: number
  /** Give every row the nearest mode (`true`), or leave rows farther than a bandwidth from every mode unlabelled. */
  readonly clusterAll: boolean
}

/** The fitted parameters of {@link meanShift}. */
export type MeanShiftParams = { readonly centres: number[][]; readonly labels: number[]; readonly bandwidth: number }

/**
 * Mean shift on the iterative base: every row seeds a hill climb on the kernel density, one move per step; the modes
 * reached, merged when within a bandwidth (the most populated first), are the cluster centres.
 *
 * @example
 * ```ts
 * import { meanShift } from 'mdml-engine/cluster'
 * meanShift.with({ bandwidth: 2 }).fit({ x: [[0], [1], [10], [11]] }).params.labels // => [1, 1, 0, 0]
 * ```
 */
export const meanShift: IterativeEstimator<MeanShiftHyper, MeanShiftState, MeanShiftParams, DensityVerbs> = iterative<
  MeanShiftHyper,
  MeanShiftState,
  MeanShiftParams,
  DensityVerbs
>({
  name: 'cluster/mean-shift',
  kind: 'clusterer',
  defaults: { bandwidth: null, kernel: 'flat', maxIter: 300, clusterAll: true },
  hyperSpace: (h) => dict({ bandwidth: real(0.05, 10, { log: true, default: h.bandwidth ?? 1, label: 'bandwidth' }) }),
  algorithm: (data, h) => meanShiftAlgorithm(data.x, undefined, h.bandwidth ?? estimateBandwidth(data.x), h.kernel),
  maxSteps: (h) => h.maxIter,
  params: (s, data, h) => {
    const m = matrixView(data.x, 'meanShift')
    const bandwidth = h.bandwidth ?? estimateBandwidth(data.x)
    const centres = meanShiftCentres(s, bandwidth)
    const dist = toFlat(pairwiseDistances(fromStorage(m.data, [m.n, m.d], 'float64'), centres, 'euclidean'))
    const labels = Array.from({ length: m.n }, (_, i) => {
      let best = 0
      for (let j = 1; j < centres.length; j++)
        if ((dist[i * centres.length + j] ?? 0) < (dist[i * centres.length + best] ?? 0)) best = j
      return !h.clusterAll && (dist[i * centres.length + best] ?? 0) > bandwidth ? -1 : best
    })
    return { centres, labels, bandwidth }
  },
  verbs: (p) => ({
    predict: labelledPredict(
      p.centres,
      p.centres.map((_, j) => j),
      Number.POSITIVE_INFINITY,
      'euclidean',
      p.centres[0]?.length ?? 0,
    ),
  }),
})
