import { WhiteboxError } from '../../lib/base/index.ts'
import { clusteringSteps } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  fitRunOptions,
  matrixView,
} from '../../lib/learn/index.ts'
import { geometry } from '../../lib/numerics/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'
import type { Merge } from './hierarchical.ts'

/** A point's role in a density clustering. */
export type DensityRole = 'core' | 'border' | 'noise'

/** The fitted verbs of a density clustering: new points join the cluster of a core point within reach, else noise. */
export interface DensityVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/** Pairwise Euclidean distances, `[n, n]` row-major. */
function distances(x: TensorLike): number[] {
  return toFlat(geometry.cdist(x, x))
}

/** Labels of new points: the label of the nearest core point within `eps`, or -1. */
function coreVerbs(points: number[][], labels: readonly number[], eps: number): DensityVerbs {
  const d = points[0]?.length ?? 0
  return {
    predict: (x) => {
      const m = matrixView(x, 'predict', { nFeatures: d })
      if (points.length === 0) return fromStorage(new Int32Array(m.n).fill(-1), [m.n], 'int32')
      const dist = toFlat(geometry.cdist(x, points))
      const out = new Int32Array(m.n)
      for (let i = 0; i < m.n; i++) {
        let best = -1
        let bestD = eps
        for (let j = 0; j < points.length; j++) {
          const v = dist[i * points.length + j] ?? 0
          if (v <= bestD) {
            bestD = v
            best = labels[j] ?? -1
          }
        }
        out[i] = best
      }
      return fromStorage(out, [m.n], 'int32')
    },
    nFeaturesIn: d,
  }
}

const rowsOf = (data: Float64Array, d: number, rows: readonly number[]): number[][] =>
  rows.map((i) => Array.from(data.subarray(i * d, (i + 1) * d)))

// ---------------------------------------------------------------------------------------------------------------
// DBSCAN

/** The state of DBSCAN's cluster expansion. */
export interface DbscanState {
  readonly labels: readonly number[]
  /** The next row to try as the seed of a new cluster. */
  readonly next: number
  readonly clusters: number
  readonly history: readonly (readonly number[])[]
}

/**
 * DBSCAN's expansion as a traced algorithm: each step takes the next unlabelled core point and grows its cluster
 * through every density-reachable point (breadth first), so the recorded assignments show one cluster appearing per
 * step.
 *
 * @param neighbours - neighbours[i]: the rows within eps of row i (itself included).
 * @param core - Whether each row is a core point.
 * @returns The algorithm.
 */
export function dbscanAlgorithm(
  neighbours: readonly (readonly number[])[],
  core: readonly boolean[],
): Algorithm<DbscanState> {
  const n = core.length
  const nextCore = (labels: readonly number[], from: number): number => {
    let i = from
    while (i < n && !(core[i] === true && (labels[i] ?? -1) < 0)) i++
    return i
  }
  return {
    name: 'dbscan',
    config: { n },
    init: () => {
      const labels = new Array<number>(n).fill(-1)
      return { labels, next: nextCore(labels, 0), clusters: 0, history: [labels] }
    },
    step: (s) => {
      const labels = [...s.labels]
      const c = s.clusters
      const queue = [s.next]
      labels[s.next] = c
      while (queue.length > 0) {
        const p = queue.shift() as number
        if (core[p] !== true) continue
        for (const q of neighbours[p] ?? []) {
          if ((labels[q] ?? -1) >= 0) continue
          labels[q] = c
          queue.push(q)
        }
      }
      return { labels, next: nextCore(labels, s.next + 1), clusters: c + 1, history: [...s.history, labels] }
    },
    done: (s) => (s.next >= n ? 'completed' : undefined),
    observe: (s) => ({ metrics: { clusters: s.clusters } }),
  }
}

/** Hyperparameters of DBSCAN. */
export type DbscanHyper = {
  /** The neighbourhood radius (default 0.5). */
  readonly eps?: number
  /** Points within eps (itself included) that make a core point (default 5). */
  readonly minSamples?: number
}

/** The parameters of a fitted DBSCAN. */
export type DbscanParams = {
  readonly labels: number[]
  readonly roles: DensityRole[]
  readonly coreSampleIndices: number[]
  readonly components: number[][]
  readonly eps: number
}

/** A fitted DBSCAN. */
export type FittedDbscan = ExplainedFitted<DbscanParams, DensityVerbs>

/**
 * DBSCAN (Ester et al. 1996; sklearn.cluster.DBSCAN): core points have at least `minSamples` points within `eps`,
 * clusters are their density-connected components, the rest is noise. Its exposition is a `clusteringSteps` record
 * with one step per cluster found and the core, border and noise role of every row.
 *
 * @param hyper - The radius and the core-point threshold.
 * @returns The estimator.
 */
export function dbscan(hyper: DbscanHyper = {}): Estimator<DbscanHyper, FittedDbscan> {
  return defineExplainedEstimator<DbscanHyper, DbscanParams, DensityVerbs>(
    {
      name: 'dbscan',
      kind: 'clusterer',
      hyperSpace: (h) =>
        dict({
          eps: real(0.01, 5, { log: true, default: h.eps ?? 0.5, label: 'radius' }),
          minSamples: int(1, 50, { default: h.minSamples ?? 5, label: 'core threshold' }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'dbscan')
        const eps = h.eps ?? 0.5
        const minSamples = h.minSamples ?? 5
        const dist = distances(data.x)
        const neighbours = Array.from({ length: m.n }, (_, i) =>
          Array.from({ length: m.n }, (_, j) => j).filter((j) => (dist[i * m.n + j] ?? 0) <= eps),
        )
        const core = neighbours.map((nb) => nb.length >= minSamples)
        const out = runAlgorithm(dbscanAlgorithm(neighbours, core), {
          ...fitRunOptions(options, 'dbscan'),
          maxSteps: m.n + 1,
        })
        const labels = [...out.state.labels]
        const roles: DensityRole[] = labels.map((l, i) => (core[i] === true ? 'core' : l >= 0 ? 'border' : 'noise'))
        const coreSampleIndices = core.flatMap((c, i) => (c ? [i] : []))
        return {
          params: {
            labels,
            roles,
            coreSampleIndices,
            components: rowsOf(m.data, m.d, coreSampleIndices),
            eps,
          },
          exposition: [clusteringSteps({ method: 'dbscan', assignments: out.state.history, roles })],
        }
      },
      methods: (p) =>
        coreVerbs(
          p.components,
          p.coreSampleIndices.map((i) => p.labels[i] ?? -1),
          p.eps,
        ),
    },
    hyper,
  )
}

/** DBSCAN as a class. */
export class DBSCAN extends EstimatorClass<DbscanHyper, FittedDbscan> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link dbscan}.
   */
  constructor(hyper: DbscanHyper = {}) {
    super(dbscan(hyper))
  }
}

// ---------------------------------------------------------------------------------------------------------------
// OPTICS

/** The OPTICS ordering: reachability and core distances by row, the processing order and predecessors. */
export interface OpticsOrdering {
  readonly reachability: number[]
  readonly coreDistances: number[]
  readonly ordering: number[]
  readonly predecessor: number[]
}

/**
 * The OPTICS ordering (Ankerst et al. 1999), as sklearn's `compute_optics_graph`: core distances to the
 * `minSamples`-th nearest point (itself included), then repeatedly the unprocessed point of smallest reachability
 * (the first on ties) is emitted and its neighbours within `maxEps` have their reachability lowered.
 *
 * @param x - Points `[n, d]`.
 * @param minSamples - The core-point threshold.
 * @param maxEps - The largest neighbourhood radius considered (default infinity).
 * @returns The ordering.
 */
export function opticsOrdering(
  x: TensorLike,
  minSamples: number,
  maxEps: number = Number.POSITIVE_INFINITY,
): OpticsOrdering {
  const n = matrixView(x, 'opticsOrdering').n
  if (!(Number.isInteger(minSamples) && minSamples >= 1 && minSamples <= n))
    throw new WhiteboxError('value', `opticsOrdering: minSamples must be in 1..${n}`)
  const dist = distances(x)
  const coreDistances = Array.from({ length: n }, (_, i) => {
    const row = dist.slice(i * n, (i + 1) * n).sort((a, b) => a - b)
    const c = row[minSamples - 1] ?? Number.POSITIVE_INFINITY
    return c > maxEps ? Number.POSITIVE_INFINITY : c
  })
  const reachability = new Array<number>(n).fill(Number.POSITIVE_INFINITY)
  const predecessor = new Array<number>(n).fill(-1)
  const processed = new Array<boolean>(n).fill(false)
  const ordering: number[] = []
  for (let t = 0; t < n; t++) {
    let p = -1
    for (let i = 0; i < n; i++) if (!processed[i] && (p < 0 || (reachability[i] ?? 0) < (reachability[p] ?? 0))) p = i
    processed[p] = true
    ordering.push(p)
    const cd = coreDistances[p] ?? Number.POSITIVE_INFINITY
    if (!Number.isFinite(cd)) continue
    for (let q = 0; q < n; q++) {
      const dpq = dist[p * n + q] ?? 0
      if (processed[q] || dpq > maxEps) continue
      const r = Math.max(cd, dpq)
      if (r < (reachability[q] ?? 0)) {
        reachability[q] = r
        predecessor[q] = p
      }
    }
  }
  return { reachability, coreDistances, ordering, predecessor }
}

/**
 * DBSCAN-equivalent clusters from an OPTICS ordering at radius eps (sklearn's `cluster_optics_dbscan`).
 *
 * @param o - The ordering.
 * @param eps - The radius (at most the ordering's maxEps).
 * @returns A label per row (-1 for noise).
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

/** Hyperparameters of OPTICS. */
export type OpticsHyper = {
  readonly minSamples?: number
  readonly maxEps?: number
  /** The radius of the DBSCAN-style extraction (default: maxEps when finite, else 0.5). */
  readonly eps?: number
}

/** The parameters of a fitted OPTICS. */
export type OpticsParams = {
  readonly labels: number[]
  readonly reachability: number[]
  readonly coreDistances: number[]
  readonly ordering: number[]
  readonly components: number[][]
  readonly componentLabels: number[]
  readonly eps: number
}

/**
 * OPTICS (sklearn.cluster.OPTICS with `cluster_method='dbscan'`): the reachability ordering, and clusters extracted
 * at one radius. Its exposition is a `clusteringSteps` record with the reachability plot (reachability in processing
 * order) and the core, border and noise roles at that radius. TODO: the xi extraction.
 *
 * @param hyper - Core threshold, the largest radius and the extraction radius.
 * @returns The estimator.
 */
export function optics(hyper: OpticsHyper = {}): Estimator<OpticsHyper, ExplainedFitted<OpticsParams, DensityVerbs>> {
  return defineExplainedEstimator<OpticsHyper, OpticsParams, DensityVerbs>(
    {
      name: 'optics',
      kind: 'clusterer',
      hyperSpace: (h) =>
        dict({
          minSamples: int(2, 50, { default: h.minSamples ?? 5, label: 'core threshold' }),
          eps: real(0.01, 5, { log: true, default: h.eps ?? 0.5, label: 'extraction radius' }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'optics')
        const o = opticsOrdering(data.x, h.minSamples ?? 5, h.maxEps ?? Number.POSITIVE_INFINITY)
        const eps = h.eps ?? (Number.isFinite(h.maxEps ?? Number.POSITIVE_INFINITY) ? (h.maxEps as number) : 0.5)
        const labels = opticsClustersAt(o, eps)
        const isCore = o.coreDistances.map((c) => c <= eps)
        const roles: DensityRole[] = labels.map((l, i) => (isCore[i] === true ? 'core' : l >= 0 ? 'border' : 'noise'))
        const cores = isCore.flatMap((c, i) => (c ? [i] : []))
        const finite = (v: number): number => (Number.isFinite(v) ? v : -1)
        return {
          params: {
            labels,
            reachability: o.reachability.map(finite),
            coreDistances: o.coreDistances.map(finite),
            ordering: o.ordering,
            components: rowsOf(m.data, m.d, cores),
            componentLabels: cores.map((i) => labels[i] ?? -1),
            eps,
          },
          exposition: [
            clusteringSteps({
              method: 'optics',
              assignments: [labels],
              roles,
              ordering: o.ordering,
              reachability: o.ordering.map((p) => finite(o.reachability[p] ?? Number.POSITIVE_INFINITY)),
            }),
          ],
        }
      },
      methods: (p) => coreVerbs(p.components, p.componentLabels, p.eps),
    },
    hyper,
  )
}

/** OPTICS as a class. */
export class OPTICS extends EstimatorClass<OpticsHyper, ExplainedFitted<OpticsParams, DensityVerbs>> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link optics}.
   */
  constructor(hyper: OpticsHyper = {}) {
    super(optics(hyper))
  }
}

// ---------------------------------------------------------------------------------------------------------------
// HDBSCAN

/** One row of HDBSCAN's condensed tree: a parent cluster, a child (a point below n, a cluster above), lambda, size. */
export type CondensedRow = [number, number, number, number]

/**
 * The single-linkage tree of the mutual reachability distance max(core(a), core(b), d(a, b)), as scipy linkage rows
 * (by Kruskal's algorithm over the complete graph).
 *
 * @param dist - Pairwise distances `[n, n]` row-major.
 * @param core - Core distances.
 * @returns The n - 1 merges.
 */
export function mutualReachabilityTree(dist: readonly number[], core: readonly number[]): Merge[] {
  const n = core.length
  const edges: [number, number, number][] = []
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) edges.push([i, j, Math.max(core[i] ?? 0, core[j] ?? 0, dist[i * n + j] ?? 0)])
  edges.sort((a, b) => a[2] - b[2])
  const parent = Array.from({ length: 2 * n - 1 }, (_, i) => i)
  const size = new Array<number>(2 * n - 1).fill(1)
  const find = (i: number): number => {
    let r = i
    while (parent[r] !== r) r = parent[r] ?? r
    let c = i
    while (parent[c] !== r) {
      const next = parent[c] ?? r
      parent[c] = r
      c = next
    }
    return r
  }
  const merges: Merge[] = []
  for (const [a, b, w] of edges) {
    const ra = find(a)
    const rb = find(b)
    if (ra === rb) continue
    const id = n + merges.length
    parent[ra] = id
    parent[rb] = id
    size[id] = (size[ra] ?? 1) + (size[rb] ?? 1)
    merges.push([Math.min(ra, rb), Math.max(ra, rb), w, size[id] ?? 0])
    if (merges.length === n - 1) break
  }
  return merges
}

/**
 * Condense a single-linkage tree (Campello et al. 2013; hdbscan's `condense_tree`): walking down from the root, a split
 * where both sides have at least `minClusterSize` points makes two new clusters; smaller sides fall out as points at
 * that lambda = 1 / distance.
 *
 * @param merges - The linkage rows.
 * @param minClusterSize - The smallest cluster.
 * @returns The condensed tree rows (cluster ids from n, the root being n).
 */
export function condenseTree(merges: readonly Merge[], minClusterSize: number): CondensedRow[] {
  const n = merges.length + 1
  const root = 2 * n - 2
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
  const relabel = new Map<number, number>([[root, n]])
  let next = n + 1
  const rows: CondensedRow[] = []
  const queue = [root]
  while (queue.length > 0) {
    const node = queue.shift() as number
    if (node < n) continue
    const [left, right, d] = merges[node - n] as Merge
    const lambda = d > 0 ? 1 / d : Number.POSITIVE_INFINITY
    const parent = relabel.get(node) as number
    const big = (c: number): boolean => sizeOf(c) >= minClusterSize
    if (big(left) && big(right)) {
      for (const c of [left, right]) {
        relabel.set(c, next)
        rows.push([parent, next, lambda, sizeOf(c)])
        next++
        queue.push(c)
      }
    } else {
      for (const c of [left, right]) {
        if (big(c)) {
          relabel.set(c, parent)
          queue.push(c)
        } else for (const p of leaves(c)) rows.push([parent, p, lambda, 1])
      }
    }
  }
  return rows
}

/**
 * The excess-of-mass selection of flat clusters from a condensed tree, and the labels it gives.
 *
 * @param rows - The condensed tree.
 * @param n - The number of points.
 * @param allowSingleCluster - Whether the root may be selected (default false).
 * @returns Labels (-1 for noise), the selected cluster ids, and the stability of each cluster.
 */
export function selectClusters(
  rows: readonly CondensedRow[],
  n: number,
  allowSingleCluster = false,
): { readonly labels: number[]; readonly selected: number[]; readonly stability: Map<number, number> } {
  const birth = new Map<number, number>([[n, 0]])
  const children = new Map<number, number[]>()
  for (const [p, c, l] of rows)
    if (c >= n) {
      birth.set(c, l)
      children.set(p, [...(children.get(p) ?? []), c])
    }
  const stability = new Map<number, number>()
  for (const [p, , l, s] of rows) {
    const b = birth.get(p) ?? 0
    const lam = Number.isFinite(l) ? l : b
    stability.set(p, (stability.get(p) ?? 0) + (lam - b) * s)
  }
  const clusters = [...birth.keys()].sort((a, b) => b - a)
  const selected = new Set<number>()
  const total = new Map<number, number>()
  for (const c of clusters) {
    const kids = children.get(c) ?? []
    const own = stability.get(c) ?? 0
    const sub = kids.reduce((a, k) => a + (total.get(k) ?? 0), 0)
    if (kids.length > 0 && sub > own) total.set(c, sub)
    else if (c === n && !allowSingleCluster) total.set(c, sub)
    else {
      total.set(c, own)
      selected.add(c)
      const stack = [...kids]
      while (stack.length > 0) {
        const k = stack.pop() as number
        selected.delete(k)
        stack.push(...(children.get(k) ?? []))
      }
    }
  }
  const parentOf = new Map<number, number>()
  for (const [p, c] of rows) parentOf.set(c, p)
  const chosen = [...selected].sort((a, b) => a - b)
  const index = new Map(chosen.map((c, i) => [c, i]))
  const labels = Array.from({ length: n }, (_, i) => {
    let c = parentOf.get(i)
    while (c !== undefined) {
      const l = index.get(c)
      if (l !== undefined) return l
      c = parentOf.get(c)
    }
    return -1
  })
  return { labels, selected: chosen, stability }
}

/** Hyperparameters of HDBSCAN. */
export type HdbscanHyper = {
  /** The smallest cluster (default 5). */
  readonly minClusterSize?: number
  /** The core-distance neighbour count, itself included (default minClusterSize). */
  readonly minSamples?: number
  readonly allowSingleCluster?: boolean
}

/** The parameters of a fitted HDBSCAN. */
export type HdbscanParams = {
  readonly labels: number[]
  readonly coreDistances: number[]
  readonly merges: Merge[]
  readonly condensedTree: CondensedRow[]
  readonly components: number[][]
  readonly componentLabels: number[]
  readonly reach: number
}

/**
 * HDBSCAN (Campello, Moulavi and Sander 2013; sklearn.cluster.HDBSCAN): core distances, the single-linkage tree of the
 * mutual reachability distance, its condensed tree and the excess-of-mass selection of clusters. Its exposition is a
 * `clusteringSteps` record with the mutual-reachability merges and the selected labels (noise -1).
 *
 * @param hyper - The smallest cluster, the core neighbour count and whether one cluster may be returned.
 * @returns The estimator.
 */
export function hdbscan(
  hyper: HdbscanHyper = {},
): Estimator<HdbscanHyper, ExplainedFitted<HdbscanParams, DensityVerbs>> {
  return defineExplainedEstimator<HdbscanHyper, HdbscanParams, DensityVerbs>(
    {
      name: 'hdbscan',
      kind: 'clusterer',
      hyperSpace: (h) =>
        dict({
          minClusterSize: int(2, 100, { default: h.minClusterSize ?? 5, label: 'smallest cluster' }),
          minSamples: int(1, 100, { default: h.minSamples ?? h.minClusterSize ?? 5, label: 'core neighbours' }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'hdbscan', { minSamples: 2 })
        const mcs = h.minClusterSize ?? 5
        const k = Math.min(h.minSamples ?? mcs, m.n)
        const dist = distances(data.x)
        const core = Array.from(
          { length: m.n },
          (_, i) => dist.slice(i * m.n, (i + 1) * m.n).sort((a, b) => a - b)[k - 1] ?? 0,
        )
        const merges = mutualReachabilityTree(dist, core)
        const condensed = condenseTree(merges, mcs)
        const { labels } = selectClusters(condensed, m.n, h.allowSingleCluster ?? false)
        const members = labels.flatMap((l, i) => (l >= 0 ? [i] : []))
        const reach = Math.max(0, ...members.map((i) => core[i] ?? 0))
        return {
          params: {
            labels,
            coreDistances: core,
            merges,
            condensedTree: condensed.map(([p, c, l, s]) => [p, c, Number.isFinite(l) ? l : -1, s]),
            components: rowsOf(m.data, m.d, members),
            componentLabels: members.map((i) => labels[i] ?? -1),
            reach,
          },
          exposition: [clusteringSteps({ method: 'hdbscan', assignments: [labels], merges })],
        }
      },
      methods: (p) => coreVerbs(p.components, p.componentLabels, p.reach),
    },
    hyper,
  )
}

/** HDBSCAN as a class. */
export class HDBSCAN extends EstimatorClass<HdbscanHyper, ExplainedFitted<HdbscanParams, DensityVerbs>> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link hdbscan}.
   */
  constructor(hyper: HdbscanHyper = {}) {
    super(hdbscan(hyper))
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Mean shift

/**
 * sklearn's `estimate_bandwidth`: the mean over rows of the distance to the k-th nearest row, k = quantile * n.
 *
 * @param x - Points `[n, d]`.
 * @param quantile - The neighbour fraction (default 0.3).
 * @returns The bandwidth.
 */
export function estimateBandwidth(x: TensorLike, quantile = 0.3): number {
  const n = matrixView(x, 'estimateBandwidth').n
  const k = Math.max(1, Math.floor(n * quantile))
  const dist = distances(x)
  let total = 0
  for (let i = 0; i < n; i++) total += dist.slice(i * n, (i + 1) * n).sort((a, b) => a - b)[Math.min(k, n) - 1] ?? 0
  return total / n
}

/** The state of mean shift: every seed's position and whether it has stopped. */
export interface MeanShiftState {
  readonly seeds: number[][]
  readonly moving: boolean[]
  readonly history: number[][][]
}

/**
 * Mean shift as a traced algorithm: each step moves every still-moving seed to the (flat or Gaussian) kernel-weighted
 * mean of the points around it; a seed stops when it moves less than 1e-3 bandwidths.
 *
 * @param x - Points `[n, d]`.
 * @param seeds - Starting positions `[s, d]`.
 * @param bandwidth - The kernel radius (flat) or standard deviation (Gaussian).
 * @param kernel - 'flat' (sklearn) or 'gaussian'.
 * @returns The algorithm.
 */
export function meanShiftAlgorithm(
  x: TensorLike,
  seeds: readonly (readonly number[])[],
  bandwidth: number,
  kernel: 'flat' | 'gaussian' = 'flat',
): Algorithm<MeanShiftState> {
  const m = matrixView(x, 'meanShift')
  return {
    name: 'mean-shift',
    config: { bandwidth, kernel },
    init: () => {
      const s = seeds.map((r) => [...r])
      return { seeds: s, moving: s.map(() => true), history: [s] }
    },
    step: (s) => {
      const dist = toFlat(geometry.cdist(s.seeds, x))
      const moving = [...s.moving]
      const next = s.seeds.map((seed, a) => {
        if (moving[a] !== true) return seed
        const mean = new Array<number>(m.d).fill(0)
        let w = 0
        for (let i = 0; i < m.n; i++) {
          const dd = dist[a * m.n + i] ?? 0
          const wi = kernel === 'flat' ? (dd <= bandwidth ? 1 : 0) : Math.exp(-0.5 * (dd / bandwidth) ** 2)
          if (wi === 0) continue
          w += wi
          for (let q = 0; q < m.d; q++) mean[q] = (mean[q] ?? 0) + wi * (m.data[i * m.d + q] ?? 0)
        }
        if (w === 0) {
          moving[a] = false
          return seed
        }
        const out = mean.map((v) => v / w)
        if (Math.hypot(...out.map((v, q) => v - (seed[q] ?? 0))) < 1e-3 * bandwidth) moving[a] = false
        return out
      })
      return { seeds: next, moving, history: [...s.history, next] }
    },
    done: (s) => (s.moving.some((v) => v) ? undefined : 'converged'),
    observe: (s) => ({ metrics: { moving: s.moving.filter((v) => v).length } }),
  }
}

/** Hyperparameters of mean shift. */
export type MeanShiftHyper = {
  /** The kernel bandwidth (default: {@link estimateBandwidth}). */
  readonly bandwidth?: number
  readonly kernel?: 'flat' | 'gaussian'
  readonly maxIter?: number
  /** Leave points farther than a bandwidth from every mode unlabelled (-1) instead of using the nearest (default false). */
  readonly clusterAll?: boolean
}

/** The parameters of a fitted mean shift. */
export type MeanShiftParams = { readonly centres: number[][]; readonly labels: number[]; readonly bandwidth: number }

/**
 * Mean shift (Comaniciu and Meer 2002; sklearn.cluster.MeanShift): every row seeds a hill climb on the kernel density;
 * the modes reached, merged when within a bandwidth (the most populated first), are the cluster centres. Its
 * exposition is a `clusteringSteps` record whose centres are the seeds' positions per step (their trajectories) and
 * whose assignments are the final labels.
 *
 * @param hyper - Bandwidth, kernel and iteration cap.
 * @returns The estimator.
 */
export function meanShift(
  hyper: MeanShiftHyper = {},
): Estimator<MeanShiftHyper, ExplainedFitted<MeanShiftParams, DensityVerbs>> {
  return defineExplainedEstimator<MeanShiftHyper, MeanShiftParams, DensityVerbs>(
    {
      name: 'mean-shift',
      kind: 'clusterer',
      hyperSpace: (h) =>
        dict({
          bandwidth: real(0.05, 10, { log: true, default: h.bandwidth ?? 1, label: 'bandwidth' }),
          kernel: choice(['flat', 'gaussian'], { default: h.kernel ?? 'flat' }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'meanShift')
        const bandwidth = h.bandwidth ?? estimateBandwidth(data.x)
        const seeds = rowsOf(
          m.data,
          m.d,
          Array.from({ length: m.n }, (_, i) => i),
        )
        const out = runAlgorithm(meanShiftAlgorithm(data.x, seeds, bandwidth, h.kernel ?? 'flat'), {
          ...fitRunOptions(options, 'mean-shift'),
          maxSteps: h.maxIter ?? 300,
        })
        const modes = out.state.seeds
        const pop = toFlat(geometry.cdist(modes, data.x)).reduce<number[]>((acc, v, idx) => {
          const a = Math.floor(idx / m.n)
          acc[a] = (acc[a] ?? 0) + (v <= bandwidth ? 1 : 0)
          return acc
        }, [])
        const order = modes.map((_, a) => a).sort((a, b) => (pop[b] ?? 0) - (pop[a] ?? 0) || a - b)
        const centres: number[][] = []
        for (const a of order) {
          const c = modes[a] as number[]
          if (centres.every((k) => Math.hypot(...k.map((v, q) => v - (c[q] ?? 0))) >= bandwidth)) centres.push(c)
        }
        const dist = toFlat(geometry.cdist(data.x, centres))
        const labels = Array.from({ length: m.n }, (_, i) => {
          let best = 0
          for (let j = 1; j < centres.length; j++)
            if ((dist[i * centres.length + j] ?? 0) < (dist[i * centres.length + best] ?? 0)) best = j
          return h.clusterAll === false && (dist[i * centres.length + best] ?? 0) > bandwidth ? -1 : best
        })
        return {
          params: { centres, labels, bandwidth },
          exposition: [
            clusteringSteps({
              method: 'meanShift',
              assignments: out.state.history.map(() => labels),
              centres: out.state.history,
            }),
          ],
        }
      },
      methods: (p) =>
        coreVerbs(
          p.centres,
          p.centres.map((_, j) => j),
          Number.POSITIVE_INFINITY,
        ),
    },
    hyper,
  )
}

/** Mean shift as a class. */
export class MeanShift extends EstimatorClass<MeanShiftHyper, ExplainedFitted<MeanShiftParams, DensityVerbs>> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link meanShift}.
   */
  constructor(hyper: MeanShiftHyper = {}) {
    super(meanShift(hyper))
  }
}
