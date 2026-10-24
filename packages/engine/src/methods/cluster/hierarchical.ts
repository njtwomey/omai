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
import { choice, dict, int } from '../../lib/space/index.ts'
import { pearson } from '../../lib/stats/index.ts'
import { type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'

/** How the distance between two clusters is measured (scipy.cluster.hierarchy.linkage's methods). */
export type Linkage = 'single' | 'complete' | 'average' | 'weighted' | 'ward' | 'centroid' | 'median'

/** One merge as scipy's linkage row: the two clusters joined (ids >= n are earlier merges), the height, the size. */
export type Merge = [number, number, number, number]

/** The state of the agglomeration: the active clusters' distance matrix and the merges so far. */
export interface AgglomerationState {
  /** dist[i * n + j] between active clusters i and j (by slot). */
  readonly dist: Float64Array
  /** Per slot: the cluster id it holds (-1 once merged away) and its size. */
  readonly ids: readonly number[]
  readonly sizes: readonly number[]
  readonly merges: readonly Merge[]
}

/**
 * The Lance-Williams update: the distance from cluster k to the union of i and j.
 *
 * @param linkage - The linkage.
 * @param dki - Distance from k to i.
 * @param dkj - Distance from k to j.
 * @param dij - Distance from i to j.
 * @param ni - Size of i.
 * @param nj - Size of j.
 * @param nk - Size of k.
 * @returns The new distance.
 */
export function lanceWilliams(
  linkage: Linkage,
  dki: number,
  dkj: number,
  dij: number,
  ni: number,
  nj: number,
  nk: number,
): number {
  switch (linkage) {
    case 'single':
      return Math.min(dki, dkj)
    case 'complete':
      return Math.max(dki, dkj)
    case 'average':
      return (ni * dki + nj * dkj) / (ni + nj)
    case 'weighted':
      return (dki + dkj) / 2
    case 'ward':
      return Math.sqrt(Math.max(0, ((nk + ni) * dki ** 2 + (nk + nj) * dkj ** 2 - nk * dij ** 2) / (ni + nj + nk)))
    case 'centroid':
      return Math.sqrt(Math.max(0, (ni * dki ** 2 + nj * dkj ** 2) / (ni + nj) - (ni * nj * dij ** 2) / (ni + nj) ** 2))
    case 'median':
      return Math.sqrt(Math.max(0, dki ** 2 / 2 + dkj ** 2 / 2 - dij ** 2 / 4))
  }
}

/**
 * Agglomerative clustering as a traced algorithm: each step merges the closest pair of active clusters and updates
 * the distances by the Lance-Williams formula (the generic O(n^3) algorithm), recording scipy's linkage rows.
 *
 * @param x - Points `[n, d]`.
 * @param linkage - The linkage (default Ward).
 * @returns The algorithm; n - 1 steps build the whole hierarchy.
 */
export function agglomerationAlgorithm(x: TensorLike, linkage: Linkage = 'ward'): Algorithm<AgglomerationState> {
  const n = matrixView(x, 'agglomeration').n
  return {
    name: 'agglomerative',
    config: { linkage, n },
    init: () => ({
      dist: Float64Array.from(toFlat(geometry.cdist(x, x))),
      ids: Array.from({ length: n }, (_, i) => i),
      sizes: new Array<number>(n).fill(1),
      merges: [],
    }),
    step: (s) => {
      let bi = -1
      let bj = -1
      let best = Number.POSITIVE_INFINITY
      for (let i = 0; i < n; i++) {
        if ((s.ids[i] ?? -1) < 0) continue
        for (let j = i + 1; j < n; j++) {
          if ((s.ids[j] ?? -1) < 0) continue
          const v = s.dist[i * n + j] ?? 0
          if (v < best) {
            best = v
            bi = i
            bj = j
          }
        }
      }
      const dist = s.dist.slice()
      const ids = [...s.ids]
      const sizes = [...s.sizes]
      const ni = sizes[bi] ?? 1
      const nj = sizes[bj] ?? 1
      for (let k = 0; k < n; k++) {
        if (k === bi || k === bj || (ids[k] ?? -1) < 0) continue
        const v = lanceWilliams(linkage, s.dist[k * n + bi] ?? 0, s.dist[k * n + bj] ?? 0, best, ni, nj, sizes[k] ?? 1)
        dist[k * n + bi] = v
        dist[bi * n + k] = v
      }
      const a = ids[bi] ?? 0
      const b = ids[bj] ?? 0
      const merge: Merge = [Math.min(a, b), Math.max(a, b), best, ni + nj]
      ids[bi] = n + s.merges.length
      ids[bj] = -1
      sizes[bi] = ni + nj
      return { dist, ids, sizes, merges: [...s.merges, merge] }
    },
    done: (s) => (s.merges.length >= n - 1 ? 'completed' : undefined),
    observe: (s) => ({ metrics: { height: s.merges.at(-1)?.[2] ?? 0, clusters: n - s.merges.length } }),
  }
}

/**
 * The linkage matrix of agglomerative clustering (scipy.cluster.hierarchy.linkage on raw observations).
 *
 * @param x - Points `[n, d]`.
 * @param linkage - The linkage (default Ward).
 * @returns The n - 1 merges.
 */
export function linkageMatrix(x: TensorLike, linkage: Linkage = 'ward'): Merge[] {
  const n = matrixView(x, 'linkageMatrix').n
  return [...runAlgorithm(agglomerationAlgorithm(x, linkage), { maxSteps: n }).state.merges]
}

/**
 * Flat clusters from a hierarchy (scipy's fcluster): apply merges until k clusters remain, or every merge below a
 * height. Labels are 0, 1, ... in order of first appearance.
 *
 * @param merges - The linkage rows.
 * @param n - The number of observations.
 * @param cut - The number of clusters, or a height threshold.
 * @returns A label per observation.
 */
export function cutTree(
  merges: readonly Merge[],
  n: number,
  cut: { readonly nClusters: number } | { readonly height: number },
): number[] {
  const parent = Array.from({ length: 2 * n - 1 }, (_, i) => i)
  const find = (i: number): number => {
    let r = i
    while (parent[r] !== r) r = parent[r] ?? r
    return r
  }
  const count = 'nClusters' in cut ? Math.max(0, n - cut.nClusters) : merges.filter((m) => m[2] <= cut.height).length
  for (let t = 0; t < Math.min(count, merges.length); t++) {
    const [a, b] = merges[t] as Merge
    parent[find(a)] = n + t
    parent[find(b)] = n + t
  }
  const labels = new Map<number, number>()
  return Array.from({ length: n }, (_, i) => {
    const r = find(i)
    if (!labels.has(r)) labels.set(r, labels.size)
    return labels.get(r) ?? 0
  })
}

/**
 * Cophenetic distances (scipy's cophenet): the height at which each pair first shares a cluster, condensed.
 *
 * @param merges - The linkage rows.
 * @param n - The number of observations.
 * @returns The condensed cophenetic distances, in pdist order.
 */
export function copheneticDistances(merges: readonly Merge[], n: number): number[] {
  const members: number[][] = Array.from({ length: n }, (_, i) => [i])
  const d = new Float64Array((n * (n - 1)) / 2)
  const at = (i: number, j: number): number => {
    const [a, b] = i < j ? [i, j] : [j, i]
    return n * a - (a * (a + 1)) / 2 + (b - a - 1)
  }
  for (const [a, b, h] of merges) {
    const left = members[a] ?? []
    const right = members[b] ?? []
    for (const i of left) for (const j of right) d[at(i, j)] = h
    members.push([...left, ...right])
  }
  return Array.from(d)
}

/**
 * The cophenetic correlation: how faithfully the hierarchy's heights preserve the original distances.
 *
 * @param x - Points `[n, d]`.
 * @param merges - The linkage rows of x.
 * @returns Pearson's correlation between cophenetic and original distances.
 */
export function copheneticCorrelation(x: TensorLike, merges: readonly Merge[]): number {
  const n = matrixView(x, 'copheneticCorrelation').n
  return pearson(copheneticDistances(merges, n), toFlat(geometry.pdist(x))).statistic
}

/** A dendrogram's layout: the leaf order and, per merge, the x position and height of its link. */
export interface Dendrogram {
  readonly leaves: number[]
  /** links[t]: merge t drawn at x (leaves at 0, 1, ...) and height, joining children at childX. */
  readonly links: { readonly x: number; readonly height: number; readonly childX: [number, number] }[]
}

/**
 * Lay out a dendrogram (scipy's dendrogram order: left child first).
 *
 * @param merges - The linkage rows.
 * @param n - The number of observations.
 * @returns The leaf order and link positions.
 */
export function dendrogram(merges: readonly Merge[], n: number): Dendrogram {
  const leaves: number[] = []
  const visit = (id: number): void => {
    if (id < n) leaves.push(id)
    else {
      const [a, b] = merges[id - n] as Merge
      visit(a)
      visit(b)
    }
  }
  if (n > 0) visit(merges.length === 0 ? 0 : n + merges.length - 1)
  const xs = new Map<number, number>(leaves.map((l, i) => [l, i]))
  const links = merges.map(([a, b, h], t) => {
    const childX: [number, number] = [xs.get(a) ?? 0, xs.get(b) ?? 0]
    const x = (childX[0] + childX[1]) / 2
    xs.set(n + t, x)
    return { x, height: h, childX }
  })
  return { leaves, links }
}

/** Hyperparameters of agglomerative clustering. */
export type AgglomerativeHyper = {
  /** Clusters to cut the tree into (default 2), unless a distance threshold is given. */
  readonly nClusters?: number
  readonly linkage?: Linkage
  /** Cut at this height instead of at a number of clusters. */
  readonly distanceThreshold?: number
}

/** The parameters of a fitted hierarchy. */
export type AgglomerativeParams = {
  readonly merges: Merge[]
  readonly labels: number[]
  readonly nClusters: number
  readonly copheneticCorrelation: number
  readonly nFeaturesIn: number
}

/** A fitted agglomerative clustering. */
export type FittedAgglomerative = ExplainedFitted<AgglomerativeParams, { readonly nFeaturesIn: number }>

/**
 * Agglomerative clustering (sklearn.cluster.AgglomerativeClustering, scipy's linkage): single, complete, average,
 * weighted, Ward, centroid or median linkage, traced merge by merge. Its exposition is a `clusteringSteps` record with
 * every merge (scipy's linkage rows) and the assignments of the flat cut.
 *
 * @param hyper - Number of clusters (or a height threshold) and the linkage.
 * @returns The estimator.
 */
export function agglomerativeClustering(
  hyper: AgglomerativeHyper = {},
): Estimator<AgglomerativeHyper, FittedAgglomerative> {
  return defineExplainedEstimator<AgglomerativeHyper, AgglomerativeParams, { readonly nFeaturesIn: number }>(
    {
      name: 'agglomerative-clustering',
      kind: 'clusterer',
      hyperSpace: (h) =>
        dict({
          nClusters: int(1, 20, { default: h.nClusters ?? 2, label: 'clusters' }),
          linkage: choice(['ward', 'single', 'complete', 'average'], { default: h.linkage ?? 'ward' }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'agglomerativeClustering')
        const k = h.nClusters ?? 2
        if (h.distanceThreshold === undefined && !(Number.isInteger(k) && k >= 1 && k <= m.n))
          throw new WhiteboxError('value', `agglomerativeClustering: nClusters must be in 1..${m.n}`)
        const run = fitRunOptions(options, 'agglomerative')
        const out = runAlgorithm(agglomerationAlgorithm(data.x, h.linkage ?? 'ward'), { ...run, maxSteps: m.n })
        const merges = [...out.state.merges]
        const labels = cutTree(
          merges,
          m.n,
          h.distanceThreshold === undefined ? { nClusters: k } : { height: h.distanceThreshold },
        )
        return {
          params: {
            merges,
            labels,
            nClusters: Math.max(...labels) + 1,
            copheneticCorrelation: m.n > 2 ? copheneticCorrelation(data.x, merges) : 1,
            nFeaturesIn: m.d,
          },
          exposition: [clusteringSteps({ method: 'agglomerative', assignments: [labels], merges })],
        }
      },
      methods: (p) => ({ nFeaturesIn: p.nFeaturesIn }),
    },
    hyper,
  )
}

/** Agglomerative clustering as a class. */
export class AgglomerativeClustering extends EstimatorClass<AgglomerativeHyper, FittedAgglomerative> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link agglomerativeClustering}.
   */
  constructor(hyper: AgglomerativeHyper = {}) {
    super(agglomerativeClustering(hyper))
  }
}
