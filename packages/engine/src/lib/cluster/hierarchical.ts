/**
 * Agglomerative hierarchical clustering as scipy's `linkage` and scikit-learn's `AgglomerativeClustering` do it: a
 * traced algorithm that takes one merge per step under single, complete, average, weighted, Ward, centroid or median
 * linkage (the Lance-Williams update), the linkage matrix, flat cuts of the tree, cophenetic distances and correlation,
 * a dendrogram layout, and the single-linkage tree of an edge list that HDBSCAN builds on. BIRCH, which summarises the
 * rows in a tree of clustering features before it clusters them, is here too.
 */

import { maxOf, toRows } from '../base/index.ts'
import { type DistanceAlias, type DistanceName, type DistanceOptions, distanceMatrix } from '../distances/index.ts'
import { UnionFind } from '../graph/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { pearson } from '../stats/index.ts'
import { fromStorage, type Tensor, type TensorLike, tensor } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { canonicalLabels } from './_support.ts'
import { nearestCentres } from './kmeans.ts'

/** How the distance between two clusters is measured (scipy.cluster.hierarchy.linkage's methods). */
export type Linkage = 'single' | 'complete' | 'average' | 'weighted' | 'ward' | 'centroid' | 'median'

/** One merge as scipy's linkage row: the two clusters joined (ids >= n are earlier merges), the height, the size. */
export type Merge = [left: number, right: number, height: number, size: number]

/** The metric slot of the distance-based clusterers: a registered distance, or the rows already are distances. */
export type MetricChoice = DistanceName | DistanceAlias | 'precomputed'

/**
 * The state of the agglomeration: the distances among the active clusters, which cluster each slot holds, and the
 * merges so far. `labels` gives the slot of the cluster each row is in now.
 */
export interface AgglomerationState {
  /** `dist[i * n + j]` between the clusters in slots i and j. */
  readonly dist: Float64Array
  /** Per slot: the cluster id it holds (-1 once merged away). */
  readonly ids: readonly number[]
  readonly sizes: readonly number[]
  readonly merges: readonly Merge[]
  readonly labels: readonly number[]
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
 * @example
 * ```ts
 * import { lanceWilliams } from 'omai-engine/cluster'
 * lanceWilliams('single', 2, 5, 1, 1, 1, 1) // => 2
 * lanceWilliams('average', 2, 5, 1, 3, 1, 1) // => 2.75
 * ```
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

/** Options of {@link agglomerativeAlgorithm}. */
export interface AgglomerationOptions {
  /** The linkage (default Ward). Ward, centroid and median expect Euclidean distances. */
  readonly linkage?: Linkage
  /** The distance among rows (default euclidean), or `'precomputed'` when `x` is a distance matrix. */
  readonly metric?: MetricChoice
  readonly distanceOptions?: DistanceOptions
}

/**
 * Agglomerative clustering as a traced {@link Algorithm}: each step merges the closest pair of clusters and updates the
 * distances by the Lance-Williams formula (the generic O(n^3) algorithm), appending one row of scipy's linkage matrix.
 * It observes the height and the number of clusters, the partition now (`labels`) and the linkage matrix so far.
 *
 * @param x - Points `[n, d]`, or with `metric: 'precomputed'` the distances `[n, n]`.
 * @param options - Linkage and metric.
 * @returns The algorithm; n - 1 steps build the whole hierarchy.
 * @example
 * ```ts
 * import { agglomerativeAlgorithm } from 'omai-engine/cluster'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = agglomerativeAlgorithm([[0], [1], [5], [6]], { linkage: 'single' })
 * runAlgorithm(alg, { maxSteps: 3 }).state.merges // => [[0, 1, 1, 2], [2, 3, 1, 2], [4, 5, 4, 4]]
 * ```
 */
export function agglomerativeAlgorithm(
  x: TensorLike,
  options: AgglomerationOptions = {},
): Algorithm<AgglomerationState> {
  const linkage = options.linkage ?? 'ward'
  const { n, data } = distanceMatrix(x, options.metric ?? 'euclidean', options.distanceOptions)
  return {
    name: 'agglomerative',
    config: { linkage, n },
    schema: channels({
      metrics: {
        height: { description: 'the height of the merge just made' },
        clusters: { description: 'the number of clusters left' },
      },
      tensors: {
        labels: { role: 'state', description: 'the cluster of each row now', shape: ['n'] },
        merges: { role: 'state', description: 'the linkage matrix so far', shape: ['m', 4] },
      },
    }),
    init: () => ({
      dist: data.slice(),
      ids: Array.from({ length: n }, (_, i) => i),
      sizes: new Array<number>(n).fill(1),
      merges: [],
      labels: Array.from({ length: n }, (_, i) => i),
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
      ids[bi] = n + s.merges.length
      ids[bj] = -1
      sizes[bi] = ni + nj
      return {
        dist,
        ids,
        sizes,
        merges: [...s.merges, [Math.min(a, b), Math.max(a, b), best, ni + nj] as Merge],
        labels: s.labels.map((l) => (l === bj ? bi : l)),
      }
    },
    done: (s) => (s.merges.length >= n - 1 ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { height: s.merges.at(-1)?.[2] ?? 0, clusters: n - s.merges.length },
      tensors: {
        labels: { role: 'state', value: canonicalLabels(s.labels), shape: [n] },
        merges: { role: 'state', value: s.merges.flat(), shape: [s.merges.length, 4] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as AgglomerationState,
  }
}

/**
 * The linkage matrix of agglomerative clustering (scipy.cluster.hierarchy.linkage on raw observations).
 *
 * @param x - Points `[n, d]`.
 * @param linkage - The linkage (default Ward).
 * @param metric - The distance among rows (default euclidean), or `'precomputed'`.
 * @returns The n - 1 merges.
 * @example
 * ```ts
 * import { linkageMatrix } from 'omai-engine/cluster'
 * linkageMatrix([[0], [1], [5]], 'complete') // => [[0, 1, 1, 2], [2, 3, 5, 3]]
 * ```
 */
export function linkageMatrix(x: TensorLike, linkage: Linkage = 'ward', metric: MetricChoice = 'euclidean'): Merge[] {
  const n = distanceMatrix(x, metric).n
  return [...runAlgorithm(agglomerativeAlgorithm(x, { linkage, metric }), { maxSteps: n }).state.merges]
}

/**
 * The single-linkage tree of a weighted edge list as scipy linkage rows: the edges are taken in order of weight (ties
 * in input order) and each one that joins two components is a merge. Single linkage is the minimum spanning tree, and
 * HDBSCAN's tree is this over the mutual reachability distances.
 *
 * @param n - The number of points.
 * @param edges - Edges `[a, b, weight]`; they must connect the points.
 * @returns The n - 1 merges.
 * @example
 * ```ts
 * import { singleLinkageTree } from 'omai-engine/cluster'
 * singleLinkageTree(3, [[0, 1, 2], [1, 2, 1]]) // => [[1, 2, 1, 2], [0, 3, 2, 3]]
 * ```
 */
export function singleLinkageTree(n: number, edges: readonly (readonly [number, number, number])[]): Merge[] {
  const uf = new UnionFind(n)
  const idOf = Array.from({ length: n }, (_, i) => i)
  const size = new Array<number>(n).fill(1)
  const order = edges.map((_, i) => i).sort((a, b) => (edges[a]?.[2] ?? 0) - (edges[b]?.[2] ?? 0) || a - b)
  const merges: Merge[] = []
  for (const e of order) {
    const [a, b, w] = edges[e] as readonly [number, number, number]
    const ra = uf.find(a)
    const rb = uf.find(b)
    if (ra === rb) continue
    const ia = idOf[ra] ?? 0
    const ib = idOf[rb] ?? 0
    const total = (size[ra] ?? 1) + (size[rb] ?? 1)
    uf.union(ra, rb)
    const root = uf.find(ra)
    idOf[root] = n + merges.length
    size[root] = total
    merges.push([Math.min(ia, ib), Math.max(ia, ib), w, total])
    if (merges.length === n - 1) break
  }
  return merges
}

/**
 * Flat clusters from a hierarchy (scipy's fcluster): apply merges until k clusters remain, or every merge below a
 * height. Labels are 0, 1, ... in order of first appearance.
 *
 * @param merges - The linkage rows.
 * @param n - The number of observations.
 * @param cut - The number of clusters, or a height threshold.
 * @returns A label per observation.
 * @example
 * ```ts
 * import { cutTree } from 'omai-engine/cluster'
 * const merges: [number, number, number, number][] = [[0, 1, 1, 2], [2, 3, 1, 2], [4, 5, 4, 4]]
 * cutTree(merges, 4, { nClusters: 2 }) // => [0, 0, 1, 1]
 * cutTree(merges, 4, { height: 0.5 }) // => [0, 1, 2, 3]
 * ```
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
 * @example
 * ```ts
 * import { copheneticDistances } from 'omai-engine/cluster'
 * copheneticDistances([[0, 1, 1, 2], [2, 3, 5, 3]], 3) // => [1, 5, 5]
 * ```
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
 * @param metric - The distance the linkage was built on (default euclidean).
 * @returns Pearson's correlation between cophenetic and original distances.
 * @example
 * ```ts
 * import { copheneticCorrelation, linkageMatrix } from 'omai-engine/cluster'
 * const x = [[0], [1], [4], [9]]
 * copheneticCorrelation(x, linkageMatrix(x, 'average')) // => 0.894832
 * ```
 */
export function copheneticCorrelation(
  x: TensorLike,
  merges: readonly Merge[],
  metric: MetricChoice = 'euclidean',
): number {
  const { n, data } = distanceMatrix(x, metric)
  const condensed: number[] = []
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) condensed.push(data[i * n + j] ?? 0)
  return pearson(copheneticDistances(merges, n), condensed).statistic
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
 * @example
 * ```ts
 * import { dendrogram } from 'omai-engine/cluster'
 * dendrogram([[0, 1, 1, 2], [2, 3, 5, 3]], 3).leaves // => [2, 0, 1]
 * ```
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

/** Hyperparameters of {@link agglomerativeClustering}. */
export interface AgglomerativeHyper {
  /** Clusters to cut the tree into (ignored when a distance threshold is given). */
  readonly nClusters: number
  readonly linkage: Linkage
  /** Cut at this height instead of at a number of clusters (`null`: cut at `nClusters`). */
  readonly distanceThreshold: number | null
  readonly metric: DistanceName | DistanceAlias
}

/** The fitted parameters of {@link agglomerativeClustering}: the hierarchy and its cut. */
export type AgglomerativeParams = {
  readonly merges: Merge[]
  readonly labels: number[]
  readonly nClusters: number
  /** The centroid of each cut cluster, for assigning new rows. */
  readonly centres: number[][]
}

/** What a fitted agglomerative clustering offers: new rows join the cut cluster with the nearest centroid. */
export interface AgglomerativeVerbs {
  readonly predict: (x: TensorLike) => Tensor
}

/**
 * Agglomerative clustering (sklearn.cluster.AgglomerativeClustering, scipy's linkage) on the iterative base: one merge
 * per step, the tree cut into `nClusters` flat clusters (or at a height). At any step the fitted value is the cut of
 * the merges so far (with the clusters not yet joined left separate).
 *
 * @index estimator Agglomerative clustering with the linkage matrix.
 * @example
 * ```ts
 * import { agglomerativeClustering } from 'omai-engine/cluster'
 * agglomerativeClustering.with({ nClusters: 2 }).fit({ x: [[0], [1], [9], [10]] }).params.labels // => [0, 0, 1, 1]
 * ```
 */
export const agglomerativeClustering: IterativeEstimator<
  AgglomerativeHyper,
  AgglomerationState,
  AgglomerativeParams,
  AgglomerativeVerbs
> = iterative<AgglomerativeHyper, AgglomerationState, AgglomerativeParams, AgglomerativeVerbs>({
  name: 'cluster/agglomerative',
  kind: 'clusterer',
  defaults: { nClusters: 2, linkage: 'ward', distanceThreshold: null, metric: 'euclidean' },
  hyperSpace: (h) =>
    dict({
      nClusters: int(1, 20, { default: h.nClusters, label: 'clusters' }),
      linkage: choice(['ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median'] as const, {
        default: h.linkage,
      }),
    }),
  algorithm: (data, h) => agglomerativeAlgorithm(data.x, { linkage: h.linkage, metric: h.metric }),
  maxSteps: (_, data) => matrixView(data.x, 'agglomerativeClustering').n,
  params: (s, data, h) => {
    const m = matrixView(data.x, 'agglomerativeClustering')
    const merges = [...s.merges]
    const labels = cutTree(
      merges,
      m.n,
      h.distanceThreshold === null ? { nClusters: Math.min(h.nClusters, m.n) } : { height: h.distanceThreshold },
    )
    const k = maxOf(labels) + 1
    const sums = Array.from({ length: k }, () => new Array<number>(m.d).fill(0))
    const counts = new Array<number>(k).fill(0)
    for (let i = 0; i < m.n; i++) {
      const c = labels[i] ?? 0
      counts[c] = (counts[c] ?? 0) + 1
      for (let q = 0; q < m.d; q++) (sums[c] as number[])[q] = (sums[c]?.[q] ?? 0) + (m.data[i * m.d + q] ?? 0)
    }
    return {
      merges,
      labels,
      nClusters: k,
      centres: sums.map((row, c) => row.map((v) => v / Math.max(counts[c] ?? 1, 1))),
    }
  },
  verbs: (p) => ({ predict: (x) => tensor(nearestCentres(x, p.centres).labels) }),
})

// ---------------------------------------------------------------------------------------------------------------
// BIRCH

/** A clustering feature: how many rows, their sum and their sum of squares, and the node below it (-1 in a leaf). */
export interface CfSubcluster {
  readonly n: number
  readonly ls: readonly number[]
  readonly ss: number
  readonly child: number
}

/** A node of the clustering-feature tree: its subclusters, and for a leaf the neighbours in the chain of leaves. */
export interface CfNode {
  readonly leaf: boolean
  readonly subs: readonly CfSubcluster[]
  readonly prev: number
  readonly next: number
}

/** The state of BIRCH: the arena of tree nodes, the root, and how many rows have been inserted. */
export interface BirchState {
  readonly nodes: readonly CfNode[]
  readonly root: number
  readonly inserted: number
}

const centroidOf = (s: CfSubcluster): number[] => s.ls.map((v) => v / s.n)
const sqNormOf = (s: CfSubcluster): number => {
  let t = 0
  for (const v of s.ls) t += (v / s.n) ** 2
  return t
}
const addSubclusters = (a: CfSubcluster, b: CfSubcluster, child = a.child): CfSubcluster => ({
  n: a.n + b.n,
  ls: a.ls.map((v, q) => v + (b.ls[q] ?? 0)),
  ss: a.ss + b.ss,
  child,
})

/** Insert a feature below a node, as scikit-learn's `_CFNode.insert_cf_subcluster`: whether the node must split. */
function insertInto(
  nodes: CfNode[],
  at: number,
  sub: CfSubcluster,
  threshold: number,
  branching: number,
): { readonly split: boolean } {
  const node = nodes[at] as CfNode
  if (node.subs.length === 0) {
    nodes[at] = { ...node, subs: [sub] }
    return { split: false }
  }
  const c = centroidOf(sub)
  let closest = 0
  let best = Number.POSITIVE_INFINITY
  for (const [i, s] of node.subs.entries()) {
    const sc = centroidOf(s)
    let dot = 0
    for (const [q, v] of sc.entries()) dot += v * (c[q] ?? 0)
    const dist = -2 * dot + sqNormOf(s)
    if (dist < best) {
      best = dist
      closest = i
    }
  }
  const near = node.subs[closest] as CfSubcluster
  if (near.child >= 0) {
    const { split } = insertInto(nodes, near.child, sub, threshold, branching)
    if (!split) {
      nodes[at] = { ...node, subs: node.subs.map((s, i) => (i === closest ? addSubclusters(s, sub) : s)) }
      return { split: false }
    }
    const [one, two] = splitNode(nodes, near.child)
    const subs = [...node.subs]
    subs[closest] = one
    subs.push(two)
    nodes[at] = { ...node, subs }
    return { split: subs.length > branching }
  }
  const merged = addSubclusters(near, sub)
  const radius = merged.ss / merged.n - sqNormOf(merged)
  if (radius <= threshold ** 2) {
    nodes[at] = { ...node, subs: node.subs.map((s, i) => (i === closest ? merged : s)) }
    return { split: false }
  }
  const subs = [...node.subs, sub]
  nodes[at] = { ...node, subs }
  return { split: subs.length > branching }
}

/** Split a node in two around its farthest pair of subclusters (`_split_node`); returns the two parent features. */
function splitNode(nodes: CfNode[], at: number): [CfSubcluster, CfSubcluster] {
  const node = nodes[at] as CfNode
  const m = node.subs.length
  const cents = node.subs.map(centroidOf)
  const sq = (a: readonly number[], b: readonly number[]): number =>
    a.reduce((t, v, q) => t + (v - (b[q] ?? 0)) ** 2, 0)
  let far: [number, number] = [0, 0]
  let farD = -1
  const dist = (i: number, j: number): number => sq(cents[i] as number[], cents[j] as number[])
  for (let i = 0; i < m; i++)
    for (let j = 0; j < m; j++) {
      const v = dist(i, j)
      if (v > farD) {
        farD = v
        far = [i, j]
      }
    }
  const first: CfSubcluster[] = []
  const second: CfSubcluster[] = []
  for (const [idx, s] of node.subs.entries()) {
    const closer = idx === far[0] || dist(far[0], idx) < dist(far[1], idx)
    ;(closer ? first : second).push(s)
  }
  const idA = nodes.length
  const idB = nodes.length + 1
  nodes.push({ leaf: node.leaf, subs: first, prev: node.leaf ? node.prev : -1, next: node.leaf ? idB : -1 })
  nodes.push({ leaf: node.leaf, subs: second, prev: node.leaf ? idA : -1, next: node.leaf ? node.next : -1 })
  if (node.leaf) {
    if (node.prev >= 0) nodes[node.prev] = { ...(nodes[node.prev] as CfNode), next: idA }
    if (node.next >= 0) nodes[node.next] = { ...(nodes[node.next] as CfNode), prev: idB }
  }
  const total = (subs: readonly CfSubcluster[], child: number): CfSubcluster =>
    subs.reduce((acc, s) => addSubclusters(acc, s, child), {
      n: 0,
      ls: new Array<number>(node.subs[0]?.ls.length ?? 0).fill(0),
      ss: 0,
      child,
    })
  return [total(first, idA), total(second, idB)]
}

/** The subclusters of the leaves, in the order of the chain of leaves. */
export function birchLeaves(state: BirchState): CfSubcluster[] {
  let start = state.root
  while (!(state.nodes[start] as CfNode).leaf) start = (state.nodes[start] as CfNode).subs[0]?.child ?? start
  while ((state.nodes[start] as CfNode).prev >= 0) start = (state.nodes[start] as CfNode).prev
  const out: CfSubcluster[] = []
  for (let at = start; at >= 0; at = (state.nodes[at] as CfNode).next) out.push(...(state.nodes[at] as CfNode).subs)
  return out
}

/**
 * BIRCH's tree of clustering features (Zhang, Ramakrishnan and Livny 1996; sklearn.cluster.Birch) as a traced
 * algorithm: each step inserts one row, which joins the nearest leaf subcluster if the merged radius stays within the
 * threshold and otherwise founds a new one; nodes split when they exceed the branching factor. It observes the
 * subcluster centres (`centres`) and their count.
 *
 * @param x - Points `[n, d]`.
 * @param options - The radius threshold (default 0.5) and the branching factor (default 50).
 * @returns The algorithm; n steps insert every row.
 * @example
 * ```ts
 * import { birchAlgorithm, birchLeaves } from 'omai-engine/cluster'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = birchAlgorithm([[0], [0.1], [5], [5.1]], { threshold: 0.5 })
 * birchLeaves(runAlgorithm(alg, { maxSteps: 4 }).state).length // => 2
 * ```
 */
export function birchAlgorithm(
  x: TensorLike,
  options: { readonly threshold?: number; readonly branchingFactor?: number } = {},
): Algorithm<BirchState> {
  const m = matrixView(x, 'birchAlgorithm')
  const threshold = options.threshold ?? 0.5
  const branching = options.branchingFactor ?? 50
  return {
    name: 'birch',
    config: { threshold, branchingFactor: branching },
    schema: channels({
      metrics: { subclusters: { description: 'the number of leaf subclusters' } },
      tensors: { centres: { role: 'state', description: 'the centres of the leaf subclusters', shape: ['s', 'd'] } },
    }),
    init: () => ({ nodes: [{ leaf: true, subs: [], prev: -1, next: -1 }], root: 0, inserted: 0 }),
    step: (s) => {
      const row = Array.from(m.data.subarray(s.inserted * m.d, (s.inserted + 1) * m.d))
      const nodes = s.nodes.map((n) => n)
      const sub: CfSubcluster = { n: 1, ls: row, ss: row.reduce((t, v) => t + v * v, 0), child: -1 }
      const { split } = insertInto(nodes, s.root, sub, threshold, branching)
      let root = s.root
      if (split) {
        const [one, two] = splitNode(nodes, s.root)
        nodes.push({ leaf: false, subs: [one, two], prev: -1, next: -1 })
        root = nodes.length - 1
      }
      return { nodes, root, inserted: s.inserted + 1 }
    },
    done: (s) => (s.inserted >= m.n ? 'completed' : undefined),
    observe: (s) => {
      const leaves = birchLeaves(s)
      return {
        metrics: { subclusters: leaves.length },
        tensors: { centres: { role: 'state', value: leaves.flatMap(centroidOf), shape: [leaves.length, m.d] } },
      }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BirchState,
  }
}

/** Hyperparameters of {@link birch}. */
export interface BirchHyper {
  /** The largest radius of a leaf subcluster. */
  readonly threshold: number
  readonly branchingFactor: number
  /** The clusters the subclusters are grouped into by Ward agglomeration (0: leave the subclusters as the clusters). */
  readonly nClusters: number
}

/** The fitted parameters of {@link birch}. */
export type BirchParams = {
  readonly subclusterCentres: number[][]
  readonly subclusterLabels: number[]
  readonly centres: number[][]
  readonly labels: number[]
}

/**
 * BIRCH (sklearn.cluster.Birch) on the iterative base: one row inserted into the tree per step; the fitted value groups
 * the leaf subclusters by Ward agglomeration into `nClusters` clusters and gives each row the cluster of its nearest
 * subcluster centre.
 *
 * @example
 * ```ts
 * import { birch } from 'omai-engine/cluster'
 * birch.with({ nClusters: 2, threshold: 1 }).fit({ x: [[0], [0.2], [9], [9.2]] }).params.labels // => [0, 0, 1, 1]
 * ```
 */
export const birch: IterativeEstimator<BirchHyper, BirchState, BirchParams, AgglomerativeVerbs> = iterative<
  BirchHyper,
  BirchState,
  BirchParams,
  AgglomerativeVerbs
>({
  name: 'cluster/birch',
  kind: 'clusterer',
  defaults: { threshold: 0.5, branchingFactor: 50, nClusters: 3 },
  hyperSpace: (h) =>
    dict({
      threshold: real(0.05, 5, { log: true, default: h.threshold }),
      nClusters: int(1, 10, { default: h.nClusters }),
    }),
  algorithm: (data, h) => birchAlgorithm(data.x, { threshold: h.threshold, branchingFactor: h.branchingFactor }),
  maxSteps: (_, data) => matrixView(data.x, 'birch').n,
  params: (s, data, h) => {
    const m = matrixView(data.x, 'birch')
    const leaves = birchLeaves(s)
    const centres = leaves.map(centroidOf)
    const k = Math.min(Math.max(h.nClusters, 1), centres.length)
    const subclusterLabels =
      h.nClusters <= 0 || centres.length <= 1
        ? centres.map((_, i) => i)
        : cutTree(linkageMatrix(centres, 'ward'), centres.length, { nClusters: k })
    const rows = toRows(m.data, m.n, m.d)
    const near = centres.length === 0 ? [] : nearestCentres(rows, centres).labels
    const kk = Math.max(0, maxOf(subclusterLabels)) + 1
    const groupCentres = Array.from({ length: kk }, (_, c) => {
      const members = leaves.flatMap((l, i) => ((subclusterLabels[i] ?? 0) === c ? [l] : []))
      const total = members.reduce((t, l) => t + l.n, 0)
      return (members[0]?.ls ?? []).map((_, q) => members.reduce((t, l) => t + (l.ls[q] ?? 0), 0) / Math.max(total, 1))
    })
    return {
      subclusterCentres: centres,
      subclusterLabels,
      centres: groupCentres,
      labels: near.map((j) => subclusterLabels[j] ?? 0),
    }
  },
  verbs: (p) => ({
    predict: (x) => {
      const m = matrixView(x, 'birch.predict')
      if (p.subclusterCentres.length === 0) return tensor(new Array<number>(m.n).fill(-1))
      const near = nearestCentres(fromStorage(m.data, [m.n, m.d], 'float64'), p.subclusterCentres).labels
      return tensor(near.map((j) => p.subclusterLabels[j] ?? 0))
    },
  }),
})
