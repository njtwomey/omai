/**
 * The k-means variants beside Lloyd's: Elkan's accelerated algorithm (the same iterations, with the triangle inequality
 * skipping distance computations, which the tape counts), bisecting k-means (one split per step) and kernel k-means
 * (Lloyd's algorithm in a kernel's feature space). Each is a traced algorithm with an estimator.
 */

import { MdmlError } from '../base/index.ts'
import { type PairwiseKernelMetric, pairwiseKernels } from '../kernels/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { eigh } from '../numerics/linalg/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, tensor, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { toRows } from './_support.ts'
import { initialCentres, lloydAlgorithm, nearestCentres, type Rows } from './kmeans.ts'

const dist = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let t = 0
  for (let q = 0; q < a.length; q++) t += ((a[q] ?? 0) - (b[q] ?? 0)) ** 2
  return Math.sqrt(t)
}

// ---------------------------------------------------------------------------------------------------------------
// Elkan

/**
 * The state of Elkan's algorithm: centres, labels, the upper bound on each row's distance to its centre, the lower
 * bounds on its distances to every centre (flat `[n, k]`), whether the upper bound is stale, and how many distances
 * the last step computed and skipped.
 */
export interface ElkanState {
  readonly centres: number[][]
  readonly labels: number[]
  readonly upper: number[]
  readonly lower: number[]
  readonly stale: boolean[]
  readonly shift: number
  readonly computed: number
  readonly skipped: number
}

/**
 * Elkan's algorithm (Elkan 2003; sklearn's `algorithm='elkan'`) as a traced {@link Algorithm}: each step moves the
 * centres to the means of their rows and reassigns the rows, using the triangle inequality with an upper bound on the
 * distance to the assigned centre and lower bounds on the others to avoid most distance computations. It reaches
 * the fixed point Lloyd's algorithm does, and observes how many distances it computed and how many it skipped.
 *
 * @param x - Points `[n, d]`.
 * @param init - Initial centres `[k, d]`.
 * @param options - Tolerance on the squared centre shift, relative to the mean feature variance (default 1e-4).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { elkanAlgorithm } from 'mdml-engine/cluster'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0, 0], [0, 1], [10, 0], [10, 1]]
 * runAlgorithm(elkanAlgorithm(x, [[0, 0], [10, 1]]), { maxSteps: 20 }).state.labels // => [0, 0, 1, 1]
 * ```
 */
export function elkanAlgorithm(
  x: TensorLike,
  init: Rows,
  options: { readonly tol?: number } = {},
): Algorithm<ElkanState> {
  const m = matrixView(x, 'elkanAlgorithm')
  const k = init.length
  const row = (i: number): Float64Array => m.data.subarray(i * m.d, (i + 1) * m.d)
  let variance = 0
  for (let q = 0; q < m.d; q++) {
    let mean = 0
    for (let i = 0; i < m.n; i++) mean += m.data[i * m.d + q] ?? 0
    mean /= m.n
    let v = 0
    for (let i = 0; i < m.n; i++) v += ((m.data[i * m.d + q] ?? 0) - mean) ** 2
    variance += v / m.n
  }
  const tol = (options.tol ?? 1e-4) * (m.d === 0 ? 0 : variance / m.d)
  const inertiaOf = (s: ElkanState): number =>
    s.labels.reduce((t, c, i) => t + dist(row(i), s.centres[c] as number[]) ** 2, 0)
  return {
    name: 'kmeans-elkan',
    config: { k, tol: options.tol ?? 1e-4 },
    schema: channels({
      metrics: {
        inertia: { description: 'sum of squared distances of the rows to their centres', direction: 'lower' },
        moved: { description: 'squared distance the centres moved this step (-1 before the first step)' },
        computed: { description: 'point-to-centre distances computed this step' },
        skipped: { description: 'point-to-centre distances the bounds made unnecessary this step' },
      },
      tensors: {
        centres: { role: 'state', description: 'the cluster centres', shape: ['k', 'd'] },
        labels: { role: 'state', description: 'the nearest centre of each row', shape: ['n'] },
        upper: { role: 'state', description: 'upper bound on the distance to the assigned centre', shape: ['n'] },
      },
    }),
    init: () => {
      const centres = init.map((r) => [...r])
      const lower = new Array<number>(m.n * k).fill(0)
      const labels = new Array<number>(m.n).fill(0)
      const upper = new Array<number>(m.n).fill(0)
      for (let i = 0; i < m.n; i++) {
        let best = 0
        for (let c = 0; c < k; c++) {
          lower[i * k + c] = dist(row(i), centres[c] as number[])
          if ((lower[i * k + c] as number) < (lower[i * k + best] as number)) best = c
        }
        labels[i] = best
        upper[i] = lower[i * k + best] as number
      }
      return {
        centres,
        labels,
        upper,
        lower,
        stale: new Array<boolean>(m.n).fill(false),
        shift: Number.MAX_VALUE,
        computed: m.n * k,
        skipped: 0,
      }
    },
    step: (s) => {
      // Move the centres to the means of their rows and loosen the bounds by how far each moved.
      const sums = init.map(() => new Array<number>(m.d).fill(0))
      const counts = new Array<number>(k).fill(0)
      for (let i = 0; i < m.n; i++) {
        const c = s.labels[i] as number
        counts[c] = (counts[c] ?? 0) + 1
        for (let q = 0; q < m.d; q++) (sums[c] as number[])[q] = (sums[c]?.[q] ?? 0) + (m.data[i * m.d + q] ?? 0)
      }
      const centres = s.centres.map((old, c) =>
        (counts[c] ?? 0) > 0 ? (sums[c] as number[]).map((v) => v / (counts[c] as number)) : old,
      )
      const delta = centres.map((c, j) => dist(c, s.centres[j] as number[]))
      const shift = delta.reduce((t, v) => t + v * v, 0)
      const lower = s.lower.map((l, idx) => Math.max(l - (delta[idx % k] ?? 0), 0))
      const upper = s.upper.map((u, i) => u + (delta[s.labels[i] as number] ?? 0))
      const stale = new Array<boolean>(m.n).fill(true)
      // Centre-centre distances and half the distance from each centre to its nearest other.
      const cc = new Array<number>(k * k).fill(0)
      for (let a = 0; a < k; a++)
        for (let b = a + 1; b < k; b++)
          cc[a * k + b] = cc[b * k + a] = dist(centres[a] as number[], centres[b] as number[])
      let computed = (k * (k - 1)) / 2
      let skipped = 0
      const near = centres.map((_, a) => {
        let best = Number.POSITIVE_INFINITY
        for (let b = 0; b < k; b++) if (b !== a) best = Math.min(best, cc[a * k + b] as number)
        return best / 2
      })
      const labels = [...s.labels]
      for (let i = 0; i < m.n; i++) {
        if ((upper[i] as number) <= (near[labels[i] as number] as number)) {
          skipped += k - 1
          continue
        }
        for (let c = 0; c < k; c++) {
          const cur = labels[i] as number
          if (
            c === cur ||
            (upper[i] as number) <= (lower[i * k + c] as number) ||
            (upper[i] as number) <= (cc[cur * k + c] as number) / 2
          ) {
            if (c !== cur) skipped++
            continue
          }
          if (stale[i] === true) {
            const dc = dist(row(i), centres[cur] as number[])
            computed++
            upper[i] = dc
            lower[i * k + cur] = dc
            stale[i] = false
          }
          if (
            (upper[i] as number) > (lower[i * k + c] as number) ||
            (upper[i] as number) > (cc[cur * k + c] as number) / 2
          ) {
            const dc = dist(row(i), centres[c] as number[])
            computed++
            lower[i * k + c] = dc
            if (dc < (upper[i] as number)) {
              upper[i] = dc
              labels[i] = c
            }
          } else skipped++
        }
      }
      return { centres, labels, upper, lower, stale, shift, computed, skipped }
    },
    done: (s) => (s.shift <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        inertia: inertiaOf(s),
        moved: s.shift === Number.MAX_VALUE ? -1 : s.shift,
        computed: s.computed,
        skipped: s.skipped,
      },
      tensors: {
        centres: { role: 'state', value: s.centres.flat(), shape: [k, m.d] },
        labels: { role: 'state', value: s.labels, shape: [m.n] },
        upper: { role: 'state', value: s.upper, shape: [m.n] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ElkanState,
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Bisecting k-means

/** The state of bisecting k-means: the centres, the labels and the inertia of each cluster so far. */
export interface BisectingState {
  readonly centres: number[][]
  readonly labels: number[]
  /** The inertia of each cluster. */
  readonly inertias: number[]
}

/** Options of {@link bisectingKMeansAlgorithm}. */
export interface BisectingOptions {
  /** Which cluster to split next: the one with the largest inertia (default) or the most rows. */
  readonly strategy?: 'biggest_inertia' | 'largest_cluster'
  /** Lloyd iterations of each 2-means (default 300). */
  readonly maxIter?: number
  /** Starts of each 2-means, the lowest inertia wins (default 1). */
  readonly nInit?: number
}

function clusterStats(
  data: Float64Array,
  d: number,
  rows: readonly number[],
): { readonly centre: number[]; readonly inertia: number } {
  const centre = new Array<number>(d).fill(0)
  for (const i of rows) for (let q = 0; q < d; q++) centre[q] = (centre[q] ?? 0) + (data[i * d + q] ?? 0) / rows.length
  let inertia = 0
  for (const i of rows) for (let q = 0; q < d; q++) inertia += ((data[i * d + q] ?? 0) - (centre[q] ?? 0)) ** 2
  return { centre, inertia }
}

/**
 * Bisecting k-means (sklearn.cluster.BisectingKMeans) as a traced {@link Algorithm}: one cluster, then each step splits
 * one cluster in two with 2-means, the one with the largest inertia or the most rows, until there are k. It observes
 * the centres, labels and total inertia after each split.
 *
 * @param x - Points `[n, d]`.
 * @param k - The number of clusters.
 * @param options - Strategy, 2-means iterations and starts.
 * @returns The algorithm; k - 1 steps.
 * @example
 * ```ts
 * import { bisectingKMeansAlgorithm } from 'mdml-engine/cluster'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0], [1], [10], [11], [20], [21]]
 * runAlgorithm(bisectingKMeansAlgorithm(x, 3), { maxSteps: 2, key: key(0) }).state.centres.length // => 3
 * ```
 */
export function bisectingKMeansAlgorithm(
  x: TensorLike,
  k: number,
  options: BisectingOptions = {},
): Algorithm<BisectingState> {
  const m = matrixView(x, 'bisectingKMeansAlgorithm')
  if (!(Number.isInteger(k) && k >= 1 && k <= m.n))
    throw new MdmlError('value', `bisectingKMeansAlgorithm: nClusters must be in 1..${m.n}`)
  const strategy = options.strategy ?? 'biggest_inertia'
  return {
    name: 'bisecting-kmeans',
    config: { k, strategy },
    schema: channels({
      metrics: { inertia: { description: 'total inertia after the split', direction: 'lower' } },
      tensors: {
        centres: { role: 'state', description: 'the cluster centres', shape: ['c', 'd'] },
        labels: { role: 'state', description: 'the cluster of each row', shape: ['n'] },
      },
    }),
    init: () => {
      const all = Array.from({ length: m.n }, (_, i) => i)
      const { centre, inertia } = clusterStats(m.data, m.d, all)
      return { centres: [centre], labels: new Array<number>(m.n).fill(0), inertias: [inertia] }
    },
    step: (s, ctx) => {
      const sizes = s.centres.map((_, c) => s.labels.filter((l) => l === c).length)
      const scores = s.centres.map((_, c) =>
        (sizes[c] ?? 0) < 2 ? -1 : strategy === 'largest_cluster' ? (sizes[c] as number) : (s.inertias[c] as number),
      )
      const target = scores.indexOf(Math.max(...scores))
      const rows = s.labels.flatMap((l, i) => (l === target ? [i] : []))
      const sub = rows.map((i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
      let best: { readonly labels: number[]; readonly inertia: number } | undefined
      for (let r = 0; r < (options.nInit ?? 1); r++) {
        const key = foldIn(ctx.key, r)
        const out = runAlgorithm(lloydAlgorithm(sub, initialCentres(sub, 2, 'k-means++', key)), {
          maxSteps: options.maxIter ?? 300,
          key,
        })
        if (best === undefined || out.state.inertia < best.inertia)
          best = { labels: out.state.labels, inertia: out.state.inertia }
      }
      const labels = [...s.labels]
      const fresh = s.centres.length
      for (const [r, i] of rows.entries()) if ((best?.labels[r] ?? 0) === 1) labels[i] = fresh
      const left = clusterStats(
        m.data,
        m.d,
        rows.filter((i) => labels[i] === target),
      )
      const right = clusterStats(
        m.data,
        m.d,
        rows.filter((i) => labels[i] === fresh),
      )
      const centres = [...s.centres]
      centres[target] = left.centre
      centres.push(right.centre)
      const inertias = [...s.inertias]
      inertias[target] = left.inertia
      inertias.push(right.inertia)
      return { centres, labels, inertias }
    },
    done: (s) => (s.centres.length >= k ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { inertia: s.inertias.reduce((a, b) => a + b, 0) },
      tensors: {
        centres: { role: 'state', value: s.centres.flat(), shape: [s.centres.length, m.d] },
        labels: { role: 'state', value: s.labels, shape: [m.n] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BisectingState,
  }
}

/** Hyperparameters of {@link bisectingKMeans}. */
export interface BisectingKMeansHyper {
  readonly nClusters: number
  readonly bisectingStrategy: 'biggest_inertia' | 'largest_cluster'
  readonly maxIter: number
  readonly nInit: number
}

/** The fitted parameters of the centroid clusterers of this file. */
export type CentroidParams = { readonly centres: number[][]; readonly labels: number[]; readonly inertia: number }

/** The verbs of a fitted centroid clusterer: the nearest centre. */
export interface CentroidVerbs {
  readonly predict: (x: TensorLike) => Tensor
}

const centroidVerbs = (p: CentroidParams): CentroidVerbs => ({
  predict: (x) => tensor(nearestCentres(x, p.centres).labels),
})

/**
 * Bisecting k-means on the iterative base: one split per step.
 *
 * @index estimator Bisecting k-means; Elkan and kernel k-means beside it.
 * @example
 * ```ts
 * import { bisectingKMeans } from 'mdml-engine/cluster'
 * bisectingKMeans.with({ nClusters: 2 }).fit({ x: [[0], [1], [10], [11]] }).params.centres.length // => 2
 * ```
 */
export const bisectingKMeans: IterativeEstimator<BisectingKMeansHyper, BisectingState, CentroidParams, CentroidVerbs> =
  iterative<BisectingKMeansHyper, BisectingState, CentroidParams, CentroidVerbs>({
    name: 'cluster/bisecting-kmeans',
    kind: 'clusterer',
    defaults: { nClusters: 3, bisectingStrategy: 'biggest_inertia', maxIter: 300, nInit: 1 },
    hyperSpace: (h) =>
      dict({
        nClusters: int(1, 20, { default: h.nClusters, label: 'clusters' }),
        bisectingStrategy: choice(['biggest_inertia', 'largest_cluster'] as const, { default: h.bisectingStrategy }),
      }),
    algorithm: (data, h) =>
      bisectingKMeansAlgorithm(data.x, h.nClusters, {
        strategy: h.bisectingStrategy,
        maxIter: h.maxIter,
        nInit: h.nInit,
      }),
    maxSteps: (h) => h.nClusters,
    params: (s) => ({ centres: s.centres, labels: s.labels, inertia: s.inertias.reduce((a, b) => a + b, 0) }),
    verbs: centroidVerbs,
  })

/** Elkan's k-means on the iterative base: Lloyd's fixed point, with the distance savings on the tape. */
export interface ElkanHyper {
  readonly nClusters: number
  readonly init: 'k-means++' | 'random'
  readonly maxIter: number
  readonly tol: number
}

/**
 * k-means by Elkan's algorithm on the iterative base.
 *
 * @example
 * ```ts
 * import { elkanKMeans } from 'mdml-engine/cluster'
 * elkanKMeans.with({ nClusters: 2 }).fit({ x: [[0], [1], [10], [11]] }).params.centres.length // => 2
 * ```
 */
export const elkanKMeans: IterativeEstimator<ElkanHyper, ElkanState, CentroidParams, CentroidVerbs> = iterative<
  ElkanHyper,
  ElkanState,
  CentroidParams,
  CentroidVerbs
>({
  name: 'cluster/kmeans-elkan',
  kind: 'clusterer',
  defaults: { nClusters: 3, init: 'k-means++', maxIter: 300, tol: 1e-4 },
  hyperSpace: (h) => dict({ nClusters: int(2, 10, { default: h.nClusters, label: 'clusters' }) }),
  algorithm: (data, h, key) => elkanAlgorithm(data.x, initialCentres(data.x, h.nClusters, h.init, key), { tol: h.tol }),
  maxSteps: (h) => h.maxIter,
  params: (s, data) => {
    const m = matrixView(data.x, 'elkanKMeans')
    return {
      centres: s.centres,
      labels: s.labels,
      inertia: nearestCentres(toRows(m.data, m.n, m.d), s.centres).inertia,
    }
  },
  verbs: centroidVerbs,
})

// ---------------------------------------------------------------------------------------------------------------
// Kernel k-means

/** The state of kernel k-means: the labels, the feature-space inertia, and whether any label changed. */
export interface KernelKMeansState {
  readonly labels: number[]
  readonly objective: number
  readonly changed: boolean
}

/**
 * Feature-space squared distances from rows to cluster means: K(x, x) - 2 mean_j K(x, j) + mean_jl K(j, l).
 *
 * @param kxx - Self-kernels of the rows `[m]`.
 * @param kxt - Cross-kernels to the training rows `[m, n]`, flat.
 * @param ktt - The training Gram matrix `[n, n]`, flat.
 * @param labels - Training labels.
 * @param k - Clusters.
 * @returns Distances `[m, k]` flat (very large for empty clusters).
 * @example
 * ```ts
 * import { featureSpaceDistances } from 'mdml-engine/cluster'
 * featureSpaceDistances([1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1], 2) // => [0, 2, 2, 0]
 * ```
 */
export function featureSpaceDistances(
  kxx: ArrayLike<number>,
  kxt: ArrayLike<number>,
  ktt: ArrayLike<number>,
  labels: readonly number[],
  k: number,
): number[] {
  const n = labels.length
  const rows = kxx.length
  const members = Array.from({ length: k }, (_, c) => labels.flatMap((l, i) => (l === c ? [i] : [])))
  const within = members.map((ms) => {
    let t = 0
    for (const i of ms) for (const j of ms) t += ktt[i * n + j] ?? 0
    return t / Math.max(1, ms.length) ** 2
  })
  const out = new Array<number>(rows * k).fill(0)
  for (let i = 0; i < rows; i++)
    for (const [c, ms] of members.entries()) {
      let cross = 0
      for (const j of ms) cross += kxt[i * n + j] ?? 0
      out[i * k + c] = ms.length === 0 ? Number.MAX_VALUE : (kxx[i] ?? 0) - (2 * cross) / ms.length + (within[c] ?? 0)
    }
  return out
}

/**
 * Kernel k-means (Dhillon, Guan and Kulis 2004) as a traced {@link Algorithm}: Lloyd's algorithm in the feature space
 * of a kernel, from given labels; each step reassigns every row to the cluster whose feature-space mean is nearest.
 *
 * @param gram - The Gram matrix `[n, n]`, flat.
 * @param n - The number of rows.
 * @param initial - Initial labels.
 * @param k - The number of clusters.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { kernelKMeansAlgorithm } from 'mdml-engine/cluster'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const gram = [1, 0.9, 0, 0, 0.9, 1, 0, 0, 0, 0, 1, 0.9, 0, 0, 0.9, 1]
 * runAlgorithm(kernelKMeansAlgorithm(gram, 4, [0, 0, 0, 1], 2), { maxSteps: 10 }).state.labels // => [0, 0, 1, 1]
 * ```
 */
export function kernelKMeansAlgorithm(
  gram: ArrayLike<number>,
  n: number,
  initial: readonly number[],
  k: number,
): Algorithm<KernelKMeansState> {
  const diag = Array.from({ length: n }, (_, i) => gram[i * n + i] ?? 0)
  return {
    name: 'kernel-kmeans',
    config: { k, n },
    schema: channels({
      metrics: {
        inertia: { description: 'feature-space sum of squared distances to the cluster means', direction: 'lower' },
      },
      tensors: { labels: { role: 'state', description: 'the cluster of each row', shape: ['n'] } },
    }),
    init: () => ({ labels: [...initial], objective: Number.MAX_VALUE, changed: true }),
    step: (s) => {
      const d = featureSpaceDistances(diag, gram, gram, s.labels, k)
      let objective = 0
      const labels = Array.from({ length: n }, (_, i) => {
        let best = 0
        for (let c = 1; c < k; c++) if ((d[i * k + c] ?? 0) < (d[i * k + best] ?? 0)) best = c
        objective += d[i * k + best] ?? 0
        return best
      })
      return { labels, objective, changed: labels.some((l, i) => l !== s.labels[i]) }
    },
    done: (s, t) => (t > 0 && !s.changed ? 'converged' : undefined),
    observe: (s) => ({
      metrics: s.objective === Number.MAX_VALUE ? {} : { inertia: s.objective },
      tensors: { labels: { role: 'state', value: s.labels, shape: [n] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as KernelKMeansState,
  }
}

/**
 * The spectral relaxation of kernel k-means (Dhillon, Guan and Kulis 2004): k-means on the unit-length rows of the
 * leading eigenvectors of the Gram matrix, as initial labels.
 *
 * @param gram - The Gram matrix `[n, n]`, flat.
 * @param n - The number of rows.
 * @param k - The number of clusters.
 * @param key - Randomness for the k-means.
 * @returns Initial labels.
 * @example
 * ```ts
 * import { spectralRelaxation } from 'mdml-engine/cluster'
 * import { key } from 'mdml-engine/random'
 * spectralRelaxation([1, 0.9, 0, 0, 0.9, 1, 0, 0, 0, 0, 1, 0.9, 0, 0, 0.9, 1], 4, 2, key(0)).length // => 4
 * ```
 */
export function spectralRelaxation(gram: ArrayLike<number>, n: number, k: number, key: Key): number[] {
  const vectors = toFlat(eigh(fromStorage(Float64Array.from(gram), [n, n], 'float64')).eigenvectors)
  const rows = Array.from({ length: n }, (_, i) => {
    const row = Array.from({ length: k }, (_, q) => vectors[i * n + (n - 1 - q)] ?? 0)
    const norm = Math.hypot(...row) || 1
    return row.map((v) => v / norm)
  })
  const out = runAlgorithm(lloydAlgorithm(rows, initialCentres(rows, k, 'k-means++', key)), { maxSteps: 300, key })
  return nearestCentres(rows, out.state.centres).labels
}

/** Hyperparameters of {@link kernelKMeans}. */
export interface KernelKMeansHyper {
  readonly nClusters: number
  readonly kernel: PairwiseKernelMetric
  /** The kernel's gamma (`null`: 1 / d). */
  readonly gamma: number | null
  readonly maxIter: number
  /**
   * How the labels start: from k-means++ in the input space, or from the spectral relaxation (k-means on the leading
   * eigenvectors of the Gram matrix, which finds structure the input space hides).
   */
  readonly init: 'spectral' | 'k-means++'
  /** Starts; the lowest feature-space inertia wins. */
  readonly nInit: number
}

/** The fitted parameters of {@link kernelKMeans}: the training rows and labels (prediction is by feature-space distances). */
export type KernelKMeansParams = {
  readonly data: number[][]
  readonly labels: number[]
  readonly objective: number
  readonly gamma: number
  readonly kernel: PairwiseKernelMetric
}

const gramOf = (a: TensorLike, b: TensorLike | undefined, kernel: PairwiseKernelMetric, gamma: number): number[] =>
  toFlat(pairwiseKernels(a, b, kernel, { gamma }))

/**
 * Kernel k-means on the iterative base: Lloyd's algorithm in a kernel's feature space, from k-means++ labels, one
 * reassignment per step. It separates what k-means cannot, such as concentric rings, with an RBF kernel.
 *
 * @example
 * ```ts
 * import { kernelKMeans } from 'mdml-engine/cluster'
 * kernelKMeans.with({ nClusters: 2, gamma: 1 }).fit({ x: [[0], [0.1], [5], [5.1]] }).params.labels.length // => 4
 * ```
 */
export const kernelKMeans: IterativeEstimator<KernelKMeansHyper, KernelKMeansState, KernelKMeansParams, CentroidVerbs> =
  iterative<KernelKMeansHyper, KernelKMeansState, KernelKMeansParams, CentroidVerbs>({
    name: 'cluster/kernel-kmeans',
    kind: 'clusterer',
    defaults: { nClusters: 2, kernel: 'rbf', gamma: null, maxIter: 100, init: 'k-means++', nInit: 3 },
    hyperSpace: (h) =>
      dict({
        nClusters: int(1, 20, { default: h.nClusters, label: 'clusters' }),
        gamma: real(1e-3, 100, { log: true, default: h.gamma ?? 1, label: 'kernel gamma' }),
      }),
    algorithm: (data, h, key) => {
      const m = matrixView(data.x, 'kernelKMeans')
      const rows = toRows(m.data, m.n, m.d)
      const gram = gramOf(rows, undefined, h.kernel, h.gamma ?? 1 / m.d)
      const labels =
        h.init === 'spectral'
          ? spectralRelaxation(gram, m.n, h.nClusters, key)
          : nearestCentres(rows, initialCentres(rows, h.nClusters, 'k-means++', key)).labels
      return kernelKMeansAlgorithm(gram, m.n, labels, h.nClusters)
    },
    maxSteps: (h) => h.maxIter,
    restarts: { count: (h) => h.nInit, better: (a, b) => a.objective < b.objective },
    params: (s, data, h) => {
      const m = matrixView(data.x, 'kernelKMeans')
      return {
        data: toRows(m.data, m.n, m.d),
        labels: s.labels,
        objective: s.objective === Number.MAX_VALUE ? 0 : s.objective,
        gamma: h.gamma ?? 1 / m.d,
        kernel: h.kernel,
      }
    },
    verbs: (p) => {
      const k = Math.max(0, ...p.labels) + 1
      const ktt = gramOf(p.data, undefined, p.kernel, p.gamma)
      return {
        predict: (x) => {
          const v = matrixView(x, 'kernelKMeans.predict', { nFeatures: p.data[0]?.length ?? 0 })
          const rows = fromStorage(v.data, [v.n, v.d], 'float64')
          const kx = gramOf(rows, p.data, p.kernel, p.gamma)
          const kxx = Array.from(
            { length: v.n },
            (_, i) =>
              gramOf(toRows(v.data.subarray(i * v.d, (i + 1) * v.d), 1, v.d), undefined, p.kernel, p.gamma)[0] ?? 0,
          )
          const d = featureSpaceDistances(kxx, kx, ktt, p.labels, k)
          return tensor(
            Array.from({ length: v.n }, (_, i) => {
              let best = 0
              for (let c = 1; c < k; c++) if ((d[i * k + c] ?? 0) < (d[i * k + best] ?? 0)) best = c
              return best
            }),
          )
        },
      }
    },
  })
