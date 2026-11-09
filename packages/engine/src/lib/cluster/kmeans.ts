/**
 * k-means as algorithms: Lloyd's algorithm (assign each row to its nearest centre, move centres to the means) and
 * mini-batch k-means, both `Algorithm`s for `runAlgorithm` whose state keeps no past (the tape records it), with the
 * initial centres (k-means++, random rows or given) and the nearest-centre assignment they share, the standalone
 * k-means++ seeding (`kmeansPlusPlus`) and the estimators `kMeans` and `miniBatchKMeans` (fit, predict, transform to
 * distances, score, `nInit` restarts) on the iterative base.
 */

import { OmaiError, toRows } from '../base/index.ts'
import { type IterativeEstimator, iterative, type LearnData, matrixView } from '../learn/index.ts'
import { geometry } from '../numerics/index.ts'
import { kmeansCodebook } from '../numerics/spatial/index.ts'
import { type Key, subsample, uniform } from '../random/index.ts'
import { dict, int } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, tensor, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'

/** Rows of a matrix as plain arrays. */
export type Rows = readonly (readonly number[])[]

/** How the initial centres are chosen: k-means++ seeding, k distinct random rows, or given centres. */
export type KMeansInit = 'k-means++' | 'random' | Rows

/** The state of Lloyd's algorithm: the centres, the labels they give, the inertia and the last squared shift. */
export interface LloydState {
  readonly centres: number[][]
  readonly labels: number[]
  readonly inertia: number
  readonly shift: number
}

const kmeansSchema = channels({
  metrics: {
    inertia: { description: 'sum of squared distances of the rows to their nearest centre', direction: 'lower' },
    moved: { description: 'squared distance the centres moved this step (-1 before the first step)' },
  },
  tensors: {
    centres: { role: 'state', description: 'the cluster centres', shape: ['k', 'd'] },
    labels: { role: 'state', description: 'the nearest centre of each row', shape: ['n'] },
  },
})

/** The squared distance between two sets of centres. */
function squaredShift(a: Rows, b: Rows): number {
  let shift = 0
  for (const [j, row] of a.entries()) for (const [q, v] of row.entries()) shift += (v - (b[j]?.[q] ?? 0)) ** 2
  return shift
}

/**
 * Squared Euclidean distances from every row to every centre, `[n, k]` row-major.
 *
 * @param x - Points `[n, d]`.
 * @param centres - Centres `[k, d]`.
 * @returns The distances, flat.
 * @example
 * ```ts
 * import { squaredDistancesToCentres } from 'omai-engine/cluster'
 * squaredDistancesToCentres([[0, 0], [3, 4]], [[0, 0]]) // => [0, 25]
 * ```
 */
export function squaredDistancesToCentres(x: TensorLike, centres: Rows): number[] {
  return toFlat(geometry.cdist(x, centres as number[][], 'sqeuclidean'))
}

/**
 * The nearest centre of each row and the inertia (sum of squared distances to it).
 *
 * @param x - Points `[n, d]`.
 * @param centres - Centres `[k, d]`.
 * @returns Labels and inertia.
 * @example
 * ```ts
 * import { nearestCentres } from 'omai-engine/cluster'
 * nearestCentres([[0, 0], [10, 0], [9, 1]], [[0, 0], [10, 0]]).labels // => [0, 1, 1]
 * ```
 */
export function nearestCentres(x: TensorLike, centres: Rows): { readonly labels: number[]; readonly inertia: number } {
  const k = centres.length
  const dist = squaredDistancesToCentres(x, centres)
  const n = dist.length / Math.max(k, 1)
  const labels: number[] = []
  let inertia = 0
  for (let i = 0; i < n; i++) {
    let best = 0
    for (let j = 1; j < k; j++) if ((dist[i * k + j] ?? 0) < (dist[i * k + best] ?? 0)) best = j
    labels.push(best)
    inertia += dist[i * k + best] ?? 0
  }
  return { labels, inertia }
}

/** The mean of the per-feature variances (sklearn's scale for the k-means tolerance). */
function meanVariance(data: Float64Array, n: number, d: number): number {
  let total = 0
  for (let j = 0; j < d; j++) {
    let m = 0
    for (let i = 0; i < n; i++) m += data[i * d + j] ?? 0
    m /= n
    let v = 0
    for (let i = 0; i < n; i++) v += ((data[i * d + j] ?? 0) - m) ** 2
    total += v / n
  }
  return d === 0 ? 0 : total / d
}

/**
 * Initial centres: k-means++ seeding (through the codebook trainer of `numerics/spatial`), k distinct random rows, or
 * the given centres.
 *
 * @param x - Points `[n, d]`.
 * @param k - The number of centres.
 * @param init - The rule or the centres.
 * @param key - Randomness for the seeding.
 * @returns The centres `[k, d]`.
 * @example
 * ```ts
 * import { initialCentres } from 'omai-engine/cluster'
 * import { key } from 'omai-engine/random'
 * initialCentres([[0, 0], [1, 1], [5, 5]], 2, [[0, 0], [5, 5]], key(0)) // => [[0, 0], [5, 5]]
 * ```
 */
export function initialCentres(x: TensorLike, k: number, init: KMeansInit, key: Key): number[][] {
  const m = matrixView(x, 'initialCentres')
  if (!(Number.isInteger(k) && k >= 1 && k <= m.n))
    throw new OmaiError('value', `initialCentres: nClusters must be in 1..${m.n}, got ${k}`)
  if (typeof init !== 'string') {
    if (init.length !== k || init.some((r) => r.length !== m.d))
      throw new OmaiError('shape', `initialCentres: init must be [${k}, ${m.d}]`)
    return init.map((r) => [...r])
  }
  if (init === 'random') {
    const rows = subsample(key, m.n, k).data
    return Array.from(rows, (i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
  }
  const seeded = kmeansCodebook(fromStorage(m.data, [m.n, m.d], 'float64'), k, { key, maxIterations: 0 })
  return toRows(toFlat(seeded.centroids), k, m.d)
}

/**
 * Lloyd's algorithm as a traced {@link Algorithm}: each step assigns every row to its nearest centre and moves each
 * centre to the mean of its rows (one iteration of the codebook trainer of `numerics/spatial`), observing centres, labels,
 * inertia and how far the centres moved.
 *
 * @param x - Points `[n, d]`.
 * @param init - Initial centres `[k, d]`.
 * @param options - Tolerance on the squared centre shift, relative to the mean feature variance (default 1e-4).
 * @returns The algorithm; run it with `runAlgorithm`.
 * @index algorithm Lloyd's k-means, one step at a time; the `kMeans` estimator wraps it.
 * @example
 * ```ts
 * import { lloydAlgorithm } from 'omai-engine/cluster'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[0, 0], [0, 1], [10, 0], [10, 1]]
 * runAlgorithm(lloydAlgorithm(x, [[0, 0], [10, 1]]), { maxSteps: 20 }).state.labels // => [0, 0, 1, 1]
 * ```
 */
export function lloydAlgorithm(
  x: TensorLike,
  init: Rows,
  options: { readonly tol?: number } = {},
): Algorithm<LloydState> {
  const m = matrixView(x, 'lloydAlgorithm')
  const data = fromStorage(m.data, [m.n, m.d], 'float64')
  const k = init.length
  const tol = (options.tol ?? 1e-4) * meanVariance(m.data, m.n, m.d)
  return {
    name: 'kmeans-lloyd',
    schema: kmeansSchema,
    config: { k, tol: options.tol ?? 1e-4 },
    init: () => {
      const centres = init.map((r) => [...r])
      const { labels, inertia } = nearestCentres(data, centres)
      return {
        centres,
        labels,
        inertia,
        shift: Number.POSITIVE_INFINITY,
      }
    },
    step: (s) => {
      const book = kmeansCodebook(data, k, { init: s.centres as number[][], maxIterations: 1 })
      const centres = toRows(toFlat(book.centroids), k, m.d)
      const shift = squaredShift(centres, s.centres)
      const labels = Array.from(book.assignments)
      return {
        centres,
        labels,
        inertia: book.inertia,
        shift,
      }
    },
    done: (s) => (s.shift <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { inertia: s.inertia, moved: Number.isFinite(s.shift) ? s.shift : -1 },
      tensors: {
        centres: { role: 'state', value: s.centres.flat(), shape: [k, m.d] },
        labels: { role: 'state', value: s.labels, shape: [m.n] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as LloydState,
  }
}

/** The state of mini-batch k-means: centres, per-centre counts, the labels of all rows and the inertia. */
export interface MiniBatchState {
  readonly centres: number[][]
  readonly counts: number[]
  readonly inertia: number
  readonly labels: number[]
  readonly moved: number
}

/**
 * Mini-batch k-means (Sculley 2010; sklearn.cluster.MiniBatchKMeans) as a traced algorithm: each step draws a batch,
 * assigns it to the nearest centres and moves each centre towards its batch rows with a per-centre learning rate of
 * 1 / count. The inertia is the full-data inertia after each step.
 *
 * @param x - Points `[n, d]`.
 * @param init - Initial centres `[k, d]`.
 * @param batchSize - Rows per batch.
 * @returns The algorithm.
 * @index algorithm Mini-batch k-means, one batch at a time; the `miniBatchKMeans` estimator wraps it.
 * @example
 * ```ts
 * import { miniBatchKMeansAlgorithm } from 'omai-engine/cluster'
 * miniBatchKMeansAlgorithm([[0, 0], [10, 1]], [[0, 0], [10, 1]], 2).name // => 'mini-batch-kmeans'
 * ```
 */
export function miniBatchKMeansAlgorithm(x: TensorLike, init: Rows, batchSize: number): Algorithm<MiniBatchState> {
  const m = matrixView(x, 'miniBatchKMeans')
  const data = fromStorage(m.data, [m.n, m.d], 'float64')
  const b = Math.min(Math.max(1, Math.floor(batchSize)), m.n)
  return {
    name: 'mini-batch-kmeans',
    schema: kmeansSchema,
    config: { k: init.length, batchSize: b },
    init: () => {
      const centres = init.map((r) => [...r])
      const { labels, inertia } = nearestCentres(data, centres)
      return {
        centres,
        counts: centres.map(() => 0),
        inertia,
        labels,
        moved: Number.POSITIVE_INFINITY,
      }
    },
    step: (s, ctx) => {
      const rows = Array.from(subsample(ctx.key, m.n, b).data)
      const batch = rows.map((i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
      const { labels: near } = nearestCentres(batch, s.centres)
      const centres = s.centres.map((c) => [...c])
      const counts = [...s.counts]
      for (const [r, row] of batch.entries()) {
        const j = near[r] ?? 0
        counts[j] = (counts[j] ?? 0) + 1
        const eta = 1 / (counts[j] ?? 1)
        const c = centres[j] as number[]
        for (let q = 0; q < m.d; q++) c[q] = (1 - eta) * (c[q] ?? 0) + eta * (row[q] ?? 0)
      }
      const { labels, inertia } = nearestCentres(data, centres)
      return {
        centres,
        counts,
        inertia,
        labels,
        moved: squaredShift(centres, s.centres),
      }
    },
    observe: (s) => ({
      metrics: { inertia: s.inertia, moved: Number.isFinite(s.moved) ? s.moved : -1 },
      tensors: {
        centres: { role: 'state', value: s.centres.flat(), shape: [init.length, m.d] },
        labels: { role: 'state', value: s.labels, shape: [m.n] },
      },
    }),
  }
}

/**
 * k-means++ seeding (Arthur and Vassilvitskii, with scikit-learn's greedy variant: at each of the `k - 1` further
 * steps several candidates are drawn with probability proportional to the squared distance to the nearest centre
 * chosen so far, and the one that lowers the total squared distance most is kept).
 *
 * @param x - Points `[n, d]`.
 * @param nClusters - The number of centres, in `1..n`.
 * @param options - The key, and the candidates per step (default `2 + floor(ln k)`, scikit-learn's).
 * @returns The centres `[k, d]` and the row indices they were taken from.
 * @throws OmaiError with code `value` unless `nClusters` is an integer in `1..n`.
 * @index function k-means++ seeding as a standalone function: centres and the rows they came from.
 * @example
 * ```ts
 * import { kmeansPlusPlus } from 'omai-engine/cluster'
 * import { key } from 'omai-engine/random'
 * const { centres, indices } = kmeansPlusPlus([[0], [0.1], [10], [10.1]], 2, { key: key(0) })
 * centres.length // => 2
 * indices.length // => 2
 * ```
 */
export function kmeansPlusPlus(
  x: TensorLike,
  nClusters: number,
  options: { readonly key: Key; readonly nLocalTrials?: number },
): { readonly centres: number[][]; readonly indices: number[] } {
  const m = matrixView(x, 'kmeansPlusPlus')
  const { n, d, data } = m
  const k = nClusters
  if (!(Number.isInteger(k) && k >= 1 && k <= n))
    throw new OmaiError('value', `kmeansPlusPlus: nClusters must be in 1..${n}, got ${k}`)
  const trials = Math.max(1, Math.floor(options.nLocalTrials ?? 2 + Math.floor(Math.log(k))))
  const u = uniform(options.key, [1 + (k - 1) * trials]).data
  const sq = (i: number, j: number): number => {
    let s = 0
    for (let q = 0; q < d; q++) s += ((data[i * d + q] ?? 0) - (data[j * d + q] ?? 0)) ** 2
    return s
  }
  const first = Math.min(n - 1, Math.floor((u[0] ?? 0) * n))
  const indices = [first]
  const closest = Float64Array.from({ length: n }, (_, i) => sq(i, first))
  let pot = closest.reduce((a, v) => a + v, 0)
  for (let c = 1; c < k; c++) {
    const cumulative = new Float64Array(n)
    let run = 0
    for (let i = 0; i < n; i++) {
      run += closest[i] ?? 0
      cumulative[i] = run
    }
    let bestPot = Number.POSITIVE_INFINITY
    let bestIndex = 0
    for (let t = 0; t < trials; t++) {
      const target = (u[1 + (c - 1) * trials + t] ?? 0) * pot
      let lo = 0
      let hi = n - 1
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if ((cumulative[mid] ?? 0) < target) lo = mid + 1
        else hi = mid
      }
      let candidatePot = 0
      for (let i = 0; i < n; i++) candidatePot += Math.min(closest[i] ?? 0, sq(i, lo))
      if (candidatePot < bestPot) {
        bestPot = candidatePot
        bestIndex = lo
      }
    }
    indices.push(bestIndex)
    for (let i = 0; i < n; i++) closest[i] = Math.min(closest[i] ?? 0, sq(i, bestIndex))
    pot = bestPot
  }
  return { centres: indices.map((i) => Array.from(data.subarray(i * d, (i + 1) * d))), indices }
}

// ---------------------------------------------------------------------------------------------------------------
// Estimators

/** The fitted parameters of {@link kMeans} and {@link miniBatchKMeans}: the centres, and the labels and inertia of the training rows. */
export type KMeansParams = { readonly centres: number[][]; readonly labels: number[]; readonly inertia: number }

/** The verbs of a fitted k-means: the nearest centre, the distances to every centre, and the negated inertia. */
export interface KMeansVerbs {
  /** The nearest centre of each row `[m]`. */
  readonly predict: (x: TensorLike) => Tensor
  /** The Euclidean distance of each row to every centre `[m, k]`. */
  readonly transform: (x: TensorLike) => Tensor
  /** Minus the inertia of the rows (scikit-learn's `score`: higher is better). */
  readonly score: (data: LearnData) => number
  readonly nFeaturesIn: number
  readonly featureNamesOut: (input?: readonly string[]) => string[]
}

const kmeansVerbs = (p: KMeansParams): KMeansVerbs => ({
  predict: (x) => tensor(nearestCentres(x, p.centres).labels),
  transform: (x) => {
    const k = p.centres.length
    const flat = squaredDistancesToCentres(x, p.centres).map(Math.sqrt)
    return fromStorage(Float64Array.from(flat), [flat.length / Math.max(k, 1), k], 'float64')
  },
  score: (data) => -nearestCentres(data.x, p.centres).inertia,
  nFeaturesIn: p.centres[0]?.length ?? 0,
  featureNamesOut: () => p.centres.map((_, j) => `kmeans${j}`),
})

const kmeansParams = (centres: number[][], data: LearnData): KMeansParams => ({
  centres,
  ...nearestCentres(data.x, centres),
})

/** Hyperparameters of {@link kMeans}. */
export interface KMeansHyper {
  /** The number of clusters (default 8). */
  readonly nClusters: number
  /** The initial centres: `'k-means++'` (default), `'random'` rows, or given centres. */
  readonly init: KMeansInit
  /** The number of restarts from different initial centres; the lowest inertia wins (default 1; given centres run once). */
  readonly nInit: number
  /** The step budget (default 300). */
  readonly maxIter: number
  /** The tolerance on the squared centre shift, relative to the mean feature variance (default 1e-4). */
  readonly tol: number
}

/**
 * k-means (scikit-learn's `KMeans` with the Lloyd algorithm) on the iterative base: `nInit` restarts from the chosen
 * initial centres, the one with the lowest inertia kept. The fitted value predicts the nearest centre, transforms to
 * the distances to the centres and scores as minus the inertia; `fit(...).params` holds the centres, the training
 * labels and the inertia (what `k_means` returns). `.method(data)` and `.session(data)` step the same fit.
 *
 * @index estimator k-means with restarts: predict, transform to centre distances, score as minus inertia.
 * @example
 * ```ts
 * import { kMeans } from 'omai-engine/cluster'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 0], [0, 1], [10, 0], [10, 1]]
 * const f = kMeans.with({ nClusters: 2, init: [[0, 0], [10, 1]] }).fit({ x })
 * f.params.labels // => [0, 0, 1, 1]
 * toFlat(f.predict([[9, 0]])) // => [1]
 * f.params.inertia // => 1
 * ```
 */
export const kMeans: IterativeEstimator<KMeansHyper, LloydState, KMeansParams, KMeansVerbs> = iterative<
  KMeansHyper,
  LloydState,
  KMeansParams,
  KMeansVerbs
>({
  name: 'cluster/kmeans',
  kind: 'clusterer',
  defaults: { nClusters: 8, init: 'k-means++', nInit: 1, maxIter: 300, tol: 1e-4 },
  hyperSpace: (h) => dict({ nClusters: int(1, 20, { default: h.nClusters, label: 'clusters' }) }),
  algorithm: (data, h, key) => lloydAlgorithm(data.x, initialCentres(data.x, h.nClusters, h.init, key), { tol: h.tol }),
  maxSteps: (h) => h.maxIter,
  restarts: { count: (h) => (typeof h.init === 'string' ? h.nInit : 1), better: (a, b) => a.inertia < b.inertia },
  params: (s, data) => kmeansParams(s.centres, data),
  verbs: kmeansVerbs,
})

/** Hyperparameters of {@link miniBatchKMeans}. */
export interface MiniBatchKMeansHyper {
  readonly nClusters: number
  readonly init: KMeansInit
  /** Restarts from different initial centres; the lowest full-data inertia wins (default 1). */
  readonly nInit: number
  /** Rows per batch (default 1024). */
  readonly batchSize: number
  /** The number of batches, one per step (default 100). */
  readonly maxIter: number
}

/**
 * Mini-batch k-means (scikit-learn's `MiniBatchKMeans`) on the iterative base: a fixed number of batches, one per step,
 * each moving the centres towards its rows by a per-centre rate of 1 / count. The fitted value is that of {@link kMeans}.
 *
 * @index estimator Mini-batch k-means with restarts; same verbs as kMeans.
 * @example
 * ```ts
 * import { miniBatchKMeans } from 'omai-engine/cluster'
 * const x = [[0, 0], [0, 1], [10, 0], [10, 1]]
 * const f = miniBatchKMeans.with({ nClusters: 2, init: [[0, 0], [10, 1]], batchSize: 4, maxIter: 5 }).fit({ x })
 * f.params.labels // => [0, 0, 1, 1]
 * ```
 */
export const miniBatchKMeans: IterativeEstimator<MiniBatchKMeansHyper, MiniBatchState, KMeansParams, KMeansVerbs> =
  iterative<MiniBatchKMeansHyper, MiniBatchState, KMeansParams, KMeansVerbs>({
    name: 'cluster/mini-batch-kmeans',
    kind: 'clusterer',
    defaults: { nClusters: 8, init: 'k-means++', nInit: 1, batchSize: 1024, maxIter: 100 },
    hyperSpace: (h) => dict({ nClusters: int(1, 20, { default: h.nClusters, label: 'clusters' }) }),
    algorithm: (data, h, key) =>
      miniBatchKMeansAlgorithm(data.x, initialCentres(data.x, h.nClusters, h.init, key), h.batchSize),
    maxSteps: (h) => h.maxIter,
    restarts: { count: (h) => (typeof h.init === 'string' ? h.nInit : 1), better: (a, b) => a.inertia < b.inertia },
    params: (s, data) => kmeansParams(s.centres, data),
    verbs: kmeansVerbs,
  })
