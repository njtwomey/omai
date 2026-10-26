/**
 * k-means as algorithms: Lloyd's algorithm (assign each row to its nearest centre, move centres to the means) and
 * mini-batch k-means, both `Algorithm`s for `runAlgorithm` whose state keeps no past (the tape records it), with the
 * initial centres (k-means++, random rows or given) and the nearest-centre assignment they share.
 */

import { MdmlError } from '../base/index.ts'
import { matrixView } from '../learn/index.ts'
import { geometry } from '../numerics/index.ts'
import { kmeansCodebook } from '../numerics/spatial/index.ts'
import { type Key, subsample } from '../random/index.ts'
import { fromStorage, type TensorLike, toFlat } from '../tensor/index.ts'
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

const rowsOf = (flat: ArrayLike<number>, n: number, d: number): number[][] =>
  Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => flat[i * d + j] ?? 0))

/**
 * Squared Euclidean distances from every row to every centre, `[n, k]` row-major.
 *
 * @param x - Points `[n, d]`.
 * @param centres - Centres `[k, d]`.
 * @returns The distances, flat.
 * @example
 * ```ts
 * import { squaredDistancesToCentres } from 'mdml-engine/cluster'
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
 * import { nearestCentres } from 'mdml-engine/cluster'
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
 * import { initialCentres } from 'mdml-engine/cluster'
 * import { key } from 'mdml-engine/random'
 * initialCentres([[0, 0], [1, 1], [5, 5]], 2, [[0, 0], [5, 5]], key(0)) // => [[0, 0], [5, 5]]
 * ```
 */
export function initialCentres(x: TensorLike, k: number, init: KMeansInit, key: Key): number[][] {
  const m = matrixView(x, 'initialCentres')
  if (!(Number.isInteger(k) && k >= 1 && k <= m.n))
    throw new MdmlError('value', `initialCentres: nClusters must be in 1..${m.n}, got ${k}`)
  if (typeof init !== 'string') {
    if (init.length !== k || init.some((r) => r.length !== m.d))
      throw new MdmlError('shape', `initialCentres: init must be [${k}, ${m.d}]`)
    return init.map((r) => [...r])
  }
  if (init === 'random') {
    const rows = subsample(key, m.n, k).data
    return Array.from(rows, (i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
  }
  const seeded = kmeansCodebook(fromStorage(m.data, [m.n, m.d], 'float64'), k, { key, maxIterations: 0 })
  return rowsOf(toFlat(seeded.centroids), k, m.d)
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
 * @example
 * ```ts
 * import { lloydAlgorithm } from 'mdml-engine/cluster'
 * import { runAlgorithm } from 'mdml-engine/trace'
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
      const centres = rowsOf(toFlat(book.centroids), k, m.d)
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
 * @example
 * ```ts
 * import { miniBatchKMeansAlgorithm } from 'mdml-engine/cluster'
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
