import { WhiteboxError } from '../../lib/base/index.ts'
import { type ClusteringSteps, clusteringSteps } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  type FitOptions,
  fitRunOptions,
  matrixView,
} from '../../lib/learn/index.ts'
import { geometry } from '../../lib/numerics/index.ts'
import { kmeansCodebook } from '../../lib/numerics/spatial/index.ts'
import { foldIn, type Key, keys, subsample } from '../../lib/random/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'

/** Rows of a matrix as plain arrays. */
export type Rows = readonly (readonly number[])[]

/** How the initial centres are chosen: k-means++ seeding, k distinct random rows, or given centres. */
export type KMeansInit = 'k-means++' | 'random' | Rows

/** Hyperparameters of k-means. */
export type KMeansHyper = {
  /** The number of clusters k (default 3). */
  readonly nClusters?: number
  /** Initial centres (default k-means++). */
  readonly init?: KMeansInit
  /** Restarts from different seeds; the lowest inertia wins (default 1). */
  readonly nInit?: number
  /** Iteration cap (default 300). */
  readonly maxIter?: number
  /** Convergence when the squared centre shift is below tol times the mean feature variance (default 1e-4). */
  readonly tol?: number
}

/** The learnt parameters of a centroid clustering. */
export type CentroidParams = {
  readonly centres: number[][]
  readonly labels: number[]
  readonly inertia: number
  readonly nIter: number
}

/** A fitted centroid clustering: nearest-centre labels, distances to the centres and the negative inertia. */
export interface CentroidVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly transform: (x: TensorLike) => Tensor
  readonly score: (data: { readonly x: TensorLike }) => number
  readonly nFeaturesIn: number
}

/** A fitted k-means model. */
export type FittedKMeans = ExplainedFitted<CentroidParams, CentroidVerbs>

/** The state of Lloyd's algorithm, with the history the exposition records. */
export interface LloydState {
  readonly centres: number[][]
  readonly labels: number[]
  readonly inertia: number
  readonly shift: number
  readonly history: { readonly assignments: number[][]; readonly centres: number[][][]; readonly objective: number[] }
}

const rowsOf = (flat: ArrayLike<number>, n: number, d: number): number[][] =>
  Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => flat[i * d + j] ?? 0))

/**
 * Squared Euclidean distances from every row to every centre, `[n, k]` row-major.
 *
 * @param x - Points `[n, d]`.
 * @param centres - Centres `[k, d]`.
 * @returns The distances, flat.
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
 */
export function initialCentres(x: TensorLike, k: number, init: KMeansInit, key: Key): number[][] {
  const m = matrixView(x, 'initialCentres')
  if (!(Number.isInteger(k) && k >= 1 && k <= m.n))
    throw new WhiteboxError('value', `initialCentres: nClusters must be in 1..${m.n}, got ${k}`)
  if (typeof init !== 'string') {
    if (init.length !== k || init.some((r) => r.length !== m.d))
      throw new WhiteboxError('shape', `initialCentres: init must be [${k}, ${m.d}]`)
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
 * centre to the mean of its rows (one iteration of the codebook trainer of `numerics/spatial`), recording assignments,
 * centres and inertia.
 *
 * @param x - Points `[n, d]`.
 * @param init - Initial centres `[k, d]`.
 * @param options - Tolerance on the squared centre shift, relative to the mean feature variance (default 1e-4).
 * @returns The algorithm; run it with `runAlgorithm`.
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
    config: { k, tol: options.tol ?? 1e-4 },
    init: () => {
      const centres = init.map((r) => [...r])
      const { labels, inertia } = nearestCentres(data, centres)
      return {
        centres,
        labels,
        inertia,
        shift: Number.POSITIVE_INFINITY,
        history: { assignments: [labels], centres: [centres], objective: [inertia] },
      }
    },
    step: (s) => {
      const book = kmeansCodebook(data, k, { init: s.centres as number[][], maxIterations: 1 })
      const centres = rowsOf(toFlat(book.centroids), k, m.d)
      let shift = 0
      for (let j = 0; j < k; j++)
        for (let q = 0; q < m.d; q++) shift += ((centres[j]?.[q] ?? 0) - (s.centres[j]?.[q] ?? 0)) ** 2
      const labels = Array.from(book.assignments)
      return {
        centres,
        labels,
        inertia: book.inertia,
        shift,
        history: {
          assignments: [...s.history.assignments, labels],
          centres: [...s.history.centres, centres],
          objective: [...s.history.objective, book.inertia],
        },
      }
    },
    done: (s) => (s.shift <= tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: { inertia: s.inertia, shift: Number.isFinite(s.shift) ? s.shift : -1 } }),
  }
}

/**
 * The clustering-steps record of a Lloyd run.
 *
 * @param method - The method name recorded.
 * @param s - The final state.
 * @returns The record.
 */
export function lloydRecord(method: string, s: LloydState): ClusteringSteps {
  return clusteringSteps({
    method,
    assignments: s.history.assignments,
    centres: s.history.centres,
    objective: s.history.objective,
  })
}

/**
 * The verbs of a centroid model.
 *
 * @param p - The parameters.
 * @returns Nearest-centre labels, distances to the centres and the negative inertia.
 */
export function centroidVerbs(p: CentroidParams): CentroidVerbs {
  const d = p.centres[0]?.length ?? 0
  return {
    predict: (x) => {
      matrixView(x, 'predict', { nFeatures: d })
      const labels = nearestCentres(x, p.centres).labels
      return fromStorage(Int32Array.from(labels), [labels.length], 'int32')
    },
    transform: (x) => {
      const n = matrixView(x, 'transform', { nFeatures: d }).n
      return fromStorage(
        Float64Array.from(squaredDistancesToCentres(x, p.centres), Math.sqrt),
        [n, p.centres.length],
        'float64',
      )
    },
    score: (data) => -nearestCentres(data.x, p.centres).inertia,
    nFeaturesIn: d,
  }
}

const kmeansSpace = (h: KMeansHyper) =>
  dict({
    nClusters: int(1, 20, { default: h.nClusters ?? 3, label: 'clusters' }),
    init: choice(['k-means++', 'random'], { default: typeof h.init === 'string' ? h.init : 'k-means++' }),
    nInit: int(1, 20, { default: h.nInit ?? 1, label: 'restarts' }),
    maxIter: int(1, 1000, { default: h.maxIter ?? 300, label: 'iterations' }),
    tol: real(1e-8, 1e-1, { log: true, default: h.tol ?? 1e-4 }),
  })

/**
 * k-means clustering (sklearn.cluster.KMeans): k-means++ or random seeding, Lloyd iterations traced through
 * `runAlgorithm`, the best of `nInit` restarts. Its exposition is a `clusteringSteps` record (assignments, centres and
 * inertia per iteration).
 *
 * @param hyper - Number of clusters, initialisation, restarts, iteration cap and tolerance.
 * @returns The estimator.
 */
export function kmeans(hyper: KMeansHyper = {}): Estimator<KMeansHyper, FittedKMeans> {
  return defineExplainedEstimator<KMeansHyper, CentroidParams, CentroidVerbs>(
    {
      name: 'kmeans',
      kind: 'clusterer',
      hyperSpace: kmeansSpace,
      fit: (data, h, options) => {
        const run = fitRunOptions(options, 'kmeans')
        const k = h.nClusters ?? 3
        const init = h.init ?? 'k-means++'
        const restarts = typeof init === 'string' ? (h.nInit ?? 1) : 1
        let best: { state: LloydState; steps: number } | undefined
        for (const [r, rk] of keys(run.key as Key, restarts).entries()) {
          const start = initialCentres(data.x, k, init, rk)
          const out = runAlgorithm(lloydAlgorithm(data.x, start, { tol: h.tol ?? 1e-4 }), {
            ...run,
            key: rk,
            maxSteps: h.maxIter ?? 300,
            ...(restarts > 1 ? { runId: `kmeans-${r}` } : {}),
          })
          if (best === undefined || out.state.inertia < best.state.inertia)
            best = { state: out.state, steps: out.steps }
        }
        const s = (best as { state: LloydState; steps: number }).state
        return {
          params: { centres: s.centres, labels: s.labels, inertia: s.inertia, nIter: best?.steps ?? 0 },
          exposition: [lloydRecord('kmeans', s)],
        }
      },
      methods: centroidVerbs,
    },
    hyper,
  )
}

/** k-means as a class: `new KMeans({ nClusters: 3 }).fit(x).predict(x)`. */
export class KMeans extends EstimatorClass<KMeansHyper, FittedKMeans> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link kmeans}.
   */
  constructor(hyper: KMeansHyper = {}) {
    super(kmeans(hyper))
  }
}

/** Hyperparameters of mini-batch k-means. */
export type MiniBatchKMeansHyper = KMeansHyper & {
  /** Rows per mini-batch (default 64). */
  readonly batchSize?: number
}

/** The state of mini-batch k-means: centres, per-centre counts and the history. */
interface MiniBatchState {
  readonly centres: number[][]
  readonly counts: number[]
  readonly inertia: number
  readonly history: LloydState['history']
}

/**
 * Mini-batch k-means (Sculley 2010; sklearn.cluster.MiniBatchKMeans) as a traced algorithm: each step draws a batch,
 * assigns it to the nearest centres and moves each centre towards its batch rows with a per-centre learning rate of
 * 1 / count. The recorded objective is the full-data inertia after each step.
 *
 * @param x - Points `[n, d]`.
 * @param init - Initial centres `[k, d]`.
 * @param batchSize - Rows per batch.
 * @returns The algorithm.
 */
export function miniBatchKMeansAlgorithm(x: TensorLike, init: Rows, batchSize: number): Algorithm<MiniBatchState> {
  const m = matrixView(x, 'miniBatchKMeans')
  const data = fromStorage(m.data, [m.n, m.d], 'float64')
  const b = Math.min(Math.max(1, Math.floor(batchSize)), m.n)
  return {
    name: 'mini-batch-kmeans',
    config: { k: init.length, batchSize: b },
    init: () => {
      const centres = init.map((r) => [...r])
      const { labels, inertia } = nearestCentres(data, centres)
      return {
        centres,
        counts: centres.map(() => 0),
        inertia,
        history: { assignments: [labels], centres: [centres], objective: [inertia] },
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
        history: {
          assignments: [...s.history.assignments, labels],
          centres: [...s.history.centres, centres],
          objective: [...s.history.objective, inertia],
        },
      }
    },
    observe: (s) => ({ metrics: { inertia: s.inertia } }),
  }
}

/**
 * Mini-batch k-means (sklearn.cluster.MiniBatchKMeans): k-means++ seeding, then `maxIter` traced mini-batch updates.
 * Its exposition is a `clusteringSteps` record with the full-data inertia per step.
 *
 * @param hyper - Number of clusters, initialisation, batch size and number of steps (default 100).
 * @returns The estimator.
 */
export function miniBatchKMeans(hyper: MiniBatchKMeansHyper = {}): Estimator<MiniBatchKMeansHyper, FittedKMeans> {
  return defineExplainedEstimator<MiniBatchKMeansHyper, CentroidParams, CentroidVerbs>(
    {
      name: 'mini-batch-kmeans',
      kind: 'clusterer',
      hyperSpace: (h) =>
        dict({
          ...kmeansSpace(h).spaces,
          maxIter: int(1, 1000, { default: h.maxIter ?? 100, label: 'steps' }),
          batchSize: int(1, 1024, { default: h.batchSize ?? 64, label: 'batch size' }),
        }),
      fit: (data, h, options) => {
        const run = fitRunOptions(options, 'mini-batch-kmeans')
        const key = run.key as Key
        const start = initialCentres(data.x, h.nClusters ?? 3, h.init ?? 'k-means++', foldIn(key, 0))
        const out = runAlgorithm(miniBatchKMeansAlgorithm(data.x, start, h.batchSize ?? 64), {
          ...run,
          key: foldIn(key, 1),
          maxSteps: h.maxIter ?? 100,
        })
        const s = out.state
        const final = nearestCentres(data.x, s.centres)
        return {
          params: { centres: s.centres, labels: final.labels, inertia: final.inertia, nIter: out.steps },
          exposition: [
            clusteringSteps({
              method: 'miniBatchKMeans',
              assignments: s.history.assignments,
              centres: s.history.centres,
              objective: s.history.objective,
            }),
          ],
        }
      },
      methods: centroidVerbs,
    },
    hyper,
  )
}

/** Mini-batch k-means as a class. */
export class MiniBatchKMeans extends EstimatorClass<MiniBatchKMeansHyper, FittedKMeans> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link miniBatchKMeans}.
   */
  constructor(hyper: MiniBatchKMeansHyper = {}) {
    super(miniBatchKMeans(hyper))
  }
}

/** Hyperparameters of bisecting k-means. */
export type BisectingKMeansHyper = {
  readonly nClusters?: number
  /** Which cluster to split next: the one with the largest inertia (default) or the most rows. */
  readonly bisectingStrategy?: 'biggest_inertia' | 'largest_cluster'
  readonly maxIter?: number
}

/**
 * Bisecting k-means (sklearn.cluster.BisectingKMeans): start with one cluster and repeatedly split one in two with
 * 2-means until there are k. Each split is a step of its `clusteringSteps` record (assignments, centres and total
 * inertia after each split).
 *
 * @param hyper - Number of clusters, the splitting strategy and the 2-means iteration cap.
 * @returns The estimator.
 */
export function bisectingKMeans(hyper: BisectingKMeansHyper = {}): Estimator<BisectingKMeansHyper, FittedKMeans> {
  return defineExplainedEstimator<BisectingKMeansHyper, CentroidParams, CentroidVerbs>(
    {
      name: 'bisecting-kmeans',
      kind: 'clusterer',
      hyperSpace: (h) =>
        dict({
          nClusters: int(1, 20, { default: h.nClusters ?? 3, label: 'clusters' }),
          bisectingStrategy: choice(['biggest_inertia', 'largest_cluster'], {
            default: h.bisectingStrategy ?? 'biggest_inertia',
          }),
        }),
      fit: (data, h, options) => {
        const key = fitRunOptions(options, 'bisecting-kmeans').key as Key
        const m = matrixView(data.x, 'bisectingKMeans')
        const k = h.nClusters ?? 3
        if (!(Number.isInteger(k) && k >= 1 && k <= m.n))
          throw new WhiteboxError('value', `bisectingKMeans: nClusters must be in 1..${m.n}`)
        const row = (i: number): number[] => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
        const mean = (rows: readonly number[]): number[] => {
          const c = new Array<number>(m.d).fill(0)
          for (const i of rows)
            for (let q = 0; q < m.d; q++) c[q] = (c[q] ?? 0) + (m.data[i * m.d + q] ?? 0) / rows.length
          return c
        }
        const sse = (rows: readonly number[], c: readonly number[]): number =>
          rows.reduce((a, i) => a + row(i).reduce((s, v, q) => s + (v - (c[q] ?? 0)) ** 2, 0), 0)
        let members: number[][] = [Array.from({ length: m.n }, (_, i) => i)]
        let centres: number[][] = [mean(members[0] as number[])]
        const labelsOf = (): number[] => {
          const out = new Array<number>(m.n).fill(0)
          for (const [c, rows] of members.entries()) for (const i of rows) out[i] = c
          return out
        }
        const total = (): number => members.reduce((a, rows, c) => a + sse(rows, centres[c] as number[]), 0)
        const assignments = [labelsOf()]
        const history = [centres.map((c) => [...c])]
        const objective = [total()]
        for (let split = 1; split < k; split++) {
          const scores = members.map((rows, c) =>
            rows.length < 2
              ? -1
              : h.bisectingStrategy === 'largest_cluster'
                ? rows.length
                : sse(rows, centres[c] as number[]),
          )
          const target = scores.indexOf(Math.max(...scores))
          const rows = members[target] as number[]
          const sub = rows.map(row)
          const start = initialCentres(sub, 2, 'k-means++', foldIn(key, split))
          const out = runAlgorithm(lloydAlgorithm(sub, start), { maxSteps: h.maxIter ?? 300 })
          const left = rows.filter((_, r) => out.state.labels[r] === 0)
          const right = rows.filter((_, r) => out.state.labels[r] === 1)
          members = [...members.slice(0, target), left, ...members.slice(target + 1), right]
          centres = [...centres.slice(0, target), mean(left), ...centres.slice(target + 1), mean(right)]
          assignments.push(labelsOf())
          history.push(centres.map((c) => [...c]))
          objective.push(total())
        }
        return {
          params: { centres, labels: labelsOf(), inertia: objective.at(-1) ?? 0, nIter: k - 1 },
          exposition: [clusteringSteps({ method: 'bisectingKMeans', assignments, centres: history, objective })],
        }
      },
      methods: centroidVerbs,
    },
    hyper,
  )
}

/** Bisecting k-means as a class. */
export class BisectingKMeans extends EstimatorClass<BisectingKMeansHyper, FittedKMeans> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link bisectingKMeans}.
   */
  constructor(hyper: BisectingKMeansHyper = {}) {
    super(bisectingKMeans(hyper))
  }
}

/**
 * The elbow curve: k-means inertia for each number of clusters, for choosing k.
 *
 * @param x - Points `[n, d]`.
 * @param ks - The numbers of clusters to try.
 * @param options - Randomness and tracing.
 * @returns The numbers of clusters and the inertia of each fit.
 */
export function kmeansElbow(
  x: TensorLike,
  ks: readonly number[],
  options: FitOptions = {},
): { readonly ks: readonly number[]; readonly inertia: number[] } {
  const key = fitRunOptions(options, 'kmeans-elbow').key as Key
  return {
    ks,
    inertia: ks.map((k, i) => kmeans({ nClusters: k, nInit: 3 }).fit({ x }, { key: foldIn(key, i) }).params.inertia),
  }
}
