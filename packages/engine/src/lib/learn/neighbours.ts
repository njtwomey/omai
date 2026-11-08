/**
 * Nearest-neighbour learners over the spatial indexes: the k-nearest classifier and regressor with uniform or
 * distance weights and any metric of `distances` (brute force, or a KD or ball tree where the metric allows), their
 * radius twins with an outlier label, and the nearest centroid classifier with Tibshirani's shrinkage (and its path
 * over the threshold). All are estimators on the iterative base that absorb the training rows one per step, so a
 * session shows the decision surface roughening as the memory fills and `fittedAt(t)` is the model on the first `t`
 * rows. `neighbourIndex` is the search itself, usable by anything that needs the neighbours of a point.
 */

import { OmaiError } from '../base/index.ts'
import {
  type DistanceAlias,
  type DistanceName,
  type DistanceOptions,
  distanceAliases,
  pairwiseDistances,
} from '../distances/index.ts'
import { inv } from '../numerics/linalg/index.ts'
import { ballTree, kdTree, type SpatialTree } from '../numerics/spatial/index.ts'
import { choice, type DictSpace, dict, int, real } from '../space/index.ts'
import { median } from '../stats/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { Log } from '../trace/index.ts'
import { accuracyOf, classifierFromScores, prefixOf, type RowsState, r2Of, rowsAlgorithm } from './_classify.ts'
import { type IterativeEstimator, iterative } from './bases/index.ts'
import type { LearnData } from './estimator.ts'
import { encodeLabels, type MatrixView, matrixView, rowsOf, targetVector } from './validation.ts'

/** How a neighbour's vote or value is weighted. */
export type NeighbourWeighting = 'uniform' | 'distance'

/** How the neighbours are found: `auto` takes a KD tree when the metric and dimension allow, else brute force. */
export type NeighbourAlgorithm = 'auto' | 'brute' | 'kd-tree' | 'ball-tree'

/** The search settings every neighbour learner shares. */
export interface NeighbourSearch {
  /** Any distance of `omai-engine/distances` (default `euclidean`). */
  readonly metric: DistanceName | DistanceAlias
  /** The metric's parameters (`p`, `weights`, `variance`, `covariance`); seuclidean and mahalanobis default to the training statistics. */
  readonly metricOptions: DistanceOptions
  readonly algorithm: NeighbourAlgorithm
  /** Points per tree leaf. */
  readonly leafSize: number
}

/** The neighbours of one query: indices into the indexed points and their distances, nearest first. */
export interface NeighbourHits {
  readonly indices: readonly number[]
  readonly distances: readonly number[]
}

/** A neighbour search over a fixed point set. */
export interface NeighbourIndex {
  /** The number of indexed points. */
  readonly size: number
  /** The algorithm actually used. */
  readonly algorithm: 'brute' | 'kd-tree' | 'ball-tree'
  /** The `k` nearest points of each query row (fewer when the index is smaller than `k`). */
  nearest(queries: TensorLike, k: number): NeighbourHits[]
  /** Every point within `radius` of each query row. */
  within(queries: TensorLike, radius: number): NeighbourHits[]
}

const SEARCH: NeighbourSearch = { metric: 'euclidean', metricOptions: {}, algorithm: 'auto', leafSize: 30 }

const canonicalMetric = (name: DistanceName | DistanceAlias): DistanceName =>
  (distanceAliases as Readonly<Record<string, DistanceName>>)[name] ?? (name as DistanceName)

const TREE_METRICS = {
  euclidean: 'euclidean',
  manhattan: 'cityblock',
  chebyshev: 'chebyshev',
  minkowski: 'minkowski',
} as const

/** Fill the statistics scipy takes from the data when a metric needs them and none were given. */
function withStatistics(name: DistanceName, m: MatrixView, options: DistanceOptions): DistanceOptions {
  if (name === 'seuclidean' && options.variance === undefined) {
    const variance = Float64Array.from({ length: m.d }, (_, j) => {
      let mean = 0
      for (let i = 0; i < m.n; i++) mean += (m.data[i * m.d + j] ?? 0) / m.n
      let s = 0
      for (let i = 0; i < m.n; i++) s += ((m.data[i * m.d + j] ?? 0) - mean) ** 2
      return s / Math.max(1, m.n - 1)
    })
    return { ...options, variance }
  }
  if (name === 'mahalanobis' && options.inverseCovariance === undefined && options.covariance === undefined) {
    const mean = new Float64Array(m.d)
    for (let i = 0; i < m.n; i++)
      for (let j = 0; j < m.d; j++) mean[j] = (mean[j] ?? 0) + (m.data[i * m.d + j] ?? 0) / m.n
    const cov = new Float64Array(m.d * m.d)
    for (let i = 0; i < m.n; i++)
      for (let a = 0; a < m.d; a++)
        for (let b = 0; b < m.d; b++)
          cov[a * m.d + b] =
            (cov[a * m.d + b] ?? 0) +
            (((m.data[i * m.d + a] ?? 0) - (mean[a] ?? 0)) * ((m.data[i * m.d + b] ?? 0) - (mean[b] ?? 0))) /
              Math.max(1, m.n - 1)
    return { ...options, inverseCovariance: inv(fromStorage(cov, [m.d, m.d], 'float64')) as Tensor }
  }
  return options
}

/**
 * A neighbour search over a point set: exact k-nearest and radius queries under any metric of `distances`, by brute
 * force or by a KD or ball tree (the same answers up to ties). Seuclidean and mahalanobis take their variances and
 * covariance from the indexed points, as scikit-learn does.
 *
 * @param points - The points `[n, d]`.
 * @param search - Metric, its options, the algorithm and the leaf size.
 * @returns The index.
 * @example
 * ```ts
 * import { neighbourIndex } from 'omai-engine/learn'
 * const index = neighbourIndex([[0, 0], [1, 0], [5, 5]])
 * index.nearest([[0.9, 0.1]], 2)[0].indices // => [1, 0]
 * index.within([[0, 0]], 1.5)[0].indices // => [0, 1]
 * ```
 */
export function neighbourIndex(points: TensorLike, search: Partial<NeighbourSearch> = {}): NeighbourIndex {
  const s: NeighbourSearch = { ...SEARCH, ...search }
  const m = matrixView(points, 'neighbourIndex')
  const name = canonicalMetric(s.metric)
  const options = withStatistics(name, m, s.metricOptions)
  const treeMetric = (TREE_METRICS as Readonly<Record<string, 'euclidean' | 'cityblock' | 'chebyshev' | 'minkowski'>>)[
    name
  ]
  const treeCapable = treeMetric !== undefined
  const wanted = s.algorithm
  if ((wanted === 'kd-tree' || wanted === 'ball-tree') && !treeCapable) {
    throw new OmaiError(
      'value',
      `neighbourIndex: the ${wanted} supports euclidean, manhattan, chebyshev and minkowski, not ${name}`,
    )
  }
  const algorithm: 'brute' | 'kd-tree' | 'ball-tree' =
    wanted === 'auto' ? (treeCapable && m.d <= 15 && m.n > 4 * s.leafSize ? 'kd-tree' : 'brute') : wanted
  const train = fromStorage(m.data, [m.n, m.d], 'float64')
  const wrap = (q: TensorLike): MatrixView => matrixView(q, 'neighbourIndex', { nFeatures: m.d })
  if (algorithm === 'brute') {
    const distances = (q: MatrixView): Float64Array =>
      Float64Array.from(
        toFlat(primalOf(pairwiseDistances(fromStorage(q.data, [q.n, q.d], 'float64'), train, s.metric, options))),
      )
    const ordered = (all: Float64Array, row: number, keep: (j: number) => boolean): number[] =>
      Array.from({ length: m.n }, (_, j) => j)
        .filter(keep)
        .sort((a, b) => (all[row * m.n + a] ?? 0) - (all[row * m.n + b] ?? 0) || a - b)
    return {
      size: m.n,
      algorithm,
      nearest: (queries, k) => {
        const q = wrap(queries)
        const all = distances(q)
        return Array.from({ length: q.n }, (_, row) => {
          const indices = ordered(all, row, () => true).slice(0, k)
          return { indices, distances: indices.map((j) => all[row * m.n + j] ?? 0) }
        })
      },
      within: (queries, radius) => {
        const q = wrap(queries)
        const all = distances(q)
        return Array.from({ length: q.n }, (_, row) => {
          const indices = ordered(all, row, (j) => (all[row * m.n + j] ?? Number.POSITIVE_INFINITY) <= radius)
          return { indices, distances: indices.map((j) => all[row * m.n + j] ?? 0) }
        })
      },
    }
  }
  const treeOptions = {
    metric: treeMetric as 'euclidean' | 'cityblock' | 'chebyshev' | 'minkowski',
    leafSize: s.leafSize,
    ...(options.p === undefined ? {} : { p: options.p }),
    ...(options.weights === undefined ? {} : { w: options.weights }),
  }
  let tree: SpatialTree | undefined
  const built = (): SpatialTree => {
    tree ??= algorithm === 'kd-tree' ? kdTree(train, treeOptions) : ballTree(train, treeOptions)
    return tree
  }
  return {
    size: m.n,
    algorithm,
    nearest: (queries, k) => {
      const q = wrap(queries)
      const kk = Math.min(k, m.n)
      if (kk === 0) return Array.from({ length: q.n }, () => ({ indices: [], distances: [] }))
      const r = built().query(fromStorage(q.data, [q.n, q.d], 'float64'), kk)
      const ix = toFlat(r.indices)
      const dx = toFlat(r.distances)
      return Array.from({ length: q.n }, (_, row) => ({
        indices: Array.from(ix.slice(row * kk, (row + 1) * kk)),
        distances: Array.from(dx.slice(row * kk, (row + 1) * kk)),
      }))
    },
    within: (queries, radius) => {
      const q = wrap(queries)
      const r = built().queryRadius(fromStorage(q.data, [q.n, q.d], 'float64'), radius)
      return Array.from({ length: q.n }, (_, row) => ({
        indices: r.indices[row] ?? [],
        distances: r.distances[row] ?? [],
      }))
    },
  }
}

/**
 * The weights of neighbours by distance: all ones for `uniform`; `1 / d` for `distance`, except that a neighbour at
 * distance zero takes all the weight (scikit-learn's rule).
 *
 * @param distances - The neighbours' distances.
 * @param weighting - The scheme.
 * @returns One weight per neighbour.
 * @example
 * ```ts
 * import { neighbourWeights } from 'omai-engine/learn'
 * neighbourWeights([1, 2, 4], 'distance') // => [1, 0.5, 0.25]
 * neighbourWeights([0, 2], 'distance') // => [1, 0]
 * ```
 */
export function neighbourWeights(distances: readonly number[], weighting: NeighbourWeighting): number[] {
  if (weighting === 'uniform') return distances.map(() => 1)
  if (distances.some((d) => d === 0)) return distances.map((d) => (d === 0 ? 1 : 0))
  return distances.map((d) => 1 / d)
}

/** The hyperparameters of the k-nearest-neighbour learners. */
export interface KNeighboursHyper extends NeighbourSearch {
  readonly nNeighbors: number
  readonly weights: NeighbourWeighting
}

/** The hyperparameters of the radius-neighbour learners. */
export interface RadiusNeighboursHyper extends NeighbourSearch {
  readonly radius: number
  readonly weights: NeighbourWeighting
  /** The label of a query with no neighbour in the radius (classifier): a class, or `'most-frequent'`; none raises. */
  readonly outlierLabel?: number | 'most-frequent'
}

/** The fitted parameters of the neighbour classifiers: the memory. */
export type NeighbourClassifierParams = {
  readonly classes: number[]
  readonly x: number[][]
  readonly codes: number[]
}

/** The fitted parameters of the neighbour regressors: the memory. */
export type NeighbourRegressorParams = {
  readonly x: number[][]
  readonly y: number[]
}

/** What a fitted neighbour learner offers beyond the regression or classification verbs. */
export interface NeighboursVerbs {
  /** The neighbours of each query (the `k` nearest, or those within the radius). */
  readonly neighbours: (x: TensorLike) => NeighbourHits[]
}

/** The verbs of a fitted neighbour classifier. */
export interface NeighbourClassifierVerbs extends NeighboursVerbs {
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
}

/** The verbs of a fitted neighbour regressor. */
export interface NeighbourRegressorVerbs extends NeighboursVerbs {
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
}

const rowsToArrays = (m: MatrixView): number[][] =>
  Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))

const arraysToTensor = (rows: readonly (readonly number[])[]): Tensor => {
  const d = rows[0]?.length ?? 0
  return fromStorage(Float64Array.from(rows.flat()), [rows.length, d], 'float64')
}

const searchSpace = (h: NeighbourSearch): { readonly metric: ReturnType<typeof choice<string>> } => {
  const named: readonly string[] = ['euclidean', 'manhattan', 'chebyshev', 'cosine']
  return { metric: choice(named.includes(h.metric) ? named : [...named, h.metric], { default: h.metric }) }
}

const readSearch = (h: NeighbourSearch): Partial<NeighbourSearch> => ({
  metric: h.metric,
  metricOptions: h.metricOptions,
  algorithm: h.algorithm,
  leafSize: h.leafSize,
})

function memoryOf(data: LearnData, n: number, who: string): { readonly m: MatrixView; readonly head: LearnData } {
  const head = prefixOf(data, n)
  return { m: matrixView(head.x, who, { minSamples: 0 }), head }
}

type Finder = (queries: TensorLike) => NeighbourHits[]

function finderOf(x: number[][], h: KNeighboursHyper | RadiusNeighboursHyper, who: string): Finder {
  if (x.length === 0) throw new OmaiError('value', `${who}: no training rows yet`)
  const index = neighbourIndex(arraysToTensor(x), readSearch(h))
  if ('nNeighbors' in h) return (q) => index.nearest(q, h.nNeighbors)
  return (q) => index.within(q, h.radius)
}

function classifierVerbs(
  p: NeighbourClassifierParams,
  h: KNeighboursHyper | RadiusNeighboursHyper,
  who: string,
): NeighbourClassifierVerbs {
  const k = p.classes.length
  const find = finderOf(p.x, h, who)
  const outlier = 'outlierLabel' in h ? h.outlierLabel : undefined
  const mostFrequent = ((): number => {
    const counts = new Array<number>(k).fill(0)
    for (const c of p.codes) counts[c] = (counts[c] ?? 0) + 1
    return counts.reduce((best, v, c) => (v > (counts[best] ?? 0) ? c : best), 0)
  })()
  const proba = (x: TensorLike): Float64Array => {
    const hits = find(x)
    const out = new Float64Array(hits.length * k)
    hits.forEach((hit, i) => {
      if (hit.indices.length === 0) {
        if (outlier === undefined) {
          throw new OmaiError(
            'value',
            `${who}: no neighbours within radius ${'radius' in h ? h.radius : ''}; set outlierLabel`,
          )
        }
        const c = outlier === 'most-frequent' ? mostFrequent : p.classes.indexOf(outlier)
        if (c >= 0) out[i * k + c] = 1
        return
      }
      const w = neighbourWeights(hit.distances, h.weights)
      const total = w.reduce((a, b) => a + b, 0)
      hit.indices.forEach((j, r) => {
        const c = p.codes[j] ?? 0
        out[i * k + c] = (out[i * k + c] ?? 0) + (w[r] ?? 0) / total
      })
    })
    return out
  }
  const predict = (x: TensorLike): Tensor => {
    const pr = proba(x)
    const n = pr.length / k
    const labels = new Float64Array(n)
    const hits =
      outlier !== undefined && typeof outlier === 'number' && !p.classes.includes(outlier) ? find(x) : undefined
    for (let i = 0; i < n; i++) {
      if (hits !== undefined && (hits[i]?.indices.length ?? 0) === 0) {
        labels[i] = outlier as number
        continue
      }
      let best = 0
      for (let c = 1; c < k; c++) if ((pr[i * k + c] ?? 0) > (pr[i * k + best] ?? 0)) best = c
      labels[i] = p.classes[best] ?? Number.NaN
    }
    return fromStorage(labels, [n], 'float64')
  }
  return {
    classes: p.classes,
    nFeaturesIn: p.x[0]?.length ?? 0,
    predict,
    predictProba: (x) => {
      const pr = proba(x)
      return fromStorage(pr, [pr.length / k, k], 'float64')
    },
    neighbours: find,
    score: accuracyOf(predict),
  }
}

function regressorVerbs(
  p: NeighbourRegressorParams,
  h: KNeighboursHyper | RadiusNeighboursHyper,
  who: string,
): NeighbourRegressorVerbs {
  const find = finderOf(p.x, h, who)
  const predict = (x: TensorLike): Tensor => {
    const hits = find(x)
    return fromStorage(
      Float64Array.from(hits, (hit) => {
        if (hit.indices.length === 0) return Number.NaN
        const w = neighbourWeights(hit.distances, h.weights)
        let s = 0
        let t = 0
        hit.indices.forEach((j, r) => {
          s += (w[r] ?? 0) * (p.y[j] ?? 0)
          t += w[r] ?? 0
        })
        return s / t
      }),
      [hits.length],
      'float64',
    )
  }
  return { nFeaturesIn: p.x[0]?.length ?? 0, predict, neighbours: find, score: r2Of(predict) }
}

const K_DEFAULTS: KNeighboursHyper = { ...SEARCH, nNeighbors: 5, weights: 'uniform' }
const RADIUS_DEFAULTS: RadiusNeighboursHyper = { ...SEARCH, radius: 1, weights: 'uniform' }

const kSpace = (h: KNeighboursHyper): DictSpace =>
  dict({
    nNeighbors: int(1, Math.max(50, h.nNeighbors), { default: h.nNeighbors, label: 'neighbours' }),
    weights: choice(['uniform', 'distance'] as const, { default: h.weights }),
    ...searchSpace(h),
  })

const radiusSpace = (h: RadiusNeighboursHyper): DictSpace =>
  dict({
    radius: real(Math.min(0.01, h.radius), Math.max(10, h.radius), { default: h.radius, log: true }),
    weights: choice(['uniform', 'distance'] as const, { default: h.weights }),
    ...searchSpace(h),
  })

function classifierParams(data: LearnData, n: number, who: string): NeighbourClassifierParams {
  const { classes } = encodeLabels(data.y ?? [], who, rowsOf(data.x))
  const { m, head } = memoryOf(data, n, who)
  const labels = targetVector(head.y ?? [], who)
  const index = new Map(classes.map((c, i) => [c, i]))
  return { classes: [...classes], x: rowsToArrays(m), codes: Array.from(labels, (v) => index.get(v) ?? 0) }
}

function regressorParams(data: LearnData, n: number, who: string): NeighbourRegressorParams {
  const { m, head } = memoryOf(data, n, who)
  return { x: rowsToArrays(m), y: Array.from(targetVector(head.y ?? [], who)) }
}

/**
 * The k-nearest-neighbour classifier (`sklearn.neighbors.KNeighborsClassifier`): the vote of the `k` nearest training
 * rows, uniform or weighted by inverse distance, under any metric of `distances`. The probability of a class is its
 * share of the weight. On the iterative base the training rows arrive one per step (logging `solver/rows`), so a
 * session's `fittedAt(t)` is the classifier on the first `t` rows; the neighbours of a query are the `neighbours` verb.
 *
 * @param hyper - `nNeighbors`, `weights`, `metric`, `metricOptions`, `algorithm`, `leafSize`.
 * @returns The estimator.
 * @index estimator k-nearest-neighbour classifier under any distance.
 * @example
 * ```ts
 * import { kNeighboursClassifier } from 'omai-engine/learn'
 * import { toFlat } from 'omai-engine/tensor'
 * const knn = kNeighboursClassifier({ nNeighbors: 3 }).fit({ x: [[0], [1], [2], [10], [11], [12]], y: [0, 0, 0, 1, 1, 1] })
 * toFlat(knn.predict([[1.5], [10.5]])) // => [0, 1]
 * ```
 */
export function kNeighboursClassifier(
  hyper: Partial<KNeighboursHyper> = {},
): IterativeEstimator<KNeighboursHyper, RowsState, NeighbourClassifierParams, NeighbourClassifierVerbs> {
  return iterative<KNeighboursHyper, RowsState, NeighbourClassifierParams, NeighbourClassifierVerbs>(
    {
      name: 'k-neighbours-classifier',
      kind: 'classifier',
      defaults: K_DEFAULTS,
      hyperSpace: kSpace,
      algorithm: (data, h) => rowsAlgorithm('k-neighbours', rowsOf(data.x), { k: h.nNeighbors }),
      maxSteps: (_h, data) => rowsOf(data.x),
      logs: { 'solver/rows': Log() },
      params: (s, data) => classifierParams(data, s.n, 'kNeighboursClassifier'),
      verbs: (p, h) => classifierVerbs(p, h, 'kNeighboursClassifier'),
    },
    { ...K_DEFAULTS, ...hyper },
  )
}

/**
 * The k-nearest-neighbour regressor (`sklearn.neighbors.KNeighborsRegressor`): the mean of the `k` nearest targets,
 * uniform or weighted by inverse distance.
 *
 * @param hyper - As {@link kNeighboursClassifier}.
 * @returns The estimator.
 * @example
 * ```ts
 * import { kNeighboursRegressor } from 'omai-engine/learn'
 * import { toFlat } from 'omai-engine/tensor'
 * const knn = kNeighboursRegressor({ nNeighbors: 2 }).fit({ x: [[0], [1], [2], [3]], y: [0, 2, 4, 6] })
 * toFlat(knn.predict([[1.4]])) // => [3]
 * ```
 */
export function kNeighboursRegressor(
  hyper: Partial<KNeighboursHyper> = {},
): IterativeEstimator<KNeighboursHyper, RowsState, NeighbourRegressorParams, NeighbourRegressorVerbs> {
  return iterative<KNeighboursHyper, RowsState, NeighbourRegressorParams, NeighbourRegressorVerbs>(
    {
      name: 'k-neighbours-regressor',
      kind: 'regressor',
      defaults: K_DEFAULTS,
      hyperSpace: kSpace,
      algorithm: (data, h) => rowsAlgorithm('k-neighbours', rowsOf(data.x), { k: h.nNeighbors }),
      maxSteps: (_h, data) => rowsOf(data.x),
      logs: { 'solver/rows': Log() },
      params: (s, data) => regressorParams(data, s.n, 'kNeighboursRegressor'),
      verbs: (p, h) => regressorVerbs(p, h, 'kNeighboursRegressor'),
    },
    { ...K_DEFAULTS, ...hyper },
  )
}

/**
 * The radius-neighbour classifier (`sklearn.neighbors.RadiusNeighborsClassifier`): the vote of every training row
 * within `radius`. A query with none takes `outlierLabel` (a class, a label outside the classes which gives a zero
 * probability row, or `'most-frequent'`), and raises without one.
 *
 * @param hyper - `radius`, `outlierLabel`, `weights`, and the search settings.
 * @returns The estimator.
 * @example
 * ```ts
 * import { radiusNeighboursClassifier } from 'omai-engine/learn'
 * import { toFlat } from 'omai-engine/tensor'
 * const c = radiusNeighboursClassifier({ radius: 1.5, outlierLabel: -1 }).fit({ x: [[0], [1], [10]], y: [0, 0, 1] })
 * toFlat(c.predict([[0.5], [10.2], [50]])) // => [0, 1, -1]
 * ```
 */
export function radiusNeighboursClassifier(
  hyper: Partial<RadiusNeighboursHyper> = {},
): IterativeEstimator<RadiusNeighboursHyper, RowsState, NeighbourClassifierParams, NeighbourClassifierVerbs> {
  return iterative<RadiusNeighboursHyper, RowsState, NeighbourClassifierParams, NeighbourClassifierVerbs>(
    {
      name: 'radius-neighbours-classifier',
      kind: 'classifier',
      defaults: RADIUS_DEFAULTS,
      hyperSpace: radiusSpace,
      algorithm: (data, h) => rowsAlgorithm('radius-neighbours', rowsOf(data.x), { radius: h.radius }),
      maxSteps: (_h, data) => rowsOf(data.x),
      logs: { 'solver/rows': Log() },
      params: (s, data) => classifierParams(data, s.n, 'radiusNeighboursClassifier'),
      verbs: (p, h) => classifierVerbs(p, h, 'radiusNeighboursClassifier'),
    },
    { ...RADIUS_DEFAULTS, ...hyper },
  )
}

/**
 * The radius-neighbour regressor (`sklearn.neighbors.RadiusNeighborsRegressor`): the (weighted) mean target of the
 * rows within `radius`, NaN where there are none.
 *
 * @param hyper - `radius`, `weights` and the search settings.
 * @returns The estimator.
 * @example
 * ```ts
 * import { radiusNeighboursRegressor } from 'omai-engine/learn'
 * import { toFlat } from 'omai-engine/tensor'
 * const r = radiusNeighboursRegressor({ radius: 1.1 }).fit({ x: [[0], [1], [2]], y: [0, 3, 6] })
 * toFlat(r.predict([[1]])) // => [3]
 * ```
 */
export function radiusNeighboursRegressor(
  hyper: Partial<RadiusNeighboursHyper> = {},
): IterativeEstimator<RadiusNeighboursHyper, RowsState, NeighbourRegressorParams, NeighbourRegressorVerbs> {
  return iterative<RadiusNeighboursHyper, RowsState, NeighbourRegressorParams, NeighbourRegressorVerbs>(
    {
      name: 'radius-neighbours-regressor',
      kind: 'regressor',
      defaults: RADIUS_DEFAULTS,
      hyperSpace: radiusSpace,
      algorithm: (data, h) => rowsAlgorithm('radius-neighbours', rowsOf(data.x), { radius: h.radius }),
      maxSteps: (_h, data) => rowsOf(data.x),
      logs: { 'solver/rows': Log() },
      params: (s, data) => regressorParams(data, s.n, 'radiusNeighboursRegressor'),
      verbs: (p, h) => regressorVerbs(p, h, 'radiusNeighboursRegressor'),
    },
    { ...RADIUS_DEFAULTS, ...hyper },
  )
}

/** The hyperparameters of the nearest centroid classifier. */
export interface NearestCentroidHyper {
  /** The distance to the centroids: the centroid is the mean, or the feature-wise median under `manhattan`. */
  readonly metric: 'euclidean' | 'manhattan'
  /** Tibshirani's threshold: deviations of a class centroid from the overall centroid shrink by it (default 0, none). */
  readonly shrinkThreshold: number
  /** `uniform`, `empirical`, or the prior of each class. */
  readonly priors: 'uniform' | 'empirical' | readonly number[]
}

/** The fitted parameters of the nearest centroid classifier. */
export type NearestCentroidParams = {
  readonly classes: number[]
  /** The (shrunken) centroids `[k][d]`. */
  readonly centroids: number[][]
  readonly datasetCentroid: number[]
  /** The pooled within-class standard deviation of each feature. */
  readonly withinStd: number[]
  /** The standardised deviation of each centroid from the overall one after shrinkage, `[k][d]`. */
  readonly deviations: number[][]
  readonly priors: number[]
}

/** The verbs of a fitted nearest centroid classifier. */
export interface NearestCentroidVerbs {
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  /** The discriminant scores `-|(x - c) / s|^2 + 2 log prior` (Euclidean metric): `[m]` for two classes, else `[m, k]`. */
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
}

function centroidFit(
  m: MatrixView,
  codes: Int32Array,
  k: number,
  hyper: NearestCentroidHyper,
): Omit<NearestCentroidParams, 'classes'> {
  const { n, d, data } = m
  const counts = new Array<number>(k).fill(0)
  for (const c of codes) counts[c] = (counts[c] ?? 0) + 1
  const centroids = Array.from({ length: k }, (_, c) =>
    Array.from({ length: d }, (_, j) => {
      const col: number[] = []
      for (let i = 0; i < n; i++) if (codes[i] === c) col.push(data[i * d + j] ?? 0)
      return hyper.metric === 'manhattan' ? median(col) : col.reduce((a, b) => a + b, 0) / Math.max(1, col.length)
    }),
  )
  const withinStd = Array.from({ length: d }, (_, j) => {
    let s = 0
    for (let i = 0; i < n; i++) s += ((data[i * d + j] ?? 0) - (centroids[codes[i] ?? 0]?.[j] ?? 0)) ** 2
    return n > k ? Math.sqrt(s / (n - k)) : 1
  })
  const datasetCentroid = Array.from({ length: d }, (_, j) => {
    let s = 0
    for (let i = 0; i < n; i++) s += data[i * d + j] ?? 0
    return s / n
  })
  const medianStd = median(withinStd)
  const deviations = centroids.map((cen, c) => {
    const mm = Math.sqrt(Math.max(0, 1 / (counts[c] ?? 1) - 1 / n))
    return cen.map((v, j) =>
      mm === 0 ? 0 : (v - (datasetCentroid[j] ?? 0)) / (mm * ((withinStd[j] ?? 0) + medianStd)),
    )
  })
  const priors =
    hyper.priors === 'uniform'
      ? new Array<number>(k).fill(1 / k)
      : hyper.priors === 'empirical'
        ? counts.map((c) => c / n)
        : (() => {
            const total = hyper.priors.reduce((a, b) => a + b, 0)
            return hyper.priors.map((p) => p / total)
          })()
  const threshold = hyper.shrinkThreshold
  if (threshold === 0) return { centroids, datasetCentroid, withinStd, deviations, priors }
  const shrunk = deviations.map((row) => row.map((v) => Math.sign(v) * Math.max(Math.abs(v) - threshold, 0)))
  const shrunkCentroids = shrunk.map((row, c) => {
    const mm = Math.sqrt(Math.max(0, 1 / (counts[c] ?? 1) - 1 / n))
    return row.map((v, j) => (datasetCentroid[j] ?? 0) + mm * ((withinStd[j] ?? 0) + medianStd) * v)
  })
  return { centroids: shrunkCentroids, datasetCentroid, withinStd, deviations: shrunk, priors }
}

function centroidVerbs(p: NearestCentroidParams, h: NearestCentroidHyper): NearestCentroidVerbs {
  const k = p.classes.length
  const d = p.datasetCentroid.length
  const uniform = p.priors.every((v) => Math.abs(v - 1 / k) <= 1e-8 + 1e-5 * (1 / k))
  const discriminant = (x: TensorLike): { readonly n: number; readonly s: Float64Array } => {
    const m = matrixView(x, 'nearestCentroid', { nFeatures: d })
    const s = new Float64Array(m.n * k)
    for (let i = 0; i < m.n; i++)
      for (let c = 0; c < k; c++) {
        let dist = 0
        for (let j = 0; j < d; j++) {
          const sd = p.withinStd[j] ?? 0
          const scale = sd === 0 ? 1 : sd
          const diff = ((m.data[i * d + j] ?? 0) - (p.centroids[c]?.[j] ?? 0)) / scale
          dist += h.metric === 'manhattan' ? Math.abs(diff) : diff * diff
        }
        s[i * k + c] = -(h.metric === 'manhattan' ? dist * dist : dist) + 2 * Math.log(p.priors[c] ?? 0)
      }
    return { n: m.n, s }
  }
  const viaScores = classifierFromScores(
    p.classes,
    d,
    (m) => discriminant(fromStorage(m.data, [m.n, m.d], 'float64')).s,
    'nearestCentroid',
  )
  const predict = (x: TensorLike): Tensor => {
    if (!uniform) return viaScores.predict(x)
    const m = matrixView(x, 'nearestCentroid', { nFeatures: d })
    const dist = pairwiseDistances(fromStorage(m.data, [m.n, m.d], 'float64'), arraysToTensor(p.centroids), h.metric)
    const flat = toFlat(primalOf(dist))
    return fromStorage(
      Float64Array.from({ length: m.n }, (_, i) => {
        let best = 0
        for (let c = 1; c < k; c++) if ((flat[i * k + c] ?? 0) < (flat[i * k + best] ?? 0)) best = c
        return p.classes[best] ?? Number.NaN
      }),
      [m.n],
      'float64',
    )
  }
  return {
    classes: p.classes,
    nFeaturesIn: d,
    predict,
    predictProba: viaScores.predictProba,
    decisionFunction: viaScores.decisionFunction,
    score: accuracyOf(predict),
  }
}

/**
 * The nearest centroid classifier with shrinkage (`sklearn.neighbors.NearestCentroid`, Tibshirani et al.'s nearest
 * shrunken centroids): each class is its centroid (mean, or median under the Manhattan metric); a threshold shrinks
 * the standardised deviation of every centroid from the overall centroid towards zero, so features whose classes
 * barely differ drop out. Prediction is the nearest centroid; with non-uniform priors it is the arg-max of the
 * discriminant score. The probabilities are the softmax of that score.
 *
 * @param hyper - `metric`, `shrinkThreshold`, `priors`.
 * @returns The estimator.
 * @reference R. Tibshirani, T. Hastie, B. Narasimhan and G. Chu, Diagnosis of multiple cancer types by shrunken
 * centroids of gene expression, PNAS 99 (2002).
 * @example
 * ```ts
 * import { nearestCentroid } from 'omai-engine/learn'
 * import { toFlat } from 'omai-engine/tensor'
 * const c = nearestCentroid().fit({ x: [[0, 0], [0, 1], [9, 9], [9, 8]], y: [0, 0, 1, 1] })
 * toFlat(c.predict([[1, 1], [8, 8]])) // => [0, 1]
 * ```
 */
export function nearestCentroid(
  hyper: Partial<NearestCentroidHyper> = {},
): IterativeEstimator<NearestCentroidHyper, RowsState, NearestCentroidParams, NearestCentroidVerbs> {
  const defaults: NearestCentroidHyper = { metric: 'euclidean', priors: 'uniform', shrinkThreshold: 0 }
  return iterative<NearestCentroidHyper, RowsState, NearestCentroidParams, NearestCentroidVerbs>(
    {
      name: 'nearest-centroid',
      kind: 'classifier',
      defaults,
      hyperSpace: (h) =>
        dict({
          shrinkThreshold: real(0, Math.max(5, h.shrinkThreshold), { default: h.shrinkThreshold }),
          metric: choice(['euclidean', 'manhattan'] as const, { default: h.metric }),
        }),
      algorithm: (data) => rowsAlgorithm('nearest-centroid', rowsOf(data.x)),
      maxSteps: (_h, data) => rowsOf(data.x),
      logs: { 'solver/rows': Log() },
      params: (s, data, h) => {
        const { classes } = encodeLabels(data.y ?? [], 'nearestCentroid', rowsOf(data.x))
        const { m, head } = memoryOf(data, s.n, 'nearestCentroid')
        const { codes: headCodes } = encodeAgainst(head.y ?? [], classes)
        // Classes unseen in the first rows have no centroid yet; they are dropped until they appear.
        const present = classes.map((_, c) => headCodes.includes(c))
        const keep = classes.filter((_, c) => present[c])
        const remap = new Map(keep.map((cl, i) => [classes.indexOf(cl), i]))
        const fit = centroidFit(
          m,
          Int32Array.from(headCodes, (c) => remap.get(c) ?? 0),
          keep.length,
          h,
        )
        return { classes: keep, ...fit }
      },
      verbs: (p, h) => centroidVerbs(p, h),
    },
    { ...defaults, ...hyper },
  )
}

function encodeAgainst(y: TensorLike, classes: readonly number[]): { readonly codes: number[] } {
  const index = new Map(classes.map((c, i) => [c, i]))
  return { codes: Array.from(targetVector(y, 'nearestCentroid'), (v) => index.get(v) ?? 0) }
}

/** One point on the path of {@link shrunkenCentroidPath}. */
export interface ShrunkenCentroids {
  readonly threshold: number
  /** The centroids at this threshold `[k][d]`. */
  readonly centroids: number[][]
  /** Whether each feature still separates some class from the overall centroid. */
  readonly active: boolean[]
  /** The number of active features. */
  readonly nActive: number
}

/**
 * The path of nearest shrunken centroids: for each threshold, the centroids and the features that still matter (those
 * with a non-zero shrunken deviation in some class), so features drop out of the classifier one by one as the
 * threshold grows.
 *
 * @param data - Features and class labels.
 * @param thresholds - The thresholds, in the order wanted.
 * @param hyper - `metric` and `priors` (the threshold is the path's).
 * @returns One entry per threshold.
 * @example
 * ```ts
 * import { shrunkenCentroidPath } from 'omai-engine/learn'
 * const x = [[0, 5], [0.2, 4], [4, 5.1], [4.2, 4.9]]
 * shrunkenCentroidPath({ x, y: [0, 0, 1, 1] }, [0, 100]).map((p) => p.nActive) // => [2, 0]
 * ```
 */
export function shrunkenCentroidPath(
  data: LearnData,
  thresholds: readonly number[],
  hyper: Partial<Omit<NearestCentroidHyper, 'shrinkThreshold'>> = {},
): ShrunkenCentroids[] {
  const m = matrixView(data.x, 'shrunkenCentroidPath')
  const { classes, codes } = encodeLabels(data.y ?? [], 'shrunkenCentroidPath', m.n)
  return thresholds.map((threshold) => {
    const fit = centroidFit(m, codes, classes.length, {
      metric: 'euclidean',
      priors: 'uniform',
      ...hyper,
      shrinkThreshold: threshold,
    })
    const active = Array.from({ length: m.d }, (_, j) => fit.deviations.some((row) => (row[j] ?? 0) !== 0))
    return { threshold, centroids: fit.centroids, active, nActive: active.filter(Boolean).length }
  })
}
