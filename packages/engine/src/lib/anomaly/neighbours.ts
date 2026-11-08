/**
 * Neighbour-based outlier detectors over the neighbour search of `learn` (any metric of `distances`, brute force or a
 * KD or ball tree): the local outlier factor (Breunig, Kriegel, Ng and Sander 2000; as in scikit-learn's
 * `LocalOutlierFactor`), the connectivity-based outlier factor (Tang, Chen, Fu and Cheung 2002) and the
 * k-nearest-neighbour distance score (Ramaswamy, Rastogi and Shim 2000; Angiulli and Pizzuti 2002). Each is a short
 * stepped algorithm (the neighbourhoods, then the densities or chaining distances, then the factors) and an estimator
 * on the iterative base whose verbs score new points in novelty mode; the training scores are in the parameters.
 */

import { maxOf, OmaiError, toRows } from '../base/index.ts'
import { pairwiseDistances } from '../distances/index.ts'
import {
  ExplainedEstimatorClass,
  type Fitted,
  type IterativeEstimator,
  type IterativeFitted,
  iterative,
  type MatrixView,
  matrixView,
  type NeighbourHits,
  type NeighbourIndex,
  type NeighbourSearch,
  neighbourIndex,
} from '../learn/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { median } from '../stats/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { contaminationThreshold, type OutlierVerbs, outlierExposition, outlierVerbs } from './combine.ts'

const DEFAULT_SEARCH: NeighbourSearch = { metric: 'euclidean', metricOptions: {}, algorithm: 'auto', leafSize: 30 }

/** scikit-learn adds this to the mean reachability distance so that duplicated points do not divide by zero. */
const LRD_EPSILON = 1e-10

const searchOf = (h: Partial<NeighbourSearch>): Partial<NeighbourSearch> => ({
  ...(h.metric === undefined ? {} : { metric: h.metric }),
  ...(h.metricOptions === undefined ? {} : { metricOptions: h.metricOptions }),
  ...(h.algorithm === undefined ? {} : { algorithm: h.algorithm }),
  ...(h.leafSize === undefined ? {} : { leafSize: h.leafSize }),
})

const asTensor2 = (m: MatrixView): Tensor => fromStorage(m.data, [m.n, m.d], 'float64')

const rowsOf = (m: MatrixView): number[][] => toRows(m.data, m.n, m.d)

/** The neighbours a detector asks for: at least one, and at most the other `n - 1` rows. */
function effectiveK(nNeighbors: number, n: number, who: string): number {
  if (!(Number.isInteger(nNeighbors) && nNeighbors >= 1))
    throw new OmaiError('value', `${who}: nNeighbors must be a positive integer`)
  if (n < 2) throw new OmaiError('shape', `${who}: needs at least 2 rows`)
  return Math.max(1, Math.min(nNeighbors, n - 1))
}

/**
 * The `k` nearest other rows of every row of an indexed point set: the row itself is dropped by index (the last of the
 * `k + 1` when duplicates push it out of the list), as scikit-learn's `kneighbors()` without a query.
 */
function trainingHits(index: NeighbourIndex, m: MatrixView, k: number): NeighbourHits[] {
  return index.nearest(asTensor2(m), k + 1).map((hit, i) => {
    const at = hit.indices.indexOf(i)
    const drop = at >= 0 ? at : hit.indices.length - 1
    const keep = hit.indices.map((_, j) => j).filter((j) => j !== drop)
    return { indices: keep.map((j) => hit.indices[j] ?? 0), distances: keep.map((j) => hit.distances[j] ?? 0) }
  })
}

const mean = (v: readonly number[]): number => (v.length === 0 ? 0 : v.reduce((a, b) => a + b, 0) / v.length)

/** The verbs every neighbour detector shares: the outlier verbs, the neighbours of new rows, and the feature count. */
export interface NeighbourDetectorVerbs extends OutlierVerbs {
  /** The nearest training rows of each row of `x` (all `k` of them: a new row is not its own neighbour). */
  readonly kneighbors: (x: TensorLike) => NeighbourHits[]
}

// ---------------------------------------------------------------------------------------------------------------
// The local outlier factor

/** Options of {@link lofAlgorithm}: the neighbourhood size and the neighbour search. */
export interface LofAlgorithmOptions extends Partial<NeighbourSearch> {
  /** The number of neighbours `k` (default 20, capped at `n - 1`). */
  readonly nNeighbors?: number
}

/**
 * The state of the local outlier factor: `stage` 0 (nothing), 1 (neighbourhoods and k-distances), 2 (local
 * reachability densities), 3 (factors).
 */
export interface LofState {
  readonly stage: number
  readonly k: number
  /** The `k` nearest other rows of each row, nearest first. */
  readonly neighbours: number[][]
  /** Their distances. */
  readonly distances: number[][]
  /** The distance to the `k`-th neighbour. */
  readonly kDistance: number[]
  /** Local reachability density: the inverse mean reachability distance to the neighbours. */
  readonly lrd: number[]
  /** The local outlier factor: the mean density ratio of the neighbours to the row (1 for an inlier). */
  readonly lof: number[]
}

/**
 * The local outlier factor (Breunig, Kriegel, Ng and Sander 2000; scikit-learn's `LocalOutlierFactor`) as a stepped
 * {@link Algorithm}: step 1 finds each row's `k` nearest other rows and its k-distance, step 2 the local reachability
 * densities `lrd(p) = 1 / (mean over o in N_k(p) of max(kdist(o), d(p, o)) + 1e-10)`, step 3 the factors
 * `LOF(p) = mean over o in N_k(p) of lrd(o) / lrd(p)`.
 *
 * @param x - Points `[n, d]`.
 * @param options - The neighbourhood size and the neighbour search (metric, algorithm).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { lofAlgorithm } from 'omai-engine/anomaly'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10]]
 * const lof = runAlgorithm(lofAlgorithm(x, { nNeighbors: 2 }), { maxSteps: 3 }).state.lof
 * lof[4] > 2 // => true
 * ```
 */
export function lofAlgorithm(x: TensorLike, options: LofAlgorithmOptions = {}): Algorithm<LofState> {
  const m = matrixView(x, 'localOutlierFactor', { minSamples: 2 })
  const k = effectiveK(options.nNeighbors ?? 20, m.n, 'localOutlierFactor')
  const empty = (): LofState => ({ stage: 0, k, neighbours: [], distances: [], kDistance: [], lrd: [], lof: [] })
  return {
    name: 'local-outlier-factor',
    config: { nNeighbors: k, metric: options.metric ?? 'euclidean' },
    schema: channels({
      metrics: {
        stage: { description: 'the stage reached: 1 neighbourhoods, 2 densities, 3 factors' },
        maxFactor: { description: 'the largest local outlier factor' },
      },
      tensors: {
        kDistance: { role: 'state', description: 'distance to the k-th neighbour of each row', shape: ['n'] },
        lrd: { role: 'state', description: 'local reachability density of each row', shape: ['n'] },
        lof: { role: 'state', description: 'local outlier factor of each row', shape: ['n'] },
      },
    }),
    init: empty,
    step: (s) => {
      if (s.stage === 0) {
        const hits = trainingHits(neighbourIndex(asTensor2(m), searchOf(options)), m, k)
        return {
          ...s,
          stage: 1,
          neighbours: hits.map((h) => [...h.indices]),
          distances: hits.map((h) => [...h.distances]),
          kDistance: hits.map((h) => h.distances[k - 1] ?? 0),
        }
      }
      if (s.stage === 1) {
        const lrd = s.neighbours.map(
          (ix, i) =>
            1 / (mean(ix.map((o, j) => Math.max(s.kDistance[o] ?? 0, s.distances[i]?.[j] ?? 0))) + LRD_EPSILON),
        )
        return { ...s, stage: 2, lrd }
      }
      const lof = s.neighbours.map((ix, i) => mean(ix.map((o) => (s.lrd[o] ?? 0) / (s.lrd[i] ?? 1))))
      return { ...s, stage: 3, lof }
    },
    done: (s) => (s.stage >= 3 ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { stage: s.stage, ...(s.stage >= 3 ? { maxFactor: maxOf(s.lof) } : {}) },
      tensors: {
        ...(s.stage >= 1 ? { kDistance: { role: 'state' as const, value: s.kDistance, shape: [m.n] } } : {}),
        ...(s.stage >= 2 ? { lrd: { role: 'state' as const, value: s.lrd, shape: [m.n] } } : {}),
        ...(s.stage >= 3 ? { lof: { role: 'state' as const, value: s.lof, shape: [m.n] } } : {}),
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as LofState,
  }
}

/**
 * The local outlier factor of every row of a data set, with its k-distances, densities and neighbourhoods.
 *
 * @param x - Points `[n, d]`.
 * @param options - The neighbourhood size and the neighbour search.
 * @returns The final {@link LofState}.
 * @example
 * ```ts
 * import { localOutlierFactors } from 'omai-engine/anomaly'
 * const r = localOutlierFactors([[0], [1], [2], [3], [20]], { nNeighbors: 2 })
 * [r.lof[4] > 3, r.lof[2] < 1.5] // => [true, true]
 * ```
 */
export function localOutlierFactors(x: TensorLike, options: LofAlgorithmOptions = {}): LofState {
  const alg = lofAlgorithm(x, options)
  let s = alg.init(undefined as never)
  while (s.stage < 3) s = alg.step(s, undefined as never)
  return s
}

/** Hyperparameters of the local outlier factor. */
export interface LofHyper extends NeighbourSearch {
  /** The number of neighbours `k` (default 20). */
  readonly nNeighbors: number
  /** The expected fraction of outliers in `(0, 0.5]`, or `'auto'` (default: the threshold 1.5 on the factor). */
  readonly contamination: number | 'auto'
}

/** The parameters of a fitted local outlier factor model: the training rows and their factors. */
export type LofParams = {
  readonly k: number
  readonly x: number[][]
  readonly kDistance: number[]
  readonly lrd: number[]
  /** The factors of the training rows (scikit-learn's `-negative_outlier_factor_`). */
  readonly scores: number[]
  /** The factor above which a row is an outlier. */
  readonly threshold: number
}

/** The verbs of a fitted local outlier factor model: `scoreSamples` is the novelty score of new rows. */
export type LofVerbs = NeighbourDetectorVerbs

/** A fitted local outlier factor model. */
export type FittedLof = IterativeFitted<LofParams, LofVerbs>

const AUTO_LOF_THRESHOLD = 1.5

function lofThreshold(scores: readonly number[], contamination: number | 'auto'): number {
  return contamination === 'auto' ? AUTO_LOF_THRESHOLD : contaminationThreshold(scores, contamination)
}

function lofVerbs(p: LofParams, h: LofHyper): LofVerbs {
  let index: NeighbourIndex | undefined
  const find = (x: TensorLike): NeighbourHits[] => {
    index ??= neighbourIndex(p.x, searchOf(h))
    matrixView(x, 'localOutlierFactor', { nFeatures: p.x[0]?.length ?? 0 })
    return index.nearest(x, p.k)
  }
  const scores = (x: TensorLike): number[] =>
    find(x).map((hit) => {
      const reach = hit.indices.map((o, j) => Math.max(p.kDistance[o] ?? 0, hit.distances[j] ?? 0))
      const lrdQuery = 1 / (mean(reach) + LRD_EPSILON)
      return mean(hit.indices.map((o) => (p.lrd[o] ?? 0) / lrdQuery))
    })
  return {
    ...outlierVerbs(scores, p.threshold, p.x[0]?.length ?? 0),
    kneighbors: find,
  }
}

/**
 * The local outlier factor (sklearn.neighbors.LocalOutlierFactor) on the iterative base. A row's factor compares its
 * local density with its neighbours': near 1 inside a cluster, well above 1 for a row less dense than its
 * neighbourhood. `fit` scores the training rows leaving each out of its own neighbourhood (`params.scores`); the verbs
 * score new rows against the training set (scikit-learn's `novelty=True`). Its exposition is an `anomalyScores` record
 * of the training factors, the densities as evidence, and a `decisionSurface` of the novelty score for one or two
 * features.
 *
 * @param hyper - `nNeighbors`, `contamination` (a fraction or `'auto'`) and the neighbour search.
 * @returns The iterative outlier estimator.
 * @index estimator Local outlier factor.
 * @example
 * ```ts
 * import { localOutlierFactor } from 'omai-engine/anomaly'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [10, 10]]
 * const f = localOutlierFactor({ nNeighbors: 3, contamination: 0.2 }).fit({ x })
 * toFlat(f.predict([[0.5, 0.4], [10, 10.5]])) // => [1, -1]
 * ```
 */
export function localOutlierFactor(
  hyper: Partial<LofHyper> = {},
): IterativeEstimator<LofHyper, LofState, LofParams, LofVerbs> {
  return iterative<LofHyper, LofState, LofParams, LofVerbs>({
    name: 'local-outlier-factor',
    kind: 'outlier',
    defaults: { ...DEFAULT_SEARCH, nNeighbors: 20, contamination: 'auto', ...hyper },
    hyperSpace: (h) =>
      dict({
        nNeighbors: int(1, 200, {
          default: h.nNeighbors,
          label: 'neighbours',
          description: 'The size k of the neighbourhood.',
        }),
        ...(typeof h.contamination === 'number'
          ? {
              contamination: real(0.01, 0.5, {
                default: h.contamination,
                label: 'contamination',
                description: 'The expected fraction of outliers: sets the threshold on the factor.',
              }),
            }
          : {}),
      }),
    algorithm: (data, h) => lofAlgorithm(data.x, { ...searchOf(h), nNeighbors: h.nNeighbors }),
    maxSteps: () => 3,
    params: (s, data, h) => ({
      k: s.k,
      x: rowsOf(matrixView(data.x, 'localOutlierFactor', { minSamples: 2 })),
      kDistance: s.kDistance,
      lrd: s.lrd,
      scores: s.lof,
      threshold: s.stage >= 3 ? lofThreshold(s.lof, h.contamination) : AUTO_LOF_THRESHOLD,
    }),
    verbs: lofVerbs,
    records: (s, data, h) => {
      const params: LofParams = {
        k: s.k,
        x: rowsOf(matrixView(data.x, 'localOutlierFactor', { minSamples: 2 })),
        kDistance: s.kDistance,
        lrd: s.lrd,
        scores: s.lof,
        threshold: lofThreshold(s.lof, h.contamination),
      }
      const fitted = { estimator: 'local-outlier-factor', kind: 'outlier' as const, params, ...lofVerbs(params, h) }
      return outlierExposition(
        fitted,
        data.x,
        'localOutlierFactor',
        params.threshold,
        { name: 'localReachabilityDensity', values: s.lrd },
        s.lof,
      )
    },
    logs: { 'solver/stage': Log(), 'solver/maxFactor': Log() },
  })
}

/**
 * The local outlier factor as a scikit-learn-style class wrapper.
 *
 * @example
 * ```ts
 * import { LocalOutlierFactor } from 'omai-engine/anomaly'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [10, 10]]
 * toFlat(new LocalOutlierFactor({ nNeighbors: 3, contamination: 0.2 }).fit(x).predict([[0.5, 0.4], [10, 10.5]])) // => [1, -1]
 * ```
 */
export class LocalOutlierFactor extends ExplainedEstimatorClass<LofHyper, Fitted<LofParams> & LofVerbs> {
  /**
   * Make an unfitted local outlier factor model.
   *
   * @param hyper - The hyperparameters of {@link localOutlierFactor}.
   */
  constructor(hyper: Partial<LofHyper> = {}) {
    super(localOutlierFactor(hyper))
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The connectivity-based outlier factor

/** Options of {@link cofAlgorithm}: the neighbourhood size and the neighbour search. */
export type CofAlgorithmOptions = LofAlgorithmOptions

/**
 * The state of the connectivity-based outlier factor: `stage` 0 (nothing), 1 (neighbourhoods), 2 (average chaining
 * distances), 3 (factors).
 */
export interface CofState {
  readonly stage: number
  readonly k: number
  readonly neighbours: number[][]
  readonly distances: number[][]
  /** The average chaining distance of each row's set-based nearest path through its neighbours. */
  readonly acDist: number[]
  /** The connectivity-based outlier factor. */
  readonly cof: number[]
}

/**
 * The average chaining distance of a point through its neighbours (Tang et al. 2002, Definition 4). The set-based
 * nearest path starts at the point and repeatedly adds the neighbour closest to any point already on the path; with
 * edge costs `c_1 .. c_r`, `ac = sum_i 2 (r + 1 - i) / (r (r + 1)) c_i`, which weights the early edges (the ones
 * nearest the point) most.
 *
 * @param within - The distances among the point (index 0) and its `r` neighbours (indices 1 to `r`), `(r + 1) x (r + 1)`.
 * @returns The average chaining distance, 0 for no neighbours.
 * @example
 * ```ts
 * import { averageChainingDistance } from 'omai-engine/anomaly'
 * // A point 1 from a neighbour that is 2 from the next: costs 1 and 2, weights 2/3 and 1/3.
 * averageChainingDistance([[0, 1, 3], [1, 0, 2], [3, 2, 0]]) // => 1.3333333333333333
 * ```
 */
export function averageChainingDistance(within: readonly (readonly number[])[]): number {
  const r = within.length - 1
  if (r <= 0) return 0
  const onPath = [0]
  let remaining = Array.from({ length: r }, (_, i) => i + 1)
  let ac = 0
  for (let step = 1; step <= r; step++) {
    let best = Number.POSITIVE_INFINITY
    let at = 0
    remaining.forEach((cand, pos) => {
      for (const v of onPath) {
        const c = within[v]?.[cand] ?? Number.POSITIVE_INFINITY
        if (c < best) {
          best = c
          at = pos
        }
      }
    })
    const next = remaining[at] ?? 0
    onPath.push(next)
    remaining = remaining.filter((_, pos) => pos !== at)
    ac += ((2 * (r + 1 - step)) / (r * (r + 1))) * best
  }
  return ac
}

/** The distances among a point (first) and its neighbours, by the metric of the search. */
function groupDistances(
  point: readonly number[],
  neighbours: readonly (readonly number[])[],
  search: Partial<NeighbourSearch>,
): number[][] {
  const g = [point, ...neighbours]
  const flat = toFlat(
    primalOf(pairwiseDistances(g, undefined, search.metric ?? 'euclidean', search.metricOptions ?? {})),
  )
  return g.map((_, a) => g.map((__, b) => flat[a * g.length + b] ?? 0))
}

const connectivity = (acPoint: number, acNeighbours: readonly number[]): number => {
  const m = mean(acNeighbours)
  return m > 0 ? acPoint / m : acPoint > 0 ? Number.POSITIVE_INFINITY : 1
}

/**
 * The connectivity-based outlier factor (Tang, Chen, Fu and Cheung 2002) as a stepped {@link Algorithm}: step 1 finds
 * the `k` nearest other rows of each row, step 2 the average chaining distance of each row through them, step 3 the
 * factors `COF(p) = ac(p) / mean over o in N_k(p) of ac(o)`. Unlike the local outlier factor it follows the chain of
 * nearest links rather than distances from the point, so a row off the end of a line or a curve scores high while a row
 * in the middle of a sparse line does not.
 *
 * @param x - Points `[n, d]`.
 * @param options - The neighbourhood size and the neighbour search.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { cofAlgorithm } from 'omai-engine/anomaly'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [2, 3]]
 * const cof = runAlgorithm(cofAlgorithm(x, { nNeighbors: 2 }), { maxSteps: 3 }).state.cof
 * cof[5] > 2 // => true
 * ```
 */
export function cofAlgorithm(x: TensorLike, options: CofAlgorithmOptions = {}): Algorithm<CofState> {
  const m = matrixView(x, 'connectivityOutlierFactor', { minSamples: 2 })
  const k = effectiveK(options.nNeighbors ?? 20, m.n, 'connectivityOutlierFactor')
  const rows = rowsOf(m)
  return {
    name: 'connectivity-outlier-factor',
    config: { nNeighbors: k, metric: options.metric ?? 'euclidean' },
    schema: channels({
      metrics: {
        stage: { description: 'the stage reached: 1 neighbourhoods, 2 chaining distances, 3 factors' },
        maxFactor: { description: 'the largest connectivity-based outlier factor' },
      },
      tensors: {
        acDist: { role: 'state', description: 'average chaining distance of each row', shape: ['n'] },
        cof: { role: 'state', description: 'connectivity-based outlier factor of each row', shape: ['n'] },
      },
    }),
    init: () => ({ stage: 0, k, neighbours: [], distances: [], acDist: [], cof: [] }),
    step: (s) => {
      if (s.stage === 0) {
        const hits = trainingHits(neighbourIndex(asTensor2(m), searchOf(options)), m, k)
        return {
          ...s,
          stage: 1,
          neighbours: hits.map((h) => [...h.indices]),
          distances: hits.map((h) => [...h.distances]),
        }
      }
      if (s.stage === 1) {
        const acDist = s.neighbours.map((ix, i) =>
          averageChainingDistance(
            groupDistances(
              rows[i] ?? [],
              ix.map((o) => rows[o] ?? []),
              options,
            ),
          ),
        )
        return { ...s, stage: 2, acDist }
      }
      const cof = s.neighbours.map((ix, i) =>
        connectivity(
          s.acDist[i] ?? 0,
          ix.map((o) => s.acDist[o] ?? 0),
        ),
      )
      return { ...s, stage: 3, cof }
    },
    done: (s) => (s.stage >= 3 ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { stage: s.stage, ...(s.stage >= 3 ? { maxFactor: maxOf(s.cof) } : {}) },
      tensors: {
        ...(s.stage >= 2 ? { acDist: { role: 'state' as const, value: s.acDist, shape: [m.n] } } : {}),
        ...(s.stage >= 3 ? { cof: { role: 'state' as const, value: s.cof, shape: [m.n] } } : {}),
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CofState,
  }
}

/**
 * The connectivity-based outlier factor of every row of a data set.
 *
 * @param x - Points `[n, d]`.
 * @param options - The neighbourhood size and the neighbour search.
 * @returns The final {@link CofState}.
 * @example
 * ```ts
 * import { connectivityOutlierFactors } from 'omai-engine/anomaly'
 * connectivityOutlierFactors([[0], [1], [2], [3], [4]], { nNeighbors: 2 }).cof.length // => 5
 * ```
 */
export function connectivityOutlierFactors(x: TensorLike, options: CofAlgorithmOptions = {}): CofState {
  const alg = cofAlgorithm(x, options)
  let s = alg.init(undefined as never)
  while (s.stage < 3) s = alg.step(s, undefined as never)
  return s
}

/** Hyperparameters of the connectivity-based outlier factor. */
export interface CofHyper extends NeighbourSearch {
  /** The number of neighbours `k` (default 20). */
  readonly nNeighbors: number
  /** The expected fraction of outliers in `(0, 0.5]` (default 0.1). */
  readonly contamination: number
}

/** The parameters of a fitted connectivity-based outlier factor model. */
export type CofParams = {
  readonly k: number
  readonly x: number[][]
  readonly acDist: number[]
  /** The factors of the training rows. */
  readonly scores: number[]
  readonly threshold: number
}

/** A fitted connectivity-based outlier factor model. */
export type FittedCof = IterativeFitted<CofParams, NeighbourDetectorVerbs>

function cofVerbs(p: CofParams, h: CofHyper): NeighbourDetectorVerbs {
  let index: NeighbourIndex | undefined
  const find = (x: TensorLike): NeighbourHits[] => {
    index ??= neighbourIndex(p.x, searchOf(h))
    return index.nearest(x, p.k)
  }
  const scores = (x: TensorLike): number[] => {
    const q = matrixView(x, 'connectivityOutlierFactor', { nFeatures: p.x[0]?.length ?? 0 })
    return find(x).map((hit, i) =>
      connectivity(
        averageChainingDistance(
          groupDistances(
            Array.from(q.data.subarray(i * q.d, (i + 1) * q.d)),
            hit.indices.map((o) => p.x[o] ?? []),
            h,
          ),
        ),
        hit.indices.map((o) => p.acDist[o] ?? 0),
      ),
    )
  }
  return { ...outlierVerbs(scores, p.threshold, p.x[0]?.length ?? 0), kneighbors: find }
}

/**
 * The connectivity-based outlier factor (Tang et al. 2002) on the iterative base: the factor of each training row
 * leaves the row out of its own neighbourhood, the verbs score new rows against the training set. Its exposition is an
 * `anomalyScores` record of the training factors with the average chaining distances as evidence.
 *
 * @param hyper - `nNeighbors`, `contamination` and the neighbour search.
 * @returns The iterative outlier estimator.
 * @example
 * ```ts
 * import { connectivityOutlierFactor } from 'omai-engine/anomaly'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [2, 4]]
 * const f = connectivityOutlierFactor({ nNeighbors: 3, contamination: 0.15 }).fit({ x })
 * toFlat(f.predict([[2.5, 0], [2.5, 4]])) // => [1, -1]
 * ```
 */
export function connectivityOutlierFactor(
  hyper: Partial<CofHyper> = {},
): IterativeEstimator<CofHyper, CofState, CofParams, NeighbourDetectorVerbs> {
  const paramsOf = (s: CofState, data: { readonly x: TensorLike }, h: CofHyper): CofParams => ({
    k: s.k,
    x: rowsOf(matrixView(data.x, 'connectivityOutlierFactor', { minSamples: 2 })),
    acDist: s.acDist,
    scores: s.cof,
    threshold: s.stage >= 3 ? contaminationThreshold(s.cof, h.contamination) : Number.POSITIVE_INFINITY,
  })
  return iterative<CofHyper, CofState, CofParams, NeighbourDetectorVerbs>({
    name: 'connectivity-outlier-factor',
    kind: 'outlier',
    defaults: { ...DEFAULT_SEARCH, nNeighbors: 20, contamination: 0.1, ...hyper },
    hyperSpace: (h) =>
      dict({
        nNeighbors: int(1, 200, {
          default: h.nNeighbors,
          label: 'neighbours',
          description: 'The size k of the neighbourhood.',
        }),
        contamination: real(0.01, 0.5, {
          default: h.contamination,
          label: 'contamination',
          description: 'The expected fraction of outliers: sets the threshold on the factor.',
        }),
      }),
    algorithm: (data, h) => cofAlgorithm(data.x, { ...searchOf(h), nNeighbors: h.nNeighbors }),
    maxSteps: () => 3,
    params: paramsOf,
    verbs: cofVerbs,
    records: (s, data, h) => {
      const params = paramsOf(s, data, h)
      const fitted = {
        estimator: 'connectivity-outlier-factor',
        kind: 'outlier' as const,
        params,
        ...cofVerbs(params, h),
      }
      return outlierExposition(
        fitted,
        data.x,
        'connectivityOutlierFactor',
        params.threshold,
        { name: 'averageChainingDistance', values: s.acDist },
        s.cof,
      )
    },
    logs: { 'solver/stage': Log(), 'solver/maxFactor': Log() },
  })
}

/**
 * The connectivity-based outlier factor as a scikit-learn-style class wrapper.
 *
 * @example
 * ```ts
 * import { ConnectivityOutlierFactor } from 'omai-engine/anomaly'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [2, 4]]
 * toFlat(new ConnectivityOutlierFactor({ nNeighbors: 3, contamination: 0.15 }).fit(x).predict([[2.5, 0], [2.5, 4]])) // => [1, -1]
 * ```
 */
export class ConnectivityOutlierFactor extends ExplainedEstimatorClass<
  CofHyper,
  Fitted<CofParams> & NeighbourDetectorVerbs
> {
  /**
   * Make an unfitted connectivity-based outlier factor model.
   *
   * @param hyper - The hyperparameters of {@link connectivityOutlierFactor}.
   */
  constructor(hyper: Partial<CofHyper> = {}) {
    super(connectivityOutlierFactor(hyper))
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The k-nearest-neighbour distance

/**
 * How the distances to the `k` nearest neighbours become a score: the `k`-th distance (`'largest'`, Ramaswamy et al.
 * 2000), their mean (Angiulli and Pizzuti 2002) or their median.
 */
export type KnnAggregation = 'largest' | 'mean' | 'median'

/** Aggregate neighbour distances (nearest first) into a score. */
function aggregateDistances(distances: readonly number[], how: KnnAggregation): number {
  if (distances.length === 0) return 0
  if (how === 'largest') return distances[distances.length - 1] ?? 0
  if (how === 'mean') return mean(distances)
  return median([...distances])
}

/** The state of the k-nearest-neighbour distance score: the neighbourhoods, then the scores. */
export interface KnnState {
  readonly stage: number
  readonly k: number
  readonly neighbours: number[][]
  readonly distances: number[][]
  readonly scores: number[]
}

/** Options of {@link knnAlgorithm}. */
export interface KnnAlgorithmOptions extends Partial<NeighbourSearch> {
  /** The number of neighbours `k` (default 5, capped at `n - 1`). */
  readonly nNeighbors?: number
  /** The aggregation of the `k` distances (default `'largest'`). */
  readonly aggregation?: KnnAggregation
}

/**
 * The k-nearest-neighbour distance score as a stepped {@link Algorithm}: step 1 finds each row's `k` nearest other
 * rows, step 2 aggregates their distances.
 *
 * @param x - Points `[n, d]`.
 * @param options - The neighbourhood size, the aggregation and the neighbour search.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { knnAlgorithm } from 'omai-engine/anomaly'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(knnAlgorithm([[0], [1], [3], [10]], { nNeighbors: 1 }), { maxSteps: 2 }).state.scores // => [1, 1, 2, 7]
 * ```
 */
export function knnAlgorithm(x: TensorLike, options: KnnAlgorithmOptions = {}): Algorithm<KnnState> {
  const m = matrixView(x, 'knnDetector', { minSamples: 2 })
  const k = effectiveK(options.nNeighbors ?? 5, m.n, 'knnDetector')
  const how = options.aggregation ?? 'largest'
  return {
    name: 'knn-distance',
    config: { nNeighbors: k, aggregation: how, metric: options.metric ?? 'euclidean' },
    schema: channels({
      metrics: {
        stage: { description: 'the stage reached: 1 neighbourhoods, 2 scores' },
        maxScore: { description: 'the largest neighbour-distance score' },
      },
      tensors: { scores: { role: 'state', description: 'neighbour-distance score of each row', shape: ['n'] } },
    }),
    init: () => ({ stage: 0, k, neighbours: [], distances: [], scores: [] }),
    step: (s) => {
      if (s.stage === 0) {
        const hits = trainingHits(neighbourIndex(asTensor2(m), searchOf(options)), m, k)
        return {
          ...s,
          stage: 1,
          neighbours: hits.map((h) => [...h.indices]),
          distances: hits.map((h) => [...h.distances]),
        }
      }
      return { ...s, stage: 2, scores: s.distances.map((d) => aggregateDistances(d, how)) }
    },
    done: (s) => (s.stage >= 2 ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { stage: s.stage, ...(s.stage >= 2 ? { maxScore: maxOf(s.scores) } : {}) },
      tensors: s.stage >= 2 ? { scores: { role: 'state' as const, value: s.scores, shape: [m.n] } } : {},
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as KnnState,
  }
}

/** Hyperparameters of the k-nearest-neighbour distance detector. */
export interface KnnDetectorHyper extends NeighbourSearch {
  /** The number of neighbours `k` (default 5). */
  readonly nNeighbors: number
  /** The aggregation of the `k` distances (default `'largest'`). */
  readonly aggregation: KnnAggregation
  /** The expected fraction of outliers in `(0, 0.5]` (default 0.1). */
  readonly contamination: number
}

/** The parameters of a fitted k-nearest-neighbour distance detector. */
export type KnnDetectorParams = {
  readonly k: number
  readonly x: number[][]
  readonly aggregation: KnnAggregation
  /** The scores of the training rows. */
  readonly scores: number[]
  readonly threshold: number
}

/** A fitted k-nearest-neighbour distance detector. */
export type FittedKnnDetector = IterativeFitted<KnnDetectorParams, NeighbourDetectorVerbs>

function knnVerbs(p: KnnDetectorParams, h: KnnDetectorHyper): NeighbourDetectorVerbs {
  let index: NeighbourIndex | undefined
  const find = (x: TensorLike): NeighbourHits[] => {
    index ??= neighbourIndex(p.x, searchOf(h))
    return index.nearest(x, p.k)
  }
  const scores = (x: TensorLike): number[] => find(x).map((hit) => aggregateDistances(hit.distances, p.aggregation))
  return { ...outlierVerbs(scores, p.threshold, p.x[0]?.length ?? 0), kneighbors: find }
}

/**
 * The k-nearest-neighbour distance detector on the iterative base: a row's score is its distance to the `k`-th nearest
 * other row (`'largest'`), or the mean or median of its `k` nearest distances, thresholded at the
 * `(1 - contamination)` quantile of the training scores. It misses an outlier next to a dense cluster that the local
 * outlier factor catches, as the density of the neighbourhood is not taken into account.
 *
 * @param hyper - `nNeighbors`, `aggregation`, `contamination` and the neighbour search.
 * @returns The iterative outlier estimator.
 * @example
 * ```ts
 * import { knnDetector } from 'omai-engine/anomaly'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [10, 10]]
 * toFlat(knnDetector({ nNeighbors: 2, contamination: 0.2 }).fit({ x }).predict([[0.5, 0.4], [10, 10.5]])) // => [1, -1]
 * ```
 */
export function knnDetector(
  hyper: Partial<KnnDetectorHyper> = {},
): IterativeEstimator<KnnDetectorHyper, KnnState, KnnDetectorParams, NeighbourDetectorVerbs> {
  const paramsOf = (s: KnnState, data: { readonly x: TensorLike }, h: KnnDetectorHyper): KnnDetectorParams => ({
    k: s.k,
    x: rowsOf(matrixView(data.x, 'knnDetector', { minSamples: 2 })),
    aggregation: h.aggregation,
    scores: s.scores,
    threshold: s.stage >= 2 ? contaminationThreshold(s.scores, h.contamination) : Number.POSITIVE_INFINITY,
  })
  return iterative<KnnDetectorHyper, KnnState, KnnDetectorParams, NeighbourDetectorVerbs>({
    name: 'knn-distance',
    kind: 'outlier',
    defaults: { ...DEFAULT_SEARCH, nNeighbors: 5, aggregation: 'largest', contamination: 0.1, ...hyper },
    hyperSpace: (h) =>
      dict({
        nNeighbors: int(1, 200, {
          default: h.nNeighbors,
          label: 'neighbours',
          description: 'The number k of neighbours.',
        }),
        aggregation: choice(['largest', 'mean', 'median'] as const, {
          default: h.aggregation,
          label: 'aggregation',
          description: 'How the k distances become a score: the k-th, their mean or their median.',
        }),
        contamination: real(0.01, 0.5, {
          default: h.contamination,
          label: 'contamination',
          description: 'The expected fraction of outliers: sets the threshold on the score.',
        }),
      }),
    algorithm: (data, h) =>
      knnAlgorithm(data.x, { ...searchOf(h), nNeighbors: h.nNeighbors, aggregation: h.aggregation }),
    maxSteps: () => 2,
    params: paramsOf,
    verbs: knnVerbs,
    records: (s, data, h) => {
      const params = paramsOf(s, data, h)
      const fitted = { estimator: 'knn-distance', kind: 'outlier' as const, params, ...knnVerbs(params, h) }
      return outlierExposition(
        fitted,
        data.x,
        'knnDistance',
        params.threshold,
        { name: 'neighbourDistance', values: s.scores },
        s.scores,
      )
    },
    logs: { 'solver/stage': Log(), 'solver/maxScore': Log() },
  })
}

/**
 * The k-nearest-neighbour distance detector as a scikit-learn-style class wrapper.
 *
 * @example
 * ```ts
 * import { KnnDetector } from 'omai-engine/anomaly'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [10, 10]]
 * toFlat(new KnnDetector({ nNeighbors: 2, contamination: 0.2 }).fit(x).predict([[0.5, 0.4], [10, 10.5]])) // => [1, -1]
 * ```
 */
export class KnnDetector extends ExplainedEstimatorClass<
  KnnDetectorHyper,
  Fitted<KnnDetectorParams> & NeighbourDetectorVerbs
> {
  /**
   * Make an unfitted k-nearest-neighbour distance detector.
   *
   * @param hyper - The hyperparameters of {@link knnDetector}.
   */
  constructor(hyper: Partial<KnnDetectorHyper> = {}) {
    super(knnDetector(hyper))
  }
}
