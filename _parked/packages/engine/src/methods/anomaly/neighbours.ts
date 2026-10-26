/**
 * Neighbour-based outlier detectors: the local outlier factor (LOF, as in scikit-learn), which compares a point's local
 * density with its neighbours' densities and can score new rows, and the k-nearest-neighbour distance detector, which
 * scores a point by its distance to its k-th neighbour.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { neighbourRecord } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  matrixView,
} from '../../lib/learn/index.ts'
import { kNearest } from '../../lib/numerics/spatial/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { contaminationThreshold, type OutlierVerbs, outlierExposition, outlierVerbs } from './combine.ts'

/** Neighbours of queries among training rows: indices and distances, nearest first. */
interface Near {
  readonly indices: number[][]
  readonly distances: number[][]
}

function near(train: TensorLike, queries: TensorLike | undefined, k: number): Near {
  const nb = kNearest(train, queries, k, { excludeSelf: queries === undefined })
  const idx = Array.from(nb.indices.data)
  const dist = toFlat(nb.distances)
  const q = idx.length / k
  return {
    indices: Array.from({ length: q }, (_, i) => idx.slice(i * k, (i + 1) * k)),
    distances: Array.from({ length: q }, (_, i) => dist.slice(i * k, (i + 1) * k)),
  }
}

/**
 * Local reachability densities and local outlier factors (Breunig et al. 2000; sklearn's LocalOutlierFactor):
 * reach_k(a, b) = max(k-distance(b), d(a, b)), lrd(a) = 1 / (mean reach + 1e-10), LOF(a) = mean lrd(neighbours) /
 * lrd(a).
 *
 * @param x - Training points `[n, d]`.
 * @param k - Neighbours.
 * @returns Per row: the k-distance, the local reachability density and the LOF; and the neighbours.
 */
export function localOutlierFactors(
  x: TensorLike,
  k: number,
): { readonly kDistance: number[]; readonly lrd: number[]; readonly lof: number[]; readonly neighbours: Near } {
  const n = matrixView(x, 'localOutlierFactor').n
  if (!(Number.isInteger(k) && k >= 1 && k < n)) throw new MdmlError('value', `lof: nNeighbors must be in 1..${n - 1}`)
  const nb = near(x, undefined, k)
  const kDistance = nb.distances.map((d) => d[k - 1] ?? 0)
  const lrdOf = (nbrs: readonly number[], dists: readonly number[]): number =>
    1 / (nbrs.reduce((a, j, t) => a + Math.max(kDistance[j] ?? 0, dists[t] ?? 0), 0) / k + 1e-10)
  const lrd = nb.indices.map((nbrs, i) => lrdOf(nbrs, nb.distances[i] ?? []))
  const lof = nb.indices.map((nbrs, i) => nbrs.reduce((a, j) => a + (lrd[j] ?? 0), 0) / k / (lrd[i] ?? 1))
  return { kDistance, lrd, lof, neighbours: nb }
}

/** Hyperparameters of the local outlier factor. */
export type LofHyper = {
  readonly nNeighbors?: number
  /** The expected fraction of outliers, or 'auto' (LOF above 1.5, sklearn's offset). */
  readonly contamination?: number | 'auto'
}

/** The parameters of a fitted LOF: the training rows and their densities. */
export type LofParams = {
  readonly data: number[][]
  readonly kDistance: number[]
  readonly lrd: number[]
  /** The training LOFs (sklearn's -negative_outlier_factor_). */
  readonly lof: number[]
  readonly k: number
  readonly threshold: number
}

/** A fitted LOF. */
export type FittedLof = ExplainedFitted<LofParams, OutlierVerbs>

/**
 * The local outlier factor (sklearn.neighbors.LocalOutlierFactor, with novelty scoring of new rows): anomaly scores
 * are the LOFs. Its exposition is an `anomalyScores` record with the local reachability densities as evidence, the
 * neighbours of every training row, and for one or two features a `decisionSurface` of the score.
 *
 * @param hyper - Neighbours and contamination.
 * @returns The estimator.
 */
export function localOutlierFactor(hyper: LofHyper = {}): Estimator<LofHyper, FittedLof> {
  const verbs = (p: LofParams): OutlierVerbs => {
    const d = p.data[0]?.length ?? 0
    return outlierVerbs(
      (x) => {
        matrixView(x, 'localOutlierFactor', { nFeatures: d })
        const nb = near(p.data, x, p.k)
        return nb.indices.map((nbrs, i) => {
          const dists = nb.distances[i] ?? []
          const lrd = 1 / (nbrs.reduce((a, j, t) => a + Math.max(p.kDistance[j] ?? 0, dists[t] ?? 0), 0) / p.k + 1e-10)
          return nbrs.reduce((a, j) => a + (p.lrd[j] ?? 0), 0) / p.k / lrd
        })
      },
      p.threshold,
      d,
    )
  }
  return defineExplainedEstimator<LofHyper, LofParams, OutlierVerbs>(
    {
      name: 'local-outlier-factor',
      kind: 'outlier',
      hyperSpace: (h) =>
        dict({
          nNeighbors: int(1, 100, { default: h.nNeighbors ?? 20, label: 'neighbours' }),
          contamination: real(0.01, 0.5, {
            label: 'contamination',
            description: 'The expected fraction of outliers: sets the threshold on the scores.',
            default: typeof h.contamination === 'number' ? h.contamination : 0.1,
          }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'localOutlierFactor', { minSamples: 2 })
        const k = Math.min(h.nNeighbors ?? 20, m.n - 1)
        const r = localOutlierFactors(data.x, k)
        const c = h.contamination ?? 'auto'
        const threshold = c === 'auto' ? 1.5 : contaminationThreshold(r.lof, c)
        const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const params: LofParams = { data: rows, kDistance: r.kDistance, lrd: r.lrd, lof: r.lof, k, threshold }
        const fitted = { estimator: 'local-outlier-factor', kind: 'outlier' as const, params, ...verbs(params) }
        // The training scores are the LOFs themselves (each row left out of its own neighbourhood).
        const exposition = outlierExposition(
          fitted,
          data.x,
          'localOutlierFactor',
          threshold,
          { name: 'localReachabilityDensity', values: r.lrd },
          r.lof,
        )
        return {
          params,
          exposition: [
            ...exposition,
            neighbourRecord({ indices: r.neighbours.indices, distances: r.neighbours.distances, metric: 'euclidean' }),
          ],
        }
      },
      methods: verbs,
    },
    hyper,
  )
}

/** The local outlier factor as a class. */
export class LocalOutlierFactor extends EstimatorClass<LofHyper, FittedLof> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link localOutlierFactor}.
   */
  constructor(hyper: LofHyper = {}) {
    super(localOutlierFactor(hyper))
  }
}

/** Hyperparameters of the k-nearest-neighbour distance detector. */
export type KnnDetectorHyper = {
  readonly nNeighbors?: number
  /** The k-th distance (default), the mean or the median of the k distances. */
  readonly method?: 'largest' | 'mean' | 'median'
  readonly contamination?: number
}

/** The parameters of a fitted k-NN detector. */
export type KnnDetectorParams = { readonly data: number[][]; readonly k: number; readonly threshold: number }

/**
 * The k-nearest-neighbour distance detector (Ramaswamy et al. 2000; PyOD's KNN): a row's score is the distance to its
 * k-th nearest training row (or the mean or median of its k distances). Its exposition is an `anomalyScores` record
 * and, for one or two features, a `decisionSurface`.
 *
 * @param hyper - Neighbours, the aggregation and contamination (default 0.1).
 * @returns The estimator.
 */
export function knnDetector(
  hyper: KnnDetectorHyper = {},
): Estimator<KnnDetectorHyper, ExplainedFitted<KnnDetectorParams, OutlierVerbs>> {
  const aggregate = (d: readonly number[], how: KnnDetectorHyper['method']): number => {
    if (how === 'mean') return d.reduce((a, v) => a + v, 0) / d.length
    if (how === 'median') {
      const s = [...d].sort((a, b) => a - b)
      return s.length % 2 === 1
        ? (s[(s.length - 1) / 2] ?? 0)
        : ((s[s.length / 2 - 1] ?? 0) + (s[s.length / 2] ?? 0)) / 2
    }
    return d.at(-1) ?? 0
  }
  const scorer = (p: KnnDetectorParams, how: KnnDetectorHyper['method']) => (x: TensorLike) =>
    near(p.data, x, p.k).distances.map((d) => aggregate(d, how))
  return defineExplainedEstimator<KnnDetectorHyper, KnnDetectorParams, OutlierVerbs>(
    {
      name: 'knn-detector',
      kind: 'outlier',
      hyperSpace: (h) =>
        dict({
          nNeighbors: int(1, 100, { default: h.nNeighbors ?? 5, label: 'neighbours' }),
          method: choice(['largest', 'mean', 'median'], {
            label: 'score',
            description: 'The outlier score: distance to the k-th neighbour, or the mean or median distance to the k.',
            default: h.method ?? 'largest',
          }),
          contamination: real(0.01, 0.5, {
            label: 'contamination',
            description: 'The expected fraction of outliers: sets the threshold on the scores.',
            default: h.contamination ?? 0.1,
          }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'knnDetector', { minSamples: 2 })
        const k = Math.min(h.nNeighbors ?? 5, m.n - 1)
        const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const train = near(rows, undefined, k).distances.map((d) => aggregate(d, h.method))
        const threshold = contaminationThreshold(train, h.contamination ?? 0.1)
        const params = { data: rows, k, threshold }
        const fitted = {
          estimator: 'knn-detector',
          kind: 'outlier' as const,
          params,
          ...outlierVerbs(scorer(params, h.method), threshold, m.d),
        }
        return {
          params,
          exposition: outlierExposition(
            fitted,
            data.x,
            'knnDistance',
            threshold,
            { name: 'kDistance', values: train },
            train,
          ),
        }
      },
      methods: (p, h) => outlierVerbs(scorer(p, h.method), p.threshold, p.data[0]?.length ?? 0),
    },
    hyper,
  )
}
