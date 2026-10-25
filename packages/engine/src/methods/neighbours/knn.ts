/**
 * Nearest-neighbour estimators, as in scikit-learn's `neighbors` module: k-nearest-neighbour classification and
 * regression (a vote or mean of the k closest training rows, optionally weighted by inverse distance) and their
 * radius-neighbour forms (every training row within a radius), by brute-force search with the core's metrics.
 */

import { argmaxOf, WhiteboxError } from '../../lib/base/index.ts'
import { categorical, type Distribution, normal } from '../../lib/dists/index.ts'
import { type Exposition, type NeighbourRecord, neighbourRecord } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  encodeLabels,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
  targetVector,
} from '../../lib/learn/index.ts'
import type { Metric } from '../../lib/numerics/geometry/index.ts'
import { kNearest, radiusNeighbours } from '../../lib/numerics/spatial/index.ts'
import { choice, dict, int, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'

// TODO: kernel-density classifiers, metric learning (NCA, LMNN), tree-backed queries (numerics/spatial has kd and
// ball trees).

/** Hyperparameters of the neighbour estimators. */
export interface NeighboursHyper {
  readonly nNeighbors: number
  /** Radius of the radius estimators. */
  readonly radius: number
  /** Uniform votes or inverse-distance weights. */
  readonly weights: 'uniform' | 'distance'
  readonly metric: Metric
  /** Minkowski order. */
  readonly p: number
  /** Training rows whose neighbours the exposition records (default 50; 0 for none). */
  readonly explainRows: number
}

/** Parameters of a fitted neighbour estimator: the training data itself. */
export type NeighboursParams = {
  readonly x: number[][]
  /** Class codes (classification) or targets. */
  readonly y: number[]
  readonly classes: number[]
}

/** Verbs of a fitted neighbour estimator. */
export interface NeighboursVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  /** Categorical neighbour votes, or normal with the neighbours' weighted mean and spread. */
  readonly predictDistribution: (x: TensorLike) => Distribution
  /** The neighbours of each query, as an exposition record. */
  readonly kneighbors: (x: TensorLike) => NeighbourRecord
  readonly classes?: number[]
  readonly nFeaturesIn: number
}

/** A fitted neighbour estimator. */
export type FittedNeighbours = Fitted<NeighboursParams> & NeighboursVerbs

type Found = { readonly indices: number[]; readonly distances: number[] }[]

function search(p: NeighboursParams, h: NeighboursHyper, radius: boolean, x: TensorLike): Found {
  const metric = { metric: h.metric, p: h.p }
  if (radius) return radiusNeighbours(p.x, x, h.radius, metric)
  const k = Math.min(h.nNeighbors, p.x.length)
  const r = kNearest(p.x, x, k, metric)
  const idx = toFlat(r.indices)
  const dist = toFlat(primalOf(r.distances))
  const q = idx.length / k
  return Array.from({ length: q }, (_, i) => ({
    indices: idx.slice(i * k, (i + 1) * k),
    distances: dist.slice(i * k, (i + 1) * k),
  }))
}

function weightsOf(distances: readonly number[], h: NeighboursHyper): number[] {
  if (h.weights === 'uniform') return distances.map(() => 1)
  // scikit-learn: a zero distance takes all the weight.
  if (distances.some((d) => d === 0)) return distances.map((d) => (d === 0 ? 1 : 0))
  return distances.map((d) => 1 / d)
}

function neighboursVerbs(
  p: NeighboursParams,
  h: NeighboursHyper,
  radius: boolean,
  classifier: boolean,
): NeighboursVerbs {
  const d = p.x[0]?.length ?? 0
  const query = (x: TensorLike): Found => {
    matrixView(x, 'neighbours', { nFeatures: d })
    return search(p, h, radius, x)
  }
  const proba = (found: Found): Float64Array => {
    const k = p.classes.length
    const out = new Float64Array(found.length * k)
    found.forEach((f, i) => {
      const w = weightsOf(f.distances, h)
      let total = 0
      f.indices.forEach((j, a) => {
        out[i * k + (p.y[j] ?? 0)] = (out[i * k + (p.y[j] ?? 0)] ?? 0) + (w[a] ?? 0)
        total += w[a] ?? 0
      })
      for (let c = 0; c < k; c++) out[i * k + c] = total > 0 ? (out[i * k + c] ?? 0) / total : 1 / k
    })
    return out
  }
  const moments = (found: Found): { mean: Float64Array; sd: Float64Array } => {
    const mean = new Float64Array(found.length)
    const sd = new Float64Array(found.length)
    found.forEach((f, i) => {
      const w = weightsOf(f.distances, h)
      const total = w.reduce((a, b) => a + b, 0)
      const m = total > 0 ? f.indices.reduce((a, j, t) => a + (w[t] ?? 0) * (p.y[j] ?? 0), 0) / total : Number.NaN
      const v =
        total > 0 ? f.indices.reduce((a, j, t) => a + (w[t] ?? 0) * ((p.y[j] ?? 0) - m) ** 2, 0) / total : Number.NaN
      mean[i] = m
      sd[i] = Math.sqrt(Math.max(v, 1e-12))
    })
    return { mean, sd }
  }
  const kneighbors = (x: TensorLike): NeighbourRecord => neighbourRecord({ ...packed(query(x)), metric: h.metric })
  if (classifier) {
    const k = p.classes.length
    const probaT = (x: TensorLike): Tensor => {
      const f = query(x)
      return fromStorage(proba(f), [f.length, k], 'float64')
    }
    return {
      predict: (x) => {
        const pr = proba(query(x))
        const m = pr.length / k
        return fromStorage(
          Float64Array.from({ length: m }, (_, i) => p.classes[argmaxOf(pr.subarray(i * k, (i + 1) * k))] ?? 0),
          [m],
          'float64',
        )
      },
      predictProba: probaT,
      predictDistribution: (x) => categorical({ probs: probaT(x) }),
      kneighbors,
      classes: p.classes,
      nFeaturesIn: d,
    }
  }
  return {
    predict: (x) => {
      const { mean } = moments(query(x))
      return fromStorage(mean, [mean.length], 'float64')
    },
    predictDistribution: (x) => {
      const { mean, sd } = moments(query(x))
      return normal(fromStorage(mean, [mean.length], 'float64'), fromStorage(sd, [sd.length], 'float64'))
    },
    kneighbors,
    nFeaturesIn: d,
  }
}

const packed = (found: Found): { indices: number[][]; distances: number[][] } => ({
  indices: found.map((f) => f.indices),
  distances: found.map((f) => f.distances),
})

function neighboursEstimator(
  name: string,
  classifier: boolean,
  radius: boolean,
  hyper: NeighboursHyper,
): ExplainedEstimator<NeighboursHyper, FittedNeighbours> {
  const est: ExplainedEstimator<NeighboursHyper, FittedNeighbours> = defineExplainedEstimator(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      hyperSpace: (h: NeighboursHyper): Space =>
        dict({
          ...(radius
            ? { radius: real(0.01, 10, { log: true, default: h.radius, label: 'radius' }) }
            : { nNeighbors: int(1, 50, { default: h.nNeighbors, label: 'neighbours' }) }),
          weights: choice(['uniform', 'distance'], {
            label: 'neighbour weights',
            description: 'Weigh neighbours equally or by inverse distance.',
            default: h.weights,
          }),
          metric: choice(['euclidean', 'cityblock', 'chebyshev'], {
            label: 'distance',
            description: 'The distance between rows.',
            default: h.metric,
          }),
        }),
      fit: (data: LearnData, h: NeighboursHyper) => {
        const m = matrixView(data.x, name)
        if (!radius && !(h.nNeighbors >= 1)) throw new WhiteboxError('value', `${name}: nNeighbors must be at least 1`)
        const x = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const labels = classifier ? encodeLabels(data.y ?? [], name, m.n) : undefined
        const params: NeighboursParams = {
          x,
          y: labels === undefined ? Array.from(targetVector(data.y ?? [], name, m.n)) : Array.from(labels.codes),
          classes: labels === undefined ? [] : [...labels.classes],
        }
        const fitted = est.revive(params)
        const exposition: Exposition[] = []
        if (h.explainRows > 0 && (radius || h.nNeighbors < m.n)) {
          const rows = x.slice(0, h.explainRows)
          const found = radius
            ? radiusNeighbours(x, rows, h.radius, { metric: h.metric, p: h.p, excludeSelf: true })
            : search({ ...params, x }, { ...h, nNeighbors: h.nNeighbors + 1 }, false, rows).map((f, i) => {
                // Leave each training row out of its own neighbours.
                const keep = f.indices
                  .map((j, a) => [j, a] as const)
                  .filter(([j]) => j !== i)
                  .slice(0, h.nNeighbors)
                return { indices: keep.map(([j]) => j), distances: keep.map(([, a]) => f.distances[a] ?? 0) }
              })
          exposition.push(neighbourRecord({ ...packed(found), metric: h.metric }))
        }
        if (m.d === 2)
          exposition.push(
            decisionSurfaceOf(fitted, data.x, {
              featureNames: featureNamesOf(data, m.d),
              ...(classifier ? { classNames: classNamesOf(data, params.classes) } : {}),
            }),
          )
        return { params, exposition }
      },
      methods: (p: NeighboursParams, h: NeighboursHyper) => neighboursVerbs(p, h, radius, classifier),
    },
    hyper,
  )
  return est
}

const defaults = (hyper: Partial<NeighboursHyper>): NeighboursHyper => ({
  nNeighbors: hyper.nNeighbors ?? 5,
  radius: hyper.radius ?? 1,
  weights: hyper.weights ?? 'uniform',
  metric: hyper.metric ?? 'euclidean',
  p: hyper.p ?? 2,
  explainRows: hyper.explainRows ?? 50,
})

/**
 * k-nearest-neighbour classification (sklearn.neighbors.KNeighborsClassifier): the (inverse-distance) weighted vote
 * of the k nearest training rows, by brute-force search with the core's metrics. Exposition: the neighbours of the
 * first training rows (each left out of its own list) and, for two features, the decision surface; `kneighbors`
 * records any query's neighbours.
 *
 * @param hyper - Neighbours (default 5), weights, metric and Minkowski order, and how many rows to explain.
 * @returns The estimator.
 */
export function kNeighborsClassifier(
  hyper: Partial<NeighboursHyper> = {},
): ExplainedEstimator<NeighboursHyper, FittedNeighbours> {
  return neighboursEstimator('k-neighbors-classifier', true, false, defaults(hyper))
}

/**
 * k-nearest-neighbour regression (sklearn.neighbors.KNeighborsRegressor): the weighted mean of the neighbours'
 * targets; `predictDistribution` is normal with their weighted mean and spread. Exposition as
 * {@link kNeighborsClassifier}.
 *
 * @param hyper - As {@link kNeighborsClassifier}.
 * @returns The estimator.
 */
export function kNeighborsRegressor(
  hyper: Partial<NeighboursHyper> = {},
): ExplainedEstimator<NeighboursHyper, FittedNeighbours> {
  return neighboursEstimator('k-neighbors-regressor', false, false, defaults(hyper))
}

/**
 * Radius-neighbour classification (sklearn.neighbors.RadiusNeighborsClassifier): the vote of every training row
 * within the radius (uniform probabilities when there is none).
 *
 * @param hyper - Radius (default 1), weights, metric and rows to explain.
 * @returns The estimator.
 */
export function radiusNeighborsClassifier(
  hyper: Partial<NeighboursHyper> = {},
): ExplainedEstimator<NeighboursHyper, FittedNeighbours> {
  return neighboursEstimator('radius-neighbors-classifier', true, true, defaults(hyper))
}

/**
 * Radius-neighbour regression (sklearn.neighbors.RadiusNeighborsRegressor): the mean target within the radius (NaN
 * when there is none).
 *
 * @param hyper - Radius (default 1), weights, metric and rows to explain.
 * @returns The estimator.
 */
export function radiusNeighborsRegressor(
  hyper: Partial<NeighboursHyper> = {},
): ExplainedEstimator<NeighboursHyper, FittedNeighbours> {
  return neighboursEstimator('radius-neighbors-regressor', false, true, defaults(hyper))
}

/** {@link kNeighborsClassifier} as a class. */
export class KNeighborsClassifier extends ExplainedEstimatorClass<NeighboursHyper, FittedNeighbours> {
  /** @param hyper - As {@link kNeighborsClassifier}. */
  constructor(hyper: Partial<NeighboursHyper> = {}) {
    super(kNeighborsClassifier(hyper))
  }
}

/** {@link kNeighborsRegressor} as a class. */
export class KNeighborsRegressor extends ExplainedEstimatorClass<NeighboursHyper, FittedNeighbours> {
  /** @param hyper - As {@link kNeighborsRegressor}. */
  constructor(hyper: Partial<NeighboursHyper> = {}) {
    super(kNeighborsRegressor(hyper))
  }
}
