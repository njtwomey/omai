/**
 * The nearest-centroid classifier (as in scikit-learn's `NearestCentroid`): each class is represented by the mean of
 * its rows and a point takes the class of the nearest one, optionally with Tibshirani's shrunken centroids, which pull
 * class means towards the overall mean and drop uninformative features.
 */

import { argminOf, quantileOf } from '../../lib/base/index.ts'
import { categorical, type Distribution } from '../../lib/dists/index.ts'
import { coefficientTable, type Exposition } from '../../lib/exposition/index.ts'
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
} from '../../lib/learn/index.ts'
import { dict, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../../lib/tensor/index.ts'

/** Hyperparameters of {@link nearestCentroid}. */
export interface NearestCentroidHyper {
  /** Euclidean distances to class means, or Manhattan distances to class medians. */
  readonly metric: 'euclidean' | 'manhattan'
  /** Shrink centroids towards the overall centroid by this many pooled standard errors (default none). */
  readonly shrinkThreshold?: number
}

/** Parameters of a fitted nearest-centroid classifier. */
export type NearestCentroidParams = { readonly centroids: number[][]; readonly classes: number[] }

/** Verbs of a fitted nearest-centroid classifier. */
export interface NearestCentroidVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** A soft assignment, softmax(-d^2 / 2) over the classes' centroid distances. */
  readonly predictDistribution: (x: TensorLike) => Distribution
  readonly classes: number[]
  readonly nFeaturesIn: number
}

/** A fitted nearest-centroid classifier. */
export type FittedNearestCentroid = Fitted<NearestCentroidParams> & NearestCentroidVerbs

function distances(p: NearestCentroidParams, h: NearestCentroidHyper, x: TensorLike): { m: number; d: Float64Array } {
  const dim = p.centroids[0]?.length ?? 0
  const v = matrixView(x, 'nearestCentroid', { nFeatures: dim })
  const k = p.centroids.length
  const d = new Float64Array(v.n * k)
  for (let i = 0; i < v.n; i++)
    for (const [c, centre] of p.centroids.entries()) {
      let s = 0
      for (let j = 0; j < dim; j++) {
        const diff = (v.data[i * dim + j] ?? 0) - (centre[j] ?? 0)
        s += h.metric === 'manhattan' ? Math.abs(diff) : diff * diff
      }
      d[i * k + c] = h.metric === 'manhattan' ? s : Math.sqrt(s)
    }
  return { m: v.n, d }
}

/**
 * The nearest-centroid classifier (sklearn.neighbors.NearestCentroid), with optional shrunken centroids (Tibshirani's
 * nearest shrunken centroids). Exposition: the centroids as a table (`class:feature`) and, for two features, the
 * decision surface.
 *
 * @param hyper - Metric (default euclidean) and shrink threshold.
 * @returns The estimator.
 */
export function nearestCentroid(
  hyper: Partial<NearestCentroidHyper> = {},
): ExplainedEstimator<NearestCentroidHyper, FittedNearestCentroid> {
  const est: ExplainedEstimator<NearestCentroidHyper, FittedNearestCentroid> = defineExplainedEstimator(
    {
      name: 'nearest-centroid',
      kind: 'classifier',
      hyperSpace: (h: NearestCentroidHyper): Space =>
        dict({ shrinkThreshold: real(0, 5, { default: h.shrinkThreshold ?? 0, label: 'shrinkage' }) }),
      fit: (data: LearnData, h: NearestCentroidHyper) => {
        const v = matrixView(data.x, 'nearestCentroid')
        const { classes, codes } = encodeLabels(data.y ?? [], 'nearestCentroid', v.n)
        const k = classes.length
        const rowsOf = (c: number): number[] => Array.from(codes.keys()).filter((i) => codes[i] === c)
        let centroids = classes.map((_, c) => {
          const rows = rowsOf(c)
          return Array.from({ length: v.d }, (_, j) => {
            const col = rows.map((i) => v.data[i * v.d + j] ?? 0)
            return h.metric === 'manhattan' ? quantileOf(col, 0.5) : col.reduce((a, b) => a + b, 0) / col.length
          })
        })
        if (h.shrinkThreshold !== undefined && h.shrinkThreshold > 0) {
          const overall = Array.from({ length: v.d }, (_, j) => {
            let s = 0
            for (let i = 0; i < v.n; i++) s += v.data[i * v.d + j] ?? 0
            return s / v.n
          })
          const variance = Array.from({ length: v.d }, (_, j) => {
            let s = 0
            for (let i = 0; i < v.n; i++) s += ((v.data[i * v.d + j] ?? 0) - (centroids[codes[i] ?? 0]?.[j] ?? 0)) ** 2
            return s
          })
          const sd = variance.map((s) => Math.sqrt(s / (v.n - k)))
          const med = quantileOf(sd, 0.5)
          const s = sd.map((x) => x + med)
          const t = h.shrinkThreshold
          centroids = centroids.map((centre, c) => {
            const m = Math.sqrt(1 / rowsOf(c).length - 1 / v.n)
            return centre.map((value, j) => {
              const ms = m * (s[j] ?? 1)
              const dev = (value - (overall[j] ?? 0)) / ms
              const shrunk = Math.sign(dev) * Math.max(Math.abs(dev) - t, 0)
              return (overall[j] ?? 0) + ms * shrunk
            })
          })
        }
        const params: NearestCentroidParams = { centroids, classes: [...classes] }
        const features = featureNamesOf(data, v.d)
        const names = classNamesOf(data, classes)
        const exposition: Exposition[] = [
          coefficientTable({
            names: names.flatMap((c) => features.map((f) => `${c}:${f}`)),
            estimates: centroids.flat(),
          }),
        ]
        if (v.d === 2)
          exposition.push(decisionSurfaceOf(est.revive(params), data.x, { featureNames: features, classNames: names }))
        return { params, exposition }
      },
      methods: (p: NearestCentroidParams, h: NearestCentroidHyper) => ({
        predict: (x: TensorLike) => {
          const { m, d } = distances(p, h, x)
          const k = p.classes.length
          return fromStorage(
            Float64Array.from({ length: m }, (_, i) => p.classes[argminOf(d.subarray(i * k, (i + 1) * k))] ?? 0),
            [m],
            'float64',
          )
        },
        predictDistribution: (x: TensorLike) => {
          const { m, d } = distances(p, h, x)
          const k = p.classes.length
          return categorical({
            logits: fromStorage(
              d.map((v) => -(v * v) / 2),
              [m, k],
              'float64',
            ),
          })
        },
        classes: p.classes,
        nFeaturesIn: p.centroids[0]?.length ?? 0,
      }),
    },
    {
      metric: hyper.metric ?? 'euclidean',
      ...(hyper.shrinkThreshold === undefined ? {} : { shrinkThreshold: hyper.shrinkThreshold }),
    },
  )
  return est
}

/** {@link nearestCentroid} as a class. */
export class NearestCentroid extends ExplainedEstimatorClass<NearestCentroidHyper, FittedNearestCentroid> {
  /** @param hyper - As {@link nearestCentroid}. */
  constructor(hyper: Partial<NearestCentroidHyper> = {}) {
    super(nearestCentroid(hyper))
  }
}
