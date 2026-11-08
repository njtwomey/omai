/**
 * Threshold curves of binary scorers, scikit-learn style: confusion counts at every threshold, the ROC curve (true
 * against false positive rate) and its area `rocAucScore` (binary, partial and multi-class), the trapezoidal `auc`, and
 * the detection error tradeoff (DET) curve.
 */

import { type MetricResult, OmaiError } from '../base/index.ts'
import { fromVector, type Tensor, type TensorLike, toVector } from '../tensor/index.ts'
import { rocRows, thresholdRows } from './_curves.ts'
import { isMultilabel, type Label, labelValues, sortLabels } from './_labels.ts'
import { averageBinary, binarise, column, positives, scoreSpec } from './_scores.ts'
import {
  type Metric,
  type MetricInput,
  type MetricOptions,
  metricResult,
  sampleWeights,
  shapedArray,
} from './metric.ts'

/** Options of the binary score curves. */
export interface CurveOptions extends MetricOptions {
  /** The positive class (default 1, or the larger of two labels). */
  readonly posLabel?: Label
}

/**
 * Weighted counts of a binary scorer at each distinct threshold, from the highest score down. The vectors are tensors
 * of shape `[T]` (migration, D-009: they were `readonly number[]`).
 */
export interface ThresholdCounts {
  readonly thresholds: Tensor<'float64'>
  readonly tps: Tensor<'float64'>
  readonly fps: Tensor<'float64'>
  readonly tns: Tensor<'float64'>
  readonly fns: Tensor<'float64'>
}

/**
 * The confusion counts of a binary scorer at every distinct score threshold, from the highest score down
 * (sklearn.metrics.confusion_matrix_at_thresholds): the building block of ROC, precision-recall and DET curves.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]` (higher means more positive).
 * @param options - posLabel and sample weights.
 * @returns The thresholds (decreasing) with true and false positives and negatives at each, as `[T]` tensors (was
 *   `number[]`).
 * @example
 * ```ts
 * import { confusionMatrixAtThresholds } from 'omai-engine/metrics'
 * confusionMatrixAtThresholds([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).tps // => [1, 1, 2, 2]
 * ```
 */
export function confusionMatrixAtThresholds(
  yTrue: MetricInput,
  yScore: TensorLike,
  options: CurveOptions = {},
): ThresholdCounts {
  const c = thresholdRows(yTrue, yScore, options)
  return {
    thresholds: fromVector(c.thresholds),
    tps: fromVector(c.tps),
    fps: fromVector(c.fps),
    tns: fromVector(c.tns),
    fns: fromVector(c.fns),
  }
}

/**
 * A ROC curve: false and true positive rates at decreasing thresholds (the first is +inf), as `[T]` tensors
 * (migration, D-009: they were `readonly number[]`).
 */
export interface RocCurve {
  readonly fpr: Tensor<'float64'>
  readonly tpr: Tensor<'float64'>
  readonly thresholds: Tensor<'float64'>
}

/**
 * The receiver operating characteristic curve (sklearn.metrics.roc_curve): FPR and TPR at each threshold, starting from
 * (0, 0) at +inf, with collinear intermediate points dropped by default.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - posLabel, dropIntermediate (default true) and sample weights.
 * @returns The curve, `[T]` tensors (was `number[]`).
 * @concept roc-curve
 * @example
 * ```ts
 * import { rocCurve } from 'omai-engine/metrics'
 * rocCurve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).tpr // => [0, 0.5, 0.5, 1, 1]
 * ```
 */
export function rocCurve(
  yTrue: MetricInput,
  yScore: TensorLike,
  options: CurveOptions & { readonly dropIntermediate?: boolean } = {},
): RocCurve {
  const r = rocRows(yTrue, yScore, options)
  return { fpr: fromVector(r.fpr), tpr: fromVector(r.tpr), thresholds: fromVector(r.thresholds) }
}

/**
 * The area under a curve by the trapezoidal rule (sklearn.metrics.auc); x must be monotonic (either direction). Takes
 * tensors (migration, D-009: `x` and `y` were `readonly number[]`; nested arrays still work).
 *
 * @param x - Abscissae, monotonic: a vector (a tensor, a typed array or a flat array).
 * @param y - Ordinates: a vector.
 * @returns The area.
 * @throws OmaiError with code `value` when x is not monotonic, and `shape` unless both are vectors.
 * @example
 * ```ts
 * import { auc } from 'omai-engine/metrics'
 * auc([0, 0.5, 1], [0, 1, 1]) // => 0.75
 * ```
 */
export function auc(x: TensorLike, y: TensorLike): number {
  const xs = toVector(x, 'auc')
  const ys = toVector(y, 'auc')
  let direction = 1
  let increasing = true
  let decreasing = true
  for (let i = 1; i < xs.length; i++) {
    if ((xs[i] ?? 0) < (xs[i - 1] ?? 0)) increasing = false
    if ((xs[i] ?? 0) > (xs[i - 1] ?? 0)) decreasing = false
  }
  if (!increasing) {
    if (!decreasing) throw new OmaiError('value', 'auc: x is neither increasing nor decreasing')
    direction = -1
  }
  let area = 0
  for (let i = 1; i < xs.length; i++)
    area += (((xs[i] ?? 0) - (xs[i - 1] ?? 0)) * ((ys[i] ?? 0) + (ys[i - 1] ?? 0))) / 2
  return direction * area
}

/** The binary ROC AUC, with the McClish-corrected partial AUC up to maxFpr. */
function binaryRocAuc(y: Float64Array, s: Float64Array, w: Float64Array | undefined, maxFpr?: number): number {
  let pos = 0
  for (const v of y) pos += v
  if (pos === 0 || pos === y.length) return Number.NaN
  const r = rocRows(Array.from(y), s, w === undefined ? {} : { sampleWeight: w })
  if (maxFpr === undefined || maxFpr === 1) return auc(r.fpr, r.tpr)
  if (!(maxFpr > 0 && maxFpr <= 1)) throw new OmaiError('value', `maxFpr must be in (0, 1], got ${maxFpr}`)
  let stop = 0
  while (stop < r.fpr.length && (r.fpr[stop] ?? 0) <= maxFpr) stop++
  const x0 = r.fpr[stop - 1] ?? 0
  const x1 = r.fpr[stop] ?? x0
  const y0 = r.tpr[stop - 1] ?? 0
  const y1 = r.tpr[stop] ?? y0
  const yAt = x1 === x0 ? y1 : y0 + ((maxFpr - x0) * (y1 - y0)) / (x1 - x0)
  const partial = auc([...r.fpr.slice(0, stop), maxFpr], [...r.tpr.slice(0, stop), yAt])
  const minArea = 0.5 * maxFpr * maxFpr
  return 0.5 * (1 + (partial - minArea) / (maxFpr - minArea))
}

/** Options of {@link rocAucScore}. */
export interface RocAucOptions extends MetricOptions {
  /** Averaging over classes or labels: 'macro' (default), 'weighted' (by prevalence) or 'micro' (multilabel). */
  readonly average?: 'macro' | 'weighted' | 'micro'
  /** Multi-class strategy: one-vs-rest ('ovr') or Hand and Till's one-vs-one ('ovo'). */
  readonly multiClass?: 'ovr' | 'ovo'
  /** Standardised partial AUC (McClish) up to this false positive rate (binary). */
  readonly maxFpr?: number
  /** Class of each score column (default the sorted labels). */
  readonly labels?: readonly Label[]
}

/**
 * Area under the ROC curve (sklearn.metrics.roc_auc_score): binary (with the McClish partial AUC up to maxFpr),
 * multi-class one-vs-rest or one-vs-one (Hand and Till), and multilabel, with macro, weighted and micro averaging.
 *
 * @param yTrue - True labels `[N]`, or a multilabel indicator `[N, L]`.
 * @param yScore - Scores `[N]` (binary) or `[N, C]` (class probabilities or label scores).
 * @param options - average, multiClass, maxFpr, labels and sample weights.
 * @returns The metric result (per-class AUCs in the breakdown).
 * @concept roc-curve
 * @index function Area under the ROC curve; rocCurve gives the curve.
 * @example
 * ```ts
 * import { rocAucScore } from 'omai-engine/metrics'
 * rocAucScore([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).estimate.value // => 0.75
 * ```
 */
export const rocAucScore: Metric<RocAucOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yScore: MetricInput, options: RocAucOptions = {}): MetricResult => {
    const name = 'rocAucScore'
    const s = shapedArray(yScore as TensorLike)
    const multilabel = isMultilabel(yTrue)
    const n = s.shape[0] ?? 0
    const w = options.sampleWeight === undefined ? undefined : sampleWeights(name, n, options.sampleWeight)
    const score = (y: Float64Array, sc: Float64Array, ww: Float64Array | undefined): number =>
      binaryRocAuc(y, sc, ww, options.maxFpr)
    if (multilabel) {
      const t = shapedArray(yTrue as TensorLike)
      const r = averageBinary(t, s, w, options.average ?? 'macro', score)
      return metricResult(name, r.value, n, Object.fromEntries(r.perClass.map((v, j) => [String(j), v])))
    }
    const labels = labelValues(yTrue)
    if (s.shape.length === 1) {
      const pos = positives(yTrue, undefined)
      return metricResult(name, binaryRocAuc(pos, s.data, w, options.maxFpr), n)
    }
    const classes = options.labels ?? sortLabels(labels)
    if (classes.length !== (s.shape[1] ?? 0)) throw new OmaiError('shape', `${name}: one score column per class`)
    if (classes.length === 2 && options.multiClass === undefined) {
      return metricResult(
        name,
        binaryRocAuc(
          positives(
            labels.map((l) => (l === classes[1] ? 1 : 0)),
            1,
          ),
          column(s, 1),
          w,
          options.maxFpr,
        ),
        n,
      )
    }
    if (options.maxFpr !== undefined && options.maxFpr !== 1)
      throw new OmaiError('value', `${name}: partial AUC is only defined for binary targets`)
    const multiClass = options.multiClass ?? 'ovr'
    const average = options.average ?? 'macro'
    if (multiClass === 'ovr') {
      const r = averageBinary(binarise(labels, classes), s, w, average, score)
      return metricResult(name, r.value, n, Object.fromEntries(r.perClass.map((v, j) => [String(classes[j]), v])))
    }
    // Hand and Till's M: the mean over class pairs of the two directional AUCs restricted to the pair.
    const pairScores: number[] = []
    const prevalence: number[] = []
    for (let a = 0; a < classes.length; a++) {
      for (let b = a + 1; b < classes.length; b++) {
        const rows = labels.map((l, i) => (l === classes[a] || l === classes[b] ? i : -1)).filter((i) => i >= 0)
        const aTrue = Float64Array.from(rows, (i) => (labels[i] === classes[a] ? 1 : 0))
        const bTrue = Float64Array.from(rows, (i) => (labels[i] === classes[b] ? 1 : 0))
        const aScore = Float64Array.from(rows, (i) => s.data[i * classes.length + a] ?? 0)
        const bScore = Float64Array.from(rows, (i) => s.data[i * classes.length + b] ?? 0)
        pairScores.push((binaryRocAuc(aTrue, aScore, undefined) + binaryRocAuc(bTrue, bScore, undefined)) / 2)
        prevalence.push(rows.length / labels.length)
      }
    }
    const value =
      average === 'weighted'
        ? pairScores.reduce((acc, v, i) => acc + v * (prevalence[i] ?? 0), 0) / prevalence.reduce((x, y) => x + y, 0)
        : pairScores.reduce((x, y) => x + y, 0) / pairScores.length
    return metricResult(name, value, n)
  },
  { spec: scoreSpec('rocAucScore') },
)

/**
 * A detection error tradeoff curve: false positive and false negative rates at increasing thresholds, as tensors
 * (migration, D-009: they were `readonly number[]`).
 */
export interface DetCurve {
  readonly fpr: Tensor<'float64'>
  readonly fnr: Tensor<'float64'>
  readonly thresholds: Tensor<'float64'>
}

/**
 * The detection error tradeoff curve (sklearn.metrics.det_curve): FPR and FNR (miss rate) at each threshold,
 * trimmed to the stretch where both change, usually plotted on normal-deviate axes.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - posLabel, dropIntermediate (default false) and sample weights.
 * @returns The curve (FPR decreasing), `[T]` tensors (was `number[]`).
 * @throws OmaiError with code `value` when only one class is present.
 * @example
 * ```ts
 * import { detCurve } from 'omai-engine/metrics'
 * detCurve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).fnr // => [0, 0.5, 0.5]
 * ```
 */
export function detCurve(
  yTrue: MetricInput,
  yScore: TensorLike,
  options: CurveOptions & { readonly dropIntermediate?: boolean } = {},
): DetCurve {
  const c = thresholdRows(yTrue, yScore, options)
  let tps = [0, ...c.tps]
  let fps = [0, ...c.fps]
  let thresholds = [Number.POSITIVE_INFINITY, ...c.thresholds]
  if (options.dropIntermediate === true && fps.length > 2) {
    const keep = tps
      .map((_, i) => i)
      .filter((i) => i === 0 || i === tps.length - 1 || tps[i] !== tps[i - 1] || tps[i + 1] !== tps[i])
    tps = keep.map((i) => tps[i] ?? 0)
    fps = keep.map((i) => fps[i] ?? 0)
    thresholds = keep.map((i) => thresholds[i] ?? 0)
  }
  const pCount = tps[tps.length - 1] ?? 0
  const nCount = fps[fps.length - 1] ?? 0
  if (pCount === 0 || nCount === 0) throw new OmaiError('value', 'detCurve: only one class is present')
  // From the last threshold with zero false positives to the first with zero false negatives.
  let first = 0
  while (first + 1 < fps.length && fps[first + 1] === fps[0]) first++
  let last = 0
  while (last < tps.length && (tps[last] ?? 0) < pCount) last++
  const idx = Array.from({ length: last + 1 - first }, (_, k) => first + k).reverse()
  return {
    fpr: fromVector(idx.map((i) => (fps[i] ?? 0) / nCount)),
    fnr: fromVector(idx.map((i) => (pCount - (tps[i] ?? 0)) / pCount)),
    thresholds: fromVector(idx.map((i) => thresholds[i] ?? 0)),
  }
}
