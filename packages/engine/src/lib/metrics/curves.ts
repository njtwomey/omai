/**
 * Threshold curves of binary scorers, scikit-learn style: confusion counts at every threshold, the ROC curve (true
 * against false positive rate) and its area `rocAucScore` (binary, partial and multi-class), the trapezoidal `auc`, and
 * the detection error tradeoff (DET) curve.
 */

import { MdmlError, type MetricResult } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { isMultilabel, type Label, labelValues, sortLabels } from './_labels.ts'
import { averageBinary, binarise, column, positives, scoreSpec } from './_scores.ts'
import {
  flat,
  type Metric,
  type MetricInput,
  type MetricOptions,
  matrix,
  metricResult,
  sampleWeights,
} from './metric.ts'

/** Options of the binary score curves. */
export interface CurveOptions extends MetricOptions {
  /** The positive class (default 1, or the larger of two labels). */
  readonly posLabel?: Label
}

/** Weighted counts of a binary scorer at each distinct threshold, from the highest score down. */
export interface ThresholdCounts {
  readonly thresholds: readonly number[]
  readonly tps: readonly number[]
  readonly fps: readonly number[]
  readonly tns: readonly number[]
  readonly fns: readonly number[]
}

/**
 * The confusion counts of a binary scorer at every distinct score threshold, from the highest score down
 * (sklearn.metrics.confusion_matrix_at_thresholds): the building block of ROC, precision-recall and DET curves.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]` (higher means more positive).
 * @param options - posLabel and sample weights.
 * @returns The thresholds (decreasing) with true and false positives and negatives at each.
 * @example
 * ```ts
 * import { confusionMatrixAtThresholds } from 'mdml-engine/metrics'
 * confusionMatrixAtThresholds([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).tps // => [1, 1, 2, 2]
 * ```
 */
export function confusionMatrixAtThresholds(
  yTrue: MetricInput,
  yScore: TensorLike,
  options: CurveOptions = {},
): ThresholdCounts {
  const y = positives(yTrue, options.posLabel)
  const s = flat(yScore)
  if (s.length !== y.length) throw new MdmlError('shape', 'yTrue and yScore have different lengths')
  const w = sampleWeights('confusionMatrixAtThresholds', y.length, options.sampleWeight)
  // A stable ascending sort, reversed (sklearn's mergesort[::-1]): decreasing scores.
  const order = Array.from(s.keys())
    .sort((a, b) => (s[a] ?? 0) - (s[b] ?? 0) || a - b)
    .reverse()
  const thresholds: number[] = []
  const tps: number[] = []
  const fps: number[] = []
  let tp = 0
  let fp = 0
  for (const [k, i] of order.entries()) {
    tp += (y[i] ?? 0) * (w[i] ?? 0)
    fp += (1 - (y[i] ?? 0)) * (w[i] ?? 0)
    const next = order[k + 1]
    if (next === undefined || (s[next] ?? 0) !== (s[i] ?? 0)) {
      thresholds.push(s[i] ?? 0)
      tps.push(tp)
      fps.push(fp)
    }
  }
  const totalTp = tps[tps.length - 1] ?? 0
  const totalFp = fps[fps.length - 1] ?? 0
  return { thresholds, tps, fps, tns: fps.map((v) => totalFp - v), fns: tps.map((v) => totalTp - v) }
}

/** A ROC curve: false and true positive rates at decreasing thresholds (the first is +inf). */
export interface RocCurve {
  readonly fpr: readonly number[]
  readonly tpr: readonly number[]
  readonly thresholds: readonly number[]
}

/**
 * The receiver operating characteristic curve (sklearn.metrics.roc_curve): FPR and TPR at each threshold, starting from
 * (0, 0) at +inf, with collinear intermediate points dropped by default.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - posLabel, dropIntermediate (default true) and sample weights.
 * @returns The curve.
 * @concept roc-curve
 * @example
 * ```ts
 * import { rocCurve } from 'mdml-engine/metrics'
 * rocCurve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).tpr // => [0, 0.5, 0.5, 1, 1]
 * ```
 */
export function rocCurve(
  yTrue: MetricInput,
  yScore: TensorLike,
  options: CurveOptions & { readonly dropIntermediate?: boolean } = {},
): RocCurve {
  const c = confusionMatrixAtThresholds(yTrue, yScore, options)
  let keep = c.thresholds.map((_, i) => i)
  if ((options.dropIntermediate ?? true) && c.fps.length > 2) {
    const second = (a: readonly number[], i: number): number => (a[i + 1] ?? 0) - 2 * (a[i] ?? 0) + (a[i - 1] ?? 0)
    keep = keep.filter((i) => i === 0 || i === c.fps.length - 1 || second(c.fps, i) !== 0 || second(c.tps, i) !== 0)
  }
  const tps = [0, ...keep.map((i) => c.tps[i] ?? 0)]
  const fps = [0, ...keep.map((i) => c.fps[i] ?? 0)]
  const nP = tps[tps.length - 1] ?? 0
  const nN = fps[fps.length - 1] ?? 0
  return {
    fpr: fps.map((v) => (nN > 0 ? v / nN : Number.NaN)),
    tpr: tps.map((v) => (nP > 0 ? v / nP : Number.NaN)),
    thresholds: [Number.POSITIVE_INFINITY, ...keep.map((i) => c.thresholds[i] ?? 0)],
  }
}

/**
 * The area under a curve by the trapezoidal rule (sklearn.metrics.auc); x must be monotonic (either direction).
 *
 * @param x - Abscissae, monotonic.
 * @param y - Ordinates.
 * @returns The area.
 * @throws MdmlError with code `value` when x is not monotonic.
 * @example
 * ```ts
 * import { auc } from 'mdml-engine/metrics'
 * auc([0, 0.5, 1], [0, 1, 1]) // => 0.75
 * ```
 */
export function auc(x: readonly number[], y: readonly number[]): number {
  let direction = 1
  let increasing = true
  let decreasing = true
  for (let i = 1; i < x.length; i++) {
    if ((x[i] ?? 0) < (x[i - 1] ?? 0)) increasing = false
    if ((x[i] ?? 0) > (x[i - 1] ?? 0)) decreasing = false
  }
  if (!increasing) {
    if (!decreasing) throw new MdmlError('value', 'auc: x is neither increasing nor decreasing')
    direction = -1
  }
  let area = 0
  for (let i = 1; i < x.length; i++) area += (((x[i] ?? 0) - (x[i - 1] ?? 0)) * ((y[i] ?? 0) + (y[i - 1] ?? 0))) / 2
  return direction * area
}

/** The binary ROC AUC, with the McClish-corrected partial AUC up to maxFpr. */
function binaryRocAuc(y: Float64Array, s: Float64Array, w: Float64Array | undefined, maxFpr?: number): number {
  let pos = 0
  for (const v of y) pos += v
  if (pos === 0 || pos === y.length) return Number.NaN
  const r = rocCurve(Array.from(y), s, w === undefined ? {} : { sampleWeight: w })
  if (maxFpr === undefined || maxFpr === 1) return auc(r.fpr as number[], r.tpr as number[])
  if (!(maxFpr > 0 && maxFpr <= 1)) throw new MdmlError('value', `maxFpr must be in (0, 1], got ${maxFpr}`)
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
 * @example
 * ```ts
 * import { rocAucScore } from 'mdml-engine/metrics'
 * rocAucScore([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).estimate.value // => 0.75
 * ```
 */
export const rocAucScore: Metric<RocAucOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yScore: MetricInput, options: RocAucOptions = {}): MetricResult => {
    const name = 'rocAucScore'
    const s = matrix(yScore as TensorLike)
    const multilabel = isMultilabel(yTrue)
    const n = s.shape[0] ?? 0
    const w = options.sampleWeight === undefined ? undefined : sampleWeights(name, n, options.sampleWeight)
    const score = (y: Float64Array, sc: Float64Array, ww: Float64Array | undefined): number =>
      binaryRocAuc(y, sc, ww, options.maxFpr)
    if (multilabel) {
      const t = matrix(yTrue as TensorLike)
      const r = averageBinary(t, s, w, options.average ?? 'macro', score)
      return metricResult(name, r.value, n, Object.fromEntries(r.perClass.map((v, j) => [String(j), v])))
    }
    const labels = labelValues(yTrue)
    if (s.shape.length === 1) {
      const pos = positives(yTrue, undefined)
      return metricResult(name, binaryRocAuc(pos, s.data, w, options.maxFpr), n)
    }
    const classes = options.labels ?? sortLabels(labels)
    if (classes.length !== (s.shape[1] ?? 0)) throw new MdmlError('shape', `${name}: one score column per class`)
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
      throw new MdmlError('value', `${name}: partial AUC is only defined for binary targets`)
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

/** A detection error tradeoff curve: false positive and false negative rates at increasing thresholds. */
export interface DetCurve {
  readonly fpr: readonly number[]
  readonly fnr: readonly number[]
  readonly thresholds: readonly number[]
}

/**
 * The detection error tradeoff curve (sklearn.metrics.det_curve): FPR and FNR (miss rate) at each threshold,
 * trimmed to the stretch where both change, usually plotted on normal-deviate axes.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - posLabel, dropIntermediate (default false) and sample weights.
 * @returns The curve (FPR decreasing).
 * @throws MdmlError with code `value` when only one class is present.
 * @example
 * ```ts
 * import { detCurve } from 'mdml-engine/metrics'
 * detCurve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).fnr // => [0, 0.5, 0.5]
 * ```
 */
export function detCurve(
  yTrue: MetricInput,
  yScore: TensorLike,
  options: CurveOptions & { readonly dropIntermediate?: boolean } = {},
): DetCurve {
  const c = confusionMatrixAtThresholds(yTrue, yScore, options)
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
  if (pCount === 0 || nCount === 0) throw new MdmlError('value', 'detCurve: only one class is present')
  // From the last threshold with zero false positives to the first with zero false negatives.
  let first = 0
  while (first + 1 < fps.length && fps[first + 1] === fps[0]) first++
  let last = 0
  while (last < tps.length && (tps[last] ?? 0) < pCount) last++
  const idx = Array.from({ length: last + 1 - first }, (_, k) => first + k).reverse()
  return {
    fpr: idx.map((i) => (fps[i] ?? 0) / nCount),
    fnr: idx.map((i) => (pCount - (tps[i] ?? 0)) / pCount),
    thresholds: idx.map((i) => thresholds[i] ?? 0),
  }
}
