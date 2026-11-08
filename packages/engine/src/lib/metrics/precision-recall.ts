/**
 * Precision-recall analysis of binary scorers: the precision-recall curve and average precision as in scikit-learn, and
 * Flach and Kull's precision-recall-gain curve with its area, which rescales precision and recall so that areas behave
 * like ROC AUC.
 */

import type { MetricResult } from '../base/index.ts'
import { fromVector, type Tensor, type TensorLike } from '../tensor/index.ts'
import { precisionRecallRows, thresholdRows } from './_curves.ts'
import { isMultilabel, type Label, labelValues, sortLabels } from './_labels.ts'
import { averageBinary, binarise, positives, scoreSpec } from './_scores.ts'
import type { CurveOptions } from './curves.ts'
import { type Metric, type MetricInput, metricResult, sampleWeights, shapedArray } from './metric.ts'

/**
 * A precision-recall curve: precision and recall at increasing thresholds, ending at (recall 0, precision 1). The
 * vectors are tensors (migration, D-009: they were `readonly number[]`); there is one threshold fewer than points.
 */
export interface PrecisionRecallCurve {
  readonly precision: Tensor<'float64'>
  readonly recall: Tensor<'float64'>
  readonly thresholds: Tensor<'float64'>
}

/**
 * The precision-recall curve (sklearn.metrics.precision_recall_curve): precision and recall at each threshold in
 * increasing order, with a final point (recall 0, precision 1) that has no threshold.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - posLabel, dropIntermediate (default false) and sample weights.
 * @returns The curve, `[T]` tensors (was `number[]`).
 * @concept precision-recall
 * @example
 * ```ts
 * import { precisionRecallCurve } from 'omai-engine/metrics'
 * precisionRecallCurve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).recall // => [1, 1, 0.5, 0.5, 0]
 * ```
 */
export function precisionRecallCurve(
  yTrue: MetricInput,
  yScore: TensorLike,
  options: CurveOptions & { readonly dropIntermediate?: boolean } = {},
): PrecisionRecallCurve {
  const r = precisionRecallRows(yTrue, yScore, options)
  return { precision: fromVector(r.precision), recall: fromVector(r.recall), thresholds: fromVector(r.thresholds) }
}

/** The step-wise average precision of binary targets: sum over thresholds of (R_n - R_{n-1}) P_n. */
function binaryAveragePrecision(y: Float64Array, s: Float64Array, w: Float64Array | undefined): number {
  const pr = precisionRecallRows(Array.from(y), s, { posLabel: 1, ...(w === undefined ? {} : { sampleWeight: w }) })
  let ap = 0
  for (let i = 0; i < pr.recall.length - 1; i++) {
    ap -= ((pr.recall[i + 1] ?? 0) - (pr.recall[i] ?? 0)) * (pr.precision[i] ?? 0)
  }
  return ap
}

/** Options of {@link averagePrecisionScore}. */
export interface AveragePrecisionOptions extends CurveOptions {
  readonly average?: 'macro' | 'weighted' | 'micro' | 'samples'
  readonly labels?: readonly Label[]
}

/**
 * Average precision (sklearn.metrics.average_precision_score): the precision-recall curve summarised as
 * sum_n (R_n - R_{n-1}) P_n without interpolation; multi-class targets are binarised (one-vs-rest) and averaged.
 *
 * @param yTrue - True labels `[N]` or a multilabel indicator `[N, L]`.
 * @param yScore - Scores `[N]` or `[N, C]`.
 * @param options - average (default 'macro'), posLabel, labels and sample weights.
 * @returns The metric result.
 * @concept precision-recall
 * @example
 * ```ts
 * import { averagePrecisionScore } from 'omai-engine/metrics'
 * averagePrecisionScore([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).estimate.value // => 0.8333333333333333
 * ```
 */
export const averagePrecisionScore: Metric<AveragePrecisionOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yScore: MetricInput, options: AveragePrecisionOptions = {}): MetricResult => {
    const name = 'averagePrecisionScore'
    const s = shapedArray(yScore as TensorLike)
    const n = s.shape[0] ?? 0
    const w = options.sampleWeight === undefined ? undefined : sampleWeights(name, n, options.sampleWeight)
    if (s.shape.length === 1) {
      return metricResult(name, binaryAveragePrecision(positives(yTrue, options.posLabel), s.data, w), n)
    }
    const isIndicator = isMultilabel(yTrue)
    const t = isIndicator
      ? shapedArray(yTrue as TensorLike)
      : binarise(labelValues(yTrue), options.labels ?? sortLabels(labelValues(yTrue)))
    const r = averageBinary(t, s, w, options.average ?? 'macro', binaryAveragePrecision)
    return metricResult(name, r.value, n, Object.fromEntries(r.perClass.map((v, j) => [String(j), v])))
  },
  { spec: scoreSpec('averagePrecisionScore', 'higher', 'precision-recall') },
)

/**
 * A precision-recall-gain curve: points sorted by recall gain, with the crossing points marked. The numeric vectors are
 * tensors (migration, D-009: they were `readonly number[]`); `isCrossing` stays a list of flags.
 */
export interface PrGainCurve {
  readonly precisionGain: Tensor<'float64'>
  readonly recallGain: Tensor<'float64'>
  readonly thresholds: Tensor<'float64'>
  /** Whether each point is an interpolated crossing of an axis rather than a threshold. */
  readonly isCrossing: readonly boolean[]
}

interface PrgPoint {
  score: number
  tp: number
  fp: number
  fn: number
  tn: number
  pg: number
  rg: number
  crossing: boolean
}

/** Precision and recall gains of one contingency table (Flach and Kull's precision_gain and recall_gain). */
function gains(tp: number, fn: number, fp: number, tn: number): { pg: number; rg: number } {
  const ratio = (tp + fn) / (fp + tn)
  if (tn + fn === 0) return { pg: 0, rg: 1 }
  return { pg: 1 - ratio * (fp / tp), rg: 1 - ratio * (fn / tp) }
}

const interpolate = (a: PrgPoint, b: PrgPoint, alpha: number): PrgPoint => ({
  score: a.score + alpha * (b.score - a.score),
  tp: a.tp + alpha * (b.tp - a.tp),
  fp: a.fp + alpha * (b.fp - a.fp),
  fn: a.fn + alpha * (b.fn - a.fn),
  tn: a.tn + alpha * (b.tn - a.tn),
  pg: a.pg + alpha * (b.pg - a.pg),
  rg: a.rg + alpha * (b.rg - a.rg),
  crossing: true,
})

/** numpy's lexsort((-pg, rg)): recall gain ascending, then precision gain descending, NaN last, stable. */
function sortPoints(points: PrgPoint[]): PrgPoint[] {
  const key = (a: number, b: number): number => {
    if (Number.isNaN(a)) return Number.isNaN(b) ? 0 : 1
    if (Number.isNaN(b)) return -1
    return a < b ? -1 : a > b ? 1 : 0
  }
  return points.sort((a, b) => key(a.rg, b.rg) || key(-a.pg, -b.pg))
}

/** The points of the precision-recall-gain curve with its crossing points (the authors' create_prg_curve). */
function prgPoints(yTrue: MetricInput, yScore: TensorLike, options: CurveOptions): PrgPoint[] {
  const c = thresholdRows(yTrue, yScore, options)
  const nPos = c.tps[c.tps.length - 1] ?? 0
  const nNeg = c.fps[c.fps.length - 1] ?? 0
  const n = nPos + nNeg
  const at = (tp: number, fp: number, score: number): PrgPoint => {
    const fn = nPos - tp
    const tn = nNeg - fp
    return { score, tp, fp, fn, tn, ...gains(tp, fn, fp, tn), crossing: false }
  }
  let points = [
    at(0, 0, Number.POSITIVE_INFINITY),
    ...c.tps.map((tp, i) => at(tp, c.fps[i] ?? 0, c.thresholds[i] ?? 0)),
  ]
  // The crossing through the y-axis (recall gain 0).
  const j = points.findIndex((p) => p.rg >= 0)
  const pj = points[j]
  const prev = points[j - 1]
  if (pj !== undefined && prev !== undefined && pj.rg > 0) {
    const dTp = pj.tp - prev.tp
    const alpha = dTp > 0 ? ((nPos * nPos) / n - prev.tp) / dTp : 0.5
    const p = interpolate(prev, pj, alpha)
    points = sortPoints([{ ...p, pg: gains(p.tp, p.fn, p.fp, p.tn).pg, rg: 0 }, ...points])
  }
  // The crossings through the non-negative part of the x-axis (precision gain 0).
  const y0 = points.map((p) => p.pg)
  const x0 = points.map((p) => p.rg)
  const indices: number[] = []
  for (let i = 1; i < points.length; i++) {
    if ((y0[i] ?? 0) * (y0[i - 1] ?? 0) < 0 && (x0[i - 1] ?? 0) >= 0) indices.push(i)
  }
  for (const [k, i0] of indices.entries()) {
    const i = i0 + k
    const a = points[i - 1]
    const b = points[i]
    if (a === undefined || b === undefined) continue
    const crossX = a.rg + (-a.pg / (b.pg - a.pg)) * (b.rg - a.rg)
    const dTp = b.tp - a.tp
    const alpha =
      dTp > 0 ? ((nPos * nPos) / (n - nNeg * crossX) - a.tp) / dTp : ((nNeg / nPos) * a.tp - a.fp) / (b.fp - a.fp)
    const p = interpolate(a, b, alpha)
    points = sortPoints([{ ...p, pg: 0, rg: gains(p.tp, p.fn, p.fp, p.tn).rg }, ...points])
  }
  return points
}

/**
 * The precision-recall-gain curve (Flach and Kull, 2015): precision and recall rescaled as gains
 * 1 - (P / N)(FP / TP) and 1 - (P / N)(FN / TP), so that the always-positive baseline sits at the origin and areas
 * are meaningful, with the crossings of both axes interpolated, as the authors' create_prg_curve.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - posLabel and sample weights.
 * @returns The curve, by increasing recall gain (points with recall gain below 0 come first); the vectors are
 *   tensors (was `number[]`).
 * @concept precision-recall
 * @example
 * ```ts
 * import { prGainCurve } from 'omai-engine/metrics'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(prGainCurve([0, 1, 1, 0, 1], [0.1, 0.9, 0.8, 0.7, 0.3]).recallGain).at(-1) // => 1
 * ```
 */
export function prGainCurve(yTrue: MetricInput, yScore: TensorLike, options: CurveOptions = {}): PrGainCurve {
  const points = prgPoints(yTrue, yScore, options)
  return {
    precisionGain: fromVector(points.map((p) => p.pg)),
    recallGain: fromVector(points.map((p) => p.rg)),
    thresholds: fromVector(points.map((p) => p.score)),
    isCrossing: points.map((p) => p.crossing),
  }
}

/**
 * Area under the precision-recall-gain curve (Flach and Kull, 2015; the authors' calc_auprg): the trapezoidal area
 * over the part of the curve with recall gain at least 0.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { prGainAuc } from 'omai-engine/metrics'
 * prGainAuc([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2]).estimate.value // => 1
 * ```
 */
export const prGainAuc: Metric<CurveOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yScore: MetricInput, options: CurveOptions = {}): MetricResult => {
    const c = prgPoints(yTrue, yScore as TensorLike, options)
    let area = 0
    for (let i = 1; i < c.length; i++) {
      const a = c[i - 1] as PrgPoint
      const b = c[i] as PrgPoint
      if (!Number.isNaN(a.rg) && a.rg >= 0) area += ((b.rg - a.rg) * (b.pg + a.pg)) / 2
    }
    return metricResult('prGainAuc', area, labelValues(yTrue).length)
  },
  { spec: scoreSpec('prGainAuc', 'higher', 'precision-recall') },
)
