/**
 * The plain-number kernels of the threshold curves: the confusion counts at every distinct score, and the ROC and
 * precision-recall points computed from them. The public functions of `curves.ts` and `precision-recall.ts` hand these
 * back as tensors; the loops that read a curve (the operating points, the areas) take them here, as rows of numbers.
 * Private to the domain.
 */

import { MdmlError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import type { Label } from './_labels.ts'
import { positives } from './_scores.ts'
import { flat, type MetricInput, type MetricOptions, sampleWeights } from './metric.ts'

/** What the curve kernels read of the options of the public curves. */
export interface RowOptions extends MetricOptions {
  readonly posLabel?: Label
}

/** Weighted counts of a binary scorer at each distinct threshold, from the highest score down, as plain numbers. */
export interface ThresholdRows {
  readonly thresholds: number[]
  readonly tps: number[]
  readonly fps: number[]
  readonly tns: number[]
  readonly fns: number[]
}

/** The confusion counts of a binary scorer at every distinct score threshold, from the highest score down. */
export function thresholdRows(yTrue: MetricInput, yScore: TensorLike, options: RowOptions): ThresholdRows {
  const y = positives(yTrue, options.posLabel)
  const s = flat(yScore)
  if (s.length !== y.length) throw new MdmlError('shape', 'yTrue and yScore have different lengths')
  // A NaN score ties with every other in the sort and would be placed by the order of the data.
  if (s.some(Number.isNaN)) throw new MdmlError('value', 'confusionMatrixAtThresholds: scores must not be NaN')
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

/** The points of a ROC curve as plain numbers: false and true positive rates at decreasing thresholds. */
export interface RocRows {
  readonly fpr: number[]
  readonly tpr: number[]
  readonly thresholds: number[]
}

/** The ROC points of a binary scorer, starting from (0, 0) at +inf, with collinear intermediate points dropped. */
export function rocRows(
  yTrue: MetricInput,
  yScore: TensorLike,
  options: RowOptions & { readonly dropIntermediate?: boolean },
): RocRows {
  const c = thresholdRows(yTrue, yScore, options)
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

/** The points of a precision-recall curve as plain numbers: at increasing thresholds, ending at (recall 0, precision 1). */
export interface PrecisionRecallRows {
  readonly precision: number[]
  readonly recall: number[]
  readonly thresholds: number[]
}

/** The precision-recall points of a binary scorer at each threshold in increasing order, and the final point. */
export function precisionRecallRows(
  yTrue: MetricInput,
  yScore: TensorLike,
  options: RowOptions & { readonly dropIntermediate?: boolean },
): PrecisionRecallRows {
  const c = thresholdRows(yTrue, yScore, options)
  let keep = c.thresholds.map((_, i) => i)
  if (options.dropIntermediate === true && c.tps.length > 2) {
    keep = keep.filter(
      (i) =>
        i === 0 ||
        i === c.tps.length - 1 ||
        (c.tps[i] ?? 0) !== (c.tps[i - 1] ?? 0) ||
        (c.tps[i + 1] ?? 0) !== (c.tps[i] ?? 0),
    )
  }
  const tps = keep.map((i) => c.tps[i] ?? 0)
  const fps = keep.map((i) => c.fps[i] ?? 0)
  const total = tps[tps.length - 1] ?? 0
  const precision = tps.map((tp, i) => (tp + (fps[i] ?? 0) === 0 ? 0 : tp / (tp + (fps[i] ?? 0))))
  const recall = tps.map((tp) => (total === 0 ? 1 : tp / total))
  return {
    precision: [...precision.reverse(), 1],
    recall: [...recall.reverse(), 0],
    thresholds: keep.map((i) => c.thresholds[i] ?? 0).reverse(),
  }
}
