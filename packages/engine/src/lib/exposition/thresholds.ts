/**
 * The threshold sweep record: a binary classifier's scores cut at every distinct threshold, with the contingency
 * counts, the rates, the misclassification cost and the ROC and precision-recall points at each, so a view can move one
 * threshold through every figure of the evaluation (score distributions, cost, ROC, PR, the contingency table).
 */

import { OmaiError } from '../base/index.ts'
import { sameLength, seal } from './_check.ts'

/**
 * A binary classifier's scores cut at every distinct threshold (predict positive when the score is at least the
 * threshold), ascending, then +Infinity (nothing positive): the four counts at each (weighted when the rows are), the
 * rates (true- and false-positive rate, precision, selection rate, accuracy; NaN where a denominator is empty), the
 * total cost under a cost matrix and the cheapest threshold when costs are given, the areas under the ROC and
 * precision-recall curves, and optionally the scored rows (so a view can draw each class's score distribution).
 * The ROC points are `(fpr[i], tpr[i])` and the precision-recall points `(tpr[i], precision[i])`.
 */
export interface ThresholdSweep {
  readonly kind: 'thresholdSweep'
  /** The names of the negative and positive class. */
  readonly classes: readonly [string, string]
  /** Every distinct score, ascending, then +Infinity. */
  readonly thresholds: readonly number[]
  readonly tp: readonly number[]
  readonly fp: readonly number[]
  readonly tn: readonly number[]
  readonly fn: readonly number[]
  /** TP / (TP + FN): recall, sensitivity. */
  readonly tpr: readonly number[]
  /** FP / (FP + TN). */
  readonly fpr: readonly number[]
  /** TP / (TP + FP): NaN where nothing is predicted positive. */
  readonly precision: readonly number[]
  /** (TP + FP) / n. */
  readonly selectionRate: readonly number[]
  readonly accuracy: readonly number[]
  /** The (weighted) number of positive and negative rows. */
  readonly positives: number
  readonly negatives: number
  /** The cost matrix, `[[cost of TN, cost of FP], [cost of FN, cost of TP]]`, when costs were given. */
  readonly costs?: readonly (readonly number[])[]
  /** The total cost at each threshold. */
  readonly cost?: readonly number[]
  /** The index of the cheapest threshold (the lowest of any ties). */
  readonly cheapest?: number
  /** The area under the ROC curve (trapezoidal). */
  readonly rocAuc: number
  /** Average precision: the sum over thresholds of precision times the step in recall (scikit-learn's). */
  readonly averagePrecision: number
  /** The scored rows: each row's score and whether it is positive (1) or negative (0). */
  readonly scored?: { readonly scores: readonly number[]; readonly labels: readonly number[] }
}

const COLUMNS = ['tp', 'fp', 'tn', 'fn', 'tpr', 'fpr', 'precision', 'selectionRate', 'accuracy'] as const

/**
 * A threshold sweep, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `shape` when a column does not have one entry per threshold, the cost matrix is not 2 by 2, or
 *   the scored rows' columns differ in length; `value` when the thresholds are not ascending or the cheapest index is
 *   not a threshold.
 * @example
 * ```ts
 * import { thresholdSweep } from 'omai-engine/exposition'
 * const r = thresholdSweep({
 *   classes: ['no', 'yes'], thresholds: [0.2, 0.8, Infinity], positives: 1, negatives: 1,
 *   tp: [1, 1, 0], fp: [1, 0, 0], tn: [0, 1, 1], fn: [0, 0, 1],
 *   tpr: [1, 1, 0], fpr: [1, 0, 0], precision: [0.5, 1, Number.NaN], selectionRate: [1, 0.5, 0], accuracy: [0.5, 1, 0.5],
 *   rocAuc: 1, averagePrecision: 1,
 * })
 * r.kind // => 'thresholdSweep'
 * ```
 */
export function thresholdSweep(record: Omit<ThresholdSweep, 'kind'>): ThresholdSweep {
  const k = record.thresholds.length
  for (const field of COLUMNS) sameLength('thresholdSweep', field, record[field].length, k)
  for (let i = 1; i < k; i++)
    if (!((record.thresholds[i] ?? 0) > (record.thresholds[i - 1] ?? 0)))
      throw new OmaiError('value', 'thresholdSweep: the thresholds must be strictly ascending')
  if (record.costs !== undefined) {
    sameLength('thresholdSweep', 'costs', record.costs.length, 2)
    for (const [i, row] of record.costs.entries()) sameLength('thresholdSweep', `costs[${i}]`, row.length, 2)
  }
  if (record.cost !== undefined) sameLength('thresholdSweep', 'cost', record.cost.length, k)
  if (
    record.cheapest !== undefined &&
    !(Number.isInteger(record.cheapest) && record.cheapest >= 0 && record.cheapest < k)
  )
    throw new OmaiError('value', `thresholdSweep: the cheapest index ${record.cheapest} is not one of ${k}`)
  if (record.scored !== undefined)
    sameLength('thresholdSweep', 'scored.labels', record.scored.labels.length, record.scored.scores.length)
  return seal({ kind: 'thresholdSweep', ...record })
}

/**
 * The sweep's index for any threshold: the first swept threshold at or above it, whose decisions (`score >= t`) are the
 * same as the threshold's own. A threshold above every score gives the last index (+Infinity, nothing positive).
 *
 * @param record - The sweep.
 * @param threshold - Any threshold.
 * @returns The index into the sweep's columns.
 * @example
 * ```ts
 * import { sweepIndex } from 'omai-engine/exposition'
 * const r = { thresholds: [0.2, 0.5, 0.9, Infinity] }
 * sweepIndex(r, 0.3) // => 1
 * sweepIndex(r, 0.5) // => 1
 * sweepIndex(r, 2) // => 3
 * ```
 */
export function sweepIndex(record: Pick<ThresholdSweep, 'thresholds'>, threshold: number): number {
  const t = record.thresholds
  let lo = 0
  let hi = t.length - 1
  if (hi < 0) return 0
  if (!(threshold <= (t[hi] ?? Number.POSITIVE_INFINITY))) return hi
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if ((t[mid] ?? 0) >= threshold) hi = mid
    else lo = mid + 1
  }
  return lo
}

/** One row of a sweep: a threshold's counts, rates and cost. */
export interface SweepPoint {
  readonly index: number
  readonly threshold: number
  /** The 2 by 2 table, actual by predicted, negative first: `[[TN, FP], [FN, TP]]`. */
  readonly table: readonly (readonly number[])[]
  readonly tpr: number
  readonly fpr: number
  readonly precision: number
  readonly selectionRate: number
  readonly accuracy: number
  /** The total cost, when the sweep has costs. */
  readonly cost?: number
}

/**
 * The sweep at a threshold: its table, rates and cost (those of the first swept threshold at or above it).
 *
 * @param record - The sweep.
 * @param threshold - Any threshold.
 * @returns The point.
 * @example
 * ```ts
 * import { sweepAt, thresholdSweep } from 'omai-engine/exposition'
 * const r = thresholdSweep({
 *   classes: ['no', 'yes'], thresholds: [0.2, 0.8, Infinity], positives: 1, negatives: 1,
 *   tp: [1, 1, 0], fp: [1, 0, 0], tn: [0, 1, 1], fn: [0, 0, 1],
 *   tpr: [1, 1, 0], fpr: [1, 0, 0], precision: [0.5, 1, Number.NaN], selectionRate: [1, 0.5, 0], accuracy: [0.5, 1, 0.5],
 *   rocAuc: 1, averagePrecision: 1,
 * })
 * sweepAt(r, 0.5).table // => [[1, 0], [0, 1]]
 * ```
 */
export function sweepAt(record: ThresholdSweep, threshold: number): SweepPoint {
  const i = sweepIndex(record, threshold)
  const at = (xs: readonly number[]): number => xs[i] ?? Number.NaN
  const cost = record.cost?.[i]
  return {
    index: i,
    threshold: at(record.thresholds),
    table: [
      [at(record.tn), at(record.fp)],
      [at(record.fn), at(record.tp)],
    ],
    tpr: at(record.tpr),
    fpr: at(record.fpr),
    precision: at(record.precision),
    selectionRate: at(record.selectionRate),
    accuracy: at(record.accuracy),
    ...(cost === undefined ? {} : { cost }),
  }
}
