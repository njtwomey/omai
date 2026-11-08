/**
 * Contingency tables of a binary or multiclass classifier and what is written around them: the margins and rates
 * (confusionMargins), the table of scores cut at a threshold (thresholdConfusion), misclassification costs of a table
 * and at every threshold (tableCost, thresholdCosts), and the shared threshold layer, analyseThresholds, which sweeps
 * every distinct threshold in one sorted pass into a thresholdSweep exposition record.
 */

import { OmaiError, toRows } from '../base/index.ts'
import { type ThresholdSweep, thresholdSweep } from '../exposition/index.ts'
import { asTensor, fromStorage, fromVector, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Label, labelValues, sortLabels } from './_labels.ts'
import type { LabelledConfusionMatrix } from './classification.ts'
import { auc } from './curves.ts'
import { flat, type MetricInput, type MetricOptions, sampleWeights, shapedArray } from './metric.ts'

/** A confusion matrix as the functions of this file take it: the classes and the `[k, k]` counts as a tensor-like. */
export interface ConfusionTable {
  readonly labels: readonly Label[]
  /** The counts, `[k, k]`: actual classes as rows, predicted classes as columns (a nested array works). */
  readonly matrix: TensorLike
}

/** The counts of a table as rows of plain numbers, checked to be `k` by `k` for its `k` labels. */
function tableRows(name: string, table: ConfusionTable): number[][] {
  const k = table.labels.length
  const t = asTensor(table.matrix)
  if (t.shape.length !== 2 || t.shape[0] !== k || t.shape[1] !== k)
    throw new OmaiError('shape', `${name}: the matrix must be ${k} by ${k}, one row and column per label`)
  return toRows(toFlat(t), k, k)
}

/**
 * The margins of a contingency table (a confusion matrix with actual classes as rows and predicted classes as
 * columns): totals and the rates read along rows and columns. A rate whose denominator is zero is NaN. The vectors
 * are `[k]` tensors (migration, D-009: they were `readonly number[]`).
 */
export interface ConfusionMargins {
  /** The grand total. */
  readonly n: number
  /** Row totals: how many of each actual class. */
  readonly actual: Tensor<'float64'>
  /** Column totals: how many predicted as each class. */
  readonly predicted: Tensor<'float64'>
  /** Diagonal over row total: each class's recall (TPR for the positive class, TNR for the negative). */
  readonly rowRate: Tensor<'float64'>
  /** Off-diagonal over row total: each class's miss rate (FNR for the positive class, FPR for the negative). */
  readonly rowMiss: Tensor<'float64'>
  /** Diagonal over column total: each class's precision (PPV for the positive class, NPV for the negative). */
  readonly columnRate: Tensor<'float64'>
  /** Off-diagonal over column total: each class's false discovery rate (FDR; FOR for the negative class). */
  readonly columnMiss: Tensor<'float64'>
  /** Row total over n: each class's prevalence. */
  readonly prevalence: Tensor<'float64'>
  /** The diagonal over n. */
  readonly accuracy: number
}

/**
 * The margins of a confusion matrix: row and column totals, each class's recall and miss rate (rows), precision and
 * false discovery rate (columns), prevalence, and accuracy — everything a contingency table writes around its cells.
 *
 * @param table - A confusion matrix of counts (`confusionMatrix`, `thresholdConfusion`), actual classes as rows; the
 *   matrix takes a tensor-like (migration, D-009: it was a nested array, which still works).
 * @returns The margins, in the table's class order.
 * @throws OmaiError `shape` when the matrix is not square or does not match its labels.
 * @concept classification-metrics
 * @example
 * ```ts
 * import { confusionMargins } from 'omai-engine/metrics'
 * const m = confusionMargins({ labels: [0, 1], matrix: [[3, 1], [2, 4]] })
 * m.accuracy // => 0.7
 * m.columnRate // => [0.6, 0.8]
 * m.actual // => [4, 6]
 * ```
 */
export function confusionMargins(table: ConfusionTable): ConfusionMargins {
  const k = table.labels.length
  const rows = tableRows('confusionMargins', table)
  const at = (i: number, j: number): number => rows[i]?.[j] ?? 0
  const actual = Array.from({ length: k }, (_, i) => rows[i]?.reduce((s, v) => s + v, 0) ?? 0)
  const predicted = Array.from({ length: k }, (_, j) => actual.reduce((s, _a, i) => s + at(i, j), 0))
  const n = actual.reduce((s, v) => s + v, 0)
  const diagonal = Array.from({ length: k }, (_, i) => at(i, i))
  const ratio = (a: number, b: number): number => (b === 0 ? Number.NaN : a / b)
  return {
    n,
    actual: fromVector(actual),
    predicted: fromVector(predicted),
    rowRate: fromVector(diagonal.map((d, i) => ratio(d, actual[i] ?? 0))),
    rowMiss: fromVector(diagonal.map((d, i) => ratio((actual[i] ?? 0) - d, actual[i] ?? 0))),
    columnRate: fromVector(diagonal.map((d, j) => ratio(d, predicted[j] ?? 0))),
    columnMiss: fromVector(diagonal.map((d, j) => ratio((predicted[j] ?? 0) - d, predicted[j] ?? 0))),
    prevalence: fromVector(actual.map((a) => ratio(a, n))),
    accuracy: ratio(
      diagonal.reduce((s, v) => s + v, 0),
      n,
    ),
  }
}

/** Options of {@link thresholdConfusion}. */
export interface ThresholdConfusionOptions extends MetricOptions {
  /** The positive class (default 1 when present, else the larger label). */
  readonly posLabel?: Label
}

/**
 * The 2 by 2 confusion matrix of scores cut at a threshold: a sample is predicted positive when its score is at least
 * the threshold. Rows are actual and columns predicted, ordered negative then positive, so the positive class is
 * index 1 (`[[TN, FP], [FN, TP]]`, as scikit-learn's `confusion_matrix(y, score >= t)`).
 *
 * @param yTrue - Binary labels.
 * @param yScore - Scores, higher meaning more positive (probabilities, decision values).
 * @param threshold - The cut.
 * @param options - posLabel and sample weights.
 * @returns The confusion matrix with labels `[negative, positive]`; the `[2, 2]` matrix is a tensor (migration,
 *   D-009: was `number[][]`).
 * @throws OmaiError `value` with more than two classes; `shape` when the lengths differ.
 * @concept classification-metrics
 * @example
 * ```ts
 * import { thresholdConfusion } from 'omai-engine/metrics'
 * thresholdConfusion([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.3).matrix // => [[1, 1], [0, 2]]
 * thresholdConfusion(['ham', 'spam'], [0.2, 0.7], 0.5, { posLabel: 'spam' }).labels // => ['ham', 'spam']
 * ```
 */
export function thresholdConfusion(
  yTrue: MetricInput,
  yScore: TensorLike,
  threshold: number,
  options: ThresholdConfusionOptions = {},
): LabelledConfusionMatrix {
  const t = labelValues(yTrue)
  const s = flat(yScore)
  if (s.length !== t.length) throw new OmaiError('shape', 'thresholdConfusion: yTrue and yScore have different lengths')
  const classes = sortLabels(t)
  if (classes.length > 2) throw new OmaiError('value', 'thresholdConfusion: needs two classes')
  const pos = options.posLabel ?? (classes.includes(1) ? 1 : (classes[classes.length - 1] ?? 1))
  const neg = classes.find((c) => c !== pos) ?? (typeof pos === 'number' ? (pos === 0 ? 1 : 0) : `not ${pos}`)
  const w = sampleWeights('thresholdConfusion', t.length, options.sampleWeight)
  const m = [
    [0, 0],
    [0, 0],
  ]
  for (let i = 0; i < t.length; i++) {
    const row = m[t[i] === pos ? 1 : 0]
    const j = (s[i] ?? Number.NaN) >= threshold ? 1 : 0
    if (row !== undefined) row[j] = (row[j] ?? 0) + (w[i] ?? 0)
  }
  return { labels: [neg, pos], matrix: fromStorage(Float64Array.of(...m.flat()), [2, 2], 'float64') }
}

/** The cost of a contingency table under a cost matrix. Holds a tensor (migration, D-009: `cells` was nested arrays). */
export interface TableCost {
  /** The `[k, k]` cells: the count of actual class i predicted as j, times its cost. */
  readonly cells: Tensor<'float64'>
  readonly total: number
  /** The total over the grand total of the table (the expected cost of one case). */
  readonly perCase: number
}

/**
 * The cost of a contingency table: each cell's count times the cost of that decision (`costs[actual][predicted]`, in
 * the table's class order; correct decisions usually cost 0 and may be negative to count as benefits), summed. Takes
 * tensors (migration, D-009: `costs` was `readonly (readonly number[])[]`; nested arrays still work).
 *
 * @param table - A confusion matrix of counts (the matrix takes a tensor-like).
 * @param costs - The cost matrix `[k, k]`, one row per actual class and one column per predicted class.
 * @returns Each cell's cost (a `[k, k]` tensor, was `number[][]`), the total and the cost per case.
 * @throws OmaiError `shape` when the cost matrix does not match the table.
 * @concept classification-metrics
 * @example
 * ```ts
 * import { tableCost } from 'omai-engine/metrics'
 * // A missed illness (FN) costs 50 times a false alarm (FP).
 * const c = tableCost({ labels: ['healthy', 'ill'], matrix: [[880, 70], [8, 42]] }, [[0, 1], [50, 0]])
 * c.total // => 470
 * c.perCase // => 0.47
 * ```
 */
export function tableCost(table: ConfusionTable, costs: TensorLike): TableCost {
  const k = table.labels.length
  const c = costRows('tableCost', costs, k, `${k} by ${k}, like the table`)
  const rows = tableRows('tableCost', table)
  const cells = rows.map((row, i) => row.map((n, j) => n * (c[i]?.[j] ?? 0)))
  const total = cells.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0)
  const n = rows.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0)
  return {
    cells: fromStorage(Float64Array.from(cells.flat()), [k, k], 'float64'),
    total,
    perCase: n === 0 ? Number.NaN : total / n,
  }
}

/**
 * The total cost of cutting scores at every threshold, and the cheapest cut. The vectors are tensors (migration, D-009:
 * they were `readonly number[]`).
 */
export interface ThresholdCosts {
  /** Every distinct score, ascending, then +Infinity (nothing predicted positive), a `[m]` tensor. */
  readonly thresholds: Tensor<'float64'>
  /** The total cost at each threshold, a `[m]` tensor. */
  readonly totals: Tensor<'float64'>
  /** The cheapest threshold (the lowest of any ties) and its total. */
  readonly best: { readonly threshold: number; readonly total: number }
}

/**
 * The total misclassification cost of cutting scores at each threshold (positive when the score is at least the
 * threshold), under a 2 by 2 cost matrix in `thresholdConfusion`'s order (`[[TN, FP], [FN, TP]]` costs), and the
 * cheapest threshold. One sorted sweep, so every threshold costs O(1) after sorting. Takes tensors (migration, D-009:
 * `costs` was `readonly (readonly number[])[]`; nested arrays still work).
 *
 * @param yTrue - Binary labels.
 * @param yScore - Scores, higher meaning more positive.
 * @param costs - `[[cost of TN, cost of FP], [cost of FN, cost of TP]]`, a `[2, 2]` matrix.
 * @param options - posLabel and sample weights.
 * @returns The thresholds, their totals (`[m]` tensors, were `number[]`) and the cheapest.
 * @throws OmaiError `shape` for a cost matrix that is not 2 by 2 or lengths that differ; `value` with more than two
 * classes.
 * @concept classification-metrics
 * @example
 * ```ts
 * import { thresholdCosts } from 'omai-engine/metrics'
 * // A false negative costs 5, a false positive 1.
 * thresholdCosts([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], [[0, 1], [5, 0]]).best // => { threshold: 0.4, total: 1 }
 * ```
 */
export function thresholdCosts(
  yTrue: MetricInput,
  yScore: TensorLike,
  costs: TensorLike,
  options: ThresholdConfusionOptions = {},
): ThresholdCosts {
  const c = costRows('thresholdCosts', costs, 2, '2 by 2')
  const sweep = sweepScores('thresholdCosts', yTrue, yScore, options)
  const totals = sweepCost(sweep, c)
  const best = cheapestIndex(totals)
  return {
    thresholds: fromVector(sweep.thresholds),
    totals: fromVector(totals),
    best: { threshold: sweep.thresholds[best] ?? Number.POSITIVE_INFINITY, total: totals[best] ?? 0 },
  }
}

/** The rows of a cost matrix that must be `k` by `k` (`what` names the size in the error). */
function costRows(name: string, costs: TensorLike, k: number, what: string): number[][] {
  const m = shapedArray(costs)
  if (m.shape.length !== 2 || m.shape[0] !== k || m.shape[1] !== k)
    throw new OmaiError('shape', `${name}: the cost matrix must be ${what}`)
  return toRows(m.data, k, k)
}

/** The counts of scores cut at every distinct threshold, from one sorted pass. */
interface ScoreSweep {
  readonly negative: Label
  readonly positive: Label
  /** Every distinct score ascending, then +Infinity. */
  readonly thresholds: number[]
  readonly tp: number[]
  readonly fp: number[]
  readonly positives: number
  readonly negatives: number
  /** Each row's score, and whether it is positive. */
  readonly scores: Float64Array
  readonly isPositive: boolean[]
}

// Sweep thresholds upwards: at a threshold equal to a distinct score, every sample at or above it is positive, so the
// counts at each threshold are the totals less what lies below it.
function sweepScores(
  name: string,
  yTrue: MetricInput,
  yScore: TensorLike,
  options: ThresholdConfusionOptions,
): ScoreSweep {
  const t = labelValues(yTrue)
  const s = flat(yScore)
  if (s.length !== t.length) throw new OmaiError('shape', `${name}: yTrue and yScore have different lengths`)
  for (const v of s) if (Number.isNaN(v)) throw new OmaiError('value', `${name}: a score is NaN`)
  const classes = sortLabels(t)
  if (classes.length > 2) throw new OmaiError('value', `${name}: needs two classes`)
  const positive = options.posLabel ?? (classes.includes(1) ? 1 : (classes[classes.length - 1] ?? 1))
  const negative =
    classes.find((c) => c !== positive) ?? (typeof positive === 'number' ? (positive === 0 ? 1 : 0) : `not ${positive}`)
  const w = sampleWeights(name, t.length, options.sampleWeight)
  const isPositive = t.map((v) => v === positive)
  const order = Array.from({ length: t.length }, (_, i) => i).sort((a, b) => (s[a] ?? 0) - (s[b] ?? 0))
  let positives = 0
  let negatives = 0
  for (let i = 0; i < t.length; i++) {
    if (isPositive[i] === true) positives += w[i] ?? 0
    else negatives += w[i] ?? 0
  }
  const thresholds: number[] = []
  const tp: number[] = []
  const fp: number[] = []
  let belowPos = 0
  let belowNeg = 0
  for (let r = 0; r < order.length; ) {
    const value = s[order[r] ?? 0] ?? 0
    thresholds.push(value)
    tp.push(positives - belowPos)
    fp.push(negatives - belowNeg)
    while (r < order.length && (s[order[r] ?? 0] ?? 0) === value) {
      const i = order[r] ?? 0
      if (isPositive[i] === true) belowPos += w[i] ?? 0
      else belowNeg += w[i] ?? 0
      r++
    }
  }
  thresholds.push(Number.POSITIVE_INFINITY)
  tp.push(positives - belowPos)
  fp.push(negatives - belowNeg)
  return { negative, positive, thresholds, tp, fp, positives, negatives, scores: s, isPositive }
}

function sweepCost(sweep: ScoreSweep, costs: readonly (readonly number[])[]): number[] {
  const [[cTn = 0, cFp = 0] = [], [cFn = 0, cTp = 0] = []] = costs
  return sweep.tp.map((tp, i) => {
    const fp = sweep.fp[i] ?? 0
    return (sweep.negatives - fp) * cTn + fp * cFp + (sweep.positives - tp) * cFn + tp * cTp
  })
}

function cheapestIndex(totals: readonly number[]): number {
  let best = 0
  for (let i = 1; i < totals.length; i++) if ((totals[i] ?? 0) < (totals[best] ?? 0)) best = i
  return best
}

/** Options of {@link analyseThresholds}. */
export interface AnalyseThresholdsOptions extends ThresholdConfusionOptions {
  /** A 2 by 2 cost matrix, `[[cost of TN, cost of FP], [cost of FN, cost of TP]]`: adds the cost and the cheapest. */
  readonly costs?: TensorLike
  /** Names of the negative and positive class (default the labels as text). */
  readonly classNames?: readonly [string, string]
  /** Keep the scored rows in the record, for score-distribution views (default true). */
  readonly keepRows?: boolean
}

/**
 * Every threshold of a binary classifier's scores at once, as a `thresholdSweep` exposition record: at each distinct
 * score (and +Infinity) the four counts, the true- and false-positive rates, precision, selection rate and accuracy,
 * the total cost and the cheapest threshold when costs are given, and the areas under the ROC and precision-recall
 * curves. One sorted pass, so every threshold costs O(1) after sorting; `sweepAt` reads any threshold back. Takes
 * tensors (migration, D-009: the `costs` option was `readonly (readonly number[])[]`; nested arrays still work).
 *
 * @param yTrue - Binary labels.
 * @param yScore - Scores, higher meaning more positive.
 * @param options - Costs, posLabel, class names, sample weights and whether to keep the rows.
 * @returns The sweep.
 * @throws OmaiError `shape` for lengths that differ or a cost matrix that is not 2 by 2; `value` with more than two
 *   classes or a NaN score.
 * @concept classification-metrics
 * @example
 * ```ts
 * import { analyseThresholds } from 'omai-engine/metrics'
 * const r = analyseThresholds([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], { costs: [[0, 1], [5, 0]] })
 * r.thresholds // => [0.1, 0.4, 0.6, 0.9, Infinity]
 * r.tp // => [2, 2, 1, 1, 0]
 * r.cost // => [2, 1, 6, 5, 10]
 * r.thresholds[r.cheapest ?? 0] // => 0.4
 * r.rocAuc // => 0.75
 * ```
 */
export function analyseThresholds(
  yTrue: MetricInput,
  yScore: TensorLike,
  options: AnalyseThresholdsOptions = {},
): ThresholdSweep {
  const costs = options.costs === undefined ? undefined : costRows('analyseThresholds', options.costs, 2, '2 by 2')
  const sweep = sweepScores('analyseThresholds', yTrue, yScore, options)
  const { positives: P, negatives: N } = sweep
  const ratio = (a: number, b: number): number => (b === 0 ? Number.NaN : a / b)
  const tn = sweep.fp.map((fp) => N - fp)
  const fn = sweep.tp.map((tp) => P - tp)
  const tpr = sweep.tp.map((tp) => ratio(tp, P))
  const fpr = sweep.fp.map((fp) => ratio(fp, N))
  const precision = sweep.tp.map((tp, i) => ratio(tp, tp + (sweep.fp[i] ?? 0)))
  let averagePrecision = P === 0 ? Number.NaN : 0
  if (P > 0)
    for (let i = 0; i + 1 < tpr.length; i++) {
      const step = (tpr[i] ?? 0) - (tpr[i + 1] ?? 0)
      if (step > 0) averagePrecision += step * (precision[i] ?? 0)
    }
  const cost = costs === undefined ? undefined : sweepCost(sweep, costs)
  return thresholdSweep({
    classes: options.classNames ?? [String(sweep.negative), String(sweep.positive)],
    thresholds: sweep.thresholds,
    tp: sweep.tp,
    fp: sweep.fp,
    tn,
    fn,
    tpr,
    fpr,
    precision,
    selectionRate: sweep.tp.map((tp, i) => ratio(tp + (sweep.fp[i] ?? 0), P + N)),
    accuracy: sweep.tp.map((tp, i) => ratio(tp + (tn[i] ?? 0), P + N)),
    positives: P,
    negatives: N,
    ...(costs === undefined || cost === undefined ? {} : { costs, cost, cheapest: cheapestIndex(cost) }),
    rocAuc: P === 0 || N === 0 ? Number.NaN : auc(fpr, tpr),
    averagePrecision,
    ...(options.keepRows === false
      ? {}
      : { scored: { scores: Array.from(sweep.scores), labels: sweep.isPositive.map((p) => (p ? 1 : 0)) } }),
  })
}
