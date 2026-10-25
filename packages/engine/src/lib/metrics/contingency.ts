/**
 * Contingency tables of a binary or multiclass classifier and what is written around them: the margins and rates
 * (confusionMargins), the table of scores cut at a threshold (thresholdConfusion), misclassification costs of a table
 * and at every threshold (tableCost, thresholdCosts), and the shared threshold layer, analyseThresholds, which sweeps
 * every distinct threshold in one sorted pass into a thresholdSweep exposition record.
 */

import { WhiteboxError } from '../base/index.ts'
import { type ThresholdSweep, thresholdSweep } from '../exposition/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { type Label, labelValues, sortLabels } from './_labels.ts'
import type { ConfusionMatrix } from './classification.ts'
import { auc } from './curves.ts'
import { flat, type MetricInput, type MetricOptions, sampleWeights } from './metric.ts'

/**
 * The margins of a contingency table (a confusion matrix with actual classes as rows and predicted classes as
 * columns): totals and the rates read along rows and columns. A rate whose denominator is zero is NaN.
 */
export interface ConfusionMargins {
  /** The grand total. */
  readonly n: number
  /** Row totals: how many of each actual class. */
  readonly actual: readonly number[]
  /** Column totals: how many predicted as each class. */
  readonly predicted: readonly number[]
  /** Diagonal over row total: each class's recall (TPR for the positive class, TNR for the negative). */
  readonly rowRate: readonly number[]
  /** Off-diagonal over row total: each class's miss rate (FNR for the positive class, FPR for the negative). */
  readonly rowMiss: readonly number[]
  /** Diagonal over column total: each class's precision (PPV for the positive class, NPV for the negative). */
  readonly columnRate: readonly number[]
  /** Off-diagonal over column total: each class's false discovery rate (FDR; FOR for the negative class). */
  readonly columnMiss: readonly number[]
  /** Row total over n: each class's prevalence. */
  readonly prevalence: readonly number[]
  /** The diagonal over n. */
  readonly accuracy: number
}

/**
 * The margins of a confusion matrix: row and column totals, each class's recall and miss rate (rows), precision and
 * false discovery rate (columns), prevalence, and accuracy — everything a contingency table writes around its cells.
 *
 * @param table - A confusion matrix of counts (`confusionMatrix`, `thresholdConfusion`), actual classes as rows.
 * @returns The margins, in the table's class order.
 * @throws WhiteboxError `shape` when the matrix is not square or does not match its labels.
 * @concept classification-metrics
 * @example
 * ```ts
 * import { confusionMargins } from 'whitebox-engine/metrics'
 * const m = confusionMargins({ labels: [0, 1], matrix: [[3, 1], [2, 4]] })
 * m.accuracy // => 0.7
 * m.columnRate // => [0.6, 0.8]
 * m.actual // => [4, 6]
 * ```
 */
export function confusionMargins(table: ConfusionMatrix): ConfusionMargins {
  const k = table.labels.length
  if (table.matrix.length !== k || table.matrix.some((row) => row.length !== k))
    throw new WhiteboxError('shape', `confusionMargins: the matrix must be ${k} by ${k}, one row and column per label`)
  const at = (i: number, j: number): number => table.matrix[i]?.[j] ?? 0
  const actual = Array.from({ length: k }, (_, i) => table.matrix[i]?.reduce((s, v) => s + v, 0) ?? 0)
  const predicted = Array.from({ length: k }, (_, j) => actual.reduce((s, _a, i) => s + at(i, j), 0))
  const n = actual.reduce((s, v) => s + v, 0)
  const diagonal = Array.from({ length: k }, (_, i) => at(i, i))
  const ratio = (a: number, b: number): number => (b === 0 ? Number.NaN : a / b)
  return {
    n,
    actual,
    predicted,
    rowRate: diagonal.map((d, i) => ratio(d, actual[i] ?? 0)),
    rowMiss: diagonal.map((d, i) => ratio((actual[i] ?? 0) - d, actual[i] ?? 0)),
    columnRate: diagonal.map((d, j) => ratio(d, predicted[j] ?? 0)),
    columnMiss: diagonal.map((d, j) => ratio((predicted[j] ?? 0) - d, predicted[j] ?? 0)),
    prevalence: actual.map((a) => ratio(a, n)),
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
 * @returns The confusion matrix with labels `[negative, positive]`.
 * @throws WhiteboxError `value` with more than two classes; `shape` when the lengths differ.
 * @concept classification-metrics
 * @example
 * ```ts
 * import { thresholdConfusion } from 'whitebox-engine/metrics'
 * thresholdConfusion([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.3).matrix // => [[1, 1], [0, 2]]
 * thresholdConfusion(['ham', 'spam'], [0.2, 0.7], 0.5, { posLabel: 'spam' }).labels // => ['ham', 'spam']
 * ```
 */
export function thresholdConfusion(
  yTrue: MetricInput,
  yScore: TensorLike,
  threshold: number,
  options: ThresholdConfusionOptions = {},
): ConfusionMatrix {
  const t = labelValues(yTrue)
  const s = flat(yScore)
  if (s.length !== t.length)
    throw new WhiteboxError('shape', 'thresholdConfusion: yTrue and yScore have different lengths')
  const classes = sortLabels(t)
  if (classes.length > 2) throw new WhiteboxError('value', 'thresholdConfusion: needs two classes')
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
  return { labels: [neg, pos], matrix: m }
}

/** The cost of a contingency table under a cost matrix. */
export interface TableCost {
  /** cells[i][j]: the count of actual class i predicted as j, times its cost. */
  readonly cells: readonly (readonly number[])[]
  readonly total: number
  /** The total over the grand total of the table (the expected cost of one case). */
  readonly perCase: number
}

/**
 * The cost of a contingency table: each cell's count times the cost of that decision (`costs[actual][predicted]`, in
 * the table's class order; correct decisions usually cost 0 and may be negative to count as benefits), summed.
 *
 * @param table - A confusion matrix of counts.
 * @param costs - The cost matrix, one row per actual class and one column per predicted class.
 * @returns Each cell's cost, the total and the cost per case.
 * @throws WhiteboxError `shape` when the cost matrix does not match the table.
 * @concept classification-metrics
 * @example
 * ```ts
 * import { tableCost } from 'whitebox-engine/metrics'
 * // A missed illness (FN) costs 50 times a false alarm (FP).
 * const c = tableCost({ labels: ['healthy', 'ill'], matrix: [[880, 70], [8, 42]] }, [[0, 1], [50, 0]])
 * c.total // => 470
 * c.perCase // => 0.47
 * ```
 */
export function tableCost(table: ConfusionMatrix, costs: readonly (readonly number[])[]): TableCost {
  const k = table.labels.length
  if (costs.length !== k || costs.some((row) => row.length !== k))
    throw new WhiteboxError('shape', `tableCost: the cost matrix must be ${k} by ${k}, like the table`)
  const cells = table.matrix.map((row, i) => row.map((n, j) => n * (costs[i]?.[j] ?? 0)))
  const total = cells.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0)
  const n = table.matrix.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0)
  return { cells, total, perCase: n === 0 ? Number.NaN : total / n }
}

/** The total cost of cutting scores at every threshold, and the cheapest cut. */
export interface ThresholdCosts {
  /** Every distinct score, ascending, then +Infinity (nothing predicted positive). */
  readonly thresholds: readonly number[]
  /** The total cost at each threshold. */
  readonly totals: readonly number[]
  /** The cheapest threshold (the lowest of any ties) and its total. */
  readonly best: { readonly threshold: number; readonly total: number }
}

/**
 * The total misclassification cost of cutting scores at each threshold (positive when the score is at least the
 * threshold), under a 2 by 2 cost matrix in `thresholdConfusion`'s order (`[[TN, FP], [FN, TP]]` costs), and the
 * cheapest threshold. One sorted sweep, so every threshold costs O(1) after sorting.
 *
 * @param yTrue - Binary labels.
 * @param yScore - Scores, higher meaning more positive.
 * @param costs - `[[cost of TN, cost of FP], [cost of FN, cost of TP]]`.
 * @param options - posLabel and sample weights.
 * @returns The thresholds, their totals and the cheapest.
 * @throws WhiteboxError `shape` for a cost matrix that is not 2 by 2 or lengths that differ; `value` with more than two
 * classes.
 * @concept classification-metrics
 * @example
 * ```ts
 * import { thresholdCosts } from 'whitebox-engine/metrics'
 * // A false negative costs 5, a false positive 1.
 * thresholdCosts([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], [[0, 1], [5, 0]]).best // => { threshold: 0.4, total: 1 }
 * ```
 */
export function thresholdCosts(
  yTrue: MetricInput,
  yScore: TensorLike,
  costs: readonly (readonly number[])[],
  options: ThresholdConfusionOptions = {},
): ThresholdCosts {
  checkCosts('thresholdCosts', costs)
  const sweep = sweepScores('thresholdCosts', yTrue, yScore, options)
  const totals = sweepCost(sweep, costs)
  const best = cheapestIndex(totals)
  return {
    thresholds: sweep.thresholds,
    totals,
    best: { threshold: sweep.thresholds[best] ?? Number.POSITIVE_INFINITY, total: totals[best] ?? 0 },
  }
}

function checkCosts(name: string, costs: readonly (readonly number[])[]): void {
  if (costs.length !== 2 || costs.some((row) => row.length !== 2))
    throw new WhiteboxError('shape', `${name}: the cost matrix must be 2 by 2`)
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
  if (s.length !== t.length) throw new WhiteboxError('shape', `${name}: yTrue and yScore have different lengths`)
  for (const v of s) if (Number.isNaN(v)) throw new WhiteboxError('value', `${name}: a score is NaN`)
  const classes = sortLabels(t)
  if (classes.length > 2) throw new WhiteboxError('value', `${name}: needs two classes`)
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
  readonly costs?: readonly (readonly number[])[]
  /** Names of the negative and positive class (default the labels as text). */
  readonly classNames?: readonly [string, string]
  /** Keep the scored rows in the record, for score-distribution views (default true). */
  readonly keepRows?: boolean
}

/**
 * Every threshold of a binary classifier's scores at once, as a `thresholdSweep` exposition record: at each distinct
 * score (and +Infinity) the four counts, the true- and false-positive rates, precision, selection rate and accuracy,
 * the total cost and the cheapest threshold when costs are given, and the areas under the ROC and precision-recall
 * curves. One sorted pass, so every threshold costs O(1) after sorting; `sweepAt` reads any threshold back.
 *
 * @param yTrue - Binary labels.
 * @param yScore - Scores, higher meaning more positive.
 * @param options - Costs, posLabel, class names, sample weights and whether to keep the rows.
 * @returns The sweep.
 * @throws WhiteboxError `shape` for lengths that differ or a cost matrix that is not 2 by 2; `value` with more than two
 *   classes or a NaN score.
 * @concept classification-metrics
 * @example
 * ```ts
 * import { analyseThresholds } from 'whitebox-engine/metrics'
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
  if (options.costs !== undefined) checkCosts('analyseThresholds', options.costs)
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
  const cost = options.costs === undefined ? undefined : sweepCost(sweep, options.costs)
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
    ...(options.costs === undefined || cost === undefined
      ? {}
      : { costs: options.costs.map((row) => [...row]), cost, cheapest: cheapestIndex(cost) }),
    rocAuc: P === 0 || N === 0 ? Number.NaN : auc(fpr, tpr),
    averagePrecision,
    ...(options.keepRows === false
      ? {}
      : { scored: { scores: Array.from(sweep.scores), labels: sweep.isPositive.map((p) => (p ? 1 : 0)) } }),
  })
}
