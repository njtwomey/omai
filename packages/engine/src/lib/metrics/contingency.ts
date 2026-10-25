import { WhiteboxError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { type Label, labelValues, sortLabels } from './_labels.ts'
import type { ConfusionMatrix } from './classification.ts'
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
