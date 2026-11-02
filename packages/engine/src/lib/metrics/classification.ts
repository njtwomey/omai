/**
 * The core classification metrics computed from the confusion matrix, scikit-learn style: precision, recall
 * (sensitivity), specificity, negative predictive value, informedness and markedness, F-beta and F1, the Jaccard index
 * and accuracy, with binary, micro, macro, weighted and per-sample averaging, plus `confusionMatrix` and the per-class
 * multilabel confusion matrices.
 */

import { MdmlError, type MetricResult, type MetricSpec } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import {
  type ClassCounts,
  classCounts,
  encodeLabels,
  indicator,
  isMultilabel,
  type Label,
  multilabelCounts,
  ratio,
} from './_labels.ts'
import { type Metric, type MetricInput, type MetricOptions, metricResult, rowCount, sampleWeights } from './metric.ts'

/**
 * How per-class scores combine: 'binary' (the positive class only), 'micro' (pooled counts), 'macro' (unweighted mean
 * over classes), 'weighted' (mean weighted by support), 'samples' (per-sample scores of multilabel data, averaged) or
 * 'none' (per-class scores in the breakdown; the estimate is their unweighted mean).
 */
export type Average = 'binary' | 'micro' | 'macro' | 'weighted' | 'samples' | 'none'

/** Options of the averaged classification metrics (scikit-learn's). */
export interface AveragingOptions extends MetricOptions {
  /** The classes to score, in order (multilabel: column indices); default every class seen. */
  readonly labels?: readonly Label[]
  /** The positive class for 'binary' averaging (default 1). */
  readonly posLabel?: Label
  /** The averaging (default 'binary'). */
  readonly average?: Average
  /** The value of a ratio whose denominator is zero (default 0; NaN excludes the class from averages). */
  readonly zeroDivision?: number
}

const spec = (name: string, direction: 'higher' | 'lower', range: readonly [number, number]): MetricSpec => ({
  name,
  direction,
  range,
  inputs: 'labels',
  category: 'classification',
  concept: 'classification-metrics',
})

/** A rate of one class's one-vs-rest counts. */
type Rate = (tp: number, fp: number, fn: number, tn: number, zeroDivision: number) => number

/** The mean of the non-NaN values, weighted (NaN when none remain). */
function nanAverage(values: readonly number[], weights?: readonly number[]): number {
  let s = 0
  let w = 0
  for (const [i, v] of values.entries()) {
    if (Number.isNaN(v)) continue
    const wi = weights === undefined ? 1 : (weights[i] ?? 0)
    s += v * wi
    w += wi
  }
  return w === 0 ? Number.NaN : s / w
}

/** Per-class counts of multi-class or multilabel targets, with their class names. */
function countsOf(
  name: string,
  yTrue: MetricInput,
  yPred: MetricInput,
  options: AveragingOptions,
): ClassCounts & { readonly classes: readonly Label[]; readonly multilabel: boolean } {
  if (isMultilabel(yTrue)) {
    const t = indicator(name, yTrue as TensorLike)
    const p = indicator(name, yPred as TensorLike)
    const w = sampleWeights(name, t.n, options.sampleWeight)
    const c = multilabelCounts(t, p, w, options.labels as readonly number[] | undefined)
    return { ...c, classes: c.labels, multilabel: true }
  }
  const e = encodeLabels(name, yTrue, yPred, options)
  return { ...classCounts(e), classes: e.classes, multilabel: false }
}

/** Score each sample of multilabel data by a rate over its labels ('samples' averaging). */
function samplesAverage(
  name: string,
  yTrue: MetricInput,
  yPred: MetricInput,
  options: AveragingOptions,
  rate: Rate,
): number {
  if (!isMultilabel(yTrue)) throw new MdmlError('value', `${name}: 'samples' averaging needs multilabel targets`)
  const t = indicator(name, yTrue as TensorLike)
  const p = indicator(name, yPred as TensorLike)
  const w = sampleWeights(name, t.n, options.sampleWeight)
  const cols = (options.labels as readonly number[] | undefined) ?? Array.from({ length: t.l }, (_, j) => j)
  const zd = options.zeroDivision ?? 0
  const scores: number[] = []
  for (let i = 0; i < t.n; i++) {
    let tp = 0
    let fp = 0
    let fn = 0
    let tn = 0
    for (const j of cols) {
      const a = (t.data[i * t.l + j] ?? 0) > 0
      const b = (p.data[i * p.l + j] ?? 0) > 0
      if (a && b) tp++
      else if (b) fp++
      else if (a) fn++
      else tn++
    }
    scores.push(rate(tp, fp, fn, tn, zd))
  }
  return nanAverage(scores, Array.from(w))
}

/** Score targets and predictions by a per-class rate, averaged as asked; per-class values go in the breakdown. */
function averaged(
  name: string,
  yTrue: MetricInput,
  yPred: MetricInput,
  options: AveragingOptions,
  rate: Rate,
): MetricResult {
  const average = options.average ?? 'binary'
  const zd = options.zeroDivision ?? 0
  const n = rowCount(yTrue)
  if (average === 'samples') return metricResult(name, samplesAverage(name, yTrue, yPred, options, rate), n)
  const c = countsOf(name, yTrue, yPred, options)
  const k = c.classes.length
  const perClass = Array.from({ length: k }, (_, i) => rate(c.tp[i] ?? 0, c.fp[i] ?? 0, c.fn[i] ?? 0, c.tn[i] ?? 0, zd))
  const size = n
  const breakdown = Object.fromEntries(c.classes.map((label, i) => [String(label), perClass[i] ?? Number.NaN]))
  if (average === 'binary') {
    if (c.multilabel) throw new MdmlError('value', `${name}: 'binary' averaging needs binary targets; pass average`)
    const pos = options.posLabel ?? 1
    if (k > 2) throw new MdmlError('value', `${name}: targets are multi-class; pass average ('micro', 'macro', ...)`)
    const i = c.classes.indexOf(pos)
    if (i < 0) {
      if (k <= 1) return metricResult(name, rate(0, 0, 0, size, zd), size)
      throw new MdmlError('value', `${name}: posLabel ${String(pos)} is not a class`)
    }
    return metricResult(name, perClass[i] ?? Number.NaN, size)
  }
  if (average === 'micro') {
    const sum = (a: Float64Array): number => a.reduce((s, v) => s + v, 0)
    return metricResult(name, rate(sum(c.tp), sum(c.fp), sum(c.fn), sum(c.tn), zd), size, breakdown)
  }
  if (average === 'weighted') {
    const support = Array.from({ length: k }, (_, i) => (c.tp[i] ?? 0) + (c.fn[i] ?? 0))
    const total = support.reduce((s, v) => s + v, 0)
    const value = total === 0 ? zd : nanAverage(perClass, support)
    return metricResult(name, value, size, breakdown)
  }
  return metricResult(name, nanAverage(perClass), size, breakdown)
}

/** Define an averaged rate metric. */
function rateMetric<O extends AveragingOptions>(
  name: string,
  rate: (options: O) => Rate,
  direction: 'higher' | 'lower' = 'higher',
  range: readonly [number, number] = [0, 1],
): Metric<O, MetricInput> {
  const metric = (yTrue: MetricInput, yPred: MetricInput, options: O = {} as O): MetricResult =>
    averaged(name, yTrue, yPred, options, rate(options))
  return Object.assign(metric, { spec: spec(name, direction, range) })
}

const precisionRate: Rate = (tp, fp, _fn, _tn, zd) => ratio(tp, tp + fp, zd)
const recallRate: Rate = (tp, _fp, fn, _tn, zd) => ratio(tp, tp + fn, zd)
const specificityRate: Rate = (_tp, fp, _fn, tn, zd) => ratio(tn, tn + fp, zd)
const npvRate: Rate = (_tp, _fp, fn, tn, zd) => ratio(tn, tn + fn, zd)
const fbetaRate =
  (beta: number): Rate =>
  (tp, fp, fn, _tn, zd) => {
    if (beta === Number.POSITIVE_INFINITY) return ratio(tp, tp + fn, zd)
    const b2 = beta * beta
    return ratio((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp, zd)
  }

/**
 * Precision tp / (tp + fp) (sklearn.metrics.precision_score), the positive predictive value.
 *
 * @param yTrue - True labels `[N]` (numbers or strings) or a multilabel indicator `[N, L]`.
 * @param yPred - Predicted labels, the same form.
 * @param options - average, labels, posLabel, zeroDivision and sample weights.
 * @returns The metric result (per-class values in the breakdown unless binary).
 * @concept classification-metrics
 * @example
 * ```ts
 * import { precisionScore } from 'mdml-engine/metrics'
 * precisionScore([0, 1, 1, 0], [1, 1, 1, 0]).estimate.value // => 0.6666666666666666
 * ```
 */
export const precisionScore: Metric<AveragingOptions, MetricInput> = rateMetric('precisionScore', () => precisionRate)

/**
 * Recall tp / (tp + fn) (sklearn.metrics.recall_score), the sensitivity or true positive rate.
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - average, labels, posLabel, zeroDivision and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { recallScore } from 'mdml-engine/metrics'
 * recallScore([0, 1, 1, 0], [1, 1, 0, 0]).estimate.value // => 0.5
 * ```
 */
export const recallScore: Metric<AveragingOptions, MetricInput> = rateMetric('recallScore', () => recallRate)

/**
 * Sensitivity, the true positive rate: the same quantity as {@link recallScore}, under its diagnostic-testing name.
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - average, labels, posLabel, zeroDivision and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { sensitivityScore } from 'mdml-engine/metrics'
 * sensitivityScore([0, 1, 1, 0], [1, 1, 0, 0]).estimate.value // => 0.5
 * ```
 */
export const sensitivityScore: Metric<AveragingOptions, MetricInput> = rateMetric('sensitivityScore', () => recallRate)

/**
 * Specificity tn / (tn + fp), the true negative rate (imbalanced-learn's specificity_score).
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - average, labels, posLabel, zeroDivision and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { specificityScore } from 'mdml-engine/metrics'
 * specificityScore([0, 1, 1, 0], [1, 1, 0, 0]).estimate.value // => 0.5
 * ```
 */
export const specificityScore: Metric<AveragingOptions, MetricInput> = rateMetric(
  'specificityScore',
  () => specificityRate,
)

/**
 * Negative predictive value tn / (tn + fn).
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - average, labels, posLabel, zeroDivision and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { negativePredictiveValue } from 'mdml-engine/metrics'
 * negativePredictiveValue([0, 1, 1, 0], [1, 1, 0, 0]).estimate.value // => 0.5
 * ```
 */
export const negativePredictiveValue: Metric<AveragingOptions, MetricInput> = rateMetric(
  'negativePredictiveValue',
  () => npvRate,
)

/**
 * Informedness (Youden's J, bookmaker informedness): sensitivity + specificity - 1, in [-1, 1].
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - average, labels, posLabel, zeroDivision and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { informedness } from 'mdml-engine/metrics'
 * informedness([0, 1, 1, 0], [0, 1, 1, 0]).estimate.value // => 1
 * ```
 */
export const informedness: Metric<AveragingOptions, MetricInput> = rateMetric(
  'informedness',
  () => (tp, fp, fn, tn, zd) => recallRate(tp, fp, fn, tn, zd) + specificityRate(tp, fp, fn, tn, zd) - 1,
  'higher',
  [-1, 1],
)

/**
 * Markedness: precision + negative predictive value - 1, in [-1, 1] (the counterpart of informedness).
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - average, labels, posLabel, zeroDivision and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { markedness } from 'mdml-engine/metrics'
 * markedness([0, 1, 1, 0], [0, 1, 1, 0]).estimate.value // => 1
 * ```
 */
export const markedness: Metric<AveragingOptions, MetricInput> = rateMetric(
  'markedness',
  () => (tp, fp, fn, tn, zd) => precisionRate(tp, fp, fn, tn, zd) + npvRate(tp, fp, fn, tn, zd) - 1,
  'higher',
  [-1, 1],
)

/** Options of {@link fbetaScore}. */
export interface FbetaOptions extends AveragingOptions {
  /** The weight of recall relative to precision (default 1). */
  readonly beta?: number
}

/**
 * F-beta score (1 + beta^2) tp / ((1 + beta^2) tp + beta^2 fn + fp) (sklearn.metrics.fbeta_score).
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - beta, average, labels, posLabel, zeroDivision and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { fbetaScore } from 'mdml-engine/metrics'
 * fbetaScore([0, 1, 1, 0], [1, 1, 0, 0], { beta: 1 }).estimate.value // => 0.5
 * ```
 */
export const fbetaScore: Metric<FbetaOptions, MetricInput> = rateMetric<FbetaOptions>('fbetaScore', (o) =>
  fbetaRate(o.beta ?? 1),
)

/**
 * F1 score, the harmonic mean of precision and recall (sklearn.metrics.f1_score).
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - average, labels, posLabel, zeroDivision and sample weights.
 * @returns The metric result.
 * @index function F1 score, with precision, recall and F-beta beside it.
 * @example
 * ```ts
 * import { f1Score } from 'mdml-engine/metrics'
 * f1Score([0, 1, 2, 2], [0, 2, 2, 2], { average: 'macro' }).estimate.value // => 0.6
 * ```
 */
export const f1Score: Metric<AveragingOptions, MetricInput> = rateMetric('f1Score', () => fbetaRate(1))

/**
 * Jaccard index tp / (tp + fp + fn) of each class (sklearn.metrics.jaccard_score), the intersection over union of the
 * predicted and true sets.
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - average, labels, posLabel, zeroDivision and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { jaccardScore } from 'mdml-engine/metrics'
 * jaccardScore([0, 1, 1, 0], [1, 1, 0, 0]).estimate.value // => 0.3333333333333333
 * ```
 */
export const jaccardScore: Metric<AveragingOptions, MetricInput> = rateMetric(
  'jaccardScore',
  () => (tp, fp, fn, _tn, zd) => ratio(tp, tp + fp + fn, zd),
)

/** Per-class precision, recall, F-score and support, with the requested average. */
export interface PrecisionRecallFscoreSupport {
  readonly labels: readonly Label[]
  readonly precision: readonly number[]
  readonly recall: readonly number[]
  readonly fscore: readonly number[]
  readonly support: readonly number[]
  /** The averaged precision, recall and F-score (absent for average 'none'). */
  readonly averaged?: { readonly precision: number; readonly recall: number; readonly fscore: number }
}

/**
 * Precision, recall, F-beta and support of every class at once (sklearn.metrics.precision_recall_fscore_support).
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - beta, average (default 'none'), labels, posLabel, zeroDivision and sample weights.
 * @returns Per-class arrays and the averages.
 * @example
 * ```ts
 * import { precisionRecallFscoreSupport } from 'mdml-engine/metrics'
 * precisionRecallFscoreSupport([0, 1, 1], [0, 1, 0]).recall // => [1, 0.5]
 * ```
 */
export function precisionRecallFscoreSupport(
  yTrue: MetricInput,
  yPred: MetricInput,
  options: FbetaOptions = {},
): PrecisionRecallFscoreSupport {
  const name = 'precisionRecallFscoreSupport'
  const c = countsOf(name, yTrue, yPred, options)
  const zd = options.zeroDivision ?? 0
  const k = c.classes.length
  const at = (rate: Rate): number[] =>
    Array.from({ length: k }, (_, i) => rate(c.tp[i] ?? 0, c.fp[i] ?? 0, c.fn[i] ?? 0, c.tn[i] ?? 0, zd))
  const average = options.average ?? 'none'
  const base = {
    labels: c.classes,
    precision: at(precisionRate),
    recall: at(recallRate),
    fscore: at(fbetaRate(options.beta ?? 1)),
    support: Array.from({ length: k }, (_, i) => (c.tp[i] ?? 0) + (c.fn[i] ?? 0)),
  }
  if (average === 'none') return base
  const opts = { ...options, average }
  const value = (m: Metric<FbetaOptions, MetricInput>): number => m(yTrue, yPred, opts).estimate.value
  return {
    ...base,
    averaged: { precision: value(precisionScore), recall: value(recallScore), fscore: value(fbetaScore) },
  }
}

/** Options of {@link accuracyScore}. */
export interface AccuracyOptions extends MetricOptions {
  /** Return the fraction (default) or the (weighted) number of correct samples. */
  readonly normalize?: boolean
}

/**
 * Accuracy (sklearn.metrics.accuracy_score): the weighted fraction of samples predicted exactly; for multilabel
 * indicators, the subset accuracy (every label of a sample right).
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - normalize and sample weights.
 * @returns The metric result.
 * @concept classification-metrics
 * @index function Classification accuracy, optionally weighted.
 * @example
 * ```ts
 * import { accuracyScore } from 'mdml-engine/metrics'
 * accuracyScore([0, 1, 2, 3], [0, 2, 1, 3]).estimate.value // => 0.5
 * ```
 */
export const accuracyScore: Metric<AccuracyOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yPred: MetricInput, options: AccuracyOptions = {}): MetricResult => {
    const name = 'accuracyScore'
    let correct: boolean[]
    if (isMultilabel(yTrue)) {
      const t = indicator(name, yTrue as TensorLike)
      const p = indicator(name, yPred as TensorLike)
      correct = Array.from({ length: t.n }, (_, i) => {
        for (let j = 0; j < t.l; j++) if ((t.data[i * t.l + j] ?? 0) !== (p.data[i * p.l + j] ?? 0)) return false
        return true
      })
    } else {
      const e = encodeLabels(name, yTrue, yPred)
      correct = Array.from(e.t, (a, i) => a === e.p[i])
    }
    const w = sampleWeights(name, correct.length, options.sampleWeight)
    let hit = 0
    let total = 0
    for (const [i, c] of correct.entries()) {
      total += w[i] ?? 0
      if (c) hit += w[i] ?? 0
    }
    const value = options.normalize === false ? hit : hit / total
    return metricResult(name, value, correct.length, { successes: hit, trials: total })
  },
  { spec: spec('accuracyScore', 'higher', [0, 1]) },
)

/**
 * Subset accuracy of multilabel predictions: the fraction of samples whose whole label set is right (the multilabel
 * case of {@link accuracyScore}).
 *
 * @param yTrue - True indicator `[N, L]`.
 * @param yPred - Predicted indicator `[N, L]`.
 * @param options - normalize and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { subsetAccuracy } from 'mdml-engine/metrics'
 * subsetAccuracy([[1, 0], [1, 1]], [[1, 0], [0, 1]]).estimate.value // => 0.5
 * ```
 */
export const subsetAccuracy: Metric<AccuracyOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yPred: MetricInput, options: AccuracyOptions = {}): MetricResult => ({
    ...accuracyScore(yTrue, yPred, options),
    metric: 'subsetAccuracy',
  }),
  { spec: spec('subsetAccuracy', 'higher', [0, 1]) },
)

/**
 * Balanced accuracy (sklearn.metrics.balanced_accuracy_score): the mean recall over classes (classes absent from the
 * targets are skipped), optionally adjusted for chance so that random guessing scores 0.
 *
 * @param yTrue - True labels.
 * @param yPred - Predicted labels.
 * @param options - adjusted and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { balancedAccuracyScore } from 'mdml-engine/metrics'
 * balancedAccuracyScore([0, 1, 0, 0], [0, 1, 0, 1]).estimate.value // => 0.8333333333333333
 * ```
 */
export const balancedAccuracyScore: Metric<MetricOptions & { readonly adjusted?: boolean }, MetricInput> =
  Object.assign(
    (
      yTrue: MetricInput,
      yPred: MetricInput,
      options: MetricOptions & { readonly adjusted?: boolean } = {},
    ): MetricResult => {
      const e = encodeLabels('balancedAccuracyScore', yTrue, yPred, options)
      const c = classCounts(e)
      const recalls: number[] = []
      for (let i = 0; i < e.classes.length; i++) {
        const support = (c.tp[i] ?? 0) + (c.fn[i] ?? 0)
        if (support > 0) recalls.push((c.tp[i] ?? 0) / support)
      }
      let value = recalls.reduce((s, v) => s + v, 0) / recalls.length
      if (options.adjusted === true) {
        const chance = 1 / recalls.length
        value = (value - chance) / (1 - chance)
      }
      return metricResult('balancedAccuracyScore', value, e.n)
    },
    { spec: spec('balancedAccuracyScore', 'higher', [0, 1]) },
  )

/** A confusion matrix with the classes labelling its rows (true) and columns (predicted). */
export interface ConfusionMatrix {
  readonly labels: readonly Label[]
  readonly matrix: readonly (readonly number[])[]
}

/**
 * Confusion matrix C[i][j]: the (weighted) number of samples of true class i predicted as j
 * (sklearn.metrics.confusion_matrix).
 *
 * @param yTrue - True labels.
 * @param yPred - Predicted labels.
 * @param options - labels (order and subset), normalize ('true' rows, 'pred' columns or 'all') and sample weights.
 * @returns The matrix and its class labels.
 * @index function Confusion matrix of labels and predictions.
 * @example
 * ```ts
 * import { confusionMatrix } from 'mdml-engine/metrics'
 * confusionMatrix([0, 1, 1, 2], [0, 1, 2, 2]).matrix // => [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
 * ```
 */
export function confusionMatrix(
  yTrue: MetricInput,
  yPred: MetricInput,
  options: MetricOptions & { readonly labels?: readonly Label[]; readonly normalize?: 'true' | 'pred' | 'all' } = {},
): ConfusionMatrix {
  const e = encodeLabels('confusionMatrix', yTrue, yPred, options)
  const k = e.classes.length
  const m = Array.from({ length: k }, () => new Array<number>(k).fill(0))
  for (let i = 0; i < e.n; i++) {
    const a = e.t[i] ?? -1
    const b = e.p[i] ?? -1
    const row = m[a]
    if (a >= 0 && b >= 0 && row !== undefined) row[b] = (row[b] ?? 0) + (e.w[i] ?? 0)
  }
  const safe = (a: number, b: number): number => (b === 0 ? 0 : a / b)
  if (options.normalize === 'true') {
    for (const row of m) {
      const s = row.reduce((x, y) => x + y, 0)
      for (let j = 0; j < k; j++) row[j] = safe(row[j] ?? 0, s)
    }
  } else if (options.normalize === 'pred') {
    for (let j = 0; j < k; j++) {
      const s = m.reduce((x, row) => x + (row[j] ?? 0), 0)
      for (const row of m) row[j] = safe(row[j] ?? 0, s)
    }
  } else if (options.normalize === 'all') {
    const s = m.reduce((x, row) => x + row.reduce((a, b) => a + b, 0), 0)
    for (const row of m) for (let j = 0; j < k; j++) row[j] = safe(row[j] ?? 0, s)
  }
  return { labels: e.classes, matrix: m }
}

/**
 * One-vs-rest 2 x 2 confusion matrices [[tn, fp], [fn, tp]] for every class or label
 * (sklearn.metrics.multilabel_confusion_matrix), or for every sample of multilabel data with `samplewise`.
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - labels, samplewise and sample weights.
 * @returns One matrix per class (or sample).
 * @example
 * ```ts
 * import { multilabelConfusionMatrix } from 'mdml-engine/metrics'
 * multilabelConfusionMatrix([0, 1, 1], [0, 1, 0])[1] // => [[1, 0], [1, 1]]
 * ```
 */
export function multilabelConfusionMatrix(
  yTrue: MetricInput,
  yPred: MetricInput,
  options: MetricOptions & { readonly labels?: readonly Label[]; readonly samplewise?: boolean } = {},
): number[][][] {
  const name = 'multilabelConfusionMatrix'
  if (options.samplewise === true) {
    if (!isMultilabel(yTrue)) throw new MdmlError('value', `${name}: samplewise needs multilabel targets`)
    const t = indicator(name, yTrue as TensorLike)
    const p = indicator(name, yPred as TensorLike)
    const w = sampleWeights(name, t.n, options.sampleWeight)
    return Array.from({ length: t.n }, (_, i) => {
      const counts = multilabelCounts(
        { n: 1, l: t.l, data: t.data.subarray(i * t.l, (i + 1) * t.l) },
        { n: 1, l: p.l, data: p.data.subarray(i * p.l, (i + 1) * p.l) },
        Float64Array.of(w[i] ?? 1),
        options.labels as readonly number[] | undefined,
      )
      const sum = (a: Float64Array): number => a.reduce((s, v) => s + v, 0)
      return [
        [sum(counts.tn), sum(counts.fp)],
        [sum(counts.fn), sum(counts.tp)],
      ]
    })
  }
  const c = countsOf(name, yTrue, yPred, options)
  return c.classes.map((_, i) => [
    [c.tn[i] ?? 0, c.fp[i] ?? 0],
    [c.fn[i] ?? 0, c.tp[i] ?? 0],
  ])
}
