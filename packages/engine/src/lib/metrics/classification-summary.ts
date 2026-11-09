/**
 * Summary scores of a classifier beyond precision and recall: the Matthews correlation coefficient, Cohen's kappa
 * (agreement beyond chance, with a quadratic-weighted form for ordinal classes), likelihood ratios and the diagnostic
 * odds ratio, Hamming and zero-one loss, top-k accuracy, the multilabel ranking metrics, and scikit-learn's
 * `classificationReport`.
 */

import { type MetricResult, type MetricSpec, maxOf, minOf, OmaiError, toRows } from '../base/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { fixed, right } from './_format.ts'
import { classCounts, encodeLabels, indicator, isMultilabel, type Label, labelValues, sortLabels } from './_labels.ts'
import { specOf } from './_spec.ts'
import { type Average, accuracyScore, confusionMatrix, precisionRecallFscoreSupport } from './classification.ts'
import {
  type Metric,
  type MetricInput,
  type MetricOptions,
  metricResult,
  rowCount,
  sampleWeights,
  shapedArray,
  weightedMean,
} from './metric.ts'

/** The counts of {@link confusionMatrix} as rows of plain numbers, the form the loops of this file read. */
function confusionRows(yTrue: MetricInput, yPred: MetricInput, options: MetricOptions): number[][] {
  const { labels, matrix } = confusionMatrix(yTrue, yPred, options)
  return toRows(matrix.data, labels.length, labels.length)
}

const labelsSpec = specOf('classification', 'labels', 'classification-metrics')
const spec = (
  name: string,
  direction: 'higher' | 'lower',
  range: readonly [number, number],
  inputs: MetricSpec['inputs'] = 'labels',
): MetricSpec => ({ ...labelsSpec(name, direction, range), inputs })

/**
 * Matthews correlation coefficient (sklearn.metrics.matthews_corrcoef), in its multi-class form
 * (c s - sum_k p_k t_k) / sqrt((s^2 - sum p_k^2)(s^2 - sum t_k^2)) over the confusion matrix; 0 when undefined.
 *
 * @param yTrue - True labels.
 * @param yPred - Predicted labels.
 * @param options - Sample weights.
 * @returns The metric result, in [-1, 1].
 * @concept classification-metrics
 * @example
 * ```ts
 * import { matthewsCorrcoef } from 'omai-engine/metrics'
 * matthewsCorrcoef([1, 1, 0, 0], [1, 0, 0, 0]).estimate.value // => 0.5773502691896258
 * ```
 */
export const matthewsCorrcoef: Metric<MetricOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yPred: MetricInput, options: MetricOptions = {}): MetricResult => {
    const c = confusionRows(yTrue, yPred, options)
    const k = c.length
    const t = c.map((row) => row.reduce((a, b) => a + b, 0))
    const p = Array.from({ length: k }, (_, j) => c.reduce((a, row) => a + (row[j] ?? 0), 0))
    const correct = c.reduce((a, row, i) => a + (row[i] ?? 0), 0)
    const s = t.reduce((a, b) => a + b, 0)
    const cov = correct * s - p.reduce((a, pk, i) => a + pk * (t[i] ?? 0), 0)
    const covP = s * s - p.reduce((a, v) => a + v * v, 0)
    const covT = s * s - t.reduce((a, v) => a + v * v, 0)
    const value = covP * covT === 0 ? 0 : cov / Math.sqrt(covP * covT)
    return metricResult('matthewsCorrcoef', value, rowCount(yTrue))
  },
  { spec: spec('matthewsCorrcoef', 'higher', [-1, 1]) },
)

/** Options of {@link cohenKappaScore}. */
export interface KappaOptions extends MetricOptions {
  readonly labels?: readonly Label[]
  /** Disagreement weights: none (0/1), 'linear' |i - j| or 'quadratic' (i - j)^2 for ordinal classes. */
  readonly weights?: 'linear' | 'quadratic'
}

/**
 * Cohen's kappa (sklearn.metrics.cohen_kappa_score): agreement between two raters beyond chance,
 * 1 - sum w O / sum w E for observed O and chance-expected E confusion matrices, with optional linear or quadratic
 * disagreement weights for ordinal labels.
 *
 * @param y1 - First rater's labels.
 * @param y2 - Second rater's labels.
 * @param options - weights, labels and sample weights.
 * @returns The metric result, at most 1.
 * @example
 * ```ts
 * import { cohenKappaScore } from 'omai-engine/metrics'
 * cohenKappaScore([0, 1, 1, 0], [0, 1, 0, 0]).estimate.value // => 0.5
 * ```
 */
export const cohenKappaScore: Metric<KappaOptions, MetricInput> = Object.assign(
  (y1: MetricInput, y2: MetricInput, options: KappaOptions = {}): MetricResult => {
    const c = confusionRows(y1, y2, options)
    const k = c.length
    const rows = c.map((row) => row.reduce((a, b) => a + b, 0))
    const cols = Array.from({ length: k }, (_, j) => c.reduce((a, row) => a + (row[j] ?? 0), 0))
    const total = rows.reduce((a, b) => a + b, 0)
    let observed = 0
    let expected = 0
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        const w =
          options.weights === 'linear'
            ? Math.abs(i - j)
            : options.weights === 'quadratic'
              ? (i - j) ** 2
              : i === j
                ? 0
                : 1
        observed += w * (c[i]?.[j] ?? 0)
        expected += (w * (rows[i] ?? 0) * (cols[j] ?? 0)) / total
      }
    }
    return metricResult('cohenKappaScore', 1 - observed / expected, rowCount(y1))
  },
  { spec: spec('cohenKappaScore', 'higher', [-1, 1]) },
)

/**
 * Quadratic weighted kappa, the agreement measure for ordinal classes: {@link cohenKappaScore} with quadratic
 * disagreement weights.
 *
 * @param yTrue - True ordinal labels.
 * @param yPred - Predicted ordinal labels.
 * @param options - labels and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { quadraticWeightedKappa } from 'omai-engine/metrics'
 * quadraticWeightedKappa([0, 1, 2, 3], [0, 1, 2, 3]).estimate.value // => 1
 * ```
 */
export const quadraticWeightedKappa: Metric<Omit<KappaOptions, 'weights'>, MetricInput> = Object.assign(
  (yTrue: MetricInput, yPred: MetricInput, options: Omit<KappaOptions, 'weights'> = {}): MetricResult => ({
    ...cohenKappaScore(yTrue, yPred, { ...options, weights: 'quadratic' }),
    metric: 'quadraticWeightedKappa',
  }),
  { spec: spec('quadraticWeightedKappa', 'higher', [-1, 1]) },
)

/** Binary counts of the positive class. */
function binaryCounts(
  name: string,
  yTrue: MetricInput,
  yPred: MetricInput,
  options: MetricOptions & { readonly posLabel?: Label },
): { tp: number; fp: number; fn: number; tn: number } {
  const e = encodeLabels(name, yTrue, yPred, options)
  if (e.classes.length > 2) throw new OmaiError('value', `${name}: needs binary labels`)
  const i = e.classes.indexOf(options.posLabel ?? 1)
  const c = classCounts(e)
  if (i < 0) {
    const total = e.w.reduce((a, b) => a + b, 0)
    return { tp: 0, fp: 0, fn: 0, tn: total }
  }
  return { tp: c.tp[i] ?? 0, fp: c.fp[i] ?? 0, fn: c.fn[i] ?? 0, tn: c.tn[i] ?? 0 }
}

/** Options of the likelihood ratios. */
export interface LikelihoodRatioOptions extends MetricOptions {
  readonly posLabel?: Label
}

/**
 * Positive likelihood ratio LR+ = sensitivity / (1 - specificity) (sklearn.metrics.class_likelihood_ratios): how much a
 * positive prediction raises the odds of the condition. NaN when there are no false positives.
 *
 * @param yTrue - True binary labels.
 * @param yPred - Predicted binary labels.
 * @param options - posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { positiveLikelihoodRatio } from 'omai-engine/metrics'
 * positiveLikelihoodRatio([1, 1, 0, 0], [1, 0, 1, 0]).estimate.value // => 1
 * ```
 */
export const positiveLikelihoodRatio: Metric<LikelihoodRatioOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yPred: MetricInput, options: LikelihoodRatioOptions = {}): MetricResult => {
    const c = binaryCounts('positiveLikelihoodRatio', yTrue, yPred, options)
    const tpr = c.tp / (c.tp + c.fn)
    const fpr = c.fp / (c.fp + c.tn)
    return metricResult('positiveLikelihoodRatio', fpr === 0 ? Number.NaN : tpr / fpr, rowCount(yTrue))
  },
  { spec: spec('positiveLikelihoodRatio', 'higher', [0, Infinity]) },
)

/**
 * Negative likelihood ratio LR- = (1 - sensitivity) / specificity (sklearn.metrics.class_likelihood_ratios): how much
 * a negative prediction lowers the odds. NaN when there are no true negatives.
 *
 * @param yTrue - True binary labels.
 * @param yPred - Predicted binary labels.
 * @param options - posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { negativeLikelihoodRatio } from 'omai-engine/metrics'
 * negativeLikelihoodRatio([1, 1, 0, 0], [1, 0, 1, 0]).estimate.value // => 1
 * ```
 */
export const negativeLikelihoodRatio: Metric<LikelihoodRatioOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yPred: MetricInput, options: LikelihoodRatioOptions = {}): MetricResult => {
    const c = binaryCounts('negativeLikelihoodRatio', yTrue, yPred, options)
    const fnr = c.fn / (c.tp + c.fn)
    const tnr = c.tn / (c.fp + c.tn)
    return metricResult('negativeLikelihoodRatio', tnr === 0 ? Number.NaN : fnr / tnr, rowCount(yTrue))
  },
  { spec: spec('negativeLikelihoodRatio', 'lower', [0, Infinity]) },
)

/** Options of {@link classLikelihoodRatios}. */
export interface ClassLikelihoodRatioOptions extends LikelihoodRatioOptions {
  /**
   * What an ill-defined ratio becomes (scikit-learn's `replace_undefined_by`): `NaN` (default), `1` for both, or one
   * value for each: the positive ratio's from 1 up (or `NaN`), the negative ratio's in `[0, 1]` (or `NaN`).
   */
  readonly replaceUndefinedBy?: number | { readonly positive: number; readonly negative: number }
}

/**
 * The positive and negative likelihood ratios of a binary classifier together (sklearn.metrics.class_likelihood_ratios),
 * from the weighted confusion counts: LR+ = tp (tn + fp) / (fp (tp + fn)) and LR- = fn (tn + fp) / (tn (tp + fn)). The
 * positive ratio is undefined when there are no false positives, the negative one when there are no true negatives, and
 * both are `NaN` when `yTrue` has no positives; `replaceUndefinedBy` says what an undefined ratio becomes where its
 * own count is zero.
 *
 * @param yTrue - True binary labels.
 * @param yPred - Predicted binary labels.
 * @param options - posLabel, sample weights and replaceUndefinedBy.
 * @returns The positive and the negative ratio as metric results.
 * @throws OmaiError with code `value` for more than two labels, or a `replaceUndefinedBy` outside its allowed values.
 * @index function Positive and negative likelihood ratios of a binary classifier together.
 * @example
 * ```ts
 * import { classLikelihoodRatios } from 'omai-engine/metrics'
 * const { positive, negative } = classLikelihoodRatios([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0])
 * [positive.estimate.value.toFixed(3), negative.estimate.value.toFixed(3)] // => ['3.333', '0.417']
 * ```
 */
export function classLikelihoodRatios(
  yTrue: MetricInput,
  yPred: MetricInput,
  options: ClassLikelihoodRatioOptions = {},
): { readonly positive: MetricResult; readonly negative: MetricResult } {
  const name = 'classLikelihoodRatios'
  const r = options.replaceUndefinedBy ?? Number.NaN
  let desired: { readonly positive: number; readonly negative: number }
  if (typeof r === 'number') {
    if (!(Number.isNaN(r) || r === 1))
      throw new OmaiError('value', `${name}: replaceUndefinedBy must be NaN, 1 or a pair, got ${r}`)
    desired = { positive: r, negative: r }
  } else {
    if (
      !(Number.isNaN(r.positive) || r.positive >= 1) ||
      !(Number.isNaN(r.negative) || (r.negative >= 0 && r.negative <= 1))
    )
      throw new OmaiError(
        'value',
        `${name}: replaceUndefinedBy needs a positive ratio from 1 up and a negative one in [0, 1]`,
      )
    desired = r
  }
  const c = binaryCounts(name, yTrue, yPred, options)
  const supportPositive = c.tp + c.fn
  const supportNegative = c.tn + c.fp
  const positive = c.fp === 0 ? desired.positive : (c.tp * supportNegative) / (c.fp * supportPositive)
  const negative = c.tn === 0 ? desired.negative : (c.fn * supportNegative) / (c.tn * supportPositive)
  const n = rowCount(yTrue)
  return {
    positive: metricResult('positiveLikelihoodRatio', positive, n),
    negative: metricResult('negativeLikelihoodRatio', negative, n),
  }
}

/**
 * Diagnostic odds ratio LR+ / LR- = (tp tn) / (fp fn): the odds of a positive prediction among positives over the
 * same odds among negatives. Infinite when fp or fn is zero.
 *
 * @param yTrue - True binary labels.
 * @param yPred - Predicted binary labels.
 * @param options - posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { diagnosticOddsRatio } from 'omai-engine/metrics'
 * diagnosticOddsRatio([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0]).estimate.value // => 4
 * ```
 */
export const diagnosticOddsRatio: Metric<LikelihoodRatioOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yPred: MetricInput, options: LikelihoodRatioOptions = {}): MetricResult => {
    const c = binaryCounts('diagnosticOddsRatio', yTrue, yPred, options)
    const den = c.fp * c.fn
    const value = den === 0 ? (c.tp * c.tn === 0 ? Number.NaN : Number.POSITIVE_INFINITY) : (c.tp * c.tn) / den
    return metricResult('diagnosticOddsRatio', value, rowCount(yTrue))
  },
  { spec: spec('diagnosticOddsRatio', 'higher', [0, Infinity]) },
)

/**
 * Hamming loss (sklearn.metrics.hamming_loss): the (weighted) fraction of wrong labels, over samples and labels for
 * multilabel indicators.
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { hammingLoss } from 'omai-engine/metrics'
 * hammingLoss([[0, 1], [1, 1]], [[0, 0], [1, 1]]).estimate.value // => 0.25
 * ```
 */
export const hammingLoss: Metric<MetricOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yPred: MetricInput, options: MetricOptions = {}): MetricResult => {
    const name = 'hammingLoss'
    if (!isMultilabel(yTrue)) {
      return metricResult(name, 1 - accuracyScore(yTrue, yPred, options).estimate.value, rowCount(yTrue))
    }
    const t = indicator(name, yTrue as TensorLike)
    const p = indicator(name, yPred as TensorLike)
    const w = sampleWeights(name, t.n, options.sampleWeight)
    const wrong = Array.from({ length: t.n }, (_, i) => {
      let m = 0
      for (let j = 0; j < t.l; j++) if ((t.data[i * t.l + j] ?? 0) !== (p.data[i * p.l + j] ?? 0)) m++
      return m / t.l
    })
    return metricResult(name, weightedMean(wrong, w), t.n)
  },
  { spec: spec('hammingLoss', 'lower', [0, 1]) },
)

/**
 * Zero-one loss (sklearn.metrics.zero_one_loss): the fraction (or, unnormalised, the weighted count) of misclassified
 * samples; 1 - subset accuracy for multilabel data.
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - normalize and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { zeroOneLoss } from 'omai-engine/metrics'
 * zeroOneLoss([1, 2, 3, 4], [2, 2, 3, 4]).estimate.value // => 0.25
 * ```
 */
export const zeroOneLoss: Metric<MetricOptions & { readonly normalize?: boolean }, MetricInput> = Object.assign(
  (
    yTrue: MetricInput,
    yPred: MetricInput,
    options: MetricOptions & { readonly normalize?: boolean } = {},
  ): MetricResult => {
    const acc = accuracyScore(yTrue, yPred, options)
    const total = acc.breakdown?.['trials'] ?? 0
    const hits = acc.breakdown?.['successes'] ?? 0
    const value = options.normalize === false ? total - hits : 1 - hits / total
    return metricResult('zeroOneLoss', value, acc.n)
  },
  { spec: spec('zeroOneLoss', 'lower', [0, 1]) },
)

/** Options of {@link topKAccuracyScore}. */
export interface TopKOptions extends MetricOptions {
  /** How many of the highest-scoring classes count as a hit (default 2). */
  readonly k?: number
  /** The class of each score column (default 0..C-1, or the sorted labels). */
  readonly labels?: readonly Label[]
  readonly normalize?: boolean
}

/**
 * Top-k accuracy (sklearn.metrics.top_k_accuracy_score): the fraction of samples whose true class is among the k
 * highest scores. Binary problems take one score per sample (of the positive class).
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N, C]` (or `[N]` for binary).
 * @param options - k, labels, normalize and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { topKAccuracyScore } from 'omai-engine/metrics'
 * topKAccuracyScore([0, 1, 2, 2], [[0.5, 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]).estimate.value // => 0.75
 * ```
 */
export const topKAccuracyScore: Metric<TopKOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yScore: MetricInput, options: TopKOptions = {}): MetricResult => {
    const name = 'topKAccuracyScore'
    const t = labelValues(yTrue)
    const s = shapedArray(yScore as TensorLike)
    const k = options.k ?? 2
    const w = sampleWeights(name, t.length, options.sampleWeight)
    const hits: number[] = []
    if (s.shape.length === 1) {
      // Binary: with k = 1 the prediction is score > 0.5 (probabilities) or > 0 (decision values); k >= 2 always hits.
      const classes = options.labels ?? sortLabels(t)
      const threshold = minOf(s.data) >= 0 && maxOf(s.data) <= 1 ? 0.5 : 0
      for (const [i, label] of t.entries()) {
        const positive = label === classes[classes.length - 1]
        const predicted = (s.data[i] ?? 0) > threshold
        hits.push(k >= 2 || predicted === positive ? 1 : 0)
      }
    } else {
      const c = s.shape[1] ?? 0
      const classes =
        options.labels ??
        (t.every((v) => typeof v === 'number') ? Array.from({ length: c }, (_, j) => j) : sortLabels(t))
      for (const [i, label] of t.entries()) {
        const j = classes.indexOf(label)
        const row = Array.from(s.data.subarray(i * c, (i + 1) * c))
        // sklearn takes the last k of a stable ascending argsort: among ties, later classes rank higher.
        const order = row.map((_, q) => q).sort((a, b) => (row[a] ?? 0) - (row[b] ?? 0) || a - b)
        hits.push(order.slice(Math.max(0, c - k)).includes(j) ? 1 : 0)
      }
    }
    const value = options.normalize === false ? hits.reduce((a, h, i) => a + h * (w[i] ?? 0), 0) : weightedMean(hits, w)
    return metricResult(name, value, t.length)
  },
  { spec: spec('topKAccuracyScore', 'higher', [0, 1], 'scores') },
)

/** Rows of multilabel targets and scores. */
function rankingRows(
  name: string,
  yTrue: TensorLike,
  yScore: TensorLike,
): { n: number; l: number; t: Float64Array; s: Float64Array } {
  const t = indicator(name, yTrue)
  const s = indicator(name, yScore)
  if (t.n !== s.n || t.l !== s.l) throw new OmaiError('shape', `${name}: yTrue and yScore shapes differ`)
  return { n: t.n, l: t.l, t: t.data, s: s.data }
}

/**
 * Label-ranking average precision (sklearn.metrics.label_ranking_average_precision_score): for each relevant label,
 * the fraction of labels ranked at or above it that are relevant, averaged per sample and over samples. Samples whose
 * labels are all or none relevant score 1.
 *
 * @param yTrue - Relevance indicator `[N, L]`.
 * @param yScore - Label scores `[N, L]`.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { labelRankingAveragePrecisionScore } from 'omai-engine/metrics'
 * labelRankingAveragePrecisionScore([[1, 0, 0], [0, 0, 1]], [[0.75, 0.5, 1], [1, 0.2, 0.1]]).estimate.value // => 0.41666666666666663
 * ```
 */
export const labelRankingAveragePrecisionScore: Metric<MetricOptions> = Object.assign(
  (yTrue: TensorLike, yScore: TensorLike, options: MetricOptions = {}): MetricResult => {
    const name = 'labelRankingAveragePrecisionScore'
    const r = rankingRows(name, yTrue, yScore)
    const w = sampleWeights(name, r.n, options.sampleWeight)
    const scores = Array.from({ length: r.n }, (_, i) => {
      const t = r.t.subarray(i * r.l, (i + 1) * r.l)
      const s = r.s.subarray(i * r.l, (i + 1) * r.l)
      const relevant = Array.from(t.keys()).filter((j) => (t[j] ?? 0) > 0)
      if (relevant.length === 0 || relevant.length === r.l) return 1
      let total = 0
      for (const j of relevant) {
        const sj = s[j] ?? 0
        let atOrAbove = 0
        let relevantAtOrAbove = 0
        for (let q = 0; q < r.l; q++) {
          if ((s[q] ?? 0) >= sj) {
            atOrAbove++
            if ((t[q] ?? 0) > 0) relevantAtOrAbove++
          }
        }
        total += relevantAtOrAbove / atOrAbove
      }
      return total / relevant.length
    })
    return metricResult(name, weightedMean(scores, w), r.n)
  },
  { spec: spec('labelRankingAveragePrecisionScore', 'higher', [0, 1], 'scores') },
)

/**
 * Coverage error (sklearn.metrics.coverage_error): how far down the ranked labels one must go, on average, to cover
 * every relevant label (ties count against). Samples without relevant labels contribute 0.
 *
 * @param yTrue - Relevance indicator `[N, L]`.
 * @param yScore - Label scores `[N, L]`.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { coverageError } from 'omai-engine/metrics'
 * coverageError([[1, 0, 0], [0, 1, 1]], [[1, 0, 0], [0, 1, 1]]).estimate.value // => 1.5
 * ```
 */
export const coverageError: Metric<MetricOptions> = Object.assign(
  (yTrue: TensorLike, yScore: TensorLike, options: MetricOptions = {}): MetricResult => {
    const name = 'coverageError'
    const r = rankingRows(name, yTrue, yScore)
    const w = sampleWeights(name, r.n, options.sampleWeight)
    const cover = Array.from({ length: r.n }, (_, i) => {
      let lowest = Number.POSITIVE_INFINITY
      for (let j = 0; j < r.l; j++) if ((r.t[i * r.l + j] ?? 0) > 0) lowest = Math.min(lowest, r.s[i * r.l + j] ?? 0)
      if (lowest === Number.POSITIVE_INFINITY) return 0
      let count = 0
      for (let j = 0; j < r.l; j++) if ((r.s[i * r.l + j] ?? 0) >= lowest) count++
      return count
    })
    return metricResult(name, weightedMean(cover, w), r.n)
  },
  { spec: spec('coverageError', 'lower', [1, Infinity], 'scores') },
)

/**
 * Label-ranking loss (sklearn.metrics.label_ranking_loss): the fraction of (relevant, irrelevant) label pairs ordered
 * wrongly (the irrelevant label scored at least as high), per sample, averaged.
 *
 * @param yTrue - Relevance indicator `[N, L]`.
 * @param yScore - Label scores `[N, L]`.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { labelRankingLoss } from 'omai-engine/metrics'
 * labelRankingLoss([[1, 0, 0], [0, 0, 1]], [[0.75, 0.5, 1], [1, 0.2, 0.1]]).estimate.value // => 0.75
 * ```
 */
export const labelRankingLoss: Metric<MetricOptions> = Object.assign(
  (yTrue: TensorLike, yScore: TensorLike, options: MetricOptions = {}): MetricResult => {
    const name = 'labelRankingLoss'
    const r = rankingRows(name, yTrue, yScore)
    const w = sampleWeights(name, r.n, options.sampleWeight)
    const losses = Array.from({ length: r.n }, (_, i) => {
      let wrong = 0
      let positives = 0
      for (let a = 0; a < r.l; a++) {
        if ((r.t[i * r.l + a] ?? 0) <= 0) continue
        positives++
        for (let b = 0; b < r.l; b++) {
          if ((r.t[i * r.l + b] ?? 0) <= 0 && (r.s[i * r.l + b] ?? 0) >= (r.s[i * r.l + a] ?? 0)) wrong++
        }
      }
      const pairs = positives * (r.l - positives)
      return pairs === 0 ? 0 : wrong / pairs
    })
    return metricResult(name, weightedMean(losses, w), r.n)
  },
  { spec: spec('labelRankingLoss', 'lower', [0, 1], 'scores') },
)

/** One row of a classification report. */
export interface ReportRow {
  readonly precision: number
  readonly recall: number
  readonly f1: number
  readonly support: number
}

/** The per-class rows, averages and printable text of {@link classificationReport}. */
export interface ClassificationReport {
  readonly classes: Readonly<Record<string, ReportRow>>
  /** Accuracy, when micro averaging equals it (multi-class or binary data scored on every class). */
  readonly accuracy?: number
  readonly averages: Readonly<Record<string, ReportRow>>
  /** sklearn's text layout. */
  readonly text: string
}

/**
 * Classification report (sklearn.metrics.classification_report): precision, recall, F1 and support per class, with
 * accuracy and macro and weighted averages (and micro and samples averages where they apply), as data and as
 * sklearn's text table.
 *
 * @param yTrue - True labels or a multilabel indicator.
 * @param yPred - Predicted labels, the same form.
 * @param options - labels, targetNames, digits (default 2), zeroDivision and sample weights.
 * @returns The report.
 * @example
 * ```ts
 * import { classificationReport } from 'omai-engine/metrics'
 * classificationReport([0, 1, 2, 2, 2], [0, 0, 2, 2, 1]).accuracy // => 0.6
 * ```
 */
export function classificationReport(
  yTrue: MetricInput,
  yPred: MetricInput,
  options: MetricOptions & {
    readonly labels?: readonly Label[]
    readonly targetNames?: readonly string[]
    readonly digits?: number
    readonly zeroDivision?: number
  } = {},
): ClassificationReport {
  const multilabel = isMultilabel(yTrue)
  const base = { ...options, average: 'none' as const }
  const prfs = precisionRecallFscoreSupport(yTrue, yPred, base)
  const names = options.targetNames ?? prfs.labels.map(String)
  const [precision, recall, fscore, support] = [prfs.precision, prfs.recall, prfs.fscore, prfs.support].map(toFlat)
  const classes: Record<string, ReportRow> = {}
  for (const [i, name] of names.entries()) {
    classes[name] = {
      precision: precision?.[i] ?? 0,
      recall: recall?.[i] ?? 0,
      f1: fscore?.[i] ?? 0,
      support: support?.[i] ?? 0,
    }
  }
  const totalSupport = (support ?? []).reduce((a, b) => a + b, 0)
  const seen = multilabel ? [] : sortLabels(labelValues(yTrue).concat(labelValues(yPred)))
  const microIsAccuracy =
    !multilabel && (options.labels === undefined || seen.every((l) => (options.labels ?? []).includes(l)))
  const averages: Record<string, ReportRow> = {}
  const kinds: Average[] = ['micro', 'macro', 'weighted', ...(multilabel ? (['samples'] as const) : [])]
  let accuracy: number | undefined
  for (const kind of kinds) {
    const r = precisionRecallFscoreSupport(yTrue, yPred, { ...options, average: kind })
    const row = {
      precision: r.averaged?.precision ?? 0,
      recall: r.averaged?.recall ?? 0,
      f1: r.averaged?.fscore ?? 0,
      support: totalSupport,
    }
    if (kind === 'micro' && microIsAccuracy) accuracy = row.f1
    else averages[`${kind} avg`] = row
  }
  const digits = options.digits ?? 2
  const width = maxOf(names.map((n) => n.length).concat(['weighted avg'.length, digits]))
  // Weighted supports are floats, which Python prints with a decimal point.
  const weighted = options.sampleWeight !== undefined
  const supportText = (s: number): string => (weighted && Number.isInteger(s) ? `${s}.0` : String(s))
  const line = (name: string, r: ReportRow): string =>
    `${right(name, width)} ${[r.precision, r.recall, r.f1].map((v) => ` ${right(fixed(v, digits), 9)}`).join('')} ${right(supportText(r.support), 9)}\n`
  let text = `${right('', width)} ${['precision', 'recall', 'f1-score', 'support'].map((h) => ` ${right(h, 9)}`).join('')}\n\n`
  for (const name of names) text += line(name, classes[name] as ReportRow)
  text += '\n'
  if (accuracy !== undefined) {
    text += `${right('accuracy', width)} ${` ${right('', 9)}`.repeat(2)} ${right(fixed(accuracy, digits), 9)} ${right(supportText(totalSupport), 9)}\n`
  }
  for (const [name, row] of Object.entries(averages)) text += line(name, row)
  return { classes, ...(accuracy === undefined ? {} : { accuracy }), averages, text }
}
