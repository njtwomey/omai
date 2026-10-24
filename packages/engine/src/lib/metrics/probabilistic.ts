import { type MetricResult, type MetricSpec, WhiteboxError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { type Label, labelValues, sortLabels } from './_labels.ts'
import { binarise, type Grid } from './_scores.ts'
import {
  type Metric,
  type MetricInput,
  type MetricOptions,
  matrix,
  metricResult,
  sampleWeights,
  weightedMean,
} from './metric.ts'

const spec = (name: string, direction: 'higher' | 'lower', range: readonly [number, number]): MetricSpec => ({
  name,
  direction,
  range,
  inputs: 'probabilities',
  category: 'probabilistic',
  concept: 'proper-scoring-rules',
})

/** Options of the probabilistic classification metrics. */
export interface ProbabilityOptions extends MetricOptions {
  /** The class of each probability column, in order (default the sorted labels of yTrue). */
  readonly labels?: readonly Label[]
}

/** One-hot targets and an `[N, C]` probability matrix (a 1-D input is the positive-class probability). */
function probabilityTable(
  name: string,
  yTrue: MetricInput,
  yProba: TensorLike,
  labels: readonly Label[] | undefined,
): { t: Grid; p: Grid; n: number; classes: readonly Label[] } {
  const y = labelValues(yTrue)
  const raw = matrix(yProba)
  const classes = labels ?? sortLabels(y)
  let p: Grid = raw
  if (raw.shape.length === 1 || raw.shape[1] === 1) {
    const n = raw.shape[0] ?? 0
    const data = new Float64Array(2 * n)
    for (let i = 0; i < n; i++) {
      data[2 * i] = 1 - (raw.data[i] ?? 0)
      data[2 * i + 1] = raw.data[i] ?? 0
    }
    p = { shape: [n, 2], data }
  }
  for (const v of p.data)
    if (v < 0 || v > 1) throw new WhiteboxError('value', `${name}: probabilities must lie in [0, 1]`)
  if (classes.length !== (p.shape[1] ?? 0)) {
    throw new WhiteboxError('shape', `${name}: ${classes.length} classes but ${p.shape[1]} probability columns`)
  }
  return { t: binarise(y, classes), p, n: y.length, classes }
}

/** Per-sample cross-entropy -sum_c t_c log clip(p_c) with sklearn's clipping at machine epsilon. */
function sampleLogLoss(t: Grid, p: Grid): number[] {
  const n = t.shape[0] ?? 0
  const c = t.shape[1] ?? 0
  const eps = Number.EPSILON
  return Array.from({ length: n }, (_, i) => {
    let s = 0
    for (let j = 0; j < c; j++) {
      const tj = t.data[i * c + j] ?? 0
      if (tj !== 0) s -= tj * Math.log(Math.min(Math.max(p.data[i * c + j] ?? 0, eps), 1 - eps))
    }
    return s
  })
}

/**
 * Log loss, the cross-entropy of predicted probabilities (sklearn.metrics.log_loss), clipped at machine epsilon; a 1-D
 * prediction is the probability of the positive (larger) class.
 *
 * @param yTrue - True labels `[N]`.
 * @param yProba - Probabilities `[N, C]` (columns in label order) or `[N]`.
 * @param options - labels, normalize (default true; false sums) and sample weights.
 * @returns The metric result.
 * @concept proper-scoring-rules
 * @example
 * ```ts
 * import { logLoss } from 'whitebox-engine/metrics'
 * logLoss([0, 1], [[0.5, 0.5], [0.5, 0.5]]).estimate.value === Math.LN2 // => true
 * ```
 */
export const logLoss: Metric<ProbabilityOptions & { readonly normalize?: boolean }, MetricInput> = Object.assign(
  (
    yTrue: MetricInput,
    yProba: MetricInput,
    options: ProbabilityOptions & { readonly normalize?: boolean } = {},
  ): MetricResult => {
    const name = 'logLoss'
    const { t, p, n } = probabilityTable(name, yTrue, yProba as TensorLike, options.labels)
    const w = sampleWeights(name, n, options.sampleWeight)
    const losses = sampleLogLoss(t, p)
    const value =
      options.normalize === false ? losses.reduce((a, v, i) => a + v * (w[i] ?? 0), 0) : weightedMean(losses, w)
    return metricResult(name, value, n)
  },
  { spec: spec('logLoss', 'lower', [0, Infinity]) },
)

/**
 * D^2 for the log loss (sklearn.metrics.d2_log_loss_score): the fraction of the log loss of the class-prior forecast
 * that the predictions explain.
 *
 * @param yTrue - True labels `[N]`.
 * @param yProba - Probabilities `[N, C]` or `[N]`.
 * @param options - labels and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { d2LogLossScore } from 'whitebox-engine/metrics'
 * d2LogLossScore([0, 1], [[0.5, 0.5], [0.5, 0.5]]).estimate.value // => 0
 * ```
 */
export const d2LogLossScore: Metric<ProbabilityOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yProba: MetricInput, options: ProbabilityOptions = {}): MetricResult => {
    const name = 'd2LogLossScore'
    const { t, p, n } = probabilityTable(name, yTrue, yProba as TensorLike, options.labels)
    const w = sampleWeights(name, n, options.sampleWeight)
    const c = t.shape[1] ?? 0
    const prior = Array.from({ length: c }, (_, j) =>
      weightedMean(
        Array.from({ length: n }, (_, i) => t.data[i * c + j] ?? 0),
        w,
      ),
    )
    const nullP: Grid = { shape: t.shape, data: Float64Array.from({ length: n * c }, (_, k) => prior[k % c] ?? 0) }
    const num = sampleLogLoss(t, p).reduce((a, v, i) => a + v * (w[i] ?? 0), 0)
    const den = sampleLogLoss(t, nullP).reduce((a, v, i) => a + v * (w[i] ?? 0), 0)
    return metricResult(name, 1 - num / den, n)
  },
  { spec: spec('d2LogLossScore', 'higher', [-Infinity, 1]) },
)

/** Options of {@link brierScoreLoss}. */
export interface BrierOptions extends ProbabilityOptions {
  /** The positive class when yProba is 1-D (default the larger label). */
  readonly posLabel?: Label
  /** Halve the score (the binary convention): 'auto' (default) halves for fewer than three columns. */
  readonly scaleByHalf?: boolean | 'auto'
}

/**
 * Brier score (sklearn.metrics.brier_score_loss): the mean squared difference between one-hot targets and predicted
 * probabilities, summed over classes; halved by default for binary problems so that it equals mean (p - y)^2.
 *
 * @param yTrue - True labels `[N]`.
 * @param yProba - Probabilities `[N]` (positive class) or `[N, C]`.
 * @param options - posLabel, labels, scaleByHalf and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { brierScoreLoss } from 'whitebox-engine/metrics'
 * brierScoreLoss([0, 1, 1, 0], [0, 1, 0.5, 0.5]).estimate.value // => 0.125
 * ```
 */
export const brierScoreLoss: Metric<BrierOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yProba: MetricInput, options: BrierOptions = {}): MetricResult => {
    const name = 'brierScoreLoss'
    const raw = matrix(yProba as TensorLike)
    const scale = options.scaleByHalf ?? 'auto'
    const half = scale === 'auto' ? raw.shape.length === 1 || (raw.shape[1] ?? 0) < 3 : scale
    if (raw.shape.length === 1 || raw.shape[1] === 1) {
      // Binary: the two one-hot columns contribute 2 (y - p)^2, halved by default.
      const { y, p } = binaryForecast(yTrue, yProba as TensorLike, options.posLabel)
      const w = sampleWeights(name, y.length, options.sampleWeight)
      const sq = Array.from(y, (v, i) => 2 * (v - (p[i] ?? 0)) ** 2)
      return metricResult(name, weightedMean(sq, w) * (half ? 0.5 : 1), y.length)
    }
    const { t, p, n } = probabilityTable(name, yTrue, yProba as TensorLike, options.labels)
    const w = sampleWeights(name, n, options.sampleWeight)
    const c = t.shape[1] ?? 0
    const per = Array.from({ length: n }, (_, i) => {
      let s = 0
      for (let j = 0; j < c; j++) s += ((t.data[i * c + j] ?? 0) - (p.data[i * c + j] ?? 0)) ** 2
      return s
    })
    return metricResult(name, weightedMean(per, w) * (half ? 0.5 : 1), n)
  },
  { spec: spec('brierScoreLoss', 'lower', [0, 2]) },
)

/** Binned forecasts: the bin of each sample and the bin edges. */
function binIndices(prob: Float64Array, nBins: number, strategy: 'uniform' | 'quantile'): Int32Array {
  let edges: number[]
  if (strategy === 'quantile') {
    const sorted = Array.from(prob).sort((a, b) => a - b)
    // numpy.percentile with linear interpolation.
    edges = Array.from({ length: nBins + 1 }, (_, k) => {
      const pos = (k / nBins) * (sorted.length - 1)
      const lo = Math.floor(pos)
      const hi = Math.min(lo + 1, sorted.length - 1)
      return (sorted[lo] ?? 0) + (pos - lo) * ((sorted[hi] ?? 0) - (sorted[lo] ?? 0))
    })
  } else {
    edges = Array.from({ length: nBins + 1 }, (_, k) => k / nBins)
  }
  const inner = edges.slice(1, -1)
  // numpy.searchsorted(inner, p) with side 'left': the number of inner edges strictly below p.
  return Int32Array.from(prob, (v) => {
    let k = 0
    while (k < inner.length && (inner[k] ?? 0) < v) k++
    return k
  })
}

/** Binary targets and positive-class probabilities. */
function binaryForecast(yTrue: MetricInput, yProb: TensorLike, posLabel?: Label): { y: Float64Array; p: Float64Array } {
  const labels = labelValues(yTrue)
  const classes = sortLabels(labels)
  if (classes.length > 2) throw new WhiteboxError('value', 'needs binary labels')
  const pos = posLabel ?? (classes.includes(1) ? 1 : classes[classes.length - 1])
  const p = matrix(yProb).data
  if (p.length !== labels.length) throw new WhiteboxError('shape', 'yTrue and yProb have different lengths')
  return { y: Float64Array.from(labels, (l) => (l === pos ? 1 : 0)), p }
}

/** Options of the binned reliability summaries. */
export interface BinningOptions extends MetricOptions {
  /** The number of bins (default 5 for the curve, 10 for the decomposition). */
  readonly nBins?: number
  /** Equal-width ('uniform', default) or equal-mass ('quantile') bins. */
  readonly strategy?: 'uniform' | 'quantile'
  readonly posLabel?: Label
}

/** A reliability diagram: the observed frequency and mean forecast of each non-empty bin. */
export interface CalibrationCurve {
  readonly probTrue: readonly number[]
  readonly probPred: readonly number[]
  readonly counts: readonly number[]
}

/**
 * The calibration (reliability) curve (sklearn.calibration.calibration_curve): forecasts binned uniformly or by
 * quantiles, with the fraction of positives and the mean forecast of each non-empty bin.
 *
 * @param yTrue - True binary labels `[N]`.
 * @param yProb - Positive-class probabilities `[N]`.
 * @param options - nBins (default 5), strategy ('uniform' or 'quantile') and posLabel.
 * @returns The curve, with the sample count of each bin.
 * @concept calibration
 * @example
 * ```ts
 * import { calibrationCurve } from 'whitebox-engine/metrics'
 * calibrationCurve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], { nBins: 2 }).probTrue // => [0, 1]
 * ```
 */
export function calibrationCurve(
  yTrue: MetricInput,
  yProb: TensorLike,
  options: BinningOptions = {},
): CalibrationCurve {
  const { y, p } = binaryForecast(yTrue, yProb, options.posLabel)
  const nBins = options.nBins ?? 5
  const bins = binIndices(p, nBins, options.strategy ?? 'uniform')
  const sums = new Float64Array(nBins)
  const trues = new Float64Array(nBins)
  const counts = new Float64Array(nBins)
  for (const [i, b] of bins.entries()) {
    sums[b] = (sums[b] ?? 0) + (p[i] ?? 0)
    trues[b] = (trues[b] ?? 0) + (y[i] ?? 0)
    counts[b] = (counts[b] ?? 0) + 1
  }
  const nonEmpty = Array.from(counts.keys()).filter((b) => (counts[b] ?? 0) > 0)
  return {
    probTrue: nonEmpty.map((b) => (trues[b] ?? 0) / (counts[b] ?? 1)),
    probPred: nonEmpty.map((b) => (sums[b] ?? 0) / (counts[b] ?? 1)),
    counts: nonEmpty.map((b) => counts[b] ?? 0),
  }
}

/**
 * Murphy's decomposition of the Brier score of binary forecasts, BS ~ reliability - resolution + uncertainty, over
 * binned forecasts: reliability sum_k n_k (pbar_k - obar_k)^2 / N, resolution sum_k n_k (obar_k - obar)^2 / N and
 * uncertainty obar (1 - obar). With continuous forecasts the identity holds up to the within-bin variance terms.
 *
 * @param yTrue - True binary labels `[N]`.
 * @param yProb - Positive-class probabilities `[N]`.
 * @param options - nBins (default 10), strategy and posLabel.
 * @returns The Brier score (mean (p - y)^2), with the three components in the breakdown.
 * @example
 * ```ts
 * import { brierDecomposition } from 'whitebox-engine/metrics'
 * brierDecomposition([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]).breakdown?.uncertainty // => 0.25
 * ```
 */
export const brierDecomposition: Metric<BinningOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yProb: MetricInput, options: BinningOptions = {}): MetricResult => {
    const { y, p } = binaryForecast(yTrue, yProb as TensorLike, options.posLabel)
    const n = y.length
    const curve = calibrationCurve(yTrue, yProb as TensorLike, { ...options, nBins: options.nBins ?? 10 })
    const obar = y.reduce((a, b) => a + b, 0) / n
    let reliability = 0
    let resolution = 0
    for (const [k, count] of curve.counts.entries()) {
      reliability += (count * ((curve.probPred[k] ?? 0) - (curve.probTrue[k] ?? 0)) ** 2) / n
      resolution += (count * ((curve.probTrue[k] ?? 0) - obar) ** 2) / n
    }
    let brier = 0
    for (let i = 0; i < n; i++) brier += ((p[i] ?? 0) - (y[i] ?? 0)) ** 2 / n
    return metricResult('brierDecomposition', brier, n, { reliability, resolution, uncertainty: obar * (1 - obar) })
  },
  { spec: spec('brierDecomposition', 'lower', [0, 1]) },
)

/** Options of the calibration errors. */
export interface CalibrationErrorOptions extends MetricOptions {
  /** The number of bins (default 15). */
  readonly nBins?: number
  /** 'l1' (expected), 'l2' (root mean square) or 'max' (maximum) over bins (default 'l1'). */
  readonly norm?: 'l1' | 'l2' | 'max'
  /** 'uniform' bins on [0, 1] (default, torchmetrics) or 'quantile' (equal-mass, the adaptive ECE). */
  readonly strategy?: 'uniform' | 'quantile'
}

/** The calibration error of confidences against hits (torchmetrics' _ce_compute), with sample weights. */
function calibrationError(
  conf: Float64Array,
  hit: Float64Array,
  w: Float64Array,
  options: CalibrationErrorOptions,
): number {
  const nBins = options.nBins ?? 15
  const strategy = options.strategy ?? 'uniform'
  // torchmetrics' bucketize(right=True) - 1 over linspace(0, 1, n + 1): a confidence of exactly 1 has its own bin.
  const bins =
    strategy === 'uniform'
      ? Int32Array.from(conf, (c) => Math.min(Math.floor(c * nBins), nBins))
      : binIndices(conf, nBins, 'quantile')
  const size = nBins + 1
  const acc = new Float64Array(size)
  const cf = new Float64Array(size)
  const count = new Float64Array(size)
  for (const [i, b] of bins.entries()) {
    acc[b] = (acc[b] ?? 0) + (hit[i] ?? 0) * (w[i] ?? 0)
    cf[b] = (cf[b] ?? 0) + (conf[i] ?? 0) * (w[i] ?? 0)
    count[b] = (count[b] ?? 0) + (w[i] ?? 0)
  }
  const total = count.reduce((a, b) => a + b, 0)
  let l1 = 0
  let l2 = 0
  let mx = 0
  for (let b = 0; b < size; b++) {
    const nb = count[b] ?? 0
    if (nb === 0) continue
    const gap = Math.abs((acc[b] ?? 0) / nb - (cf[b] ?? 0) / nb)
    l1 += (gap * nb) / total
    l2 += (gap * gap * nb) / total
    mx = Math.max(mx, gap)
  }
  const norm = options.norm ?? 'l1'
  return norm === 'l1' ? l1 : norm === 'max' ? mx : Math.sqrt(l2)
}

/** Confidences and hits: binary forecasts as they are, multi-class as the top-label confidence and its correctness. */
function confidences(name: string, yTrue: MetricInput, yProba: TensorLike): { conf: Float64Array; hit: Float64Array } {
  const raw = matrix(yProba)
  if (raw.shape.length === 1) {
    const { y, p } = binaryForecast(yTrue, yProba)
    return { conf: p, hit: y }
  }
  const labels = labelValues(yTrue)
  const n = raw.shape[0] ?? 0
  const c = raw.shape[1] ?? 0
  if (labels.length !== n) throw new WhiteboxError('shape', `${name}: yTrue and yProba have different lengths`)
  const conf = new Float64Array(n)
  const hit = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let best = 0
    for (let j = 1; j < c; j++) if ((raw.data[i * c + j] ?? 0) > (raw.data[i * c + best] ?? 0)) best = j
    conf[i] = raw.data[i * c + best] ?? 0
    hit[i] = best === labels[i] ? 1 : 0
  }
  return { conf, hit }
}

/**
 * Expected calibration error (Naeini et al., 2015; torchmetrics' CalibrationError): the bin-weighted mean of
 * |accuracy - confidence| over confidence bins; for multi-class probabilities, of the top-label confidence (Guo et al.,
 * 2017). The 'l2' norm gives the RMS calibration error and 'quantile' bins the adaptive ECE.
 *
 * @param yTrue - True labels `[N]` (class indices for multi-class).
 * @param yProba - Positive-class probabilities `[N]` or class probabilities `[N, C]`.
 * @param options - nBins, norm, strategy and sample weights.
 * @returns The metric result.
 * @concept calibration
 * @example
 * ```ts
 * import { expectedCalibrationError } from 'whitebox-engine/metrics'
 * expectedCalibrationError([0, 1, 1, 0], [0.25, 0.75, 0.75, 0.25], { nBins: 4 }).estimate.value // => 0.25
 * ```
 */
export const expectedCalibrationError: Metric<CalibrationErrorOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yProba: MetricInput, options: CalibrationErrorOptions = {}): MetricResult => {
    const name = 'expectedCalibrationError'
    const { conf, hit } = confidences(name, yTrue, yProba as TensorLike)
    const w = sampleWeights(name, conf.length, options.sampleWeight)
    return metricResult(name, calibrationError(conf, hit, w, options), conf.length)
  },
  { spec: { ...spec('expectedCalibrationError', 'lower', [0, 1]), concept: 'calibration' } },
)

/**
 * Maximum calibration error: the largest |accuracy - confidence| over non-empty confidence bins (the 'max' norm of
 * {@link expectedCalibrationError}).
 *
 * @param yTrue - True labels `[N]`.
 * @param yProba - Probabilities `[N]` or `[N, C]`.
 * @param options - nBins, strategy and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { maximumCalibrationError } from 'whitebox-engine/metrics'
 * maximumCalibrationError([0, 1, 1, 0], [0.25, 0.75, 0.75, 0.25], { nBins: 4 }).estimate.value // => 0.25
 * ```
 */
export const maximumCalibrationError: Metric<Omit<CalibrationErrorOptions, 'norm'>, MetricInput> = Object.assign(
  (yTrue: MetricInput, yProba: MetricInput, options: Omit<CalibrationErrorOptions, 'norm'> = {}): MetricResult => ({
    ...expectedCalibrationError(yTrue, yProba, { ...options, norm: 'max' }),
    metric: 'maximumCalibrationError',
  }),
  { spec: { ...spec('maximumCalibrationError', 'lower', [0, 1]), concept: 'calibration' } },
)

/**
 * Classwise expected calibration error (Kull et al., 2019): the mean over classes of the binary calibration error of
 * each class's probability column against its one-vs-rest indicator.
 *
 * @param yTrue - True class indices `[N]`.
 * @param yProba - Class probabilities `[N, C]`.
 * @param options - nBins, norm, strategy and sample weights.
 * @returns The metric result, with the per-class errors in the breakdown.
 * @example
 * ```ts
 * import { classwiseCalibrationError } from 'whitebox-engine/metrics'
 * classwiseCalibrationError([0, 1], [[1, 0], [0, 1]]).estimate.value // => 0
 * ```
 */
export const classwiseCalibrationError: Metric<CalibrationErrorOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yProba: MetricInput, options: CalibrationErrorOptions = {}): MetricResult => {
    const name = 'classwiseCalibrationError'
    const raw = matrix(yProba as TensorLike)
    const labels = labelValues(yTrue)
    const n = raw.shape[0] ?? 0
    const c = raw.shape[1] ?? 0
    if (raw.shape.length !== 2) throw new WhiteboxError('shape', `${name}: needs [N, C] probabilities`)
    const w = sampleWeights(name, n, options.sampleWeight)
    const per = Array.from({ length: c }, (_, j) => {
      const conf = Float64Array.from({ length: n }, (_, i) => raw.data[i * c + j] ?? 0)
      const hit = Float64Array.from(labels, (l) => (l === j ? 1 : 0))
      return calibrationError(conf, hit, w, options)
    })
    return metricResult(
      name,
      per.reduce((a, b) => a + b, 0) / c,
      n,
      Object.fromEntries(per.map((v, j) => [String(j), v])),
    )
  },
  { spec: { ...spec('classwiseCalibrationError', 'lower', [0, 1]), concept: 'calibration' } },
)
