import { type MetricResult, type MetricSpec, WhiteboxError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { labelValues } from './_labels.ts'
import { type CurveOptions, confusionMatrixAtThresholds, rocCurve } from './curves.ts'
import { type Metric, type MetricInput, metricResult } from './metric.ts'
import { precisionRecallCurve } from './precision-recall.ts'

const spec = (
  name: string,
  direction: 'higher' | 'lower' = 'higher',
  range: readonly [number, number] = [0, 1],
): MetricSpec => ({
  name,
  direction,
  range,
  inputs: 'scores',
  category: 'classification',
  concept: 'operating-points',
})

/** The ROC points without dropping intermediate thresholds. */
const roc = (yTrue: MetricInput, yScore: MetricInput, options: CurveOptions) =>
  rocCurve(yTrue, yScore as TensorLike, { ...options, dropIntermediate: false })

const at = (name: string, value: number, threshold: number, yTrue: MetricInput): MetricResult =>
  metricResult(name, value, labelValues(yTrue).length, { threshold })

/** Options of the fixed-rate operating points. */
export interface FixedRateOptions extends CurveOptions {
  /** The constraint: the minimum rate the operating point must reach. */
  readonly min: number
}

/**
 * The highest recall among thresholds whose precision is at least `min` (torchmetrics'
 * binary_recall_at_fixed_precision); the threshold goes in the breakdown (NaN when no threshold qualifies).
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - min (the precision floor), posLabel and sample weights.
 * @returns The metric result.
 * @concept operating-points
 * @example
 * ```ts
 * import { recallAtFixedPrecision } from 'whitebox-engine/metrics'
 * recallAtFixedPrecision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], { min: 0.6 }).estimate.value // => 1
 * ```
 */
export const recallAtFixedPrecision: Metric<FixedRateOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yScore: MetricInput, options: FixedRateOptions = { min: 0 }): MetricResult => {
    const pr = precisionRecallCurve(yTrue, yScore as TensorLike, options)
    let best: [number, number, number] | undefined
    for (const [i, t] of pr.thresholds.entries()) {
      const p = pr.precision[i] ?? 0
      const r = pr.recall[i] ?? 0
      if (p < options.min) continue
      const cand: [number, number, number] = [r, p, t]
      if (
        best === undefined ||
        cand[0] > best[0] ||
        (cand[0] === best[0] && (cand[1] > best[1] || (cand[1] === best[1] && cand[2] > best[2])))
      )
        best = cand
    }
    const recall = best?.[0] ?? 0
    return at('recallAtFixedPrecision', recall, recall === 0 ? Number.NaN : (best?.[2] ?? Number.NaN), yTrue)
  },
  { spec: spec('recallAtFixedPrecision') },
)

/**
 * The highest precision among thresholds whose recall is at least `min` (torchmetrics'
 * binary_precision_at_fixed_recall); the threshold goes in the breakdown.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - min (the recall floor), posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { precisionAtFixedRecall } from 'whitebox-engine/metrics'
 * precisionAtFixedRecall([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], { min: 1 }).estimate.value // => 0.6666666666666666
 * ```
 */
export const precisionAtFixedRecall: Metric<FixedRateOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yScore: MetricInput, options: FixedRateOptions = { min: 0 }): MetricResult => {
    const pr = precisionRecallCurve(yTrue, yScore as TensorLike, options)
    let best: [number, number, number] | undefined
    for (const [i, t] of pr.thresholds.entries()) {
      const p = pr.precision[i] ?? 0
      const r = pr.recall[i] ?? 0
      if (r < options.min) continue
      const cand: [number, number, number] = [p, r, t]
      if (
        best === undefined ||
        cand[0] > best[0] ||
        (cand[0] === best[0] && (cand[1] > best[1] || (cand[1] === best[1] && cand[2] > best[2])))
      )
        best = cand
    }
    const precision = best?.[0] ?? 0
    return at('precisionAtFixedRecall', precision, precision === 0 ? Number.NaN : (best?.[2] ?? Number.NaN), yTrue)
  },
  { spec: spec('precisionAtFixedRecall') },
)

/**
 * The highest sensitivity (TPR) among thresholds whose specificity (1 - FPR) is at least `min` (torchmetrics'
 * binary_sensitivity_at_specificity): the true positive rate at a fixed false positive rate.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - min (the specificity floor), posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { sensitivityAtSpecificity } from 'whitebox-engine/metrics'
 * sensitivityAtSpecificity([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], { min: 0.5 }).estimate.value // => 1
 * ```
 */
export const sensitivityAtSpecificity: Metric<FixedRateOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yScore: MetricInput, options: FixedRateOptions = { min: 0 }): MetricResult => {
    const r = roc(yTrue, yScore, options)
    let best = -1
    for (const [i, f] of r.fpr.entries())
      if (1 - f >= options.min && (best < 0 || (r.tpr[i] ?? 0) > (r.tpr[best] ?? 0))) best = i
    if (best < 0) return at('sensitivityAtSpecificity', 0, 1e6, yTrue)
    return at('sensitivityAtSpecificity', r.tpr[best] ?? 0, firstThreshold(r.thresholds[best] ?? 0), yTrue)
  },
  { spec: spec('sensitivityAtSpecificity') },
)

/**
 * The highest specificity (1 - FPR) among thresholds whose sensitivity (TPR) is at least `min` (torchmetrics'
 * binary_specificity_at_sensitivity).
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - min (the sensitivity floor), posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { specificityAtSensitivity } from 'whitebox-engine/metrics'
 * specificityAtSensitivity([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], { min: 1 }).estimate.value // => 0.5
 * ```
 */
export const specificityAtSensitivity: Metric<FixedRateOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yScore: MetricInput, options: FixedRateOptions = { min: 0 }): MetricResult => {
    const r = roc(yTrue, yScore, options)
    let best = -1
    for (const [i, t] of r.tpr.entries())
      if (t >= options.min && (best < 0 || 1 - (r.fpr[i] ?? 0) > 1 - (r.fpr[best] ?? 0))) best = i
    if (best < 0) return at('specificityAtSensitivity', 0, 1e6, yTrue)
    return at('specificityAtSensitivity', 1 - (r.fpr[best] ?? 0), firstThreshold(r.thresholds[best] ?? 0), yTrue)
  },
  { spec: spec('specificityAtSensitivity') },
)

/** torchmetrics puts the always-negative threshold at 1, where sklearn puts it at +inf. */
const firstThreshold = (t: number): number => (t === Number.POSITIVE_INFINITY ? 1 : t)

/**
 * The false positive rate at the first threshold whose true positive rate reaches `min` (for example FPR at 95% TPR,
 * the usual out-of-distribution detection score), with the threshold in the breakdown.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - min (the TPR to reach, default 0.95), posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { fprAtTpr } from 'whitebox-engine/metrics'
 * fprAtTpr([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], { min: 0.95 }).estimate.value // => 0.5
 * ```
 */
export const fprAtTpr: Metric<FixedRateOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yScore: MetricInput, options: FixedRateOptions = { min: 0.95 }): MetricResult => {
    const r = roc(yTrue, yScore, options)
    const i = r.tpr.findIndex((t) => t >= options.min)
    return at('fprAtTpr', r.fpr[i] ?? 1, r.thresholds[i] ?? Number.NaN, yTrue)
  },
  { spec: spec('fprAtTpr', 'lower') },
)

/**
 * The equal error rate: the error where the false positive rate equals the false negative rate. 'interpolated'
 * (default) intersects the ROC polyline with FPR = 1 - TPR exactly (the brentq-on-interp1d recipe of speaker
 * verification); 'nearest' takes the threshold minimising |FPR - FNR| and averages the two rates (torchmetrics'
 * binary_eer).
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - method, posLabel and sample weights.
 * @returns The metric result (with the threshold for 'nearest').
 * @example
 * ```ts
 * import { equalErrorRate } from 'whitebox-engine/metrics'
 * equalErrorRate([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).estimate.value // => 0.5
 * ```
 */
export const equalErrorRate: Metric<CurveOptions & { readonly method?: 'interpolated' | 'nearest' }, MetricInput> =
  Object.assign(
    (
      yTrue: MetricInput,
      yScore: MetricInput,
      options: CurveOptions & { readonly method?: 'interpolated' | 'nearest' } = {},
    ): MetricResult => {
      const r = roc(yTrue, yScore, options)
      if (options.method === 'nearest') {
        let best = 0
        for (let i = 1; i < r.fpr.length; i++) {
          if (
            Math.abs((r.fpr[i] ?? 0) - (1 - (r.tpr[i] ?? 0))) < Math.abs((r.fpr[best] ?? 0) - (1 - (r.tpr[best] ?? 0)))
          )
            best = i
        }
        const value = ((r.fpr[best] ?? 0) + (1 - (r.tpr[best] ?? 0))) / 2
        return at('equalErrorRate', value, firstThreshold(r.thresholds[best] ?? 0), yTrue)
      }
      // g = TPR - (1 - FPR) rises from -1 to 1 along the curve; find where it crosses zero.
      const g = r.fpr.map((f, i) => (r.tpr[i] ?? 0) - (1 - f))
      for (let i = 1; i < g.length; i++) {
        const a = g[i - 1] ?? 0
        const b = g[i] ?? 0
        if (a <= 0 && b >= 0) {
          const t = b === a ? 0 : -a / (b - a)
          const fpr = (r.fpr[i - 1] ?? 0) + t * ((r.fpr[i] ?? 0) - (r.fpr[i - 1] ?? 0))
          return metricResult('equalErrorRate', fpr, labelValues(yTrue).length)
        }
      }
      throw new WhiteboxError('value', 'equalErrorRate: the ROC curve does not cross FPR = FNR')
    },
    { spec: spec('equalErrorRate', 'lower') },
  )

/**
 * Youden's index J = max over thresholds of TPR - FPR (informedness at the best threshold), with that threshold in
 * the breakdown.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { youdenIndex } from 'whitebox-engine/metrics'
 * youdenIndex([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).breakdown?.threshold // => 0.8
 * ```
 */
export const youdenIndex: Metric<CurveOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yScore: MetricInput, options: CurveOptions = {}): MetricResult => {
    const r = roc(yTrue, yScore, options)
    let best = 0
    for (let i = 1; i < r.fpr.length; i++) {
      if ((r.tpr[i] ?? 0) - (r.fpr[i] ?? 0) > (r.tpr[best] ?? 0) - (r.fpr[best] ?? 0)) best = i
    }
    return at('youdenIndex', (r.tpr[best] ?? 0) - (r.fpr[best] ?? 0), r.thresholds[best] ?? 0, yTrue)
  },
  { spec: spec('youdenIndex', 'higher', [-1, 1]) },
)

/** Options of {@link minimumExpectedCost}. */
export interface CostOptions extends CurveOptions {
  /** The cost of a false positive (false alarm; default 1). */
  readonly costFp?: number
  /** The cost of a false negative (miss; default 1). */
  readonly costFn?: number
  /** The prior probability of the positive class (default its weighted frequency in yTrue). */
  readonly prior?: number
  /** Divide by the cost of the better trivial system, min(pi c_fn, (1 - pi) c_fp) (default true). */
  readonly normalize?: boolean
}

/**
 * The minimum expected cost over thresholds, pi c_fn FNR + (1 - pi) c_fp FPR (the minimum detection cost function of
 * NIST speaker recognition evaluations, minDCF), normalised by the cost of the better trivial system; the threshold
 * goes in the breakdown.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - costFp, costFn, prior, normalize, posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { minimumExpectedCost } from 'whitebox-engine/metrics'
 * minimumExpectedCost([0, 1], [0.2, 0.9]).estimate.value // => 0
 * ```
 */
export const minimumExpectedCost: Metric<CostOptions, MetricInput> = Object.assign(
  (yTrue: MetricInput, yScore: MetricInput, options: CostOptions = {}): MetricResult => {
    const c = confusionMatrixAtThresholds(yTrue, yScore as TensorLike, options)
    const nP = c.tps[c.tps.length - 1] ?? 0
    const nN = c.fps[c.fps.length - 1] ?? 0
    const pi = options.prior ?? nP / (nP + nN)
    const cFp = options.costFp ?? 1
    const cFn = options.costFn ?? 1
    // Every threshold, plus the always-negative system (FNR 1, FPR 0).
    let best = pi * cFn
    let threshold = Number.POSITIVE_INFINITY
    for (const [i, tp] of c.tps.entries()) {
      const cost = pi * cFn * (1 - tp / nP) + (1 - pi) * cFp * ((c.fps[i] ?? 0) / nN)
      if (cost < best) {
        best = cost
        threshold = c.thresholds[i] ?? 0
      }
    }
    const value = (options.normalize ?? true) ? best / Math.min(pi * cFn, (1 - pi) * cFp) : best
    return at('minimumExpectedCost', value, threshold, yTrue)
  },
  { spec: spec('minimumExpectedCost', 'lower', [0, Infinity]) },
)

/** A cost curve: normalised expected cost against the probability cost of the positive class. */
export interface CostCurve {
  readonly probabilityCost: readonly number[]
  readonly normalizedExpectedCost: readonly number[]
}

/**
 * The cost curve (Drummond and Holte, 2006): for each probability cost PC(+) in [0, 1], the lowest normalised expected
 * cost over the ROC's operating points, min_i (1 - TPR_i) PC + FPR_i (1 - PC), the lower envelope of the lines of the
 * operating points. It is the point-line dual of the ROC curve.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param options - points on the PC axis (default 101), posLabel and sample weights.
 * @returns The curve.
 * @example
 * ```ts
 * import { costCurve } from 'whitebox-engine/metrics'
 * costCurve([0, 1], [0.2, 0.9], { points: 3 }).normalizedExpectedCost // => [0, 0, 0]
 * ```
 */
export function costCurve(
  yTrue: MetricInput,
  yScore: TensorLike,
  options: CurveOptions & { readonly points?: number } = {},
): CostCurve {
  const r = roc(yTrue, yScore, options)
  const m = options.points ?? 101
  const pc = Array.from({ length: m }, (_, i) => (m === 1 ? 0 : i / (m - 1)))
  const nec = pc.map((p) => {
    let best = Number.POSITIVE_INFINITY
    for (const [i, f] of r.fpr.entries()) best = Math.min(best, (1 - (r.tpr[i] ?? 0)) * p + f * (1 - p))
    return best
  })
  return { probabilityCost: pc, normalizedExpectedCost: nec }
}
