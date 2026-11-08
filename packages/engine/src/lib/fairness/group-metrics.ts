/**
 * Group fairness metrics for classification (fairlearn's MetricFrame and parity metrics, AIF360's and Aequitas's group
 * measures): rates per group, their differences and ratios between groups or to the overall rate, disparate impact, and
 * calibration within groups.
 */

import { type MetricResult, type MetricSpec, maxOf, minOf, OmaiError } from '../base/index.ts'
import { type Label, labelValues, sortLabels } from '../metrics/_labels.ts'
import {
  defineMetric,
  type MetricFunction,
  type MetricInput,
  type MetricOptions,
  metricResult,
  sampleWeights,
} from '../metrics/metric.ts'
import { expectedCalibrationError } from '../metrics/probabilistic.ts'
import type { TensorLike } from '../tensor/index.ts'

const spec = (name: string, direction: 'higher' | 'lower', range: readonly [number, number]): MetricSpec => ({
  name,
  direction,
  range,
  inputs: 'labels',
  category: 'fairness',
  concept: 'algorithmic-fairness',
})

/** The rates of one group's binary predictions. */
export interface GroupRates {
  /** The (weighted) number of members. */
  readonly count: number
  /** P(yhat = 1): the selection (positive prediction) rate. */
  readonly selectionRate: number
  readonly truePositiveRate: number
  readonly falsePositiveRate: number
  readonly trueNegativeRate: number
  readonly falseNegativeRate: number
  /** P(y = 1 | yhat = 1): the positive predictive value. */
  readonly precision: number
  /** P(y = 0 | yhat = 0). */
  readonly negativePredictiveValue: number
  /** P(y = 0 | yhat = 1). */
  readonly falseDiscoveryRate: number
  /** P(y = 1 | yhat = 0). */
  readonly falseOmissionRate: number
  readonly accuracy: number
  /** FN / FP, the ratio treatment equality compares (Berk et al.); NaN without false positives. */
  readonly errorRatio: number
}

/** Options of the group fairness metrics. */
export interface FairnessOptions extends MetricOptions {
  /** The positive class (default 1). */
  readonly posLabel?: Label
}

/** Binary predictions and targets by group. */
function rates(t: readonly number[], p: readonly number[], w: Float64Array, members: readonly number[]): GroupRates {
  let tp = 0
  let fp = 0
  let tn = 0
  let fn = 0
  for (const i of members) {
    const wi = w[i] ?? 0
    const y = t[i] ?? 0
    const yh = p[i] ?? 0
    if (y === 1 && yh === 1) tp += wi
    else if (y === 0 && yh === 1) fp += wi
    else if (y === 0 && yh === 0) tn += wi
    else fn += wi
  }
  const n = tp + fp + tn + fn
  const r = (a: number, b: number): number => (b === 0 ? 0 : a / b)
  return {
    count: n,
    selectionRate: r(tp + fp, n),
    truePositiveRate: r(tp, tp + fn),
    falsePositiveRate: r(fp, fp + tn),
    trueNegativeRate: r(tn, tn + fp),
    falseNegativeRate: r(fn, fn + tp),
    precision: r(tp, tp + fp),
    negativePredictiveValue: r(tn, tn + fn),
    falseDiscoveryRate: r(fp, tp + fp),
    falseOmissionRate: r(fn, tn + fn),
    accuracy: r(tp + tn, n),
    errorRatio: fp === 0 ? Number.NaN : fn / fp,
  }
}

/** Encode targets, predictions and groups. */
function prepare(
  name: string,
  yTrue: MetricInput,
  yPred: MetricInput,
  groups: MetricInput,
  options: FairnessOptions,
): { t: number[]; p: number[]; w: Float64Array; byGroup: Map<Label, number[]> } {
  const pos = options.posLabel ?? 1
  const t = labelValues(yTrue).map((v) => (v === pos ? 1 : 0))
  const p = labelValues(yPred).map((v) => (v === pos ? 1 : 0))
  const g = labelValues(groups)
  if (t.length !== p.length || t.length !== g.length) throw new OmaiError('shape', `${name}: inputs differ in length`)
  const byGroup = new Map<Label, number[]>()
  for (const label of sortLabels(g)) byGroup.set(label, [])
  for (const [i, label] of g.entries()) byGroup.get(label)?.push(i)
  return { t, p, w: sampleWeights(name, t.length, options.sampleWeight), byGroup }
}

/**
 * Binary classification rates per group of a sensitive feature (fairlearn's MetricFrame with selection rate, true and
 * false positive and negative rates, precision and accuracy), plus the overall rates.
 *
 * @param yTrue - True binary labels `[N]`.
 * @param yPred - Predicted binary labels `[N]`.
 * @param groups - The group of each sample `[N]` (numbers or strings).
 * @param options - posLabel and sample weights.
 * @returns The rates of each group and overall.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { groupRates } from 'omai-engine/fairness'
 * groupRates([1, 0, 1, 0], [1, 1, 0, 0], ['a', 'a', 'b', 'b']).groups.a?.selectionRate // => 1
 * ```
 */
export function groupRates(
  yTrue: MetricInput,
  yPred: MetricInput,
  groups: MetricInput,
  options: FairnessOptions = {},
): { readonly groups: Readonly<Record<string, GroupRates>>; readonly overall: GroupRates } {
  const d = prepare('groupRates', yTrue, yPred, groups, options)
  const out: Record<string, GroupRates> = {}
  for (const [label, members] of d.byGroup) out[String(label)] = rates(d.t, d.p, d.w, members)
  return {
    groups: out,
    overall: rates(
      d.t,
      d.p,
      d.w,
      d.t.map((_, i) => i),
    ),
  }
}

/**
 * How a disparity summarises the groups: the largest gap between groups, the largest gap to the overall rate, or the
 * largest gap to a reference group (AIF360's privileged group; give `reference`).
 */
export type DisparityMethod = 'between_groups' | 'to_overall' | 'to_reference'

/** The difference (max - min, or max |group - overall|) and ratio (min / max, or min of group/overall and back). */
function disparity(
  values: readonly number[],
  overall: number,
  method: DisparityMethod,
  kind: 'difference' | 'ratio',
): number {
  if (method === 'to_overall' || method === 'to_reference') {
    if (kind === 'difference') return values.reduce((a, v) => Math.max(a, Math.abs(v - overall)), 0)
    return values.reduce((a, v) => Math.min(a, v === 0 || overall === 0 ? 0 : Math.min(v / overall, overall / v)), 1)
  }
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of values) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  if (kind === 'difference') return hi - lo
  return hi === 0 ? 0 : lo / hi
}

/** Options of the disparity metrics. */
export interface DisparityOptions extends FairnessOptions {
  readonly method?: DisparityMethod
  /** The reference group of `to_reference`. */
  readonly reference?: Label
}

/** The value the groups are compared with: the overall rate, or the reference group's for `to_reference`. */
function baseline(
  name: string,
  r: { readonly groups: Readonly<Record<string, GroupRates>>; readonly overall: GroupRates },
  pick: (r: GroupRates) => number,
  options: DisparityOptions,
): number {
  if (options.method !== 'to_reference') return pick(r.overall)
  const ref = options.reference === undefined ? undefined : r.groups[String(options.reference)]
  if (ref === undefined)
    throw new OmaiError('value', `${name}: to_reference needs a reference group present in the data`)
  return pick(ref)
}

/** A disparity metric of one group rate. */
function rateDisparity(
  name: string,
  pick: (r: GroupRates) => number,
  kind: 'difference' | 'ratio',
): MetricFunction<[MetricInput, MetricInput, MetricInput, DisparityOptions?]> {
  return defineMetric(
    spec(name, kind === 'difference' ? 'lower' : 'higher', [0, 1]),
    (yTrue: MetricInput, yPred: MetricInput, groups: MetricInput, options: DisparityOptions = {}): MetricResult => {
      const r = groupRates(yTrue, yPred, groups, options)
      const values = Object.values(r.groups)
        .map(pick)
        .filter((v) => !Number.isNaN(v))
      const value = disparity(values, baseline(name, r, pick, options), options.method ?? 'between_groups', kind)
      return metricResult(
        name,
        value,
        labelValues(yTrue).length,
        Object.fromEntries(Object.entries(r.groups).map(([g, v]) => [g, pick(v)])),
      )
    },
  )
}

/**
 * Demographic parity difference (fairlearn.metrics.demographic_parity_difference): the largest gap in selection rate
 * P(yhat = 1 | group) between groups (or to the overall rate); 0 is parity.
 *
 * @param yTrue - True labels `[N]` (unused by the selection rate, kept for a uniform signature).
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - method, posLabel and sample weights.
 * @returns The metric result, with each group's selection rate in the breakdown.
 * @index function Demographic parity gap; equalized odds and others beside it.
 * @example
 * ```ts
 * import { demographicParityDifference } from 'omai-engine/fairness'
 * demographicParityDifference([1, 0, 1, 0], [1, 1, 0, 0], ['a', 'a', 'b', 'b']).estimate.value // => 1
 * ```
 */
export const demographicParityDifference: MetricFunction<[MetricInput, MetricInput, MetricInput, DisparityOptions?]> =
  rateDisparity('demographicParityDifference', (r) => r.selectionRate, 'difference')

/**
 * Demographic parity ratio (fairlearn.metrics.demographic_parity_ratio): the smallest over the largest group
 * selection rate; 1 is parity.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - method, posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { demographicParityRatio } from 'omai-engine/fairness'
 * demographicParityRatio([0, 0, 0, 0], [1, 1, 1, 0], ['a', 'a', 'b', 'b']).estimate.value // => 0.5
 * ```
 */
export const demographicParityRatio: MetricFunction<[MetricInput, MetricInput, MetricInput, DisparityOptions?]> =
  rateDisparity('demographicParityRatio', (r) => r.selectionRate, 'ratio')

/**
 * Equal opportunity difference (fairlearn.metrics.equal_opportunity_difference): the largest gap in true positive
 * rate between groups.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - method, posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { equalOpportunityDifference } from 'omai-engine/fairness'
 * equalOpportunityDifference([1, 1, 1, 1], [1, 1, 1, 0], ['a', 'a', 'b', 'b']).estimate.value // => 0.5
 * ```
 */
export const equalOpportunityDifference: MetricFunction<[MetricInput, MetricInput, MetricInput, DisparityOptions?]> =
  rateDisparity('equalOpportunityDifference', (r) => r.truePositiveRate, 'difference')

/**
 * Equal opportunity ratio (fairlearn.metrics.equal_opportunity_ratio): the smallest over the largest group TPR.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - method, posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { equalOpportunityRatio } from 'omai-engine/fairness'
 * equalOpportunityRatio([1, 1, 1, 1], [1, 1, 1, 0], ['a', 'a', 'b', 'b']).estimate.value // => 0.5
 * ```
 */
export const equalOpportunityRatio: MetricFunction<[MetricInput, MetricInput, MetricInput, DisparityOptions?]> =
  rateDisparity('equalOpportunityRatio', (r) => r.truePositiveRate, 'ratio')

/**
 * Predictive parity difference: the largest gap in positive predictive value (precision) between groups, the
 * criterion of Chouldechova (2017).
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - method, posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { predictiveParityDifference } from 'omai-engine/fairness'
 * predictiveParityDifference([1, 0, 1, 1], [1, 1, 1, 1], ['a', 'a', 'b', 'b']).estimate.value // => 0.5
 * ```
 */
export const predictiveParityDifference: MetricFunction<[MetricInput, MetricInput, MetricInput, DisparityOptions?]> =
  rateDisparity('predictiveParityDifference', (r) => r.precision, 'difference')

/** Options of the equalised-odds metrics. */
export interface EqualizedOddsOptions extends DisparityOptions {
  /** Combine the TPR and FPR disparities by their worst case (default) or their mean (fairlearn's agg). */
  readonly agg?: 'worst_case' | 'mean'
}

const equalizedOdds = (
  name: string,
  kind: 'difference' | 'ratio',
): MetricFunction<[MetricInput, MetricInput, MetricInput, EqualizedOddsOptions?]> =>
  defineMetric(
    spec(name, kind === 'difference' ? 'lower' : 'higher', [0, 1]),
    (yTrue: MetricInput, yPred: MetricInput, groups: MetricInput, options: EqualizedOddsOptions = {}): MetricResult => {
      const r = groupRates(yTrue, yPred, groups, options)
      const method = options.method ?? 'between_groups'
      const g = Object.values(r.groups)
      const tpr = disparity(
        g.map((v) => v.truePositiveRate),
        baseline(name, r, (v) => v.truePositiveRate, options),
        method,
        kind,
      )
      const fpr = disparity(
        g.map((v) => v.falsePositiveRate),
        baseline(name, r, (v) => v.falsePositiveRate, options),
        method,
        kind,
      )
      const worst = kind === 'difference' ? Math.max(tpr, fpr) : Math.min(tpr, fpr)
      const value = options.agg === 'mean' ? (tpr + fpr) / 2 : worst
      return metricResult(name, value, labelValues(yTrue).length, { truePositiveRate: tpr, falsePositiveRate: fpr })
    },
  )

/**
 * Equalised odds difference (Hardt, Price and Srebro, 2016; fairlearn.metrics.equalized_odds_difference): the larger
 * of the between-group gaps in true positive rate and false positive rate (or their mean).
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - method, agg, posLabel and sample weights.
 * @returns The metric result, with the two gaps in the breakdown.
 * @example
 * ```ts
 * import { equalizedOddsDifference } from 'omai-engine/fairness'
 * equalizedOddsDifference([1, 0, 1, 0], [1, 1, 1, 0], ['a', 'a', 'b', 'b']).estimate.value // => 1
 * ```
 */
export const equalizedOddsDifference: MetricFunction<[MetricInput, MetricInput, MetricInput, EqualizedOddsOptions?]> =
  equalizedOdds('equalizedOddsDifference', 'difference')

/**
 * Equalised odds ratio (fairlearn.metrics.equalized_odds_ratio): the smaller of the between-group ratios of true and
 * false positive rates.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - method, agg, posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { equalizedOddsRatio } from 'omai-engine/fairness'
 * equalizedOddsRatio([1, 0, 1, 0], [1, 0, 1, 0], ['a', 'a', 'b', 'b']).estimate.value // => 0
 * ```
 */
export const equalizedOddsRatio: MetricFunction<[MetricInput, MetricInput, MetricInput, EqualizedOddsOptions?]> =
  equalizedOdds('equalizedOddsRatio', 'ratio')

/**
 * Disparate impact: the selection rate of the unprivileged groups over that of the privileged group (the
 * four-fifths rule flags values below 0.8); without a privileged group, the smallest over the largest group rate.
 *
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - privileged (the reference group), posLabel and sample weights.
 * @returns The metric result: the smallest ratio over unprivileged groups, each group's ratio in the breakdown.
 * @example
 * ```ts
 * import { disparateImpact } from 'omai-engine/fairness'
 * disparateImpact([1, 1, 1, 0], ['a', 'a', 'b', 'b'], { privileged: 'a' }).estimate.value // => 0.5
 * ```
 */
export const disparateImpact: MetricFunction<
  [MetricInput, MetricInput, (FairnessOptions & { readonly privileged?: Label })?]
> = defineMetric(
  spec('disparateImpact', 'higher', [0, Infinity]),
  (
    yPred: MetricInput,
    groups: MetricInput,
    options: FairnessOptions & { readonly privileged?: Label } = {},
  ): MetricResult => {
    const r = groupRates(yPred, yPred, groups, options)
    const entries = Object.entries(r.groups)
    if (options.privileged === undefined) {
      const values = entries.map(([, v]) => v.selectionRate)
      return metricResult('disparateImpact', disparity(values, 0, 'between_groups', 'ratio'), labelValues(yPred).length)
    }
    const reference = r.groups[String(options.privileged)]
    if (reference === undefined) throw new OmaiError('value', `disparateImpact: no group ${String(options.privileged)}`)
    const ratios = entries
      .filter(([g]) => g !== String(options.privileged))
      .map(
        ([g, v]) =>
          [g, reference.selectionRate === 0 ? Number.NaN : v.selectionRate / reference.selectionRate] as const,
      )
    const value = ratios.reduce((a, [, v]) => Math.min(a, v), Number.POSITIVE_INFINITY)
    return metricResult('disparateImpact', value, labelValues(yPred).length, Object.fromEntries(ratios))
  },
)

/**
 * Calibration by group: the expected calibration error of each group's probabilities, and the largest gap between
 * groups (sufficiency, or calibration within groups, in Kleinberg et al.'s sense).
 *
 * @param yTrue - True binary labels `[N]`.
 * @param yProb - Positive-class probabilities `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - nBins (default 10) and the calibration norm.
 * @returns The largest between-group ECE gap, with each group's ECE in the breakdown.
 * @example
 * ```ts
 * import { calibrationByGroup } from 'omai-engine/fairness'
 * calibrationByGroup([1, 0, 1, 0], [1, 0, 0.25, 0.25], ['a', 'a', 'b', 'b'], { nBins: 2 }).estimate.value // => 0.25
 * ```
 */
export const calibrationByGroup: MetricFunction<
  [MetricInput, TensorLike, MetricInput, { readonly nBins?: number; readonly norm?: 'l1' | 'l2' | 'max' }?]
> = defineMetric(
  { ...spec('calibrationByGroup', 'lower', [0, 1]), inputs: 'probabilities' },
  (
    yTrue: MetricInput,
    yProb: TensorLike,
    groups: MetricInput,
    options: { readonly nBins?: number; readonly norm?: 'l1' | 'l2' | 'max' } = {},
  ): MetricResult => {
    const t = labelValues(yTrue)
    const p = labelValues(yProb as MetricInput) as number[]
    const g = labelValues(groups)
    const per: Record<string, number> = {}
    for (const label of sortLabels(g)) {
      const idx = g.map((v, i) => (v === label ? i : -1)).filter((i) => i >= 0)
      per[String(label)] = expectedCalibrationError(
        idx.map((i) => t[i] ?? 0) as number[],
        idx.map((i) => p[i] ?? 0),
        { nBins: options.nBins ?? 10, ...(options.norm === undefined ? {} : { norm: options.norm }) },
      ).estimate.value
    }
    const values = Object.values(per)
    return metricResult('calibrationByGroup', disparity(values, 0, 'between_groups', 'difference'), t.length, per)
  },
)

/**
 * Accuracy parity difference (overall accuracy equality, Berk et al. 2018): the largest gap in accuracy between groups.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - method, reference, posLabel and sample weights.
 * @returns The metric result, with each group's accuracy in the breakdown.
 * @example
 * ```ts
 * import { accuracyDifference } from 'omai-engine/fairness'
 * accuracyDifference([1, 0, 1, 0], [1, 0, 0, 0], ['a', 'a', 'b', 'b']).estimate.value // => 0.5
 * ```
 */
export const accuracyDifference: MetricFunction<[MetricInput, MetricInput, MetricInput, DisparityOptions?]> =
  rateDisparity('accuracyDifference', (r) => r.accuracy, 'difference')

/**
 * Accuracy parity ratio: the smallest over the largest group accuracy.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - method, reference, posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { accuracyRatio } from 'omai-engine/fairness'
 * accuracyRatio([1, 0, 1, 0], [1, 0, 0, 0], ['a', 'a', 'b', 'b']).estimate.value // => 0.5
 * ```
 */
export const accuracyRatio: MetricFunction<[MetricInput, MetricInput, MetricInput, DisparityOptions?]> = rateDisparity(
  'accuracyRatio',
  (r) => r.accuracy,
  'ratio',
)

/**
 * Predictive equality difference (Chouldechova; Corbett-Davies et al.): the largest gap in false-positive rate between
 * groups, equalised odds' negative half.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - method, reference, posLabel and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { predictiveEqualityDifference } from 'omai-engine/fairness'
 * predictiveEqualityDifference([0, 0, 0, 0], [1, 0, 0, 0], ['a', 'a', 'b', 'b']).estimate.value // => 0.5
 * ```
 */
export const predictiveEqualityDifference: MetricFunction<[MetricInput, MetricInput, MetricInput, DisparityOptions?]> =
  rateDisparity('predictiveEqualityDifference', (r) => r.falsePositiveRate, 'difference')

/**
 * Treatment equality difference (Berk et al. 2018): the largest gap between groups in the ratio of false negatives to
 * false positives (FN / FP); groups without false positives are left out.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - method, reference, posLabel and sample weights.
 * @returns The metric result, with each group's FN / FP in the breakdown.
 * @example
 * ```ts
 * import { treatmentEqualityDifference } from 'omai-engine/fairness'
 * // a: 2 FN, 1 FP; b: 1 FN, 1 FP.
 * treatmentEqualityDifference([1, 1, 0, 1, 0], [0, 0, 1, 0, 1], ['a', 'a', 'a', 'b', 'b']).estimate.value // => 1
 * ```
 */
export const treatmentEqualityDifference: MetricFunction<[MetricInput, MetricInput, MetricInput, DisparityOptions?]> =
  rateDisparity('treatmentEqualityDifference', (r) => r.errorRatio, 'difference')

/** Options of {@link conditionalDemographicParityDifference}. */
export interface ConditionalParityOptions extends FairnessOptions {
  /** Combine the strata by their size-weighted mean (default) or their worst case. */
  readonly agg?: 'mean' | 'worst_case'
}

/**
 * Conditional statistical parity (Kamiran, Žliobaitė and Calders 2013; Corbett-Davies et al. 2017): demographic parity
 * within each stratum of legitimate factors (the groups' selection rates compared among people with the same
 * legitimate characteristics), combined over strata by their weighted mean or their worst case.
 *
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param strata - The legitimate factor of each row `[N]` (a combined label for several factors).
 * @param options - agg, posLabel and sample weights.
 * @returns The metric result, with each stratum's selection-rate gap in the breakdown.
 * @example
 * ```ts
 * import { conditionalDemographicParityDifference } from 'omai-engine/fairness'
 * // Within each stratum both groups are selected alike, though overall they are not.
 * const yPred = [1, 1, 0, 1, 0, 0]
 * const groups = ['a', 'a', 'b', 'b', 'b', 'b']
 * const strata = ['hi', 'hi', 'lo', 'hi', 'lo', 'lo']
 * conditionalDemographicParityDifference(yPred, groups, strata).estimate.value // => 0
 * ```
 */
export const conditionalDemographicParityDifference: MetricFunction<
  [MetricInput, MetricInput, MetricInput, ConditionalParityOptions?]
> = defineMetric(
  spec('conditionalDemographicParityDifference', 'lower', [0, 1]),
  (
    yPred: MetricInput,
    groups: MetricInput,
    strata: MetricInput,
    options: ConditionalParityOptions = {},
  ): MetricResult => {
    const s = labelValues(strata)
    const p = labelValues(yPred)
    if (s.length !== p.length)
      throw new OmaiError('shape', 'conditionalDemographicParityDifference: inputs differ in length')
    const w = sampleWeights('conditionalDemographicParityDifference', p.length, options.sampleWeight)
    const g = labelValues(groups)
    const per: Record<string, number> = {}
    let total = 0
    let sum = 0
    let worst = 0
    for (const stratum of sortLabels(s)) {
      const rows = s.flatMap((v, i) => (v === stratum ? [i] : []))
      const pos = options.posLabel ?? 1
      const decided = rows.map((i) => (p[i] === pos ? 1 : 0))
      const r = groupRates(
        decided,
        decided,
        rows.map((i) => String(g[i])),
        { sampleWeight: rows.map((i) => w[i] ?? 0) },
      )
      const gap = disparity(
        Object.values(r.groups).map((v) => v.selectionRate),
        0,
        'between_groups',
        'difference',
      )
      per[String(stratum)] = gap
      total += r.overall.count
      sum += r.overall.count * gap
      worst = Math.max(worst, gap)
    }
    const value = options.agg === 'worst_case' ? worst : total === 0 ? 0 : sum / total
    return metricResult('conditionalDemographicParityDifference', value, p.length, per)
  },
)

/** Mean scores of one class by group (Kleinberg, Mullainathan and Raghavan's balance). */
function balance(name: string, cls: 0 | 1): MetricFunction<[MetricInput, TensorLike, MetricInput, FairnessOptions?]> {
  return defineMetric(
    { ...spec(name, 'lower', [0, Infinity]), inputs: 'scores' },
    (yTrue: MetricInput, yScore: TensorLike, groups: MetricInput, options: FairnessOptions = {}): MetricResult => {
      const pos = options.posLabel ?? 1
      const t = labelValues(yTrue).map((v) => (v === pos ? 1 : 0))
      const s = labelValues(yScore as MetricInput) as number[]
      const g = labelValues(groups)
      if (t.length !== s.length || t.length !== g.length)
        throw new OmaiError('shape', `${name}: inputs differ in length`)
      const w = sampleWeights(name, t.length, options.sampleWeight)
      const per: Record<string, number> = {}
      for (const label of sortLabels(g)) {
        let num = 0
        let den = 0
        for (const [i, v] of g.entries())
          if (v === label && t[i] === cls) {
            num += (w[i] ?? 0) * (s[i] ?? 0)
            den += w[i] ?? 0
          }
        if (den > 0) per[String(label)] = num / den
      }
      return metricResult(name, disparity(Object.values(per), 0, 'between_groups', 'difference'), t.length, per)
    },
  )
}

/**
 * Balance for the positive class (Kleinberg, Mullainathan and Raghavan 2017): the largest gap between groups in the
 * mean score of the people who are actually positive (score parity among positives).
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - posLabel and sample weights.
 * @returns The metric result, with each group's mean positive score in the breakdown.
 * @example
 * ```ts
 * import { balanceForPositiveClass } from 'omai-engine/fairness'
 * balanceForPositiveClass([1, 1, 1, 0], [0.75, 0.25, 0.25, 0.5], ['a', 'a', 'b', 'b']).estimate.value // => 0.25
 * ```
 */
export const balanceForPositiveClass: MetricFunction<[MetricInput, TensorLike, MetricInput, FairnessOptions?]> =
  balance('balanceForPositiveClass', 1)

/**
 * Balance for the negative class (Kleinberg, Mullainathan and Raghavan 2017): the largest gap between groups in the
 * mean score of the people who are actually negative.
 *
 * @param yTrue - True labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - posLabel and sample weights.
 * @returns The metric result, with each group's mean negative score in the breakdown.
 * @example
 * ```ts
 * import { balanceForNegativeClass } from 'omai-engine/fairness'
 * balanceForNegativeClass([0, 0, 0, 1], [0.1, 0.3, 0.5, 0.9], ['a', 'a', 'b', 'b']).estimate.value // => 0.3
 * ```
 */
export const balanceForNegativeClass: MetricFunction<[MetricInput, TensorLike, MetricInput, FairnessOptions?]> =
  balance('balanceForNegativeClass', 0)

/** A metric of one group's targets and predictions. */
export type GroupMetricFn = (yTrue: Label[], yPred: Label[], sampleWeight: number[]) => number

/** Options of {@link metricFrame}. */
export interface MetricFrameOptions extends MetricOptions {
  /** Which way is better, for the worst group (default 'higher'). */
  readonly direction?: 'higher' | 'lower'
  /** The reference group for `toReference` gaps (default the largest group). */
  readonly reference?: Label
}

/** The result of {@link metricFrame}. */
export interface MetricFrame {
  /** The metric per group, in sorted group order. */
  readonly byGroup: Readonly<Record<string, number>>
  readonly overall: number
  readonly groupMin: number
  readonly groupMax: number
  /** groupMax - groupMin (fairlearn's `difference()`). */
  readonly difference: number
  /** groupMin / groupMax (fairlearn's `ratio()`; 0 when the max is 0). */
  readonly ratio: number
  /** The worst group by the direction, and its value. */
  readonly worst: { readonly group: string; readonly value: number }
  /** Each group's value minus the reference group's. */
  readonly toReference: Readonly<Record<string, number>>
  readonly reference: string
}

/**
 * Any metric disaggregated by group (fairlearn's MetricFrame): its value in each group and overall, the group minimum
 * and maximum, their difference and ratio, the worst group (the lowest of a higher-is-better metric), and each group's
 * gap to a reference group.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predictions `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param metric - The metric of one group's rows (labels, predictions and weights).
 * @param options - direction, reference and sample weights.
 * @returns The frame.
 * @example
 * ```ts
 * import { metricFrame } from 'omai-engine/fairness'
 * const accuracy = (t: unknown[], p: unknown[]) => t.filter((v, i) => v === p[i]).length / t.length
 * const f = metricFrame([1, 0, 1, 0], [1, 0, 0, 0], ['a', 'a', 'b', 'b'], accuracy)
 * f.byGroup // => { a: 1, b: 0.5 }
 * f.worst // => { group: 'b', value: 0.5 }
 * f.difference // => 0.5
 * ```
 */
export function metricFrame(
  yTrue: MetricInput,
  yPred: MetricInput,
  groups: MetricInput,
  metric: GroupMetricFn,
  options: MetricFrameOptions = {},
): MetricFrame {
  const t = labelValues(yTrue)
  const p = labelValues(yPred)
  const g = labelValues(groups)
  if (t.length !== p.length || t.length !== g.length)
    throw new OmaiError('shape', 'metricFrame: inputs differ in length')
  const w = Array.from(sampleWeights('metricFrame', t.length, options.sampleWeight))
  const byGroup: Record<string, number> = {}
  const sizes: Record<string, number> = {}
  for (const label of sortLabels(g)) {
    const rows = g.flatMap((v, i) => (v === label ? [i] : []))
    byGroup[String(label)] = metric(
      rows.map((i) => t[i] ?? 0),
      rows.map((i) => p[i] ?? 0),
      rows.map((i) => w[i] ?? 0),
    )
    sizes[String(label)] = rows.length
  }
  const entries = Object.entries(byGroup)
  const values = entries.map(([, v]) => v)
  const groupMin = minOf(values)
  const groupMax = maxOf(values)
  const higher = (options.direction ?? 'higher') === 'higher'
  const [worstGroup, worstValue] = entries.reduce((a, b) => ((higher ? b[1] < a[1] : b[1] > a[1]) ? b : a))
  let reference = options.reference === undefined ? undefined : String(options.reference)
  if (reference === undefined) reference = Object.entries(sizes).reduce((a, b) => (b[1] > a[1] ? b : a))[0]
  const ref = byGroup[reference]
  if (ref === undefined) throw new OmaiError('value', `metricFrame: no group ${reference}`)
  return {
    byGroup,
    overall: metric(t, p, w),
    groupMin,
    groupMax,
    difference: groupMax - groupMin,
    ratio: groupMax === 0 ? 0 : groupMin / groupMax,
    worst: { group: worstGroup, value: worstValue },
    toReference: Object.fromEntries(entries.map(([k, v]) => [k, v - ref])),
    reference,
  }
}

/** Options of {@link multicalibration}. */
export interface MulticalibrationOptions {
  /** Equal-width probability bins (default 10). */
  readonly nBins?: number
  /** Leave out cells (subgroup by bin) holding less than this share of the rows (default 0). */
  readonly minMass?: number
  /** Include the intersections of the attributes, not only each attribute's groups (default true). */
  readonly intersections?: boolean
}

/**
 * Multicalibration (Hébert-Johnson, Kim, Reingold and Rothblum 2018): calibration within every subgroup of a
 * collection, at every level of the prediction. The collection is every group of each attribute and (by default) every
 * combination of the attributes; for each subgroup and probability bin holding enough rows, the violation is
 * |mean(y - p)|, and the result is the largest.
 *
 * @param yTrue - True binary labels `[N]`.
 * @param yProb - Positive-class probabilities `[N]`.
 * @param attributes - One or more sensitive features, each `[N]`.
 * @param options - nBins, minMass and whether to include intersections.
 * @returns The largest violation, with each subgroup's largest violation in the breakdown.
 * @example
 * ```ts
 * import { multicalibration } from 'omai-engine/fairness'
 * // Calibrated overall (0.5 against half positives) but not within either group.
 * const r = multicalibration([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5], [['a', 'a', 'b', 'b']], { nBins: 2 })
 * r.estimate.value // => 0.5
 * ```
 */
export const multicalibration: MetricFunction<
  [MetricInput, TensorLike, readonly MetricInput[], MulticalibrationOptions?]
> = defineMetric(
  { ...spec('multicalibration', 'lower', [0, 1]), inputs: 'probabilities' },
  (
    yTrue: MetricInput,
    yProb: TensorLike,
    attributes: readonly MetricInput[],
    options: MulticalibrationOptions = {},
  ): MetricResult => {
    const t = labelValues(yTrue).map((v) => (v === 1 ? 1 : 0))
    const p = labelValues(yProb as MetricInput) as number[]
    const columns = attributes.map(labelValues)
    if (columns.some((c) => c.length !== t.length) || p.length !== t.length)
      throw new OmaiError('shape', 'multicalibration: inputs differ in length')
    const nBins = options.nBins ?? 10
    const minRows = (options.minMass ?? 0) * t.length
    const keyed = new Map<string, number[]>()
    const add = (name: string, i: number): void => {
      const rows = keyed.get(name)
      if (rows === undefined) keyed.set(name, [i])
      else rows.push(i)
    }
    for (let i = 0; i < t.length; i++) {
      for (const [a, c] of columns.entries()) add(columns.length > 1 ? `${a}:${String(c[i])}` : String(c[i]), i)
      if (columns.length > 1 && options.intersections !== false)
        add(columns.map((c, a) => `${a}:${String(c[i])}`).join(' & '), i)
    }
    const per: Record<string, number> = {}
    let worst = 0
    for (const [name, rows] of [...keyed].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
      const sums = new Array<number>(nBins).fill(0)
      const counts = new Array<number>(nBins).fill(0)
      for (const i of rows) {
        const b = Math.min(nBins - 1, Math.max(0, Math.floor((p[i] ?? 0) * nBins)))
        sums[b] = (sums[b] ?? 0) + (t[i] ?? 0) - (p[i] ?? 0)
        counts[b] = (counts[b] ?? 0) + 1
      }
      let v = 0
      for (let b = 0; b < nBins; b++)
        if ((counts[b] ?? 0) > 0 && (counts[b] ?? 0) >= minRows)
          v = Math.max(v, Math.abs((sums[b] ?? 0) / (counts[b] ?? 1)))
      per[name] = v
      worst = Math.max(worst, v)
    }
    return metricResult('multicalibration', worst, t.length, per)
  },
)
