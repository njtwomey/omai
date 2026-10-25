/**
 * Group fairness metrics for classification (fairlearn's MetricFrame and parity metrics, AIF360's and Aequitas's group
 * measures): rates per group, their differences and ratios between groups or to the overall rate, disparate impact, and
 * calibration within groups.
 */

import { type MetricResult, type MetricSpec, WhiteboxError } from '../base/index.ts'
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
  readonly accuracy: number
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
    accuracy: r(tp + tn, n),
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
  if (t.length !== p.length || t.length !== g.length)
    throw new WhiteboxError('shape', `${name}: inputs differ in length`)
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
 * import { groupRates } from 'whitebox-engine/fairness'
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

/** How a disparity summarises the groups: the largest gap between groups, or the largest gap to the overall rate. */
export type DisparityMethod = 'between_groups' | 'to_overall'

/** The difference (max - min, or max |group - overall|) and ratio (min / max, or min of group/overall and back). */
function disparity(
  values: readonly number[],
  overall: number,
  method: DisparityMethod,
  kind: 'difference' | 'ratio',
): number {
  if (method === 'to_overall') {
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
      const values = Object.values(r.groups).map(pick)
      const value = disparity(values, pick(r.overall), options.method ?? 'between_groups', kind)
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
 * @example
 * ```ts
 * import { demographicParityDifference } from 'whitebox-engine/fairness'
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
 * import { demographicParityRatio } from 'whitebox-engine/fairness'
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
 * import { equalOpportunityDifference } from 'whitebox-engine/fairness'
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
 * import { equalOpportunityRatio } from 'whitebox-engine/fairness'
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
 * import { predictiveParityDifference } from 'whitebox-engine/fairness'
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
        r.overall.truePositiveRate,
        method,
        kind,
      )
      const fpr = disparity(
        g.map((v) => v.falsePositiveRate),
        r.overall.falsePositiveRate,
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
 * import { equalizedOddsDifference } from 'whitebox-engine/fairness'
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
 * import { equalizedOddsRatio } from 'whitebox-engine/fairness'
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
 * import { disparateImpact } from 'whitebox-engine/fairness'
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
    if (reference === undefined)
      throw new WhiteboxError('value', `disparateImpact: no group ${String(options.privileged)}`)
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
 * import { calibrationByGroup } from 'whitebox-engine/fairness'
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
