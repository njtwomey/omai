/**
 * The data-bias record (`dataBias`): what a dataset holds about a protected attribute before any model is fitted, its
 * groups' representation (against a population when known), their base rates and the gaps between them, label bias
 * (labels flipped against clean ones, when known), the features that act as proxies for the group, and how much a
 * model amplifies the base-rate gap.
 */

import { sameLength, seal } from './_check.ts'
import type { GroupDisparity } from './fairness.ts'

/** The features that predict the protected attribute: each feature's dependence and how well it predicts the group. */
export interface ProxySummary {
  readonly features: readonly string[]
  /** Each feature's area under the ROC curve for telling the groups apart (0.5 is none; at least 0.5). */
  readonly auc: readonly number[]
  /** Each feature's normalised mutual information with the group (0 independent, 1 determined). */
  readonly dependence: readonly number[]
  /** The cross-validated AUC of every feature together (a linear model predicting the group). */
  readonly combinedAuc?: number
}

/**
 * Bias in a dataset across the groups of a protected attribute (AIF360's dataset metrics, Aequitas's group counts):
 * each group's count, share (and the population's share when known), base rate, the label disparities (statistical
 * parity difference and disparate impact of the labels), label flips against clean labels when known, proxies, and a
 * model's bias amplification (each group's predicted minus actual positive rate) when given.
 */
export interface DataBias {
  readonly kind: 'dataBias'
  readonly label?: string
  readonly attribute: string
  readonly groups: readonly string[]
  /** The group the others are compared with. */
  readonly reference: number
  readonly counts: readonly number[]
  readonly shares: readonly number[]
  /** Each group's share of the population the data should represent. */
  readonly populationShares?: readonly number[]
  readonly baseRates: readonly number[]
  readonly overallBaseRate: number
  /** Statistical parity difference and disparate impact of the labels against the reference. */
  readonly disparities: readonly GroupDisparity[]
  /** Per group, the share of clean positives recorded negative and of clean negatives recorded positive. */
  readonly labelFlips?: {
    readonly positiveToNegative: readonly number[]
    readonly negativeToPositive: readonly number[]
  }
  readonly proxies?: ProxySummary
  /** A model's selection rate per group, and its bias amplification (Wang and Russakovsky's BiasAmp). */
  readonly amplification?: { readonly selectionRates: readonly number[]; readonly value: number }
}

/**
 * A data-bias record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `shape` when a per-group field does not have one entry per group, or the proxies' columns
 *   differ in length.
 * @example
 * ```ts
 * import { dataBias } from 'omai-engine/exposition'
 * const r = dataBias({ attribute: 'sex', groups: ['f', 'm'], reference: 1, counts: [40, 60], shares: [0.4, 0.6],
 *   baseRates: [0.3, 0.5], overallBaseRate: 0.42, disparities: [] })
 * r.kind // => 'dataBias'
 * ```
 */
export function dataBias(record: Omit<DataBias, 'kind'>): DataBias {
  const k = record.groups.length
  for (const field of ['counts', 'shares', 'baseRates'] as const) sameLength('dataBias', field, record[field].length, k)
  if (record.populationShares !== undefined)
    sameLength('dataBias', 'populationShares', record.populationShares.length, k)
  if (record.labelFlips !== undefined) {
    sameLength('dataBias', 'labelFlips.positiveToNegative', record.labelFlips.positiveToNegative.length, k)
    sameLength('dataBias', 'labelFlips.negativeToPositive', record.labelFlips.negativeToPositive.length, k)
  }
  if (record.proxies !== undefined) {
    const f = record.proxies.features.length
    sameLength('dataBias', 'proxies.auc', record.proxies.auc.length, f)
    sameLength('dataBias', 'proxies.dependence', record.proxies.dependence.length, f)
  }
  if (record.amplification !== undefined)
    sameLength('dataBias', 'amplification.selectionRates', record.amplification.selectionRates.length, k)
  return seal({ kind: 'dataBias', ...record })
}
