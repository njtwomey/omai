/**
 * Intersectional audits of a fitted classifier: its rates in every combination of several protected attributes, the
 * worst subgroup, the generalised entropy index between and within subgroups, and a rich-subgroup audit over linear
 * functions of the attributes, as a `subgroupReport` record.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { type Explained, type SubgroupReport, subgroupReport } from '../../lib/exposition/index.ts'
import { entropyDecomposition, richSubgroupAudit, subgroupRates } from '../../lib/fairness/index.ts'
import type { Fitted } from '../../lib/learn/index.ts'
import {
  binaryLabels,
  type FairnessData,
  positiveScores,
  type SensitiveSpec,
  sensitiveGroups,
  withoutColumns,
} from './groups.ts'

/** Options of {@link intersectionalAudit}. */
export interface IntersectionalAuditOptions {
  /** The protected attributes: columns (name or index) or labels per row (default the dataset's sensitive ones). */
  readonly attributes?: readonly SensitiveSpec[]
  /** Names for the attributes (default a column's name, `attribute 1`, ... for labels). */
  readonly names?: readonly string[]
  /** The decision threshold (default 0.5 for probabilities, 0 for decision values). */
  readonly threshold?: number
  /** Score the model without the protected columns (it was fitted without them). Default false. */
  readonly dropSensitive?: boolean
  /** Leave subgroups smaller than this out of the gaps (default 20). */
  readonly minSize?: number
  /** The order of the generalised entropy index (default 2). */
  readonly alpha?: number
  /** The error rate of the rich-subgroup audit (default false positives); false to skip it. */
  readonly audit?: false | 'falsePositive' | 'falseNegative'
  readonly label?: string
}

/** The result of {@link intersectionalAudit}. */
export interface IntersectionalAudit {
  readonly attributes: readonly string[]
  /** Each subgroup's name (its values joined by ` × `). */
  readonly subgroups: readonly string[]
  readonly counts: readonly number[]
  /** The largest selection-rate gap between subgroups large enough. */
  readonly selectionGap: number
  /** The subgroup furthest from the overall selection rate. */
  readonly worstSubgroup: string
}

/**
 * An intersectional audit of a fitted binary classifier: its rates in every combination of several protected
 * attributes (a gap can hide in an intersection that neither attribute shows alone), the largest gaps and the worst
 * subgroup, the generalised entropy index of benefits split between and within subgroups, and a GerryFair audit of the
 * linear subgroup of the attributes with the largest weighted error-rate gap. Exposition: a `subgroupReport`.
 *
 * @param model - A fitted binary classifier.
 * @param data - Labelled data holding the attributes.
 * @param options - The attributes, threshold, minimum subgroup size, entropy order and audit.
 * @returns The subgroups and the headline gap, with the record.
 * @throws WhiteboxError `value` without at least one attribute.
 */
export function intersectionalAudit(
  model: Fitted,
  data: FairnessData,
  options: IntersectionalAuditOptions = {},
): Explained<IntersectionalAudit> {
  const specs = options.attributes ?? Object.keys(data.meta?.sensitive ?? {})
  if (specs.length === 0) throw new WhiteboxError('value', 'intersectionalAudit: give the protected attributes')
  const groups = specs.map((s, a) => {
    const g = sensitiveGroups(data, s)
    const name = options.names?.[a]
    if (name !== undefined) return { ...g, attribute: name }
    return typeof s === 'object' ? { ...g, attribute: `attribute ${a + 1}` } : g
  })
  const labels = binaryLabels(data, 'intersectionalAudit')
  const columns = groups.flatMap((g) => (g.column === undefined ? [] : [g.column]))
  const x = options.dropSensitive === true ? withoutColumns(data, columns).x : data.x
  const { scores, probability } = positiveScores(model, x)
  const threshold = options.threshold ?? (probability ? 0.5 : 0)
  const predicted = scores.map((s) => (s >= threshold ? 1 : 0))
  const attributes = groups.map((g) => g.index.map((i) => g.names[i] ?? String(i)))
  const r = subgroupRates(labels, predicted, attributes, { minSize: options.minSize ?? 20 })
  const e = entropyDecomposition(labels, predicted, attributes, { alpha: options.alpha ?? 2 })
  const auditRate = options.audit ?? 'falsePositive'
  // The audit's features: an indicator per non-reference value of each attribute.
  const indicators = groups.flatMap((g) =>
    g.names.slice(1).map((name, k) => ({ g, k: k + 1, name: `${g.attribute}=${name}` })),
  )
  const audited =
    auditRate === false || indicators.length === 0
      ? undefined
      : (() => {
          try {
            const features = labels.map((_, i) => indicators.map(({ g, k }) => (g.index[i] === k ? 1 : 0)))
            const a = richSubgroupAudit(features, labels, predicted, { rate: auditRate })
            // The regression's scale is arbitrary (its target is divided by n): report the weights scaled so the largest
            // magnitude, intercept included, is 1.
            const scale = Math.max(Math.abs(a.intercept), ...a.weights.map(Math.abs)) || 1
            return {
              rate: a.rate,
              weightedDisparity: a.weightedDisparity,
              disparity: a.disparity,
              size: a.size,
              subgroupRate: a.subgroupRate,
              baseline: a.baseline,
              features: indicators.map((f) => f.name),
              weights: a.weights.map((w) => w / scale),
              intercept: a.intercept / scale,
            }
          } catch {
            return undefined
          }
        })()
  const record: SubgroupReport = subgroupReport({
    label: options.label ?? 'intersectional audit',
    attributes: groups.map((g) => g.attribute),
    values: r.groups.values,
    subgroups: r.groups.subgroups,
    counts: r.groups.counts,
    tables: r.tables,
    rates: r.rates,
    overall: r.overall,
    minSize: r.minSize,
    gaps: r.gaps,
    worst: r.worst,
    entropy: e,
    ...(audited === undefined ? {} : { audit: audited }),
  })
  return {
    attributes: record.attributes,
    subgroups: r.groups.names,
    counts: r.groups.counts,
    selectionGap: r.gaps.selectionRate,
    worstSubgroup: r.groups.names[r.worst.selectionRate.subgroup] ?? '',
    exposition: [record],
  }
}
