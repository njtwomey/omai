/**
 * What the fairness mitigations share (private to the area): the rows without the protected column, the labels and
 * groups (the reference group first as the privileged one), an unconstrained logistic regression's scores to compare
 * with, and the group reports before and after a mitigation.
 */

import type { GroupReport } from '../../lib/exposition/index.ts'
import { featureNamesOf, matrixView } from '../../lib/learn/index.ts'
import { type Key, key as makeKey } from '../../lib/random/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { buildGroupReport, groupReportReference } from './audit.ts'
import {
  binaryLabels,
  type FairnessData,
  positiveScores,
  type SensitiveGroups,
  type SensitiveSpec,
  sensitiveGroups,
  withoutColumns,
} from './groups.ts'

/** The common options of the mitigations. */
export interface MitigationOptions {
  /** The protected attribute (default the data's first). */
  readonly sensitive?: SensitiveSpec
  /** The privileged (reference) group by name or index (default the largest). */
  readonly privileged?: number | string
  readonly key?: Key
}

/** A dataset made ready for a mitigation. */
export interface MitigationData {
  readonly groups: SensitiveGroups
  readonly labels: number[]
  /** The rows without the protected column. */
  readonly rows: number[][]
  readonly featureNames: string[]
  /** The data without the protected column. */
  readonly unaware: FairnessData
  /** The privileged group's index. */
  readonly privileged: number
  /** Each row's group with the privileged group as 0 and the rest as 1, for the two-group methods. */
  readonly binary: number[]
  readonly key: Key
  readonly classes?: readonly [string, string]
}

/** The rows, labels and groups of a dataset, ready for a mitigation. */
export function mitigationData(data: FairnessData, options: MitigationOptions, who: string): MitigationData {
  const groups = sensitiveGroups(data, options.sensitive)
  const labels = binaryLabels(data, who)
  const unaware = groups.column === undefined ? data : withoutColumns(data, [groups.column])
  const m = matrixView(unaware.x, who)
  const rows = Array.from({ length: m.n }, (_, i) => Array.from({ length: m.d }, (_, j) => m.data[i * m.d + j] ?? 0))
  const privileged = groupReportReference(groups, options.privileged)
  const classes = data.classes?.length === 2 ? ([data.classes[0], data.classes[1]] as [string, string]) : undefined
  return {
    groups,
    labels,
    rows,
    featureNames: featureNamesOf(unaware, m.d),
    unaware,
    privileged,
    binary: groups.index.map((g) => (g === privileged ? 0 : 1)),
    key: options.key ?? makeKey(0),
    ...(classes === undefined ? {} : { classes }),
  }
}

/** An unconstrained logistic regression on the unaware rows: its scores, to compare a mitigation with. */
export function baselineScores(d: MitigationData): number[] {
  return positiveScores(logisticRegression().fit(d.unaware, { key: d.key, explain: false }), d.unaware.x).scores
}

/** A group report of scores at 0.5, or of decisions (probabilities of a positive decision), labelled. */
export function reportOf(
  d: MitigationData,
  label: string,
  scores: readonly number[],
  decisions?: readonly number[],
  thresholds?: readonly number[],
): GroupReport {
  return buildGroupReport({
    scores,
    labels: d.labels,
    groups: d.groups,
    thresholds: thresholds ?? d.groups.names.map(() => 0.5),
    reference: d.privileged,
    label,
    ...(decisions === undefined ? {} : { decisions }),
    ...(d.classes === undefined ? {} : { classes: d.classes }),
  })
}
