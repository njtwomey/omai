/**
 * The subgroup report record (`subgroupReport`): a classifier's rates in every intersection of several protected
 * attributes, with the largest gaps, the worst subgroup in each rate, and optionally the generalised entropy index
 * split between and within subgroups and a rich-subgroup (GerryFair) audit. Views draw it as a heatmap of one rate over
 * the attributes' values.
 */

import { MdmlError } from '../base/index.ts'
import { rowsOf, sameLength } from './_check.ts'
import type { GroupRateName, GroupRateValues } from './fairness.ts'

/** The rich-subgroup audit of a subgroup report: the linear subgroup with the largest weighted error-rate gap. */
export interface SubgroupAuditSummary {
  readonly rate: 'falsePositive' | 'falseNegative'
  readonly weightedDisparity: number
  readonly disparity: number
  /** The subgroup's share of the rows of the audited class. */
  readonly size: number
  readonly subgroupRate: number
  readonly baseline: number
  /** The defining function's weights on the protected features (scaled so the largest magnitude is 1), with names. */
  readonly features: readonly string[]
  readonly weights: readonly number[]
  readonly intercept: number
}

/**
 * A classifier's rates in every intersection of several protected attributes (Buolamwini and Gebru's intersectional
 * audit; fairlearn's MetricFrame over several sensitive features): each subgroup's values, size, 2 by 2 table and
 * rates, the overall rates, the largest gap in each rate between subgroups at least `minSize` large and the subgroup
 * furthest from the overall rate, with the generalised entropy decomposition and a rich-subgroup audit when computed.
 */
export interface SubgroupReport {
  readonly kind: 'subgroupReport'
  readonly label?: string
  /** The attributes' names (`sex`, `age`). */
  readonly attributes: readonly string[]
  /** Each attribute's values, in order. */
  readonly values: readonly (readonly string[])[]
  /** Each subgroup's value of every attribute. */
  readonly subgroups: readonly (readonly string[])[]
  readonly counts: readonly number[]
  /** tables[s][actual][predicted]. */
  readonly tables: readonly (readonly (readonly number[])[])[]
  readonly rates: readonly GroupRateValues[]
  readonly overall: GroupRateValues
  /** Subgroups smaller than this are left out of the gaps. */
  readonly minSize: number
  readonly gaps: Readonly<Record<GroupRateName, number>>
  readonly worst: Readonly<Record<GroupRateName, { readonly subgroup: number; readonly gap: number }>>
  /** The generalised entropy index of benefits, between and within the subgroups. */
  readonly entropy?: {
    readonly alpha: number
    readonly total: number
    readonly between: number
    readonly within: number
  }
  readonly audit?: SubgroupAuditSummary
}

/**
 * A subgroup report, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when a per-subgroup field does not have one entry per subgroup, a subgroup does not have
 *   one value per attribute, or a table is not 2 by 2; `value` when a worst subgroup is not a subgroup.
 * @example
 * ```ts
 * import { subgroupReport, tableRates } from 'mdml-engine/exposition'
 * const t = [[1, 0], [0, 1]]
 * const rate = { subgroup: 0, gap: 0 }
 * const r = subgroupReport({
 *   attributes: ['sex'], values: [['f']], subgroups: [['f']], counts: [2], tables: [t], rates: [tableRates(t)],
 *   overall: tableRates(t), minSize: 1,
 *   gaps: { selectionRate: 0, truePositiveRate: 0, falsePositiveRate: 0, precision: 0, accuracy: 0, baseRate: 0 },
 *   worst: { selectionRate: rate, truePositiveRate: rate, falsePositiveRate: rate, precision: rate, accuracy: rate, baseRate: rate },
 * })
 * r.kind // => 'subgroupReport'
 * ```
 */
export function subgroupReport(record: Omit<SubgroupReport, 'kind'>): SubgroupReport {
  const k = record.subgroups.length
  for (const field of ['counts', 'tables', 'rates'] as const)
    sameLength('subgroupReport', field, record[field].length, k)
  sameLength('subgroupReport', 'values', record.values.length, record.attributes.length)
  for (const [s, v] of record.subgroups.entries())
    sameLength('subgroupReport', `subgroups[${s}]`, v.length, record.attributes.length)
  for (const [s, t] of record.tables.entries()) {
    sameLength('subgroupReport', `tables[${s}]`, t.length, 2)
    rowsOf('subgroupReport', `tables[${s}]`, t, 2)
  }
  for (const w of Object.values(record.worst))
    if (!(Number.isInteger(w.subgroup) && w.subgroup >= 0 && w.subgroup < Math.max(k, 1)))
      throw new MdmlError('value', `subgroupReport: the worst subgroup ${w.subgroup} is not one of ${k}`)
  if (record.audit !== undefined)
    sameLength('subgroupReport', 'audit.weights', record.audit.weights.length, record.audit.features.length)
  return { kind: 'subgroupReport', ...record }
}

/**
 * A subgroup report's rate laid out as a grid over the attributes: rows are the first attribute's values, columns
 * every combination of the others (one column when there is one attribute); a cell is NaN where its subgroup has no
 * rows, and `counts` gives each cell's size.
 *
 * @param record - The report.
 * @param rate - The rate to lay out.
 * @returns The row and column labels, the values and the counts, `[rows][columns]`.
 * @example
 * ```ts
 * import { subgroupGrid, subgroupReport, tableRates } from 'mdml-engine/exposition'
 * const tables = [[[1, 1], [0, 2]], [[2, 0], [1, 1]]]
 * const rate = { subgroup: 0, gap: 0 }
 * const zero = { selectionRate: 0, truePositiveRate: 0, falsePositiveRate: 0, precision: 0, accuracy: 0, baseRate: 0 }
 * const r = subgroupReport({
 *   attributes: ['sex', 'age'], values: [['f'], ['old', 'young']], subgroups: [['f', 'old'], ['f', 'young']],
 *   counts: [4, 4], tables, rates: tables.map(tableRates), overall: zero, minSize: 1, gaps: zero,
 *   worst: { selectionRate: rate, truePositiveRate: rate, falsePositiveRate: rate, precision: rate, accuracy: rate, baseRate: rate },
 * })
 * subgroupGrid(r, 'selectionRate').values // => [[0.75, 0.25]]
 * subgroupGrid(r, 'selectionRate').columns // => ['old', 'young']
 * ```
 */
export function subgroupGrid(
  record: SubgroupReport,
  rate: GroupRateName,
): { rows: string[]; columns: string[]; values: number[][]; counts: number[][] } {
  const rows = [...(record.values[0] ?? [])]
  const rest = record.values.slice(1)
  const columns = rest.reduce<string[][]>((acc, vals) => acc.flatMap((a) => vals.map((v) => [...a, v])), [[]])
  const columnNames = rest.length === 0 ? [record.attributes[0] ?? ''] : columns.map((c) => c.join(' × '))
  const values = rows.map(() => columnNames.map(() => Number.NaN))
  const counts = rows.map(() => columnNames.map(() => 0))
  for (const [s, v] of record.subgroups.entries()) {
    const i = rows.indexOf(v[0] ?? '')
    const j = rest.length === 0 ? 0 : columns.findIndex((c) => c.every((x, a) => x === v[a + 1]))
    if (i < 0 || j < 0) continue
    const row = values[i]
    const count = counts[i]
    if (row !== undefined) row[j] = record.rates[s]?.[rate] ?? Number.NaN
    if (count !== undefined) count[j] = record.counts[s] ?? 0
  }
  return { rows, columns: columnNames, values, counts }
}
