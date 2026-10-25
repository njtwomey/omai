import { WhiteboxError } from '../base/index.ts'
import { rowsOf, sameLength } from './_check.ts'

/** The rates a group report gives for each group (and overall), each a probability. */
export type GroupRateName =
  | 'selectionRate'
  | 'truePositiveRate'
  | 'falsePositiveRate'
  | 'precision'
  | 'accuracy'
  | 'baseRate'

/** The rates of a group report, in the order views show them. */
export const GROUP_RATES: readonly GroupRateName[] = [
  'selectionRate',
  'truePositiveRate',
  'falsePositiveRate',
  'precision',
  'accuracy',
  'baseRate',
]

/** One group's (or the whole population's) rates. */
export type GroupRateValues = Readonly<Record<GroupRateName, number>>

/** A summary disparity across the groups: a difference (ideal 0) or a ratio (ideal 1). */
export interface GroupDisparity {
  readonly name: string
  readonly value: number
  readonly ideal: number
  readonly kind: 'difference' | 'ratio'
}

/** One group's ROC curve: false- and true-positive rates at each threshold, thresholds decreasing. */
export interface GroupRoc {
  readonly fpr: readonly number[]
  readonly tpr: readonly number[]
  readonly thresholds: readonly number[]
}

/** The scored rows behind a report, so a view can re-threshold it (drag a group's threshold). */
export interface GroupRows {
  /** The positive-class score of each row. */
  readonly scores: readonly number[]
  /** The true label of each row (1 positive, 0 negative). */
  readonly labels: readonly number[]
  /** The group of each row, an index into `groups`. */
  readonly groups: readonly number[]
}

/**
 * Rates by threshold for every group: `selectionRate[g][t]` and so on at `thresholds[t]`, for a figure of how each
 * group's rates move with one shared threshold.
 */
export interface GroupSweep {
  readonly thresholds: readonly number[]
  readonly selectionRate: readonly (readonly number[])[]
  readonly truePositiveRate: readonly (readonly number[])[]
  readonly falsePositiveRate: readonly (readonly number[])[]
}

/**
 * A fairness audit of a binary classifier across the groups of a protected attribute (fairlearn's MetricFrame with
 * its disparities): per group the count, the decision threshold applied, the 2 by 2 table (actual by predicted; expected
 * counts for randomised decisions), the rates (selection, true- and false-positive, precision, accuracy, base rate),
 * the summary disparities, and optionally each group's ROC curve with its operating point, the scored rows (to move a
 * group's threshold) and a sweep of the rates over thresholds.
 */
export interface GroupReport {
  readonly kind: 'groupReport'
  /** What produced the decisions (`unmitigated`, `threshold optimiser (equalised odds)`). */
  readonly label?: string
  /** The protected attribute (`sex`, `group`). */
  readonly attribute: string
  readonly groups: readonly string[]
  /** The group the others are compared with (an index into `groups`). */
  readonly reference: number
  /** The names of the negative and positive class. */
  readonly classes?: readonly [string, string]
  /** The decision threshold of each group (`score >= threshold` predicts positive). */
  readonly thresholds: readonly number[]
  readonly counts: readonly number[]
  readonly rates: readonly GroupRateValues[]
  readonly overall: GroupRateValues
  /** tables[g][actual][predicted]. */
  readonly tables: readonly (readonly (readonly number[])[])[]
  readonly disparities: readonly GroupDisparity[]
  readonly roc?: readonly GroupRoc[]
  /** Each group's operating point (false- and true-positive rate) when it is not a threshold of its ROC curve. */
  readonly operatingPoints?: readonly { readonly fpr: number; readonly tpr: number }[]
  readonly scored?: GroupRows
  readonly sweep?: GroupSweep
}

/**
 * A group report, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when a per-group field does not have one entry per group, a table is not 2 by 2, or
 *   the rows' columns differ in length; `value` when the reference is not a group.
 * @example
 * ```ts
 * import { groupReport, groupReportFromTables } from 'whitebox-engine/exposition'
 * const r = groupReport(groupReportFromTables({ attribute: 'sex', groups: ['m', 'f'], tables: [[[5, 1], [1, 3]], [[4, 0], [3, 2]]] }))
 * r.rates[1]?.selectionRate // => 0.2222222222222222
 * ```
 */
export function groupReport(record: Omit<GroupReport, 'kind'>): GroupReport {
  const k = record.groups.length
  for (const field of ['thresholds', 'counts', 'rates', 'tables'] as const)
    sameLength('groupReport', field, record[field].length, k)
  for (const [g, t] of record.tables.entries()) {
    sameLength('groupReport', `tables[${g}]`, t.length, 2)
    rowsOf('groupReport', `tables[${g}]`, t, 2)
  }
  if (!(Number.isInteger(record.reference) && record.reference >= 0 && record.reference < k))
    throw new WhiteboxError('value', `groupReport: the reference ${record.reference} is not one of ${k} groups`)
  if (record.roc !== undefined) {
    sameLength('groupReport', 'roc', record.roc.length, k)
    for (const [g, c] of record.roc.entries()) {
      sameLength('groupReport', `roc[${g}].tpr`, c.tpr.length, c.fpr.length)
      sameLength('groupReport', `roc[${g}].thresholds`, c.thresholds.length, c.fpr.length)
    }
  }
  if (record.operatingPoints !== undefined)
    sameLength('groupReport', 'operatingPoints', record.operatingPoints.length, k)
  if (record.scored !== undefined) {
    sameLength('groupReport', 'scored.labels', record.scored.labels.length, record.scored.scores.length)
    sameLength('groupReport', 'scored.groups', record.scored.groups.length, record.scored.scores.length)
  }
  if (record.sweep !== undefined)
    for (const field of ['selectionRate', 'truePositiveRate', 'falsePositiveRate'] as const) {
      sameLength('groupReport', `sweep.${field}`, record.sweep[field].length, k)
      rowsOf('groupReport', `sweep.${field}`, record.sweep[field], record.sweep.thresholds.length)
    }
  return { kind: 'groupReport', ...record }
}

const ratio = (a: number, b: number): number => (b === 0 ? 0 : a / b)

/**
 * The rates of a 2 by 2 table (actual by predicted, the positive class second): selection, true- and false-positive
 * rates, precision, accuracy and base rate (0 where a rate's denominator is empty, as fairlearn).
 *
 * @param table - `table[actual][predicted]`.
 * @returns The rates.
 * @example
 * ```ts
 * import { tableRates } from 'whitebox-engine/exposition'
 * tableRates([[8, 2], [1, 9]]).truePositiveRate // => 0.9
 * ```
 */
export function tableRates(table: readonly (readonly number[])[]): GroupRateValues {
  const [tn, fp] = [table[0]?.[0] ?? 0, table[0]?.[1] ?? 0]
  const [fn, tp] = [table[1]?.[0] ?? 0, table[1]?.[1] ?? 0]
  const n = tn + fp + fn + tp
  return {
    selectionRate: ratio(tp + fp, n),
    truePositiveRate: ratio(tp, tp + fn),
    falsePositiveRate: ratio(fp, fp + tn),
    precision: ratio(tp, tp + fp),
    accuracy: ratio(tp + tn, n),
    baseRate: ratio(tp + fn, n),
  }
}

/** The largest minus the smallest of some values. */
function spread(values: readonly number[]): number {
  return values.length === 0 ? 0 : Math.max(...values) - Math.min(...values)
}

/**
 * The summary disparities of per-group rates, between groups (fairlearn's `between_groups`): demographic parity,
 * equal opportunity, equalised odds (the larger of the true- and false-positive gaps), predictive parity and accuracy
 * differences, and the disparate-impact ratio (the smallest selection rate over the reference group's).
 *
 * @param rates - Each group's rates.
 * @param reference - The reference group for disparate impact.
 * @returns The disparities.
 * @example
 * ```ts
 * import { groupDisparities, tableRates } from 'whitebox-engine/exposition'
 * const d = groupDisparities([tableRates([[5, 5], [0, 10]]), tableRates([[6, 4], [4, 6]])], 0)
 * d[0]?.value // => 0.25
 * ```
 */
export function groupDisparities(rates: readonly GroupRateValues[], reference: number): GroupDisparity[] {
  const pick = (r: GroupRateName): number[] => rates.map((v) => v[r])
  const ref = rates[reference]?.selectionRate ?? 0
  const others = rates.filter((_, g) => g !== reference).map((v) => v.selectionRate)
  const impact = ref === 0 ? Number.NaN : Math.min(...others.map((v) => v / ref), Number.POSITIVE_INFINITY)
  return [
    { name: 'demographic parity difference', value: spread(pick('selectionRate')), ideal: 0, kind: 'difference' },
    { name: 'equal opportunity difference', value: spread(pick('truePositiveRate')), ideal: 0, kind: 'difference' },
    {
      name: 'equalised odds difference',
      value: Math.max(spread(pick('truePositiveRate')), spread(pick('falsePositiveRate'))),
      ideal: 0,
      kind: 'difference',
    },
    { name: 'predictive parity difference', value: spread(pick('precision')), ideal: 0, kind: 'difference' },
    { name: 'accuracy difference', value: spread(pick('accuracy')), ideal: 0, kind: 'difference' },
    { name: 'disparate impact', value: Number.isFinite(impact) ? impact : Number.NaN, ideal: 1, kind: 'ratio' },
  ]
}

/** What {@link groupReportFromTables} needs: the groups and their tables, and optionally the rest of a report. */
export type GroupTablesInput = Pick<GroupReport, 'attribute' | 'groups' | 'tables'> &
  Partial<
    Omit<GroupReport, 'kind' | 'attribute' | 'groups' | 'tables' | 'counts' | 'rates' | 'overall' | 'disparities'>
  >

/**
 * A group report's fields from its per-group tables: counts, rates, the overall rates and the disparities (the
 * reference defaults to the largest group, the thresholds to 0.5).
 *
 * @param input - The attribute, groups and tables, with any other fields of the report.
 * @returns The report's fields (pass them to {@link groupReport}).
 * @example
 * ```ts
 * import { groupReportFromTables } from 'whitebox-engine/exposition'
 * groupReportFromTables({ attribute: 'a', groups: ['x', 'y'], tables: [[[1, 1], [0, 2]], [[3, 0], [1, 0]]] }).counts // => [4, 4]
 * ```
 */
export function groupReportFromTables(input: GroupTablesInput): Omit<GroupReport, 'kind'> {
  const counts = input.tables.map((t) => t.flat().reduce((a, b) => a + b, 0))
  let largest = 0
  for (const [g, c] of counts.entries()) if (c > (counts[largest] ?? 0)) largest = g
  const reference = input.reference ?? largest
  const rates = input.tables.map(tableRates)
  const total = [0, 1].map((a) => [0, 1].map((p) => input.tables.reduce((s, t) => s + (t[a]?.[p] ?? 0), 0)))
  return {
    ...input,
    reference,
    thresholds: input.thresholds ?? input.groups.map(() => 0.5),
    counts,
    rates,
    overall: tableRates(total),
    disparities: groupDisparities(rates, reference),
  }
}

/**
 * The report at other thresholds, recomputed from its scored rows: each group's table, rates and the disparities,
 * with each group's operating point on its ROC curve. Views use it to let a reader drag a group's threshold.
 *
 * @param record - A report with `scored` rows.
 * @param thresholds - A threshold per group (`score >= threshold` predicts positive).
 * @returns The report at those thresholds.
 * @throws WhiteboxError `value` when the report has no scored rows; `shape` without one threshold per group.
 * @example
 * ```ts
 * import { groupReport, groupReportAt, groupReportFromTables } from 'whitebox-engine/exposition'
 * const rows = { scores: [0.2, 0.6, 0.4, 0.9], labels: [0, 1, 0, 1], groups: [0, 0, 1, 1] }
 * const r = groupReport(groupReportFromTables({ attribute: 'a', groups: ['x', 'y'], tables: [[[1, 0], [0, 1]], [[1, 0], [0, 1]]], scored: rows }))
 * groupReportAt(r, [0.5, 0.3]).rates[1]?.falsePositiveRate // => 1
 * ```
 */
export function groupReportAt(record: GroupReport, thresholds: readonly number[]): GroupReport {
  const rows = record.scored
  if (rows === undefined) throw new WhiteboxError('value', 'groupReportAt: the report has no scored rows')
  sameLength('groupReportAt', 'thresholds', thresholds.length, record.groups.length)
  const tables = record.groups.map(() => [
    [0, 0],
    [0, 0],
  ])
  for (const [i, s] of rows.scores.entries()) {
    const g = rows.groups[i] ?? 0
    const actual = rows.labels[i] === 1 ? 1 : 0
    const predicted = s >= (thresholds[g] ?? 0.5) ? 1 : 0
    const row = tables[g]?.[actual]
    if (row !== undefined) row[predicted] = (row[predicted] ?? 0) + 1
  }
  const { operatingPoints: _, ...rest } = record
  return groupReport(groupReportFromTables({ ...rest, tables, thresholds, reference: record.reference }))
}
