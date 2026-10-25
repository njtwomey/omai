import { linspaceNumbers, WhiteboxError } from '../../lib/base/index.ts'
import {
  type Explained,
  type GroupReport,
  type GroupRoc,
  groupReport,
  groupReportFromTables,
} from '../../lib/exposition/index.ts'
import {
  demographicParityDifference,
  disparateImpact,
  equalizedOddsDifference,
  equalOpportunityDifference,
  type GroupRates,
  groupRates,
  predictiveParityDifference,
} from '../../lib/fairness/index.ts'
import type { Fitted } from '../../lib/learn/index.ts'
import { rocCurve } from '../../lib/metrics/index.ts'
import {
  binaryLabels,
  type FairnessData,
  positiveScores,
  type SensitiveGroups,
  type SensitiveSpec,
  sensitiveGroups,
  withoutColumns,
} from './groups.ts'

/** What a group report is built from: scores or decision probabilities, labels and groups. */
export interface GroupReportInput {
  /** Each row's positive-class score. */
  readonly scores: readonly number[]
  /** Each row's binary label (1 positive). */
  readonly labels: readonly number[]
  readonly groups: SensitiveGroups
  /** A threshold per group (`score >= threshold` predicts positive). */
  readonly thresholds: readonly number[]
  /**
   * Each row's probability of a positive decision, for randomised decisions (a threshold optimiser's): the tables then
   * hold expected counts. Default: the thresholds' hard decisions.
   */
  readonly decisions?: readonly number[]
  /** The reference group, by index or name (default the largest). */
  readonly reference?: number | string
  readonly label?: string
  readonly classes?: readonly [string, string]
  /** Thresholds in the sweep of rates (default 51; 0 for none). */
  readonly sweep?: number
  /** Keep the scored rows so a view can move the thresholds (default true, up to 5000 rows). */
  readonly keepRows?: boolean
  /** Each group's operating point when it is not on its ROC curve (randomised decisions). */
  readonly operatingPoints?: readonly { readonly fpr: number; readonly tpr: number }[]
}

/** Rows of each group. */
function membersOf(groups: SensitiveGroups): number[][] {
  const out = groups.names.map((): number[] => [])
  for (const [i, g] of groups.index.entries()) out[g]?.push(i)
  return out
}

/** A group's ROC curve, thinned to at most 200 points (empty when the group has one class only). */
function groupRoc(labels: readonly number[], scores: readonly number[]): GroupRoc {
  if (!(labels.includes(0) && labels.includes(1))) return { fpr: [], tpr: [], thresholds: [] }
  const c = rocCurve(labels, scores)
  const step = Math.max(1, Math.ceil(c.fpr.length / 200))
  const keep = c.fpr.map((_, i) => i).filter((i) => i % step === 0 || i === c.fpr.length - 1)
  return {
    fpr: keep.map((i) => c.fpr[i] ?? 0),
    tpr: keep.map((i) => c.tpr[i] ?? 0),
    // The first threshold is +infinity (nothing positive); JSON-safe as the largest score.
    thresholds: keep.map((i) => {
      const t = c.thresholds[i] ?? 0
      return Number.isFinite(t) ? t : Math.max(...scores)
    }),
  }
}

/**
 * A `groupReport` record from scores, labels and groups at per-group thresholds (or from randomised decisions): each
 * group's table, rates and ROC curve, the disparities, the rates over a sweep of thresholds and the scored rows.
 *
 * @param input - Scores, labels, groups, thresholds and the record's options.
 * @returns The record.
 */
export function buildGroupReport(input: GroupReportInput): GroupReport {
  const { scores, labels, groups } = input
  const members = membersOf(groups)
  const decision = (i: number): number =>
    input.decisions?.[i] ?? ((scores[i] ?? 0) >= (input.thresholds[groups.index[i] ?? 0] ?? 0.5) ? 1 : 0)
  const tables = members.map((rows) => {
    const t = [
      [0, 0],
      [0, 0],
    ]
    for (const i of rows) {
      const row = t[labels[i] === 1 ? 1 : 0] as number[]
      const d = decision(i)
      row[1] = (row[1] ?? 0) + d
      row[0] = (row[0] ?? 0) + 1 - d
    }
    return t
  })
  const reference =
    typeof input.reference === 'string' ? Math.max(0, groups.names.indexOf(input.reference)) : input.reference
  const probability = scores.every((s) => s >= 0 && s <= 1)
  const nSweep = input.sweep ?? 51
  const lo = probability ? 0 : Math.min(...scores)
  const hi = probability ? 1 : Math.max(...scores)
  const grid = nSweep > 1 ? linspaceNumbers(lo, hi, nSweep) : []
  const sweepOf = (pick: (tp: number, fp: number, p: number, n: number) => number): number[][] =>
    members.map((rows) =>
      grid.map((t) => {
        let tp = 0
        let fp = 0
        let p = 0
        for (const i of rows) {
          const yes = (scores[i] ?? 0) >= t
          if (labels[i] === 1) {
            p++
            if (yes) tp++
          } else if (yes) fp++
        }
        return pick(tp, fp, p, rows.length)
      }),
    )
  const keep = (input.keepRows ?? true) && scores.length <= 5000
  return groupReport(
    groupReportFromTables({
      attribute: groups.attribute,
      groups: groups.names,
      tables,
      thresholds: input.thresholds,
      ...(reference === undefined ? {} : { reference }),
      ...(input.label === undefined ? {} : { label: input.label }),
      ...(input.classes === undefined ? {} : { classes: input.classes }),
      roc: members.map((rows) =>
        groupRoc(
          rows.map((i) => labels[i] ?? 0),
          rows.map((i) => scores[i] ?? 0),
        ),
      ),
      ...(input.operatingPoints === undefined ? {} : { operatingPoints: input.operatingPoints }),
      ...(keep ? { scored: { scores: [...scores], labels: [...labels], groups: [...groups.index] } } : {}),
      ...(grid.length === 0
        ? {}
        : {
            sweep: {
              thresholds: grid,
              selectionRate: sweepOf((tp, fp, _p, n) => (n === 0 ? 0 : (tp + fp) / n)),
              truePositiveRate: sweepOf((tp, _fp, p) => (p === 0 ? 0 : tp / p)),
              falsePositiveRate: sweepOf((_tp, fp, p, n) => (n - p === 0 ? 0 : fp / (n - p))),
            },
          }),
    }),
  )
}

/** Options of {@link fairnessAudit}. */
export interface FairnessAuditOptions {
  /** The protected attribute: a feature column (name or index) or a label per row (default the data's first). */
  readonly sensitive?: SensitiveSpec
  /** The decision threshold, one for all or one per group (default 0.5 for probabilities, 0 for decision values). */
  readonly threshold?: number | readonly number[]
  /** Score the model without the sensitive column (it was fitted without it). Default false. */
  readonly dropSensitive?: boolean
  /** The reference group by name or index (default the largest). */
  readonly reference?: number | string
  /** A label for the record (default `audit`). */
  readonly label?: string
  /** Thresholds in the sweep of rates (default 51). */
  readonly sweep?: number
}

/** The result of {@link fairnessAudit}. */
export interface FairnessAudit {
  readonly attribute: string
  readonly groups: readonly string[]
  readonly thresholds: readonly number[]
  /** fairlearn's MetricFrame rates per group and overall (`whitebox-engine/metrics` `groupRates`). */
  readonly rates: Readonly<Record<string, GroupRates>>
  readonly overall: GroupRates
  /** The parity metrics of `whitebox-engine/metrics` at the thresholds. */
  readonly disparities: {
    readonly demographicParityDifference: number
    readonly equalOpportunityDifference: number
    readonly equalizedOddsDifference: number
    readonly predictiveParityDifference: number
    readonly disparateImpact: number
  }
  /** Each row's positive-class score and group. */
  readonly scores: readonly number[]
  readonly groupIndex: readonly number[]
}

/**
 * A fairness audit of a fitted binary classifier (fairlearn's MetricFrame and parity metrics): score the rows, decide
 * at a threshold per group, and report each group's rates (selection, true- and false-positive, precision, accuracy,
 * base rate), the disparities between groups (demographic parity, equal opportunity, equalised odds, predictive
 * parity, disparate impact against the reference group), each group's ROC curve with its operating point, and the
 * rates over a sweep of shared thresholds. Exposition: a `groupReport` with the scored rows, so a view can move each
 * group's threshold.
 *
 * @param model - A fitted binary classifier (probabilities, or a decision function).
 * @param data - Labelled data holding the protected attribute (a column, or labels given in `sensitive`).
 * @param options - The attribute, thresholds, reference group, whether the model saw the attribute, and the sweep.
 * @returns The rates and disparities, with the record.
 * @throws WhiteboxError `value` when the data has no labels or a threshold list does not match the groups.
 */
export function fairnessAudit(
  model: Fitted,
  data: FairnessData,
  options: FairnessAuditOptions = {},
): Explained<FairnessAudit> {
  const groups = sensitiveGroups(data, options.sensitive)
  const labels = binaryLabels(data, 'fairnessAudit')
  const x =
    options.dropSensitive === true && groups.column !== undefined ? withoutColumns(data, [groups.column]).x : data.x
  const { scores, probability } = positiveScores(model, x)
  const k = groups.names.length
  const given = options.threshold ?? (probability ? 0.5 : 0)
  const thresholds = typeof given === 'number' ? groups.names.map(() => given) : [...given]
  if (thresholds.length !== k)
    throw new WhiteboxError('value', `fairnessAudit: ${thresholds.length} thresholds for ${k} groups`)
  const predicted = scores.map((s, i) => (s >= (thresholds[groups.index[i] ?? 0] ?? 0) ? 1 : 0))
  const named = groups.index.map((g) => groups.names[g] ?? String(g))
  const r = groupRates(labels, predicted, named)
  const reference = groupReportReference(groups, options.reference)
  const value = (m: { readonly estimate: { readonly value: number } }): number => m.estimate.value
  const disparities = {
    demographicParityDifference: value(demographicParityDifference(labels, predicted, named)),
    equalOpportunityDifference: value(equalOpportunityDifference(labels, predicted, named)),
    equalizedOddsDifference: value(equalizedOddsDifference(labels, predicted, named)),
    predictiveParityDifference: value(predictiveParityDifference(labels, predicted, named)),
    disparateImpact:
      k < 2 ? 1 : value(disparateImpact(predicted, named, { privileged: groups.names[reference] ?? '' })),
  }
  const classes = data.classes?.length === 2 ? ([data.classes[0], data.classes[1]] as [string, string]) : undefined
  const record = buildGroupReport({
    scores,
    labels,
    groups,
    thresholds,
    reference,
    label: options.label ?? 'audit',
    ...(classes === undefined ? {} : { classes }),
    ...(options.sweep === undefined ? {} : { sweep: options.sweep }),
  })
  return {
    attribute: groups.attribute,
    groups: groups.names,
    thresholds,
    rates: r.groups,
    overall: r.overall,
    disparities,
    scores,
    groupIndex: groups.index,
    exposition: [record],
  }
}

/** The reference group's index: the one named, else the largest group. */
export function groupReportReference(groups: SensitiveGroups, reference?: number | string): number {
  if (typeof reference === 'number') return reference
  if (typeof reference === 'string') {
    const at = groups.names.indexOf(reference)
    if (at < 0) throw new WhiteboxError('value', `fairness: no group ${reference}`)
    return at
  }
  const counts = groups.names.map(() => 0)
  for (const g of groups.index) counts[g] = (counts[g] ?? 0) + 1
  let best = 0
  for (const [g, c] of counts.entries()) if (c > (counts[best] ?? 0)) best = g
  return best
}
