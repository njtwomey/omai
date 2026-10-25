/**
 * Post-processing methods: the threshold optimiser over a fitted classifier, with group reports before and after and
 * the trade-off curves.
 */

import { curveSet, type Explained, type Exposition } from '../../lib/exposition/index.ts'
import {
  type GroupThresholdRule,
  type OptimisedThresholds,
  type OptimiseThresholdsOptions,
  optimiseThresholds,
  thresholdDecisions,
} from '../../lib/fairness/index.ts'
import type { Fitted } from '../../lib/learn/index.ts'
import { buildGroupReport, groupReportReference } from './audit.ts'
import {
  binaryLabels,
  type FairnessData,
  positiveScores,
  type SensitiveSpec,
  sensitiveGroups,
  withoutColumns,
} from './groups.ts'

/** A finite threshold to show for a rule, within the scores' range. */
function shownThreshold(r: GroupThresholdRule, lo: number, hi: number): number {
  const main = r.p0 >= r.p1 ? r.rule0 : r.rule1
  return Math.min(hi, Math.max(lo, main.threshold))
}

/** Options of {@link thresholdOptimizer}. */
export interface ThresholdOptimizerOptions extends OptimiseThresholdsOptions {
  /** The protected attribute (default the data's first). */
  readonly sensitive?: SensitiveSpec
  /** Score the model without the sensitive column (it was fitted without it). Default false. */
  readonly dropSensitive?: boolean
  /** The reference group of the reports (default the largest). */
  readonly reference?: number | string
}

/** The result of {@link thresholdOptimizer}. */
export interface ThresholdOptimizer extends OptimisedThresholds {
  readonly groups: readonly string[]
  /** P(positive decision) for each row of the data. */
  readonly decisions: readonly number[]
  /** Decide for new rows: P(positive) from the model's scores and the rows' groups. */
  readonly predictProbabilities: (scores: readonly number[], groups: readonly number[]) => number[]
}

/**
 * Post-process a fitted classifier with per-group randomised thresholds that meet a fairness constraint
 * (`optimiseThresholds`, fairlearn's `ThresholdOptimizer`). Exposition: the group report before (one threshold of 0.5
 * or 0) and after (the expected decisions), and a `curves` record of the trade-off the choice was made on (each
 * group's hull and the overall objective, or for equalised odds each group's ROC hull and the shared true-positive
 * rate) with the chosen point marked.
 *
 * @param model - A fitted binary classifier.
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, constraint, objective, grid and reference group.
 * @returns The rules and decisions, with the records.
 */
export function thresholdOptimizer(
  model: Fitted,
  data: FairnessData,
  options: ThresholdOptimizerOptions = {},
): Explained<ThresholdOptimizer> {
  const groups = sensitiveGroups(data, options.sensitive)
  const labels = binaryLabels(data, 'thresholdOptimizer')
  const x =
    options.dropSensitive === true && groups.column !== undefined ? withoutColumns(data, [groups.column]).x : data.x
  const { scores, probability } = positiveScores(model, x)
  const fitted = optimiseThresholds(scores, labels, groups.index, options)
  const decisions = thresholdDecisions(fitted.rules, scores, groups.index)
  const reference = groupReportReference(groups, options.reference)
  const classes = data.classes?.length === 2 ? ([data.classes[0], data.classes[1]] as [string, string]) : undefined
  const common = { scores, labels, groups, reference, ...(classes === undefined ? {} : { classes }) }
  const lo = Math.min(...scores)
  const hi = Math.max(...scores)
  const before = buildGroupReport({
    ...common,
    thresholds: groups.names.map(() => (probability ? 0.5 : 0)),
    label: 'before',
  })
  const after = buildGroupReport({
    ...common,
    thresholds: fitted.rules.map((r) => shownThreshold(r, lo, hi)),
    decisions,
    label: `after (${fitted.constraint})`,
    sweep: 0,
  })
  const eo = fitted.constraint === 'equalisedOdds'
  // Every tenth grid point is plenty for a figure.
  const keep = fitted.grid.map((_, j) => j).filter((j) => j % 10 === 0 || j === fitted.grid.length - 1)
  const pick = (v: readonly number[]): number[] => keep.map((j) => v[j] ?? 0)
  const xLabel = eo
    ? 'false-positive rate'
    : fitted.constraint === 'demographicParity'
      ? 'selection rate'
      : fitted.constraint === 'truePositiveRateParity'
        ? 'true-positive rate'
        : 'false-positive rate'
  const tradeoff = curveSet({
    x: pick(fitted.grid),
    xLabel,
    yLabel: eo ? 'true-positive rate (ROC hull)' : fitted.objective,
    series: [
      ...groups.names.map((name, g) => ({ name, y: pick(fitted.curves[g] ?? []) })),
      {
        name: eo ? 'shared (lowest group)' : 'overall',
        y: pick(
          eo
            ? fitted.curves.reduce((m, c) => m.map((v, j) => Math.min(v, c[j] ?? 0)), [...(fitted.curves[0] ?? [])])
            : fitted.overall,
        ),
      },
    ],
    marks: [{ x: fitted.xBest, label: 'chosen' }],
  })
  const exposition: Exposition[] = [before, after, tradeoff]
  return {
    ...fitted,
    groups: groups.names,
    decisions,
    predictProbabilities: (s, g) => thresholdDecisions(fitted.rules, s, g),
    exposition,
  }
}
