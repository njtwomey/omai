/**
 * Fairness after training: `thresholdOptimizer` post-processes a fitted classifier with per-group randomised thresholds
 * that meet a fairness constraint while keeping as much accuracy as possible (fairlearn's `ThresholdOptimizer`),
 * returning the audits before and after.
 */

import { curveSet, type Explained, type Exposition } from '../../lib/exposition/index.ts'
import {
  type CalibratedCost,
  type CalibratedEqualisedOdds,
  calibratedEqualisedOdds,
  calibratedEqualisedOddsDecisions,
  type GroupThresholdRule,
  type OptimisedThresholds,
  type OptimiseThresholdsOptions,
  optimiseThresholds,
  type RejectOption,
  type RejectOptionOptions,
  rejectOptionClassification,
  rejectOptionDecisions,
  thresholdDecisions,
} from '../../lib/fairness/index.ts'
import type { Fitted } from '../../lib/learn/index.ts'
import { type MitigationOptions, mitigationData, reportOf } from './_mitigation.ts'
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

/**
 * Calibrated equalised odds (`calibratedEqualisedOdds` of `mdml-engine/fairness`, Pleiss et al.) on a fitted
 * classifier: the cheaper group's scores are mixed with its base rate until the groups' generalised costs match.
 * Exposition: the group reports before and after (expected decisions).
 *
 * @param model - A fitted binary classifier with probabilities.
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, privileged group, cost and whether the model saw the attribute.
 * @returns The fitted post-processor and the decisions, with the records.
 */
export function calibratedEqOdds(
  model: Fitted,
  data: FairnessData,
  options: MitigationOptions & { readonly cost?: CalibratedCost; readonly dropSensitive?: boolean } = {},
): Explained<{ readonly fit: CalibratedEqualisedOdds; readonly decisions: readonly number[] }> {
  const d = mitigationData(data, options, 'calibratedEqOdds')
  const x = options.dropSensitive === true ? d.unaware.x : data.x
  const { scores } = positiveScores(model, x)
  const fit = calibratedEqualisedOdds(scores, d.labels, d.groups.index, {
    privileged: d.privileged,
    ...(options.cost === undefined ? {} : { cost: options.cost }),
  })
  const decisions = calibratedEqualisedOddsDecisions(fit, scores, d.groups.index)
  return {
    fit,
    decisions,
    exposition: [
      reportOf(d, 'before', scores),
      reportOf(
        d,
        `calibrated equalised odds (mix ${fit.mixRates.map((r) => r.toFixed(3)).join(', ')})`,
        scores,
        decisions,
      ),
    ],
  }
}

/**
 * Reject-option classification (`rejectOptionClassification` of `mdml-engine/fairness`, Kamiran et al.) on a fitted
 * classifier. Exposition: the group reports before and after, and a `curves` record of balanced accuracy and the
 * fairness metric over thresholds at the chosen margin.
 *
 * @param model - A fitted binary classifier with probabilities.
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, privileged group, metric, bounds, grids and whether the model saw the attribute.
 * @returns The fitted rule and the decisions, with the records.
 */
export function rejectOption(
  model: Fitted,
  data: FairnessData,
  options: MitigationOptions & Omit<RejectOptionOptions, 'privileged'> & { readonly dropSensitive?: boolean } = {},
): Explained<{ readonly fit: RejectOption; readonly decisions: readonly number[] }> {
  const d = mitigationData(data, options, 'rejectOption')
  const x = options.dropSensitive === true ? d.unaware.x : data.x
  const { scores } = positiveScores(model, x)
  const fit = rejectOptionClassification(scores, d.labels, d.groups.index, { ...options, privileged: d.privileged })
  const decisions = rejectOptionDecisions(fit, scores, d.groups.index)
  // The grid along the thresholds at the chosen margin's position in each threshold's row of margins.
  const perThreshold = fit.grid.margins.length / new Set(fit.grid.thresholds).size
  const chosen = fit.grid.thresholds.findIndex((t, i) => t === fit.threshold && fit.grid.margins[i] === fit.margin)
  const position = chosen < 0 ? 0 : chosen % perThreshold
  const atMargin = fit.grid.thresholds.flatMap((_, i) => (i % perThreshold === position ? [i] : []))
  return {
    fit,
    decisions,
    exposition: [
      reportOf(d, 'before', scores),
      reportOf(
        d,
        `reject option (threshold ${fit.threshold.toFixed(2)}, margin ${fit.margin.toFixed(3)})`,
        scores,
        decisions,
      ),
      curveSet({
        x: atMargin.map((i) => fit.grid.thresholds[i] ?? 0),
        xLabel: 'classification threshold',
        yLabel: 'value',
        series: [
          { name: 'balanced accuracy', y: atMargin.map((i) => fit.grid.balancedAccuracy[i] ?? 0) },
          { name: 'fairness metric', y: atMargin.map((i) => fit.grid.metric[i] ?? 0) },
        ],
        marks: [{ x: fit.threshold, label: 'chosen' }],
      }),
    ],
  }
}
