/**
 * Fair classification during training on a dataset with a protected attribute, wrapping the reductions of
 * `mdml-engine/fairness`: exponentiated gradient, which yields a randomised classifier meeting a demographic-parity
 * or equalised-odds constraint, and a grid search over multipliers, each audited before and after (as in fairlearn).
 */

import { curveSet, type Explained, type Exposition } from '../../lib/exposition/index.ts'
import {
  type AdversarialDebiasing,
  type AdversarialDebiasingOptions,
  adversarialDebiasing,
  type ExponentiatedGradientReduction,
  type ExponentiatedGradientReductionOptions,
  exponentiatedGradientReduction,
  type FairConstrainedLogistic,
  type FairConstrainedOptions,
  fairConstrainedLogistic,
  type GerryFair,
  type GerryFairOptions,
  type GridPoint,
  gerryFairClassifier,
  gridReduction,
  type LinearClassifier,
  linearProbabilities,
  type MetaFair,
  type MetaFairOptions,
  metaFairClassifier,
  type PrejudiceRemover,
  type PrejudiceRemoverOptions,
  prejudiceRemover,
  type ReductionConstraint,
} from '../../lib/fairness/index.ts'
import type { Estimator } from '../../lib/learn/index.ts'
import type { Key } from '../../lib/random/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { baselineScores, type MitigationOptions, mitigationData, reportOf } from './_mitigation.ts'
import { buildGroupReport, groupReportReference } from './audit.ts'
import { binaryLabels, type FairnessData, type SensitiveSpec, sensitiveGroups, withoutColumns } from './groups.ts'

/** Options of {@link exponentiatedGradient}. */
export interface ExponentiatedGradientOptions extends ExponentiatedGradientReductionOptions {
  readonly sensitive?: SensitiveSpec
  /** The classifier fitted in each best response (default logistic regression); it must accept sample weights. */
  readonly estimator?: Estimator
  /** Fit without the sensitive column (default false). */
  readonly dropSensitive?: boolean
  readonly reference?: number | string
}

/** The result of {@link exponentiatedGradient}. */
export interface ExponentiatedGradient extends ExponentiatedGradientReduction {
  readonly groups: readonly string[]
}

/**
 * Fair classification by exponentiated gradient (`exponentiatedGradientReduction` of `mdml-engine/fairness`) on a
 * dataset with a protected attribute, logistic regression by default. Exposition: the trade-off moving over the
 * iterations (error and violation of the randomised classifier and of each best response, the price on the
 * violation), the path in (violation, error) space from the unconstrained classifier, and the group reports before
 * and after.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - Attribute, constraint, epsilon, difference bound, iterations, step size, estimator and key.
 * @returns The members, decisions and steps, with the records.
 */
export function exponentiatedGradient(
  data: FairnessData,
  options: ExponentiatedGradientOptions = {},
): Explained<ExponentiatedGradient> {
  const groups = sensitiveGroups(data, options.sensitive)
  const labels = binaryLabels(data, 'exponentiatedGradient')
  const x =
    options.dropSensitive === true && groups.column !== undefined ? withoutColumns(data, [groups.column]).x : data.x
  const constraint = options.constraint ?? 'demographicParity'
  const allowed = options.differenceBound ?? 0.01
  const r = exponentiatedGradientReduction(
    options.estimator ?? logisticRegression(),
    { x, labels, groups: groups.index },
    options,
  )
  const { steps, baseline, decisions } = r
  const reference = groupReportReference(groups, options.reference)
  const classes = data.classes?.length === 2 ? ([data.classes[0], data.classes[1]] as [string, string]) : undefined
  const thresholds = groups.names.map(() => 0.5)
  const common = { labels, groups, reference, thresholds, ...(classes === undefined ? {} : { classes }) }
  const exposition: Exposition[] = [
    curveSet({
      x: steps.map((s) => s.iteration),
      xLabel: 'iteration',
      yLabel: 'rate',
      y2Label: 'price on the violation',
      series: [
        { name: 'error (randomised classifier)', y: steps.map((s) => s.averageError), group: 'error' },
        { name: 'error (best response)', y: steps.map((s) => s.error), group: 'error' },
        { name: 'violation (randomised classifier)', y: steps.map((s) => s.averageViolation), group: 'violation' },
        { name: 'violation (best response)', y: steps.map((s) => s.violation), group: 'violation' },
        { name: 'price', y: steps.map((s) => s.lambdaTotal), axis: 'right', group: 'price' },
      ],
    }),
    curveSet({
      xLabel: 'constraint violation',
      yLabel: 'error',
      series: [
        {
          name: 'randomised classifier, iteration by iteration',
          x: [baseline.violation, ...steps.map((s) => s.averageViolation)],
          y: [baseline.error, ...steps.map((s) => s.averageError)],
        },
      ],
      marks: [{ x: allowed, label: 'allowed violation' }],
    }),
    buildGroupReport({ ...common, scores: baseline.scores, label: 'unconstrained' }),
    buildGroupReport({
      ...common,
      scores: decisions,
      decisions,
      label: `exponentiated gradient (${constraint})`,
      sweep: 0,
    }),
  ]
  return { ...r, groups: groups.names, exposition }
}

/** Options of {@link gridSearchReduction}. */
export interface GridSearchReductionOptions {
  readonly sensitive?: SensitiveSpec
  /** The constraint (default `demographicParity`). */
  readonly constraint?: ReductionConstraint
  /** Multiplier vectors tried (default 10, as fairlearn). */
  readonly gridSize?: number
  /** The largest multiplier (default 2, fairlearn's grid_limit). */
  readonly bound?: number
  readonly estimator?: Estimator
  readonly dropSensitive?: boolean
  readonly key?: Key
}

/**
 * Fair classification by a grid of multipliers (`gridReduction` of `mdml-engine/fairness`, fairlearn's
 * `GridSearch`) on a dataset with a protected attribute. Exposition: a `curves` record of error against violation over
 * the grid, with fairlearn's choice (the least half error plus half violation) marked.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - Attribute, constraint, grid size, bound, estimator and key.
 * @returns The classifiers with their error and violation, with the record.
 */
export function gridSearchReduction(
  data: FairnessData,
  options: GridSearchReductionOptions = {},
): Explained<{ readonly points: readonly GridPoint[] }> {
  const groups = sensitiveGroups(data, options.sensitive)
  const labels = binaryLabels(data, 'gridSearchReduction')
  const x =
    options.dropSensitive === true && groups.column !== undefined ? withoutColumns(data, [groups.column]).x : data.x
  const points = gridReduction(
    options.estimator ?? logisticRegression(),
    { x, labels, groups: groups.index },
    {
      ...(options.constraint === undefined ? {} : { constraint: options.constraint }),
      ...(options.gridSize === undefined ? {} : { gridSize: options.gridSize }),
      ...(options.bound === undefined ? {} : { bound: options.bound }),
      ...(options.key === undefined ? {} : { key: options.key }),
    },
  )
  const sorted = [...points].sort((a, b) => a.violation - b.violation)
  return {
    points,
    exposition: [
      curveSet({
        xLabel: 'constraint violation',
        yLabel: 'error',
        series: [{ name: 'grid of multipliers', x: sorted.map((p) => p.violation), y: sorted.map((p) => p.error) }],
        marks: points.flatMap((p) =>
          p.best ? [{ x: p.violation, label: `chosen: error ${p.error.toFixed(3)}` }] : [],
        ),
      }),
    ],
  }
}

/** A linear classifier fitted under a fairness term, with its scores on the rows. */
export interface FairLinearFit {
  readonly scores: readonly number[]
}

/**
 * The prejudice remover (`prejudiceRemover` of `mdml-engine/fairness`) on a dataset: one logistic regression per
 * group, penalised by eta times the mutual information between decision and group. Exposition: the group reports
 * with eta 0 and at the eta asked.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, eta, L2 penalty and key.
 * @returns The fit and its scores, with the records.
 */
export function prejudiceRemoverFit(
  data: FairnessData,
  options: MitigationOptions & PrejudiceRemoverOptions = {},
): Explained<FairLinearFit & { readonly fit: PrejudiceRemover }> {
  const d = mitigationData(data, options, 'prejudiceRemoverFit')
  const scoresOf = (fit: PrejudiceRemover): number[] =>
    d.rows.map((r, i) => linearProbabilities(fit.models[d.groups.index[i] ?? 0] as LinearClassifier, [r])[0] ?? 0)
  const plain = prejudiceRemover(d.rows, d.labels, d.groups.index, { ...options, eta: 0 })
  const fit = prejudiceRemover(d.rows, d.labels, d.groups.index, options)
  const scores = scoresOf(fit)
  return {
    fit,
    scores,
    exposition: [
      reportOf(d, `eta 0 (prejudice index ${plain.prejudiceIndex.toFixed(4)})`, scoresOf(plain)),
      reportOf(d, `eta ${options.eta ?? 1} (prejudice index ${fit.prejudiceIndex.toFixed(4)})`, scores),
    ],
  }
}

/**
 * Adversarial debiasing (`adversarialDebiasing` of `mdml-engine/fairness`) on a dataset: a logistic predictor
 * trained against an adversary recovering the group from its output. Exposition: the group reports of the predictor
 * without the adversary (weight 0) and with it, and a `curves` record of both losses over training.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, adversary weight, constraint, steps and key.
 * @returns The fit and its scores, with the records.
 */
export function adversarialDebiasingFit(
  data: FairnessData,
  options: MitigationOptions & Omit<AdversarialDebiasingOptions, 'key'> = {},
): Explained<FairLinearFit & { readonly fit: AdversarialDebiasing }> {
  const d = mitigationData(data, options, 'adversarialDebiasingFit')
  const plain = adversarialDebiasing(d.rows, d.labels, d.binary, { ...options, adversaryWeight: 0, key: d.key })
  const fit = adversarialDebiasing(d.rows, d.labels, d.binary, { ...options, key: d.key })
  const scores = linearProbabilities(fit.model, d.rows)
  return {
    fit,
    scores,
    exposition: [
      reportOf(d, 'without the adversary', linearProbabilities(plain.model, d.rows)),
      reportOf(d, 'against the adversary', scores),
      curveSet({
        x: fit.steps.map((s) => s.step),
        xLabel: 'step',
        yLabel: 'log loss',
        series: [
          { name: 'predictor', y: fit.steps.map((s) => s.predictorLoss) },
          { name: 'adversary (ln 2 is chance)', y: fit.steps.map((s) => s.adversaryLoss) },
        ],
      }),
    ],
  }
}

/**
 * Fairness-constrained logistic regression (`fairConstrainedLogistic` of `mdml-engine/fairness`, Zafar et al.) on a
 * dataset. Exposition: the group reports of the unconstrained and the constrained fit.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, covariance bound, L2 penalty and key.
 * @returns The fit and its scores, with the records.
 */
export function fairConstrainedFit(
  data: FairnessData,
  options: MitigationOptions & FairConstrainedOptions = {},
): Explained<FairLinearFit & { readonly fit: FairConstrainedLogistic }> {
  const d = mitigationData(data, options, 'fairConstrainedFit')
  const fit = fairConstrainedLogistic(d.rows, d.labels, d.binary, options)
  const scores = linearProbabilities(fit.model, d.rows)
  return {
    fit,
    scores,
    exposition: [
      reportOf(d, `unconstrained (covariance ${fit.unconstrainedCovariance.toFixed(4)})`, baselineScores(d)),
      reportOf(d, `covariance at most ${options.covariance ?? 0}`, scores),
    ],
  }
}

/**
 * The meta fair classifier for the statistical rate (`metaFairClassifier` of `mdml-engine/fairness`) on a
 * dataset: a logistic regression's posteriors thresholded per group to the most accurate decisions whose selection-rate
 * ratio is at least tau. Exposition: the group reports at 0.5 and at the per-group thresholds.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, tau and key.
 * @returns The thresholds and decisions, with the records.
 */
export function metaFairFit(
  data: FairnessData,
  options: MitigationOptions & MetaFairOptions = {},
): Explained<{ readonly fit: MetaFair; readonly decisions: readonly number[] }> {
  const d = mitigationData(data, options, 'metaFairFit')
  const scores = baselineScores(d)
  const fit = metaFairClassifier(scores, d.groups.index, options)
  const thresholds = fit.thresholds.map((t) => (Number.isFinite(t) ? t : 1))
  const decisions = scores.map((s, i) => (s >= (fit.thresholds[d.groups.index[i] ?? 0] ?? 0.5) ? 1 : 0))
  return {
    fit,
    decisions,
    exposition: [
      reportOf(d, 'posterior at 0.5', scores),
      reportOf(d, `statistical rate at least ${options.tau ?? 0.8}`, scores, undefined, thresholds),
    ],
  }
}

/**
 * GerryFair (`gerryFairClassifier` of `mdml-engine/fairness`) on a dataset, the protected attribute's indicators as
 * the subgroup features. Exposition: the group reports of a logistic regression and of GerryFair's average classifier,
 * and a `curves` record of the error and the worst subgroup violation over the rounds.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, rate, C, gamma, rounds and key.
 * @returns The fit and its decisions, with the records.
 */
export function gerryFairFit(
  data: FairnessData,
  options: MitigationOptions & GerryFairOptions = {},
): Explained<{ readonly fit: GerryFair }> {
  const d = mitigationData(data, options, 'gerryFairFit')
  const indicators = d.groups.index.map((g) => d.groups.names.slice(1).map((_, k) => (g === k + 1 ? 1 : 0)))
  const fit = gerryFairClassifier(d.rows, indicators, d.labels, options)
  return {
    fit,
    exposition: [
      reportOf(d, 'logistic regression', baselineScores(d)),
      reportOf(d, 'GerryFair (average classifier)', fit.decisions, fit.decisions),
      curveSet({
        x: fit.steps.map((s) => s.iteration),
        xLabel: 'round',
        yLabel: 'rate',
        series: [
          { name: 'error', y: fit.steps.map((s) => s.error) },
          { name: 'worst subgroup violation', y: fit.steps.map((s) => s.violation) },
        ],
      }),
    ],
  }
}
