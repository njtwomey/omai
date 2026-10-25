/**
 * In-processing methods: the exponentiated-gradient and grid-search reductions over a fitted learner, with the group
 * reports before and after and the trade-off over iterations.
 */

import { curveSet, type Explained, type Exposition } from '../../lib/exposition/index.ts'
import {
  type ExponentiatedGradientReduction,
  type ExponentiatedGradientReductionOptions,
  exponentiatedGradientReduction,
  type GridPoint,
  gridReduction,
  type ReductionConstraint,
} from '../../lib/fairness/index.ts'
import type { Estimator } from '../../lib/learn/index.ts'
import type { Key } from '../../lib/random/index.ts'
import { logisticRegression } from '../linear/index.ts'
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
 * Fair classification by exponentiated gradient (`exponentiatedGradientReduction` of `whitebox-engine/fairness`) on a
 * dataset with a protected attribute, logistic regression by default. Exposition: the trade-off moving over the
 * iterations (error and violation of the randomised classifier and of each best response, the price on the
 * violation), the path in (violation, error) space from the unconstrained classifier, and the group reports before
 * and after.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - Attribute, constraint, epsilon, iterations, bound, step size, estimator and key.
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
  const epsilon = options.epsilon ?? 0.01
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
      marks: [{ x: epsilon, label: 'ε' }],
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
  /** Multipliers tried, from -bound to bound (default 21). */
  readonly gridSize?: number
  /** The largest multiplier (default 0.5). */
  readonly bound?: number
  readonly estimator?: Estimator
  readonly dropSensitive?: boolean
  readonly key?: Key
}

/**
 * Fair classification by a grid of multipliers (`gridReduction` of `whitebox-engine/fairness`, fairlearn's
 * `GridSearch`) on a dataset with a protected attribute, priced against its largest group. Exposition: a `curves`
 * record of error against violation over the grid.
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
      reference: groupReportReference(groups),
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
      }),
    ],
  }
}
