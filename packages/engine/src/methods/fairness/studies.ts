/**
 * Fairness studies: the accuracy-fairness frontier of a dataset (how much accuracy each amount of unfairness buys,
 * traced by reductions and by thresholds), and the impossibility result of Kleinberg, Mullainathan and Raghavan and of
 * Chouldechova, shown by how calibrated scores break equalised odds as groups' base rates move apart.
 */

import { generate } from '../../lib/data/index.ts'
import { biasedCredit } from '../../lib/datasets/index.ts'
import { curveSet, type Explained, type Exposition, paretoFront } from '../../lib/exposition/index.ts'
import {
  type ImpossibilityPoint,
  impossibilityAt,
  optimiseThresholds,
  paretoFrontier,
  type ReductionConstraint,
  selectFairModel,
  thresholdDecisions,
  thresholdFrontier,
} from '../../lib/fairness/index.ts'
import type { Estimator } from '../../lib/learn/index.ts'
import { foldIn, type Key, key as makeKey } from '../../lib/random/index.ts'
import { type Tensor, toFlat } from '../../lib/tensor/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { buildGroupReport } from './audit.ts'
import {
  binaryLabels,
  type FairnessData,
  positiveScores,
  type SensitiveSpec,
  sensitiveGroups,
  withoutColumns,
} from './groups.ts'
import { gridSearchReduction } from './in-processing.ts'

/** Options of {@link fairnessFrontier}. */
export interface FairnessFrontierOptions {
  readonly sensitive?: SensitiveSpec
  /** The constraint whose violation is traded (default `demographicParity`). */
  readonly constraint?: ReductionConstraint
  /** Multipliers in the reduction's grid (default 41). */
  readonly gridSize?: number
  /** The largest multiplier of the grid (default 2, fairlearn's grid_limit). */
  readonly bound?: number
  /** The violation budget of the model selection (default 0.05). */
  readonly maxViolation?: number
  /** Thresholds tried for the disadvantaged group (default 41). */
  readonly thresholds?: number
  readonly estimator?: Estimator
  readonly dropSensitive?: boolean
  readonly key?: Key
}

/** A point of the frontier: a classifier's accuracy and its fairness violation. */
export interface FrontierPoint {
  readonly accuracy: number
  readonly violation: number
  readonly how: string
}

/**
 * The accuracy–fairness frontier of a dataset: how much accuracy each amount of unfairness buys, traced two ways. In
 * training, fairlearn's grid of multipliers on the constraint's violation (`gridSearchReduction`); after training, the
 * unconstrained model's scores with the disadvantaged group's threshold moved while the others keep theirs
 * (`thresholdFrontier`). Each family is reduced to its Pareto front, and fairness-aware model selection
 * (`selectFairModel`) picks the most accurate candidate within the violation budget. Exposition: a `curves` record of
 * both fronts with the exact threshold optimiser (zero violation in expectation) marked, and a `paretoFront` record of
 * every candidate, both fronts, the optimiser and the choice.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - Constraint, grid, thresholds, budget, estimator and key.
 * @returns Both fronts and the chosen candidate, with the record.
 */
export function fairnessFrontier(
  data: FairnessData,
  options: FairnessFrontierOptions = {},
): Explained<{
  readonly reduction: readonly FrontierPoint[]
  readonly thresholds: readonly FrontierPoint[]
  readonly chosen: FrontierPoint
}> {
  const constraint = options.constraint ?? 'demographicParity'
  const key = options.key ?? makeKey(0)
  const estimator = options.estimator ?? logisticRegression()
  const groups = sensitiveGroups(data, options.sensitive)
  const labels = binaryLabels(data, 'fairnessFrontier')
  const drop = options.dropSensitive === true && groups.column !== undefined
  const grid = gridSearchReduction(data, {
    constraint,
    gridSize: options.gridSize ?? 41,
    ...(options.bound === undefined ? {} : { bound: options.bound }),
    estimator,
    dropSensitive: drop,
    key: foldIn(key, 0),
    ...(options.sensitive === undefined ? {} : { sensitive: options.sensitive }),
  })
  const reductionAll = grid.points.map((p) => ({
    accuracy: 1 - p.error,
    violation: p.violation,
    how: `multiplier ${p.lambdas.map((l) => l.toFixed(2)).join(', ')}`,
  }))
  // Post-processing: the unconstrained scores, the lowest-selected group's threshold swept.
  const x = drop ? withoutColumns(data, [groups.column as number]).x : data.x
  const model = estimator.fit({ x, y: labels }, { key: foldIn(key, 1), explain: false })
  const { scores } = positiveScores(model, x)
  const swept = thresholdFrontier(scores, labels, groups.index, {
    constraint,
    count: options.thresholds ?? 41,
  })
  const thresholdAll = swept.points.map((p, t) => ({
    ...p,
    how: `${groups.names[swept.group] ?? ''} threshold ${(swept.thresholds[t] ?? 0).toFixed(2)}`,
  }))
  const reductionFront = paretoFrontier(reductionAll)
  const optimiser = optimiseThresholds(scores, labels, groups.index, {
    constraint: constraint === 'demographicParity' ? 'demographicParity' : 'equalisedOdds',
  })
  const decisions = thresholdDecisions(optimiser.rules, scores, groups.index)
  const exact = buildGroupReport({
    scores,
    labels,
    groups,
    thresholds: groups.names.map(() => 0.5),
    decisions,
    sweep: 0,
    keepRows: false,
    intervals: false,
  })
  const name = constraint === 'demographicParity' ? 'demographic parity difference' : 'equalised odds difference'
  const exactViolation = exact.disparities.find((d) => d.name === name)?.value ?? 0
  const budget = options.maxViolation ?? 0.05
  const all = [...reductionAll, ...thresholdAll]
  const pick = selectFairModel(all, { maxViolation: budget })
  const family = pick < reductionAll.length ? 0 : 1
  const exposition: Exposition[] = [
    curveSet({
      xLabel: name,
      yLabel: 'accuracy',
      series: [
        {
          name: 'in training (grid of multipliers)',
          x: reductionFront.map((i) => reductionAll[i]?.violation ?? 0),
          y: reductionFront.map((i) => reductionAll[i]?.accuracy ?? 0),
        },
        {
          name: `after training (${groups.names[swept.group] ?? ''} threshold)`,
          x: swept.front.map((i) => thresholdAll[i]?.violation ?? 0),
          y: swept.front.map((i) => thresholdAll[i]?.accuracy ?? 0),
        },
      ],
      marks: [{ x: exactViolation, label: `threshold optimiser: accuracy ${exact.overall.accuracy.toFixed(3)}` }],
    }),
    paretoFront({
      label: 'accuracy–fairness frontier',
      violationName: name,
      families: [
        {
          name: 'in training (grid of multipliers)',
          violation: reductionAll.map((p) => p.violation),
          accuracy: reductionAll.map((p) => p.accuracy),
          labels: reductionAll.map((p) => p.how),
          front: reductionFront,
        },
        {
          name: `after training (${groups.names[swept.group] ?? ''} threshold)`,
          violation: thresholdAll.map((p) => p.violation),
          accuracy: thresholdAll.map((p) => p.accuracy),
          labels: thresholdAll.map((p) => p.how),
          front: [...swept.front],
        },
      ],
      marks: [{ violation: exactViolation, accuracy: exact.overall.accuracy, label: 'threshold optimiser' }],
      chosen: {
        family,
        index: family === 0 ? pick : pick - reductionAll.length,
        rule: `most accurate with violation at most ${budget}`,
      },
    }),
  ]
  return {
    reduction: reductionFront.map((i) => reductionAll[i] as FrontierPoint),
    thresholds: swept.front.map((i) => thresholdAll[i] as FrontierPoint),
    chosen: all[pick] as FrontierPoint,
    exposition,
  }
}

/** Options of {@link impossibilityStudy}. */
export interface ImpossibilityStudyOptions {
  /** The base-rate gaps (qualification gaps of `biasedCredit`) to try (default 0 to 1.5 in 7 steps). */
  readonly gaps?: readonly number[]
  /** Applicants per dataset (default 4000). */
  readonly n?: number
  readonly key?: Key
}

/** One base-rate gap of {@link impossibilityStudy}: `impossibilityAt` of the generated data's true posterior. */
export type ImpossibilityStudyPoint = ImpossibilityPoint & { readonly gap: number }

/**
 * The impossibility of calibration and equalised odds together (Kleinberg, Mullainathan and Raghavan, 2016;
 * Chouldechova, 2017): when groups' base rates differ, perfectly calibrated scores give unequal error rates, and
 * equalising the error rates makes the positive predictive value (what a positive decision means) differ. On
 * `biasedCredit` data without label bias, with a growing real base-rate gap, the true posterior (calibrated within
 * each group by construction) is cut at 0.5, then post-processed to equalised odds. Exposition: a `curves` record of
 * the disparities against the base-rate gap, before and after.
 *
 * @param options - The gaps, the sample size and the key.
 * @returns The points, with the record.
 */
export function impossibilityStudy(
  options: ImpossibilityStudyOptions = {},
): Explained<{ readonly points: readonly ImpossibilityStudyPoint[] }> {
  const gaps = options.gaps ?? [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
  const key = options.key ?? makeKey(0)
  const points = gaps.map((gap, s) => {
    const { data, truth } = generate(biasedCredit, {
      key: foldIn(key, s),
      knobs: { n: options.n ?? 4000, baseRateGap: gap, labelBias: 0, minorityNoise: 1, proxyStrength: 0 },
    })
    const p = toFlat(truth.posterior?.(data.x) as Tensor)
    const scores = Array.from({ length: p.length / 2 }, (_, i) => p[2 * i + 1] ?? 0)
    const labels = Array.from(toFlat(data.y))
    const groups = sensitiveGroups(data)
    return { gap, ...impossibilityAt(scores, labels, groups.index) }
  })
  const x = points.map((q) => Math.abs((q.baseRates[0] ?? 0) - (q.baseRates[1] ?? 0)))
  return {
    points,
    exposition: [
      curveSet({
        x,
        xLabel: 'base-rate gap between the groups',
        yLabel: 'disparity',
        series: [
          { name: 'equalised odds, calibrated scores', y: points.map((q) => q.equalisedOdds), group: 'odds' },
          { name: 'equalised odds, after post-processing', y: points.map((q) => q.equalisedOddsAfter), group: 'odds' },
          { name: 'predictive parity, calibrated scores', y: points.map((q) => q.predictiveParity), group: 'ppv' },
          {
            name: 'predictive parity, after post-processing',
            y: points.map((q) => q.predictiveParityAfter),
            group: 'ppv',
          },
          { name: 'calibration gap within groups', y: points.map((q) => q.calibrationGap), group: 'calibration' },
        ],
      }),
    ],
  }
}
