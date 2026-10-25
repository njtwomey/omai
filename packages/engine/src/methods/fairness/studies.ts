/**
 * Fairness studies: the accuracy-fairness frontier of a dataset (how much accuracy each amount of unfairness buys,
 * traced by reductions and by thresholds), and the impossibility result of Kleinberg, Mullainathan and Raghavan and of
 * Chouldechova, shown by how calibrated scores break equalised odds as groups' base rates move apart.
 */

import { generate } from '../../lib/data/index.ts'
import { biasedCredit } from '../../lib/datasets/index.ts'
import { curveSet, type Explained, type Exposition } from '../../lib/exposition/index.ts'
import {
  calibrationByGroup,
  optimiseThresholds,
  type ReductionConstraint,
  thresholdDecisions,
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
  /** The largest multiplier of the grid (default 0.5). */
  readonly bound?: number
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

/** The upper-left frontier of points: those no other point beats in both accuracy and violation, by violation. */
function paretoFront(points: readonly FrontierPoint[]): FrontierPoint[] {
  const sorted = [...points].sort((a, b) => a.violation - b.violation || b.accuracy - a.accuracy)
  const out: FrontierPoint[] = []
  for (const p of sorted) if (out.length === 0 || p.accuracy > (out[out.length - 1]?.accuracy ?? 0)) out.push(p)
  return out
}

/**
 * The accuracy–fairness frontier of a dataset: how much accuracy each amount of unfairness buys, traced two ways. In
 * training, a grid of prices on the constraint's violation (`gridSearchReduction`); after training, the unconstrained
 * model's scores with the disadvantaged group's threshold moved while the others keep theirs. Each family is reduced
 * to its Pareto front. Exposition: a `curves` record of accuracy against violation, one series per family, with the
 * exact threshold optimiser's violation (zero in expectation) marked.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - Constraint, grid, thresholds, estimator and key.
 * @returns Both fronts, with the record.
 */
export function fairnessFrontier(
  data: FairnessData,
  options: FairnessFrontierOptions = {},
): Explained<{ readonly reduction: readonly FrontierPoint[]; readonly thresholds: readonly FrontierPoint[] }> {
  const constraint = options.constraint ?? 'demographicParity'
  const key = options.key ?? makeKey(0)
  const estimator = options.estimator ?? logisticRegression()
  const groups = sensitiveGroups(data, options.sensitive)
  const labels = binaryLabels(data, 'fairnessFrontier')
  const drop = options.dropSensitive === true && groups.column !== undefined
  const grid = gridSearchReduction(data, {
    constraint,
    gridSize: options.gridSize ?? 41,
    bound: options.bound ?? 0.5,
    estimator,
    dropSensitive: drop,
    key: foldIn(key, 0),
    ...(options.sensitive === undefined ? {} : { sensitive: options.sensitive }),
  })
  const reduction = paretoFront(
    grid.points.map((p) => ({
      accuracy: 1 - p.error,
      violation: p.violation,
      how: `multiplier ${p.lambda.toFixed(2)}`,
    })),
  )
  // Post-processing: the unconstrained scores, the lowest-selected group's threshold swept.
  const x = drop ? withoutColumns(data, [groups.column as number]).x : data.x
  const model = estimator.fit({ x, y: labels }, { key: foldIn(key, 1), explain: false })
  const { scores } = positiveScores(model, x)
  const base = buildGroupReport({
    scores,
    labels,
    groups,
    thresholds: groups.names.map(() => 0.5),
    sweep: 0,
    keepRows: false,
  })
  let target = 0
  for (const [g, r] of base.rates.entries()) if (r.selectionRate < (base.rates[target]?.selectionRate ?? 1)) target = g
  const count = options.thresholds ?? 41
  const violationOf = (r: typeof base): number => {
    const name = constraint === 'demographicParity' ? 'demographic parity difference' : 'equalised odds difference'
    return r.disparities.find((d) => d.name === name)?.value ?? 0
  }
  const swept = Array.from({ length: count }, (_, t) => {
    const cut = 0.05 + (0.9 * t) / Math.max(1, count - 1)
    const thresholds = groups.names.map((_, g) => (g === target ? cut : 0.5))
    const r = buildGroupReport({ scores, labels, groups, thresholds, sweep: 0, keepRows: false })
    return { accuracy: r.overall.accuracy, violation: violationOf(r), how: `threshold ${cut.toFixed(2)}` }
  })
  const front = paretoFront(swept)
  const optimiser = optimiseThresholds(scores, labels, groups.index, {
    constraint: constraint === 'demographicParity' ? 'demographicParity' : 'equalisedOdds',
  })
  const decisions = thresholdDecisions(optimiser.rules, scores, groups.index)
  const exact = buildGroupReport({
    scores,
    labels,
    groups,
    thresholds: base.thresholds,
    decisions,
    sweep: 0,
    keepRows: false,
  })
  const exposition: Exposition[] = [
    curveSet({
      xLabel: constraint === 'demographicParity' ? 'demographic parity difference' : 'equalised odds difference',
      yLabel: 'accuracy',
      series: [
        {
          name: 'in training (grid of multipliers)',
          x: reduction.map((p) => p.violation),
          y: reduction.map((p) => p.accuracy),
        },
        {
          name: `after training (${groups.names[target] ?? ''} threshold)`,
          x: front.map((p) => p.violation),
          y: front.map((p) => p.accuracy),
        },
      ],
      marks: [{ x: violationOf(exact), label: `threshold optimiser: accuracy ${exact.overall.accuracy.toFixed(3)}` }],
    }),
  ]
  return { reduction, thresholds: front, exposition }
}

/** Options of {@link impossibilityStudy}. */
export interface ImpossibilityStudyOptions {
  /** The base-rate gaps (qualification gaps of `biasedCredit`) to try (default 0 to 1.5 in 7 steps). */
  readonly gaps?: readonly number[]
  /** Applicants per dataset (default 4000). */
  readonly n?: number
  readonly key?: Key
}

/** One base-rate gap of {@link impossibilityStudy}. */
export interface ImpossibilityPoint {
  readonly gap: number
  /** The groups' base rates. */
  readonly baseRates: readonly number[]
  /** Calibrated scores at one threshold: the within-group calibration gap and the error-rate disparities. */
  readonly calibrationGap: number
  readonly equalisedOdds: number
  readonly predictiveParity: number
  /** After equalising odds by post-processing: the disparities again. */
  readonly equalisedOddsAfter: number
  readonly predictiveParityAfter: number
}

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
): Explained<{ readonly points: readonly ImpossibilityPoint[] }> {
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
    const before = buildGroupReport({ scores, labels, groups, thresholds: [0.5, 0.5], sweep: 0, keepRows: false })
    const opt = optimiseThresholds(scores, labels, groups.index, { constraint: 'equalisedOdds' })
    const decisions = thresholdDecisions(opt.rules, scores, groups.index)
    const after = buildGroupReport({
      scores,
      labels,
      groups,
      thresholds: [0.5, 0.5],
      decisions,
      sweep: 0,
      keepRows: false,
    })
    const named = (r: typeof before, name: string): number => r.disparities.find((d) => d.name === name)?.value ?? 0
    return {
      gap,
      baseRates: before.rates.map((r) => r.baseRate),
      calibrationGap: calibrationByGroup(labels, scores, groups.index, { nBins: 10 }).estimate.value,
      equalisedOdds: named(before, 'equalised odds difference'),
      predictiveParity: named(before, 'predictive parity difference'),
      equalisedOddsAfter: named(after, 'equalised odds difference'),
      predictiveParityAfter: named(after, 'predictive parity difference'),
    }
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
