/**
 * Trade-offs and selection: the accuracy–fairness Pareto frontier of a set of classifiers, fairness-aware model
 * selection (the most accurate within a violation budget, or the best of accuracy minus a priced violation), the
 * frontier traced by moving one group's threshold, and the impossibility of calibration together with equal error
 * rates when base rates differ (Kleinberg, Mullainathan and Raghavan; Chouldechova).
 */

import { WhiteboxError } from '../base/index.ts'
import { type GroupRateValues, groupDisparities, tableRates } from '../exposition/index.ts'
import { calibrationByGroup } from './group-metrics.ts'
import { optimiseThresholds, thresholdDecisions } from './post-processing.ts'

/** A classifier as the frontier sees it: its accuracy and its fairness violation. */
export interface TradeoffPoint {
  readonly accuracy: number
  readonly violation: number
}

/**
 * The accuracy–fairness Pareto frontier: the points no other point beats in both accuracy (higher) and violation
 * (lower), by increasing violation.
 *
 * @param points - Classifiers' accuracies and violations.
 * @returns The indices of the points on the frontier, by increasing violation.
 * @example
 * ```ts
 * import { paretoFrontier } from 'whitebox-engine/fairness'
 * paretoFrontier([{ accuracy: 0.8, violation: 0.2 }, { accuracy: 0.7, violation: 0.25 }, { accuracy: 0.75, violation: 0.05 }]) // => [2, 0]
 * ```
 */
export function paretoFrontier(points: readonly TradeoffPoint[]): number[] {
  const order = points
    .map((_, i) => i)
    .sort(
      (a, b) =>
        (points[a]?.violation ?? 0) - (points[b]?.violation ?? 0) ||
        (points[b]?.accuracy ?? 0) - (points[a]?.accuracy ?? 0),
    )
  const out: number[] = []
  for (const i of order) {
    const last = out[out.length - 1]
    if (last === undefined || (points[i]?.accuracy ?? 0) > (points[last]?.accuracy ?? 0)) out.push(i)
  }
  return out
}

/** Options of {@link selectFairModel}. */
export interface SelectionOptions {
  /** The largest violation allowed: the most accurate point within it wins (default: no budget). */
  readonly maxViolation?: number
  /** Without a budget, the price of a unit of violation: the best accuracy - price x violation wins (default 1). */
  readonly price?: number
}

/**
 * Fairness-aware model selection: the most accurate classifier whose violation is within a budget (the least
 * violating when none is), or, without a budget, the one with the best accuracy minus a priced violation.
 *
 * @param points - Candidates' accuracies and violations (from a grid of models or hyperparameters).
 * @param options - The budget or the price.
 * @returns The index of the chosen candidate.
 * @throws WhiteboxError `value` without candidates.
 * @example
 * ```ts
 * import { selectFairModel } from 'whitebox-engine/fairness'
 * const points = [{ accuracy: 0.8, violation: 0.2 }, { accuracy: 0.75, violation: 0.05 }]
 * selectFairModel(points, { maxViolation: 0.1 }) // => 1
 * selectFairModel(points, { price: 0.1 }) // => 0
 * ```
 */
export function selectFairModel(points: readonly TradeoffPoint[], options: SelectionOptions = {}): number {
  if (points.length === 0) throw new WhiteboxError('value', 'selectFairModel: no candidates')
  const budget = options.maxViolation
  let best = -1
  if (budget !== undefined) {
    for (const [i, p] of points.entries())
      if (p.violation <= budget && (best < 0 || p.accuracy > (points[best]?.accuracy ?? 0))) best = i
    if (best >= 0) return best
    for (const [i, p] of points.entries()) if (best < 0 || p.violation < (points[best]?.violation ?? 0)) best = i
    return best
  }
  const price = options.price ?? 1
  const value = (p: TradeoffPoint): number => p.accuracy - price * p.violation
  for (const [i, p] of points.entries()) if (best < 0 || value(p) > value(points[best] as TradeoffPoint)) best = i
  return best
}

/** Each group's 2 by 2 table from decisions (probabilities of a positive decision allowed). */
function tablesOf(
  labels: readonly number[],
  decisions: readonly number[],
  groups: readonly number[],
  k: number,
): number[][][] {
  const t = Array.from({ length: k }, () => [
    [0, 0],
    [0, 0],
  ])
  for (const [i, g] of groups.entries()) {
    const row = t[g]?.[labels[i] === 1 ? 1 : 0]
    if (row === undefined) continue
    const d = decisions[i] ?? 0
    row[1] = (row[1] ?? 0) + d
    row[0] = (row[0] ?? 0) + 1 - d
  }
  return t
}

/** The fairness violation a frontier trades: the demographic parity or the equalised odds difference. */
export type FrontierConstraint = 'demographicParity' | 'equalisedOdds'

/** Accuracy and violation of per-group tables. */
function pointOf(
  tables: readonly (readonly (readonly number[])[])[],
  constraint: FrontierConstraint,
): TradeoffPoint & { readonly rates: GroupRateValues[] } {
  const rates = tables.map(tableRates)
  const d = groupDisparities(rates, 0)
  const name = constraint === 'demographicParity' ? 'demographic parity difference' : 'equalised odds difference'
  let right = 0
  let n = 0
  for (const t of tables) {
    right += (t[0]?.[0] ?? 0) + (t[1]?.[1] ?? 0)
    n += (t[0]?.[0] ?? 0) + (t[0]?.[1] ?? 0) + (t[1]?.[0] ?? 0) + (t[1]?.[1] ?? 0)
  }
  return { accuracy: n === 0 ? 0 : right / n, violation: d.find((x) => x.name === name)?.value ?? 0, rates }
}

/** Options of {@link thresholdFrontier}. */
export interface ThresholdFrontierOptions {
  readonly constraint?: FrontierConstraint
  /** The group whose threshold moves (default the one selected least at the base threshold). */
  readonly group?: number
  /** The others' threshold (default 0.5). */
  readonly base?: number
  /** Thresholds tried for the moving group, from 0.05 to 0.95 (default 41). */
  readonly count?: number
}

/** The points of {@link thresholdFrontier}. */
export interface ThresholdFrontier {
  readonly group: number
  readonly thresholds: readonly number[]
  readonly points: readonly TradeoffPoint[]
  /** Indices of the points on the Pareto frontier. */
  readonly front: readonly number[]
}

/**
 * The accuracy–fairness trade-off after training: one group's threshold moved while the others keep the base
 * threshold, each threshold's accuracy and violation, and the Pareto frontier among them.
 *
 * @param scores - Scores `[n]`.
 * @param labels - Binary labels `[n]`.
 * @param groups - Group indices `[n]`.
 * @param options - Constraint, moving group, base threshold and count.
 * @returns The thresholds, their points and the frontier.
 * @example
 * ```ts
 * import { thresholdFrontier } from 'whitebox-engine/fairness'
 * const r = thresholdFrontier([0.9, 0.7, 0.4, 0.2, 0.6, 0.45, 0.3, 0.1], [1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1], { count: 5 })
 * r.group // => 1
 * r.points.length // => 5
 * ```
 */
export function thresholdFrontier(
  scores: readonly number[],
  labels: readonly number[],
  groups: readonly number[],
  options: ThresholdFrontierOptions = {},
): ThresholdFrontier {
  if (scores.length !== labels.length || scores.length !== groups.length)
    throw new WhiteboxError('shape', 'thresholdFrontier: inputs differ in length')
  const k = Math.max(...groups) + 1
  const base = options.base ?? 0.5
  const constraint = options.constraint ?? 'demographicParity'
  const cut = (thresholds: readonly number[]): number[] =>
    scores.map((s, i) => (s >= (thresholds[groups[i] ?? 0] ?? base) ? 1 : 0))
  let group = options.group ?? -1
  if (group < 0) {
    const at = pointOf(tablesOf(labels, cut(new Array<number>(k).fill(base)), groups, k), constraint)
    group = 0
    for (const [g, r] of at.rates.entries()) if (r.selectionRate < (at.rates[group]?.selectionRate ?? 1)) group = g
  }
  const count = options.count ?? 41
  const thresholds = Array.from({ length: count }, (_, t) => 0.05 + (0.9 * t) / Math.max(1, count - 1))
  const points = thresholds.map((t) => {
    const p = pointOf(
      tablesOf(labels, cut(Array.from({ length: k }, (_, g) => (g === group ? t : base))), groups, k),
      constraint,
    )
    return { accuracy: p.accuracy, violation: p.violation }
  })
  return { group, thresholds, points, front: paretoFrontier(points) }
}

/** What calibrated scores give at one threshold, and after equalising odds. */
export interface ImpossibilityPoint {
  readonly baseRates: readonly number[]
  /** The largest gap between groups' expected calibration errors. */
  readonly calibrationGap: number
  readonly equalisedOdds: number
  readonly predictiveParity: number
  /** After the threshold optimiser for equalised odds. */
  readonly equalisedOddsAfter: number
  readonly predictiveParityAfter: number
}

/**
 * The impossibility of calibration with equal error rates (Kleinberg, Mullainathan and Raghavan, 2016; Chouldechova,
 * 2017) on one set of scores: their calibration gap between groups and, cut at a threshold, their equalised-odds and
 * predictive-parity differences; then the same after post-processing to equalised odds (the threshold optimiser),
 * which closes the error-rate gap and opens the predictive-parity gap whenever the base rates differ.
 *
 * @param scores - Scores, calibrated within groups for the result to mean what it says `[n]`.
 * @param labels - Binary labels `[n]`.
 * @param groups - Group indices `[n]`.
 * @param threshold - The threshold (default 0.5).
 * @returns The base rates and the disparities before and after.
 * @example
 * ```ts
 * import { impossibilityAt } from 'whitebox-engine/fairness'
 * const r = impossibilityAt([0.8, 0.8, 0.8, 0.2, 0.8, 0.2, 0.2, 0.2], [1, 1, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1])
 * r.baseRates // => [0.5, 0.25]
 * ```
 */
export function impossibilityAt(
  scores: readonly number[],
  labels: readonly number[],
  groups: readonly number[],
  threshold = 0.5,
): ImpossibilityPoint {
  const k = Math.max(...groups) + 1
  const named = (tables: number[][][], name: string): number =>
    groupDisparities(tables.map(tableRates), 0).find((d) => d.name === name)?.value ?? 0
  const before = tablesOf(
    labels,
    scores.map((s) => (s >= threshold ? 1 : 0)),
    groups,
    k,
  )
  const opt = optimiseThresholds(scores, labels, groups, { constraint: 'equalisedOdds' })
  const after = tablesOf(labels, thresholdDecisions(opt.rules, scores, groups), groups, k)
  return {
    baseRates: before.map((t) => tableRates(t).baseRate),
    calibrationGap: calibrationByGroup(labels, scores, groups, { nBins: 10 }).estimate.value,
    equalisedOdds: named(before, 'equalised odds difference'),
    predictiveParity: named(before, 'predictive parity difference'),
    equalisedOddsAfter: named(after, 'equalised odds difference'),
    predictiveParityAfter: named(after, 'predictive parity difference'),
  }
}
