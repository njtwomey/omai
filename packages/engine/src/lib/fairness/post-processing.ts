/**
 * Fairness after training. Threshold adjustment (Hardt, Price and Srebro, 2016; fairlearn's `ThresholdOptimizer`):
 * per-group randomised thresholds on a score that satisfy a constraint such as demographic parity or equalised odds
 * exactly in expectation while maximising accuracy or another objective, and the decisions they give each row;
 * calibrated equalised odds (Pleiss et al., 2017), which mixes the cheaper group's scores with its base rate; and
 * reject-option classification (Kamiran et al., 2012), which decides near the threshold in favour of the unprivileged
 * group (both as AIF360).
 */

import { MdmlError, maxOf, minOf } from '../base/index.ts'

/** The fairness constraint a threshold optimiser equalises across groups. */
export type ThresholdConstraint =
  | 'demographicParity'
  | 'equalisedOdds'
  | 'truePositiveRateParity'
  | 'falsePositiveRateParity'

/** What a threshold optimiser maximises. */
export type ThresholdObjective = 'accuracy' | 'balancedAccuracy'

/** A threshold rule: `score > threshold` (or `score < threshold` when flipped). */
export interface ThresholdRule {
  readonly operator: '>' | '<'
  readonly threshold: number
}

/**
 * One group's randomised classifier: rule 0 with probability `p0`, rule 1 with `p1`; for equalised odds, with
 * probability `pIgnore` instead a constant positive decision with probability `constant`.
 */
export interface GroupThresholdRule {
  readonly p0: number
  readonly rule0: ThresholdRule
  readonly p1: number
  readonly rule1: ThresholdRule
  readonly pIgnore?: number
  readonly constant?: number
}

interface Counts {
  readonly tp: number
  readonly fp: number
  readonly tn: number
  readonly fn: number
}

type RateMetric = 'selectionRate' | 'truePositiveRate' | 'falsePositiveRate'
type CurveMetric = RateMetric | ThresholdObjective

const METRIC: Readonly<Record<CurveMetric, (c: Counts) => number>> = {
  selectionRate: (c) => (c.tp + c.fp) / (c.tp + c.fp + c.tn + c.fn),
  truePositiveRate: (c) => c.tp / (c.tp + c.fn),
  falsePositiveRate: (c) => c.fp / (c.fp + c.tn),
  accuracy: (c) => (c.tp + c.tn) / (c.tp + c.fp + c.tn + c.fn),
  balancedAccuracy: (c) => (0.5 * c.tp) / (c.tp + c.fn) + (0.5 * c.tn) / (c.tn + c.fp),
}

interface TradeoffPoint {
  readonly x: number
  readonly y: number
  readonly rule: ThresholdRule
}

/**
 * Every threshold of one group's scores as a point (x metric, y metric), sorted by x then y (fairlearn's
 * `_calculate_tradeoff_points`): thresholds halfway between consecutive distinct scores, from +infinity down.
 */
function tradeoffPoints(
  scores: readonly number[],
  labels: readonly number[],
  xMetric: CurveMetric,
  yMetric: CurveMetric,
  flip: boolean,
  group: string,
): TradeoffPoint[] {
  const order = scores.map((_, i) => i).sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0))
  const n = order.length
  const pos = labels.reduce((s, v) => s + (v === 1 ? 1 : 0), 0)
  const neg = n - pos
  if (pos === 0 || neg === 0) throw new MdmlError('value', `thresholdOptimizer: group ${group} has one class only`)
  const sorted = [...order.map((i) => scores[i] ?? 0), Number.NEGATIVE_INFINITY]
  const out: TradeoffPoint[] = []
  const count = [0, 0]
  let i = 0
  let first = true
  while (i < n) {
    let threshold = Number.POSITIVE_INFINITY
    if (!first) {
      const t = sorted[i] ?? 0
      while (sorted[i] === t) {
        const label = labels[order[i] ?? 0] === 1 ? 1 : 0
        count[label] = (count[label] ?? 0) + 1
        i++
      }
      threshold = (t + (sorted[i] ?? 0)) / 2
    }
    first = false
    const [fp, tp] = [count[0] ?? 0, count[1] ?? 0]
    const ops: [ThresholdRule['operator'], Counts][] = [['>', { tp, fp, tn: neg - fp, fn: pos - tp }]]
    if (flip) ops.push(['<', { tp: pos - tp, fp: neg - fp, tn: fp, fn: tp }])
    for (const [operator, c] of ops)
      out.push({ x: METRIC[xMetric](c), y: METRIC[yMetric](c), rule: { operator, threshold } })
  }
  return out.sort((a, b) => a.x - b.x || a.y - b.y)
}

/** The upper convex hull of points sorted by x (Andrew's monotone chain, as fairlearn). */
function upperHull(points: readonly TradeoffPoint[]): TradeoffPoint[] {
  const hull: TradeoffPoint[] = []
  for (const r2 of points) {
    while (hull.length >= 2) {
      const r1 = hull[hull.length - 1] as TradeoffPoint
      const r0 = hull[hull.length - 2] as TradeoffPoint
      if ((r1.y - r0.y) * (r2.x - r0.x) <= (r2.y - r0.y) * (r1.x - r0.x)) hull.pop()
      else break
    }
    hull.push(r2)
  }
  return hull
}

interface Interpolated {
  readonly y: number[]
  readonly p0: number[]
  readonly rule0: ThresholdRule[]
  readonly p1: number[]
  readonly rule1: ThresholdRule[]
}

/** The hull interpolated on a grid of x, each grid point a mixture of two hull points (fairlearn's `_interpolate_curve`). */
function interpolate(hull: readonly TradeoffPoint[], grid: readonly number[]): Interpolated {
  const xs = hull.map((p) => p.x)
  const out: Interpolated = { y: [], p0: [], rule0: [], p1: [], rule1: [] }
  for (const [g, x] of grid.entries()) {
    // The last hull point with x_k <= x (searchsorted right, minus one), stepped back when it equals x.
    let k = -1
    while (k + 1 < xs.length && (xs[k + 1] ?? 0) <= x) k++
    if (g > 0 && xs[k] === x) k--
    k = Math.max(0, Math.min(k, xs.length - 2))
    const a = hull[k] as TradeoffPoint
    const b = hull[k + 1] as TradeoffPoint
    const p0 = (b.x - x) / (b.x - a.x)
    out.y.push(p0 * a.y + (1 - p0) * b.y)
    out.p0.push(p0)
    out.rule0.push(a.rule)
    out.p1.push(1 - p0)
    out.rule1.push(b.rule)
  }
  return out
}

/** Options of {@link optimiseThresholds}. */
export interface OptimiseThresholdsOptions {
  /** The constraint (default `demographicParity`). */
  readonly constraint?: ThresholdConstraint
  /** The objective (default `accuracy`). */
  readonly objective?: ThresholdObjective
  /** Points in the grid of the constrained rate (default 1000, fairlearn's `grid_size`). */
  readonly gridSize?: number
  /** Also consider flipped rules, `score < threshold` (default false). */
  readonly flip?: boolean
}

/** The result of {@link optimiseThresholds}. */
export interface OptimisedThresholds {
  readonly constraint: ThresholdConstraint
  readonly objective: ThresholdObjective
  /** One randomised rule per group. */
  readonly rules: readonly GroupThresholdRule[]
  /** The chosen value of the equalised rate (the false-positive rate for equalised odds). */
  readonly xBest: number
  /** The objective there (the shared true-positive rate for equalised odds). */
  readonly yBest: number
  /** The grid, each group's trade-off curve on it, and the overall curve the best point maximises. */
  readonly grid: readonly number[]
  readonly curves: readonly (readonly number[])[]
  readonly overall: readonly number[]
  /** The rules that put every group at another grid point of the constrained rate (the nearest), not the best one. */
  readonly rulesAt: (x: number) => GroupThresholdRule[]
}

/**
 * Per-group randomised thresholds that satisfy a fairness constraint exactly (in expectation) while maximising an
 * objective (Hardt, Price and Srebro, 2016; fairlearn's `ThresholdOptimizer`). For a rate parity (demographic parity:
 * selection rate; true- or false-positive rate parity), each group's achievable (rate, objective) points over all
 * thresholds are reduced to their upper convex hull and interpolated on a grid of the rate; the rate maximising the
 * groups' weighted objective is chosen and each group mixes the two hull thresholds that reach it. For equalised odds,
 * each group's ROC hull is interpolated on a grid of false-positive rates, the shared true-positive rate is the
 * lowest group's, the objective picks the false-positive rate, and groups above it mix in a constant decision.
 *
 * @param scores - Each row's score (higher is more positive).
 * @param labels - Each row's binary label.
 * @param groups - Each row's group index.
 * @param options - Constraint, objective, grid size and flipping.
 * @returns The rules, the chosen point and the trade-off curves.
 * @throws MdmlError `value` when a group has one class only.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { optimiseThresholds } from 'mdml-engine/fairness'
 * const r = optimiseThresholds([0.1, 0.3, 0.6, 0.8], [0, 0, 1, 1], [0, 0, 0, 0], { gridSize: 100 })
 * r.yBest // => 1
 * ```
 */
export function optimiseThresholds(
  scores: readonly number[],
  labels: readonly number[],
  groups: readonly number[],
  options: OptimiseThresholdsOptions = {},
): OptimisedThresholds {
  const constraint = options.constraint ?? 'demographicParity'
  const objective = options.objective ?? 'accuracy'
  const size = options.gridSize ?? 1000
  const grid = Array.from({ length: size + 1 }, (_, i) => i / size)
  const k = maxOf(groups) + 1
  const members = Array.from({ length: k }, (): number[] => [])
  for (const [i, g] of groups.entries()) members[g]?.push(i)
  const n = scores.length
  const eo = constraint === 'equalisedOdds'
  const xMetric: RateMetric =
    constraint === 'demographicParity'
      ? 'selectionRate'
      : constraint === 'truePositiveRateParity'
        ? 'truePositiveRate'
        : 'falsePositiveRate'
  const curves = members.map((rows, g) =>
    interpolate(
      upperHull(
        tradeoffPoints(
          rows.map((i) => scores[i] ?? 0),
          rows.map((i) => labels[i] ?? 0),
          eo ? 'falsePositiveRate' : xMetric,
          eo ? 'truePositiveRate' : objective,
          options.flip ?? false,
          String(g),
        ),
      ),
      grid,
    ),
  )
  const at = (x: number): number => Math.min(size, Math.max(0, Math.round(x * size)))
  const mix = (c: Interpolated, j: number): GroupThresholdRule => ({
    p0: c.p0[j] ?? 0,
    rule0: c.rule0[j] as ThresholdRule,
    p1: c.p1[j] ?? 0,
    rule1: c.rule1[j] as ThresholdRule,
  })
  if (!eo) {
    const overall = grid.map((_, j) =>
      members.reduce((s, rows, g) => s + (rows.length / n) * (curves[g]?.y[j] ?? 0), 0),
    )
    let best = 0
    for (const [j, v] of overall.entries()) if (v > (overall[best] ?? 0)) best = j
    return {
      constraint,
      objective,
      rules: curves.map((c) => mix(c, best)),
      xBest: grid[best] ?? 0,
      yBest: overall[best] ?? 0,
      grid,
      curves: curves.map((c) => c.y),
      overall,
      rulesAt: (x) => curves.map((c) => mix(c, at(x))),
    }
  }
  const pos = labels.reduce((s, v) => s + (v === 1 ? 1 : 0), 0)
  const neg = n - pos
  const yMin = grid.map((_, j) => minOf(curves.map((c) => c.y[j] ?? 0)))
  const overall = grid.map((x, j) => {
    const y = yMin[j] ?? 0
    const c = { tp: pos * y, fn: pos * (1 - y), fp: neg * x, tn: neg * (1 - x) }
    return Math.round(METRIC[objective](c) * 1e15) / 1e15
  })
  let best = 0
  for (const [j, v] of overall.entries()) if (v > (overall[best] ?? 0)) best = j
  // Each group at (x_j, the shared rate): its hull mixture, then a constant decision for the part above the shared rate.
  const rulesOf = (j: number): GroupThresholdRule[] => {
    const xj = grid[j] ?? 0
    const shared = yMin[j] ?? 0
    return curves.map((c) => {
      const y = c.y[j] ?? 0
      return { ...mix(c, j), pIgnore: y === xj ? 0 : (y - shared) / (y - xj), constant: xj }
    })
  }
  return {
    constraint,
    objective,
    rules: rulesOf(best),
    xBest: grid[best] ?? 0,
    yBest: yMin[best] ?? 0,
    grid,
    curves: curves.map((c) => c.y),
    overall,
    rulesAt: (x) => rulesOf(at(x)),
  }
}

const applies = (r: ThresholdRule, s: number): number =>
  (r.operator === '>' ? s > r.threshold : s < r.threshold) ? 1 : 0

/**
 * Each row's probability of a positive decision under per-group randomised threshold rules (fairlearn's
 * `_pmf_predict`).
 *
 * @param rules - One rule per group.
 * @param scores - Each row's score.
 * @param groups - Each row's group index.
 * @returns P(positive decision) per row.
 * @example
 * ```ts
 * import { thresholdDecisions } from 'mdml-engine/fairness'
 * const rule = { p0: 0.5, rule0: { operator: '>' as const, threshold: 0.5 }, p1: 0.5, rule1: { operator: '>' as const, threshold: 0.2 } }
 * thresholdDecisions([rule], [0.1, 0.3, 0.9], [0, 0, 0]) // => [0, 0.5, 1]
 * ```
 */
export function thresholdDecisions(
  rules: readonly GroupThresholdRule[],
  scores: readonly number[],
  groups: readonly number[],
): number[] {
  return scores.map((s, i) => {
    const r = rules[groups[i] ?? 0]
    if (r === undefined) return 0
    const p = r.p0 * applies(r.rule0, s) + r.p1 * applies(r.rule1, s)
    return r.pIgnore === undefined ? p : r.pIgnore * (r.constant ?? 0) + (1 - r.pIgnore) * p
  })
}

/** The cost calibrated equalised odds equalises: false positives, false negatives, or both weighted by base rates. */
export type CalibratedCost = 'falsePositive' | 'falseNegative' | 'weighted'

/** Options of {@link calibratedEqualisedOdds}. */
export interface CalibratedEqualisedOddsOptions {
  /** The privileged group's index (default 0); every other group is unprivileged. */
  readonly privileged?: number
  /** The generalised cost equalised (default `weighted`, as AIF360). */
  readonly cost?: CalibratedCost
}

/** A fitted calibrated-equalised-odds post-processor. */
export interface CalibratedEqualisedOdds {
  readonly privileged: number
  /** The base rate of the privileged and the unprivileged rows. */
  readonly baseRates: readonly [number, number]
  /** The generalised cost of the privileged and unprivileged rows' scores, and of their base rates (trivial). */
  readonly costs: readonly [number, number]
  readonly trivialCosts: readonly [number, number]
  /** The share of each side's scores replaced by its base rate (one of the two is 0). */
  readonly mixRates: readonly [number, number]
}

/** Generalised false-positive rate (mean score of negatives) and false-negative rate (mean 1 - score of positives). */
function generalisedRates(
  scores: readonly number[],
  labels: readonly number[],
  rows: readonly number[],
): [number, number] {
  let fp = 0
  let neg = 0
  let fn = 0
  let pos = 0
  for (const i of rows) {
    const s = scores[i] ?? 0
    if (labels[i] === 1) {
      fn += 1 - s
      pos++
    } else {
      fp += s
      neg++
    }
  }
  return [neg === 0 ? Number.NaN : fp / neg, pos === 0 ? Number.NaN : fn / pos]
}

/**
 * Calibrated equalised odds (Pleiss et al., 2017; AIF360's `CalibratedEqOddsPostprocessing`): keep scores calibrated
 * while equalising a generalised error cost between the privileged and unprivileged groups, by replacing a share of the
 * cheaper group's scores with its base rate (the trivial calibrated classifier) until both groups' costs match.
 *
 * @param scores - Calibrated scores (probabilities) `[N]`.
 * @param labels - Binary labels `[N]`.
 * @param groups - Group indices `[N]`.
 * @param options - The privileged group and the cost.
 * @returns The base rates, costs and mixing rates.
 * @throws MdmlError `shape` when the inputs differ in length.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { calibratedEqualisedOdds } from 'mdml-engine/fairness'
 * const r = calibratedEqualisedOdds([0.9, 0.2, 0.8, 0.1, 0.6, 0.4, 0.6, 0.4], [1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 1, 1, 1])
 * r.mixRates[0] > 0 // => true
 * r.mixRates[1] // => 0
 * ```
 */
export function calibratedEqualisedOdds(
  scores: readonly number[],
  labels: readonly number[],
  groups: readonly number[],
  options: CalibratedEqualisedOddsOptions = {},
): CalibratedEqualisedOdds {
  if (scores.length !== labels.length || scores.length !== groups.length)
    throw new MdmlError('shape', 'calibratedEqualisedOdds: inputs differ in length')
  const privileged = options.privileged ?? 0
  const sides = [0, 1].map((side) => groups.flatMap((g, i) => ((g === privileged) === (side === 0) ? [i] : [])))
  const baseRates = sides.map(
    (rows) => rows.reduce((s, i) => s + (labels[i] === 1 ? 1 : 0), 0) / Math.max(1, rows.length),
  )
  const cost = options.cost ?? 'weighted'
  const costOf = (s: readonly number[], side: number): number => {
    const [gfpr, gfnr] = generalisedRates(s, labels, sides[side] ?? [])
    const b = baseRates[side] ?? 0
    if (cost === 'falsePositive') return gfpr
    if (cost === 'falseNegative') return gfnr
    // AIF360's weighted cost with equal weights (fp_rate = fn_rate = 1, normalised to 1/2 each).
    return 0.5 * gfpr * (1 - b) + 0.5 * gfnr * b
  }
  const trivial = scores.map((_, i) => baseRates[groups[i] === privileged ? 0 : 1] ?? 0)
  const costs: [number, number] = [costOf(scores, 0), costOf(scores, 1)]
  const trivialCosts: [number, number] = [costOf(trivial, 0), costOf(trivial, 1)]
  const unprivCostsMore = costs[1] > costs[0]
  const mixRates: [number, number] = unprivCostsMore
    ? [(costs[1] - costs[0]) / (trivialCosts[0] - costs[0]), 0]
    : [0, (costs[0] - costs[1]) / (trivialCosts[1] - costs[1])]
  return { privileged, baseRates: [baseRates[0] ?? 0, baseRates[1] ?? 0], costs, trivialCosts, mixRates }
}

/**
 * The decisions of a fitted calibrated-equalised-odds post-processor: each row's probability of a positive decision
 * (its own score's decision with probability 1 - mix, its group's base rate's with probability mix), or, given a key,
 * the randomised scores themselves.
 *
 * @param fitted - The fitted post-processor.
 * @param scores - Scores `[N]`.
 * @param groups - Group indices `[N]`.
 * @param threshold - The decision threshold (default 0.5; positive when the score is at least it).
 * @returns Each row's probability of a positive decision.
 * @example
 * ```ts
 * import { calibratedEqualisedOdds, calibratedEqualisedOddsDecisions } from 'mdml-engine/fairness'
 * const fit = { privileged: 0, baseRates: [0.4, 0.4], costs: [0, 0], trivialCosts: [0, 0], mixRates: [0.25, 0] } as const
 * calibratedEqualisedOddsDecisions(fit, [0.9, 0.1, 0.9], [0, 0, 1]) // => [0.75, 0, 1]
 * ```
 */
export function calibratedEqualisedOddsDecisions(
  fitted: CalibratedEqualisedOdds,
  scores: readonly number[],
  groups: readonly number[],
  threshold = 0.5,
): number[] {
  return scores.map((s, i) => {
    const side = groups[i] === fitted.privileged ? 0 : 1
    const mix = fitted.mixRates[side] ?? 0
    const own = s >= threshold ? 1 : 0
    const base = (fitted.baseRates[side] ?? 0) >= threshold ? 1 : 0
    return (1 - mix) * own + mix * base
  })
}

/** The fairness metric reject-option classification bounds. */
export type RejectOptionMetric = 'statisticalParity' | 'averageOdds' | 'equalOpportunity'

/** Options of {@link rejectOptionClassification}. */
export interface RejectOptionOptions {
  /** The privileged group's index (default 0); every other group is unprivileged. */
  readonly privileged?: number
  /** The metric bounded (default statistical parity, as AIF360). */
  readonly metric?: RejectOptionMetric
  /** The metric's allowed range, unprivileged minus privileged (default [-0.05, 0.05]). */
  readonly bounds?: readonly [number, number]
  /** Classification thresholds searched: from, to and how many (default 0.01, 0.99, 100). */
  readonly thresholds?: readonly [number, number, number]
  /** Margins searched at each threshold, from 0 to the distance to 0 or 1 (default 50). */
  readonly margins?: number
}

/** A fitted reject-option classifier, with the search behind it. */
export interface RejectOption {
  readonly privileged: number
  readonly threshold: number
  /** Half the width of the critical region around the threshold. */
  readonly margin: number
  readonly balancedAccuracy: number
  /** The metric at the chosen threshold and margin. */
  readonly metric: number
  /** Whether a threshold and margin met the bounds (else the one with the smallest metric). */
  readonly feasible: boolean
  /** Every point searched. */
  readonly grid: {
    readonly thresholds: readonly number[]
    readonly margins: readonly number[]
    readonly balancedAccuracy: readonly number[]
    readonly metric: readonly number[]
  }
}

/**
 * Reject-option decisions: positive when the score is above the threshold, except in the critical region (within the
 * margin of it), where the unprivileged rows are decided positive and the privileged negative.
 *
 * @param fitted - The threshold, margin and privileged group.
 * @param scores - Scores `[N]`.
 * @param groups - Group indices `[N]`.
 * @returns Each row's decision.
 * @example
 * ```ts
 * import { rejectOptionDecisions } from 'mdml-engine/fairness'
 * rejectOptionDecisions({ privileged: 0, threshold: 0.5, margin: 0.1 }, [0.55, 0.45, 0.8, 0.2], [0, 1, 0, 1]) // => [0, 1, 1, 0]
 * ```
 */
export function rejectOptionDecisions(
  fitted: Pick<RejectOption, 'privileged' | 'threshold' | 'margin'>,
  scores: readonly number[],
  groups: readonly number[],
): number[] {
  const { threshold: t, margin: m } = fitted
  return scores.map((s, i) => {
    if (s <= t + m && s > t - m) return groups[i] === fitted.privileged ? 0 : 1
    return s > t ? 1 : 0
  })
}

/** numpy's linspace. */
function linspace(a: number, b: number, n: number): number[] {
  if (n === 1) return [a]
  const step = (b - a) / (n - 1)
  return Array.from({ length: n }, (_, i) => (i === n - 1 ? b : a + i * step))
}

/**
 * Reject-option classification (Kamiran, Karim and Zhang, 2012; AIF360's `RejectOptionClassification`): in a critical
 * region around the decision threshold, where the classifier is least sure, decide in favour of the unprivileged group
 * and against the privileged. The threshold and the region's margin are chosen by grid search to maximise balanced
 * accuracy while keeping the fairness metric (unprivileged minus privileged) within bounds, as AIF360.
 *
 * @param scores - Scores in [0, 1] `[N]`.
 * @param labels - Binary labels `[N]`.
 * @param groups - Group indices `[N]`.
 * @param options - The privileged group, metric, bounds and grids.
 * @returns The threshold, margin, their balanced accuracy and metric, and the grid.
 * @throws MdmlError `shape` when the inputs differ in length.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { rejectOptionClassification } from 'mdml-engine/fairness'
 * const scores = [0.9, 0.7, 0.6, 0.3, 0.55, 0.45, 0.4, 0.1]
 * const r = rejectOptionClassification(scores, [1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1], { thresholds: [0.1, 0.9, 9], margins: 5 })
 * r.feasible // => true
 * ```
 */
export function rejectOptionClassification(
  scores: readonly number[],
  labels: readonly number[],
  groups: readonly number[],
  options: RejectOptionOptions = {},
): RejectOption {
  if (scores.length !== labels.length || scores.length !== groups.length)
    throw new MdmlError('shape', 'rejectOptionClassification: inputs differ in length')
  const privileged = options.privileged ?? 0
  const metricName = options.metric ?? 'statisticalParity'
  const [lb, ub] = options.bounds ?? [-0.05, 0.05]
  const [lo, hi, count] = options.thresholds ?? [0.01, 0.99, 100]
  const nMargins = options.margins ?? 50
  const priv = groups.map((g) => g === privileged)
  const evaluate = (d: readonly number[]): { bacc: number; metric: number } => {
    // Counts per side (0 privileged, 1 unprivileged) and overall: [tp, fp, tn, fn, selected, n].
    const c = [new Array<number>(6).fill(0), new Array<number>(6).fill(0)]
    for (const [i, y] of labels.entries()) {
      const side = c[priv[i] === true ? 0 : 1] as number[]
      const p = d[i] === 1
      const k = y === 1 ? (p ? 0 : 3) : p ? 1 : 2
      side[k] = (side[k] ?? 0) + 1
      if (p) side[4] = (side[4] ?? 0) + 1
      side[5] = (side[5] ?? 0) + 1
    }
    const sum = (k: number): number => (c[0]?.[k] ?? 0) + (c[1]?.[k] ?? 0)
    const tpr = (s: readonly number[]): number => (s[0] ?? 0) / ((s[0] ?? 0) + (s[3] ?? 0))
    const fpr = (s: readonly number[]): number => (s[1] ?? 0) / ((s[1] ?? 0) + (s[2] ?? 0))
    const all = [sum(0), sum(1), sum(2), sum(3)]
    const bacc = 0.5 * (tpr(all) + (all[2] ?? 0) / ((all[2] ?? 0) + (all[1] ?? 0)))
    const [p0 = [], p1 = []] = c
    const metric =
      metricName === 'statisticalParity'
        ? (p1[4] ?? 0) / (p1[5] ?? 1) - (p0[4] ?? 0) / (p0[5] ?? 1)
        : metricName === 'equalOpportunity'
          ? tpr(p1) - tpr(p0)
          : 0.5 * (fpr(p1) - fpr(p0) + (tpr(p1) - tpr(p0)))
    return { bacc, metric }
  }
  const thresholds: number[] = []
  const margins: number[] = []
  const baccs: number[] = []
  const metrics: number[] = []
  for (const t of linspace(lo, hi, count))
    for (const m of linspace(0, t <= 0.5 ? t : 1 - t, nMargins)) {
      const e = evaluate(rejectOptionDecisions({ privileged, threshold: t, margin: m }, scores, groups))
      thresholds.push(t)
      margins.push(m)
      baccs.push(e.bacc)
      metrics.push(e.metric)
    }
  const feasible = metrics.map((v) => v >= lb && v <= ub)
  const anyFeasible = feasible.some(Boolean)
  let best = -1
  for (let i = 0; i < metrics.length; i++) {
    if (anyFeasible && feasible[i] !== true) continue
    if (best < 0) best = i
    else if (anyFeasible ? (baccs[i] ?? 0) > (baccs[best] ?? 0) : (metrics[i] ?? 0) < (metrics[best] ?? 0)) best = i
  }
  return {
    privileged,
    threshold: thresholds[best] ?? 0.5,
    margin: margins[best] ?? 0,
    balancedAccuracy: baccs[best] ?? Number.NaN,
    metric: metrics[best] ?? Number.NaN,
    feasible: anyFeasible,
    grid: { thresholds, margins, balancedAccuracy: baccs, metric: metrics },
  }
}
