import { WhiteboxError } from '../base/index.ts'

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
  if (pos === 0 || neg === 0) throw new WhiteboxError('value', `thresholdOptimizer: group ${group} has one class only`)
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
 * @throws WhiteboxError `value` when a group has one class only.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { optimiseThresholds } from 'whitebox-engine/fairness'
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
  const k = Math.max(...groups) + 1
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
  const yMin = grid.map((_, j) => Math.min(...curves.map((c) => c.y[j] ?? 0)))
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
 * import { thresholdDecisions } from 'whitebox-engine/fairness'
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
