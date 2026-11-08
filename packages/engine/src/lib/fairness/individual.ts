/**
 * Individual fairness: similar people treated similarly. Consistency of decisions among nearest neighbours (Zemel et
 * al. 2013; AIF360), Lipschitz fairness of scores under a task metric (Dwork et al. 2012) with every violating pair,
 * and fairness through awareness, the most useful randomised decisions that are Lipschitz in the metric (a linear
 * programme).
 */

import { OmaiError } from '../base/index.ts'
import { linprog } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey, randint } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'

const euclidean = (a: readonly number[], b: readonly number[]): number =>
  Math.sqrt(a.reduce((s, v, j) => s + (v - (b[j] ?? 0)) ** 2, 0))

/** A distance between two rows (the task metric of individual fairness). */
export type RowMetric = (a: readonly number[], b: readonly number[]) => number

/**
 * Consistency (Zemel et al., 2013; AIF360's `consistency`): one minus the mean gap between each row's decision and the
 * mean decision of its k nearest neighbours in feature space (the row itself among them, as AIF360), 1 when
 * neighbours are always decided alike.
 *
 * @param x - Features `[n, d]`.
 * @param decisions - Decisions or scores `[n]`.
 * @param options - k (default 5) and the metric (default Euclidean).
 * @returns The consistency, in [0, 1] for decisions in [0, 1].
 * @throws OmaiError `shape` when the inputs differ in length.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { consistency } from 'omai-engine/fairness'
 * consistency([[0], [0.1], [5], [5.1]], [1, 1, 0, 0], { k: 2 }) // => 1
 * consistency([[0], [0.1], [5], [5.1]], [1, 0, 0, 1], { k: 2 }) // => 0.5
 * ```
 */
export function consistency(
  x: readonly (readonly number[])[],
  decisions: readonly number[],
  options: { readonly k?: number; readonly metric?: RowMetric } = {},
): number {
  const n = x.length
  if (decisions.length !== n) throw new OmaiError('shape', 'consistency: one decision per row')
  const k = Math.min(n, options.k ?? 5)
  const metric = options.metric ?? euclidean
  let total = 0
  for (let i = 0; i < n; i++) {
    const row = x[i] ?? []
    const near = Array.from({ length: n }, (_, j) => j)
      .map((j) => [metric(row, x[j] ?? []), j] as const)
      .sort((a, b) => a[0] - b[0] || a[1] - b[1])
      .slice(0, k)
    const mean = near.reduce((s, [, j]) => s + (decisions[j] ?? 0), 0) / k
    total += Math.abs((decisions[i] ?? 0) - mean)
  }
  return 1 - total / n
}

/** Options of {@link lipschitzFairness}. */
export interface LipschitzOptions {
  /** The Lipschitz constant (default 1). */
  readonly constant?: number
  readonly metric?: RowMetric
  /** Check this many random pairs instead of all of them (default all, up to 2000 rows). */
  readonly pairs?: number
  readonly key?: Key
}

/** The result of {@link lipschitzFairness}. */
export interface LipschitzFairness {
  /** The pairs checked: each pair's distance and output gap. */
  readonly distances: readonly number[]
  readonly gaps: readonly number[]
  readonly pairs: readonly (readonly [number, number])[]
  /** The share of pairs with gap > constant x distance. */
  readonly violationRate: number
  /** The largest gap over distance (the smallest constant the scores satisfy). */
  readonly worstRatio: number
  readonly constant: number
}

/**
 * Lipschitz (individual) fairness (Dwork, Hardt, Pitassi, Reingold and Zemel, 2012): a classifier treats similar
 * people similarly when |f(x) - f(x')| <= L d(x, x') for a task metric d. Every pair (or a sample of pairs) with its
 * distance and output gap, the share that violate the condition, and the smallest L that holds.
 *
 * @param x - Features `[n, d]`, in the units of the task metric.
 * @param outputs - Scores or probabilities of a positive decision `[n]`.
 * @param options - The constant, metric, number of sampled pairs and key.
 * @returns The pairs, the violation rate and the worst ratio.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { lipschitzFairness } from 'omai-engine/fairness'
 * const r = lipschitzFairness([[0], [0.1], [1]], [0, 0.5, 1])
 * r.violationRate.toFixed(3) // => '0.333'
 * r.worstRatio // => 5
 * ```
 */
export function lipschitzFairness(
  x: readonly (readonly number[])[],
  outputs: readonly number[],
  options: LipschitzOptions = {},
): LipschitzFairness {
  const n = x.length
  if (outputs.length !== n) throw new OmaiError('shape', 'lipschitzFairness: one output per row')
  const L = options.constant ?? 1
  const metric = options.metric ?? euclidean
  const pairs: [number, number][] = []
  const all = (n * (n - 1)) / 2
  const wanted = options.pairs ?? (n <= 2000 ? all : 200_000)
  if (wanted >= all) for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) pairs.push([i, j])
  else {
    const k = options.key ?? makeKey(0)
    const a = toFlat(randint(foldIn(k, 0), [wanted], 0, n))
    const b = toFlat(randint(foldIn(k, 1), [wanted], 0, n - 1))
    for (let t = 0; t < wanted; t++) {
      const i = a[t] ?? 0
      const j0 = b[t] ?? 0
      pairs.push([i, j0 >= i ? j0 + 1 : j0])
    }
  }
  const distances = pairs.map(([i, j]) => metric(x[i] ?? [], x[j] ?? []))
  const gaps = pairs.map(([i, j]) => Math.abs((outputs[i] ?? 0) - (outputs[j] ?? 0)))
  let violations = 0
  let worst = 0
  for (const [p, d] of distances.entries()) {
    const g = gaps[p] ?? 0
    if (g > L * d + 1e-12) violations++
    worst = Math.max(worst, d === 0 ? (g > 0 ? Number.POSITIVE_INFINITY : 0) : g / d)
  }
  return {
    distances,
    gaps,
    pairs,
    violationRate: pairs.length === 0 ? 0 : violations / pairs.length,
    worstRatio: worst,
    constant: L,
  }
}

/**
 * Fairness through awareness (Dwork et al., 2012): the randomised decisions (each row's probability of a positive
 * decision) that maximise expected utility, sum_i p_i (2 s_i - 1) for scores s, subject to |p_i - p_j| <= L d(x_i, x_j)
 * for every pair (the total-variation Lipschitz condition), as a linear programme. Quadratic in the rows: for up to a
 * few hundred.
 *
 * @param x - Features `[n, d]`, in the units of the task metric.
 * @param scores - Each row's probability of being positive (the utility of deciding positive) `[n]`.
 * @param options - The constant and metric.
 * @returns Each row's probability of a positive decision.
 * @throws OmaiError `value` above 400 rows.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { fairnessThroughAwareness } from 'omai-engine/fairness'
 * // Two close rows with opposite scores must be decided nearly alike.
 * fairnessThroughAwareness([[0], [0.2]], [0.9, 0.2]).map((p) => p.toFixed(1)) // => ['1.0', '0.8']
 * ```
 */
export function fairnessThroughAwareness(
  x: readonly (readonly number[])[],
  scores: readonly number[],
  options: { readonly constant?: number; readonly metric?: RowMetric } = {},
): number[] {
  const n = x.length
  if (scores.length !== n) throw new OmaiError('shape', 'fairnessThroughAwareness: one score per row')
  if (n > 400) throw new OmaiError('value', 'fairnessThroughAwareness: at most 400 rows (the programme is quadratic)')
  const L = options.constant ?? 1
  const metric = options.metric ?? euclidean
  const aUb: number[][] = []
  const bUb: number[] = []
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const d = L * metric(x[i] ?? [], x[j] ?? [])
      if (d >= 1) continue
      const row = new Array<number>(n).fill(0)
      row[i] = 1
      row[j] = -1
      aUb.push(
        row,
        row.map((v) => -v),
      )
      bUb.push(d, d)
    }
  const r = linprog({
    c: scores.map((s) => -(2 * s - 1)),
    ...(aUb.length === 0 ? {} : { aUb, bUb }),
    bounds: [0, 1],
  })
  if (!r.success) throw new OmaiError('value', `fairnessThroughAwareness: ${r.message}`)
  return Array.from(toFlat(r.x))
}
