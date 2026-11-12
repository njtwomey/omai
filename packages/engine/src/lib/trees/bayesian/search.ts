/**
 * The split search of a Bayesian tree over every feature and every threshold at once. The rows of a node are sorted per
 * feature in one `argsort`, their per-row statistics are gathered in that order and accumulated with one `cumsum`, so the
 * statistics of the left side at every threshold of every feature are one `[m - 1, d, size]` tensor; the right side is the
 * total minus the left. The exact log Bayes factor of each split against no split is the sum of two leaf evidences minus
 * the parent's, evaluated on the whole tensor, and one `argmax` picks the best. A node costs a handful of tensor
 * operations, not a loop over thresholds.
 */

import {
  add,
  argmax,
  argsort,
  cast,
  cumsum,
  greaterEqual,
  less,
  logicalAnd,
  mul,
  reshape,
  sliceAxis,
  sub,
  sum,
  take,
  takeAlongAxis,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import type { BayesData } from './tree.ts'

/** The best split of a node and what the search saw. */
export interface BayesSplitSearch {
  /** How many (feature, threshold) rules are available: they leave at least `minLeaf` rows on each side. */
  readonly available: number
  /** The log evidence of the node as a leaf. */
  readonly logEvidence: number
  /** The best rule by log Bayes factor, when one is available. */
  readonly best?: {
    readonly feature: number
    readonly threshold: number
    /** The log Bayes factor of the split against no split: log p(rows | split) - log p(rows | leaf), leaf parameters integrated out. */
    readonly gain: number
    /** The statistics of the two sides. */
    readonly left: readonly number[]
    readonly right: readonly number[]
  }
  /** Every rule's log Bayes factor, `[m - 1, d]`, `-Infinity` where a rule is not available (a tie, or a side too small). */
  readonly gains: Value
  /** The threshold of each rule, `[m - 1, d]`: the midpoint of two consecutive sorted values. */
  readonly thresholds: Value
}

/**
 * Search every feature and every threshold of a node for the split with the largest log Bayes factor, with tensor
 * operations over all of them at once. A threshold is the midpoint of two consecutive distinct values of a feature; a rule
 * is available when it leaves at least `minLeaf` rows on each side.
 *
 * @param data - The data.
 * @param rows - The indices of the node's rows (int32), or `undefined` for all rows.
 * @param minLeaf - The fewest rows a side may hold.
 * @returns The best split, the number of available rules, and the gain of every rule.
 * @concept bayesian-trees
 * @index function Best split of a node over every feature and threshold at once, by exact Bayes factor.
 * @example
 * ```ts
 * import { normalGammaLeaf, searchBayesSplits, bayesData } from 'omai-engine/trees'
 * const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 })
 * const data = bayesData([[0], [1], [2], [3], [4], [5]], [0, 0.1, 0, 5, 5.1, 5], model)
 * const found = searchBayesSplits(data, undefined, 2)
 * found.best?.threshold // => 2.5
 * found.available // => 3
 * ```
 */
export function searchBayesSplits(data: BayesData, rows: Value | undefined, minLeaf: number): BayesSplitSearch {
  const { model } = data
  const xn = rows === undefined ? data.x : take(data.x, rows, 0)
  const rn = rows === undefined ? data.rows : take(data.rows, rows, 0)
  const m = xn.shape[0] ?? 0
  const d = data.d
  const total = sum(rn, 0)
  const logEvidence = toFlat(model.logEvidence(total))[0] ?? 0
  if (m < 2) {
    const empty = reshape(sum(xn, 0), [1, d])
    return { available: 0, logEvidence, gains: empty, thresholds: empty }
  }
  const order = argsort(xn, 0)
  const xs = takeAlongAxis(xn, order, 0)
  const cum = cumsum(take(rn, order, 0), 0)
  const left = sliceAxis(cum, 0, 0, m - 1)
  const right = sub(total, left)
  const lower = sliceAxis(xs, 0, 0, m - 1)
  const upper = sliceAxis(xs, 0, 1, m)
  const thresholds = mul(add(lower, upper), 0.5)
  const nLeft = model.count(left)
  const nRight = model.count(right)
  const distinct = less(lower, upper)
  const roomy = logicalAnd(greaterEqual(nLeft, minLeaf), greaterEqual(nRight, minLeaf))
  const valid = logicalAnd(distinct, roomy)
  const gain = sub(add(model.logEvidence(left), model.logEvidence(right)), logEvidence)
  const gains = where(valid, gain, Number.NEGATIVE_INFINITY)
  const available = toFlat(sum(cast(valid, 'float64')))[0] ?? 0
  if (available === 0) return { available, logEvidence, gains, thresholds }
  const flat = reshape(gains, [-1])
  const at = toFlat(argmax(flat))[0] ?? 0
  const feature = at % d
  const pick = (stats: Value): number[] => toFlat(take(reshape(stats, [(m - 1) * d, model.size]), [at], 0))
  return {
    available,
    logEvidence,
    best: {
      feature,
      threshold: toFlat(reshape(thresholds, [-1]))[at] ?? 0,
      gain: toFlat(flat)[at] ?? 0,
      left: pick(left),
      right: pick(right),
    },
    gains,
    thresholds,
  }
}
