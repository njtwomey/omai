/**
 * The binning and the loss of the cyclic booster. `quantileEdges` and `binIndex` cut a feature into bins, and two
 * option tables are the slots they fill: how the bins are cut (`boostingBinnings`: evenly spaced quantiles, or equal
 * widths) and which loss the booster descends (`boostingLosses`: squared error for regression, the logistic loss with
 * a logit link for binary classification). Each table is one `union` space for controls, grids and tuners.
 */

import { dict, type Option, type OptionTable, optionTable } from '../space/index.ts'

/**
 * Quantile cut points of a column: between consecutive values when there are few, else at evenly spaced quantiles.
 *
 * @param col - The column.
 * @param bins - The most bins.
 * @returns The ascending cut points (one fewer than the bins).
 * @example
 * ```ts
 * import { quantileEdges } from 'mdml-engine/ensembles'
 * quantileEdges([1, 2, 3, 4], 2) // => [2.5]
 * quantileEdges([1, 2, 3], 8) // => [1.5, 2.5]
 * ```
 */
export function quantileEdges(col: ArrayLike<number>, bins: number): number[] {
  const sorted = Float64Array.from(col).sort()
  const n = sorted.length
  const distinct = Array.from(new Set(sorted))
  // Few values: cut between each pair of consecutive values, so no bin is empty.
  if (distinct.length <= bins) return distinct.slice(1).map((v, i) => ((distinct[i] ?? 0) + v) / 2)
  const cuts: number[] = []
  for (let b = 1; b < bins; b++) {
    const h = (b / bins) * (n - 1)
    const lo = Math.floor(h)
    const v = (sorted[lo] ?? 0) + (h - lo) * ((sorted[Math.min(lo + 1, n - 1)] ?? 0) - (sorted[lo] ?? 0))
    if (v > (sorted[0] ?? 0) && v <= (sorted[n - 1] ?? 0) && (cuts.length === 0 || v > (cuts[cuts.length - 1] ?? 0)))
      cuts.push(v)
  }
  return cuts
}

/**
 * The bin of a value among ascending cut points: the number of cut points at or below it.
 *
 * @param edges - The cut points.
 * @param v - The value.
 * @returns The bin index.
 * @example
 * ```ts
 * import { binIndex } from 'mdml-engine/ensembles'
 * binIndex([1, 2], 1.5) // => 1
 * binIndex([1, 2], 5) // => 2
 * ```
 */
export function binIndex(edges: readonly number[], v: number): number {
  let lo = 0
  let hi = edges.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (v < (edges[mid] ?? 0)) hi = mid
    else lo = mid + 1
  }
  return lo
}

/**
 * Cut points of equal width between the extremes of a column (fewer when the column has fewer distinct values).
 *
 * @param col - The column.
 * @param bins - The most bins.
 * @returns The ascending cut points (one fewer than the bins).
 * @example
 * ```ts
 * import { uniformEdges } from 'mdml-engine/ensembles'
 * uniformEdges([0, 1, 2, 3, 4, 8], 4) // => [2, 4, 6]
 * ```
 */
export function uniformEdges(col: ArrayLike<number>, bins: number): number[] {
  const values = Array.from(col)
  const distinct = Array.from(new Set(values)).sort((a, b) => a - b)
  if (distinct.length <= bins) return distinct.slice(1).map((v, i) => ((distinct[i] ?? 0) + v) / 2)
  const lo = distinct[0] ?? 0
  const hi = distinct[distinct.length - 1] ?? 0
  return Array.from({ length: bins - 1 }, (_, b) => lo + ((b + 1) * (hi - lo)) / bins)
}

/** Cuts a column into at most `bins` bins and returns the cut points. */
export type BinningFunction = (col: ArrayLike<number>, bins: number) => number[]

/** What a binning does, for validity rules and selectors. */
export interface BinningProperties {
  /** Whether bins hold about the same number of rows (so none is empty or crowded). */
  readonly balanced: boolean
}

/**
 * The binnings of the cyclic booster as one slot: `quantile` (evenly spaced quantiles, balanced bins, the default) and
 * `uniform` (equal widths, which keeps the scale of the feature but can leave bins nearly empty).
 *
 * @concept boosting
 * @example
 * ```ts
 * import { boostingBinnings } from 'mdml-engine/ensembles'
 * boostingBinnings.make({ kind: 'uniform' })([0, 1, 2, 3, 4, 8], 4) // => [2, 4, 6]
 * ```
 */
export const boostingBinnings: OptionTable<BinningFunction, BinningProperties> = optionTable(
  {
    quantile: {
      description: 'cut at evenly spaced quantiles, so every bin holds about as many rows',
      properties: { balanced: true },
      space: dict({}),
      make: (): BinningFunction => quantileEdges,
    } satisfies Option<BinningFunction, BinningProperties>,
    uniform: {
      description: 'cut into equal widths between the extremes',
      properties: { balanced: false },
      space: dict({}),
      make: (): BinningFunction => uniformEdges,
    } satisfies Option<BinningFunction, BinningProperties>,
  },
  { default: 'quantile', label: 'binning', description: 'How a feature is cut into bins.' },
)

/** The loss a booster descends and the link that turns its scores into predictions. */
export type BoostingTask = 'regression' | 'classification'

/** What a loss does, for validity rules and selectors. */
export interface BoostingLossProperties {
  /** Whether it is a binary classification loss (labels 0/1, scores on the logit scale). */
  readonly classification: boolean
}

/**
 * The losses of the cyclic booster as one slot: `squared` (squared error, identity link: regression) and `logistic`
 * (log loss with a logit link: binary classification with labels 0 and 1).
 *
 * @concept boosting
 * @example
 * ```ts
 * import { boostingLosses } from 'mdml-engine/ensembles'
 * boostingLosses.make({ kind: 'logistic' }) // => 'classification'
 * ```
 */
export const boostingLosses: OptionTable<BoostingTask, BoostingLossProperties> = optionTable(
  {
    squared: {
      description: 'squared error with the identity link (regression)',
      properties: { classification: false },
      space: dict({}),
      make: (): BoostingTask => 'regression',
    } satisfies Option<BoostingTask, BoostingLossProperties>,
    logistic: {
      description: 'log loss with the logit link (binary classification)',
      properties: { classification: true },
      space: dict({}),
      make: (): BoostingTask => 'classification',
    } satisfies Option<BoostingTask, BoostingLossProperties>,
  },
  { default: 'squared', label: 'loss', description: 'The loss descended and the link from scores to predictions.' },
)
