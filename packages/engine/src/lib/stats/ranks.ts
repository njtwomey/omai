/**
 * Ranking with ties, the basis of rank-based tests: `rankdata` (scipy.stats.rankdata) with average, minimum, maximum,
 * dense or ordinal ties, the same for a vector (`ranksOf`), and the tie counts and tie corrections that rank-test
 * variances need.
 */

import { fromStorage, type Tensor, type TensorLike, toVector } from '../tensor/index.ts'
import { rankKernel, tieCountKernel } from './_ranks.ts'
import { sample, vec } from './_util.ts'

/** How {@link rankdata} ranks tied values. */
export type RankMethod = 'average' | 'min' | 'max' | 'dense' | 'ordinal'

/**
 * Ranks of a vector, 1-based (scipy.stats.rankdata on a flat sample), for code that wants the ranks of a list it holds
 * as a tensor, a typed array or a plain array. Takes and returns tensors (migration, D-009: `x` was `ArrayLike<number>`
 * and the ranks a `Float64Array`; the plain-number kernel the loops of this domain use is private).
 *
 * @param x - The values (no NaN): a vector (a tensor, a typed array or a flat array).
 * @param method - Tie handling (default `'average'`).
 * @returns A float64 tensor of shape `[n]`, the ranks.
 * @throws OmaiError with code `value` for NaN entries and `shape` unless `x` is a vector.
 * @example
 * ```ts
 * import { ranksOf } from 'omai-engine/stats'
 * ranksOf([10, 20, 10]) // => [1.5, 3, 1.5]
 * ranksOf([10, 20, 10], 'min') // => [1, 3, 1]
 * ```
 */
export function ranksOf(x: TensorLike, method: RankMethod = 'average'): Tensor<'float64'> {
  const ranks = rankKernel(toVector(x, 'ranksOf'), method)
  return fromStorage(ranks, [ranks.length], 'float64')
}

/**
 * Rank data (scipy.stats.rankdata): 1-based ranks with ties averaged (default) or given the minimum, maximum, dense
 * or ordinal (first-occurrence) rank.
 *
 * @param x - The sample (flattened).
 * @param options - method (default `'average'`).
 * @returns The ranks.
 * @throws OmaiError with code `value` for NaN entries.
 * @example
 * ```ts
 * import { rankdata } from 'omai-engine/stats'
 * rankdata([0, 2, 3, 2]) // => [1, 2.5, 4, 2.5]
 * rankdata([0, 2, 3, 2], { method: 'dense' }) // => [1, 2, 3, 2]
 * rankdata([0, 2, 3, 2], { method: 'ordinal' }) // => [1, 2, 4, 3]
 * ```
 */
export function rankdata(x: TensorLike, options: { readonly method?: RankMethod } = {}): Tensor {
  const a = sample('rankdata', x, 0, true)
  return vec(rankKernel(a, options.method ?? 'average'))
}

/**
 * The sizes of the groups of tied values (every group, including singletons, in ascending order of value). Returns a
 * tensor (migration, D-009: was `number[]`).
 *
 * @param x - The values.
 * @returns An int32 tensor of shape `[k]`: one count per distinct value.
 * @example
 * ```ts
 * import { tieCounts } from 'omai-engine/stats'
 * tieCounts([3, 1, 3, 2, 3]) // => [1, 1, 3]
 * ```
 */
export function tieCounts(x: TensorLike): Tensor<'int32'> {
  const counts = tieCountKernel(sample('tieCounts', x, 0))
  return fromStorage(Int32Array.from(counts), [counts.length], 'int32')
}

/**
 * The tie correction factor of rank tests, 1 - sum(t^3 - t) / (n^3 - n) over tie groups of size t
 * (scipy.stats.tiecorrect); 1 without ties.
 *
 * @param x - The values or their ranks.
 * @returns The factor in (0, 1].
 * @example
 * ```ts
 * import { tieCorrection } from 'omai-engine/stats'
 * tieCorrection([1, 2, 3]) // => 1
 * tieCorrection([1, 2.5, 2.5, 4]) // => 0.9
 * ```
 */
export function tieCorrection(x: TensorLike): number {
  const counts = tieCountKernel(sample('tieCorrection', x, 0))
  const n = counts.reduce((s, t) => s + t, 0)
  if (n < 2) return 1
  let s = 0
  for (const t of counts) s += t ** 3 - t
  return 1 - s / (n ** 3 - n)
}

/**
 * sum(t^3 - t) over tie groups, the quantity rank-test variances subtract.
 *
 * @param x - The values or their ranks.
 * @returns The tie term (0 without ties).
 * @example
 * ```ts
 * import { tieTerm } from 'omai-engine/stats'
 * tieTerm([1, 1, 2, 3, 3, 3]) // => 30
 * ```
 */
export function tieTerm(x: TensorLike): number {
  let s = 0
  for (const t of tieCountKernel(sample('tieTerm', x, 0))) s += t ** 3 - t
  return s
}
