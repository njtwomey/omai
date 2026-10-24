import { WhiteboxError } from '../base/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { sample, vec } from './_util.ts'

/** How {@link rankdata} ranks tied values. */
export type RankMethod = 'average' | 'min' | 'max' | 'dense' | 'ordinal'

/** The stable ascending order of a list (indices). */
function order(x: ArrayLike<number>): number[] {
  return Array.from({ length: x.length }, (_, i) => i).sort((a, b) => (x[a] ?? 0) - (x[b] ?? 0) || a - b)
}

/**
 * Ranks of a list of numbers, 1-based (scipy.stats.rankdata on plain numbers): the kernel behind {@link rankdata},
 * for code that ranks inside loops.
 *
 * @param x - The values (no NaN).
 * @param method - Tie handling (default `'average'`).
 * @returns The ranks.
 * @example
 * ```ts
 * import { ranksOf } from 'whitebox-engine/stats'
 * ranksOf([10, 20, 10]) // => [1.5, 3, 1.5]
 * ranksOf([10, 20, 10], 'min') // => [1, 3, 1]
 * ```
 */
export function ranksOf(x: ArrayLike<number>, method: RankMethod = 'average'): Float64Array {
  const n = x.length
  const idx = order(x)
  const ranks = new Float64Array(n)
  let dense = 0
  let i = 0
  while (i < n) {
    let j = i
    while (j + 1 < n && x[idx[j + 1] ?? 0] === x[idx[i] ?? 0]) j++
    dense++
    for (let k = i; k <= j; k++) {
      const at = idx[k] ?? 0
      if (method === 'average') ranks[at] = (i + j) / 2 + 1
      else if (method === 'min') ranks[at] = i + 1
      else if (method === 'max') ranks[at] = j + 1
      else if (method === 'dense') ranks[at] = dense
      else ranks[at] = k + 1
    }
    i = j + 1
  }
  return ranks
}

/**
 * Rank data (scipy.stats.rankdata): 1-based ranks with ties averaged (default) or given the minimum, maximum, dense
 * or ordinal (first-occurrence) rank.
 *
 * @param x - The sample (flattened).
 * @param options - method (default `'average'`).
 * @returns The ranks.
 * @throws WhiteboxError with code `value` for NaN entries.
 * @example
 * ```ts
 * import { rankdata } from 'whitebox-engine/stats'
 * rankdata([0, 2, 3, 2]) // => [1, 2.5, 4, 2.5]
 * rankdata([0, 2, 3, 2], { method: 'dense' }) // => [1, 2, 3, 2]
 * rankdata([0, 2, 3, 2], { method: 'ordinal' }) // => [1, 2, 4, 3]
 * ```
 */
export function rankdata(x: TensorLike, options: { readonly method?: RankMethod } = {}): Tensor {
  const a = sample('rankdata', x, 0)
  for (const v of a) if (Number.isNaN(v)) throw new WhiteboxError('value', 'rankdata: NaN values cannot be ranked')
  return vec(ranksOf(a, options.method ?? 'average'))
}

/**
 * The sizes of the groups of tied values (every group, including singletons, in ascending order of value).
 *
 * @param x - The values.
 * @returns One count per distinct value.
 * @example
 * ```ts
 * import { tieCounts } from 'whitebox-engine/stats'
 * tieCounts([3, 1, 3, 2, 3]) // => [1, 1, 3]
 * ```
 */
export function tieCounts(x: TensorLike): number[] {
  const a = sample('tieCounts', x, 0)
  a.sort()
  const out: number[] = []
  let i = 0
  while (i < a.length) {
    let j = i
    while (j + 1 < a.length && a[j + 1] === a[i]) j++
    out.push(j - i + 1)
    i = j + 1
  }
  return out
}

/**
 * The tie correction factor of rank tests, 1 - sum(t^3 - t) / (n^3 - n) over tie groups of size t
 * (scipy.stats.tiecorrect); 1 without ties.
 *
 * @param x - The values or their ranks.
 * @returns The factor in (0, 1].
 * @example
 * ```ts
 * import { tieCorrection } from 'whitebox-engine/stats'
 * tieCorrection([1, 2, 3]) // => 1
 * tieCorrection([1, 2.5, 2.5, 4]) // => 0.9
 * ```
 */
export function tieCorrection(x: TensorLike): number {
  const counts = tieCounts(x)
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
 * import { tieTerm } from 'whitebox-engine/stats'
 * tieTerm([1, 1, 2, 3, 3, 3]) // => 30
 * ```
 */
export function tieTerm(x: TensorLike): number {
  let s = 0
  for (const t of tieCounts(x)) s += t ** 3 - t
  return s
}
