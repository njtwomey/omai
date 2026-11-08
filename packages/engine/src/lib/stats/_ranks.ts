// Private plain-number kernels of ranking, for the loops of the rank-based tests (a tensor per call would cost more than
// the ranking): average, minimum, maximum, dense and ordinal ranks of a list, and the sizes of its tie groups.
import { OmaiError } from '../base/index.ts'

/** How ties are ranked (the values of `RankMethod`). */
export type TieMethod = 'average' | 'min' | 'max' | 'dense' | 'ordinal'

/** The stable ascending order of a list (indices). */
function order(x: ArrayLike<number>): number[] {
  return Array.from({ length: x.length }, (_, i) => i).sort((a, b) => (x[a] ?? 0) - (x[b] ?? 0) || a - b)
}

/** 1-based ranks of a list of numbers (scipy.stats.rankdata); `name` labels the error for NaN entries. */
export function rankKernel(x: ArrayLike<number>, method: TieMethod = 'average', name = 'ranksOf'): Float64Array {
  const n = x.length
  for (let k = 0; k < n; k++)
    if (Number.isNaN(x[k])) throw new OmaiError('value', `${name}: NaN values cannot be ranked`)
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

/** The sizes of the groups of tied values of a list, in ascending order of value (the list is sorted in place). */
export function tieCountKernel(a: Float64Array): number[] {
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
