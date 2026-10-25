/**
 * Pre-processing mitigation: changing the data before a model sees it. Reweighing (Kamiran and Calders; AIF360's
 * Reweighing) and disparate-impact repair of features (Feldman et al.; AIF360's DisparateImpactRemover).
 */

import { WhiteboxError } from '../base/index.ts'

/** One cell of {@link reweighingWeights}: a group and a label, how many rows it has and expects, and its weight. */
export interface ReweighingCell {
  readonly group: number
  readonly label: number
  readonly count: number
  /** The count were group and label independent: n P(group) P(label). */
  readonly expected: number
  /** expected / count. */
  readonly weight: number
}

/**
 * Reweighing (Kamiran and Calders, 2012; AIF360's `Reweighing`): weight each row by `P(group) P(label) / P(group,
 * label)`, so that in the weighted data the label is independent of the group.
 *
 * @param labels - Each row's binary label.
 * @param groups - Each row's group index.
 * @returns A weight per row, and the group-and-label cells (group-major, label 0 then 1).
 * @throws WhiteboxError `shape` when the two differ in length.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { reweighingWeights } from 'whitebox-engine/fairness'
 * reweighingWeights([1, 1, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1]).weights // => [0.75, 0.75, 1.5, 0.75, 0.75, 1.5]
 * ```
 */
export function reweighingWeights(
  labels: readonly number[],
  groups: readonly number[],
): { readonly weights: number[]; readonly cells: ReweighingCell[] } {
  if (labels.length !== groups.length) throw new WhiteboxError('shape', 'reweighingWeights: one group per label')
  const n = labels.length
  const k = Math.max(-1, ...groups) + 1
  const joint = Array.from({ length: k }, () => [0, 0])
  for (const [i, g] of groups.entries()) {
    const row = joint[g] as number[]
    const y = labels[i] === 1 ? 1 : 0
    row[y] = (row[y] ?? 0) + 1
  }
  const byGroup = joint.map((r) => (r[0] ?? 0) + (r[1] ?? 0))
  const byLabel = [0, 1].map((y) => joint.reduce((s, r) => s + (r[y] ?? 0), 0))
  const cells: ReweighingCell[] = joint.flatMap((r, g) =>
    [0, 1].map((y) => {
      const count = r[y] ?? 0
      const expected = ((byGroup[g] ?? 0) * (byLabel[y] ?? 0)) / n
      return { group: g, label: y, count, expected, weight: count === 0 ? 0 : expected / count }
    }),
  )
  return { weights: groups.map((g, i) => cells[2 * g + (labels[i] === 1 ? 1 : 0)]?.weight ?? 1), cells }
}

/** The empirical quantile (type 7, numpy's default) of sorted values. */
function quantileSorted(sorted: readonly number[], q: number): number {
  const h = (sorted.length - 1) * q
  const lo = Math.floor(h)
  const hi = Math.min(sorted.length - 1, lo + 1)
  return (sorted[lo] ?? 0) + (h - lo) * ((sorted[hi] ?? 0) - (sorted[lo] ?? 0))
}

/** The position of v in sorted values: the number below it plus half of the others equal to it. */
function midRank(sorted: readonly number[], v: number): number {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if ((sorted[mid] ?? 0) < v) lo = mid + 1
    else hi = mid
  }
  let end = lo
  while (end < sorted.length && sorted[end] === v) end++
  return lo + (end - lo - 1) / 2
}

/**
 * Disparate-impact repair (Feldman et al., 2015; AIF360's `DisparateImpactRemover`): move each feature column towards
 * a distribution shared by every group while keeping each value's rank within its group: a value at quantile u of its
 * group goes towards the median, across groups, of the groups' quantiles at u, by `level`.
 *
 * @param x - Row-major features `[n, d]`.
 * @param d - The number of columns.
 * @param groups - Each row's group index.
 * @param options - The repair level in [0, 1] (default 1) and columns to leave as they are (the protected one).
 * @returns The repaired features, row-major.
 * @throws WhiteboxError `value` for a level outside [0, 1]; `shape` when the groups do not match the rows.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { repairFeatures } from 'whitebox-engine/fairness'
 * Array.from(repairFeatures([0, 1, 10, 11], 1, [0, 0, 1, 1])) // => [5, 6, 5, 6]
 * ```
 */
export function repairFeatures(
  x: ArrayLike<number>,
  d: number,
  groups: readonly number[],
  options: { readonly level?: number; readonly skip?: readonly number[] } = {},
): Float64Array {
  const level = options.level ?? 1
  if (!(level >= 0 && level <= 1)) throw new WhiteboxError('value', 'repairFeatures: the level is in [0, 1]')
  const n = groups.length
  if (x.length !== n * d) throw new WhiteboxError('shape', `repairFeatures: ${x.length} values for ${n} rows of ${d}`)
  const k = Math.max(-1, ...groups) + 1
  const members = Array.from({ length: k }, (): number[] => [])
  for (const [i, g] of groups.entries()) members[g]?.push(i)
  const skip = new Set(options.skip ?? [])
  const out = Float64Array.from(x)
  for (let j = 0; j < d; j++) {
    if (skip.has(j)) continue
    const sorted = members.map((rows) => rows.map((i) => x[i * d + j] ?? 0).sort((a, b) => a - b))
    const present = sorted.filter((s) => s.length > 0)
    for (const [g, rows] of members.entries()) {
      const own = sorted[g] ?? []
      for (const i of rows) {
        const v = x[i * d + j] ?? 0
        const u = own.length <= 1 ? 0.5 : midRank(own, v) / (own.length - 1)
        const target = quantileSorted(
          present.map((s) => quantileSorted(s, u)).sort((a, b) => a - b),
          0.5,
        )
        out[i * d + j] = (1 - level) * v + level * target
      }
    }
  }
  return out
}
