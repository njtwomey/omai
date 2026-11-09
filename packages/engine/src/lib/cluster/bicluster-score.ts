/**
 * Comparing sets of biclusters: the consensus score of Hochreiter et al. (scikit-learn's `consensus_score`), the
 * similarity of the best one-to-one matching between two sets, used to score a biclustering against the planted truth
 * of `datasets` (`makeBiclusters`, `makeCheckerboard`).
 */

import { OmaiError } from '../base/index.ts'
import { linearSumAssignment } from '../optim/index.ts'

export interface Biclusters {
  readonly rows: readonly (readonly (boolean | number)[])[]
  readonly columns: readonly (readonly (boolean | number)[])[]
}

/** How two biclusters are compared: the Jaccard index of their cells, or a function of the four indicator rows. */
export type BiclusterSimilarity =
  | 'jaccard'
  | ((
      aRows: readonly number[],
      aColumns: readonly number[],
      bRows: readonly number[],
      bColumns: readonly number[],
    ) => number)

const count = (v: readonly (boolean | number)[]): number => v.reduce<number>((s, x) => s + (x ? 1 : 0), 0)
const both = (a: readonly (boolean | number)[], b: readonly (boolean | number)[]): number =>
  a.reduce<number>((s, x, i) => s + (x && b[i] ? 1 : 0), 0)

/**
 * The consensus score of two sets of biclusters (Hochreiter et al. 2010; scikit-learn's `consensus_score`): the
 * biclusters of the two sets are matched one to one by the assignment that maximises the total similarity (Jaccard
 * index of the cells, by default), and the sum of the matched similarities is divided by the larger set's size. It is
 * 1 for identical sets up to order, 0 for disjoint ones.
 *
 * @param a - A set of biclusters (row and column indicators).
 * @param b - Another set.
 * @param options - The similarity (default Jaccard).
 * @returns The score in [0, 1].
 * @throws OmaiError `shape` when a set's rows and columns lists differ in length or its indicators in size.
 * @example
 * ```ts
 * import { consensusScore } from 'omai-engine/cluster'
 * const a = { rows: [[1, 1, 0, 0], [0, 0, 1, 1]], columns: [[1, 0, 0], [0, 1, 1]] }
 * const b = { rows: [[0, 0, 1, 1], [1, 1, 0, 0]], columns: [[0, 1, 1], [1, 0, 0]] }
 * consensusScore(a, b) // => 1
 * ```
 */
export function consensusScore(
  a: Biclusters,
  b: Biclusters,
  options: { readonly similarity?: BiclusterSimilarity } = {},
): number {
  for (const [name, s] of [
    ['a', a],
    ['b', b],
  ] as const)
    if (s.rows.length !== s.columns.length)
      throw new OmaiError(
        'shape',
        `consensusScore: ${name} has ${s.rows.length} row indicators and ${s.columns.length} column indicators`,
      )
  const similarity = options.similarity ?? 'jaccard'
  const sim = (i: number, j: number): number => {
    const ar = a.rows[i] ?? []
    const ac = a.columns[i] ?? []
    const br = b.rows[j] ?? []
    const bc = b.columns[j] ?? []
    if (typeof similarity !== 'string')
      return similarity(ar.map(Number), ac.map(Number), br.map(Number), bc.map(Number))
    const intersection = both(ar, br) * both(ac, bc)
    return intersection / (count(ar) * count(ac) + count(br) * count(bc) - intersection)
  }
  const na = a.rows.length
  const nb = b.rows.length
  if (na === 0 || nb === 0) return 0
  const matrix = Array.from({ length: na }, (_, i) => Array.from({ length: nb }, (_, j) => sim(i, j)))
  const { rowInd, colInd } = linearSumAssignment(matrix.map((r) => r.map((v) => 1 - v)))
  return rowInd.reduce((s, i, t) => s + (matrix[i]?.[colInd[t] ?? 0] ?? 0), 0) / Math.max(na, nb)
}
