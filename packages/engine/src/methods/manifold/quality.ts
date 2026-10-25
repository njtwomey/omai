/**
 * Measures of how well an embedding preserves its data: trustworthiness and continuity (Venna and Kaski, 2001),
 * neighbourhood preservation, the co-ranking matrix of original against embedded neighbour ranks (Lee and Verleysen,
 * 2009) and Kruskal's stress.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { matrixView } from '../../lib/learn/index.ts'
import { geometry } from '../../lib/numerics/index.ts'
import { type TensorLike, toFlat } from '../../lib/tensor/index.ts'

/**
 * Neighbour ranks: ranks[i][j] is the rank (1 = nearest) of row j among row i's neighbours by Euclidean distance,
 * ties broken by index; the diagonal is 0.
 *
 * @param x - Points `[n, d]`.
 * @returns The rank matrix `[n][n]`.
 */
export function neighbourRanks(x: TensorLike): number[][] {
  const n = matrixView(x, 'neighbourRanks').n
  const d = toFlat(geometry.cdist(x, x))
  return Array.from({ length: n }, (_, i) => {
    const order = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => (d[i * n + a] ?? 0) - (d[i * n + b] ?? 0) || a - b)
    const ranks = new Array<number>(n).fill(0)
    for (const [r, j] of order.entries()) ranks[j] = r + 1
    return ranks
  })
}

function checkK(n: number, k: number, who: string): void {
  if (!(Number.isInteger(k) && k >= 1 && k < n / 2))
    throw new WhiteboxError('value', `${who}: nNeighbors must be in 1..${Math.ceil(n / 2) - 1}`)
}

/** The penalty of Venna and Kaski: sum over i of the ranks in `a` beyond k of the k nearest in `b`, normalised. */
function rankPenalty(a: number[][], b: number[][], k: number): number {
  const n = a.length
  let t = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) if (j !== i && (b[i]?.[j] ?? 0) <= k) t += Math.max(0, (a[i]?.[j] ?? 0) - k)
  return 1 - (t * 2) / (n * k * (2 * n - 3 * k - 1))
}

/**
 * Trustworthiness (Venna and Kaski 2001; sklearn.manifold.trustworthiness): how far the embedding's k nearest
 * neighbours lie from each point's true neighbourhood; 1 when every embedded neighbour is a true one.
 *
 * @param x - The original points `[n, d]`.
 * @param embedded - The embedding `[n, e]`.
 * @param nNeighbors - k (default 5).
 * @returns The trustworthiness in [0, 1].
 */
export function trustworthiness(x: TensorLike, embedded: TensorLike, nNeighbors = 5): number {
  const high = neighbourRanks(x)
  checkK(high.length, nNeighbors, 'trustworthiness')
  return rankPenalty(high, neighbourRanks(embedded), nNeighbors)
}

/**
 * Continuity (Venna and Kaski 2001): trustworthiness with the roles swapped; how many true neighbours stay near in
 * the embedding.
 *
 * @param x - The original points `[n, d]`.
 * @param embedded - The embedding `[n, e]`.
 * @param nNeighbors - k (default 5).
 * @returns The continuity in [0, 1].
 */
export function continuity(x: TensorLike, embedded: TensorLike, nNeighbors = 5): number {
  const low = neighbourRanks(embedded)
  checkK(low.length, nNeighbors, 'continuity')
  return rankPenalty(low, neighbourRanks(x), nNeighbors)
}

/**
 * Neighbourhood preservation: the mean fraction of each point's k nearest neighbours that are also among its k
 * nearest in the embedding.
 *
 * @param x - The original points `[n, d]`.
 * @param embedded - The embedding `[n, e]`.
 * @param nNeighbors - k (default 5).
 * @returns The preserved fraction in [0, 1].
 */
export function neighbourhoodPreservation(x: TensorLike, embedded: TensorLike, nNeighbors = 5): number {
  const a = neighbourRanks(x)
  const b = neighbourRanks(embedded)
  const n = a.length
  let total = 0
  for (let i = 0; i < n; i++) {
    let shared = 0
    for (let j = 0; j < n; j++)
      if (j !== i && (a[i]?.[j] ?? 0) <= nNeighbors && (b[i]?.[j] ?? 0) <= nNeighbors) shared++
    total += shared / nNeighbors
  }
  return total / n
}

/**
 * The co-ranking matrix (Lee and Verleysen 2009): q[k][l] counts the pairs ranked k + 1 in the original space and
 * l + 1 in the embedding; a perfect embedding is diagonal.
 *
 * @param x - The original points `[n, d]`.
 * @param embedded - The embedding `[n, e]`.
 * @returns The (n - 1) x (n - 1) counts.
 */
export function coRankingMatrix(x: TensorLike, embedded: TensorLike): number[][] {
  const a = neighbourRanks(x)
  const b = neighbourRanks(embedded)
  const n = a.length
  const q = Array.from({ length: n - 1 }, () => new Array<number>(n - 1).fill(0))
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j) continue
      const row = q[(a[i]?.[j] ?? 1) - 1] as number[]
      const c = (b[i]?.[j] ?? 1) - 1
      row[c] = (row[c] ?? 0) + 1
    }
  return q
}

/**
 * Raw and normalised stress: how well embedded distances reproduce the original ones (Kruskal's stress-1).
 *
 * @param x - The original points `[n, d]` (or a dissimilarity matrix when `precomputed`).
 * @param embedded - The embedding `[n, e]`.
 * @param options - Whether x is a dissimilarity matrix.
 * @returns Raw stress (the sum of squared differences over pairs) and stress-1.
 */
export function embeddingStress(
  x: TensorLike,
  embedded: TensorLike,
  options: { readonly precomputed?: boolean } = {},
): { readonly raw: number; readonly normalised: number } {
  const d = options.precomputed === true ? toFlat(geometry.squareform(x)) : toFlat(geometry.pdist(x))
  const e = toFlat(geometry.pdist(embedded))
  let raw = 0
  let total = 0
  for (const [i, v] of d.entries()) {
    raw += (v - (e[i] ?? 0)) ** 2
    total += v * v
  }
  return { raw, normalised: Math.sqrt(raw / Math.max(total, Number.MIN_VALUE)) }
}
