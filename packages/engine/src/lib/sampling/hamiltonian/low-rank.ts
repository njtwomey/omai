/**
 * A low-rank-plus-diagonal estimate of an inverse mass matrix (Sountsov and Hoffman 2021, Focusing on difficult directions for
 * learning HMC trajectory lengths and mass matrices; BlackJAX `window_adaptation_low_rank`): the diagonal of the sample covariance sets
 * the scale of each coordinate, and of the correlation structure that remains only the `rank` directions whose eigenvalues
 * differ most from one (in log) are kept. It is the estimator that suits a covariance learned from few draws in many dimensions,
 * where the full matrix is noisy, and it keeps the long and the short directions of a correlated posterior, which are the ones a
 * trajectory length has to be tuned to.
 */

import { OmaiError } from '../../base/index.ts'
import { eigh } from '../../numerics/linalg/index.ts'
import { fromStorage, toFlat } from '../../tensor/index.ts'

/**
 * The inverse mass matrix diag(s) (I + U (Lambda - I) U^T) diag(s) from a covariance: s is the square root of its diagonal, Lambda and
 * U the `rank` eigenpairs of the correlation matrix whose eigenvalues are farthest from one in the log. It is positive definite, equals
 * the covariance when `rank` is the dimension, and is returned dense (row-major) for the Euclidean metric.
 *
 * @param covariance - A symmetric positive-definite covariance, row-major `[dim, dim]`.
 * @param dim - The dimension.
 * @param rank - The number of correlation directions to keep (0 to `dim`).
 * @returns The inverse mass matrix, row-major `[dim, dim]`.
 * @throws OmaiError with code `value` for a rank outside `[0, dim]` or a covariance with a non-positive diagonal.
 * @concept mass-matrix-adaptation
 * @example
 * ```ts
 * import { lowRankInverseMassMatrix } from 'omai-engine/sampling'
 * // A rank-0 estimate is the diagonal of the covariance; the full rank gives the covariance back.
 * const cov = Float64Array.of(4, 1, 1, 1)
 * [Array.from(lowRankInverseMassMatrix(cov, 2, 0)), Array.from(lowRankInverseMassMatrix(cov, 2, 2))] // => [[4, 0, 0, 1], [4, 1, 1, 1]]
 * ```
 */
export function lowRankInverseMassMatrix(covariance: Float64Array, dim: number, rank: number): Float64Array {
  if (!(Number.isInteger(rank) && rank >= 0 && rank <= dim)) {
    throw new OmaiError('value', `lowRankInverseMassMatrix: the rank must be an integer in [0, ${dim}]`)
  }
  const scale = Float64Array.from({ length: dim }, (_, i) => Math.sqrt(covariance[i * dim + i] ?? 0))
  if (!scale.every((s) => s > 0 && Number.isFinite(s))) {
    throw new OmaiError('value', 'lowRankInverseMassMatrix: the covariance needs a positive diagonal')
  }
  const out = new Float64Array(dim * dim)
  for (let a = 0; a < dim; a++) out[a * dim + a] = (scale[a] ?? 1) ** 2
  if (rank === 0) return out
  const correlation = Float64Array.from(
    { length: dim * dim },
    (_, k) => (covariance[k] ?? 0) / ((scale[Math.floor(k / dim)] ?? 1) * (scale[k % dim] ?? 1)),
  )
  const e = eigh(fromStorage(correlation, [dim, dim], 'float64'))
  const lambda = toFlat(e.eigenvalues)
  const vectors = toFlat(e.eigenvectors)
  const order = Array.from({ length: dim }, (_, i) => i)
    .sort(
      (i, j) =>
        Math.abs(Math.log(Math.max(lambda[j] ?? 1, 1e-10))) - Math.abs(Math.log(Math.max(lambda[i] ?? 1, 1e-10))),
    )
    .slice(0, rank)
  for (const i of order) {
    const l = Math.max(lambda[i] ?? 1, 1e-10) - 1
    for (let a = 0; a < dim; a++)
      for (let b = 0; b < dim; b++)
        out[a * dim + b] =
          (out[a * dim + b] ?? 0) +
          (scale[a] ?? 1) * (scale[b] ?? 1) * l * (vectors[a * dim + i] ?? 0) * (vectors[b * dim + i] ?? 0)
  }
  return out
}
