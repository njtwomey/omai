/**
 * Spectral regularisation of ill-posed linear problems, in the manner of Hansen's Regularization Tools: the SVD with
 * data coefficients, filter factors, Tikhonov solutions in standard and general form (with a smoothness operator such
 * as second differences), truncated SVD solutions, and Tikhonov paths with residual and solution norms for L-curve
 * figures.
 */

import { MdmlError } from '../../base/index.ts'
import {
  add,
  concat,
  div,
  greater,
  matmul,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  swapaxes,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
  zeros,
} from '../../tensor/index.ts'
import { lstsq, svd } from '../linalg/index.ts'

/** The SVD of a matrix with the data's coefficients, the pieces every spectral regularisation needs. */
export interface SpectralData {
  readonly u: Value
  readonly s: Value
  readonly vh: Value
  /** u_i^T b. */
  readonly beta: Value
}

/**
 * The SVD of A and the coefficients u_i^T b (Hansen's Regularization Tools csvd plus the data step).
 *
 * @param a - Matrix `[m, n]`.
 * @param b - Data `[m]`.
 * @returns u, s, vh and beta.
 * @example
 * ```ts
 * import { spectralData } from 'mdml-engine/numerics/inverse'
 * spectralData([[2, 0], [0, 1]], [2, 3]).s.shape // => [2]
 * ```
 */
export function spectralData(a: TensorLike, b: TensorLike): SpectralData {
  const [av, bv] = promoteFloat(a, b) as [Value, Value]
  const { u, s, vh } = svd(av)
  return { u, s, vh, beta: reshape(matmul(swapaxes(u, 0, 1), reshape(bv, [-1, 1])), [-1]) }
}

/**
 * Filter factors of spectral regularisation: Tikhonov f_i = s_i^2 / (s_i^2 + lambda^2); truncated SVD f_i = 1 for the
 * first k values and 0 after.
 *
 * @param s - Singular values (descending).
 * @param parameter - lambda for Tikhonov, k for TSVD.
 * @param method - `tikhonov` (default) or `tsvd`.
 * @returns The factors.
 * @concept regularisation
 * @example
 * ```ts
 * import { filterFactors } from 'mdml-engine/numerics/inverse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(filterFactors([2, 1], 1)) // => [0.8, 0.5]
 * ```
 */
export function filterFactors(s: TensorLike, parameter: number, method: 'tikhonov' | 'tsvd' = 'tikhonov'): Value {
  const [sv] = promoteFloat(s) as [Value]
  if (method === 'tsvd') {
    const n = sv.shape[0] ?? 0
    return promoteFloat(Array.from({ length: n }, (_, i) => (i < parameter ? 1 : 0)))[0] as Value
  }
  const s2 = mul(sv, sv)
  const denominator = add(s2, parameter * parameter)
  // A zero singular value with lambda = 0 has no filter factor (0 / 0): it is dropped, as the pseudo-inverse does.
  const positive = greater(denominator, 0)
  return where(positive, div(s2, where(positive, denominator, 1)), 0)
}

/**
 * The Tikhonov solution in standard form, x = argmin |A x - b|^2 + lambda^2 |x|^2 = sum_i f_i (u_i^T b / s_i) v_i, from
 * the SVD. Differentiable in A and b.
 *
 * @param data - From {@link spectralData}.
 * @param lambda - The regularisation parameter.
 * @returns The solution `[n]`.
 * @index function Tikhonov-regularised solution of an ill-posed linear problem.
 * @example
 * ```ts
 * import { spectralData, tikhonov } from 'mdml-engine/numerics/inverse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(tikhonov(spectralData([[1, 0], [0, 1]], [2, 4]), 1)) // => [1, 2]
 * ```
 */
export function tikhonov(data: SpectralData, lambda: number): Value {
  const f = filterFactors(data.s, lambda)
  return reshape(matmul(reshape(coefficientsOf(f, data), [1, -1]), data.vh), [-1])
}

/** f_i u_i^T b / s_i, with the term of a zero singular value taken as 0 (the pseudo-inverse's convention, not 0 / 0). */
function coefficientsOf(f: Value, data: SpectralData): Value {
  const positive = greater(data.s, 0)
  return where(positive, div(mul(f, data.beta), where(positive, data.s, 1)), 0)
}

/**
 * The truncated SVD solution x_k = sum_{i <= k} (u_i^T b / s_i) v_i.
 *
 * @param data - From {@link spectralData}.
 * @param k - Number of terms kept.
 * @returns The solution `[n]`.
 * @example
 * ```ts
 * import { spectralData, tsvd } from 'mdml-engine/numerics/inverse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(tsvd(spectralData([[2, 0], [0, 1e-8]], [2, 1]), 1)) // => [1, 0]
 * ```
 */
export function tsvd(data: SpectralData, k: number): Value {
  const f = filterFactors(data.s, k, 'tsvd')
  return reshape(matmul(reshape(coefficientsOf(f, data), [1, -1]), data.vh), [-1])
}

/**
 * Tikhonov regularisation in general form, x = argmin |A x - b|^2 + lambda^2 |L (x - x0)|^2, solved as the stacked least
 * squares problem [A; lambda L] x = [b; lambda L x0] (equivalent to the GSVD solution, without forming it).
 * Differentiable in A, b and L.
 *
 * @param a - Matrix `[m, n]`.
 * @param b - Data `[m]`.
 * @param lambda - The parameter.
 * @param options - The regulariser L `[p, n]` (default identity) and prior x0.
 * @returns The solution.
 * @example
 * ```ts
 * import { tikhonovGeneral } from 'mdml-engine/numerics/inverse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(tikhonovGeneral([[1, 0], [0, 1]], [2, 4], 1)).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 2]
 * ```
 */
export function tikhonovGeneral(
  a: TensorLike,
  b: TensorLike,
  lambda: number,
  options: { readonly l?: TensorLike; readonly x0?: TensorLike } = {},
): Value {
  const [av, bv] = promoteFloat(a, b) as [Value, Value]
  const n = av.shape[1] ?? 0
  const lv = options.l === undefined ? identity(n) : (promoteFloat(options.l)[0] as Value)
  const p = lv.shape[0] ?? 0
  const prior =
    options.x0 === undefined
      ? zeros([p])
      : reshape(matmul(lv, reshape(promoteFloat(options.x0)[0] as Value, [-1, 1])), [-1])
  return lstsq(concat([av, mul(lv, lambda)], 0), concat([bv, mul(prior, lambda)], 0))
}

/** The second-difference regulariser L `[n - 2, n]` (rows [1, -2, 1]), a common smoothness prior. */
export function secondDifference(n: number): Tensor {
  if (n < 3) throw new MdmlError('value', 'secondDifference: n must be at least 3')
  const rows = Array.from({ length: n - 2 }, (_, i) =>
    Array.from({ length: n }, (_, j) => (j === i || j === i + 2 ? 1 : j === i + 1 ? -2 : 0)),
  )
  return promoteFloat(rows)[0] as Tensor
}

function identity(n: number): Value {
  return promoteFloat(
    Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))),
  )[0] as Value
}

/** One point of a regularisation path. */
export interface PathPoint {
  readonly lambda: number
  readonly residualNorm: number
  readonly solutionNorm: number
  readonly x: readonly number[]
}

/**
 * Tikhonov solutions along a range of parameters, with residual and solution norms (the data of an L-curve figure).
 *
 * @param data - From {@link spectralData}.
 * @param b - The data vector (for residual norms including the part outside the range of A).
 * @param lambdas - The parameters.
 * @returns One point per parameter.
 * @example
 * ```ts
 * import { spectralData, tikhonovPath } from 'mdml-engine/numerics/inverse'
 * tikhonovPath(spectralData([[1, 0], [0, 0.1]], [1, 1]), [1, 1], [0.01, 1]).length // => 2
 * ```
 */
export function tikhonovPath(data: SpectralData, b: TensorLike, lambdas: readonly number[]): PathPoint[] {
  const s = toFlat(primalOf(data.s))
  const beta = toFlat(primalOf(data.beta))
  const bn2 = toFlat(primalOf(promoteFloat(b)[0] as Value)).reduce((acc, v) => acc + v * v, 0)
  const outside = Math.max(0, bn2 - beta.reduce((acc, v) => acc + v * v, 0))
  return lambdas.map((lambda) => {
    let res = outside
    let sol = 0
    s.forEach((si, i) => {
      const f = (si * si) / (si * si + lambda * lambda)
      res += ((1 - f) * (beta[i] ?? 0)) ** 2
      sol += ((f * (beta[i] ?? 0)) / si) ** 2
    })
    return {
      lambda,
      residualNorm: Math.sqrt(res),
      solutionNorm: Math.sqrt(sol),
      x: toFlat(primalOf(tikhonov(data, lambda))),
    }
  })
}
