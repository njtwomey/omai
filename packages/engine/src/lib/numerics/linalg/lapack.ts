// Dense LAPACK-style routines on plain row-major number arrays, for algorithms that iterate on primal values
// (Newton and trust-region methods, quasi-Newton updates, active-set solvers). Differentiable code uses the tensor
// operations of this folder instead.
import { eighQlKernel, qrKernel, solveKernel, svdKernel, triangularKernel } from './_kernels.ts'

/**
 * The Cholesky factorisation A = U^T U of a symmetric n x n matrix (LAPACK dpotrf with uplo 'U', reading the upper
 * triangle), reporting failure instead of throwing: `info` is 0 on success, else the order k (1-based) of the first
 * leading minor that is not positive definite, with rows 0..k-2 of U computed (as scipy's potrf leaves them).
 *
 * @param a - The matrix, row-major.
 * @param n - Its order.
 * @returns The upper factor (zero below the diagonal) and `info`.
 * @example
 * ```ts
 * import { dpotrf } from 'whitebox-engine/numerics/linalg'
 * Array.from(dpotrf([4, 2, 2, 2], 2).factor) // => [2, 1, 0, 1]
 * dpotrf([1, 2, 2, 1], 2).info // => 2
 * ```
 */
export function dpotrf(a: ArrayLike<number>, n: number): { readonly factor: Float64Array; readonly info: number } {
  const u = new Float64Array(n * n)
  for (let j = 0; j < n; j++) {
    let d = a[j * n + j] ?? 0
    for (let k = 0; k < j; k++) d -= (u[k * n + j] ?? 0) ** 2
    if (!(d > 0)) return { factor: u, info: j + 1 }
    const ujj = Math.sqrt(d)
    u[j * n + j] = ujj
    for (let i = j + 1; i < n; i++) {
      let s = a[j * n + i] ?? 0
      for (let k = 0; k < j; k++) s -= (u[k * n + j] ?? 0) * (u[k * n + i] ?? 0)
      u[j * n + i] = s / ujj
    }
  }
  return { factor: u, info: 0 }
}

/**
 * Solve A x = b from the upper Cholesky factor U of A (LAPACK dpotrs): U^T y = b, then U x = y.
 *
 * @param u - The upper factor from {@link dpotrf}.
 * @param b - The right-hand side (length n).
 * @param n - The order.
 * @returns x.
 * @example
 * ```ts
 * import { dpotrf, dpotrs } from 'whitebox-engine/numerics/linalg'
 * Array.from(dpotrs(dpotrf([4, 2, 2, 2], 2).factor, [6, 4], 2)) // => [1, 1]
 * ```
 */
export function dpotrs(u: ArrayLike<number>, b: ArrayLike<number>, n: number): Float64Array {
  const uf = Float64Array.from(u)
  const y = triangularKernel(uf, Float64Array.from(b), n, 1, { lower: false, transpose: true, unitDiagonal: false })
  return triangularKernel(uf, y, n, 1, { lower: false, transpose: false, unitDiagonal: false })
}

/**
 * Solve a triangular system T x = b or T^T x = b (BLAS dtrsv).
 *
 * @param t - The n x n matrix, row-major (only the chosen triangle is read).
 * @param b - The right-hand side.
 * @param n - The order.
 * @param options - `lower` (default false: upper), `transpose` and `unitDiagonal`.
 * @returns x.
 * @example
 * ```ts
 * import { dtrsv } from 'whitebox-engine/numerics/linalg'
 * Array.from(dtrsv([2, 1, 0, 4], [4, 8], 2)) // => [1, 2]
 * ```
 */
export function dtrsv(
  t: ArrayLike<number>,
  b: ArrayLike<number>,
  n: number,
  options: { readonly lower?: boolean; readonly transpose?: boolean; readonly unitDiagonal?: boolean } = {},
): Float64Array {
  return triangularKernel(Float64Array.from(t), Float64Array.from(b), n, 1, {
    lower: options.lower ?? false,
    transpose: options.transpose ?? false,
    unitDiagonal: options.unitDiagonal ?? false,
  })
}

/**
 * Solve A x = b for a general square matrix by LU with partial pivoting (LAPACK dgesv).
 *
 * @param a - The n x n matrix, row-major.
 * @param b - The right-hand side.
 * @param n - The order.
 * @returns x.
 * @throws WhiteboxError with code `value` when A is exactly singular.
 * @example
 * ```ts
 * import { dgesv } from 'whitebox-engine/numerics/linalg'
 * Array.from(dgesv([0, 1, 1, 0], [2, 3], 2)) // => [3, 2]
 * ```
 */
export function dgesv(a: ArrayLike<number>, b: ArrayLike<number>, n: number): Float64Array {
  return solveKernel(Float64Array.from(a), Float64Array.from(b), n, 1)
}

/**
 * The symmetric eigendecomposition A = V diag(w) V^T (LAPACK dsyev): eigenvalues ascending, eigenvectors as columns
 * of V (row-major), of the symmetric part of A.
 *
 * @param a - The n x n matrix, row-major.
 * @param n - The order.
 * @returns The eigenvalues and eigenvectors.
 * @example
 * ```ts
 * import { dsyev } from 'whitebox-engine/numerics/linalg'
 * Array.from(dsyev([2, 1, 1, 2], 2).w).map((x) => Math.round(x * 1e12) / 1e12) // => [1, 3]
 * ```
 */
export function dsyev(a: ArrayLike<number>, n: number): { readonly w: Float64Array; readonly v: Float64Array } {
  return eighQlKernel(Float64Array.from(a), n)
}

/**
 * The reduced QR factorisation A = Q R of an m x n matrix by Householder reflections (LAPACK dgeqrf and dorgqr).
 *
 * @param a - The matrix, row-major.
 * @param m - Rows.
 * @param n - Columns.
 * @returns Q (m x k) and R (k x n), k = min(m, n).
 * @example
 * ```ts
 * import { dgeqrf } from 'whitebox-engine/numerics/linalg'
 * Array.from(dgeqrf([3, 4], 2, 1).r).map((x) => Math.abs(x)) // => [5]
 * ```
 */
export function dgeqrf(
  a: ArrayLike<number>,
  m: number,
  n: number,
): { readonly q: Float64Array; readonly r: Float64Array } {
  return qrKernel(Float64Array.from(a), m, n)
}

/**
 * The reduced singular value decomposition A = U diag(s) V^T of an m x n matrix (LAPACK dgesvd): s descending.
 *
 * @param a - The matrix, row-major.
 * @param m - Rows.
 * @param n - Columns.
 * @returns U (m x k), s (k) and V (n x k), k = min(m, n).
 * @example
 * ```ts
 * import { dgesvd } from 'whitebox-engine/numerics/linalg'
 * Array.from(dgesvd([3, 0, 0, 4], 2, 2).s) // => [4, 3]
 * ```
 */
export function dgesvd(
  a: ArrayLike<number>,
  m: number,
  n: number,
): { readonly u: Float64Array; readonly s: Float64Array; readonly v: Float64Array } {
  return svdKernel(Float64Array.from(a), m, n)
}
