/**
 * Level-1 and level-2 BLAS kernels on plain number arrays (`ddot`, `dnrm2`, `daxpy`, `dscal`, `dgemv` and friends), for
 * algorithms that iterate on primal values such as Krylov solvers, Lanczos and regularising iterations. These are the
 * shared helpers every such algorithm uses instead of local copies; differentiable code uses the tensor operations
 * instead.
 *
 * @index function BLAS-style vector kernels on plain arrays: ddot, dnrm2, daxpy, dgemv.
 */

/**
 * The dot product sum_i x_i y_i (BLAS ddot).
 *
 * @param x - First vector.
 * @param y - Second vector (same length).
 * @returns The dot product.
 * @example
 * ```ts
 * import { ddot } from 'mdml-engine/numerics/linalg'
 * ddot([1, 2, 3], [4, 5, 6]) // => 32
 * ```
 */
export function ddot(x: ArrayLike<number>, y: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < x.length; i++) s += (x[i] ?? 0) * (y[i] ?? 0)
  return s
}

/**
 * The index of the entry of largest absolute value (BLAS idamax, 0-based; the first on ties, -1 for an empty vector).
 * `Math.abs(x[idamax(x)])` is the infinity norm.
 *
 * @param x - The vector.
 * @returns The index.
 * @example
 * ```ts
 * import { idamax } from 'mdml-engine/numerics/linalg'
 * idamax([1, -3, 2]) // => 1
 * ```
 */
export function idamax(x: ArrayLike<number>): number {
  let best = -1
  let m = -1
  for (let i = 0; i < x.length; i++) {
    const a = Math.abs(x[i] ?? 0)
    if (a > m || Number.isNaN(a)) {
      best = i
      m = a
      if (Number.isNaN(a)) break
    }
  }
  return best
}

/**
 * The sum of absolute values sum_i |x_i| (BLAS dasum), the L1 norm.
 *
 * @param x - The vector.
 * @returns The sum.
 * @example
 * ```ts
 * import { dasum } from 'mdml-engine/numerics/linalg'
 * dasum([1, -3, 2]) // => 6
 * ```
 */
export function dasum(x: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < x.length; i++) s += Math.abs(x[i] ?? 0)
  return s
}

/**
 * The Euclidean norm |x| (BLAS dnrm2), accumulated with scaling so that it neither overflows nor underflows. An infinite
 * entry gives Infinity and a NaN gives NaN (it propagates; the scaled sum used to turn two infinite entries into NaN).
 *
 * @param x - The vector.
 * @returns |x|.
 * @example
 * ```ts
 * import { dnrm2 } from 'mdml-engine/numerics/linalg'
 * dnrm2([3e200, 4e200]) // => 5e200
 * dnrm2([Infinity, -Infinity]) // => Infinity
 * ```
 */
export function dnrm2(x: ArrayLike<number>): number {
  let scale = 0
  let ssq = 1
  let infinite = false
  for (let i = 0; i < x.length; i++) {
    const a = Math.abs(x[i] ?? 0)
    if (a === 0) continue
    // An infinite entry would make the scaled sum Infinity / Infinity: the norm is Infinity (NaN if any entry is NaN).
    if (a === Number.POSITIVE_INFINITY) {
      infinite = true
      continue
    }
    if (scale < a) {
      ssq = 1 + ssq * (scale / a) ** 2
      scale = a
    } else ssq += (a / scale) ** 2
  }
  return infinite ? (Number.isNaN(ssq) ? Number.NaN : Number.POSITIVE_INFINITY) : scale * Math.sqrt(ssq)
}

/**
 * y + alpha x as a new array (BLAS daxpy, without overwriting y).
 *
 * @param alpha - Scale of x.
 * @param x - Vector.
 * @param y - Vector (same length).
 * @returns y + alpha x.
 * @example
 * ```ts
 * import { daxpy } from 'mdml-engine/numerics/linalg'
 * Array.from(daxpy(2, [1, 1], [1, 2])) // => [3, 4]
 * ```
 */
export function daxpy(alpha: number, x: ArrayLike<number>, y: ArrayLike<number>): Float64Array {
  const out = new Float64Array(y.length)
  for (let i = 0; i < y.length; i++) out[i] = (y[i] ?? 0) + alpha * (x[i] ?? 0)
  return out
}

/**
 * alpha x as a new array (BLAS dscal, without overwriting x).
 *
 * @param alpha - The scale.
 * @param x - The vector.
 * @returns alpha x.
 * @example
 * ```ts
 * import { dscal } from 'mdml-engine/numerics/linalg'
 * Array.from(dscal(0.5, [2, 4])) // => [1, 2]
 * ```
 */
export function dscal(alpha: number, x: ArrayLike<number>): Float64Array {
  const out = new Float64Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = alpha * (x[i] ?? 0)
  return out
}

/**
 * The matrix-vector product A x, or A^T x, of a row-major m x n matrix (BLAS dgemv with alpha 1, beta 0).
 *
 * @param a - The matrix, row-major.
 * @param m - Rows.
 * @param n - Columns.
 * @param x - The vector (length n, or m when transposed).
 * @param transpose - Whether to multiply by A^T (default false).
 * @returns The product.
 * @example
 * ```ts
 * import { dgemv } from 'mdml-engine/numerics/linalg'
 * Array.from(dgemv([1, 2, 3, 4], 2, 2, [1, 1], true)) // => [4, 6]
 * ```
 */
export function dgemv(
  a: ArrayLike<number>,
  m: number,
  n: number,
  x: ArrayLike<number>,
  transpose = false,
): Float64Array {
  if (transpose) {
    const out = new Float64Array(n)
    for (let i = 0; i < m; i++) {
      const xi = x[i] ?? 0
      for (let j = 0; j < n; j++) out[j] = (out[j] ?? 0) + (a[i * n + j] ?? 0) * xi
    }
    return out
  }
  const out = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += (a[i * n + j] ?? 0) * (x[j] ?? 0)
    out[i] = s
  }
  return out
}

/**
 * The matrix product A B of row-major matrices, A m x k and B k x n, either optionally transposed (BLAS dgemm with
 * alpha 1, beta 0): the small dense products of geometry and calibration code.
 *
 * @param a - A, row-major (k x m when transposed).
 * @param b - B, row-major (n x k when transposed).
 * @param m - Rows of the product.
 * @param k - The inner dimension.
 * @param n - Columns of the product.
 * @param options - transposeA and transposeB (default false).
 * @returns The product, m x n row-major.
 * @example
 * ```ts
 * import { dgemm } from 'mdml-engine/numerics/linalg'
 * Array.from(dgemm([1, 2, 3, 4], [5, 6, 7, 8], 2, 2, 2)) // => [19, 22, 43, 50]
 * Array.from(dgemm([1, 2, 3, 4], [1, 0, 0, 1], 2, 2, 2, { transposeA: true })) // => [1, 3, 2, 4]
 * ```
 */
export function dgemm(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  m: number,
  k: number,
  n: number,
  options: { readonly transposeA?: boolean; readonly transposeB?: boolean } = {},
): Float64Array {
  const ta = options.transposeA === true
  const tb = options.transposeB === true
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++)
    for (let p = 0; p < k; p++) {
      const aip = (ta ? a[p * m + i] : a[i * k + p]) ?? 0
      if (aip === 0) continue
      for (let j = 0; j < n; j++)
        out[i * n + j] = (out[i * n + j] ?? 0) + aip * ((tb ? b[j * k + p] : b[p * n + j]) ?? 0)
    }
  return out
}
