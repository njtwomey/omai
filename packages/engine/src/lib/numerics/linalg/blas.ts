// Level-1 and level-2 BLAS kernels on plain number arrays, for algorithms that iterate on primal values (Krylov
// solvers, Lanczos, regularising iterations). Differentiable code uses the tensor operations instead.

/**
 * The dot product sum_i x_i y_i (BLAS ddot).
 *
 * @param x - First vector.
 * @param y - Second vector (same length).
 * @returns The dot product.
 * @example
 * ```ts
 * import { ddot } from 'whitebox-engine/numerics/linalg'
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
 * import { idamax } from 'whitebox-engine/numerics/linalg'
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
 * import { dasum } from 'whitebox-engine/numerics/linalg'
 * dasum([1, -3, 2]) // => 6
 * ```
 */
export function dasum(x: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < x.length; i++) s += Math.abs(x[i] ?? 0)
  return s
}

/**
 * The Euclidean norm |x| (BLAS dnrm2), accumulated with scaling so that it neither overflows nor underflows.
 *
 * @param x - The vector.
 * @returns |x|.
 * @example
 * ```ts
 * import { dnrm2 } from 'whitebox-engine/numerics/linalg'
 * dnrm2([3e200, 4e200]) // => 5e200
 * ```
 */
export function dnrm2(x: ArrayLike<number>): number {
  let scale = 0
  let ssq = 1
  for (let i = 0; i < x.length; i++) {
    const a = Math.abs(x[i] ?? 0)
    if (a === 0) continue
    if (scale < a) {
      ssq = 1 + ssq * (scale / a) ** 2
      scale = a
    } else ssq += (a / scale) ** 2
  }
  return scale * Math.sqrt(ssq)
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
 * import { daxpy } from 'whitebox-engine/numerics/linalg'
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
 * import { dscal } from 'whitebox-engine/numerics/linalg'
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
 * import { dgemv } from 'whitebox-engine/numerics/linalg'
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
