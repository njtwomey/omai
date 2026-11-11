// Small dense symmetric positive-definite algebra for the multivariate Gaussian message: a Cholesky factorisation that
// answers "not positive definite" with `undefined` instead of an exception, and the solves, inverse and log determinant
// that follow from it. Matrices are row-major `Float64Array`s over numerics/linalg.
import { cholesky, choSolve } from '../../numerics/linalg/index.ts'
import { fromStorage, toFlat } from '../../tensor/index.ts'

/** A factored symmetric positive-definite matrix. */
export interface Spd {
  readonly n: number
  /** log det A. */
  readonly logDet: number
  /** A^-1 b. */
  solve(b: ArrayLike<number>): Float64Array
  /** A^-1, row-major. */
  inverse(): Float64Array
}

const mat = (a: ArrayLike<number>, n: number, m = n): ReturnType<typeof fromStorage> =>
  fromStorage(Float64Array.from(a), [n, m], 'float64')

/**
 * Factor a symmetric matrix, or say it is not positive definite.
 *
 * @param a - The matrix, row-major `n x n`.
 * @param n - Its order.
 * @returns The factorisation, or `undefined` when the matrix is not positive definite (or not finite).
 */
export function spd(a: ArrayLike<number>, n: number): Spd | undefined {
  for (let i = 0; i < n * n; i++) if (!Number.isFinite(a[i] ?? Number.NaN)) return undefined
  let l: ReturnType<typeof cholesky>
  try {
    l = cholesky(mat(a, n))
  } catch {
    return undefined
  }
  const lf = toFlat(l)
  let logDet = 0
  for (let i = 0; i < n; i++) logDet += 2 * Math.log(lf[i * n + i] ?? 1)
  return {
    n,
    logDet,
    solve: (b) => Float64Array.from(toFlat(choSolve(l, fromStorage(Float64Array.from(b), [n], 'float64')))),
    inverse: () => {
      const eye = new Float64Array(n * n)
      for (let i = 0; i < n; i++) eye[i * n + i] = 1
      return Float64Array.from(toFlat(choSolve(l, mat(eye, n))))
    },
  }
}
