/**
 * Structured matrices and their solvers, scipy.linalg style: Toeplitz, circulant, Hankel and block-diagonal
 * constructors, tridiagonal, banded (LAPACK band storage) and Toeplitz solves, and Kronecker-product systems solved
 * without forming the product.
 */

import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  concat,
  expandDims,
  eye,
  flip,
  mul,
  pad,
  promoteFloat,
  reshape,
  sliceAxis,
  swapaxes,
  type TensorLike,
  take,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { solve } from './lu.ts'

const indexMatrix = (rows: number, cols: number, f: (i: number, j: number) => number): number[] =>
  Array.from({ length: rows * cols }, (_, k) => f(Math.floor(k / cols), k % cols))

/**
 * A Toeplitz matrix with first column `c` and first row `r` (scipy.linalg.toeplitz; `r` defaults to `c`, and `r[0]`
 * is ignored).
 *
 * @param c - First column `[m]`.
 * @param r - First row `[n]`.
 * @returns The matrix `[m, n]`, differentiable in `c` and `r`.
 * @concept structured-matrices
 * @example
 * ```ts
 * import { toeplitz } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(toeplitz([1, 2, 3], [1, 4])) // => [[1, 4], [2, 1], [3, 2]]
 * ```
 */
export function toeplitz(c: TensorLike, r?: TensorLike): Value {
  const [col] = promoteFloat(c) as [Value]
  const row = r === undefined ? col : (promoteFloat(r)[0] as Value)
  const m = col.shape[0] ?? 0
  const n = row.shape[0] ?? 0
  // v = [r[n-1], ..., r[1], c[0], ..., c[m-1]]; T[i, j] = v[n - 1 + i - j].
  const v = concat([flip(sliceAxis(row, 0, 1, n)), col])
  return reshape(
    take(
      v,
      indexMatrix(m, n, (i, j) => n - 1 + i - j),
    ),
    [m, n],
  )
}

/**
 * A circulant matrix with first column `c` (scipy.linalg.circulant): C[i, j] = c[(i - j) mod n].
 *
 * @param c - First column `[n]`.
 * @returns The matrix `[n, n]`, differentiable.
 * @example
 * ```ts
 * import { circulant } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(circulant([1, 2, 3])) // => [[1, 3, 2], [2, 1, 3], [3, 2, 1]]
 * ```
 */
export function circulant(c: TensorLike): Value {
  const [col] = promoteFloat(c) as [Value]
  const n = col.shape[0] ?? 0
  return reshape(
    take(
      col,
      indexMatrix(n, n, (i, j) => (((i - j) % n) + n) % n),
    ),
    [n, n],
  )
}

/**
 * A Hankel matrix with first column `c` and last row `r` (scipy.linalg.hankel; `r` defaults to zeros, and `r[0]` is
 * ignored): H[i, j] = c[i + j] or r[i + j - m + 1].
 *
 * @param c - First column `[m]`.
 * @param r - Last row `[n]`.
 * @returns The matrix `[m, n]`, differentiable.
 * @example
 * ```ts
 * import { hankel } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(hankel([1, 2, 3])) // => [[1, 2, 3], [2, 3, 0], [3, 0, 0]]
 * ```
 */
export function hankel(c: TensorLike, r?: TensorLike): Value {
  const [col] = promoteFloat(c) as [Value]
  const m = col.shape[0] ?? 0
  const row = r === undefined ? zeros([m]) : (promoteFloat(r)[0] as Value)
  const n = row.shape[0] ?? 0
  const v = concat([col, sliceAxis(row, 0, 1, n)])
  return reshape(
    take(
      v,
      indexMatrix(m, n, (i, j) => i + j),
    ),
    [m, n],
  )
}

/**
 * A block-diagonal matrix (scipy.linalg.block_diag).
 *
 * @param blocks - The diagonal blocks (2-D).
 * @returns The block-diagonal matrix, differentiable in every block.
 * @example
 * ```ts
 * import { blockDiag } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(blockDiag([[1]], [[2, 3]])) // => [[1, 0, 0], [0, 2, 3]]
 * ```
 */
export function blockDiag(...blocks: readonly TensorLike[]): Value {
  const bs = blocks.map((b) => promoteFloat(b)[0] as Value)
  const widths = bs.map((b) => b.shape[1] ?? 0)
  const total = widths.reduce((a, b) => a + b, 0)
  let before = 0
  const rows = bs.map((b, i) => {
    const w = widths[i] ?? 0
    const padded = pad(b, [
      [0, 0],
      [before, total - before - w],
    ])
    before += w
    return padded
  })
  return concat(rows, 0)
}

/**
 * Solve a tridiagonal system (the matrix built from its three diagonals, then an LU solve).
 *
 * @param lower - Sub-diagonal `[n - 1]`.
 * @param diagonal - Diagonal `[n]`.
 * @param upper - Super-diagonal `[n - 1]`.
 * @param b - Right-hand side `[n]` or `[n, k]`.
 * @returns The solution, differentiable in every argument.
 * @example
 * ```ts
 * import { solveTridiagonal } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(solveTridiagonal([1], [2, 2], [1], [3, 3])) // => [1, 1]
 * ```
 */
export function solveTridiagonal(lower: TensorLike, diagonal: TensorLike, upper: TensorLike, b: TensorLike): Value {
  const [l, d, u] = promoteFloat(lower, diagonal, upper) as [Value, Value, Value]
  const n = d.shape[0] ?? 0
  const matrix = add(
    add(mul(expandDims(d, -1), eye(n)), mul(expandDims(pad(u, [[0, 1]]), -1), eye(n, n, 1))),
    mul(expandDims(pad(l, [[1, 0]]), -1), eye(n, n, -1)),
  )
  return solve(matrix, b)
}

/**
 * Solve a banded system given in LAPACK's band storage (scipy.linalg.solve_banded): `ab[u + i - j, j] = A[i, j]`.
 *
 * @param bandwidths - `[l, u]`, the numbers of sub- and super-diagonals.
 * @param ab - Band storage `[l + u + 1, n]`.
 * @param b - Right-hand side `[n]` or `[n, k]`.
 * @returns The solution, differentiable.
 * @example
 * ```ts
 * import { solveBanded } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(solveBanded([1, 1], [[0, 1], [2, 2], [1, 0]], [3, 3])) // => [1, 1]
 * ```
 */
export function solveBanded(bandwidths: readonly [number, number], ab: TensorLike, b: TensorLike): Value {
  const [band] = promoteFloat(ab) as [Value]
  const [l, u] = bandwidths
  const n = band.shape[1] ?? 0
  if ((band.shape[0] ?? 0) !== l + u + 1) throw new WhiteboxError('shape', `solveBanded: ab needs ${l + u + 1} rows`)
  const flat = concat([reshape(band, [(l + u + 1) * n]), zeros([1])])
  const zero = (l + u + 1) * n
  const idx = indexMatrix(n, n, (i, j) => (i - j > l || j - i > u ? zero : (u + i - j) * n + j))
  return solve(reshape(take(flat, idx), [n, n]), b)
}

/**
 * Solve a Toeplitz system T x = b (scipy.linalg.solve_toeplitz).
 *
 * @param c - First column.
 * @param r - First row (defaults to `c`).
 * @param b - Right-hand side.
 * @returns The solution, differentiable.
 * @example
 * ```ts
 * import { solveToeplitz } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(solveToeplitz([2, 1], undefined, [3, 3])) // => [1, 1]
 * ```
 */
export function solveToeplitz(c: TensorLike, r: TensorLike | undefined, b: TensorLike): Value {
  return solve(toeplitz(c, r), b)
}

/**
 * Solve (A kron B) x = c without forming the Kronecker product: with row-major vectorisation, (A kron B) vec(X) =
 * vec(A X B^T), so X = A^{-1} C B^{-T}.
 *
 * @param a - Matrix `[m, m]`.
 * @param b - Matrix `[n, n]`.
 * @param c - Right-hand side `[m n]`.
 * @returns x `[m n]`, differentiable.
 * @example
 * ```ts
 * import { kroneckerSolve } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(kroneckerSolve([[2]], [[1, 0], [0, 4]], [2, 8])) // => [1, 1]
 * ```
 */
export function kroneckerSolve(a: TensorLike, b: TensorLike, c: TensorLike): Value {
  const [av, bv, cv] = promoteFloat(a, b, c) as [Value, Value, Value]
  const m = av.shape[0] ?? 0
  const n = bv.shape[0] ?? 0
  const cm = reshape(cv, [m, n])
  // X = A^{-1} (C B^{-T}) = A^{-1} (B^{-1} C^T)^T.
  return reshape(solve(av, swapaxes(solve(bv, swapaxes(cm, 0, 1)), 0, 1)), [m * n])
}
