// Small dense-matrix conveniences over numerics/linalg for the Gaussian models (not re-exported): plain number[][] in,
// plain number[][] out.
import { OmaiError } from '../base/index.ts'
import { cholesky, ddot, dgemv, inv, slogdet, solve } from '../numerics/linalg/index.ts'
import { item, toArray } from '../tensor/index.ts'

export type Rows = readonly (readonly number[])[]

/** A^{-1}. */
export function inverse(a: Rows): number[][] {
  if (a.length === 0) return []
  return toArray(inv(a.map((r) => [...r]))) as number[][]
}

/** A^{-1} b for a vector b. */
export function solveVector(a: Rows, b: readonly number[]): number[] {
  if (a.length === 0) return []
  return toArray(
    solve(
      a.map((r) => [...r]),
      [...b],
    ),
  ) as number[]
}

/** log |det A| for a positive-definite A (throws otherwise). */
export function logdetPositive(a: Rows, name: string): number {
  if (a.length === 0) return 0
  const { sign, logabsdet } = slogdet(a.map((r) => [...r]))
  if (item(sign) <= 0) throw new OmaiError('value', `${name}: the matrix is not positive definite`)
  return item(logabsdet)
}

/** The lower Cholesky factor. */
export function choleskyLower(a: Rows): number[][] {
  if (a.length === 0) return []
  return toArray(cholesky(a.map((r) => [...r]))) as number[][]
}

/** Rows and columns of a matrix at some indices. */
export function sub(a: Rows, rows: readonly number[], cols: readonly number[]): number[][] {
  return rows.map((i) => cols.map((j) => a[i]?.[j] ?? 0))
}

/** A x (BLAS dgemv on the flattened rows). */
export function matVec(a: Rows, x: readonly number[]): number[] {
  return Array.from(dgemv(a.flat(), a.length, a[0]?.length ?? x.length, x))
}

/** A B, column by column. */
export function matMul(a: Rows, b: Rows): number[][] {
  const cols = transposeMatrix(b).map((c) => matVec(a, c))
  return transposeMatrix(cols)
}

/** A^T. */
export function transposeMatrix(a: Rows): number[][] {
  const n = a[0]?.length ?? 0
  return Array.from({ length: n }, (_, j) => a.map((r) => r[j] ?? 0))
}

/** x . y (BLAS ddot). */
export function dotVec(x: readonly number[], y: readonly number[]): number {
  return ddot(x, y)
}

/** (A + A^T) / 2, to remove rounding asymmetry. */
export function symmetrise(a: Rows): number[][] {
  return a.map((r, i) => r.map((v, j) => (v + (a[j]?.[i] ?? 0)) / 2))
}

/** Check a square symmetric matrix of a given size. */
export function checkSquare(a: Rows, n: number, name: string): void {
  if (a.length !== n || a.some((r) => r.length !== n))
    throw new OmaiError('shape', `${name}: expected a ${n} x ${n} matrix`)
}
