/**
 * Shared plumbing of the geometry files: small row-major matrix helpers over the BLAS and LAPACK kernels of
 * `numerics/linalg`, the null vector of a homogeneous system, and nonlinear refinement of reprojection residuals by
 * `optim`'s Levenberg-Marquardt. Private to vision.
 */

import { MdmlError, toRows } from '../base/index.ts'
import { dgemm, dgesv, dgesvd } from '../numerics/linalg/index.ts'
import { levenbergMarquardt } from '../optim/index.ts'
import { primalOf, type Tensor, type TensorLike, toFlat, type Value } from '../tensor/index.ts'
import { valuesOf } from './_image.ts'

/** A dense matrix as rows. */
export type MutableRows = number[][]

/** Rows to a row-major array. */
export const flatOf = (m: readonly (readonly number[])[]): Float64Array => Float64Array.from(m.flat())

/** The product of two matrices given as rows. */
export function matMul(a: readonly (readonly number[])[], b: readonly (readonly number[])[]): MutableRows {
  const m = a.length
  const k = b.length
  const n = b[0]?.length ?? 0
  return toRows(dgemm(flatOf(a), flatOf(b), m, k, n), m, n)
}

/** The transpose of a matrix given as rows. */
export const transposeOf = (a: readonly (readonly number[])[]): MutableRows =>
  Array.from({ length: a[0]?.length ?? 0 }, (_, j) => a.map((r) => r[j] ?? 0))

/** The inverse of a square matrix (by LU solves against the identity). */
export function invOf(a: readonly (readonly number[])[]): MutableRows {
  const n = a.length
  const eye = Float64Array.from({ length: n * n }, (_, i) => (Math.floor(i / n) === i % n ? 1 : 0))
  // dgesv solves A X = B for B (and X) row-major n x n.
  return toRows(dgesv(flatOf(a), eye, n, n), n, n)
}

/** The determinant of a 3 x 3 matrix. */
export function det3(m: readonly (readonly number[])[]): number {
  const [a = [], b = [], c = []] = m
  return (
    (a[0] ?? 0) * ((b[1] ?? 0) * (c[2] ?? 0) - (b[2] ?? 0) * (c[1] ?? 0)) -
    (a[1] ?? 0) * ((b[0] ?? 0) * (c[2] ?? 0) - (b[2] ?? 0) * (c[0] ?? 0)) +
    (a[2] ?? 0) * ((b[0] ?? 0) * (c[1] ?? 0) - (b[1] ?? 0) * (c[0] ?? 0))
  )
}

/** The SVD of a matrix given as rows: U (m x k), s (k) and V (n x k) as rows, k = min(m, n). */
export function svdOf(a: readonly (readonly number[])[]): { u: MutableRows; s: number[]; v: MutableRows } {
  const m = a.length
  const n = a[0]?.length ?? 0
  const { u, s, v } = dgesvd(flatOf(a), m, n)
  const k = s.length
  return { u: toRows(u, m, k), s: Array.from(s), v: toRows(v, n, k) }
}

/**
 * The unit vector x minimising |A x| (the right singular vector of the smallest singular value). Fewer rows than
 * columns are padded with zero rows, since the reduced SVD keeps only min(m, n) singular vectors.
 */
export function nullVector(rows: readonly (readonly number[])[]): number[] {
  const n = rows[0]?.length ?? 0
  const padded = [
    ...rows.map((r) => [...r]),
    ...Array.from({ length: Math.max(0, n - rows.length) }, () => new Array<number>(n).fill(0)),
  ]
  const { v, s } = svdOf(padded)
  const k = s.length
  return v.map((r) => r[k - 1] ?? 0)
}

/** 3-D points as rows, from a tensor `[n, 3]` or nested arrays. */
export function points3Of(
  points: TensorLike | readonly (readonly number[])[],
  name: string,
): [number, number, number][] {
  if (Array.isArray(points) && points.length === 0) return []
  const { data, shape } = valuesOf(points as TensorLike)
  if (shape.length !== 2 || shape[1] !== 3) throw new MdmlError('shape', `${name}: points are [n, 3]`)
  return Array.from({ length: shape[0] ?? 0 }, (_, i) => [data[3 * i] ?? 0, data[3 * i + 1] ?? 0, data[3 * i + 2] ?? 0])
}

/** Minimise half the squared norm of residuals by Levenberg-Marquardt from x0; returns the solution. */
export function refine(residuals: (p: Value) => Value, x0: readonly number[], maxIterations = 100): number[] {
  if (x0.length === 0) return []
  const r = levenbergMarquardt(residuals, [...x0], { maxIterations, ftol: 1e-15, xtol: 1e-15, gradTol: 1e-15 })
  return toFlat(primalOf(r.x as Value) as Tensor)
}
