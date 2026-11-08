// Private helpers of the survival domain: scalar views of the special functions that p-values and intervals need, and
// the symmetric positive-definite inverse the covariances come from.
import { OmaiError } from '../base/index.ts'
import { dpotrf, dpotrs } from '../numerics/linalg/index.ts'
import { chdtrc, ndtr, ndtri } from '../numerics/special/index.ts'
import { item } from '../tensor/index.ts'

export const normCdf = (x: number): number => item(ndtr(x))
export const normSf = (x: number): number => item(ndtr(-x))
export const normPpf = (p: number): number => item(ndtri(p))
export const chi2Sf = (df: number, x: number): number => item(chdtrc(df, x))

/** The inverse of a symmetric positive-definite matrix (row-major, `n x n`), or `undefined` when it is not. */
export function inverseSpd(a: ArrayLike<number>, n: number): Float64Array | undefined {
  const { factor, info } = dpotrf(a, n)
  if (info !== 0) return undefined
  const out = new Float64Array(n * n)
  for (let c = 0; c < n; c++) {
    const e = new Float64Array(n)
    e[c] = 1
    const col = dpotrs(factor, e, n)
    for (let r = 0; r < n; r++) out[r * n + c] = col[r] ?? 0
  }
  return out
}

/** A matrix of numbers as rows of arrays. */
export function rowsOfMatrix(a: ArrayLike<number>, n: number): number[][] {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => a[i * n + j] ?? Number.NaN))
}

/** Throw unless `ok`, with a `value` error. */
export function require(ok: boolean, message: string): void {
  if (!ok) throw new OmaiError('value', message)
}
