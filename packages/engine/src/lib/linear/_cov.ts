/**
 * Small covariance helpers of the likelihood-style fits of this folder: the inverse of a square matrix (by a
 * pseudo-inverse when it is singular), the sandwich `A^-1 B A^-1`, and a flat-to-nested conversion for stored
 * covariances.
 */

import { inv, pinv } from '../numerics/linalg/index.ts'
import { primalOf, toFlat } from '../tensor/index.ts'
import { matrixTensor } from './design.ts'

/** The inverse of a flat row-major `n x n` matrix; the pseudo-inverse when it is singular or not finite. */
export function inverseOf(a: ArrayLike<number>, n: number): Float64Array {
  const t = matrixTensor(a, n, n)
  try {
    const out = Float64Array.from(toFlat(primalOf(inv(t))))
    if (out.every(Number.isFinite)) return out
  } catch {
    // fall through to the pseudo-inverse
  }
  return Float64Array.from(toFlat(primalOf(pinv(t))))
}

/** `A B A` for flat row-major `n x n` matrices (the sandwich covariance when `a` is the inverse bread). */
export function sandwich(a: ArrayLike<number>, b: ArrayLike<number>, n: number): Float64Array {
  const ab = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let k = 0; k < n; k++) {
      const aik = a[i * n + k] ?? 0
      if (aik !== 0) for (let j = 0; j < n; j++) ab[i * n + j] = (ab[i * n + j] ?? 0) + aik * (b[k * n + j] ?? 0)
    }
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let k = 0; k < n; k++) {
      const v = ab[i * n + k] ?? 0
      if (v !== 0) for (let j = 0; j < n; j++) out[i * n + j] = (out[i * n + j] ?? 0) + v * (a[k * n + j] ?? 0)
    }
  return out
}

/** A flat row-major `n x n` matrix as nested arrays (what the parameters of a fit store). */
export function nested(a: ArrayLike<number>, n: number): number[][] {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => a[i * n + j] ?? Number.NaN))
}

/** The flat row-major form of nested arrays. */
export function flat(a: readonly (readonly number[])[]): Float64Array {
  const n = a.length
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i * n + j] = a[i]?.[j] ?? Number.NaN
  return out
}
