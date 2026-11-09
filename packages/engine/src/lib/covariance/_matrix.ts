// Small dense helpers shared by the covariance modules: symmetric eigendecompositions as plain arrays (eigenvalues
// descending, eigenvectors as the columns of a row-major matrix), the matrix rebuilt from them, correlation scaling,
// condition numbers and the Gaussian log likelihood of held-out rows.

import { OmaiError } from '../base/index.ts'
import { matrixView } from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { asTensor, type TensorLike, toArray, toFlat } from '../tensor/index.ts'

/** A dense matrix as rows. */
export type Rows = number[][]

/** A symmetric eigendecomposition: eigenvalues descending and the matching eigenvectors as the columns of `vectors`. */
export interface Eigen {
  readonly values: number[]
  /** Row-major `[p, p]`: column `k` is the eigenvector of `values[k]`. */
  readonly vectors: Float64Array
}

/** The symmetric part of a square matrix, as rows. */
export function symmetrise(a: readonly (readonly number[])[]): Rows {
  const p = a.length
  return Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (_, j) => 0.5 * ((a[i]?.[j] ?? 0) + (a[j]?.[i] ?? 0))),
  )
}

/** The eigendecomposition of the symmetric part of `a`, eigenvalues descending. */
export function symmetricEigen(a: readonly (readonly number[])[]): Eigen {
  const p = a.length
  const { eigenvalues, eigenvectors } = linalg.eigh(symmetrise(a))
  const ascending = toFlat(eigenvalues)
  const v = toFlat(eigenvectors)
  const values = Array.from({ length: p }, (_, k) => ascending[p - 1 - k] ?? 0)
  const vectors = new Float64Array(p * p)
  for (let k = 0; k < p; k++) for (let i = 0; i < p; i++) vectors[i * p + k] = v[i * p + (p - 1 - k)] ?? 0
  return { values, vectors }
}

/** `V diag(values) V^T` from the columns of `vectors`, as rows. */
export function fromEigen(values: readonly number[], vectors: Float64Array): Rows {
  const p = values.length
  const out = new Float64Array(p * p)
  for (let k = 0; k < p; k++) {
    const lam = values[k] ?? 0
    if (lam === 0) continue
    for (let i = 0; i < p; i++) {
      const vik = (vectors[i * p + k] ?? 0) * lam
      for (let j = 0; j < p; j++) out[i * p + j] = (out[i * p + j] ?? 0) + vik * (vectors[j * p + k] ?? 0)
    }
  }
  return Array.from({ length: p }, (_, i) => Array.from(out.subarray(i * p, (i + 1) * p)))
}

/** The square root of the diagonal and the matrix scaled to unit diagonal (`C_ij / sqrt(C_ii C_jj)`). */
export function toCorrelation(cov: readonly (readonly number[])[]): { correlation: Rows; scale: number[] } {
  const p = cov.length
  const scale = Array.from({ length: p }, (_, i) => Math.sqrt(Math.max(cov[i]?.[i] ?? 0, 0)))
  const correlation = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (_, j) => {
      if (i === j) return 1
      const d = (scale[i] ?? 0) * (scale[j] ?? 0)
      return d > 0 ? Math.max(-1, Math.min(1, (cov[i]?.[j] ?? 0) / d)) : 0
    }),
  )
  return { correlation, scale }
}

/** `D C D` with `D = diag(scale)`: a correlation matrix back to a covariance. */
export function fromCorrelation(corr: readonly (readonly number[])[], scale: readonly number[]): Rows {
  return corr.map((row, i) => row.map((v, j) => v * (scale[i] ?? 0) * (scale[j] ?? 0)))
}

/** The Frobenius norm of `a - b`. */
export function frobeniusDistance(a: readonly (readonly number[])[], b: readonly (readonly number[])[]): number {
  let s = 0
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < (a[i]?.length ?? 0); j++) s += ((a[i]?.[j] ?? 0) - (b[i]?.[j] ?? 0)) ** 2
  return Math.sqrt(s)
}

/** The ratio of the largest to the smallest of eigenvalues (`Infinity` unless the smallest is positive). */
export function conditionOf(values: readonly number[]): number {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of values) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  return lo > 0 ? hi / lo : Number.POSITIVE_INFINITY
}

/** The rows of a data matrix, its column means (zeros when `assumeCentered`) and the empirical covariance `X^T X / n`. */
export function moments(
  x: TensorLike,
  who: string,
  options: { readonly assumeCentered?: boolean; readonly minSamples?: number } = {},
): { n: number; d: number; location: number[]; centred: Float64Array; covariance: Rows } {
  const m = matrixView(x, who, { minSamples: options.minSamples ?? 2 })
  const { n, d } = m
  const location = new Array<number>(d).fill(0)
  if (options.assumeCentered !== true) {
    for (let i = 0; i < n; i++)
      for (let j = 0; j < d; j++) location[j] = (location[j] ?? 0) + (m.data[i * d + j] ?? 0) / n
  }
  const centred = Float64Array.from(m.data, (v, k) => v - (location[k % d] ?? 0))
  const c = new Float64Array(d * d)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < d; a++) {
      const xa = centred[i * d + a] ?? 0
      for (let b = a; b < d; b++) c[a * d + b] = (c[a * d + b] ?? 0) + xa * (centred[i * d + b] ?? 0)
    }
  const covariance = Array.from({ length: d }, (_, a) =>
    Array.from({ length: d }, (_, b) => (c[Math.min(a, b) * d + Math.max(a, b)] ?? 0) / n),
  )
  return { n, d, location, centred, covariance }
}

/**
 * The mean Gaussian log likelihood of the rows of `x` under `N(location, covariance)`, from the eigendecomposition of
 * the covariance; `-Infinity` when the covariance is singular or not positive definite (its smallest eigenvalue at
 * most `1e-12` of its largest), where the density does not exist.
 */
export function gaussianLogLikelihood(
  x: TensorLike,
  location: readonly number[],
  covariance: readonly (readonly number[])[],
): number {
  const d = location.length
  const m = matrixView(x, 'gaussianLogLikelihood', { nFeatures: d })
  const eig = symmetricEigen(covariance)
  const top = eig.values[0] ?? 0
  const bottom = eig.values[d - 1] ?? 0
  if (!(bottom > 1e-12 * Math.max(top, Number.MIN_VALUE))) return Number.NEGATIVE_INFINITY
  let logdet = 0
  for (const v of eig.values) logdet += Math.log(v)
  let total = 0
  for (let i = 0; i < m.n; i++) {
    let q = 0
    for (let k = 0; k < d; k++) {
      let proj = 0
      for (let j = 0; j < d; j++)
        proj += ((m.data[i * d + j] ?? 0) - (location[j] ?? 0)) * (eig.vectors[j * d + k] ?? 0)
      q += (proj * proj) / (eig.values[k] ?? 1)
    }
    total += -0.5 * (q + logdet + d * Math.log(2 * Math.PI))
  }
  return total / m.n
}

/** Rows of a matrix-like value (a tensor or nested arrays), checked square. */
export function squareRows(a: TensorLike, who: string): Rows {
  const rows = toArray(asTensor(a)) as unknown as Rows
  const p = rows.length
  if (p === 0 || rows.some((r) => !Array.isArray(r) || r.length !== p))
    throw new OmaiError('shape', `${who}: expected a square matrix`)
  return rows.map((r) => Array.from(r))
}
