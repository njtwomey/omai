// Private dense linear-model helpers of the stats domain, composed from numerics/linalg's LAPACK-style routines:
// least squares with coefficient covariances, Gram matrices and SPD inverses on row-major number arrays.
import { WhiteboxError } from '../base/index.ts'
import { dgeqrf, dpotrf, dpotrs, dsyev, dtrsv } from '../numerics/linalg/index.ts'

/** An ordinary least-squares fit. */
export interface OlsFit {
  readonly beta: Float64Array
  readonly residuals: Float64Array
  /** Residual sum of squares. */
  readonly rss: number
  /** Residual degrees of freedom n - p. */
  readonly dfResid: number
  /** (X^T X)^-1, row-major p x p. */
  readonly xtxInv: Float64Array
  /** Standard errors sqrt(sigma^2 diag((X^T X)^-1)) with sigma^2 = rss / (n - p). */
  readonly se: Float64Array
}

/** Ordinary least squares y ~ X by Householder QR (X row-major n x p, full column rank). */
export function ols(X: ArrayLike<number>, n: number, p: number, y: ArrayLike<number>): OlsFit {
  if (n < p) throw new WhiteboxError('value', 'ols: fewer observations than coefficients')
  const { q, r } = dgeqrf(X, n, p)
  for (let i = 0; i < p; i++)
    if (!(Math.abs(r[i * p + i] ?? 0) > 1e-12 * Math.abs(r[0] ?? 1)))
      throw new WhiteboxError('singular', 'ols: the design matrix is rank deficient')
  const qty = new Float64Array(p)
  for (let j = 0; j < p; j++) {
    let s = 0
    for (let i = 0; i < n; i++) s += (q[i * p + j] ?? 0) * (y[i] ?? 0)
    qty[j] = s
  }
  const beta = dtrsv(r, qty, p, { lower: false })
  const residuals = new Float64Array(n)
  let rss = 0
  for (let i = 0; i < n; i++) {
    let f = 0
    for (let j = 0; j < p; j++) f += (X[i * p + j] ?? 0) * (beta[j] ?? 0)
    const e = (y[i] ?? 0) - f
    residuals[i] = e
    rss += e * e
  }
  // (X^T X)^-1 = R^-1 R^-T.
  const rinv = new Float64Array(p * p)
  for (let j = 0; j < p; j++) {
    const e = new Float64Array(p)
    e[j] = 1
    const col = dtrsv(r, e, p, { lower: false })
    for (let i = 0; i < p; i++) rinv[i * p + j] = col[i] ?? 0
  }
  const xtxInv = new Float64Array(p * p)
  for (let i = 0; i < p; i++)
    for (let j = 0; j < p; j++) {
      let s = 0
      for (let k = 0; k < p; k++) s += (rinv[i * p + k] ?? 0) * (rinv[j * p + k] ?? 0)
      xtxInv[i * p + j] = s
    }
  const dfResid = n - p
  const sigma2 = dfResid > 0 ? rss / dfResid : Number.NaN
  const se = Float64Array.from({ length: p }, (_, i) => Math.sqrt(sigma2 * (xtxInv[i * p + i] ?? 0)))
  return { beta, residuals, rss, dfResid, xtxInv, se }
}

/** The inverse of a symmetric positive-definite matrix (row-major n x n) via Cholesky. */
export function invSpd(a: ArrayLike<number>, n: number): Float64Array {
  const { factor, info } = dpotrf(a, n)
  if (info !== 0) throw new WhiteboxError('singular', 'the matrix is not positive definite')
  const out = new Float64Array(n * n)
  for (let j = 0; j < n; j++) {
    const e = new Float64Array(n)
    e[j] = 1
    const col = dpotrs(factor, e, n)
    for (let i = 0; i < n; i++) out[i * n + j] = col[i] ?? 0
  }
  return out
}

/**
 * The Moore-Penrose pseudo-inverse of a symmetric matrix (numpy.linalg.pinv with hermitian=True): eigenvalues below
 * rcond times the largest in magnitude are treated as zero.
 */
export function pinvSym(a: ArrayLike<number>, n: number, rcond = 1e-15): Float64Array {
  const { w, v } = dsyev(a, n)
  let top = 0
  for (const e of w) top = Math.max(top, Math.abs(e))
  const out = new Float64Array(n * n)
  for (let k = 0; k < n; k++) {
    const e = w[k] ?? 0
    if (!(Math.abs(e) > rcond * top)) continue
    for (let i = 0; i < n; i++) {
      const vi = (v[i * n + k] ?? 0) / e
      for (let j = 0; j < n; j++) out[i * n + j] = (out[i * n + j] ?? 0) + vi * (v[j * n + k] ?? 0)
    }
  }
  return out
}

/** log det of a symmetric positive-definite matrix (NaN when not positive definite). */
export function logdetSpd(a: ArrayLike<number>, n: number): number {
  const { factor, info } = dpotrf(a, n)
  if (info !== 0) return Number.NaN
  let s = 0
  for (let i = 0; i < n; i++) s += 2 * Math.log(factor[i * n + i] ?? 0)
  return s
}

/** Eigenvalues of a symmetric matrix, ascending. */
export function eigvalsSym(a: ArrayLike<number>, n: number): Float64Array {
  return dsyev(a, n).w
}

/** C = A B for row-major A (m x k) and B (k x n). */
export function matmul(a: ArrayLike<number>, b: ArrayLike<number>, m: number, k: number, n: number): Float64Array {
  const c = new Float64Array(m * n)
  for (let i = 0; i < m; i++)
    for (let l = 0; l < k; l++) {
      const v = a[i * k + l] ?? 0
      if (v === 0) continue
      for (let j = 0; j < n; j++) c[i * n + j] = (c[i * n + j] ?? 0) + v * (b[l * n + j] ?? 0)
    }
  return c
}

/** The transpose of a row-major m x n matrix. */
export function transpose(a: ArrayLike<number>, m: number, n: number): Float64Array {
  const t = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) t[j * m + i] = a[i * n + j] ?? 0
  return t
}

/** The sample covariance matrix (ddof 1) of the columns of a row-major n x p matrix. */
export function covarianceOfColumns(x: ArrayLike<number>, n: number, p: number, ddof = 1): Float64Array {
  const mean = new Float64Array(p)
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) + (x[i * p + j] ?? 0) / n
  const c = new Float64Array(p * p)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < p; a++) {
      const da = (x[i * p + a] ?? 0) - (mean[a] ?? 0)
      for (let b = a; b < p; b++) c[a * p + b] = (c[a * p + b] ?? 0) + da * ((x[i * p + b] ?? 0) - (mean[b] ?? 0))
    }
  for (let a = 0; a < p; a++)
    for (let b = a; b < p; b++) {
      const v = (c[a * p + b] ?? 0) / (n - ddof)
      c[a * p + b] = v
      c[b * p + a] = v
    }
  return c
}

/**
 * An orthonormal basis of the contrasts of k conditions (k x (k - 1), row-major): normalised Helmert contrasts, each
 * column orthogonal to the constant.
 */
export function orthonormalContrasts(k: number): Float64Array {
  const c = new Float64Array(k * (k - 1))
  for (let j = 1; j < k; j++) {
    const norm = Math.sqrt(j * (j + 1))
    for (let i = 0; i < j; i++) c[i * (k - 1) + (j - 1)] = 1 / norm
    c[j * (k - 1) + (j - 1)] = -j / norm
  }
  return c
}
