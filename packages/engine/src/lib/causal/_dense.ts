// Private plumbing of the causal domain: the causal data read once as plain arrays, weighted least squares with
// classical, heteroscedasticity-robust and cluster-robust covariances on number arrays, scalar normal and t
// distribution functions, and the small row and column selections the estimators share. Nothing here is exported
// from the domain.
import { MdmlError } from '../base/index.ts'
import { matrixView, targetVector } from '../learn/validation.ts'
import { dgesv } from '../numerics/linalg/index.ts'
import { betainc, chdtrc, fdtrc, ndtr, ndtri, stdtr, stdtrit } from '../numerics/special/index.ts'
import { primalOf, type TensorLike, toFlat } from '../tensor/index.ts'

const item = (v: Parameters<typeof toFlat>[0]): number => toFlat(primalOf(v))[0] ?? Number.NaN

/** The standard normal CDF. */
export const normCdf = (x: number): number => item(ndtr(x))
/** The standard normal quantile. */
export const normPpf = (p: number): number => item(ndtri(p))
/** Student's t CDF. */
export const tCdf = (df: number, t: number): number => item(stdtr(df, t))
/** Student's t quantile. */
export const tPpf = (df: number, p: number): number => item(stdtrit(df, p))
/** The regularised incomplete beta function. */
export const betaInc = (a: number, b: number, x: number): number => item(betainc(a, b, x))
/** The upper tail of the chi-squared distribution. */
export const chi2Sf = (df: number, x: number): number => item(chdtrc(df, x))
/** The upper tail of the F distribution. */
export const fSf = (d1: number, d2: number, x: number): number => item(fdtrc(d1, d2, x))

/** Two-sided p-value of a statistic against the normal (`df` undefined) or t distribution. */
export function twoSided(stat: number, df?: number): number {
  if (!Number.isFinite(stat)) return Number.isNaN(stat) ? Number.NaN : 0
  return df === undefined ? 2 * normCdf(-Math.abs(stat)) : 2 * tCdf(df, -Math.abs(stat))
}

/** Causal data read once: `x` row-major `[n, d]`, binary treatment `t` and outcome `y`. */
export interface Prepared {
  readonly n: number
  readonly d: number
  readonly x: Float64Array
  readonly t: Float64Array
  readonly y: Float64Array
}

/** Read covariates, a 0/1 treatment and an outcome, checking lengths and values. */
export function prepare(x: TensorLike, treatment: TensorLike, outcome: TensorLike, who: string): Prepared {
  const m = matrixView(x, who)
  const t = targetVector(treatment, who, m.n)
  const y = targetVector(outcome, who, m.n)
  for (const v of t) if (v !== 0 && v !== 1) throw new MdmlError('value', `${who}: the treatment must be 0 or 1`)
  return { n: m.n, d: m.d, x: m.data, t, y }
}

/** Rows of a prepared dataset (with repeats allowed, as a bootstrap resample is). */
export function takePrepared(p: Prepared, rows: ArrayLike<number>): Prepared {
  const n = rows.length
  const x = new Float64Array(n * p.d)
  const t = new Float64Array(n)
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const r = rows[i] ?? 0
    for (let j = 0; j < p.d; j++) x[i * p.d + j] = p.x[r * p.d + j] ?? 0
    t[i] = p.t[r] ?? 0
    y[i] = p.y[r] ?? 0
  }
  return { n, d: p.d, x, t, y }
}

/** The covariates with the treatment appended as the last column. */
export function withTreatment(p: Prepared, t: ArrayLike<number> = p.t): Float64Array {
  const w = p.d + 1
  const out = new Float64Array(p.n * w)
  for (let i = 0; i < p.n; i++) {
    for (let j = 0; j < p.d; j++) out[i * w + j] = p.x[i * p.d + j] ?? 0
    out[i * w + p.d] = t[i] ?? 0
  }
  return out
}

/** The rows of `x` (width `d`) listed in `rows`. */
export function pickRows(x: Float64Array, d: number, rows: ArrayLike<number>): Float64Array {
  const out = new Float64Array(rows.length * d)
  for (let i = 0; i < rows.length; i++) for (let j = 0; j < d; j++) out[i * d + j] = x[(rows[i] ?? 0) * d + j] ?? 0
  return out
}

/** The entries of `v` listed in `rows`. */
export function pick(v: ArrayLike<number>, rows: ArrayLike<number>): Float64Array {
  return Float64Array.from({ length: rows.length }, (_, i) => v[rows[i] ?? 0] ?? 0)
}

/** The sample mean. */
export function mean(v: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] ?? 0
  return s / Math.max(1, v.length)
}

/** The sample variance with `ddof` degrees of freedom removed. */
export function variance(v: ArrayLike<number>, ddof = 1): number {
  const m = mean(v)
  let s = 0
  for (let i = 0; i < v.length; i++) s += ((v[i] ?? 0) - m) ** 2
  return s / Math.max(1, v.length - ddof)
}

/** The inverse of a symmetric positive-definite (or general, as a fallback) `p x p` matrix. */
export function invert(a: Float64Array, p: number): Float64Array {
  const eye = new Float64Array(p * p)
  for (let i = 0; i < p; i++) eye[i * p + i] = 1
  return dgesv(a, eye, p, p)
}

/** The covariance types of {@link wls}. */
export type CovKind = 'nonrobust' | 'HC0' | 'HC1' | 'cluster'

/** A weighted least-squares fit on a row-major design. */
export interface Wls {
  readonly beta: Float64Array
  readonly residuals: Float64Array
  /** `(A' W A)^-1`. */
  readonly bread: Float64Array
  readonly n: number
  readonly p: number
  /** The sum of squared weighted residuals over the residual degrees of freedom. */
  readonly sigma2: number
}

/** Weighted least squares of `y` on the `p` columns of `a` (no intercept is added). */
export function wls(a: Float64Array, n: number, p: number, y: ArrayLike<number>, w?: ArrayLike<number>): Wls {
  const g = new Float64Array(p * p)
  const b = new Float64Array(p)
  for (let i = 0; i < n; i++) {
    const wi = w === undefined ? 1 : (w[i] ?? 0)
    if (wi === 0) continue
    for (let r = 0; r < p; r++) {
      const ar = (a[i * p + r] ?? 0) * wi
      b[r] = (b[r] ?? 0) + ar * (y[i] ?? 0)
      for (let c = r; c < p; c++) g[r * p + c] = (g[r * p + c] ?? 0) + ar * (a[i * p + c] ?? 0)
    }
  }
  for (let r = 0; r < p; r++) for (let c = 0; c < r; c++) g[r * p + c] = g[c * p + r] ?? 0
  const bread = invert(g, p)
  const beta = new Float64Array(p)
  for (let r = 0; r < p; r++)
    for (let c = 0; c < p; c++) beta[r] = (beta[r] ?? 0) + (bread[r * p + c] ?? 0) * (b[c] ?? 0)
  const residuals = new Float64Array(n)
  let ss = 0
  let count = 0
  for (let i = 0; i < n; i++) {
    let f = 0
    for (let j = 0; j < p; j++) f += (a[i * p + j] ?? 0) * (beta[j] ?? 0)
    residuals[i] = (y[i] ?? 0) - f
    const wi = w === undefined ? 1 : (w[i] ?? 0)
    ss += wi * (residuals[i] ?? 0) ** 2
    if (wi > 0) count++
  }
  return { beta, residuals, bread, n, p, sigma2: ss / Math.max(1, count - p) }
}

/** The coefficient covariance of a {@link wls} fit (HC1 and the cluster-robust CR1 carry statsmodels' small-sample factors). */
export function covariance(
  fit: Wls,
  a: Float64Array,
  kind: CovKind,
  w?: ArrayLike<number>,
  clusters?: ArrayLike<number>,
  parameters?: number,
): Float64Array {
  const { n, p, bread } = fit
  const k = parameters ?? p
  if (kind === 'nonrobust')
    return Float64Array.from(bread, (v) => (v * fit.sigma2 * Math.max(1, n - p)) / Math.max(1, n - k))
  const meat = new Float64Array(p * p)
  const add = (score: Float64Array): void => {
    for (let r = 0; r < p; r++)
      for (let c = 0; c < p; c++) meat[r * p + c] = (meat[r * p + c] ?? 0) + (score[r] ?? 0) * (score[c] ?? 0)
  }
  let groups = 0
  if (kind === 'cluster') {
    if (clusters === undefined) throw new MdmlError('value', 'cluster-robust covariance needs the cluster of each row')
    const sums = new Map<number, Float64Array>()
    for (let i = 0; i < n; i++) {
      const g = clusters[i] ?? 0
      let s = sums.get(g)
      if (s === undefined) {
        s = new Float64Array(p)
        sums.set(g, s)
      }
      const wi = w === undefined ? 1 : (w[i] ?? 0)
      for (let j = 0; j < p; j++) s[j] = (s[j] ?? 0) + wi * (a[i * p + j] ?? 0) * (fit.residuals[i] ?? 0)
    }
    groups = sums.size
    for (const s of sums.values()) add(s)
  } else {
    for (let i = 0; i < n; i++) {
      const wi = w === undefined ? 1 : (w[i] ?? 0)
      const s = new Float64Array(p)
      for (let j = 0; j < p; j++) s[j] = wi * (a[i * p + j] ?? 0) * (fit.residuals[i] ?? 0)
      add(s)
    }
  }
  const scale =
    kind === 'HC0'
      ? 1
      : kind === 'HC1'
        ? n / Math.max(1, n - k)
        : (groups / Math.max(1, groups - 1)) * ((n - 1) / Math.max(1, n - k))
  const out = new Float64Array(p * p)
  for (let r = 0; r < p; r++)
    for (let c = 0; c < p; c++) {
      let s = 0
      for (let k = 0; k < p; k++)
        for (let l = 0; l < p; l++) s += (bread[r * p + k] ?? 0) * (meat[k * p + l] ?? 0) * (bread[l * p + c] ?? 0)
      out[r * p + c] = scale * s
    }
  return out
}

/** The design with a leading column of ones. */
export function addIntercept(a: Float64Array, n: number, d: number): Float64Array {
  const out = new Float64Array(n * (d + 1))
  for (let i = 0; i < n; i++) {
    out[i * (d + 1)] = 1
    for (let j = 0; j < d; j++) out[i * (d + 1) + 1 + j] = a[i * d + j] ?? 0
  }
  return out
}

/** The logit of a probability. */
export const logit = (p: number): number => Math.log(p / (1 - p))
/** The logistic function. */
export const expit = (z: number): number => 1 / (1 + Math.exp(-z))
