/**
 * Statistical inference for linear models, statsmodels style: Wald tables of standard errors, t or z statistics,
 * p-values and confidence intervals, the covariance of GLM estimates, and `olsSummary`, an ordinary or weighted
 * least-squares fit with classical or heteroscedasticity-consistent (HC0 to HC3) standard errors, R^2, the F test and
 * information criteria.
 *
 * @index function Wald tables, covariances and confidence intervals for linear models.
 */

import { OmaiError } from '../base/index.ts'
import { type CoefficientTable, coefficientTable, type Explained } from '../exposition/index.ts'
import { featureNamesOf, type LearnData } from '../learn/index.ts'
import { inv, pinv } from '../numerics/linalg/index.ts'
import { fdtrc, ndtr, ndtri, stdtr, stdtrit } from '../numerics/special/index.ts'
import { primalOf, toFlat } from '../tensor/index.ts'
import { type Design, designOf, matrixTensor, withIntercept } from './design.ts'
import { leastSquaresSolve } from './solvers.ts'

/** Heteroscedasticity-consistent covariance types (statsmodels' `cov_type`). */
export type CovarianceType = 'nonrobust' | 'HC0' | 'HC1' | 'HC2' | 'HC3'

const scalar = (v: Parameters<typeof toFlat>[0]): number => toFlat(primalOf(v))[0] ?? Number.NaN

/** (A^T W A)^{-1}, by a pseudo-inverse when singular. */
function bread(a: Float64Array, n: number, p: number, w: ArrayLike<number>): Float64Array {
  const g = new Float64Array(p * p)
  for (let i = 0; i < n; i++) {
    const wi = w[i] ?? 0
    for (let r = 0; r < p; r++) {
      const ar = (a[i * p + r] ?? 0) * wi
      for (let c = 0; c < p; c++) g[r * p + c] = (g[r * p + c] ?? 0) + ar * (a[i * p + c] ?? 0)
    }
  }
  const t = matrixTensor(g, p, p)
  try {
    const out = Float64Array.from(toFlat(primalOf(inv(t))))
    if (out.every(Number.isFinite)) return out
  } catch {
    // fall through to the pseudo-inverse
  }
  return Float64Array.from(toFlat(primalOf(pinv(t))))
}

/**
 * A Wald table from estimates and their covariance: standard errors, t (with `df`) or z statistics, two-sided p-values
 * and intervals at `level`.
 *
 * @param names - Coefficient names.
 * @param estimates - Estimates.
 * @param covariance - Their covariance, row-major `[p, p]`.
 * @param options - Residual degrees of freedom (t statistics; omit for z) and the interval level (default 0.95).
 * @returns The coefficient table.
 */
export function waldTable(
  names: readonly string[],
  estimates: readonly number[],
  covariance: ArrayLike<number>,
  options: { readonly df?: number; readonly level?: number } = {},
): CoefficientTable {
  const p = names.length
  const level = options.level ?? 0.95
  const df = options.df
  const crit = df === undefined ? scalar(ndtri(0.5 + level / 2)) : scalar(stdtrit(df, 0.5 + level / 2))
  const se = Array.from({ length: p }, (_, j) => Math.sqrt(Math.max(covariance[j * p + j] ?? 0, 0)))
  const pValues = estimates.map((b, j) => {
    const t = Math.abs(b / (se[j] ?? Number.NaN))
    return 2 * (1 - (df === undefined ? scalar(ndtr(t)) : scalar(stdtr(df, t))))
  })
  return coefficientTable({
    names,
    estimates,
    standardErrors: se,
    intervals: estimates.map((b, j): [number, number] => [b - crit * (se[j] ?? 0), b + crit * (se[j] ?? 0)]),
    pValues,
    level,
  })
}

/**
 * The covariance of GLM estimates, phi (X^T W X)^{-1} with the working weights W at the solution (the intercept last).
 *
 * @param design - Features and the prior weights.
 * @param workingWeights - The IRLS weights at the solution (the prior weights for least squares).
 * @param scale - The dispersion phi.
 * @param fitIntercept - Whether the model has an intercept.
 * @returns The covariance, row-major.
 */
export function glmCovariance(
  design: Design,
  workingWeights: ArrayLike<number>,
  scale: number,
  fitIntercept: boolean,
): Float64Array {
  const { a, p } = withIntercept(design, fitIntercept)
  return Float64Array.from(bread(a, design.x.n, p, workingWeights), (v) => v * scale)
}

/** Options of {@link olsSummary}. */
export interface OlsSummaryOptions {
  readonly fitIntercept?: boolean
  readonly covType?: CovarianceType
  /** Interval level (default 0.95). */
  readonly level?: number
}

/** A statsmodels-style summary of an ordinary (or weighted) least-squares fit. */
export interface OlsSummary {
  /** Coefficients with standard errors, t statistics' p-values and intervals; the intercept is `const`, last. */
  readonly table: CoefficientTable
  readonly rSquared: number
  readonly adjRSquared: number
  readonly fStatistic: number
  readonly fPValue: number
  /** The residual standard error. */
  readonly sigma: number
  readonly dfResid: number
  readonly logLikelihood: number
  readonly aic: number
  readonly bic: number
  readonly residuals: readonly number[]
  readonly fitted: readonly number[]
  /** Diagonal of the hat matrix. */
  readonly leverage: readonly number[]
  readonly cooksDistance: readonly number[]
  /** Internally studentised residuals. */
  readonly studentised: readonly number[]
  /** Variance inflation factor of each feature. */
  readonly vif: readonly number[]
}

/**
 * Ordinary or weighted least squares with statsmodels' summary: classical or heteroscedasticity-consistent (HC0 to
 * HC3) standard errors, t tests and intervals, R^2 and the overall F test, the Gaussian log-likelihood with AIC and
 * BIC, leverage, Cook's distances, studentised residuals and variance inflation factors.
 *
 * @param data - Features, targets and optional weights.
 * @param options - Intercept (default true), covariance type (default nonrobust) and interval level.
 * @returns The summary, with its coefficient table as exposition.
 * @throws OmaiError `value` when there are no residual degrees of freedom.
 */
export function olsSummary(data: LearnData, options: OlsSummaryOptions = {}): Explained<OlsSummary> {
  const fitIntercept = options.fitIntercept ?? true
  const covType = options.covType ?? 'nonrobust'
  const design = designOf(data, 'olsSummary')
  const { n, d } = design.x
  const { a, p } = withIntercept(design, fitIntercept)
  const dfResid = n - p
  if (dfResid <= 0) throw new OmaiError('value', 'olsSummary: needs more samples than coefficients')
  const fit = leastSquaresSolve(design, 0, fitIntercept, 'svd')
  const beta = [...fit.coef, ...(fitIntercept ? [fit.intercept] : [])]
  const w = design.w
  const fitted = Array.from({ length: n }, (_, i) => {
    let s = 0
    for (let j = 0; j < p; j++) s += (a[i * p + j] ?? 0) * (beta[j] ?? 0)
    return s
  })
  const resid = fitted.map((f, i) => (design.y[i] ?? 0) - f)
  let ssr = 0
  let sw = 0
  let ybar = 0
  for (let i = 0; i < n; i++) {
    ssr += (w[i] ?? 0) * (resid[i] ?? 0) ** 2
    sw += w[i] ?? 0
    ybar += (w[i] ?? 0) * (design.y[i] ?? 0)
  }
  ybar /= sw
  let sst = 0
  for (let i = 0; i < n; i++) sst += (w[i] ?? 0) * ((design.y[i] ?? 0) - (fitIntercept ? ybar : 0)) ** 2
  const sigma2 = ssr / dfResid
  const xtxInv = bread(a, n, p, w)
  const leverage = Array.from({ length: n }, (_, i) => {
    let h = 0
    for (let r = 0; r < p; r++) {
      let t = 0
      for (let c = 0; c < p; c++) t += (xtxInv[r * p + c] ?? 0) * (a[i * p + c] ?? 0)
      h += (a[i * p + r] ?? 0) * t
    }
    return h * (w[i] ?? 0)
  })
  let cov: Float64Array
  if (covType === 'nonrobust') cov = Float64Array.from(xtxInv, (v) => v * sigma2)
  else {
    const omega = resid.map((e, i) => {
      const e2 = (w[i] ?? 0) * e * e
      const h = leverage[i] ?? 0
      if (covType === 'HC0') return e2
      if (covType === 'HC1') return (e2 * n) / dfResid
      if (covType === 'HC2') return e2 / (1 - h)
      return e2 / (1 - h) ** 2
    })
    const meat = new Float64Array(p * p)
    for (let i = 0; i < n; i++)
      for (let r = 0; r < p; r++)
        for (let c = 0; c < p; c++)
          meat[r * p + c] =
            (meat[r * p + c] ?? 0) + (omega[i] ?? 0) * (w[i] ?? 0) * (a[i * p + r] ?? 0) * (a[i * p + c] ?? 0)
    cov = sandwich(xtxInv, meat, p)
  }
  const names = [...featureNamesOf(data, d), ...(fitIntercept ? ['const'] : [])]
  const table = waldTable(names, beta, cov, {
    df: dfResid,
    ...(options.level === undefined ? {} : { level: options.level }),
  })
  const dfModel = p - (fitIntercept ? 1 : 0)
  const rSquared = 1 - ssr / sst
  const fStatistic = (sst - ssr) / dfModel / sigma2
  let logw = 0
  for (let i = 0; i < n; i++) logw += Math.log(w[i] ?? 1)
  const logLikelihood = -(n / 2) * Math.log((2 * Math.PI * ssr) / n) - n / 2 + logw / 2
  const studentised = resid.map((e, i) => (Math.sqrt(w[i] ?? 0) * e) / Math.sqrt(sigma2 * (1 - (leverage[i] ?? 0))))
  return {
    table,
    rSquared,
    adjRSquared: 1 - ((1 - rSquared) * (n - (fitIntercept ? 1 : 0))) / dfResid,
    fStatistic,
    fPValue: scalar(fdtrc(dfModel, dfResid, fStatistic)),
    sigma: Math.sqrt(sigma2),
    dfResid,
    logLikelihood,
    aic: -2 * logLikelihood + 2 * p,
    bic: -2 * logLikelihood + Math.log(n) * p,
    residuals: resid,
    fitted,
    leverage,
    cooksDistance: studentised.map((r, i) => ((r * r) / p) * ((leverage[i] ?? 0) / (1 - (leverage[i] ?? 0)))),
    studentised,
    vif: vifOf(design),
    exposition: [table],
  }
}

function sandwich(b: Float64Array, m: Float64Array, p: number): Float64Array {
  const t = new Float64Array(p * p)
  for (let r = 0; r < p; r++)
    for (let c = 0; c < p; c++) {
      let s = 0
      for (let k = 0; k < p; k++) s += (b[r * p + k] ?? 0) * (m[k * p + c] ?? 0)
      t[r * p + c] = s
    }
  const out = new Float64Array(p * p)
  for (let r = 0; r < p; r++)
    for (let c = 0; c < p; c++) {
      let s = 0
      for (let k = 0; k < p; k++) s += (t[r * p + k] ?? 0) * (b[k * p + c] ?? 0)
      out[r * p + c] = s
    }
  return out
}

/** Variance inflation factors: 1 / (1 - R_j^2) regressing each feature on the others (with an intercept). */
function vifOf(design: Design): number[] {
  const { n, d } = design.x
  return Array.from({ length: d }, (_, j) => {
    if (d === 1) return 1
    const others = new Float64Array(n * (d - 1))
    const target = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let k = 0
      for (let c = 0; c < d; c++) {
        if (c === j) target[i] = design.x.data[i * d + c] ?? 0
        else others[i * (d - 1) + k++] = design.x.data[i * d + c] ?? 0
      }
    }
    const sub: Design = { x: { n, d: d - 1, data: others }, y: target, w: design.w }
    const f = leastSquaresSolve(sub, 0, true, 'svd')
    let ssr = 0
    let mean = 0
    for (let i = 0; i < n; i++) mean += (target[i] ?? 0) / n
    let sst = 0
    for (let i = 0; i < n; i++) {
      let pred = f.intercept
      for (let c = 0; c < d - 1; c++) pred += (others[i * (d - 1) + c] ?? 0) * (f.coef[c] ?? 0)
      ssr += ((target[i] ?? 0) - pred) ** 2
      sst += ((target[i] ?? 0) - mean) ** 2
    }
    return sst / ssr
  })
}
