/**
 * Vector autoregression, statsmodels' `VAR`: the least-squares fit of y_t = c + A_1 y_(t-1) + ... + A_p y_(t-p) + u_t
 * with its covariance, likelihood and information criteria, the choice of the lag order, forecasts with intervals,
 * the moving-average representation with impulse responses (plain, orthogonalised by the Cholesky factor, cumulative),
 * the forecast-error variance decomposition, the stability of the companion matrix, and Granger-causality tests.
 */

import { MdmlError, type TestResult } from '../base/index.ts'
import { chdtrc, fdtrc, ndtri } from '../numerics/special/index.ts'
import { item } from '../tensor/index.ts'
import {
  add,
  choleskyLower,
  eigenvalueModuli,
  identity,
  invert,
  logDeterminant,
  type MutableRows,
  multiply,
  transpose,
} from './_matrix.ts'
import { leastSquares } from './_ml.ts'

/** The deterministic terms of a VAR: none, a constant, a constant and linear trend (statsmodels' `trend`). */
export type VarTrend = 'n' | 'c' | 'ct'

/** A fitted VAR(p): plain data. */
export interface VarFit {
  readonly lags: number
  readonly nVars: number
  readonly trend: VarTrend
  /** The deterministic coefficients, one row per term (constant, trend) and one column per equation. */
  readonly deterministic: readonly (readonly number[])[]
  /** The lag matrices: `coefficients[i][eq][var]` multiplies y_(t-i-1)[var] in equation eq. */
  readonly coefficients: readonly (readonly (readonly number[])[])[]
  /** The innovation covariance, with the degrees-of-freedom correction T - K p - terms. */
  readonly sigma: readonly (readonly number[])[]
  /** The maximum-likelihood covariance, the residual sum of squares over T. */
  readonly sigmaMle: readonly (readonly number[])[]
  readonly logLikelihood: number
  readonly aic: number
  readonly bic: number
  readonly hqic: number
  readonly fpe: number
  /** Observations used, T - p. */
  readonly nobs: number
  readonly dfResid: number
  /** Standard errors of the coefficients, rows ordered as the regressors (deterministic, lag 1, lag 2, ...), columns by equation. */
  readonly stdErrors: readonly (readonly number[])[]
  /** The inverse of Z'Z for the regressor matrix Z (to rebuild the covariance of the coefficients). */
  readonly zzInverse: readonly (readonly number[])[]
}

/** Options of {@link varFit}. */
export interface VarOptions {
  /** The lag order p (default 1). */
  readonly lags?: number
  /** The deterministic terms (default `c`). */
  readonly trend?: VarTrend
  /** Drop this many leading observations (for comparing orders on one sample). */
  readonly offset?: number
}

const termsOf = (trend: VarTrend): number => (trend === 'n' ? 0 : trend === 'c' ? 1 : 2)

/** Rows of a multivariate series as arrays; accepts rows or a tensor-like `[T, k]`. */
function rowsOfSeries(y: ArrayLike<ArrayLike<number>>, who: string): MutableRows {
  const rows = Array.from(y, (r) => Array.from(r))
  const k = rows[0]?.length ?? 0
  if (k < 1 || rows.some((r) => r.length !== k)) throw new MdmlError('shape', `${who}: expected rows of equal length`)
  return rows
}

/** The Gaussian log-likelihood of a VAR from its residual covariance: -T K / 2 log 2 pi - T / 2 (log |S| + K). */
function varLogLikelihood(sigmaMle: MutableRows, nobs: number): number {
  const k = sigmaMle.length
  return -((nobs * k) / 2) * Math.log(2 * Math.PI) - (nobs / 2) * (logDeterminant(sigmaMle) + k)
}

/**
 * Fit a VAR(p) by least squares equation by equation (Lutkepohl; statsmodels' `VAR(y).fit(p, trend)`).
 *
 * @param y - The series, one row per time and one column per variable.
 * @param options - The lag order, trend and offset.
 * @returns The fitted model with its covariance, likelihood, criteria and standard errors.
 * @throws MdmlError with code `value` when the sample is too short for the model.
 * @example
 * ```ts
 * import { varFit } from 'mdml-engine/timeseries'
 * // y2 follows y1 one step later: the lag-1 coefficient of y1 in y2's equation is 1.
 * const y1 = [0, 1, 0, -1, 2, 0, 1, -2, 1, 0, 2, -1]
 * const y = y1.map((v, t) => [v, t > 0 ? (y1[t - 1] ?? 0) : 0])
 * varFit(y, { lags: 1, trend: 'n' }).coefficients[0][1][0].toFixed(6) // => '1.000000'
 * ```
 */
export function varFit(y: ArrayLike<ArrayLike<number>>, options: VarOptions = {}): VarFit {
  const rows = rowsOfSeries(y, 'varFit').slice(options.offset ?? 0)
  const p = options.lags ?? 1
  const trend = options.trend ?? 'c'
  const k = rows[0]?.length ?? 0
  const terms = termsOf(trend)
  const nobs = rows.length - p
  const regressors = terms + k * p
  if (!(Number.isInteger(p) && p >= 1) || nobs <= regressors)
    throw new MdmlError('value', 'varFit: the sample is too short for the model')
  // Regressors: the constant, the trend (statsmodels counts it from lags + 1), then lag 1 of all variables, lag 2, ...
  const z: MutableRows = Array.from({ length: nobs }, (_, t) => [
    ...(terms >= 1 ? [1] : []),
    ...(terms >= 2 ? [t + 1 + p] : []),
    ...Array.from({ length: p }, (_, lag) => rows[t + p - 1 - lag] ?? []).flat(),
  ])
  const columns = Array.from({ length: regressors }, (_, j) => z.map((row) => row[j] ?? 0))
  const params: MutableRows = Array.from({ length: regressors }, () => new Array<number>(k).fill(0))
  const resid: MutableRows = rows.slice(p).map((r) => [...r])
  for (let eq = 0; eq < k; eq++) {
    const target = rows.slice(p).map((r) => r[eq] ?? 0)
    const beta = leastSquares(columns, target)
    beta.forEach((b, j) => {
      ;(params[j] as number[])[eq] = b
    })
    for (let t = 0; t < nobs; t++) {
      let fit = 0
      for (let j = 0; j < regressors; j++) fit += (beta[j] ?? 0) * (z[t]?.[j] ?? 0)
      ;(resid[t] as number[])[eq] = (target[t] ?? 0) - fit
    }
  }
  const sse = multiply(transpose(resid), resid)
  const dfResid = nobs - regressors
  const sigma = sse.map((r) => r.map((v) => v / dfResid))
  const sigmaMle = sigma.map((r) => r.map((v) => (v * dfResid) / nobs))
  const zz = invert(multiply(transpose(z), z))
  const stdErrors = zz.map((r, j) => sigma.map((_, eq) => Math.sqrt((r[j] ?? 0) * (sigma[eq]?.[eq] ?? 0))))
  const llf = varLogLikelihood(sigmaMle, nobs)
  const ld = logDeterminant(sigmaMle)
  const free = p * k * k + k * terms
  return {
    lags: p,
    nVars: k,
    trend,
    deterministic: params.slice(0, terms),
    coefficients: Array.from({ length: p }, (_, i) =>
      Array.from({ length: k }, (_, eq) => Array.from({ length: k }, (_, v) => params[terms + i * k + v]?.[eq] ?? 0)),
    ),
    sigma,
    sigmaMle,
    logLikelihood: llf,
    aic: ld + (2 / nobs) * free,
    bic: ld + (Math.log(nobs) / nobs) * free,
    hqic: ld + ((2 * Math.log(Math.log(nobs))) / nobs) * free,
    fpe: ((nobs + regressors) / dfResid) ** k * Math.exp(ld),
    nobs,
    dfResid,
    stdErrors,
    zzInverse: zz,
  }
}

/** The criteria of a lag-order search. */
export interface VarOrderSelection {
  /** The criteria for each order 0..maxLags (order 0 has only the deterministic terms). */
  readonly criteria: readonly {
    readonly lags: number
    readonly aic: number
    readonly bic: number
    readonly hqic: number
    readonly fpe: number
  }[]
  /** The order minimising each criterion. */
  readonly selected: { readonly aic: number; readonly bic: number; readonly hqic: number; readonly fpe: number }
}

/**
 * Choose the lag order by information criteria (statsmodels' `select_order`): every order from 1 to `maxLags` is fitted
 * to the same sample (the last T - maxLags observations).
 *
 * @param y - The series.
 * @param maxLags - The largest order (default 12 (T / 100)^(1/4), rounded).
 * @param trend - The deterministic terms (default `c`).
 * @returns The criteria and the selected orders.
 * @example
 * ```ts
 * import { varSelectOrder } from 'mdml-engine/timeseries'
 * const y = Array.from({ length: 60 }, (_, t) => [Math.sin(t / 3), Math.cos(t / 5)])
 * varSelectOrder(y, 3).criteria.length // => 3
 * ```
 */
export function varSelectOrder(
  y: ArrayLike<ArrayLike<number>>,
  maxLags?: number,
  trend: VarTrend = 'c',
): VarOrderSelection {
  const rows = rowsOfSeries(y, 'varSelectOrder')
  const total = rows.length
  const top = maxLags ?? Math.round(12 * (total / 100) ** 0.25)
  const criteria = Array.from({ length: top }, (_, i) => {
    const lags = i + 1
    const f = varFit(rows, { lags, trend, offset: top - lags })
    return { lags, aic: f.aic, bic: f.bic, hqic: f.hqic, fpe: f.fpe }
  })
  const best = (key: 'aic' | 'bic' | 'hqic' | 'fpe'): number =>
    criteria.reduce((b, c) => (c[key] < b[key] ? c : b), criteria[0] as (typeof criteria)[number]).lags
  return { criteria, selected: { aic: best('aic'), bic: best('bic'), hqic: best('hqic'), fpe: best('fpe') } }
}

/** The moving-average matrices Phi_0 = I, Phi_i = sum_j A_j Phi_(i-j), for i = 0..steps. */
function maMatrices(fit: Pick<VarFit, 'coefficients' | 'nVars'>, steps: number): MutableRows[] {
  const phis: MutableRows[] = [identity(fit.nVars)]
  for (let i = 1; i <= steps; i++) {
    let acc: MutableRows = fit.coefficients[0]?.map((r) => r.map(() => 0)) ?? []
    for (let j = 1; j <= Math.min(i, fit.coefficients.length); j++)
      acc = add(acc, multiply(fit.coefficients[j - 1] as MutableRows, phis[i - j] as MutableRows))
    phis.push(acc)
  }
  return phis
}

/**
 * The impulse responses of a VAR: the response of each variable to a unit shock in each, at horizons 0..steps
 * (statsmodels' `irf`): the MA matrices Phi_h, or Phi_h P with P the Cholesky factor of the covariance for orthogonal
 * shocks, optionally accumulated over horizons.
 *
 * @param fit - The fitted model.
 * @param steps - The largest horizon.
 * @param options - `orthogonal` (default false) and `cumulative` (default false).
 * @returns `[steps + 1][response variable][shocked variable]`.
 * @example
 * ```ts
 * import { varImpulseResponse } from 'mdml-engine/timeseries'
 * // A scalar AR(1) with phi = 0.5: the response decays geometrically.
 * const fit = { coefficients: [[[0.5]]], nVars: 1, sigma: [[1]] }
 * varImpulseResponse(fit, 3).map((m) => m[0]?.[0]) // => [1, 0.5, 0.25, 0.125]
 * ```
 */
export function varImpulseResponse(
  fit: Pick<VarFit, 'coefficients' | 'nVars' | 'sigma'>,
  steps: number,
  options: { readonly orthogonal?: boolean; readonly cumulative?: boolean } = {},
): MutableRows[] {
  let phis = maMatrices(fit, steps)
  if (options.orthogonal === true) {
    const P = choleskyLower(fit.sigma)
    phis = phis.map((m) => multiply(m, P))
  }
  if (options.cumulative === true) {
    let acc = phis[0]?.map((r) => r.map(() => 0)) ?? []
    phis = phis.map((m) => {
      acc = add(acc, m)
      return acc
    })
  }
  return phis
}

/**
 * The forecast-error variance decomposition (statsmodels' `fevd`): the share of each shock in the variance of the
 * h-step forecast error of each variable, from the orthogonalised responses.
 *
 * @param fit - The fitted model.
 * @param periods - The number of horizons (1..periods).
 * @returns `[variable][horizon][shock]`, each row summing to one.
 * @example
 * ```ts
 * import { varForecastErrorVarianceDecomposition } from 'mdml-engine/timeseries'
 * const fit = { coefficients: [[[0.5, 0], [0, 0.5]]], nVars: 2, sigma: [[1, 0], [0, 4]] }
 * varForecastErrorVarianceDecomposition(fit, 2)[1]?.[0] // => [0, 1]
 * ```
 */
export function varForecastErrorVarianceDecomposition(
  fit: Pick<VarFit, 'coefficients' | 'nVars' | 'sigma'>,
  periods: number,
): number[][][] {
  const orth = varImpulseResponse(fit, periods - 1, { orthogonal: true })
  const k = fit.nVars
  const cumulative: number[][][] = []
  let acc = Array.from({ length: k }, () => new Array<number>(k).fill(0))
  for (const m of orth) {
    acc = acc.map((row, i) => row.map((v, j) => v + (m[i]?.[j] ?? 0) ** 2))
    cumulative.push(acc.map((r) => [...r]))
  }
  return Array.from({ length: k }, (_, i) =>
    cumulative.map((c) => {
      const total = (c[i] ?? []).reduce((a, b) => a + b, 0)
      return (c[i] ?? []).map((v) => v / total)
    }),
  )
}

/** Forecasts of a VAR with normal intervals. */
export interface VarForecast {
  /** `[steps][variable]`. */
  readonly mean: MutableRows
  readonly lower: MutableRows
  readonly upper: MutableRows
  /** The forecast error covariance at each horizon, `[steps][k][k]` (parameter uncertainty ignored, as statsmodels' `mse`). */
  readonly covariance: MutableRows[]
}

/**
 * Forecast `steps` ahead (statsmodels' `forecast_interval`): the iterated conditional mean from the last p
 * observations, and intervals from the forecast error covariance sum_(i<h) Phi_i Sigma Phi_i'.
 *
 * @param fit - The fitted model.
 * @param y - The series (its last `lags` rows start the forecast).
 * @param steps - How many steps ahead.
 * @param alpha - The significance level of the intervals (default 0.05).
 * @returns Means, bounds and covariances.
 * @example
 * ```ts
 * import { varForecast } from 'mdml-engine/timeseries'
 * const fit = { lags: 1, nVars: 1, trend: 'n', deterministic: [], coefficients: [[[0.5]]], sigma: [[1]] } as never
 * varForecast(fit, [[4]], 2).mean // => [[2], [1]]
 * ```
 */
export function varForecast(
  fit: Pick<VarFit, 'lags' | 'nVars' | 'trend' | 'deterministic' | 'coefficients' | 'sigma'>,
  y: ArrayLike<ArrayLike<number>>,
  steps: number,
  alpha = 0.05,
): VarForecast {
  const rows = rowsOfSeries(y, 'varForecast')
  const p = fit.lags
  const k = fit.nVars
  if (rows.length < p) throw new MdmlError('value', 'varForecast: need at least `lags` observations')
  const history: MutableRows = rows.slice(rows.length - p)
  const total = rows.length
  const mean: MutableRows = []
  for (let h = 1; h <= steps; h++) {
    const point = new Array<number>(k).fill(0)
    const terms = fit.deterministic
    for (let eq = 0; eq < k; eq++) {
      let v = 0
      if (fit.trend !== 'n') v += terms[0]?.[eq] ?? 0
      // statsmodels counts the trend from lags + 1 within the estimation sample, as the fit does.
      if (fit.trend === 'ct') v += (terms[1]?.[eq] ?? 0) * (total - p + h + p)
      for (let i = 1; i <= p; i++) {
        const prior = h - i <= 0 ? (history[history.length + h - i - 1] ?? []) : (mean[h - i - 1] ?? [])
        for (let j = 0; j < k; j++) v += (fit.coefficients[i - 1]?.[eq]?.[j] ?? 0) * (prior[j] ?? 0)
      }
      point[eq] = v
    }
    mean.push(point)
  }
  const phis = maMatrices(fit, steps)
  const covariance: MutableRows[] = []
  let prior: MutableRows = identity(k).map((r) => r.map(() => 0))
  for (let h = 0; h < steps; h++) {
    const phi = phis[h] as MutableRows
    prior = add(prior, multiply(multiply(phi, fit.sigma as MutableRows), transpose(phi)))
    covariance.push(prior)
  }
  const q = item(ndtri(1 - alpha / 2))
  const sd = covariance.map((c) => c.map((r, i) => Math.sqrt(r[i] ?? 0)))
  return {
    mean,
    lower: mean.map((m, h) => m.map((v, i) => v - q * (sd[h]?.[i] ?? 0))),
    upper: mean.map((m, h) => m.map((v, i) => v + q * (sd[h]?.[i] ?? 0))),
    covariance,
  }
}

/**
 * Whether the VAR is stable: every eigenvalue of the companion matrix lies inside the unit circle.
 *
 * @param fit - The fitted model.
 * @returns The moduli of the eigenvalues of the companion matrix (largest first) and the verdict.
 * @example
 * ```ts
 * import { varStability } from 'mdml-engine/timeseries'
 * varStability({ coefficients: [[[0.5]]], nVars: 1 }).stable // => true
 * ```
 */
export function varStability(fit: Pick<VarFit, 'coefficients' | 'nVars'>): {
  readonly moduli: number[]
  readonly stable: boolean
} {
  const k = fit.nVars
  const p = fit.coefficients.length
  const n = k * p
  const companion: MutableRows = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < p; i++)
    for (let r = 0; r < k; r++)
      for (let c = 0; c < k; c++) (companion[r] as number[])[i * k + c] = fit.coefficients[i]?.[r]?.[c] ?? 0
  for (let r = k; r < n; r++) (companion[r] as number[])[r - k] = 1
  const moduli = eigenvalueModuli(companion).sort((a, b) => b - a)
  return { moduli, stable: moduli.every((m) => m < 1) }
}

/**
 * Test whether some variables do not Granger-cause others (statsmodels' `test_causality`): a Wald test that all lag
 * coefficients of the `causing` variables in the equations of the `caused` ones are zero. The F form divides by the
 * number of restrictions and uses the system's degrees of freedom k (T - K p - terms).
 *
 * @param fit - The fitted model.
 * @param caused - The indices of the tested equations.
 * @param causing - The indices of the lagged variables whose coefficients are tested.
 * @param kind - `f` (default) or `wald` (chi-square).
 * @returns The test result.
 * @example
 * ```ts
 * import { varFit, grangerCausality } from 'mdml-engine/timeseries'
 * const y1 = [0, 1, 0, -1, 2, 0, 1, -2, 1, 0, 2, -1, 0, 1, -1, 2]
 * const y = y1.map((v, t) => [v + 0.1 * Math.sin(t * 7), (t > 0 ? (y1[t - 1] ?? 0) : 0) + 0.3 * Math.cos(t * 5)])
 * grangerCausality(varFit(y, { lags: 1 }), [1], [0]).pValue < 0.05 // => true
 * ```
 */
export function grangerCausality(
  fit: VarFit,
  caused: readonly number[],
  causing: readonly number[],
  kind: 'f' | 'wald' = 'f',
): TestResult {
  const k = fit.nVars
  const p = fit.lags
  const terms = termsOf(fit.trend)
  const cols = terms * k + k * k * p
  // Covariance of vec(params'): the Kronecker product of (Z'Z)^-1 and Sigma, regressor-major with the equation inner.
  const cov = (a: number, b: number): number => {
    const ra = Math.floor(a / k)
    const rb = Math.floor(b / k)
    return (fit.zzInverse[ra]?.[rb] ?? 0) * (fit.sigma[a % k]?.[b % k] ?? 0)
  }
  const indices: number[] = []
  const stacked: number[] = []
  for (let j = 0; j < p; j++)
    for (const ing of causing)
      for (const ed of caused) {
        const col = terms * k + ed + k * ing + k * k * j
        indices.push(col)
        stacked.push(fit.coefficients[j]?.[ed]?.[ing] ?? 0)
      }
  if (indices.some((c) => c >= cols)) throw new MdmlError('value', 'grangerCausality: index out of range')
  const middle = invert(indices.map((a) => indices.map((b) => cov(a, b))))
  let wald = 0
  stacked.forEach((u, i) => {
    stacked.forEach((v, j) => {
      wald += u * (middle[i]?.[j] ?? 0) * v
    })
  })
  const restrictions = indices.length
  if (kind === 'wald')
    return {
      test: 'Granger causality (Wald)',
      statistic: wald,
      pValue: item(chdtrc(restrictions, wald)),
      alternative: 'greater',
      df: restrictions,
    }
  const f = wald / restrictions
  const dfDen = k * fit.dfResid
  return {
    test: 'Granger causality (F)',
    statistic: f,
    pValue: item(fdtrc(restrictions, dfDen, f)),
    alternative: 'greater',
    df: [restrictions, dfDen],
  }
}
