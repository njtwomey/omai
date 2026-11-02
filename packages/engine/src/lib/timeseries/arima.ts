/**
 * ARIMA and seasonal ARIMA with regression terms (SARIMAX), statsmodels style: the fit by exact Gaussian maximum
 * likelihood (through the state-space form of the differenced series), by conditional sum of squares, or by the second
 * started from the first, as a traced BFGS `Algorithm` with the log-likelihood at every iteration; the residuals and the
 * forecasts with prediction intervals on the original scale. A fitted model is plain JSON, so it serialises.
 */

import { MdmlError } from '../base/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import type { BfgsState } from '../optim/index.ts'
import { yuleWalker } from '../signal/index.ts'
import { item, toFlat } from '../tensor/index.ts'
import { type Algorithm, type RunOptions, runAlgorithm } from '../trace/index.ts'
import {
  type ArimaSpec,
  constrain,
  cssLogLikelihood,
  differencingOf,
  exactLogLikelihood,
  expandPolynomials,
  type NaturalParameters,
  type Prepared,
  prepare,
  regressionColumns,
  regressionNames,
  regressionResidual,
  resolveSpec,
  type Trend,
} from './_arima-model.ts'
import { inverseDiagonalRoots, leastSquares, likelihoodAlgorithm, numericalHessian } from './_ml.ts'
import { unconstrainStationary } from './_params.ts'
import { asSeries } from './_series.ts'
import { armaForm, filterForm, forecastStateSpace } from './state-space.ts'

export type { ArimaSpec, Trend } from './_arima-model.ts'

/** How an ARIMA is fitted. */
export type ArimaMethod = 'mle' | 'css' | 'css-mle'

/** Options of {@link arimaFit}. */
export interface ArimaOptions extends Partial<Pick<RunOptions, 'key' | 'tracer' | 'policy' | 'parent'>> {
  /** Exact maximum likelihood (default), conditional sum of squares, or CSS followed by exact likelihood. */
  readonly method?: ArimaMethod
  /** The BFGS iteration cap (default 200). */
  readonly maxIterations?: number
  /** The gradient tolerance of the per-observation objective (default 1e-6). */
  readonly gradTol?: number
}

/** A fitted ARIMA(p, d, q)(P, D, Q)_s model with regression terms: plain data, so it serialises as JSON. */
export interface ArimaFit {
  readonly order: readonly [number, number, number]
  readonly seasonalOrder: readonly [number, number, number]
  readonly period: number
  readonly trend: Trend
  /** The names and values of the regression coefficients (trend first, then exogenous regressors). */
  readonly regressionNames: readonly string[]
  readonly regression: readonly number[]
  /** phi_1..phi_p. */
  readonly ar: readonly number[]
  /** theta_1..theta_q. */
  readonly ma: readonly number[]
  /** Phi_1..Phi_P. */
  readonly seasonalAr: readonly number[]
  /** Theta_1..Theta_Q. */
  readonly seasonalMa: readonly number[]
  /** The innovation variance. */
  readonly sigma2: number
  readonly logLikelihood: number
  readonly aic: number
  readonly bic: number
  /** Observations in the likelihood (after differencing). */
  readonly nobs: number
  /** Standard errors of regression, ar, ma, seasonalAr, seasonalMa, sigma2 from the numerical Hessian (in that order). */
  readonly stdErrors: readonly number[]
  readonly method: ArimaMethod
  readonly converged: boolean
  readonly iterations: number
}

const NEG_BIG = 1e10

function naturalOf(fit: ArimaFit): NaturalParameters {
  return {
    beta: [...fit.regression],
    ar: [...fit.ar],
    ma: [...fit.ma],
    sar: [...fit.seasonalAr],
    sma: [...fit.seasonalMa],
  }
}

function specOf(fit: ArimaFit, exog?: ArimaSpec['exog']): ArimaSpec {
  return {
    order: fit.order,
    seasonal: { order: fit.seasonalOrder, period: fit.period },
    trend: fit.trend,
    ...(exog === undefined ? {} : { exog }),
  }
}

function startVector(prep: Prepared): number[] {
  const { rs } = prep
  const beta = leastSquares(prep.columns, prep.z)
  const r = regressionResidual(prep, beta)
  const start = [...beta]
  let ar: number[] = new Array<number>(rs.p).fill(0)
  if (rs.p > 0 && r.length > rs.p + 2) {
    const yw = yuleWalker(r, rs.p, { method: 'mle' })
    const u = unconstrainStationary(toFlat(yw.ar))
    if (u !== undefined) ar = Array.from(u)
  }
  start.push(...ar)
  start.push(...new Array<number>(rs.q + rs.P + rs.Q).fill(0))
  return start
}

/**
 * The objective and bookkeeping shared by the ARIMA fit algorithms: the per-observation negative log-likelihood of the
 * unconstrained parameters under a method.
 */
function objectiveOf(prep: Prepared, method: 'mle' | 'css'): (u: Float64Array) => number {
  const kReg = prep.columns.length
  return (u) => {
    try {
      const nat = constrain(u, prep.rs, kReg)
      const ll = method === 'mle' ? exactLogLikelihood(prep, nat) : cssLogLikelihood(prep, nat)
      return Number.isFinite(ll) ? -ll / prep.z.length : NEG_BIG
    } catch {
      return NEG_BIG
    }
  }
}

function fullLogLikelihood(prep: Prepared, nat: NaturalParameters, sigma2: number): number {
  const { ar, ma } = expandPolynomials(nat.ar, nat.ma, nat.sar, nat.sma, prep.rs.s)
  const f = filterForm(regressionResidual(prep, nat.beta), armaForm(ar, ma))
  return -0.5 * (f.nobs * Math.log(2 * Math.PI * sigma2) + f.sumLogF + f.sumSquares / sigma2)
}

function assemble(
  prep: Prepared,
  state: BfgsState,
  method: ArimaMethod,
  iterations: number,
  converged: boolean,
): ArimaFit {
  const { rs } = prep
  const nat = constrain(state.x, rs, prep.columns.length)
  const { ar, ma } = expandPolynomials(nat.ar, nat.ma, nat.sar, nat.sma, rs.s)
  const f = filterForm(regressionResidual(prep, nat.beta), armaForm(ar, ma))
  const n = f.nobs
  const sigma2 = f.sumSquares / n
  const logLikelihood = -0.5 * (n * (Math.log(2 * Math.PI) + 1 + Math.log(sigma2)) + f.sumLogF)
  const k = prep.columns.length + rs.p + rs.q + rs.P + rs.Q + 1
  const flat = [...nat.beta, ...nat.ar, ...nat.ma, ...nat.sar, ...nat.sma, sigma2]
  const objective = (x: Float64Array): number => {
    try {
      const parts = {
        beta: Array.from(x.subarray(0, nat.beta.length)),
        ar: Array.from(x.subarray(nat.beta.length, nat.beta.length + rs.p)),
        ma: Array.from(x.subarray(nat.beta.length + rs.p, nat.beta.length + rs.p + rs.q)),
        sar: Array.from(x.subarray(nat.beta.length + rs.p + rs.q, nat.beta.length + rs.p + rs.q + rs.P)),
        sma: Array.from(x.subarray(nat.beta.length + rs.p + rs.q + rs.P, k - 1)),
      }
      const v = -fullLogLikelihood(prep, parts, x[k - 1] ?? 1)
      return Number.isFinite(v) ? v : Number.NaN
    } catch {
      return Number.NaN
    }
  }
  const stdErrors = inverseDiagonalRoots(numericalHessian(objective, flat), k)
  return {
    order: [rs.p, rs.d, rs.q],
    seasonalOrder: [rs.P, rs.D, rs.Q],
    period: rs.s,
    trend: rs.trend,
    regressionNames: regressionNames(rs),
    regression: nat.beta,
    ar: nat.ar,
    ma: nat.ma,
    seasonalAr: nat.sar,
    seasonalMa: nat.sma,
    sigma2,
    logLikelihood,
    aic: -2 * logLikelihood + 2 * k,
    bic: -2 * logLikelihood + k * Math.log(n),
    nobs: n,
    stdErrors,
    method,
    converged,
    iterations,
  }
}

/** An ARIMA fit as an algorithm plus the map from its final state to the fitted model. */
export interface ArimaProblem {
  /** BFGS on the negative log-likelihood per observation, one iteration per step. */
  readonly algorithm: Algorithm<BfgsState>
  /** The fitted model for a state of the algorithm. */
  readonly fit: (state: BfgsState, iterations?: number, converged?: boolean) => ArimaFit
}

/**
 * ARIMA(p, d, q)(P, D, Q)_s with regression terms as a traced algorithm: BFGS on the unconstrained parameters (partial
 * autocorrelations keep the AR and MA polynomials stationary and invertible), one iteration per step, with the
 * log-likelihood recorded at each. The series is differenced once and the likelihood is that of the differenced
 * series (the diffuse limit statsmodels approximates with `simple_differencing`). For the method `css-mle` the
 * conditional fit runs first, to completion, and the exact-likelihood algorithm starts from it.
 *
 * @param y - The series.
 * @param spec - The orders, trend and exogenous regressors.
 * @param options - The method, iteration cap and gradient tolerance.
 * @returns The algorithm and the map from its state to the fit.
 * @example
 * ```ts
 * import { arimaFitAlgorithm } from 'mdml-engine/timeseries'
 * const y = [0.3, 0.9, 0.4, -0.2, 0.5, 1.1, 0.6, -0.1, 0.2, 0.8, 0.5, -0.3]
 * arimaFitAlgorithm(y, { order: [1, 0, 0] }).algorithm.name // => 'arima-mle'
 * ```
 */
export function arimaFitAlgorithm(y: ArrayLike<number>, spec: ArimaSpec, options: ArimaOptions = {}): ArimaProblem {
  const prep = prepare(asSeries(Float64Array.from(y), 'arimaFit'), spec)
  const method = options.method ?? 'mle'
  let x0 = startVector(prep)
  let first: 'mle' | 'css' = method === 'css' ? 'css' : 'mle'
  if (method === 'css-mle') {
    first = 'mle'
    const css = likelihoodAlgorithm('arima-css', {}, objectiveOf(prep, 'css'), x0, prep.z.length, options)
    x0 = Array.from(runAlgorithm(css, { maxSteps: options.maxIterations ?? 200 }).state.x)
  }
  const algorithm = likelihoodAlgorithm(
    `arima-${first}`,
    { order: spec.order, seasonal: spec.seasonal ?? null, trend: prep.rs.trend, method },
    objectiveOf(prep, first),
    x0,
    prep.z.length,
    options,
  )
  return {
    algorithm,
    fit: (state, iterations = 0, converged = true) => assemble(prep, state, method, iterations, converged),
  }
}

/**
 * Fit an ARIMA(p, d, q)(P, D, Q)_s model with optional trend and regressors (statsmodels' `ARIMA` and `SARIMAX`).
 *
 * @param y - The series.
 * @param spec - The orders, trend and exogenous regressors.
 * @param options - The method (`mle`, `css` or `css-mle`), iteration cap, gradient tolerance and run options.
 * @returns The fitted model with its likelihood, information criteria and standard errors.
 * @index function Fit ARIMA or seasonal ARIMA by exact Gaussian likelihood.
 * @example
 * ```ts
 * import { arimaFit } from 'mdml-engine/timeseries'
 * const y = [0.3, 0.9, 0.4, -0.2, 0.5, 1.1, 0.6, -0.1, 0.2, 0.8, 0.5, -0.3, 0.1, 0.7, 0.4, 0.0]
 * arimaFit(y, { order: [1, 0, 0], trend: 'n' }).ar.length // => 1
 * ```
 */
export function arimaFit(y: ArrayLike<number>, spec: ArimaSpec, options: ArimaOptions = {}): ArimaFit {
  const { algorithm, fit } = arimaFitAlgorithm(y, spec, options)
  const { state, steps, status } = runAlgorithm(algorithm, {
    maxSteps: options.maxIterations ?? 200,
    ...(options.key === undefined ? {} : { key: options.key }),
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  })
  return fit(state, steps, status === 'converged')
}

/**
 * The one-step prediction errors of a fitted model on the differenced scale (the innovations of its Kalman filter).
 *
 * @param fit - The fitted model.
 * @param y - The series it was fitted to.
 * @param exog - The exogenous regressors it was fitted with.
 * @returns The innovations, `d + D s` fewer than the series, and their variances relative to `sigma2`.
 * @example
 * ```ts
 * import { arimaFit, arimaResiduals } from 'mdml-engine/timeseries'
 * const y = [0.3, 0.9, 0.4, -0.2, 0.5, 1.1, 0.6, -0.1, 0.2, 0.8, 0.5, -0.3, 0.1, 0.7, 0.4, 0.0]
 * arimaResiduals(arimaFit(y, { order: [1, 0, 0] }), y).innovations.length // => 16
 * ```
 */
export function arimaResiduals(
  fit: ArimaFit,
  y: ArrayLike<number>,
  exog?: ArimaSpec['exog'],
): { readonly innovations: Float64Array; readonly relativeVariances: Float64Array } {
  const prep = prepare(Float64Array.from(y), specOf(fit, exog))
  const nat = naturalOf(fit)
  const { ar, ma } = expandPolynomials(nat.ar, nat.ma, nat.sar, nat.sma, prep.rs.s)
  const f = filterForm(regressionResidual(prep, nat.beta), armaForm(ar, ma))
  return { innovations: f.innovations, relativeVariances: f.relativeVariances }
}

/** Forecasts of an ARIMA with prediction intervals. */
export interface ArimaForecast {
  readonly mean: readonly number[]
  /** Standard errors of the forecasts. */
  readonly se: readonly number[]
  /** The central prediction intervals, one per requested level. */
  readonly intervals: readonly {
    readonly level: number
    readonly lower: readonly number[]
    readonly upper: readonly number[]
  }[]
}

/** Options of {@link arimaForecast}. */
export interface ArimaForecastOptions {
  /** Levels of the prediction intervals (default `[0.95]`). */
  readonly levels?: readonly number[]
  /** Exogenous regressors for the observed period followed by the forecast period (`n + steps` rows). */
  readonly exog?: ArimaSpec['exog']
}

/**
 * Forecast `steps` ahead on the original scale with normal prediction intervals. The mean and variance are exact for
 * the fitted model: the state of the differenced series is filtered to the end of the sample and carried forward
 * together with the last observations, which the differencing operator integrates, so the intervals widen with the
 * order of integration as statsmodels' do.
 *
 * @param fit - The fitted model.
 * @param y - The series it was fitted to.
 * @param steps - How many steps ahead.
 * @param options - Interval levels and the exogenous regressors (history then future).
 * @returns Means, standard errors and intervals.
 * @example
 * ```ts
 * import { arimaForecast } from 'mdml-engine/timeseries'
 * const fit = { order: [0, 1, 0], seasonalOrder: [0, 0, 0], period: 1, trend: 'n', regressionNames: [], regression: [],
 *   ar: [], ma: [], seasonalAr: [], seasonalMa: [], sigma2: 4, logLikelihood: 0, aic: 0, bic: 0, nobs: 3, stdErrors: [],
 *   method: 'mle', converged: true, iterations: 0 } as const
 * // A random walk: flat forecasts, standard error 2 sqrt(h).
 * const f = arimaForecast(fit, [1, 2, 5], 3)
 * [f.mean, f.se.map((v) => Math.round(v * 1e6) / 1e6)] // => [[5, 5, 5], [2, 2.828427, 3.464102]]
 * ```
 */
export function arimaForecast(
  fit: ArimaFit,
  y: ArrayLike<number>,
  steps: number,
  options: ArimaForecastOptions = {},
): ArimaForecast {
  if (!(Number.isInteger(steps) && steps >= 1)) throw new MdmlError('value', 'arimaForecast: steps must be at least 1')
  const n = y.length
  const all = options.exog
  const nTrend = fit.trend === 'ct' ? 2 : fit.trend === 'n' ? 0 : 1
  const kExog = fit.regression.length - nTrend
  if (kExog > 0 && (all === undefined || all.length !== n + steps || (all[0]?.length ?? 0) !== kExog))
    throw new MdmlError('shape', `arimaForecast: exog needs ${n + steps} rows of ${kExog} columns (history and future)`)
  const rs = resolveSpec(specOf(fit, all))
  const nat = naturalOf(fit)
  const columns = regressionColumns(n + steps, rs, all)
  const effect = (t: number): number => columns.reduce((acc, c, k) => acc + (nat.beta[k] ?? 0) * (c[t] ?? 0), 0)
  const u = Float64Array.from({ length: n }, (_, t) => (y[t] ?? Number.NaN) - effect(t))
  const poly = differencingOf(rs)
  const kd = poly.length - 1
  const z = u.length > kd ? differenceFrom(u, rs) : new Float64Array(0)
  if (z.length === 0) throw new MdmlError('value', 'arimaForecast: the series is too short')
  const { ar, ma } = expandPolynomials(nat.ar, nat.ma, nat.sar, nat.sma, rs.s)
  const form = armaForm(ar, ma)
  const f = filterForm(z, form)
  const r = form.r
  const m = r + kd
  const A = new Float64Array(m * m)
  const g = new Float64Array(m)
  const h = new Float64Array(m)
  for (let i = 0; i < r; i++) {
    A[i * m] = form.phi[i] ?? 0
    if (i + 1 < r) A[i * m + i + 1] = 1
    g[i] = form.loading[i] ?? 0
  }
  h[0] = 1
  if (kd > 0) {
    A[r * m] = 1
    for (let j = 1; j <= kd; j++) {
      A[r * m + r + j - 1] = -(poly[j] ?? 0)
      h[r + j - 1] = -(poly[j] ?? 0)
    }
    for (let i = 1; i < kd; i++) A[(r + i) * m + r + i - 1] = 1
  }
  const m0 = new Float64Array(m)
  m0.set(f.state)
  for (let j = 0; j < kd; j++) m0[r + j] = u[n - 1 - j] ?? 0
  const P0 = new Float64Array(m * m)
  for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) P0[i * m + j] = f.covariance[i * r + j] ?? 0
  const fc = forecastStateSpace(A, g, h, m0, P0, fit.sigma2, steps)
  const mean = Array.from(fc.mean, (v, k) => v + effect(n + k))
  const se = Array.from(fc.variance, (v) => Math.sqrt(v))
  const levels = options.levels ?? [0.95]
  return {
    mean,
    se,
    intervals: levels.map((level) => {
      const q = item(ndtri((1 + level) / 2))
      return { level, lower: mean.map((v, k) => v - q * (se[k] ?? 0)), upper: mean.map((v, k) => v + q * (se[k] ?? 0)) }
    }),
  }
}

function differenceFrom(u: Float64Array, rs: ReturnType<typeof resolveSpec>): Float64Array {
  const poly = differencingOf(rs)
  const kd = poly.length - 1
  const out = new Float64Array(u.length - kd)
  for (let t = kd; t < u.length; t++) {
    let v = 0
    for (let j = 0; j <= kd; j++) v += (poly[j] ?? 0) * (u[t - j] ?? 0)
    out[t - kd] = v
  }
  return out
}
