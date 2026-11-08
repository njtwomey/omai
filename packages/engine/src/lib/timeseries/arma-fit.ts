/**
 * Estimators of AR, MA and ARMA models: Yule-Walker, Burg and least squares for autoregressions, the innovations
 * estimator of moving averages, the Hannan-Rissanen regression for ARMA, and the likelihood fits (conditional sum of
 * squares, exact maximum likelihood through the state-space form) that run as traced algorithms with the objective at
 * each iteration. Every fit is plain data, so it serialises.
 *
 * @index function AR, MA and ARMA estimators: Yule-Walker, Burg, innovations.
 */

import { meanOf, minOf, OmaiError } from '../base/index.ts'
import { burg, yuleWalker } from '../signal/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { leastSquares } from './_ml.ts'
import { asSeries } from './_series.ts'
import { autocovariance } from './acf.ts'
import { type ArimaOptions, type ArimaProblem, arimaFit, arimaFitAlgorithm } from './arima.ts'
import { innovations, isInvertible, isStationary } from './arma.ts'

/** A fitted AR, MA or ARMA model: x_t - mean = sum phi_j (x_(t-j) - mean) + e_t + sum theta_j e_(t-j). */
export interface ArmaEstimate {
  readonly ar: readonly number[]
  readonly ma: readonly number[]
  /** The innovation variance. */
  readonly sigma2: number
  /** The process mean (0 when the series was not demeaned). */
  readonly mean: number
  /** The log-likelihood, for the likelihood-based methods. */
  readonly logLikelihood?: number
  readonly method: string
}

/** The estimators of an autoregression. */
export type ArMethod = 'yule-walker' | 'yule-walker-mle' | 'burg' | 'ols'

/** Options of {@link fitAr}. */
export interface FitArOptions {
  /** The estimator (default `yule-walker`, with the adjusted autocovariance; `yule-walker-mle` divides by n). */
  readonly method?: ArMethod
  /** Remove the sample mean first (default true); for `ols` this decides whether a constant is fitted. */
  readonly demean?: boolean
}

/**
 * Fit an AR(p) by Yule-Walker, Burg's recursion or least squares (statsmodels' `yule_walker`, `burg` and `AutoReg`). The
 * least-squares fit conditions on the first p observations and fits a constant, from which the mean follows.
 *
 * @param y - The series.
 * @param order - The order p.
 * @param options - The method and whether to demean.
 * @returns The coefficients, innovation variance and mean.
 * @example
 * ```ts
 * import { fitAr } from 'omai-engine/timeseries'
 * // An exact AR(1) with phi = 0.5 and no noise: least squares recovers it.
 * const y = [8, 4, 2, 1, 0.5, 0.25]
 * fitAr(y, 1, { method: 'ols', demean: false }).ar // => [0.5]
 * ```
 */
export function fitAr(y: ArrayLike<number>, order: number, options: FitArOptions = {}): ArmaEstimate {
  const x = asSeries(Float64Array.from(y), 'fitAr')
  const method = options.method ?? 'yule-walker'
  const demean = options.demean ?? true
  if (method === 'ols') return fitArLeastSquares(x, order, demean)
  const centre = demean ? meanOf(x) : 0
  const model =
    method === 'burg'
      ? burg(x, order, demean)
      : yuleWalker(x, order, { method: method === 'yule-walker' ? 'adjusted' : 'mle', demean })
  return { ar: toFlat(model.ar), ma: [], sigma2: toFlat(model.sigma2)[0] ?? Number.NaN, mean: centre, method }
}

function fitArLeastSquares(x: Float64Array, p: number, constant: boolean): ArmaEstimate {
  const n = x.length
  const m = n - p
  if (m <= p + 1) throw new OmaiError('value', 'fitAr: the series is too short for the order')
  const columns: number[][] = constant ? [new Array<number>(m).fill(1)] : []
  for (let j = 1; j <= p; j++) columns.push(Array.from({ length: m }, (_, t) => x[p + t - j] ?? 0))
  const target = Array.from({ length: m }, (_, t) => x[p + t] ?? 0)
  const beta = leastSquares(columns, target)
  let ssr = 0
  for (let t = 0; t < m; t++) {
    let fit = 0
    columns.forEach((c, k) => {
      fit += (beta[k] ?? 0) * (c[t] ?? 0)
    })
    ssr += ((target[t] ?? 0) - fit) ** 2
  }
  const ar = constant ? beta.slice(1) : beta
  const intercept = constant ? (beta[0] ?? 0) : 0
  const mean = constant ? intercept / (1 - ar.reduce((a, b) => a + b, 0)) : 0
  return { ar, ma: [], sigma2: ssr / m, mean, method: 'ols' }
}

/** An information criterion for choosing an order: -2 loglik + penalty k, the penalty a function of the sample size. */
export type OrderCriterion = 'aic' | 'bic' | 'hqic'

/**
 * The penalty per parameter of each information criterion, as a function of the sample size: 2 for the AIC, ln m for
 * the BIC and 2 ln ln m for the Hannan-Quinn.
 *
 * @param criterion - `aic`, `bic` or `hqic`.
 * @param m - The sample size.
 * @returns The penalty.
 * @example
 * ```ts
 * import { criterionPenalty } from 'omai-engine/timeseries'
 * criterionPenalty('aic', 100) // => 2
 * ```
 */
export function criterionPenalty(criterion: OrderCriterion, m: number): number {
  return criterion === 'aic' ? 2 : criterion === 'bic' ? Math.log(m) : 2 * Math.log(Math.log(m))
}

/** The state of {@link arOrderAlgorithm}: the orders tried and their criterion values. */
export interface ArOrderState {
  /** The next order to try (the orders 0..p-1 are done). */
  readonly p: number
  /** The criterion at orders 0..p-1. */
  readonly scores: readonly number[]
  /** The order with the smallest criterion so far (-1 before the first). */
  readonly best: number
}

/** The paths {@link arOrderAlgorithm} logs, as a type. */
export type ArOrderPath = 'select/order' | 'select/criterion' | 'select/sigma2' | 'select/best'

/**
 * What {@link arOrderAlgorithm} logs, by path: the order tried, its criterion, the residual variance of its fit and
 * the best order so far.
 */
export const arOrderContract: Contract<ArOrderState, ArOrderPath> = defineContract({
  'select/order': scalar({ description: 'the order tried at this step' }),
  'select/criterion': scalar({ description: 'the information criterion at that order', direction: 'lower' }),
  'select/sigma2': scalar({ description: 'the residual variance of the least-squares fit at that order' }),
  'select/best': scalar({ description: 'the order with the smallest criterion so far' }),
})

/**
 * The search for an AR order as a steppable algorithm: each step fits one order p by least squares on the common
 * sample and scores it by an information criterion; the state holds the scores so far and the best order.
 *
 * @param y - The series.
 * @param maxOrder - The largest order tried.
 * @param criterion - `aic` (default), `bic` or `hqic`.
 * @returns The algorithm; it is done after order `maxOrder`.
 * @example
 * ```ts
 * import { arOrderAlgorithm } from 'omai-engine/timeseries'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const y = [1, 1.3, 0.2, -0.6, 0.1, 0.7, 0.2, -0.3, -0.1, 0.2, 0.1, -0.1, 0, 0.1, 0.02, -0.04]
 * runAlgorithm(arOrderAlgorithm(y, 3), { maxSteps: 4 }).state.scores.length // => 4
 * ```
 */
export function arOrderAlgorithm(
  y: ArrayLike<number>,
  maxOrder: number,
  criterion: OrderCriterion = 'aic',
): Algorithm<ArOrderState, ArOrderPath> {
  const x = asSeries(Float64Array.from(y), 'selectArOrder')
  const m = x.length - maxOrder
  return {
    name: 'ar-order-selection',
    config: { maxOrder, criterion, steps: maxOrder + 1 },
    init: () => ({ p: 0, scores: [], best: -1 }),
    step: (s, ctx) => {
      const p = s.p
      const columns: number[][] = [new Array<number>(m).fill(1)]
      for (let j = 1; j <= p; j++) columns.push(Array.from({ length: m }, (_, t) => x[maxOrder + t - j] ?? 0))
      const target = Array.from({ length: m }, (_, t) => x[maxOrder + t] ?? 0)
      const beta = leastSquares(columns, target)
      let ssr = 0
      for (let t = 0; t < m; t++) {
        let fit = 0
        columns.forEach((c, k) => {
          fit += (beta[k] ?? 0) * (c[t] ?? 0)
        })
        ssr += ((target[t] ?? 0) - fit) ** 2
      }
      const sigma2 = ssr / m
      const k = p + 2
      const score = m * (Math.log(2 * Math.PI * sigma2) + 1) + criterionPenalty(criterion, m) * k
      const scores = [...s.scores, score]
      const best = scores.indexOf(minOf(scores))
      if (ctx.wants('select/order')) ctx.log('select/order', p)
      if (ctx.wants('select/criterion')) ctx.log('select/criterion', score)
      if (ctx.wants('select/sigma2')) ctx.log('select/sigma2', sigma2)
      if (ctx.wants('select/best')) ctx.log('select/best', best)
      return { p: p + 1, scores, best }
    },
    done: (s) => (s.p > maxOrder ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ArOrderState,
  }
}

/**
 * The AR order chosen by an information criterion over least-squares fits on a common sample (statsmodels'
 * `ar_select_order`; {@link arOrderAlgorithm} run to the end): the criterion is -2 loglik + penalty k, with k the lags, the constant and the variance.
 *
 * @param y - The series.
 * @param maxOrder - The largest order tried.
 * @param criterion - `aic` (default), `bic` or `hqic`.
 * @returns The chosen order and the criterion at every order 0..maxOrder.
 * @example
 * ```ts
 * import { selectArOrder } from 'omai-engine/timeseries'
 * // A noiseless AR(2) is identified as order 2 (an AR(0) fits far worse).
 * const y = [1, 1.3, 0.2, -0.6, 0.1, 0.7, 0.2, -0.3, -0.1, 0.2, 0.1, -0.1, 0, 0.1, 0.02, -0.04]
 * selectArOrder(y, 3).order >= 1 // => true
 * ```
 */
export function selectArOrder(
  y: ArrayLike<number>,
  maxOrder: number,
  criterion: OrderCriterion = 'aic',
): { readonly order: number; readonly scores: readonly number[] } {
  const state = runAlgorithm(arOrderAlgorithm(y, maxOrder, criterion), { maxSteps: maxOrder + 1 }).state
  return { order: state.best, scores: state.scores }
}

/**
 * Fit an MA(q) by the innovations algorithm (Brockwell and Davis 5.1.3; statsmodels' `innovations` estimator): run the
 * algorithm on the sample autocovariances to row `m` and take theta_(m,1..q) and v_m.
 *
 * @param y - The series.
 * @param order - The order q.
 * @param options - `m` (default q, statsmodels' choice; a larger m is less biased) and `demean` (default true).
 * @returns The moving-average coefficients and innovation variance.
 * @example
 * ```ts
 * import { fitMa } from 'omai-engine/timeseries'
 * // The innovations estimate of an MA(1) from the autocovariances (1.25, 0.5) of its model.
 * fitMa([1, 0.5, -0.5, 0.2, -0.3, 0.1, 0.4, -0.2], 1).ma.length // => 1
 * ```
 */
export function fitMa(
  y: ArrayLike<number>,
  order: number,
  options: { readonly m?: number; readonly demean?: boolean } = {},
): ArmaEstimate {
  const x = asSeries(Float64Array.from(y), 'fitMa')
  const m = options.m ?? order
  if (!(Number.isInteger(order) && order >= 1 && m >= order && m < x.length))
    throw new OmaiError('value', 'fitMa: need 1 <= order <= m < n')
  const demean = options.demean ?? true
  const gamma = autocovariance(x, m, { demean })
  const { theta, v } = innovations(gamma)
  return {
    ar: [],
    ma: (theta[m] ?? []).slice(0, order),
    sigma2: v[m] ?? Number.NaN,
    mean: demean ? meanOf(x) : 0,
    method: 'innovations',
  }
}

/** Options of {@link hannanRissanen}. */
export interface HannanRissananOptions {
  /** Remove the mean first (default true). */
  readonly demean?: boolean
  /** The order of the long autoregression of the first step (default max(floor(log(n)^2), 2 max(p, q))). */
  readonly initialArOrder?: number
  /** Apply the bias-correction step (default: when the second-step estimate is stationary and invertible). */
  readonly unbiased?: boolean
}

/**
 * The Hannan-Rissanen estimator of an ARMA(p, q) (Brockwell and Davis 5.1.4; statsmodels' `hannan_rissanen`): a long
 * autoregression gives residuals, the ARMA coefficients come from least squares on lagged series and residuals, and
 * a final regression removes the bias when the estimate implies a stationary, invertible process.
 *
 * @param y - The series.
 * @param order - `[p, q]`.
 * @param options - Demeaning, the initial AR order and the bias correction.
 * @returns The estimate.
 * @example
 * ```ts
 * import { hannanRissanen } from 'omai-engine/timeseries'
 * const y = Array.from({ length: 60 }, (_, t) => Math.sin(t * 0.7) + 0.3 * Math.cos(t * 1.9))
 * hannanRissanen(y, [1, 1]).method // => 'hannan-rissanen'
 * ```
 */
export function hannanRissanen(
  y: ArrayLike<number>,
  order: readonly [number, number],
  options: HannanRissananOptions = {},
): ArmaEstimate {
  const [p, q] = order
  const raw = asSeries(Float64Array.from(y), 'hannanRissanen')
  const mu = (options.demean ?? true) ? meanOf(raw) : 0
  const x = Float64Array.from(raw, (v) => v - mu)
  const n = x.length
  if (p === 0 && q === 0) return { ar: [], ma: [], sigma2: sumSquares(x) / n, mean: mu, method: 'hannan-rissanen' }
  const lag = (series: ArrayLike<number>, k: number, j: number): number => series[k - j] ?? 0
  if (q === 0) {
    const { beta, sigma2 } = regress(x, p, 0, p, [], 0)
    return { ar: beta, ma: [], sigma2, mean: mu, method: 'hannan-rissanen' }
  }
  const initial = options.initialArOrder ?? Math.max(Math.floor(Math.log(n) ** 2), 2 * Math.max(p, q))
  if (initial + q + 1 >= n) throw new OmaiError('value', 'hannanRissanen: the series is too short')
  const long = toFlat(yuleWalker(x, initial, { method: 'mle' }).ar)
  const resid = new Float64Array(n - initial)
  for (let t = initial; t < n; t++) {
    let v = x[t] ?? 0
    for (let j = 1; j <= initial; j++) v -= (long[j - 1] ?? 0) * lag(x, t, j)
    resid[t - initial] = v
  }
  const start = initial + q
  const { beta, sigma2: s2 } = regress(x, p, q, start, resid, initial)
  let ar = beta.slice(0, p)
  let ma = beta.slice(p)
  let sigma2 = s2
  const ok = isStationary(ar) && isInvertible(ma)
  const unbiased = options.unbiased ?? ok
  if (unbiased) {
    if (!ok) throw new OmaiError('value', 'hannanRissanen: the second-step estimate is not stationary and invertible')
    const k = Math.max(p, q)
    const z = new Float64Array(n)
    for (let t = k; t < n; t++) {
      let v = x[t] ?? 0
      for (let j = 1; j <= p; j++) v -= (ar[j - 1] ?? 0) * lag(x, t, j)
      for (let j = 1; j <= q; j++) v -= (ma[j - 1] ?? 0) * lag(z, t, j)
      z[t] = v
    }
    // V filters z by 1 / phi(B); W filters it by theta(B)^-1-style recursion with the signs of statsmodels.
    const vv = new Float64Array(n)
    const ww = new Float64Array(n)
    for (let t = 0; t < n; t++) {
      let a = z[t] ?? 0
      for (let j = 1; j <= p; j++) a += (ar[j - 1] ?? 0) * lag(vv, t, j)
      vv[t] = a
      let b = z[t] ?? 0
      for (let j = 1; j <= q; j++) b -= (ma[j - 1] ?? 0) * lag(z, t, j)
      ww[t] = b
    }
    const rows = n - k
    const columns: number[][] = []
    for (let j = 1; j <= p; j++) columns.push(Array.from({ length: rows }, (_, i) => lag(vv, k + i, j)))
    for (let j = 1; j <= q; j++) columns.push(Array.from({ length: rows }, (_, i) => lag(ww, k + i, j)))
    const delta = leastSquares(columns, Array.from(z.subarray(k)))
    ar = ar.map((v, j) => v + (delta[j] ?? 0))
    ma = ma.map((v, j) => v + (delta[p + j] ?? 0))
    // Recompute the variance from the second-step regression design, as statsmodels does.
    const design = lagColumns(x, p, resid, q, start, initial)
    const target = Array.from(x.subarray(start))
    let ssr = 0
    for (let i = 0; i < target.length; i++) {
      let fit = 0
      design.forEach((c, j) => {
        fit += (j < p ? (ar[j] ?? 0) : (ma[j - p] ?? 0)) * (c[i] ?? 0)
      })
      ssr += ((target[i] ?? 0) - fit) ** 2
    }
    sigma2 = ssr / target.length
  }
  return { ar, ma, sigma2, mean: mu, method: 'hannan-rissanen' }
}

function sumSquares(x: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < x.length; i++) s += (x[i] ?? 0) ** 2
  return s
}

/** The regression columns of lagged series (p lags) and lagged residuals (q lags) for targets x[start..]. */
function lagColumns(
  x: ArrayLike<number>,
  p: number,
  resid: ArrayLike<number>,
  q: number,
  start: number,
  residOffset: number,
): number[][] {
  const rows = x.length - start
  const columns: number[][] = []
  for (let j = 1; j <= p; j++) columns.push(Array.from({ length: rows }, (_, i) => x[start + i - j] ?? 0))
  for (let j = 1; j <= q; j++)
    columns.push(Array.from({ length: rows }, (_, i) => resid[start + i - residOffset - j] ?? 0))
  return columns
}

function regress(
  x: ArrayLike<number>,
  p: number,
  q: number,
  start: number,
  resid: ArrayLike<number>,
  residOffset: number,
): { readonly beta: number[]; readonly sigma2: number; readonly residuals: Float64Array } {
  const columns = lagColumns(x, p, resid, q, start, residOffset)
  const target = Array.from({ length: x.length - start }, (_, i) => x[start + i] ?? 0)
  const beta = leastSquares(columns, target)
  const residuals = new Float64Array(target.length)
  for (let i = 0; i < target.length; i++) {
    let fit = 0
    columns.forEach((c, j) => {
      fit += (beta[j] ?? 0) * (c[i] ?? 0)
    })
    residuals[i] = (target[i] ?? 0) - fit
  }
  // statsmodels' OLS scale: the residual sum of squares over the residual degrees of freedom.
  return { beta, sigma2: sumSquares(residuals) / (target.length - columns.length), residuals }
}

/** The estimators of an ARMA(p, q). */
export type ArmaMethod = 'mle' | 'css' | 'css-mle' | 'hannan-rissanen'

/** Options of {@link fitArma} and {@link armaFitAlgorithm}. */
export interface FitArmaOptions extends Omit<ArimaOptions, 'method'> {
  /** The estimator (default `mle`). */
  readonly method?: ArmaMethod
  /** Fit a mean (default true). */
  readonly demean?: boolean
}

function estimateOf(f: ReturnType<typeof arimaFit>, method: string): ArmaEstimate {
  return {
    ar: f.ar,
    ma: f.ma,
    sigma2: f.sigma2,
    mean: f.regression[0] ?? 0,
    logLikelihood: f.logLikelihood,
    method,
  }
}

/**
 * Fit an ARMA(p, q) with an optional mean by exact maximum likelihood (`mle`), conditional sum of squares (`css`), CSS
 * followed by the exact likelihood (`css-mle`), or the Hannan-Rissanen regression. The likelihood methods run the
 * traced BFGS algorithm of {@link armaFitAlgorithm}.
 *
 * @param y - The series.
 * @param order - `[p, q]`.
 * @param options - The method, mean and optimiser options.
 * @returns The estimate.
 * @example
 * ```ts
 * import { fitArma } from 'omai-engine/timeseries'
 * const y = [0.3, 0.9, 0.4, -0.2, 0.5, 1.1, 0.6, -0.1, 0.2, 0.8, 0.5, -0.3, 0.1, 0.7, 0.4, 0.0]
 * fitArma(y, [1, 0]).ar.length // => 1
 * ```
 */
export function fitArma(
  y: ArrayLike<number>,
  order: readonly [number, number],
  options: FitArmaOptions = {},
): ArmaEstimate {
  const method = options.method ?? 'mle'
  if (method === 'hannan-rissanen')
    return hannanRissanen(y, order, options.demean === undefined ? {} : { demean: options.demean })
  const fit = arimaFit(
    y,
    { order: [order[0], 0, order[1]], trend: (options.demean ?? true) ? 'c' : 'n' },
    { ...options, method },
  )
  return estimateOf(fit, method)
}

/**
 * The likelihood fit of an ARMA(p, q) as a traced algorithm, one BFGS iteration per step with the log-likelihood as a
 * metric (see {@link arimaFitAlgorithm}, which this specialises to d = 0).
 *
 * @param y - The series.
 * @param order - `[p, q]`.
 * @param options - The method (`mle`, `css` or `css-mle`), mean and optimiser options.
 * @returns The algorithm and the map from its state to the fit.
 * @example
 * ```ts
 * import { armaFitAlgorithm } from 'omai-engine/timeseries'
 * const y = [0.3, 0.9, 0.4, -0.2, 0.5, 1.1, 0.6, -0.1, 0.2, 0.8, 0.5, -0.3, 0.1, 0.7, 0.4, 0.0]
 * armaFitAlgorithm(y, [1, 1]).algorithm.name // => 'arima-mle'
 * ```
 */
export function armaFitAlgorithm(
  y: ArrayLike<number>,
  order: readonly [number, number],
  options: FitArmaOptions = {},
): ArimaProblem {
  const method = options.method === 'hannan-rissanen' ? 'mle' : (options.method ?? 'mle')
  return arimaFitAlgorithm(
    y,
    { order: [order[0], 0, order[1]], trend: (options.demean ?? true) ? 'c' : 'n' },
    { ...options, method },
  )
}
