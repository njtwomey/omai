/**
 * Tests on time series, statsmodels style: the Ljung-Box and Box-Pierce tests of autocorrelation, the Durbin-Watson
 * statistic, the augmented Dickey-Fuller unit-root test with MacKinnon's p-values and critical values, the KPSS test of
 * stationarity, and the Wald-Wolfowitz runs test of randomness.
 */

import { MdmlError, meanOf, type TestResult } from '../../base/index.ts'
import type { TensorLike } from '../../tensor/index.ts'
import { ols } from '../_linear.ts'
import { chi2Sf, normCdf, normSf, sample } from '../_util.ts'
import { median } from '../quantiles.ts'
import { testResult } from './result.ts'

/** The sample autocorrelations r_1..r_maxlag (statsmodels acf without fft or adjustment). */
function acf(x: Float64Array, maxlag: number): Float64Array {
  const n = x.length
  const m = meanOf(x)
  let c0 = 0
  for (const v of x) c0 += (v - m) ** 2
  const r = new Float64Array(maxlag + 1)
  r[0] = 1
  for (let k = 1; k <= maxlag; k++) {
    let s = 0
    for (let t = 0; t + k < n; t++) s += ((x[t] ?? 0) - m) * ((x[t + k] ?? 0) - m)
    r[k] = s / c0
  }
  return r
}

/** Options of {@link ljungBox}. */
export interface LjungBoxOptions {
  /** The lag (default min(10, n / 5), statsmodels' default). */
  readonly lags?: number
  /** Use the Box-Pierce statistic n sum r_k^2 instead (default false). */
  readonly boxPierce?: boolean
  /** Degrees of freedom used by a fitted model, subtracted from the lags (default 0). */
  readonly modelDf?: number
}

/**
 * The Ljung-Box (or Box-Pierce) portmanteau test of no autocorrelation up to a lag (statsmodels acorr_ljungbox): Q =
 * n (n + 2) sum r_k^2 / (n - k), against chi-square with lags - modelDf degrees of freedom.
 *
 * @param x - The series (residuals of a fitted model, typically).
 * @param options - lags, boxPierce and modelDf.
 * @returns Q at the lag and its p-value.
 * @example
 * ```ts
 * import { ljungBox } from 'mdml-engine/stats/tests'
 * ljungBox([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], { lags: 1 }).statistic // => 10.8
 * ```
 */
export function ljungBox(x: TensorLike, options: LjungBoxOptions = {}): TestResult {
  const a = sample('ljungBox', x, 3)
  const n = a.length
  const lags = options.lags ?? Math.min(10, Math.floor(n / 5))
  if (!(Number.isInteger(lags) && lags >= 1 && lags < n))
    throw new MdmlError('value', 'ljungBox: lags must be in [1, n)')
  const r = acf(a, lags)
  let q = 0
  for (let k = 1; k <= lags; k++) q += options.boxPierce === true ? (r[k] ?? 0) ** 2 : (r[k] ?? 0) ** 2 / (n - k)
  q *= options.boxPierce === true ? n : n * (n + 2)
  const df = lags - (options.modelDf ?? 0)
  return testResult({
    test: options.boxPierce === true ? 'Box-Pierce test' : 'Ljung-Box test',
    statistic: q,
    pValue: df > 0 ? chi2Sf(df, q) : Number.NaN,
    df,
    alternative: 'greater',
  })
}

/**
 * The Durbin-Watson statistic of residuals, sum (e_t - e_t-1)^2 / sum e_t^2 (statsmodels durbin_watson), near 2 without
 * first-order autocorrelation. The exact null distribution depends on the regressors, so the p-value here is the
 * asymptotic one, from d ~ N(2, 4 / n).
 *
 * @param residuals - The residuals.
 * @returns d and its asymptotic two-sided p-value.
 * @example
 * ```ts
 * import { durbinWatson } from 'mdml-engine/stats/tests'
 * durbinWatson([1, -1, 1, -1]).statistic // => 3
 * ```
 */
export function durbinWatson(residuals: TensorLike): TestResult {
  const e = sample('durbinWatson', residuals, 2)
  let num = 0
  let den = 0
  for (let t = 0; t < e.length; t++) {
    den += (e[t] ?? 0) ** 2
    if (t > 0) num += ((e[t] ?? 0) - (e[t - 1] ?? 0)) ** 2
  }
  const d = num / den
  const z = (d - 2) / Math.sqrt(4 / e.length)
  return testResult({
    test: 'Durbin-Watson test',
    statistic: d,
    pValue: 2 * normSf(Math.abs(z)),
    method: 'asymptotic normal',
  })
}

/** The deterministic terms of a unit-root regression. */
export type TrendTerms = 'n' | 'c' | 'ct' | 'ctt'

// MacKinnon (1994) p-value surfaces and MacKinnon (2010) critical values for N = 1 (statsmodels adfvalues).
const TAU_MAX: Readonly<Record<TrendTerms, number>> = { n: Number.POSITIVE_INFINITY, c: 2.74, ct: 0.7, ctt: 0.54 }
const TAU_MIN: Readonly<Record<TrendTerms, number>> = { n: -19.04, c: -18.83, ct: -16.18, ctt: -17.17 }
const TAU_STAR: Readonly<Record<TrendTerms, number>> = { n: -1.04, c: -1.61, ct: -2.89, ctt: -3.21 }
const TAU_SMALL: Readonly<Record<TrendTerms, readonly number[]>> = {
  n: [0.6344, 1.2378, 0.032496],
  c: [2.1659, 1.4412, 0.038269],
  ct: [3.2512, 1.6047, 0.049588],
  ctt: [4.0003, 1.658, 0.048288],
}
const TAU_LARGE: Readonly<Record<TrendTerms, readonly number[]>> = {
  n: [0.4797, 0.93557, -0.06999, 0.033066],
  c: [1.7339, 0.93202, -0.12745, -0.010368],
  ct: [2.5261, 0.61654, -0.37956, -0.060285],
  ctt: [3.0778, 0.49529, -0.41477, -0.059359],
}
const TAU_CRIT: Readonly<Record<TrendTerms, readonly (readonly number[])[]>> = {
  n: [
    [-2.56574, -2.2358, -3.627, 0],
    [-1.941, -0.2686, -3.365, 31.223],
    [-1.61682, 0.2656, -2.714, 25.364],
  ],
  c: [
    [-3.43035, -6.5393, -16.786, -79.433],
    [-2.86154, -2.8903, -4.234, -40.04],
    [-2.56677, -1.5384, -2.809, 0],
  ],
  ct: [
    [-3.95877, -9.0531, -28.428, -134.155],
    [-3.41049, -4.3904, -9.036, -45.374],
    [-3.12705, -2.5856, -3.925, -22.38],
  ],
  ctt: [
    [-4.37113, -11.5882, -35.819, -334.047],
    [-3.83239, -5.9057, -12.49, -118.284],
    [-3.55326, -3.6596, -5.293, -63.559],
  ],
}

const poly = (c: readonly number[], x: number): number => {
  let r = 0
  for (let i = c.length - 1; i >= 0; i--) r = r * x + (c[i] ?? 0)
  return r
}

/**
 * MacKinnon's approximate p-value of a Dickey-Fuller t statistic (statsmodels mackinnonp, N = 1).
 *
 * @param stat - The t statistic of the lagged level.
 * @param regression - The deterministic terms (default `'c'`).
 * @returns The p-value.
 * @example
 * ```ts
 * import { mackinnonp } from 'mdml-engine/stats/tests'
 * mackinnonp(-2.86154) // => 0.05
 * ```
 */
export function mackinnonp(stat: number, regression: TrendTerms = 'c'): number {
  if (stat > TAU_MAX[regression]) return 1
  if (stat < TAU_MIN[regression]) return 0
  return normCdf(poly(stat <= TAU_STAR[regression] ? TAU_SMALL[regression] : TAU_LARGE[regression], stat))
}

/**
 * MacKinnon's (2010) finite-sample critical values of the Dickey-Fuller test at 1%, 5% and 10% (statsmodels
 * mackinnoncrit, N = 1).
 *
 * @param nobs - The regression's sample size (Infinity for the asymptotic values).
 * @param regression - The deterministic terms (default `'c'`).
 * @returns The three critical values.
 * @example
 * ```ts
 * import { mackinnoncrit } from 'mdml-engine/stats/tests'
 * mackinnoncrit(Number.POSITIVE_INFINITY) // => [-3.43035, -2.86154, -2.56677]
 * ```
 */
export function mackinnoncrit(nobs: number, regression: TrendTerms = 'c'): [number, number, number] {
  const t = TAU_CRIT[regression]
  const v = (row: readonly number[]): number => (Number.isFinite(nobs) ? poly(row, 1 / nobs) : (row[0] ?? 0))
  return [v(t[0] ?? []), v(t[1] ?? []), v(t[2] ?? [])]
}

/** The design of an ADF regression with `lag` lagged differences, on the last `nobs` usable observations. */
function adfDesign(
  x: Float64Array,
  lag: number,
  regression: TrendTerms,
  nobs: number,
): { X: Float64Array; y: Float64Array; p: number } {
  const dx = Float64Array.from({ length: x.length - 1 }, (_, i) => (x[i + 1] ?? 0) - (x[i] ?? 0))
  const ntrend = regression === 'n' ? 0 : regression.length
  const p = ntrend + 1 + lag
  const X = new Float64Array(nobs * p)
  const y = new Float64Array(nobs)
  const first = dx.length - nobs
  for (let r = 0; r < nobs; r++) {
    const t = first + r
    y[r] = dx[t] ?? 0
    let c = 0
    if (ntrend >= 1) X[r * p + c++] = 1
    if (ntrend >= 2) X[r * p + c++] = r + 1
    if (ntrend >= 3) X[r * p + c++] = (r + 1) ** 2
    X[r * p + c++] = x[t] ?? 0
    for (let j = 1; j <= lag; j++) X[r * p + c++] = dx[t - j] ?? 0
  }
  return { X, y, p }
}

/** Options of {@link adfuller}. */
export interface AdfOptions {
  /** The largest lag considered (default 12 (n / 100)^(1/4), capped at n / 2 - ntrend - 1). */
  readonly maxlag?: number
  /** Deterministic terms (default `'c'`). */
  readonly regression?: TrendTerms
  /** Lag selection: `'AIC'` (default), `'BIC'`, `'t-stat'` (general to specific at 1.645), or none (use maxlag). */
  readonly autolag?: 'AIC' | 'BIC' | 't-stat' | null
}

/**
 * The augmented Dickey-Fuller unit-root test (statsmodels adfuller): the t statistic of the lagged level in the
 * regression of the differences on deterministic terms, the level and lagged differences, with the lag chosen by
 * information criterion on a common sample, MacKinnon's p-value and critical values in `assumptions`.
 *
 * @param x - The series.
 * @param options - maxlag, regression and autolag.
 * @returns The ADF statistic and its p-value (small p rejects a unit root).
 * @example
 * ```ts
 * import { adfuller } from 'mdml-engine/stats/tests'
 * const x = [0.5, -0.2, 0.3, 0.1, -0.4, 0.2, 0.6, -0.3, 0.1, 0.4, -0.5, 0.2, 0.3, -0.1, 0.2, -0.6, 0.4, 0.1, -0.2, 0.3]
 * adfuller(x, { autolag: null, maxlag: 1 }).pValue < 0.05 // => true
 * ```
 */
export function adfuller(x: TensorLike, options: AdfOptions = {}): TestResult {
  const a = sample('adfuller', x, 4)
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of a) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  if (lo === hi) throw new MdmlError('value', 'adfuller: the series is constant')
  const regression = options.regression ?? 'c'
  const n = a.length
  const ntrend = regression === 'n' ? 0 : regression.length
  let maxlag = options.maxlag ?? Math.min(Math.floor(n / 2) - ntrend - 1, Math.ceil(12 * (n / 100) ** 0.25))
  if (maxlag < 0) throw new MdmlError('value', 'adfuller: the sample is too short for the regression')
  if (options.maxlag !== undefined && maxlag > Math.floor(n / 2) - ntrend - 1)
    throw new MdmlError('value', 'adfuller: maxlag must be less than n / 2 - 1 - ntrend')
  maxlag = Math.floor(maxlag)
  const autolag = options.autolag === undefined ? 'AIC' : options.autolag
  let usedlag = maxlag
  let icbest = Number.NaN
  if (autolag !== null) {
    const nobs = n - 1 - maxlag
    const fits: { lag: number; aic: number; bic: number; tLast: number }[] = []
    for (let lag = 0; lag <= maxlag; lag++) {
      const { X, y, p } = adfDesign(a, lag, regression, nobs)
      const f = ols(X, nobs, p, y)
      const llf = (-nobs / 2) * (Math.log(2 * Math.PI) + Math.log(f.rss / nobs) + 1)
      fits.push({
        lag,
        aic: -2 * llf + 2 * p,
        bic: -2 * llf + Math.log(nobs) * p,
        tLast: (f.beta[p - 1] ?? 0) / (f.se[p - 1] ?? 1),
      })
    }
    if (autolag === 't-stat') {
      usedlag = 0
      icbest = 0
      for (let lag = maxlag; lag >= 0; lag--) {
        icbest = Math.abs(fits[lag]?.tLast ?? 0)
        usedlag = lag
        if (icbest >= 1.6448536269514722) break
      }
    } else {
      let best = Number.POSITIVE_INFINITY
      for (const f of fits) {
        const v = autolag === 'AIC' ? f.aic : f.bic
        if (v < best) {
          best = v
          usedlag = f.lag
        }
      }
      icbest = best
    }
  }
  const nobs = n - 1 - usedlag
  const { X, y, p } = adfDesign(a, usedlag, regression, nobs)
  const fit = ols(X, nobs, p, y)
  const level = ntrend
  const stat = (fit.beta[level] ?? 0) / (fit.se[level] ?? 1)
  const [c1, c5, c10] = mackinnoncrit(nobs, regression)
  return testResult({
    test: 'augmented Dickey-Fuller test',
    statistic: stat,
    pValue: mackinnonp(stat, regression),
    alternative: 'less',
    method: `regression '${regression}', ${usedlag} lags`,
    assumptions: { nobs, usedlag, icbest, critical1: c1, critical5: c5, critical10: c10 },
  })
}

/** Options of {@link kpss}. */
export interface KpssOptions {
  /** Level (`'c'`, default) or trend (`'ct'`) stationarity. */
  readonly regression?: 'c' | 'ct'
  /** Bartlett lags: `'auto'` (Hobijn et al.'s data-dependent rule, default), `'legacy'` or a number. */
  readonly nlags?: 'auto' | 'legacy' | number
}

/**
 * The KPSS test of stationarity (statsmodels kpss): the scaled sum of squared partial sums of the residuals from a
 * level (or trend) fit over a Bartlett long-run variance, with the p-value interpolated in Kwiatkowski et al.'s table
 * (so it is capped to [0.01, 0.1]); small p rejects stationarity.
 *
 * @param x - The series.
 * @param options - regression and nlags.
 * @returns The KPSS statistic and its interpolated p-value.
 * @example
 * ```ts
 * import { kpss } from 'mdml-engine/stats/tests'
 * kpss([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], { nlags: 2 }).pValue // => 0.037877
 * ```
 */
export function kpss(x: TensorLike, options: KpssOptions = {}): TestResult {
  const a = sample('kpss', x, 3)
  const n = a.length
  const regression = options.regression ?? 'c'
  let resid: Float64Array
  let crit: number[]
  if (regression === 'ct') {
    const X = Float64Array.from({ length: 2 * n }, (_, k) => (k % 2 === 0 ? 1 : Math.floor(k / 2) + 1))
    resid = ols(X, n, 2, a).residuals
    crit = [0.119, 0.146, 0.176, 0.216]
  } else {
    const m = meanOf(a)
    resid = Float64Array.from(a, (v) => v - m)
    crit = [0.347, 0.463, 0.574, 0.739]
  }
  const dot = (lag: number): number => {
    let s = 0
    for (let t = lag; t < n; t++) s += (resid[t] ?? 0) * (resid[t - lag] ?? 0)
    return s
  }
  let nlags: number
  const opt = options.nlags ?? 'auto'
  if (opt === 'legacy') nlags = Math.min(Math.ceil(12 * (n / 100) ** 0.25), n - 1)
  else if (opt === 'auto') {
    const cov = Math.floor(n ** (2 / 9))
    let s0 = dot(0) / n
    let s1 = 0
    for (let i = 1; i <= cov; i++) {
      const p = dot(i) / (n / 2)
      s0 += p
      s1 += i * p
    }
    const sHat = s1 / s0
    nlags = Math.min(Math.trunc(1.1447 * (sHat * sHat) ** (1 / 3) * n ** (1 / 3)), n - 1)
  } else {
    if (!(Number.isInteger(opt) && opt >= 0 && opt < n)) throw new MdmlError('value', 'kpss: nlags must be in [0, n)')
    nlags = opt
  }
  let cum = 0
  let eta = 0
  for (const v of resid) {
    cum += v
    eta += cum * cum
  }
  eta /= n * n
  let s = dot(0)
  for (let i = 1; i <= nlags; i++) s += 2 * dot(i) * (1 - i / (nlags + 1))
  s /= n
  const stat = eta / s
  const pvals = [0.1, 0.05, 0.025, 0.01]
  let p: number
  if (stat <= (crit[0] ?? 0)) p = 0.1
  else if (stat >= (crit[3] ?? 0)) p = 0.01
  else {
    let j = 0
    while (stat > (crit[j + 1] ?? 0)) j++
    p =
      (pvals[j] ?? 0) +
      ((stat - (crit[j] ?? 0)) / ((crit[j + 1] ?? 0) - (crit[j] ?? 0))) * ((pvals[j + 1] ?? 0) - (pvals[j] ?? 0))
  }
  return testResult({
    test: 'KPSS test',
    statistic: stat,
    pValue: p,
    alternative: 'greater',
    assumptions: { nlags },
    method: `${regression === 'c' ? 'level' : 'trend'} stationarity, ${nlags} lags`,
  })
}

/**
 * The Wald-Wolfowitz runs test of randomness (statsmodels runstest_1samp): the number of runs above and below a
 * cutoff, standardised, with a continuity correction for samples under 50.
 *
 * @param x - The series.
 * @param options - cutoff (`'mean'`, default, `'median'` or a number) and correction (default true).
 * @returns z and its two-sided p-value.
 * @example
 * ```ts
 * import { runsTest } from 'mdml-engine/stats/tests'
 * runsTest([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]).statistic > 2 // => true
 * ```
 */
export function runsTest(
  x: TensorLike,
  options: { readonly cutoff?: 'mean' | 'median' | number; readonly correction?: boolean } = {},
): TestResult {
  const a = sample('runsTest', x, 2)
  const cut = options.cutoff ?? 'mean'
  const c = cut === 'mean' ? meanOf(a) : cut === 'median' ? median(a) : cut
  const ind = Array.from(a, (v) => (v >= c ? 1 : 0))
  let runs = 1
  for (let i = 1; i < ind.length; i++) if (ind[i] !== ind[i - 1]) runs++
  const npo = ind.filter((v) => v === 1).length
  const nne = ind.length - npo
  const n = npo + nne
  if (runs === 1) {
    const pval = 1 / 2 ** (Math.min(n, 1024) - 1)
    return testResult({ test: 'runs test', statistic: Number.NEGATIVE_INFINITY, pValue: 2 * pval })
  }
  const npn = npo * nne
  const mean = (2 * npn) / n + 1
  const variance = (2 * npn * (2 * npn - n)) / n ** 2 / (n - 1)
  const d = runs - mean
  let z: number
  if (n >= 50 || options.correction === false) z = d
  else if (d > 0.5) z = d - 0.5
  else if (d < 0.5) z = d + 0.5
  else z = 0
  z /= Math.sqrt(variance)
  return testResult({ test: 'runs test', statistic: z, pValue: 2 * normSf(Math.abs(z)), assumptions: { runs } })
}
