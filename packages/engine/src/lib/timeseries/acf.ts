/**
 * Sample correlation functions of a series, statsmodels style: the autocovariance, the autocorrelation function with
 * Bartlett confidence bands, the partial autocorrelation function (Yule-Walker, with the biased or the adjusted
 * autocovariance, or by regressions on lags) and the cross-correlation of two series.
 */

import { meanOf, OmaiError } from '../base/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import { item, type TensorLike } from '../tensor/index.ts'
import { leastSquares } from './_ml.ts'
import { asSeries } from './_series.ts'

/** Options of {@link autocovariance}. */
export interface AutocovarianceOptions {
  /** Divide the lag-k sum by n - k instead of n (default false, the biased estimator). */
  readonly adjusted?: boolean
  /** Subtract the mean first (default true). */
  readonly demean?: boolean
}

function checkLags(nlags: number, n: number, who: string): void {
  if (!(Number.isInteger(nlags) && nlags >= 0 && nlags < n))
    throw new OmaiError('value', `${who}: nlags must be an integer in [0, n)`)
}

/**
 * The sample autocovariances at lags 0..nlags (statsmodels' `acovf`).
 *
 * @param y - The series.
 * @param nlags - The largest lag (default n - 1).
 * @param options - `adjusted` (divide by n - k) and `demean`.
 * @returns gamma(0..nlags).
 * @example
 * ```ts
 * import { autocovariance } from 'omai-engine/timeseries'
 * Array.from(autocovariance([1, 2, 3, 4], 2)) // => [1.25, 0.3125, -0.375]
 * ```
 */
export function autocovariance(y: TensorLike, nlags?: number, options: AutocovarianceOptions = {}): Float64Array {
  const x = asSeries(y, 'autocovariance')
  const n = x.length
  const lags = nlags ?? n - 1
  checkLags(lags, n, 'autocovariance')
  const mu = options.demean === false ? 0 : meanOf(x)
  const out = new Float64Array(lags + 1)
  for (let k = 0; k <= lags; k++) {
    let s = 0
    for (let t = k; t < n; t++) s += ((x[t] ?? 0) - mu) * ((x[t - k] ?? 0) - mu)
    out[k] = s / (options.adjusted === true ? n - k : n)
  }
  return out
}

/** The autocorrelations with their Bartlett confidence band. */
export interface AcfResult {
  /** rho(0..nlags). */
  readonly acf: Float64Array
  /** The half-width of the band at each lag under Bartlett's formula (undefined without `alpha`). */
  readonly band?: Float64Array
}

/**
 * The sample autocorrelation function (statsmodels' `acf`), optionally with the Bartlett confidence band for a
 * moving-average process of order lag - 1: sqrt((1 + 2 sum_(j<k) r_j^2) / n) times the normal quantile.
 *
 * @param y - The series.
 * @param nlags - The largest lag (default min(10 log10 n, n - 1)).
 * @param options - `adjusted`, `demean` and `alpha` for the band (e.g. 0.05).
 * @returns The autocorrelations and the band.
 * @example
 * ```ts
 * import { acf } from 'omai-engine/timeseries'
 * Array.from(acf([1, 2, 3, 4], 2).acf) // => [1, 0.25, -0.3]
 * ```
 */
export function acf(
  y: TensorLike,
  nlags?: number,
  options: AutocovarianceOptions & { readonly alpha?: number } = {},
): AcfResult {
  const x = asSeries(y, 'acf')
  const n = x.length
  const lags = nlags ?? Math.min(Math.floor(10 * Math.log10(n)), n - 1)
  const g = autocovariance(x, lags, options)
  const r = Float64Array.from(g, (v) => v / (g[0] ?? 1))
  if (options.alpha === undefined) return { acf: r }
  const q = item(ndtri(1 - options.alpha / 2))
  const band = new Float64Array(lags + 1)
  let cum = 0
  for (let k = 1; k <= lags; k++) {
    band[k] = q * Math.sqrt((1 + 2 * cum) / n)
    cum += (r[k] ?? 0) ** 2
  }
  return { acf: r, band }
}

/** How {@link pacf} estimates the partial autocorrelations. */
export type PacfMethod = 'yule-walker' | 'yule-walker-mle' | 'ols'

/**
 * The sample partial autocorrelation function (statsmodels' `pacf`): the last coefficient of the AR(k) fit for each k,
 * by the Levinson-Durbin recursion on the adjusted (`yule-walker`) or biased (`yule-walker-mle`) autocovariances, or by
 * least squares of the series on its first k lags and a constant (`ols`).
 *
 * @param y - The series.
 * @param nlags - The largest lag (default min(10 log10 n, n / 2 - 1)).
 * @param method - The estimator (default `yule-walker`, the adjusted autocovariance).
 * @returns phi_kk for k = 0..nlags (phi_00 = 1).
 * @example
 * ```ts
 * import { pacf } from 'omai-engine/timeseries'
 * Array.from(pacf([1, 3, 2, 5, 4, 6, 5, 8], 2, 'yule-walker-mle')).map((v) => Math.round(v * 1e6) / 1e6) // => [1, 0.244718, 0.374513]
 * ```
 */
export function pacf(y: TensorLike, nlags?: number, method: PacfMethod = 'yule-walker'): Float64Array {
  const x = asSeries(y, 'pacf')
  const n = x.length
  const lags = nlags ?? Math.min(Math.floor(10 * Math.log10(n)), Math.floor(n / 2) - 1)
  checkLags(lags, n, 'pacf')
  const out = new Float64Array(lags + 1)
  out[0] = 1
  if (method === 'ols') {
    const mu = meanOf(x)
    for (let k = 1; k <= lags; k++) {
      const m = n - k
      const columns: number[][] = [new Array<number>(m).fill(1)]
      for (let j = 1; j <= k; j++) columns.push(Array.from({ length: m }, (_, t) => (x[k + t - j] ?? 0) - mu))
      const beta = leastSquares(
        columns,
        Array.from({ length: m }, (_, t) => (x[k + t] ?? 0) - mu),
      )
      out[k] = beta[k] ?? Number.NaN
    }
    return out
  }
  const g = autocovariance(x, lags, { adjusted: method === 'yule-walker' })
  let phi: number[] = []
  let v = g[0] ?? 1
  for (let k = 1; k <= lags; k++) {
    let num = g[k] ?? 0
    for (let j = 1; j < k; j++) num -= (phi[j - 1] ?? 0) * (g[k - j] ?? 0)
    const r = num / v
    phi = [...phi.map((p, j) => p - r * (phi[k - 2 - j] ?? 0)), r]
    v *= 1 - r * r
    out[k] = r
  }
  return out
}

/**
 * The sample cross-correlation of x with y at lags 0..nlags (statsmodels' `ccf`): corr(x_(t+k), y_t).
 *
 * @param x - The first series.
 * @param y - The second series.
 * @param nlags - The largest lag (default n - 1).
 * @param adjusted - Divide by n - k instead of n (default true, as statsmodels).
 * @returns The cross-correlations.
 * @example
 * ```ts
 * import { ccf } from 'omai-engine/timeseries'
 * // A series against itself is its autocorrelation.
 * Array.from(ccf([1, 2, 3, 4], [1, 2, 3, 4], 2, false)) // => [1, 0.25, -0.3]
 * ```
 */
export function ccf(x: TensorLike, y: TensorLike, nlags?: number, adjusted = true): Float64Array {
  const a = asSeries(x, 'ccf')
  const b = asSeries(y, 'ccf')
  if (a.length !== b.length) throw new OmaiError('shape', 'ccf: the series must have the same length')
  const n = a.length
  const lags = nlags ?? n - 1
  checkLags(lags, n, 'ccf')
  const ma = meanOf(a)
  const mb = meanOf(b)
  const sa = Math.sqrt(a.reduce((s, v) => s + (v - ma) ** 2, 0) / n)
  const sb = Math.sqrt(b.reduce((s, v) => s + (v - mb) ** 2, 0) / n)
  const out = new Float64Array(lags + 1)
  for (let k = 0; k <= lags; k++) {
    let s = 0
    for (let t = k; t < n; t++) s += ((a[t] ?? 0) - ma) * ((b[t - k] ?? 0) - mb)
    out[k] = s / (adjusted ? n - k : n) / (sa * sb)
  }
  return out
}
