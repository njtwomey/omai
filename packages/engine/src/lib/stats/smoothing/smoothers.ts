/**
 * Scatterplot smoothers, nonparametric estimates of a regression curve: LOWESS and LOESS (local weighted polynomial
 * fits, as statsmodels and R), Nadaraya-Watson and local linear kernel regression, running (moving-window) statistics
 * as pandas rolling, and cubic smoothing splines with the penalty chosen by degrees of freedom or generalised
 * cross-validation.
 */

import { MdmlError } from '../../base/index.ts'
import { makeSmoothingSpline, type SmoothingSpline } from '../../numerics/interpolate/index.ts'
import { brentq } from '../../numerics/roots/index.ts'
import { asTensor, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { ols } from '../_linear.ts'
import { paired, sample, vec } from '../_util.ts'
import { median } from '../quantiles.ts'

function sortedPairs(name: string, x: TensorLike, y: TensorLike): { x: Float64Array; y: Float64Array } {
  const [a, b] = paired(name, x, y, 2)
  const order = Array.from({ length: a.length }, (_, i) => i).sort((p, q) => (a[p] ?? 0) - (a[q] ?? 0) || p - q)
  return { x: Float64Array.from(order, (i) => a[i] ?? 0), y: Float64Array.from(order, (i) => b[i] ?? 0) }
}

/** Options of {@link lowess}. */
export interface LowessOptions {
  /** The fraction of the data in each local fit (default 2/3). */
  readonly frac?: number
  /** Robustifying iterations (default 3). */
  readonly it?: number
  /** Points within delta of the last fitted point are linearly interpolated instead of fitted (default 0). */
  readonly delta?: number
}

/** The result of {@link lowess}: the sorted covariate and the smoothed values. */
export interface LowessResult {
  readonly x: Tensor
  readonly fitted: Tensor
}

/**
 * Cleveland's LOWESS (statsmodels.nonparametric.lowess): locally weighted linear regression on the k = frac n nearest
 * points with tricube weights, then `it` robustifying passes reweighting by the bisquare of residuals over six median
 * absolute residuals; points within delta of a fitted one are interpolated.
 *
 * @param x - The covariate.
 * @param y - The response.
 * @param options - frac, it and delta.
 * @returns The sorted covariate and the fit at it.
 * @example
 * ```ts
 * import { lowess } from 'mdml-engine/stats/smoothing'
 * lowess([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]).fitted // => [2, 4, 6, 8, 10]
 * ```
 */
export function lowess(x: TensorLike, y: TensorLike, options: LowessOptions = {}): LowessResult {
  const { x: xs, y: ys } = sortedPairs('lowess', x, y)
  const n = xs.length
  const frac = options.frac ?? 2 / 3
  if (!(frac >= 0 && frac <= 1)) throw new MdmlError('value', 'lowess: frac must lie in [0, 1]')
  const delta = options.delta ?? 0
  let k = Math.trunc(frac * n + 1e-10)
  k = Math.min(n, Math.max(2, k))
  let robust = new Float64Array(n).fill(1)
  const fit = new Float64Array(n)
  const w = new Float64Array(n)
  for (let iter = 0; iter <= (options.it ?? 3); iter++) {
    fit.fill(0)
    let i = 0
    let last = -1
    let left = 0
    let right = k
    for (;;) {
      const xv = xs[i] ?? 0
      while (right < n && xv > ((xs[left] ?? 0) + (xs[right] ?? 0)) / 2) {
        left++
        right++
      }
      const radius = Math.max(xv - (xs[left] ?? 0), (xs[right - 1] ?? 0) - xv)
      w.fill(0)
      let sumW = 0
      let nonzero = 0
      for (let j = left; j < right; j++) {
        const u = Math.abs((xs[j] ?? 0) - xv) / radius
        const t = 1 - u * u * u
        const v = t * t * t * (robust[j] ?? 0)
        w[j] = v
        sumW += v
        if (v > 1e-12) nonzero++
      }
      if (nonzero < 2) fit[i] = ys[i] ?? 0
      else {
        let mx = 0
        for (let j = left; j < right; j++) {
          w[j] = (w[j] ?? 0) / sumW
          mx += (w[j] ?? 0) * (xs[j] ?? 0)
        }
        let ss = 0
        for (let j = left; j < right; j++) ss += (w[j] ?? 0) * ((xs[j] ?? 0) - mx) ** 2
        ss = Math.max(ss, 1e-12)
        let f = 0
        for (let j = left; j < right; j++)
          f += (w[j] ?? 0) * (1 + ((xv - mx) * ((xs[j] ?? 0) - mx)) / ss) * (ys[j] ?? 0)
        fit[i] = f
      }
      if (last < i - 1) {
        const denom = xv - (xs[last] ?? 0)
        for (let j = last + 1; j < i; j++) {
          const a = ((xs[j] ?? 0) - (xs[last] ?? 0)) / denom
          fit[j] = a * (fit[i] ?? 0) + (1 - a) * (fit[last] ?? 0)
        }
      }
      // statsmodels' update_indices.
      last = i
      const cut = (xs[last] ?? 0) + delta
      let kk = last
      for (let j = last + 1; j < n; j++) {
        kk = j
        if ((xs[j] ?? 0) > cut) break
        if (xs[j] === xs[last]) {
          fit[j] = fit[last] ?? 0
          last = j
        }
      }
      i = Math.max(kk - 1, last + 1)
      if (last >= n - 1) break
    }
    const resid = Float64Array.from(ys, (v, j) => Math.abs(v - (fit[j] ?? 0)))
    const med = median(resid)
    robust = Float64Array.from(resid, (r) => {
      const u = Math.min(1, med === 0 ? (r > 0 ? 1 : 0) : r / (6 * med))
      const t = 1 - u * u
      return t * t
    })
  }
  return { x: vec(xs), fitted: vec(fit) }
}

/** Options of {@link loess}. */
export interface LoessOptions {
  /** The fraction of points in each neighbourhood (default 0.75). */
  readonly span?: number
  /** The local polynomial degree, 0, 1 or 2 (default 2). */
  readonly degree?: 0 | 1 | 2
  /** `'gaussian'` (default) or `'symmetric'` (robust bisquare reweighting, as R's loess). */
  readonly family?: 'gaussian' | 'symmetric'
  /** Robustness iterations for the symmetric family (default 4). */
  readonly iterations?: number
}

/**
 * LOESS (Cleveland and Devlin 1988; R's loess with surface "direct"): at each point a local polynomial of the chosen
 * degree is fitted by weighted least squares to the floor(span n) nearest observations with tricube weights scaled by
 * the largest of their distances; the symmetric family adds bisquare robustness weights.
 *
 * @param x - The covariate.
 * @param y - The response.
 * @param options - span, degree, family and iterations.
 * @param at - Points to evaluate at (default the observations, in their original order).
 * @returns The fitted values.
 * @example
 * ```ts
 * import { loess } from 'mdml-engine/stats/smoothing'
 * loess([0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 4, 9, 16, 25, 36, 49]) // => [0, 1, 4, 9, 16, 25, 36, 49]
 * ```
 */
export function loess(x: TensorLike, y: TensorLike, options: LoessOptions = {}, at?: TensorLike): Tensor {
  const [xs, ys] = paired('loess', x, y, 2)
  const n = xs.length
  const degree = options.degree ?? 2
  const q = Math.min(n, Math.max(degree + 1, Math.floor(n * (options.span ?? 0.75))))
  const points = at === undefined ? xs : sample('loess', at, 0)
  let robust = new Float64Array(n).fill(1)
  const fitAt = (x0: number, wts: Float64Array): number => {
    const d = Float64Array.from(xs, (v) => Math.abs(v - x0))
    const sorted = Float64Array.from(d).sort()
    let dq = sorted[q - 1] ?? 0
    const span = options.span ?? 0.75
    if (span > 1) dq *= span
    if (!(dq > 0)) dq = 1e-300
    const p = degree + 1
    const X: number[] = []
    const yy: number[] = []
    for (let i = 0; i < n; i++) {
      const u = (d[i] ?? 0) / dq
      if (u >= 1) continue
      const t = 1 - u * u * u
      const wt = t * t * t * (wts[i] ?? 0)
      if (!(wt > 0)) continue
      const s = Math.sqrt(wt)
      const dx = (xs[i] ?? 0) - x0
      for (let c = 0; c < p; c++) X.push(s * dx ** c)
      yy.push(s * (ys[i] ?? 0))
    }
    const m = yy.length
    if (m < p) {
      // Too few points for the polynomial: fall back to the weighted mean.
      return yy.length === 0 ? Number.NaN : yy.reduce((s, v) => s + v, 0) / yy.length
    }
    return ols(X, m, p, yy).beta[0] ?? Number.NaN
  }
  const iterations = options.family === 'symmetric' ? (options.iterations ?? 4) : 0
  for (let it = 0; it < iterations; it++) {
    const fitted = Float64Array.from(xs, (v) => fitAt(v, robust))
    const resid = Float64Array.from(ys, (v, i) => Math.abs(v - (fitted[i] ?? 0)))
    const s = median(resid)
    robust = Float64Array.from(resid, (r) => {
      const u = s === 0 ? (r > 0 ? 1 : 0) : r / (6 * s)
      return u < 1 ? (1 - u * u) ** 2 : 0
    })
  }
  return vec(Array.from(points, (v) => fitAt(v, robust)))
}

/** A kernel for {@link nadarayaWatson} and {@link localLinear}. */
export type SmoothingKernel = 'gaussian' | 'epanechnikov' | 'uniform' | 'tricube'

function kernelWeight(kind: SmoothingKernel, u: number): number {
  const a = Math.abs(u)
  switch (kind) {
    case 'epanechnikov':
      return a < 1 ? 0.75 * (1 - u * u) : 0
    case 'uniform':
      return a < 1 ? 0.5 : 0
    case 'tricube':
      return a < 1 ? (70 / 81) * (1 - a ** 3) ** 3 : 0
    default:
      return Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI)
  }
}

/** Options of the kernel smoothers. */
export interface KernelSmootherOptions {
  /** The bandwidth (default Silverman's rule 1.06 sd n^(-1/5)). */
  readonly bandwidth?: number
  /** The kernel (default Gaussian). */
  readonly kernel?: SmoothingKernel
}

function bandwidthOf(xs: Float64Array, h: number | undefined): number {
  if (h !== undefined) {
    if (!(h > 0)) throw new MdmlError('value', 'kernel smoother: the bandwidth must be positive')
    return h
  }
  let m = 0
  for (const v of xs) m += v / xs.length
  let s = 0
  for (const v of xs) s += (v - m) ** 2
  return 1.06 * Math.sqrt(s / (xs.length - 1)) * xs.length ** -0.2
}

/**
 * The Nadaraya-Watson kernel regression estimate (statsmodels KernelReg with reg_type 'lc'): the kernel-weighted
 * mean of the responses near each point.
 *
 * @param x - The covariate.
 * @param y - The response.
 * @param at - Points to estimate at.
 * @param options - bandwidth and kernel.
 * @returns The estimates.
 * @example
 * ```ts
 * import { nadarayaWatson } from 'mdml-engine/stats/smoothing'
 * nadarayaWatson([0, 1, 2], [1, 1, 1], [0.5], { bandwidth: 1 }) // => [1]
 * ```
 */
export function nadarayaWatson(
  x: TensorLike,
  y: TensorLike,
  at: TensorLike,
  options: KernelSmootherOptions = {},
): Tensor {
  const [xs, ys] = paired('nadarayaWatson', x, y, 2)
  const h = bandwidthOf(xs, options.bandwidth)
  const kind = options.kernel ?? 'gaussian'
  return vec(
    toFlat(asTensor(at)).map((x0) => {
      let sw = 0
      let swy = 0
      for (let i = 0; i < xs.length; i++) {
        const w = kernelWeight(kind, ((xs[i] ?? 0) - x0) / h)
        sw += w
        swy += w * (ys[i] ?? 0)
      }
      return sw > 0 ? swy / sw : Number.NaN
    }),
  )
}

/**
 * Local linear kernel regression (statsmodels KernelReg with reg_type 'll'): the intercept of a kernel-weighted linear
 * fit at each point, which removes the boundary bias of Nadaraya-Watson.
 *
 * @param x - The covariate.
 * @param y - The response.
 * @param at - Points to estimate at.
 * @param options - bandwidth and kernel.
 * @returns The estimates.
 * @example
 * ```ts
 * import { localLinear } from 'mdml-engine/stats/smoothing'
 * localLinear([0, 1, 2, 3], [1, 3, 5, 7], [0, 1.5], { bandwidth: 1 }) // => [1, 4]
 * ```
 */
export function localLinear(x: TensorLike, y: TensorLike, at: TensorLike, options: KernelSmootherOptions = {}): Tensor {
  const [xs, ys] = paired('localLinear', x, y, 2)
  const h = bandwidthOf(xs, options.bandwidth)
  const kind = options.kernel ?? 'gaussian'
  return vec(
    toFlat(asTensor(at)).map((x0) => {
      let s0 = 0
      let s1 = 0
      let s2 = 0
      let t0 = 0
      let t1 = 0
      for (let i = 0; i < xs.length; i++) {
        const d = (xs[i] ?? 0) - x0
        const w = kernelWeight(kind, d / h)
        s0 += w
        s1 += w * d
        s2 += w * d * d
        t0 += w * (ys[i] ?? 0)
        t1 += w * d * (ys[i] ?? 0)
      }
      const det = s0 * s2 - s1 * s1
      return det !== 0 ? (s2 * t0 - s1 * t1) / det : s0 > 0 ? t0 / s0 : Number.NaN
    }),
  )
}

/** Options of the running statistics. */
export interface RunningOptions {
  /** Label each window at its centre (default true) rather than its right end, as pandas' rolling(center). */
  readonly center?: boolean
}

/**
 * A running (moving-window) statistic of a series (pandas Series.rolling(window, center).apply): NaN where the window
 * is incomplete.
 *
 * @param x - The series.
 * @param window - The window length.
 * @param statistic - The statistic of a window's values.
 * @param options - center (default true).
 * @returns The statistic at each position.
 * @example
 * ```ts
 * import { runningStatistic } from 'mdml-engine/stats/smoothing'
 * runningStatistic([1, 5, 2, 8, 3], 3, (w) => w.reduce((m, v) => Math.max(m, v), -Infinity)) // => [NaN, 5, 8, 8, NaN]
 * ```
 */
export function runningStatistic(
  x: TensorLike,
  window: number,
  statistic: (values: Float64Array) => number,
  options: RunningOptions = {},
): Tensor {
  const a = sample('runningStatistic', x, 1)
  if (!(Number.isInteger(window) && window >= 1))
    throw new MdmlError('value', 'running: the window must be a positive integer')
  const n = a.length
  const offset = options.center === false ? window - 1 : Math.floor(window / 2)
  return vec(
    Array.from({ length: n }, (_, i) => {
      const start = i - offset
      const end = start + window
      if (start < 0 || end > n) return Number.NaN
      return statistic(a.subarray(start, end))
    }),
  )
}

/**
 * The running mean of a series (pandas rolling mean).
 *
 * @param x - The series.
 * @param window - The window length.
 * @param options - center (default true).
 * @returns The running mean.
 * @example
 * ```ts
 * import { runningMean } from 'mdml-engine/stats/smoothing'
 * runningMean([1, 2, 3, 4, 5], 3) // => [NaN, 2, 3, 4, NaN]
 * ```
 */
export function runningMean(x: TensorLike, window: number, options: RunningOptions = {}): Tensor {
  return runningStatistic(x, window, (w) => w.reduce((s, v) => s + v, 0) / w.length, options)
}

/**
 * The running median of a series (pandas rolling median), a robust smoother.
 *
 * @param x - The series.
 * @param window - The window length.
 * @param options - center (default true).
 * @returns The running median.
 * @example
 * ```ts
 * import { runningMedian } from 'mdml-engine/stats/smoothing'
 * runningMedian([1, 9, 2, 3, 8], 3) // => [NaN, 2, 3, 3, NaN]
 * ```
 */
export function runningMedian(x: TensorLike, window: number, options: RunningOptions = {}): Tensor {
  return runningStatistic(x, window, (w) => median(w), options)
}

/**
 * A cubic smoothing-spline smoother (scipy.interpolate.make_smoothing_spline): the penalty lambda given, chosen to give
 * `df` effective degrees of freedom, or by generalised cross-validation.
 *
 * @param x - The covariate (sorted and distinct, at least five points).
 * @param y - The response.
 * @param options - lambda or df.
 * @returns The fitted spline with its penalty, GCV score and degrees of freedom.
 * @example
 * ```ts
 * import { smoothingSpline } from 'mdml-engine/stats/smoothing'
 * smoothingSpline([0, 1, 2, 3, 4, 5, 6], [0, 1, 0, 1, 0, 1, 0], { df: 3 }).degreesOfFreedom // => 3
 * ```
 */
export function smoothingSpline(
  x: TensorLike,
  y: TensorLike,
  options: { readonly lambda?: number; readonly df?: number; readonly weights?: readonly number[] } = {},
): SmoothingSpline {
  const w = options.weights === undefined ? {} : { w: options.weights }
  if (options.lambda !== undefined) return makeSmoothingSpline(x, y, { ...w, lam: options.lambda })
  if (options.df === undefined) return makeSmoothingSpline(x, y, w)
  const target = options.df
  const n = sample('smoothingSpline', x, 5).length
  if (!(target > 2 && target < n)) throw new MdmlError('value', 'smoothingSpline: df must lie in (2, n)')
  const f = (ll: number): number => makeSmoothingSpline(x, y, { ...w, lam: Math.exp(ll) }).degreesOfFreedom - target
  let lo = -30
  let hi = 30
  while (f(lo) < 0 && lo > -200) lo -= 20
  while (f(hi) > 0 && hi < 200) hi += 20
  const ll = brentq(f, lo, hi, { xtol: 1e-12 }).root
  return makeSmoothingSpline(x, y, { ...w, lam: Math.exp(ll) })
}
