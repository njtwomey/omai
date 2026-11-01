/**
 * State-space forms of ARMA models and the filters that serve them: the Harvey form of an ARMA(p, q) as a
 * `LinearGaussianModel` (so the library's Kalman filter, smoother and likelihood apply to it), a fast number-level
 * filter for the same form that evaluates the exact Gaussian likelihood with the variance concentrated out (the inner
 * loop of maximum-likelihood fits), and the h-step forecast of any linear Gaussian state-space form with one source of
 * noise, which also carries the forecasts of integrated (ARIMA) models.
 */

import { MdmlError } from '../base/index.ts'
import type { LinearGaussianModel } from '../dynamics/filtering/index.ts'
import { dgemm } from '../numerics/linalg/index.ts'

/** The Harvey form of an ARMA(p, q): x_(t+1) = T x_t + R e_(t+1), y_t = x_t[0], with r = max(p, q + 1) states. */
export interface ArmaForm {
  /** The number of states. */
  readonly r: number
  /** The autoregressive coefficients padded with zeros to length r (the first column of T). */
  readonly phi: Float64Array
  /** The loading R = [1, theta_1, ..., theta_(r-1)] of the innovation on the state. */
  readonly loading: Float64Array
  /** The stationary state covariance in units of the innovation variance, `r x r` row-major. */
  readonly p0: Float64Array
}

/** T x for the companion-like transition of the Harvey form. */
function applyT(phi: Float64Array, a: ArrayLike<number>, out: Float64Array): void {
  const r = phi.length
  const a0 = a[0] ?? 0
  for (let i = 0; i < r; i++) out[i] = (phi[i] ?? 0) * a0 + (i + 1 < r ? (a[i + 1] ?? 0) : 0)
}

/** T P T' + R R' (in place into `out`, `tmp` is scratch), O(r^2) by the structure of T. */
function propagate(
  phi: Float64Array,
  loading: Float64Array,
  p: Float64Array,
  tmp: Float64Array,
  out: Float64Array,
): void {
  const r = phi.length
  for (let i = 0; i < r; i++)
    for (let j = 0; j < r; j++)
      tmp[i * r + j] = (phi[i] ?? 0) * (p[j] ?? 0) + (i + 1 < r ? (p[(i + 1) * r + j] ?? 0) : 0)
  for (let i = 0; i < r; i++)
    for (let j = 0; j < r; j++)
      out[i * r + j] =
        (phi[j] ?? 0) * (tmp[i * r] ?? 0) +
        (j + 1 < r ? (tmp[i * r + j + 1] ?? 0) : 0) +
        (loading[i] ?? 0) * (loading[j] ?? 0)
}

/** The stationary covariance P = T P T' + R R' by the doubling algorithm (P_(k+1) = P_k + A_k P_k A_k', A_(k+1) = A_k^2). */
function stationaryCovariance(phi: Float64Array, loading: Float64Array): Float64Array {
  const r = phi.length
  let a: Float64Array = new Float64Array(r * r)
  for (let i = 0; i < r; i++) {
    a[i * r] = phi[i] ?? 0
    if (i + 1 < r) a[i * r + i + 1] = 1
  }
  const p = new Float64Array(r * r)
  for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) p[i * r + j] = (loading[i] ?? 0) * (loading[j] ?? 0)
  for (let it = 0; it < 80; it++) {
    const ap = dgemm(a, p, r, r, r)
    const add = dgemm(ap, a, r, r, r, { transposeB: true })
    let change = 0
    let scale = 0
    for (let i = 0; i < p.length; i++) {
      change += Math.abs(add[i] ?? 0)
      scale += Math.abs(p[i] ?? 0)
      p[i] = (p[i] ?? 0) + (add[i] ?? 0)
    }
    if (change <= 1e-16 * scale) break
    a = dgemm(a, a, r, r, r)
  }
  return p
}

/**
 * The Harvey state-space form of an ARMA(p, q) with x_t = sum phi_j x_(t-j) + e_t + sum theta_j e_(t-j), including the
 * stationary covariance of the initial state.
 *
 * @param ar - The autoregressive coefficients phi_1..phi_p (stationary).
 * @param ma - The moving-average coefficients theta_1..theta_q.
 * @returns The form.
 * @throws MdmlError with code `value` when the process is not stationary.
 * @example
 * ```ts
 * import { armaForm } from 'mdml-engine/timeseries'
 * // The AR(1) with phi = 0.5 has state variance 1 / (1 - 0.25).
 * armaForm([0.5], []).p0[0].toFixed(6) // => '1.333333'
 * ```
 */
export function armaForm(ar: ArrayLike<number>, ma: ArrayLike<number>): ArmaForm {
  const r = Math.max(ar.length, ma.length + 1)
  const phi = new Float64Array(r)
  for (let i = 0; i < ar.length; i++) phi[i] = ar[i] ?? 0
  const loading = new Float64Array(r)
  loading[0] = 1
  for (let i = 0; i < ma.length; i++) loading[i + 1] = ma[i] ?? 0
  const p0 = stationaryCovariance(phi, loading)
  if (!p0.every(Number.isFinite) || (p0[0] ?? 0) <= 0)
    throw new MdmlError('value', 'armaForm: the autoregressive polynomial is not stationary')
  return { r, phi, loading, p0 }
}

/**
 * An ARMA as a {@link LinearGaussianModel}: F the companion transition, H = [1, 0, ...], Q = sigma2 R R', no
 * observation noise and the stationary initial state, so that `kalmanLogLikelihood` gives the exact Gaussian
 * likelihood of the series.
 *
 * @param ar - The autoregressive coefficients.
 * @param ma - The moving-average coefficients.
 * @param sigma2 - The innovation variance (default 1).
 * @returns The model.
 * @example
 * ```ts
 * import { armaStateSpace } from 'mdml-engine/timeseries'
 * armaStateSpace([0.5], [0.4]).F // => [[0.5, 1], [0, 0]]
 * ```
 */
export function armaStateSpace(ar: ArrayLike<number>, ma: ArrayLike<number>, sigma2 = 1): LinearGaussianModel {
  const { r, phi, loading, p0 } = armaForm(ar, ma)
  const F = Array.from({ length: r }, (_, i) =>
    Array.from({ length: r }, (_, j) => (j === 0 ? (phi[i] ?? 0) : j === i + 1 ? 1 : 0)),
  )
  const Q = Array.from({ length: r }, (_, i) =>
    Array.from({ length: r }, (_, j) => sigma2 * (loading[i] ?? 0) * (loading[j] ?? 0)),
  )
  const P0 = Array.from({ length: r }, (_, i) => Array.from({ length: r }, (_, j) => sigma2 * (p0[i * r + j] ?? 0)))
  return {
    F,
    H: [Array.from({ length: r }, (_, j) => (j === 0 ? 1 : 0))],
    Q,
    R: [[0]],
    x0: new Array<number>(r).fill(0),
    P0,
  }
}

/** What {@link armaFilter} returns. */
export interface ArmaFilterResult {
  /** Observations used (the finite ones). */
  readonly nobs: number
  /** The sum of the log relative innovation variances, sum log f_t. */
  readonly sumLogF: number
  /** The sum of v_t^2 / f_t, so that the maximum-likelihood innovation variance is `sumSquares / nobs`. */
  readonly sumSquares: number
  /** One-step prediction errors v_t (NaN where the observation is missing). */
  readonly innovations: Float64Array
  /** Their variances relative to the innovation variance, f_t. */
  readonly relativeVariances: Float64Array
  /** The predicted state for the step after the last observation. */
  readonly state: Float64Array
  /** Its covariance in units of the innovation variance, `r x r`. */
  readonly covariance: Float64Array
}

/**
 * The Kalman filter of the Harvey form of an ARMA, specialised to scalar observations and no measurement noise: O(r^2)
 * per step. Returns the innovations and their relative variances, from which the exact Gaussian likelihood follows
 * with the innovation variance concentrated out, and the predicted state after the last observation for forecasting.
 * Missing observations (NaN) are skipped. It is the general `filtering.kalmanFilter` of `dynamics` specialised on purpose
 * (one observation row, no measurement noise, the shift structure of the companion matrix), so a fit's likelihood costs
 * O(r^2) per step instead of O(r^3); `armaStateSpace` gives the same model for the general filter.
 *
 * @param y - The series (a zero-mean ARMA).
 * @param ar - The autoregressive coefficients.
 * @param ma - The moving-average coefficients.
 * @returns The filter output.
 * @example
 * ```ts
 * import { armaFilter } from 'mdml-engine/timeseries'
 * // White noise: every prediction error is the observation itself with unit relative variance.
 * armaFilter([1, -2, 3], [], []).sumSquares // => 14
 * ```
 */
export function armaFilter(y: ArrayLike<number>, ar: ArrayLike<number>, ma: ArrayLike<number>): ArmaFilterResult {
  const form = armaForm(ar, ma)
  return filterForm(y, form)
}

/**
 * {@link armaFilter} for a prepared form (so a fit builds the form once per parameter vector).
 *
 * @param y - The series.
 * @param form - The form from {@link armaForm}.
 * @returns The filter output.
 * @example
 * ```ts
 * import { armaForm, filterForm } from 'mdml-engine/timeseries'
 * filterForm([1, -2, 3], armaForm([], [])).sumSquares // => 14
 * ```
 */
export function filterForm(y: ArrayLike<number>, form: ArmaForm): ArmaFilterResult {
  const { r, phi, loading } = form
  const n = y.length
  let a: Float64Array = new Float64Array(r)
  let p: Float64Array = Float64Array.from(form.p0)
  let a2: Float64Array = new Float64Array(r)
  let p2: Float64Array = new Float64Array(r * r)
  const tmp = new Float64Array(r * r)
  const col = new Float64Array(r)
  const innovations = new Float64Array(n)
  const rel = new Float64Array(n)
  let sumLogF = 0
  let sumSquares = 0
  let nobs = 0
  for (let t = 0; t < n; t++) {
    const yt = y[t] ?? Number.NaN
    if (Number.isFinite(yt)) {
      const v = yt - (a[0] ?? 0)
      const f = p[0] ?? 0
      if (!(f > 0)) {
        innovations.fill(Number.NaN, t)
        rel.fill(Number.NaN, t)
        return {
          nobs,
          sumLogF: Number.NaN,
          sumSquares: Number.NaN,
          innovations,
          relativeVariances: rel,
          state: a,
          covariance: p,
        }
      }
      innovations[t] = v
      rel[t] = f
      sumLogF += Math.log(f)
      sumSquares += (v * v) / f
      nobs += 1
      for (let i = 0; i < r; i++) col[i] = p[i * r] ?? 0
      for (let i = 0; i < r; i++) {
        a[i] = (a[i] ?? 0) + ((col[i] ?? 0) * v) / f
        for (let j = 0; j < r; j++) p[i * r + j] = (p[i * r + j] ?? 0) - ((col[i] ?? 0) * (col[j] ?? 0)) / f
      }
    } else {
      innovations[t] = Number.NaN
      rel[t] = Number.NaN
    }
    applyT(phi, a, a2)
    propagate(phi, loading, p, tmp, p2)
    ;[a, a2] = [a2, a]
    ;[p, p2] = [p2, p]
  }
  return { nobs, sumLogF, sumSquares, innovations, relativeVariances: rel, state: a, covariance: p }
}

/**
 * The exact Gaussian log-likelihood of a zero-mean ARMA series by the prediction-error decomposition of its state-space
 * form (equal to `kalmanLogLikelihood(armaStateSpace(...), y)`, in O(r^2) per step).
 *
 * @param y - The series.
 * @param ar - The autoregressive coefficients.
 * @param ma - The moving-average coefficients.
 * @param sigma2 - The innovation variance, or `undefined` to maximise over it (the concentrated likelihood).
 * @returns The log-likelihood.
 * @example
 * ```ts
 * import { armaLogLikelihood } from 'mdml-engine/timeseries'
 * armaLogLikelihood([0.5], [], [], 1).toFixed(6) // => '-1.043939'
 * ```
 */
export function armaLogLikelihood(
  y: ArrayLike<number>,
  ar: ArrayLike<number>,
  ma: ArrayLike<number>,
  sigma2?: number,
): number {
  const f = armaFilter(y, ar, ma)
  const n = f.nobs
  if (sigma2 === undefined) return -0.5 * (n * (Math.log(2 * Math.PI) + 1 + Math.log(f.sumSquares / n)) + f.sumLogF)
  return -0.5 * (n * (Math.log(2 * Math.PI) + Math.log(sigma2)) + f.sumLogF + f.sumSquares / sigma2)
}

/** The h-step forecast of a linear Gaussian state-space form: means and variances of the observation. */
export interface StateForecast {
  readonly mean: Float64Array
  readonly variance: Float64Array
}

/**
 * Forecast a linear Gaussian state-space form with one source of noise: the state follows s_(k+1) = A s_k + g e_(k+1)
 * with e of variance `sigma2`, the observation is y_k = h's_k, and the first state s_1 has mean `m` and covariance
 * `sigma2 P`. Gives the mean and variance of y_1..y_steps.
 *
 * @param A - The transition, `n x n` row-major.
 * @param g - The noise loading `[n]`.
 * @param h - The observation vector `[n]`.
 * @param m - The mean of the first state.
 * @param P - Its covariance in units of `sigma2`, `n x n` row-major.
 * @param sigma2 - The innovation variance.
 * @param steps - The number of steps.
 * @returns Means and variances.
 * @example
 * ```ts
 * import { forecastStateSpace } from 'mdml-engine/timeseries'
 * // A random walk from 3: the variance grows by sigma2 per step.
 * const f = forecastStateSpace([1], [1], [1], [3], [0], 2, 3)
 * [Array.from(f.mean), Array.from(f.variance)] // => [[3, 3, 3], [0, 2, 4]]
 * ```
 */
export function forecastStateSpace(
  A: ArrayLike<number>,
  g: ArrayLike<number>,
  h: ArrayLike<number>,
  m: ArrayLike<number>,
  P: ArrayLike<number>,
  sigma2: number,
  steps: number,
): StateForecast {
  const n = h.length
  const mean = new Float64Array(steps)
  const variance = new Float64Array(steps)
  let s: Float64Array = Float64Array.from(m)
  let p: Float64Array = Float64Array.from(P)
  for (let k = 0; k < steps; k++) {
    let mu = 0
    for (let i = 0; i < n; i++) mu += (h[i] ?? 0) * (s[i] ?? 0)
    mean[k] = mu
    const ph = dgemm(p, h, n, n, 1)
    let v = 0
    for (let i = 0; i < n; i++) v += (h[i] ?? 0) * (ph[i] ?? 0)
    variance[k] = Math.max(v, 0) * sigma2
    s = dgemm(A, s, n, n, 1)
    const ap = dgemm(dgemm(A, p, n, n, n), A, n, n, n, { transposeB: true })
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) ap[i * n + j] = (ap[i * n + j] ?? 0) + (g[i] ?? 0) * (g[j] ?? 0)
    p = ap
  }
  return { mean, variance }
}
