/**
 * Kernel density estimation, the smooth estimate of a density by placing a kernel at every data point: Gaussian KDE in
 * any dimension (scipy.stats.gaussian_kde) with optional weights and boundary correction, and bandwidth selection by
 * Scott's and Silverman's rules or by likelihood and least-squares cross-validation (as statsmodels).
 */

import { MdmlError, varianceOf } from '../base/index.ts'
import { dpotrf } from '../numerics/linalg/index.ts'
import { categorical, foldIn, type Key, normal as normalNoise } from '../random/index.ts'
import {
  div,
  exp,
  fromStorage,
  item,
  log,
  mul,
  neg,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'
import { maximiseSmooth, slopeOf } from './_optimise.ts'
import { mat, normCdf, rowsOf, sample, vec } from './_util.ts'
import { iqr } from './quantiles.ts'

/**
 * Scott's bandwidth factor neff^(-1 / (d + 4)) (scipy.stats.gaussian_kde.scotts_factor).
 *
 * @param n - The (effective) sample size.
 * @param d - The dimension (default 1).
 * @returns The factor multiplying the data's standard deviation.
 * @example
 * ```ts
 * import { scottFactor } from 'mdml-engine/stats'
 * scottFactor(32) // => 0.5
 * ```
 */
export function scottFactor(n: number, d = 1): number {
  return n ** (-1 / (d + 4))
}

/**
 * Silverman's bandwidth factor (n (d + 2) / 4)^(-1 / (d + 4)) (scipy.stats.gaussian_kde.silverman_factor).
 *
 * @param n - The (effective) sample size.
 * @param d - The dimension (default 1).
 * @returns The factor.
 * @example
 * ```ts
 * import { silvermanFactor } from 'mdml-engine/stats'
 * silvermanFactor(4 / 3) // => 1
 * ```
 */
export function silvermanFactor(n: number, d = 1): number {
  return ((n * (d + 2)) / 4) ** (-1 / (d + 4))
}

function spread(x: Float64Array): number {
  const sd = Math.sqrt(varianceOf(x, 1))
  const r = iqr(x) / 1.349
  return r > 0 ? Math.min(sd, r) : sd
}

/**
 * Scott's rule-of-thumb bandwidth for a univariate Gaussian kernel, 1.059 A n^(-1/5) with A = min(sd, IQR / 1.349)
 * (statsmodels.nonparametric.bandwidths.bw_scott).
 *
 * @param x - The sample.
 * @returns The bandwidth.
 * @example
 * ```ts
 * import { bandwidthScott } from 'mdml-engine/stats'
 * bandwidthScott([1, 2, 3, 4, 5]) // => 1.137942
 * ```
 */
export function bandwidthScott(x: TensorLike): number {
  const a = sample('bandwidthScott', x, 2)
  return 1.059 * spread(a) * a.length ** -0.2
}

/**
 * Silverman's rule-of-thumb bandwidth, 0.9 A n^(-1/5) (statsmodels.nonparametric.bandwidths.bw_silverman).
 *
 * @param x - The sample.
 * @returns The bandwidth.
 * @example
 * ```ts
 * import { bandwidthSilverman } from 'mdml-engine/stats'
 * bandwidthSilverman([1, 2, 3, 4, 5]) // => 0.967089
 * ```
 */
export function bandwidthSilverman(x: TensorLike): number {
  const a = sample('bandwidthSilverman', x, 2)
  return 0.9 * spread(a) * a.length ** -0.2
}

/** A cross-validation criterion for a univariate Gaussian kernel bandwidth. */
export type BandwidthCriterion = 'likelihood' | 'least-squares'

/**
 * The cross-validation criterion of a univariate Gaussian kernel density estimate at a bandwidth, to be maximised:
 * the leave-one-out log-likelihood sum log f_(-i)(x_i), or minus the least-squares criterion (the integral of f^2
 * minus 2 / n sum f_(-i)(x_i)).
 *
 * @param x - The sample.
 * @param bandwidth - The bandwidth h > 0.
 * @param options - method (default `'likelihood'`).
 * @returns The criterion.
 * @example
 * ```ts
 * import { bandwidthCriterion } from 'mdml-engine/stats'
 * bandwidthCriterion([0, 1], 1) // => -2.837877
 * ```
 */
export function bandwidthCriterion(
  x: TensorLike,
  bandwidth: number,
  options: { readonly method?: BandwidthCriterion } = {},
): number {
  const a = sample('bandwidthCriterion', x, 2)
  return item(criterion(a, options.method ?? 'likelihood')(vec([Math.log(bandwidth)])))
}

// The criterion as a tensor function of log h, so that its derivative comes from reverse-mode differentiation.
function criterion(a: Float64Array, method: BandwidthCriterion): (lh: Value) => Value {
  const n = a.length
  const d2 = new Float64Array(n * n)
  const mask = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      d2[i * n + j] = ((a[i] ?? 0) - (a[j] ?? 0)) ** 2
      mask[i * n + j] = i === j ? 0 : 1
    }
  const D = mat(d2, n, n)
  const M = mat(mask, n, n)
  const kernel = (lh: Value, scale: number): Value => exp(mul(neg(D), mul(scale, exp(mul(lh, -2)))))
  if (method === 'likelihood')
    return (lh) => {
      const rows = sum(mul(kernel(lh, 0.5), M), 1)
      return sum(sub(log(rows), sum(log(mul(exp(lh), (n - 1) * Math.sqrt(2 * Math.PI))))))
    }
  return (lh) => {
    const h = exp(lh)
    const isq = mul(sum(kernel(lh, 0.25)), 1 / (n * n * Math.sqrt(4 * Math.PI)))
    const loo = mul(sum(mul(kernel(lh, 0.5), M)), 2 / (n * (n - 1) * Math.sqrt(2 * Math.PI)))
    return sum(div(sub(loo, isq), h))
  }
}

/**
 * A univariate Gaussian kernel bandwidth by cross-validation: maximum leave-one-out likelihood (`'likelihood'`, as
 * statsmodels' cv_ml) or least-squares cross-validation (`'least-squares'`, cv_ls, minimising the estimated integrated
 * squared error); searched on a log scale within a factor of ten of Silverman's bandwidth and refined at a root of the
 * criterion's derivative.
 *
 * @param x - The sample.
 * @param options - method (default `'likelihood'`).
 * @returns The bandwidth.
 * @example
 * ```ts
 * import { bandwidthCrossValidation } from 'mdml-engine/stats'
 * bandwidthCrossValidation([0, 0.5, 1.1, 1.4, 2.2, 2.3, 3.1, 4]) > 0 // => true
 * ```
 */
export function bandwidthCrossValidation(
  x: TensorLike,
  options: { readonly method?: BandwidthCriterion } = {},
): number {
  const a = sample('bandwidthCrossValidation', x, 3)
  const f = criterion(a, options.method ?? 'likelihood')
  const value = (lh: number): number => item(f(vec([lh])))
  const h0 = Math.log(bandwidthSilverman(a))
  return Math.exp(maximiseSmooth(value, slopeOf(f), h0 - Math.LN10, h0 + Math.LN10, 80))
}

/** A boundary correction for a univariate density on [lower, upper]. */
export interface KdeBoundary {
  readonly lower?: number
  readonly upper?: number
  /**
   * `'reflection'` (default) adds the mirror images of the kernels across each boundary; `'renormalisation'` divides by
   * the kernel mass inside the support (Jones 1993).
   */
  readonly method?: 'reflection' | 'renormalisation'
}

/** Options of {@link gaussianKde}. */
export interface KdeOptions {
  /**
   * `'scott'` (default) or `'silverman'` factors, a numeric factor, or (univariate only) a cross-validated bandwidth
   * `'cv-likelihood'` or `'cv-least-squares'`.
   */
  readonly bandwidth?: 'scott' | 'silverman' | 'cv-likelihood' | 'cv-least-squares' | number
  /** Observation weights (normalised to sum to one). */
  readonly weights?: TensorLike
  /** Boundary correction (univariate only). */
  readonly boundary?: KdeBoundary
}

/** A fitted Gaussian kernel density estimate. */
export interface GaussianKde {
  readonly d: number
  readonly n: number
  /** Effective sample size (sum w)^2 / sum w^2. */
  readonly neff: number
  /** The bandwidth factor multiplying the data covariance's square root. */
  readonly factor: number
  /** The kernel covariance: the (weighted) data covariance times factor^2. */
  readonly covariance: Tensor
  /** Density at points `[M]` (univariate) or `[M, D]`. */
  evaluate(points: TensorLike): Tensor
  /** Log density at points. */
  logpdf(points: TensorLike): Tensor
  /** Draw `size` points from the estimate (without boundary correction). */
  resample(key: Key, size: number): Tensor
}

/**
 * Gaussian kernel density estimation (scipy.stats.gaussian_kde): a weighted mixture of Gaussians at the data with
 * covariance factor^2 times the data covariance, the factor by Scott's or Silverman's rule (or cross-validation for one
 * dimension), with optional boundary correction by reflection or renormalisation.
 *
 * @param data - Observations `[N]` or `[N, D]`.
 * @param options - bandwidth, weights and boundary.
 * @returns The estimate.
 * @index function Gaussian kernel density estimate with bandwidth rules.
 * @example
 * ```ts
 * import { gaussianKde } from 'mdml-engine/stats'
 * const kde = gaussianKde([-1, 0, 1])
 * kde.evaluate([0]) // => [0.318156]
 * ```
 */
export function gaussianKde(data: TensorLike, options: KdeOptions = {}): GaussianKde {
  const m = rowsOf('gaussianKde', data)
  const n = m.rows
  const d = m.cols
  if (n < 2) throw new MdmlError('value', 'gaussianKde: needs at least two observations')
  const w = options.weights === undefined ? new Float64Array(n).fill(1 / n) : sample('gaussianKde', options.weights)
  if (w.length !== n) throw new MdmlError('shape', 'gaussianKde: weights must match the data')
  let ws = 0
  for (const v of w) ws += v
  for (let i = 0; i < n; i++) w[i] = (w[i] ?? 0) / ws
  let w2 = 0
  for (const v of w) w2 += v * v
  const neff = 1 / w2
  // The weighted data covariance (numpy.cov with aweights, ddof 1).
  const mu = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mu[j] = (mu[j] ?? 0) + (w[i] ?? 0) * (m.data[i * d + j] ?? 0)
  const C = new Float64Array(d * d)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < d; a++)
      for (let b = 0; b < d; b++)
        C[a * d + b] =
          (C[a * d + b] ?? 0) +
          (w[i] ?? 0) * ((m.data[i * d + a] ?? 0) - (mu[a] ?? 0)) * ((m.data[i * d + b] ?? 0) - (mu[b] ?? 0))
  for (let k = 0; k < d * d; k++) C[k] = (C[k] ?? 0) / (1 - w2)
  const bw = options.bandwidth ?? 'scott'
  let factor: number
  if (typeof bw === 'number') factor = bw
  else if (bw === 'scott') factor = scottFactor(neff, d)
  else if (bw === 'silverman') factor = silvermanFactor(neff, d)
  else {
    if (d !== 1) throw new MdmlError('value', 'gaussianKde: cross-validated bandwidths are univariate')
    const h = bandwidthCrossValidation(m.data, { method: bw === 'cv-likelihood' ? 'likelihood' : 'least-squares' })
    factor = h / Math.sqrt(C[0] ?? 1)
  }
  if (!(factor > 0)) throw new MdmlError('value', 'gaussianKde: the bandwidth factor must be positive')
  const cov = Float64Array.from(C, (v) => v * factor * factor)
  const { factor: U, info } = dpotrf(cov, d)
  if (info !== 0) throw new MdmlError('singular', 'gaussianKde: the data covariance is singular')
  // Lower Cholesky factor L = U^T; log normaliser log((2 pi)^(d/2) det L).
  let logNorm = (d / 2) * Math.log(2 * Math.PI)
  for (let i = 0; i < d; i++) logNorm += Math.log(U[i * d + i] ?? 1)
  const boundary = options.boundary
  if (boundary !== undefined && d !== 1) throw new MdmlError('value', 'gaussianKde: boundary correction is univariate')
  const h = Math.sqrt(cov[0] ?? 0)
  // Squared Mahalanobis distance of z to x_i by forward substitution with L.
  const mahalanobis = (z: ArrayLike<number>, i: number): number => {
    const y = new Float64Array(d)
    let s = 0
    for (let r = 0; r < d; r++) {
      let v = (z[r] ?? 0) - (m.data[i * d + r] ?? 0)
      for (let c = 0; c < r; c++) v -= (U[c * d + r] ?? 0) * (y[c] ?? 0)
      const yr = v / (U[r * d + r] ?? 1)
      y[r] = yr
      s += yr * yr
    }
    return s
  }
  const raw = (z: ArrayLike<number>): number => {
    let top = Number.NEGATIVE_INFINITY
    const terms = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      const t = Math.log(w[i] ?? 0) - 0.5 * mahalanobis(z, i)
      terms[i] = t
      if (t > top) top = t
    }
    if (top === Number.NEGATIVE_INFINITY) return 0
    let s = 0
    for (const t of terms) s += Math.exp(t - top)
    return Math.exp(top + Math.log(s) - logNorm)
  }
  const density = (z: ArrayLike<number>): number => {
    if (boundary === undefined) return raw(z)
    const x0 = z[0] ?? 0
    const lo = boundary.lower ?? Number.NEGATIVE_INFINITY
    const hi = boundary.upper ?? Number.POSITIVE_INFINITY
    if (x0 < lo || x0 > hi) return 0
    if ((boundary.method ?? 'reflection') === 'reflection') {
      let f = raw([x0])
      if (Number.isFinite(lo)) f += raw([2 * lo - x0])
      if (Number.isFinite(hi)) f += raw([2 * hi - x0])
      return f
    }
    // Renormalisation: divide by the mass of the kernel centred at x that lies inside the support.
    return raw([x0]) / (normCdf((x0 - lo) / h) - normCdf((x0 - hi) / h))
  }
  const pointsOf = (points: TensorLike): Float64Array[] => {
    const p = rowsOf('gaussianKde', points)
    if (p.cols !== d) throw new MdmlError('shape', `gaussianKde: points must have dimension ${d}`)
    return Array.from({ length: p.rows }, (_, i) => p.data.subarray(i * d, (i + 1) * d))
  }
  return {
    d,
    n,
    neff,
    factor,
    covariance: mat(cov, d, d),
    evaluate: (points) => vec(pointsOf(points).map(density)),
    logpdf: (points) => vec(pointsOf(points).map((z) => Math.log(density(z)))),
    resample: (key, size) => {
      const idx = categorical(foldIn(key, 0), Array.from(w), [size]).data
      const eps = normalNoise(foldIn(key, 1), [size, d]).data
      const out = new Float64Array(size * d)
      for (let s = 0; s < size; s++)
        for (let r = 0; r < d; r++) {
          let v = m.data[(idx[s] ?? 0) * d + r] ?? 0
          for (let c = 0; c <= r; c++) v += (U[c * d + r] ?? 0) * (eps[s * d + c] ?? 0)
          out[s * d + r] = v
        }
      return d === 1 ? vec(out) : fromStorage(out, [size, d], 'float64')
    },
  }
}
