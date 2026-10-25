/**
 * Fractional Brownian motion and long-range dependence, where the Hurst exponent H sets how strongly increments are
 * correlated over long lags: fBm covariances, exact simulation of fractional Gaussian noise by the Davies-Harte
 * circulant embedding, and estimators of H from data by rescaled range (R/S), detrended fluctuation analysis and
 * aggregated variance (as nolds).
 */

import { WhiteboxError } from '../base/index.ts'
import { fft } from '../numerics/fourier/index.ts'
import { choleskyJitter } from '../numerics/linalg/index.ts'
import { polyfit } from '../numerics/polynomial/index.ts'
import { lgamma } from '../numerics/special/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import {
  abs,
  add,
  asTensor,
  concat,
  cumsum,
  expandDims,
  flip,
  fromStorage,
  matmul,
  maximum,
  mul,
  pow,
  promoteFloat,
  sliceAxis,
  sqrt,
  sub,
  type TensorLike,
  transpose,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { positiveInteger, primalNumbers } from './_shared.ts'
import type { CovarianceFunction } from './gaussian-processes.ts'
import type { Paths } from './paths.ts'

function checkHurst(h: number, where: string): void {
  if (!(h > 0 && h < 1)) throw new WhiteboxError('value', `${where}: the Hurst exponent must be in (0, 1)`)
}

/**
 * The covariance of fractional Brownian motion with Hurst exponent H, sigma^2 / 2 (|s|^{2H} + |t|^{2H} - |s - t|^{2H})
 * (Brownian motion for H = 1/2). Differentiable in H and sigma.
 *
 * @param hurst - H in (0, 1).
 * @param sigma - The scale (default 1).
 * @returns The covariance function (times on the first coordinate).
 * @concept fractional-brownian-motion
 * @example
 * ```ts
 * import { fbmCovariance } from 'whitebox-engine/stochastic'
 * import { item } from 'whitebox-engine/tensor'
 * item(fbmCovariance(0.5)([[1]], [[2]])) // => 1
 * ```
 */
export function fbmCovariance(hurst: TensorLike, sigma: TensorLike = 1): CovarianceFunction {
  const [h, s] = promoteFloat(hurst, sigma) as [Value, Value]
  const twoH = mul(h, 2)
  const power = (x: Value): Value => pow(abs(x), twoH)
  return (x, y) => {
    const a = sliceAxis(asTensor(x), 1, 0, 1)
    const b = transpose(sliceAxis(asTensor(y), 1, 0, 1))
    return mul(mul(mul(s, s), 0.5), sub(add(power(a), power(b)), power(sub(a, b))))
  }
}

/**
 * The autocovariance of unit fractional Gaussian noise (the increments of fBm on a unit grid),
 * gamma(k) = (|k + 1|^{2H} - 2 |k|^{2H} + |k - 1|^{2H}) / 2: zero for k != 0 when H = 1/2, positive and decaying like
 * H (2H - 1) k^{2H - 2} (not summable: long-range dependence) when H > 1/2, negative when H < 1/2. Differentiable.
 *
 * @param k - Lags.
 * @param hurst - H in (0, 1).
 * @returns gamma(k).
 * @concept long-range-dependence
 * @example
 * ```ts
 * import { fgnAutocovariance } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(fgnAutocovariance([0, 1, 2], 0.5)) // => [1, 0, 0]
 * ```
 */
export function fgnAutocovariance(k: TensorLike, hurst: TensorLike): Value {
  const [kk, h] = promoteFloat(k, hurst) as [Value, Value]
  const twoH = mul(h, 2)
  const p = (x: Value): Value => pow(abs(x), twoH)
  return mul(add(sub(p(add(kk, 1)), mul(p(kk), 2)), p(sub(kk, 1))), 0.5)
}

/** Options of {@link fractionalGaussianNoise}. */
export interface FgnOptions {
  /** The number of independent series (default 1). */
  readonly paths?: number
  /** `daviesHarte` (default; circulant embedding by the FFT, O(n log n)) or `cholesky` (exact, O(n^3)). */
  readonly method?: 'daviesHarte' | 'cholesky'
}

/**
 * Fractional Gaussian noise: a stationary Gaussian series with the fGn autocovariance, by the Davies-Harte method
 * (Wood and Chan's form): embed the covariance in a circulant of size 2n, whose eigenvalues are the FFT of its first
 * row, and transform complex white noise scaled by their square roots. When an eigenvalue is negative (not for fGn, but
 * kept as a guard) the Cholesky method is used. Differentiable in H (the FFT and the scaling are tensor operations).
 *
 * @param key - The key.
 * @param n - The length.
 * @param hurst - H in (0, 1).
 * @param options - Paths and method.
 * @returns `[paths, n]` with unit variance.
 * @concept fractional-brownian-motion
 * @reference R. B. Davies and D. S. Harte, Tests for Hurst effect, Biometrika 74 (1987); A. T. A. Wood and G. Chan,
 *   Simulation of stationary Gaussian processes in [0, 1]^d, J. Comput. Graph. Stat. 3 (1994).
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { fractionalGaussianNoise } from 'whitebox-engine/stochastic'
 * fractionalGaussianNoise(key(0), 100, 0.7, { paths: 3 }).shape // => [3, 100]
 * ```
 */
export function fractionalGaussianNoise(key: Key, n: number, hurst: TensorLike, options: FgnOptions = {}): Value {
  positiveInteger(n, 'n', 'fractionalGaussianNoise')
  const [h] = promoteFloat(hurst) as [Value]
  checkHurst(primalNumbers(h)[0] ?? Number.NaN, 'fractionalGaussianNoise')
  const paths = positiveInteger(options.paths ?? 1, 'paths', 'fractionalGaussianNoise')
  const lags = fromStorage(
    Float64Array.from({ length: n }, (_, i) => i),
    [n],
    'float64',
  )
  const gamma = fgnAutocovariance(lags, h)
  if (options.method !== 'cholesky' && n > 1) {
    // First row of the circulant: gamma(0), ..., gamma(n - 1), gamma(n), gamma(n - 1), ..., gamma(1).
    const top = fgnAutocovariance(n, h)
    const row = concat([gamma, expandDims(top, 0), flip(sliceAxis(gamma, 0, 1, n), 0)], 0)
    const m = 2 * n
    const lambda = fft(row).re
    const eig = primalNumbers(lambda)
    let ok = true
    for (const v of eig) if (v < -1e-10 * Math.max(1, Math.abs(eig[0] ?? 1))) ok = false
    if (ok) {
      const scale = sqrt(mul(maximum(lambda, 0), 1 / m))
      const re = mul(normal(foldIn(key, 0), [paths, m]), scale)
      const im = mul(normal(foldIn(key, 1), [paths, m]), scale)
      const y = fft({ re, im }).re
      return sliceAxis(y, 1, 0, n)
    }
  }
  const idx = fromStorage(
    Float64Array.from({ length: n }, (_, i) => i),
    [n, 1],
    'float64',
  )
  const diff = sub(idx, transpose(idx))
  const cov = fgnAutocovariance(diff, h)
  const { l } = choleskyJitter(cov)
  return matmul(normal(foldIn(key, 2), [paths, n]), transpose(l))
}

/**
 * Fractional Brownian motion B_H on an equally spaced grid from 0: the cumulative sum of fractional Gaussian noise
 * scaled by dt^H, so Var B_H(t) = sigma^2 t^{2H} and the increments are stationary and self-similar. Differentiable in H
 * and sigma.
 *
 * @param key - The key.
 * @param horizon - The end time T.
 * @param steps - The number of steps.
 * @param hurst - H in (0, 1).
 * @param options - Paths, the scale sigma (default 1) and the method.
 * @returns Paths `[paths, steps + 1]` starting at 0.
 * @concept fractional-brownian-motion
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { fractionalBrownianMotion } from 'whitebox-engine/stochastic'
 * fractionalBrownianMotion(key(0), 1, 64, 0.3, { paths: 2 }).values.shape // => [2, 65]
 * ```
 */
export function fractionalBrownianMotion(
  key: Key,
  horizon: number,
  steps: number,
  hurst: TensorLike,
  options: FgnOptions & { readonly sigma?: TensorLike } = {},
): Paths {
  if (!(horizon > 0)) throw new WhiteboxError('value', 'fractionalBrownianMotion: the horizon must be positive')
  positiveInteger(steps, 'steps', 'fractionalBrownianMotion')
  const [h, sigma] = promoteFloat(hurst, options.sigma ?? 1) as [Value, Value]
  const noise = fractionalGaussianNoise(key, steps, h, options)
  const dt = horizon / steps
  const scaled = mul(mul(noise, pow(dt, h)), sigma)
  const paths = noise.shape[0] ?? 1
  const times = Float64Array.from({ length: steps + 1 }, (_, i) => (i === steps ? horizon : i * dt))
  return { times, values: concat([zeros([paths, 1]), cumsum(scaled, 1)], 1) }
}

function slopeOf(x: readonly number[], y: readonly number[]): number {
  const c = primalNumbers(polyfit(x, y, 1))
  return c[1] ?? Number.NaN
}

function series(x: TensorLike, where: string): number[] {
  const v = primalNumbers(x)
  if (v.length < 8) throw new WhiteboxError('value', `${where}: needs at least 8 observations`)
  return v
}

function windowsOf(windows: readonly number[] | undefined, n: number, where: string): number[] {
  const w = windows ?? defaultWindows(n)
  if (w.length < 2) throw new WhiteboxError('value', `${where}: needs at least two window sizes`)
  for (const s of w) {
    if (!(Number.isInteger(s) && s >= 2 && s < n)) throw new WhiteboxError('value', `${where}: bad window size ${s}`)
  }
  return [...w]
}

function defaultWindows(n: number): number[] {
  // logarithmically spaced sizes from 8 to n / 4
  const out = new Set<number>()
  const lo = Math.log(8)
  const hi = Math.log(Math.max(9, Math.floor(n / 4)))
  for (let i = 0; i < 12; i++) out.add(Math.round(Math.exp(lo + ((hi - lo) * i) / 11)))
  return [...out].filter((w) => w >= 2 && w < n)
}

/** The expected R/S of white noise for a window of n (Anis and Lloyd, corrected by Peters). */
function expectedRs(n: number): number {
  const front = (n - 0.5) / n
  let back = 0
  for (let i = 1; i < n; i++) back += Math.sqrt((n - i) / i)
  const middle =
    n <= 340
      ? Math.exp((primalNumbers(lgamma((n - 1) / 2))[0] ?? 0) - (primalNumbers(lgamma(n / 2))[0] ?? 0)) /
        Math.sqrt(Math.PI)
      : 1 / Math.sqrt((n * Math.PI) / 2)
  return front * middle * back
}

function rescaledRange(x: readonly number[], n: number, unbiased: boolean): number {
  const m = Math.floor(x.length / n)
  let total = 0
  let count = 0
  for (let s = 0; s < m; s++) {
    let mean = 0
    for (let i = 0; i < n; i++) mean += x[s * n + i] ?? 0
    mean /= n
    let walk = 0
    let hi = Number.NEGATIVE_INFINITY
    let lo = Number.POSITIVE_INFINITY
    let ss = 0
    for (let i = 0; i < n; i++) {
      const d = (x[s * n + i] ?? 0) - mean
      walk += d
      hi = Math.max(hi, walk)
      lo = Math.min(lo, walk)
      ss += d * d
    }
    const range = hi - lo
    const sd = Math.sqrt(ss / (unbiased ? n - 1 : n))
    if (range !== 0) {
      total += range / sd
      count += 1
    }
  }
  return count === 0 ? Number.NaN : total / count
}

/**
 * The Hurst exponent by rescaled range (R/S) analysis (as nolds.hurst_rs with a least-squares fit): the slope of
 * log (R/S)_n against log n over window sizes n, with the Anis-Lloyd-Peters correction (the slope of the excess over
 * white noise's expected R/S, plus 1/2) by default.
 *
 * @param x - The series (an increment series such as fractional Gaussian noise).
 * @param options - Window sizes, `corrected` (default true) and `unbiased` standard deviations (default true).
 * @returns The estimate of H.
 * @concept hurst-exponent
 * @reference H. E. Hurst, Long-term storage capacity of reservoirs (1951); A. A. Anis and E. H. Lloyd (1976).
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { fractionalGaussianNoise, hurstRs } from 'whitebox-engine/stochastic'
 * Math.abs(hurstRs(fractionalGaussianNoise(key(0), 4096, 0.5)) - 0.5) < 0.1 // => true
 * ```
 */
export function hurstRs(
  x: TensorLike,
  options: { readonly windows?: readonly number[]; readonly corrected?: boolean; readonly unbiased?: boolean } = {},
): number {
  const data = series(x, 'hurstRs')
  const windows = windowsOf(options.windows, data.length, 'hurstRs')
  const corrected = options.corrected ?? true
  const logN: number[] = []
  const logRs: number[] = []
  for (const n of windows) {
    const rs = rescaledRange(data, n, options.unbiased ?? true)
    if (!Number.isFinite(rs)) continue
    logN.push(Math.log(n))
    logRs.push(Math.log(rs) - (corrected ? Math.log(expectedRs(n)) : 0))
  }
  if (logN.length < 2) return Number.NaN
  const slope = slopeOf(logN, logRs)
  return corrected ? slope + 0.5 : slope
}

/**
 * Detrended fluctuation analysis (Peng et al. 1994, as nolds.dfa with least-squares fits): integrate the centred
 * series, remove a polynomial trend of the given order in (half-overlapping) windows of size n, and fit the slope alpha
 * of log F(n) against log n, where F(n) is the root mean square residual. For fractional Gaussian noise alpha
 * estimates H; for fractional Brownian motion, H + 1.
 *
 * @param x - The series.
 * @param options - Window sizes, `overlap` (default true) and the detrending `order` (default 1).
 * @returns alpha.
 * @concept hurst-exponent
 * @reference C.-K. Peng et al., Mosaic organization of DNA nucleotides, Phys. Rev. E 49 (1994).
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { detrendedFluctuation, fractionalGaussianNoise } from 'whitebox-engine/stochastic'
 * Math.abs(detrendedFluctuation(fractionalGaussianNoise(key(1), 4096, 0.8)) - 0.8) < 0.1 // => true
 * ```
 */
export function detrendedFluctuation(
  x: TensorLike,
  options: { readonly windows?: readonly number[]; readonly overlap?: boolean; readonly order?: number } = {},
): number {
  const data = series(x, 'detrendedFluctuation')
  const n = data.length
  const windows = windowsOf(options.windows, n, 'detrendedFluctuation')
  const order = options.order ?? 1
  if (!(Number.isInteger(order) && order >= 0)) throw new WhiteboxError('value', 'detrendedFluctuation: bad order')
  const mean = data.reduce((a, b) => a + b, 0) / n
  const walk: number[] = []
  let acc = 0
  for (const v of data) {
    acc += v - mean
    walk.push(acc)
  }
  const logN: number[] = []
  const logF: number[] = []
  for (const w of windows) {
    const starts: number[] = []
    if (options.overlap ?? true) for (let i = 0; i < n - w; i += Math.floor(w / 2)) starts.push(i)
    else for (let i = 0; i + w <= n; i += w) starts.push(i)
    if (starts.length === 0) continue
    // every window's trend at once: polyfit with one column per window
    const cols = new Float64Array(w * starts.length)
    for (const [c, s] of starts.entries()) for (let i = 0; i < w; i++) cols[i * starts.length + c] = walk[s + i] ?? 0
    const xs = Array.from({ length: w }, (_, i) => i)
    const y = fromStorage(cols, [w, starts.length], 'float64')
    const coef = primalNumbers(polyfit(xs, y, order))
    const k = starts.length
    const trend = new Float64Array(w * k)
    for (let i = 0; i < w; i++) {
      for (let c = 0; c < k; c++) {
        let v = 0
        for (let d = order; d >= 0; d--) v = v * i + (coef[d * k + c] ?? 0)
        trend[i * k + c] = v
      }
    }
    let total = 0
    for (const [c] of starts.entries()) {
      let s = 0
      for (let i = 0; i < w; i++) s += ((cols[i * starts.length + c] ?? 0) - (trend[i * starts.length + c] ?? 0)) ** 2
      total += s / w
    }
    const f = Math.sqrt(total / starts.length)
    if (f > 0) {
      logN.push(Math.log(w))
      logF.push(Math.log(f))
    }
  }
  return logN.length < 2 ? Number.NaN : slopeOf(logN, logF)
}

/**
 * The Hurst exponent by the aggregated variance method: the variance of block means of size m of a stationary series
 * scales like m^{2H - 2}, so H = 1 + slope / 2 of log variance against log m.
 *
 * @param x - The series (increments, such as fractional Gaussian noise).
 * @param options - Block sizes.
 * @returns The estimate of H.
 * @concept hurst-exponent
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { aggregatedVarianceHurst, fractionalGaussianNoise } from 'whitebox-engine/stochastic'
 * Math.abs(aggregatedVarianceHurst(fractionalGaussianNoise(key(2), 4096, 0.7)) - 0.7) < 0.1 // => true
 * ```
 */
export function aggregatedVarianceHurst(x: TensorLike, options: { readonly windows?: readonly number[] } = {}): number {
  const data = series(x, 'aggregatedVarianceHurst')
  const windows = windowsOf(options.windows, data.length, 'aggregatedVarianceHurst')
  const logM: number[] = []
  const logV: number[] = []
  for (const m of windows) {
    const k = Math.floor(data.length / m)
    if (k < 2) continue
    const means: number[] = []
    for (let b = 0; b < k; b++) {
      let s = 0
      for (let i = 0; i < m; i++) s += data[b * m + i] ?? 0
      means.push(s / m)
    }
    const mu = means.reduce((a, b) => a + b, 0) / k
    const v = means.reduce((a, b) => a + (b - mu) ** 2, 0) / (k - 1)
    if (v > 0) {
      logM.push(Math.log(m))
      logV.push(Math.log(v))
    }
  }
  return logM.length < 2 ? Number.NaN : 1 + slopeOf(logM, logV) / 2
}

/** The long-range dependence summary of {@link longRangeDependence}. */
export interface LongRangeDependence {
  /** Whether the autocorrelations are not summable (H > 1/2). */
  readonly longMemory: boolean
  /** The decay exponent 2H - 2 of the autocorrelation gamma(k) ~ c k^{2H - 2}. */
  readonly exponent: number
  /** The constant c = H (2H - 1). */
  readonly coefficient: number
  /** The partial sums sum_{k=-K}^{K} gamma(k) for K = 1, 10, 100, 1000 (diverging for long memory). */
  readonly partialSums: readonly number[]
}

/**
 * Long-range dependence of fractional Gaussian noise with Hurst exponent H: its autocorrelations decay like
 * H (2H - 1) k^{2H - 2}, too slowly to be summable when H > 1/2 (long memory: the variance of a mean of n terms falls
 * like n^{2H - 2} instead of 1 / n), and sum to zero over all lags when H < 1/2 (antipersistence).
 *
 * @param hurst - H in (0, 1).
 * @returns The asymptotics and partial sums.
 * @concept long-range-dependence
 * @example
 * ```ts
 * import { longRangeDependence } from 'whitebox-engine/stochastic'
 * longRangeDependence(0.75).longMemory // => true
 * longRangeDependence(0.5).partialSums // => [1, 1, 1, 1]
 * ```
 */
export function longRangeDependence(hurst: number): LongRangeDependence {
  checkHurst(hurst, 'longRangeDependence')
  const partialSums = [1, 10, 100, 1000].map((kmax) => {
    // gamma telescopes: sum_{k=-K}^{K} gamma(k) = (K + 1)^{2H} - K^{2H}
    return (kmax + 1) ** (2 * hurst) - kmax ** (2 * hurst)
  })
  return {
    longMemory: hurst > 0.5,
    exponent: 2 * hurst - 2,
    coefficient: hurst * (2 * hurst - 1),
    partialSums,
  }
}
