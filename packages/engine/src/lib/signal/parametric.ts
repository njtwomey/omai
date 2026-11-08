/**
 * Parametric spectral estimation, which fits a model and reads the spectrum from it: autoregressive models by the
 * Levinson-Durbin recursion, Yule-Walker and Burg (as statsmodels) with their power spectral density, the MUSIC and
 * ESPRIT subspace estimators of sinusoid frequencies, and Prony's method for sums of damped complex exponentials.
 */

import { OmaiError } from '../base/index.ts'
import { complexOps as C } from '../numerics/differentiate/index.ts'
import { eigh, eigvals, lstsq, pinv, solveToeplitz } from '../numerics/linalg/index.ts'
import { polyroots } from '../numerics/polynomial/index.ts'
import {
  add,
  div,
  matmul,
  mean,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  stack,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { constant, float, samples } from './_shared.ts'
import type { Spectrum } from './signal.ts'

// Parametric spectral estimation: autoregressive models by Yule-Walker (Levinson-Durbin) and Burg, their spectra,
// and subspace and exponential-fitting frequency estimators (MUSIC, ESPRIT, Prony). AR coefficients follow
// statsmodels: x_t = sum_k phi_k x_(t-k) + e_t.

/** An autoregressive model: coefficients phi_1 .. phi_p, the innovation variance and the reflection coefficients. */
export interface ArModel {
  readonly ar: Value
  readonly sigma2: Value
  readonly reflection: Value
}

/**
 * The Levinson-Durbin recursion (statsmodels.tsa.stattools.levinson_durbin with isacov): solve the Yule-Walker
 * equations of autocovariances r_0 .. r_p for the AR coefficients, the innovation variance and the partial
 * autocorrelations (reflection coefficients), in O(p^2). Differentiable in r.
 *
 * @param r - Autocovariances r_0 .. r_p.
 * @returns The AR(p) model.
 * @concept autoregressive-spectrum
 * @example
 * ```ts
 * import { levinsonDurbin } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(levinsonDurbin([1, 0.5, 0.25]).ar).map((v) => Math.round(v * 1e12) / 1e12) // => [0.5, 0]
 * ```
 */
export function levinsonDurbin(r: TensorLike): ArModel {
  const v = reshape(float(r), [-1])
  const p = (v.shape[0] ?? 1) - 1
  if (p < 1) throw new OmaiError('value', 'levinsonDurbin: need autocovariances up to lag 1 at least')
  const at = (k: number): Value => reshape(sliceAxis(v, 0, k, k + 1), [])
  let phi: Value[] = [div(at(1), at(0))]
  let sig = sub(at(0), mul(phi[0] as Value, at(1)))
  const reflection: Value[] = [phi[0] as Value]
  for (let k = 2; k <= p; k++) {
    let acc = at(k)
    for (let j = 1; j < k; j++) acc = sub(acc, mul(phi[j - 1] as Value, at(k - j)))
    const kk = div(acc, sig)
    phi = [...phi.map((pj, j) => sub(pj, mul(kk, phi[k - 2 - j] as Value))), kk]
    reflection.push(kk)
    sig = mul(sig, sub(1, mul(kk, kk)))
  }
  return { ar: stack(phi), sigma2: sig, reflection: stack(reflection) }
}

/** Options for {@link yuleWalker}. */
export interface YuleWalkerOptions {
  /** 'adjusted' (divide lag-k sums by n - k, default) or 'mle' (by n). */
  readonly method?: 'adjusted' | 'mle'
  /** Remove the mean first (default true). */
  readonly demean?: boolean
}

/**
 * Yule-Walker AR estimation (statsmodels.regression.linear_model.yule_walker): sample autocovariances, then the
 * Toeplitz normal equations. Differentiable in x.
 *
 * @param x - Signal.
 * @param order - AR order p.
 * @param options - Autocovariance normalisation and demeaning.
 * @returns The coefficients and innovation variance (reflection coefficients by Levinson-Durbin).
 * @concept autoregressive-spectrum
 * @example
 * ```ts
 * import { yuleWalker } from 'omai-engine/signal'
 * yuleWalker([1, 2, 0, -1, 1, 3, 1, -2, 0, 1], 2).ar.shape // => [2]
 * ```
 */
export function yuleWalker(x: TensorLike, order: number, options: YuleWalkerOptions = {}): ArModel {
  let v = reshape(float(x), [-1])
  const n = v.shape[0] ?? 0
  if (!Number.isInteger(order) || order < 1 || order >= n) throw new OmaiError('value', 'yuleWalker: 1 <= order < n')
  if (options.demean ?? true) v = sub(v, mean(v))
  const adjusted = (options.method ?? 'adjusted') === 'adjusted'
  const r: Value[] = [div(sum(mul(v, v)), n)]
  for (let k = 1; k <= order; k++)
    r.push(div(sum(mul(sliceAxis(v, 0, 0, n - k), sliceAxis(v, 0, k, n))), adjusted ? n - k : n))
  const rv = stack(r)
  const rho = solveToeplitz(sliceAxis(rv, 0, 0, order), undefined, sliceAxis(rv, 0, 1, order + 1))
  const sigma2 = sub(reshape(sliceAxis(rv, 0, 0, 1), []), sum(mul(sliceAxis(rv, 0, 1, order + 1), rho)))
  return { ar: rho, sigma2, reflection: levinsonDurbin(rv).reflection }
}

/**
 * Burg's AR estimation (statsmodels.regression.linear_model.burg): reflection coefficients minimising the forward
 * and backward prediction errors at each order, so the model is always stable.
 *
 * @param x - Signal.
 * @param order - AR order p.
 * @param demean - Remove the mean first (default true).
 * @returns The coefficients, innovation variance and reflection coefficients.
 * @concept autoregressive-spectrum
 * @example
 * ```ts
 * import { burg } from 'omai-engine/signal'
 * burg([1, 2, 0, -1, 1, 3, 1, -2, 0, 1], 3).ar.shape // => [3]
 * ```
 */
export function burg(x: TensorLike, order: number, demean = true): ArModel {
  const raw = samples(x, 'burg')
  const n = raw.length
  if (!Number.isInteger(order) || order < 1 || order > n - 1) throw new OmaiError('value', 'burg: 1 <= order < n')
  let m = 0
  if (demean) for (const v of raw) m += v / n
  const xs = Float64Array.from(raw, (v) => v - m)
  const p = order
  const d = new Float64Array(p + 1)
  const pacf = new Float64Array(p + 1)
  let x2 = 0
  for (const v of xs) x2 += v * v
  d[0] = 2 * x2
  let u = Float64Array.from(xs).reverse()
  let v = Float64Array.from(u)
  const dot = (a: Float64Array, ao: number, b: Float64Array, bo: number, len: number): number => {
    let s = 0
    for (let i = 0; i < len; i++) s += (a[ao + i] ?? 0) * (b[bo + i] ?? 0)
    return s
  }
  d[1] = dot(u, 0, u, 0, n - 1) + dot(v, 1, v, 1, n - 1)
  pacf[1] = (2 / (d[1] ?? 1)) * dot(v, 1, u, 0, n - 1)
  for (let i = 1; i < p; i++) {
    const lu = u
    const lv = v
    u = Float64Array.from(lu)
    v = Float64Array.from(lv)
    for (let j = 1; j < n; j++) {
      u[j] = (lu[j - 1] ?? 0) - (pacf[i] ?? 0) * (lv[j] ?? 0)
      v[j] = (lv[j] ?? 0) - (pacf[i] ?? 0) * (lu[j - 1] ?? 0)
    }
    d[i + 1] = (1 - (pacf[i] ?? 0) ** 2) * (d[i] ?? 0) - (v[i] ?? 0) ** 2 - (u[n - 1] ?? 0) ** 2
    pacf[i + 1] = (2 / (d[i + 1] ?? 1)) * dot(v, i + 1, u, i, n - 1 - i)
  }
  const sigma2 = ((1 - (pacf[p] ?? 0) ** 2) * (d[p] ?? 0)) / (2 * (n - p))
  // Coefficients from the partial autocorrelations (statsmodels' levinson_durbin_pacf).
  const k = Array.from(pacf.slice(1))
  const ar = [...k]
  for (let i = 1; i < p; i++) {
    const prev = ar.slice(0, i)
    for (let j = 0; j < i; j++) ar[j] = (prev[j] ?? 0) - (ar[i] ?? 0) * (prev[i - 1 - j] ?? 0)
  }
  return { ar: constant(ar), sigma2: constant([sigma2], []), reflection: constant(k) }
}

/** Options for {@link arSpectrum}. */
export interface ArSpectrumOptions {
  readonly fs?: number
  /** Number of frequencies over [0, fs / 2] (default 256). */
  readonly n?: number
  /** One-sided density (doubled away from DC and Nyquist, default true) or two-sided. */
  readonly onesided?: boolean
}

/**
 * The power spectral density of an AR model: sigma^2 / fs / |1 - sum_k phi_k e^{-2 pi i f k / fs}|^2 (one-sided by
 * default, in the units of welch). Differentiable in the coefficients and the variance.
 *
 * @param model - AR coefficients and innovation variance.
 * @param options - Sampling rate, grid size and sides.
 * @returns The frequencies and the density.
 * @concept autoregressive-spectrum
 * @example
 * ```ts
 * import { arSpectrum } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(arSpectrum({ ar: [0], sigma2: 1 }, { n: 3, onesided: false }).values) // => [1, 1, 1]
 * ```
 */
export function arSpectrum(
  model: { readonly ar: TensorLike; readonly sigma2: TensorLike },
  options: ArSpectrumOptions = {},
): Spectrum {
  const fs = options.fs ?? 1
  const n = options.n ?? 256
  const onesided = options.onesided ?? true
  const phi = reshape(float(model.ar), [-1])
  const p = phi.shape[0] ?? 0
  const f = Array.from({ length: n }, (_, j) => (n === 1 ? 0 : (j * fs) / (2 * (n - 1))))
  const c = new Float64Array(p * n)
  const s = new Float64Array(p * n)
  for (let k = 0; k < p; k++) {
    for (const [j, fj] of f.entries()) {
      c[k * n + j] = Math.cos((2 * Math.PI * fj * (k + 1)) / fs)
      s[k * n + j] = Math.sin((2 * Math.PI * fj * (k + 1)) / fs)
    }
  }
  const re = sub(1, matmul(phi, constant(c, [p, n])))
  const im = matmul(phi, constant(s, [p, n]))
  const fold = f.map((_, j) => (onesided && j > 0 && j < n - 1 ? 2 : 1))
  const values = mul(div(div(reshape(float(model.sigma2), []), fs), add(mul(re, re), mul(im, im))), constant(fold))
  return { frequencies: constant(f), values, kind: 'density', fs }
}

/** Forward-backward correlation matrix of order m from overlapping snapshots of a real signal. */
function correlationMatrix(x: Float64Array, m: number): Float64Array {
  const n = x.length
  const count = n - m + 1
  if (count < 1) throw new OmaiError('value', 'the correlation order must not exceed the signal length')
  const r = new Float64Array(m * m)
  for (let s = 0; s < count; s++) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        const f = (x[s + i] ?? 0) * (x[s + j] ?? 0)
        const b = (x[s + m - 1 - i] ?? 0) * (x[s + m - 1 - j] ?? 0)
        r[i * m + j] = (r[i * m + j] ?? 0) + (f + b) / (2 * count)
      }
    }
  }
  return r
}

/** Options shared by the subspace estimators. */
export interface SubspaceOptions {
  /** Size of the correlation matrix (default 2 p + 2, at most n). */
  readonly m?: number
  readonly fs?: number
}

/** The MUSIC pseudospectrum and the frequencies of its largest peaks. */
export interface MusicResult {
  readonly spectrum: Spectrum
  readonly frequencies: number[]
}

/**
 * The MUSIC frequency estimator: the noise subspace of the forward-backward correlation matrix (eigenvectors past
 * the p largest), and the pseudospectrum 1 / sum_k |e(f)^H v_k|^2, which peaks at the p complex exponentials (two per
 * real sinusoid).
 *
 * @param x - Real signal.
 * @param p - Signal subspace dimension (2 per real sinusoid).
 * @param options - Correlation size, sampling rate, and the pseudospectrum grid size `nfft` (default 1024).
 * @returns The pseudospectrum and the p / 2 strongest peak frequencies, ascending.
 * @concept subspace-methods
 * @example
 * ```ts
 * import { music } from 'omai-engine/signal'
 * const x = Array.from({ length: 64 }, (_, n) => Math.cos(2 * Math.PI * 0.12 * n) + Math.cos(2 * Math.PI * 0.31 * n + 1))
 * music(x, 4).frequencies.map((f) => Math.round(f * 1e3) / 1e3) // => [0.12, 0.31]
 * ```
 */
export function music(
  x: TensorLike,
  p: number,
  options: SubspaceOptions & { readonly nfft?: number } = {},
): MusicResult {
  const xs = samples(x, 'music')
  const fs = options.fs ?? 1
  const m = Math.min(options.m ?? 2 * p + 2, xs.length)
  if (!(p >= 1 && p < m)) throw new OmaiError('value', 'music: need 1 <= p < m')
  const v = toFlat(primalOf(eigh(constant(correlationMatrix(xs, m), [m, m])).eigenvectors))
  const nfft = options.nfft ?? 1024
  const bins = Math.floor(nfft / 2) + 1
  const ps = new Float64Array(bins)
  // Noise eigenvectors: the m - p smallest (eigh sorts ascending).
  for (let j = 0; j < bins; j++) {
    const w = (2 * Math.PI * j) / nfft
    let total = 0
    for (let col = 0; col < m - p; col++) {
      let re = 0
      let im = 0
      for (let i = 0; i < m; i++) {
        re += (v[i * m + col] ?? 0) * Math.cos(w * i)
        im -= (v[i * m + col] ?? 0) * Math.sin(w * i)
      }
      total += re * re + im * im
    }
    ps[j] = 1 / total
  }
  const peaks: number[] = []
  for (let j = 1; j + 1 < bins; j++)
    if ((ps[j] ?? 0) > (ps[j - 1] ?? 0) && (ps[j] ?? 0) >= (ps[j + 1] ?? 0)) peaks.push(j)
  peaks.sort((a, b) => (ps[b] ?? 0) - (ps[a] ?? 0))
  const chosen = peaks.slice(0, Math.max(1, Math.floor(p / 2))).map((j) => {
    // Parabolic refinement on the log pseudospectrum.
    const a = Math.log(ps[j - 1] ?? 1)
    const b = Math.log(ps[j] ?? 1)
    const c = Math.log(ps[j + 1] ?? 1)
    const den = a - 2 * b + c
    return ((j + (den === 0 ? 0 : (0.5 * (a - c)) / den)) * fs) / nfft
  })
  return {
    spectrum: {
      frequencies: constant(Array.from({ length: bins }, (_, j) => (j * fs) / nfft)),
      values: constant(ps),
      kind: 'pseudospectrum',
      fs,
    },
    frequencies: chosen.sort((a, b) => a - b),
  }
}

/**
 * The ESPRIT frequency estimator: the rotational invariance of the signal subspace between the first and last m - 1
 * rows, Phi = U1^+ U2, whose eigenvalue angles are the frequencies of the p complex exponentials.
 *
 * @param x - Real signal.
 * @param p - Signal subspace dimension (2 per real sinusoid).
 * @param options - Correlation size and sampling rate.
 * @returns The non-negative frequencies, ascending.
 * @concept subspace-methods
 * @example
 * ```ts
 * import { esprit } from 'omai-engine/signal'
 * const x = Array.from({ length: 64 }, (_, n) => Math.cos(2 * Math.PI * 0.12 * n) + Math.cos(2 * Math.PI * 0.31 * n + 1))
 * esprit(x, 4).map((f) => Math.round(f * 1e6) / 1e6) // => [0.12, 0.31]
 * ```
 */
export function esprit(x: TensorLike, p: number, options: SubspaceOptions = {}): number[] {
  const xs = samples(x, 'esprit')
  const fs = options.fs ?? 1
  const m = Math.min(options.m ?? 2 * p + 2, xs.length)
  if (!(p >= 1 && p < m)) throw new OmaiError('value', 'esprit: need 1 <= p < m')
  const us = sliceAxis(eigh(constant(correlationMatrix(xs, m), [m, m])).eigenvectors, 1, m - p, m)
  const u1 = sliceAxis(us, 0, 0, m - 1)
  const u2 = sliceAxis(us, 0, 1, m)
  const phi = matmul(pinv(u1), u2)
  const ev = eigvals(phi)
  const re = toFlat(ev.re)
  const im = toFlat(ev.im)
  return re
    .map((r, i) => (Math.atan2(im[i] ?? 0, r) / (2 * Math.PI)) * fs)
    .filter((f) => f >= -1e-12)
    .map((f) => Math.abs(f))
    .sort((a, b) => a - b)
}

/** A sum of damped complex exponentials fitted by Prony's method. */
export interface PronyResult {
  /** The poles z_k (x[n] = sum_k A_k z_k^n). */
  readonly poles: { readonly re: number[]; readonly im: number[] }
  /** The complex amplitudes A_k. */
  readonly amplitudes: { readonly re: number[]; readonly im: number[] }
  /** Frequencies angle(z_k) fs / (2 pi). */
  readonly frequencies: number[]
  /** Damping factors log |z_k| fs (negative for decay). */
  readonly damping: number[]
}

/**
 * Prony's method: fit x[n] = sum_k A_k z_k^n with p exponentials, by least-squares linear prediction (the
 * characteristic polynomial's roots are the poles) and then least-squares amplitudes on the Vandermonde system.
 * Exact for noiseless data with at least 2 p samples.
 *
 * @param x - Real signal.
 * @param p - Number of exponentials (2 per real damped sinusoid).
 * @param fs - Sampling rate (default 1).
 * @returns The poles, amplitudes, frequencies and damping factors.
 * @concept prony
 * @example
 * ```ts
 * import { prony } from 'omai-engine/signal'
 * const r = prony([1, 0.5, 0.25, 0.125, 0.0625], 1)
 * [Math.round((r.poles.re[0] ?? 0) * 1e12) / 1e12, Math.round((r.amplitudes.re[0] ?? 0) * 1e12) / 1e12] // => [0.5, 1]
 * ```
 */
export function prony(x: TensorLike, p: number, fs = 1): PronyResult {
  const xs = samples(x, 'prony')
  const n = xs.length
  if (!Number.isInteger(p) || p < 1 || 2 * p > n) throw new OmaiError('value', 'prony: need 1 <= p and 2 p <= n')
  // x[k] = -sum_j a_j x[k - j] for k = p .. n - 1.
  const rows = n - p
  const a = new Float64Array(rows * p)
  const b = new Float64Array(rows)
  for (let r = 0; r < rows; r++) {
    for (let j = 0; j < p; j++) a[r * p + j] = -(xs[p + r - 1 - j] ?? 0)
    b[r] = xs[p + r] ?? 0
  }
  const coef = toFlat(lstsq(constant(a, [rows, p]), constant(b)))
  // The characteristic polynomial z^p + a_1 z^(p-1) + ... + a_p, lowest degree first.
  const roots = polyroots([...[...coef].reverse(), 1])
  const zr = toFlat(roots.re)
  const zi = toFlat(roots.im)
  // Amplitudes: the complex Vandermonde system as a real least-squares problem of twice the size.
  const vr = new Float64Array(2 * n * 2 * p)
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < p; j++) {
      const zk = C.pow(C.of(zr[j] ?? 0, zi[j] ?? 0), k)
      const z = k === 0 ? C.of(1) : zk
      // [Re; Im] of (ar + i ai) z = [zr ar - zi ai; zi ar + zr ai]
      vr[k * 2 * p + j] = z.re
      vr[k * 2 * p + p + j] = -z.im
      vr[(n + k) * 2 * p + j] = z.im
      vr[(n + k) * 2 * p + p + j] = z.re
    }
  }
  const rhs = new Float64Array(2 * n)
  rhs.set(xs)
  const amp = toFlat(lstsq(constant(vr, [2 * n, 2 * p]), constant(rhs)))
  const angles = zr.map((r, j) => Math.atan2(zi[j] ?? 0, r))
  const order = angles.map((_, i) => i).sort((i, j) => (angles[i] ?? 0) - (angles[j] ?? 0))
  const pick = <T>(xs2: readonly T[]): T[] => order.map((i) => xs2[i] as T)
  const pr = pick(zr)
  const pi = pick(zi)
  return {
    poles: { re: pr, im: pi },
    amplitudes: { re: pick(Array.from(amp.slice(0, p))), im: pick(Array.from(amp.slice(p))) },
    frequencies: pr.map((r, j) => (Math.atan2(pi[j] ?? 0, r) * fs) / (2 * Math.PI)),
    damping: pr.map((r, j) => Math.log(Math.hypot(r, pi[j] ?? 0)) * fs),
  }
}
