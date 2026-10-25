/**
 * Quadratic time-frequency distributions: the Wigner-Ville distribution of the analytic signal, perfectly concentrated
 * on linear chirps but with cross terms between components, and the pseudo and smoothed pseudo versions that window the
 * lag and time to suppress those cross terms, as in the tftb toolbox.
 */

import { MdmlError } from '../base/index.ts'
import { fft } from '../numerics/fourier/index.ts'
import { type ComplexLike, complex, primalOf, type TensorLike, toFlat } from '../tensor/index.ts'
import { constant, float } from './_shared.ts'
import { hilbert } from './fourier.ts'
import type { TimeFrequency } from './signal.ts'
import { hamming } from './windows/index.ts'

// Quadratic time-frequency distributions of the Cohen class, as the Time-Frequency Toolbox (tftb) computes them: the
// Wigner-Ville distribution, its pseudo (lag-windowed) and smoothed pseudo (lag- and time-windowed) variants. The
// frequency of bin k is k fs / (2 N): the lag product z[n + m] z*[n - m] advances twice as fast as the signal.

/** Options shared by the Wigner-Ville distributions. */
export interface WignerOptions {
  /** Sampling rate (default 1). */
  readonly fs?: number
  /** Number of frequency bins (default the signal length). */
  readonly n?: number
  /** Form the analytic signal of a real input first (default true), suppressing interference with negative frequencies. */
  readonly analytic?: boolean
}

interface Analytic {
  readonly re: Float64Array
  readonly im: Float64Array
}

function signalOf(x: TensorLike | ComplexLike, analytic: boolean): Analytic {
  const z = analytic ? hilbert(float(x as TensorLike)) : complex(x as ComplexLike)
  if (z.re.shape.length !== 1) throw new MdmlError('shape', 'wigner: expected a 1-D signal')
  return { re: Float64Array.from(primalOf(z.re).data), im: Float64Array.from(primalOf(z.im).data) }
}

/** FFT each column of a `[n, t]` complex lag table; the real part is the distribution. */
function finish(re: Float64Array, im: Float64Array, n: number, t: number, fs: number): TimeFrequency {
  const spec = fft({ re: constant(re, [n, t]), im: constant(im, [n, t]) }, undefined, 0)
  return {
    frequencies: constant(Array.from({ length: n }, (_, k) => (k * fs) / (2 * n))),
    times: constant(Array.from({ length: t }, (_, i) => i / fs)),
    values: constant(toFlat(spec.re), [n, t]),
    fs,
  }
}

function window(h: TensorLike | undefined, defaultLength: number): Float64Array {
  const w = h === undefined ? toFlat(hamming(defaultLength)) : toFlat(primalOf(float(h)))
  if (w.length % 2 === 0) throw new MdmlError('value', 'wigner: windows must have odd length')
  return Float64Array.from(w)
}

/** The lag table of the (pseudo) Wigner-Ville distribution with lag window h (all ones for the plain WVD). */
function lagTable(z: Analytic, n: number, h: Float64Array | null): { re: Float64Array; im: Float64Array } {
  const len = z.re.length
  const re = new Float64Array(n * len)
  const im = new Float64Array(n * len)
  const lh = h === null ? Number.POSITIVE_INFINITY : (h.length - 1) / 2
  const hn = h === null ? null : Float64Array.from(h, (v) => v / (h[(h.length - 1) / 2] ?? 1))
  const half = Math.round(n / 2)
  // product z[a] conj(z[b]) added into row r of column c
  const put = (r: number, c: number, a: number, b: number, scale: number): void => {
    const ar = z.re[a] ?? 0
    const ai = z.im[a] ?? 0
    const br = z.re[b] ?? 0
    const bi = z.im[b] ?? 0
    re[r * len + c] = (re[r * len + c] ?? 0) + scale * (ar * br + ai * bi)
    im[r * len + c] = (im[r * len + c] ?? 0) + scale * (ai * br - ar * bi)
  }
  for (let ti = 0; ti < len; ti++) {
    const taumax = Math.min(ti, len - ti - 1, half - 1, lh)
    for (let tau = -taumax; tau <= taumax; tau++) {
      const w = hn === null ? 1 : (hn[lh + tau] ?? 0)
      put((n + tau) % n, ti, ti + tau, ti - tau, w)
    }
    if (ti + half <= len - 1 && ti - half >= 0 && half <= lh) {
      // tftb assigns this row (for odd N it overwrites the last lag of the loop).
      re[half * len + ti] = 0
      im[half * len + ti] = 0
      put(half, ti, ti + half, ti - half, 0.5 * (hn === null ? 1 : (hn[lh + half] ?? 0)))
      put(half, ti, ti - half, ti + half, 0.5 * (hn === null ? 1 : (hn[lh - half] ?? 0)))
    }
  }
  return { re, im }
}

/**
 * The Wigner-Ville distribution W[k, n] = sum_m z[n + m] z*[n - m] e^{-2 pi i k m / N} of the analytic signal (tftb
 * tfrwv): perfectly concentrated on linear chirps, with cross terms between components.
 *
 * @param x - Real signal (or a complex signal with `analytic: false`).
 * @param options - Sampling rate, bins and analytic conversion.
 * @returns Frequencies k fs / (2 N), times and the real distribution `[N, n]`.
 * @concept wigner-ville
 * @example
 * ```ts
 * import { wignerVille } from 'mdml-engine/signal'
 * wignerVille(new Array(16).fill(0).map((_, i) => Math.cos(i))).values.shape // => [16, 16]
 * ```
 */
export function wignerVille(x: TensorLike | ComplexLike, options: WignerOptions = {}): TimeFrequency {
  const z = signalOf(x, options.analytic ?? true)
  const n = options.n ?? z.re.length
  const t = lagTable(z, n, null)
  return finish(t.re, t.im, n, z.re.length, options.fs ?? 1)
}

/**
 * The pseudo Wigner-Ville distribution (tftb tfrpwv): the WVD with the lag products tapered by a window h (default
 * Hamming of length about N / 4, odd), which attenuates cross terms between components separated in frequency.
 *
 * @param x - Real signal (or complex with `analytic: false`).
 * @param options - Sampling rate, bins, analytic conversion and the lag window h (odd length).
 * @returns The distribution `[N, n]`.
 * @concept wigner-ville
 * @example
 * ```ts
 * import { pseudoWignerVille } from 'mdml-engine/signal'
 * pseudoWignerVille(new Array(16).fill(0).map((_, i) => Math.cos(i))).values.shape // => [16, 16]
 * ```
 */
export function pseudoWignerVille(
  x: TensorLike | ComplexLike,
  options: WignerOptions & { readonly h?: TensorLike } = {},
): TimeFrequency {
  const z = signalOf(x, options.analytic ?? true)
  const n = options.n ?? z.re.length
  const h = window(options.h, 2 * Math.floor(n / 8) + 1)
  const t = lagTable(z, n, h)
  return finish(t.re, t.im, n, z.re.length, options.fs ?? 1)
}

/**
 * The smoothed pseudo Wigner-Ville distribution (tftb tfrspwv): lag window h and a separable time-smoothing window g
 * (normalised to unit sum at each time), trading resolution for the suppression of cross terms in time and frequency.
 *
 * @param x - Real signal (or complex with `analytic: false`).
 * @param options - Sampling rate, bins, analytic conversion, lag window h and time window g (odd lengths).
 * @returns The distribution `[N, n]`.
 * @concept wigner-ville
 * @example
 * ```ts
 * import { smoothedPseudoWignerVille } from 'mdml-engine/signal'
 * smoothedPseudoWignerVille(new Array(16).fill(0).map((_, i) => Math.cos(i))).values.shape // => [16, 16]
 * ```
 */
export function smoothedPseudoWignerVille(
  x: TensorLike | ComplexLike,
  options: WignerOptions & { readonly h?: TensorLike; readonly g?: TensorLike } = {},
): TimeFrequency {
  const z = signalOf(x, options.analytic ?? true)
  const len = z.re.length
  const n = options.n ?? len
  const h0 = window(options.h, 2 * Math.floor(n / 8) + 1)
  const g = window(options.g, 2 * Math.floor(n / 20) + 1)
  const lh = (h0.length - 1) / 2
  const lg = (g.length - 1) / 2
  const h = Float64Array.from(h0, (v) => v / (h0[lh] ?? 1))
  const re = new Float64Array(n * len)
  const im = new Float64Array(n * len)
  const half = Math.round(n / 2)
  /** sum_p g2[p] z[a - p] conj(z[b - p]) over p in [-pLo, pHi], with g2 = g / sum g over those points. */
  const smoothed = (a: number, b: number, pLo: number, pHi: number): [number, number] => {
    let gs = 0
    for (let p = -pLo; p <= pHi; p++) gs += g[lg + p] ?? 0
    let sr = 0
    let si = 0
    for (let p = -pLo; p <= pHi; p++) {
      const w = (g[lg + p] ?? 0) / gs
      const ar = z.re[a - p] ?? 0
      const ai = z.im[a - p] ?? 0
      const br = z.re[b - p] ?? 0
      const bi = z.im[b - p] ?? 0
      sr += w * (ar * br + ai * bi)
      si += w * (ai * br - ar * bi)
    }
    return [sr, si]
  }
  for (let ti = 0; ti < len; ti++) {
    const taumax = Math.min(ti + lg, len - 1 - ti + lg, half - 1, lh)
    const [r0, i0] = smoothed(ti, ti, Math.min(lg, len - 1 - ti), Math.min(lg, ti))
    re[ti] = r0
    im[ti] = i0
    for (let tau = 1; tau <= taumax; tau++) {
      const lo = Math.min(lg, len - 1 - ti - tau)
      const hi = Math.min(lg, ti - tau)
      if (lo < -hi) continue
      const [ra, ia] = smoothed(ti + tau, ti - tau, lo, hi)
      re[tau * len + ti] = (h[lh + tau] ?? 0) * ra
      im[tau * len + ti] = (h[lh + tau] ?? 0) * ia
      const [rb, ib] = smoothed(ti - tau, ti + tau, lo, hi)
      re[(n - tau) * len + ti] = (h[lh - tau] ?? 0) * rb
      im[(n - tau) * len + ti] = (h[lh - tau] ?? 0) * ib
    }
    if (ti + half <= len - 1 && ti - half >= 0 && half <= lh) {
      const lo = Math.min(lg, len - 1 - ti - half)
      const hi = Math.min(lg, ti - half)
      const [ra, ia] = smoothed(ti + half, ti - half, lo, hi)
      const [rb, ib] = smoothed(ti - half, ti + half, lo, hi)
      re[half * len + ti] = 0.5 * ((h[lh + half] ?? 0) * ra + (h[lh - half] ?? 0) * rb)
      im[half * len + ti] = 0.5 * ((h[lh + half] ?? 0) * ia + (h[lh - half] ?? 0) * ib)
    }
  }
  return finish(re, im, n, len, options.fs ?? 1)
}
