/**
 * Fourier analysis of sampled signals beyond the bare FFT: the discrete-time Fourier transform at arbitrary frequencies
 * (differentiable), zero padding, frequency resolution, sub-bin peak-frequency estimation by parabolic interpolation,
 * spectral leakage of a window and Parseval's identity, plus the Hilbert transform, the analytic signal and the
 * amplitude envelope (scipy.signal.hilbert).
 */

import { OmaiError } from '../base/index.ts'
import { fft, ifft, rfft } from '../numerics/fourier/index.ts'
import {
  absComplex,
  add,
  type ComplexLike,
  type ComplexValue,
  complex,
  cScale,
  div,
  matmul,
  mul,
  pad,
  primalOf,
  sliceAxis,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { axisOf, constant, float, samples } from './_shared.ts'
import { getWindow, type WindowSpec } from './windows/index.ts'

// Fourier analysis of sampled signals: the DTFT at arbitrary frequencies, zero padding and resolution, leakage,
// Parseval's identity and the analytic signal. The transforms themselves are in numerics/fourier.

/**
 * The discrete-time Fourier transform at given frequencies: X(f) = sum_n x_n exp(-2 pi i f n / fs), as a product
 * with the cosine and sine matrices, so differentiable in x (real or complex).
 *
 * @param x - Samples `[n]` (real or complex).
 * @param frequencies - Frequencies in the units of fs.
 * @param fs - Sampling rate (default 1: cycles per sample).
 * @returns The transform at each frequency.
 * @concept fourier-transform
 * @example
 * ```ts
 * import { dtft } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(dtft([1, 1], [0, 0.5]).re).map((v) => Math.round(v * 1e12) / 1e12) // => [2, 0]
 * ```
 */
export function dtft(x: ComplexLike, frequencies: TensorLike, fs = 1): ComplexValue {
  const z = complex(x)
  if (z.re.shape.length !== 1) throw new OmaiError('shape', 'dtft: expected a 1-D signal')
  const n = z.re.shape[0] ?? 0
  const f = toFlat(primalOf(float(frequencies)))
  const c = new Float64Array(n * f.length)
  const s = new Float64Array(n * f.length)
  for (let k = 0; k < n; k++) {
    for (const [j, fj] of f.entries()) {
      const t = (2 * Math.PI * fj * k) / fs
      c[k * f.length + j] = Math.cos(t)
      s[k * f.length + j] = Math.sin(t)
    }
  }
  const cm = constant(c, [n, f.length])
  const sm = constant(s, [n, f.length])
  // (xr + i xi)(cos - i sin)
  return {
    re: add(matmul(z.re, cm), matmul(z.im, sm)),
    im: sub(matmul(z.im, cm), matmul(z.re, sm)),
  }
}

/**
 * Zero-pad (or crop) along an axis to length n, refining the frequency grid of a following DFT without adding
 * resolution. Differentiable.
 *
 * @param x - Input.
 * @param n - Target length.
 * @param axis - Axis (default -1).
 * @returns The padded input.
 * @example
 * ```ts
 * import { zeroPad } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(zeroPad([1, 2], 4)) // => [1, 2, 0, 0]
 * ```
 */
export function zeroPad(x: TensorLike, n: number, axis = -1): Value {
  const v = float(x)
  const a = axisOf(axis, v.shape.length, 'zeroPad')
  const len = v.shape[a] ?? 0
  if (n <= len) return sliceAxis(v, a, 0, n)
  return pad(
    v,
    v.shape.map((_, i): [number, number] => (i === a ? [0, n - len] : [0, 0])),
  )
}

/**
 * The frequency resolution of an observation: the Rayleigh limit fs / n (two tones closer than this are not
 * resolved, however much the record is zero-padded), and the bin spacing fs / nfft of a padded DFT.
 *
 * @param n - Number of samples observed.
 * @param fs - Sampling rate (default 1).
 * @param nfft - DFT length (default n).
 * @returns The resolution and the bin spacing.
 * @concept frequency-resolution
 * @example
 * ```ts
 * import { frequencyResolution } from 'omai-engine/signal'
 * frequencyResolution(100, 1000, 400) // => { resolution: 10, binSpacing: 2.5 }
 * ```
 */
export function frequencyResolution(
  n: number,
  fs = 1,
  nfft: number = n,
): { readonly resolution: number; readonly binSpacing: number } {
  if (!(n > 0) || !(nfft > 0)) throw new OmaiError('value', 'frequencyResolution: lengths must be positive')
  return { resolution: fs / n, binSpacing: fs / nfft }
}

/** Options for {@link peakFrequency}. */
export interface PeakFrequencyOptions {
  readonly fs?: number
  /** DFT length (default the signal length; zero-padding refines the grid). */
  readonly nfft?: number
  /** Window (default 'hann'). */
  readonly window?: WindowSpec
}

/**
 * The frequency of the strongest spectral peak, refined by fitting a parabola to the log magnitudes of the peak bin
 * and its neighbours (quadratic interpolation, accurate to a small fraction of a bin for windowed tones).
 *
 * @param x - Real samples `[n]`.
 * @param options - Sampling rate, DFT length and window.
 * @returns The estimated frequency, in the units of fs.
 * @concept spectral-leakage
 * @example
 * ```ts
 * import { peakFrequency } from 'omai-engine/signal'
 * const x = Array.from({ length: 64 }, (_, n) => Math.cos(2 * Math.PI * 0.1234 * n))
 * Math.abs(peakFrequency(x) - 0.1234) < 1e-3 // => true
 * ```
 */
export function peakFrequency(x: TensorLike, options: PeakFrequencyOptions = {}): number {
  const s = samples(x, 'peakFrequency')
  const fs = options.fs ?? 1
  const nfft = options.nfft ?? s.length
  const w = toFlat(getWindow(options.window ?? 'hann', s.length))
  const spec = rfft(
    Float64Array.from(s, (v, i) => v * (w[i] ?? 0)),
    nfft,
  )
  const re = toFlat(spec.re)
  const im = toFlat(spec.im)
  const mag = re.map((r, k) => Math.log(Math.hypot(r, im[k] ?? 0) + 1e-300))
  let k = 1
  for (let j = 1; j < mag.length; j++) if ((mag[j] ?? 0) > (mag[k] ?? 0)) k = j
  const a = mag[k - 1] ?? mag[k] ?? 0
  const b = mag[k] ?? 0
  const c = mag[k + 1] ?? b
  const denom = a - 2 * b + c
  const delta = denom === 0 ? 0 : (0.5 * (a - c)) / denom
  return ((k + delta) * fs) / nfft
}

/**
 * Spectral leakage of a window: the fraction of a complex tone's power that falls outside the `bins` DFT bins
 * either side of its frequency, for a tone `offset` bins from a bin centre (0.5 is the worst case between bins).
 *
 * @param window - The window (or its length for a rectangular window).
 * @param offset - Offset of the tone from the nearest bin, in bins (default 0.5).
 * @param bins - Bins counted as the tone's own on each side (default 1).
 * @returns The leaked fraction in [0, 1].
 * @concept spectral-leakage
 * @example
 * ```ts
 * import { hann, leakageFraction } from 'omai-engine/signal'
 * leakageFraction(64, 0) < 1e-20 // => true
 * leakageFraction(hann(64, { sym: false }), 0.5) < leakageFraction(64, 0.5) // => true
 * ```
 */
export function leakageFraction(window: TensorLike | number, offset = 0.5, bins = 1): number {
  const w = typeof window === 'number' ? new Float64Array(window).fill(1) : samples(window, 'leakageFraction')
  const n = w.length
  const k0 = Math.floor(n / 4)
  const f = (k0 + offset) / n
  const re = Float64Array.from(w, (v, i) => v * Math.cos(2 * Math.PI * f * i))
  const im = Float64Array.from(w, (v, i) => v * Math.sin(2 * Math.PI * f * i))
  const spec = fft({ re: constant(re), im: constant(im) })
  const sr = toFlat(spec.re)
  const si = toFlat(spec.im)
  let total = 0
  let near = 0
  for (let k = 0; k < n; k++) {
    const p = (sr[k] ?? 0) ** 2 + (si[k] ?? 0) ** 2
    total += p
    if (Math.abs(k - (k0 + offset)) <= bins) near += p
  }
  return total === 0 ? 0 : 1 - near / total
}

/**
 * Both sides of Parseval's identity for the DFT: the energy sum |x_n|^2 and sum |X_k|^2 / N, equal up to rounding.
 *
 * @param x - Samples (real or complex, 1-D).
 * @returns The time-domain and frequency-domain energies (differentiable values).
 * @concept parseval
 * @example
 * ```ts
 * import { parseval } from 'omai-engine/signal'
 * import { item } from 'omai-engine/tensor'
 * const { time, frequency } = parseval([1, 2, 3, 4])
 * [item(time), Math.round(item(frequency) * 1e12) / 1e12] // => [30, 30]
 * ```
 */
export function parseval(x: ComplexLike): { readonly time: Value; readonly frequency: Value } {
  const z = complex(x)
  const n = z.re.shape[z.re.shape.length - 1] ?? 1
  const time = sum(add(mul(z.re, z.re), mul(z.im, z.im)))
  const spec = fft(z)
  const frequency = div(sum(add(mul(spec.re, spec.re), mul(spec.im, spec.im))), n)
  return { time, frequency }
}

/**
 * The analytic signal x + i H[x] (scipy.signal.hilbert): FFT, keep DC (and Nyquist), double the positive
 * frequencies, zero the negative ones, invert. Differentiable.
 *
 * @param x - Real input.
 * @param n - Transform length (default the axis length).
 * @param axis - Axis (default -1).
 * @returns The analytic signal; its imaginary part is the Hilbert transform.
 * @concept analytic-signal
 * @example
 * ```ts
 * import { hilbert } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(hilbert([1, 0, -1, 0]).im).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 1, 0, -1]
 * ```
 */
export function hilbert(x: TensorLike, n?: number, axis = -1): ComplexValue {
  const v = float(x)
  const a = axisOf(axis, v.shape.length, 'hilbert')
  const len = n ?? v.shape[a] ?? 0
  if (len < 1) throw new OmaiError('value', 'hilbert: the length must be positive')
  const h = new Float64Array(len)
  h[0] = 1
  if (len % 2 === 0) {
    h[len / 2] = 1
    for (let k = 1; k < len / 2; k++) h[k] = 2
  } else for (let k = 1; k < (len + 1) / 2; k++) h[k] = 2
  const shape = v.shape.map((_, i) => (i === a ? len : 1))
  const spec = fft(v, len, a)
  return ifft(cScale(spec, constant(h, shape)), undefined, a)
}

/**
 * The Hilbert transform H[x]: the imaginary part of the analytic signal (a 90-degree phase shift of every component).
 *
 * @param x - Real input.
 * @param axis - Axis (default -1).
 * @returns H[x].
 * @concept analytic-signal
 * @example
 * ```ts
 * import { hilbertTransform } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(hilbertTransform([0, 1, 0, -1])).map((v) => Math.round(v * 1e12) / 1e12) // => [-1, 0, 1, 0]
 * ```
 */
export function hilbertTransform(x: TensorLike, axis = -1): Value {
  return hilbert(x, undefined, axis).im
}

/**
 * The amplitude envelope |x + i H[x]| of a real signal.
 *
 * @param x - Real input.
 * @param axis - Axis (default -1).
 * @returns The envelope.
 * @concept analytic-signal
 * @example
 * ```ts
 * import { envelope } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(envelope([1, 0, -1, 0])).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 1, 1, 1]
 * ```
 */
export function envelope(x: TensorLike, axis = -1): Value {
  return absComplex(hilbert(x, undefined, axis))
}
