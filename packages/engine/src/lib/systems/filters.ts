/**
 * The frequency responses of filters given by coefficients, as scipy.signal: `freqs` for an analog filter b(s) / a(s)
 * and `freqz` for a digital filter in powers of z^-1 (through the real FFT for FIR filters on a uniform grid), both
 * differentiable in the coefficients.
 */

import { fft, nextFastLen, rfft } from '../numerics/fourier/index.ts'
import {
  type ComplexValue,
  cDiv,
  div,
  flip,
  reshape,
  sliceAxis,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { constant, horner, type Points, tfAt } from './_points.ts'
import { float } from './_shared.ts'
import { type FrequencyResponse, findfreqs } from './frequency.ts'

// Frequency responses of filters given by their coefficients, as scipy.signal's freqs (analog, powers of s) and freqz
// (digital, powers of z^-1).

/**
 * The frequency response of an analog filter (scipy.signal.freqs): H(jw) = b(jw) / a(jw), differentiable in the
 * coefficients.
 *
 * @param b - Numerator, highest degree first.
 * @param a - Denominator, highest degree first.
 * @param worN - Frequencies, or a count chosen by findfreqs (default 200).
 * @returns The frequencies and the response.
 * @concept frequency-response
 * @example
 * ```ts
 * import { freqs } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(freqs([1], [1, 1], [0]).h.re) // => [1]
 * ```
 */
export function freqs(b: TensorLike, a: TensorLike, worN: number | TensorLike = 200): FrequencyResponse {
  const bv = float(b)
  const av = float(a)
  const w = typeof worN === 'number' ? findfreqs(bv, av, worN) : (float(worN) as Tensor)
  const ws = toFlat(w)
  return { w, h: tfAt(bv, av, { re: constant(ws.map(() => 0)), im: constant(ws) }) }
}

/** Options for {@link freqz}. */
export interface FreqzOptions {
  /** Frequencies over the whole circle [0, fs) instead of [0, fs / 2). */
  readonly whole?: boolean
  /** The sampling frequency (default 2 pi: frequencies in radians per sample). */
  readonly fs?: number
  /** Include the Nyquist frequency in a counted grid (when not `whole`). */
  readonly includeNyquist?: boolean
}

/**
 * The frequency response of a digital filter (scipy.signal.freqz): H = B(e^{-jw}) / A(e^{-jw}) with coefficients in
 * powers of z^{-1}; an FIR filter on a fast grid goes through the real FFT, as scipy. Differentiable in the
 * coefficients.
 *
 * @param b - Numerator coefficients b0 + b1 z^-1 + ...
 * @param a - Denominator coefficients (default 1).
 * @param worN - Frequencies (in the units of fs), or a count (default 512).
 * @param options - Whole circle, sampling frequency, Nyquist inclusion.
 * @returns The frequencies and the response.
 * @concept frequency-response
 * @example
 * ```ts
 * import { freqz } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(freqz([0.5, 0.5], [1], 2).h.re) // => [1, 0.5]
 * ```
 */
export function freqz(
  b: TensorLike,
  a: TensorLike = [1],
  worN: number | TensorLike = 512,
  options: FreqzOptions = {},
): FrequencyResponse {
  const fs = options.fs ?? 2 * Math.PI
  const bv = reshape(float(b), [-1])
  const av = reshape(float(a), [-1])
  if (typeof worN === 'number') {
    const n = worN
    const whole = options.whole ?? false
    const nyquist = !whole && (options.includeNyquist ?? false)
    const last = whole ? fs : fs / 2
    const w = constant(
      Array.from({ length: n }, (_, k) => (nyquist ? (n === 1 ? 0 : (k * last) / (n - 1)) : (k * last) / n)),
    )
    const fir = (av.shape[0] ?? 0) === 1 && nextFastLen(n) === n && n >= (bv.shape[0] ?? 0)
    if (fir) {
      const nfft = whole ? n : nyquist ? 2 * (n - 1) : 2 * n
      const spectrum = whole ? fft(bv, nfft) : rfft(bv, nfft)
      const head = (x: Value): Value => sliceAxis(x, 0, 0, n)
      const a0 = reshape(av, [])
      return { w, h: { re: div(head(spectrum.re), a0), im: div(head(spectrum.im), a0) } }
    }
    return { w, h: digital(bv, av, toFlat(w), fs) }
  }
  const w = float(worN) as Tensor
  return { w, h: digital(bv, av, toFlat(w), fs) }
}

/** B(e^{-jw}) / A(e^{-jw}): Horner in z^{-1} over the reversed coefficients. */
function digital(b: Value, a: Value, w: readonly number[], fs: number): ComplexValue {
  const theta = w.map((v) => (2 * Math.PI * v) / fs)
  const zm1: Points = { re: constant(theta.map(Math.cos)), im: constant(theta.map((t) => -Math.sin(t))) }
  return cDiv(horner(flip(b, 0), zm1), horner(flip(a, 0), zm1))
}
