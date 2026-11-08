/**
 * The cepstrum, the inverse Fourier transform of a log spectrum, which separates a slowly varying spectral envelope
 * from a periodic excitation: the real cepstrum, the complex cepstrum and its inverse (as MATLAB cceps and
 * python-acoustics), liftering of low or high quefrencies, and pitch estimation from the strongest cepstral peak.
 */

import { OmaiError } from '../base/index.ts'
import { fft, ifft } from '../numerics/fourier/index.ts'
import {
  absComplex,
  add,
  angle,
  cExp,
  log,
  maximum,
  mul,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { constant, float, samples, vector } from './_shared.ts'
import { unwrap } from './signal.ts'

// Cepstral analysis: the real cepstrum (the inverse transform of the log magnitude), the complex cepstrum with phase
// unwrapping and removal of the linear phase (as MATLAB's cceps and python-acoustics), its inverse, liftering, and
// pitch from the cepstral peak.

/**
 * The real cepstrum: Re IFFT(log |FFT(x)|). Differentiable where the spectrum does not vanish.
 *
 * @param x - Signal.
 * @param n - Transform length (default the signal length).
 * @returns The cepstrum `[n]`.
 * @concept cepstrum
 * @example
 * ```ts
 * import { realCepstrum } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(realCepstrum([1, 0, 0, 0])) // => [0, 0, 0, 0]
 * ```
 */
export function realCepstrum(x: TensorLike, n?: number): Value {
  const v = vector(x, 'realCepstrum')
  const spec = fft(v, n)
  return ifft({ re: log(maximum(absComplex(spec), Number.MIN_VALUE)), im: mul(spec.re, 0) }).re
}

/** A complex cepstrum and the linear-phase delay removed before taking it. */
export interface ComplexCepstrum {
  readonly cepstrum: Value
  readonly delay: number
}

/**
 * The complex cepstrum (MATLAB cceps, python-acoustics): IFFT(log |X| + i arg X) with the unwrapped phase made
 * periodic by removing the linear phase of a delay of round(phase[centre] / pi) samples, which is returned.
 *
 * @param x - Signal.
 * @param n - Transform length (default the signal length).
 * @returns The cepstrum and the removed delay.
 * @concept cepstrum
 * @example
 * ```ts
 * import { complexCepstrum } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(complexCepstrum([1, 0.5], 64).cepstrum).slice(0, 3).map((v) => Math.round(v * 1e9) / 1e9) // => [0, 0.5, -0.125]
 * ```
 */
export function complexCepstrum(x: TensorLike, n?: number): ComplexCepstrum {
  const v = vector(x, 'complexCepstrum')
  const len = n ?? v.shape[0] ?? 0
  const spec = fft(v, len)
  const phase = unwrap(angle(spec))
  const centre = Math.floor((len + 1) / 2)
  const delay = Math.round((toFlat(phase)[centre] ?? 0) / Math.PI)
  const ramp = constant(Array.from({ length: len }, (_, k) => (-Math.PI * delay * k) / centre))
  const logSpec = { re: log(maximum(absComplex(spec), Number.MIN_VALUE)), im: add(phase, ramp) }
  return { cepstrum: ifft(logSpec).re, delay }
}

/**
 * Invert a complex cepstrum (python-acoustics inverse_complex_cepstrum): exponentiate its spectrum and restore the
 * linear phase of the removed delay.
 *
 * @param c - The cepstrum and delay from {@link complexCepstrum}.
 * @returns The signal.
 * @concept cepstrum
 * @example
 * ```ts
 * import { complexCepstrum, inverseComplexCepstrum } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(inverseComplexCepstrum(complexCepstrum([2, -1, 0.5, 0.25]))).map((v) => Math.round(v * 1e12) / 1e12) // => [2, -1, 0.5, 0.25]
 * ```
 */
export function inverseComplexCepstrum(c: ComplexCepstrum): Value {
  const len = c.cepstrum.shape[0] ?? 0
  const centre = Math.floor((len + 1) / 2)
  const spec = fft(c.cepstrum)
  const ramp = constant(Array.from({ length: len }, (_, k) => (Math.PI * c.delay * k) / centre))
  return ifft(cExp({ re: spec.re, im: add(spec.im, ramp) })).re
}

/**
 * Lifter a cepstrum: keep the low quefrencies (the spectral envelope, 'low'), the high ones (the excitation, 'high'),
 * or weight MFCCs by librosa's sinusoidal lifter 1 + (L / 2) sin(pi (n + 1) / L) ('sine').
 *
 * @param c - Cepstrum `[n]` (or MFCCs `[..., n]` for 'sine', along the first axis of coefficients).
 * @param cutoff - Quefrency cutoff in samples (or L for 'sine').
 * @param kind - 'low' (default), 'high' or 'sine'.
 * @returns The liftered cepstrum.
 * @concept cepstrum
 * @example
 * ```ts
 * import { lifter } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(lifter([1, 2, 3, 4, 3, 2], 2)) // => [1, 2, 0, 0, 0, 2]
 * ```
 */
export function lifter(c: TensorLike, cutoff: number, kind: 'low' | 'high' | 'sine' = 'low'): Value {
  const v = float(c)
  const n = v.shape[0] ?? 0
  if (kind === 'sine') {
    const w = Array.from({ length: n }, (_, i) => 1 + (cutoff / 2) * Math.sin((Math.PI * (i + 1)) / cutoff))
    const shape = v.shape.map((s, i) => (i === 0 ? s : 1))
    return mul(v, constant(w, shape))
  }
  // A symmetric window over quefrencies below the cutoff (and the mirrored negative quefrencies).
  const keep = Array.from({ length: n }, (_, i) => {
    const q = Math.min(i, n - i)
    const low = q < cutoff
    return kind === 'low' ? (low ? 1 : 0) : low ? 0 : 1
  })
  return mul(v, constant(keep))
}

/**
 * Pitch by the cepstrum: the quefrency of the largest real-cepstrum peak between fs / fmax and fs / fmin samples.
 *
 * @param x - A voiced frame.
 * @param fs - Sampling rate.
 * @param options - Pitch range (default 50 to 500 Hz).
 * @returns The fundamental frequency in Hz.
 * @concept pitch-estimation
 * @example
 * ```ts
 * import { cepstralPitch } from 'omai-engine/signal'
 * const fs = 8000
 * const x = Array.from({ length: 1024 }, (_, n) => [1, 2, 3, 4, 5].reduce((a, h) => a + Math.cos((2 * Math.PI * 200 * h * n) / fs) / h, 0))
 * Math.abs(cepstralPitch(x, fs) - 200) < 10 // => true
 * ```
 */
export function cepstralPitch(
  x: TensorLike,
  fs: number,
  options: { readonly fmin?: number; readonly fmax?: number } = {},
): number {
  const s = samples(x, 'cepstralPitch')
  const fmin = options.fmin ?? 50
  const fmax = options.fmax ?? 500
  const c = toFlat(realCepstrum(s))
  const lo = Math.max(1, Math.floor(fs / fmax))
  const hi = Math.min(c.length - 1, Math.ceil(fs / fmin))
  if (lo >= hi) throw new OmaiError('value', 'cepstralPitch: the frame is too short for this pitch range')
  let best = lo
  for (let q = lo; q <= hi; q++) if ((c[q] ?? 0) > (c[best] ?? 0)) best = q
  return fs / best
}
