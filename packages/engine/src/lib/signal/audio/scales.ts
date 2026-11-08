/**
 * Perceptual frequency scales and their conversions to and from hertz: the mel scale (Slaney and HTK variants, as
 * librosa), the Bark scale (Traunmüller, Zwicker and Terhardt, Schroeder) and the equivalent rectangular bandwidth
 * (ERB) scale of the auditory filter, with helpers that space frequencies evenly on them and list the rFFT bin
 * frequencies.
 */

import { OmaiError } from '../../base/index.ts'
import {
  add,
  arcsinh,
  arctan,
  div,
  exp,
  greaterEqual,
  less,
  log,
  log10,
  mul,
  pow,
  sinh,
  sub,
  type Tensor,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { constant, float } from '../_shared.ts'

// Perceptual frequency scales: mel (HTK and Slaney, as librosa), Bark (Traunmüller, Zwicker-Terhardt, Schroeder) and
// the equivalent-rectangular-bandwidth scale of Glasberg and Moore. Elementwise and differentiable.

const SLANEY_SP = 200 / 3
const SLANEY_LOG_HZ = 1000
const SLANEY_LOG_MEL = SLANEY_LOG_HZ / SLANEY_SP
const SLANEY_STEP = Math.log(6.4) / 27

/**
 * Hertz to mels (librosa.hz_to_mel): Slaney's scale (linear below 1 kHz, logarithmic above; the default) or HTK's
 * 2595 log10(1 + f / 700).
 *
 * @param f - Frequencies in Hz.
 * @param htk - Use the HTK formula (default false).
 * @returns Mels.
 * @concept mel-scale
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(audio.hzToMel([0, 1000, 2000])).map((v) => Math.round(v * 1e6) / 1e6) // => [0, 15, 25.08188]
 * ```
 */
export function hzToMel(f: TensorLike, htk = false): Value {
  const x = float(f)
  if (htk) return mul(log10(add(div(x, 700), 1)), 2595)
  const linear = div(x, SLANEY_SP)
  const logs = add(
    div(log(div(where(greaterEqual(x, SLANEY_LOG_HZ), x, SLANEY_LOG_HZ), SLANEY_LOG_HZ)), SLANEY_STEP),
    SLANEY_LOG_MEL,
  )
  return where(greaterEqual(x, SLANEY_LOG_HZ), logs, linear)
}

/**
 * Mels to hertz (librosa.mel_to_hz), the inverse of {@link hzToMel}.
 *
 * @param m - Mels.
 * @param htk - Use the HTK formula (default false).
 * @returns Frequencies in Hz.
 * @concept mel-scale
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(audio.melToHz([0, 15])).map((v) => Math.round(v * 1e6) / 1e6) // => [0, 1000]
 * ```
 */
export function melToHz(m: TensorLike, htk = false): Value {
  const x = float(m)
  if (htk) return mul(sub(pow(10, div(x, 2595)), 1), 700)
  const linear = mul(x, SLANEY_SP)
  const safe = where(greaterEqual(x, SLANEY_LOG_MEL), x, SLANEY_LOG_MEL)
  const logs = mul(exp(mul(sub(safe, SLANEY_LOG_MEL), SLANEY_STEP)), SLANEY_LOG_HZ)
  return where(greaterEqual(x, SLANEY_LOG_MEL), logs, linear)
}

/**
 * Frequencies equally spaced on the mel scale (librosa.mel_frequencies).
 *
 * @param n - Number of frequencies (default 128).
 * @param fmin - Lowest frequency in Hz (default 0).
 * @param fmax - Highest frequency in Hz (default 11025).
 * @param htk - Use the HTK formula (default false).
 * @returns The frequencies `[n]`.
 * @concept mel-scale
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(audio.melFrequencies(3, 0, 1000)).map((v) => Math.round(v * 1e6) / 1e6) // => [0, 500, 1000]
 * ```
 */
export function melFrequencies(n = 128, fmin = 0, fmax = 11025, htk = false): Tensor {
  if (!Number.isInteger(n) || n < 1) throw new OmaiError('value', 'melFrequencies: n must be a positive integer')
  const lo = Number((hzToMel([fmin], htk) as Tensor).data[0] ?? 0)
  const hi = Number((hzToMel([fmax], htk) as Tensor).data[0] ?? 0)
  const mels = Float64Array.from({ length: n }, (_, i) => (n === 1 ? lo : lo + ((hi - lo) * i) / (n - 1)))
  return melToHz(constant(mels), htk) as Tensor
}

/**
 * The centre frequencies of the rFFT bins (librosa.fft_frequencies): k sr / nfft for k = 0 .. nfft / 2.
 *
 * @param sr - Sampling rate (default 22050).
 * @param nfft - FFT length (default 2048).
 * @returns The frequencies `[nfft / 2 + 1]`.
 * @concept mel-scale
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(audio.fftFrequencies(8, 4)) // => [0, 2, 4]
 * ```
 */
export function fftFrequencies(sr = 22050, nfft = 2048): Tensor {
  return constant(Float64Array.from({ length: Math.floor(nfft / 2) + 1 }, (_, k) => (k * sr) / nfft))
}

/** The Bark-scale formulas. */
export type BarkFormula = 'traunmuller' | 'zwicker' | 'schroeder'

/**
 * Hertz to Bark: Traunmüller's 26.81 f / (1960 + f) - 0.53 with his end corrections (default), Zwicker and
 * Terhardt's 13 atan(0.00076 f) + 3.5 atan((f / 7500)^2), or Schroeder's 7 asinh(f / 650).
 *
 * @param f - Frequencies in Hz.
 * @param formula - The formula (default 'traunmuller').
 * @returns Critical-band rates in Bark.
 * @concept bark-scale
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(audio.hzToBark([1000])).map((v) => Math.round(v * 1e4) / 1e4) // => [8.5274]
 * ```
 */
export function hzToBark(f: TensorLike, formula: BarkFormula = 'traunmuller'): Value {
  const x = float(f)
  if (formula === 'schroeder') return mul(arcsinh(div(x, 650)), 7)
  if (formula === 'zwicker') return add(mul(arctan(mul(x, 0.00076)), 13), mul(arctan(pow(div(x, 7500), 2)), 3.5))
  const z = sub(div(mul(x, 26.81), add(x, 1960)), 0.53)
  const low = add(z, mul(sub(2, z), 0.15))
  const high = add(z, mul(sub(z, 20.1), 0.22))
  return where(less(z, 2), low, where(greaterEqual(z, 20.1), high, z))
}

/**
 * Bark to hertz: the inverse of Traunmüller's formula (with its corrections) or of Schroeder's (Zwicker and
 * Terhardt's has no closed-form inverse).
 *
 * @param z - Critical-band rates in Bark.
 * @param formula - 'traunmuller' (default) or 'schroeder'.
 * @returns Frequencies in Hz.
 * @concept bark-scale
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(audio.barkToHz(audio.hzToBark([300, 1000, 9000]))).map((v) => Math.round(v * 1e6) / 1e6) // => [300, 1000, 9000]
 * ```
 */
export function barkToHz(z: TensorLike, formula: Exclude<BarkFormula, 'zwicker'> = 'traunmuller'): Value {
  const b = float(z)
  if (formula === 'schroeder') return mul(sinh(div(b, 7)), 650)
  // Undo the corrections: below 2, z' = z + 0.15 (2 - z) = 0.85 z + 0.3; above 20.1, z' = 1.22 z - 4.422.
  const raw = where(less(b, 2), div(sub(b, 0.3), 0.85), where(greaterEqual(b, 20.1), div(add(b, 4.422), 1.22), b))
  return div(mul(add(raw, 0.53), 1960), sub(26.28, raw))
}

/**
 * The equivalent rectangular bandwidth of the auditory filter at f (Glasberg and Moore 1990): 24.7 (4.37 f / 1000 + 1).
 *
 * @param f - Frequencies in Hz.
 * @returns Bandwidths in Hz.
 * @concept erb-scale
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(audio.erbBandwidth([1000])).map((v) => Math.round(v * 1e6) / 1e6) // => [132.639]
 * ```
 */
export function erbBandwidth(f: TensorLike): Value {
  return mul(add(mul(float(f), 4.37e-3), 1), 24.7)
}

/**
 * Hertz to the ERB-rate scale (number of ERBs below f): 21.4 log10(1 + 4.37 f / 1000).
 *
 * @param f - Frequencies in Hz.
 * @returns ERB numbers.
 * @concept erb-scale
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(audio.hzToErb([1000])).map((v) => Math.round(v * 1e4) / 1e4) // => [15.6214]
 * ```
 */
export function hzToErb(f: TensorLike): Value {
  return mul(log10(add(mul(float(f), 4.37e-3), 1)), 21.4)
}

/**
 * The ERB-rate scale to hertz, the inverse of {@link hzToErb}.
 *
 * @param e - ERB numbers.
 * @returns Frequencies in Hz.
 * @concept erb-scale
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(audio.erbToHz(audio.hzToErb([440]))).map((v) => Math.round(v * 1e9) / 1e9) // => [440]
 * ```
 */
export function erbToHz(e: TensorLike): Value {
  return div(sub(pow(10, div(float(e), 21.4)), 1), 4.37e-3)
}

/**
 * Frequencies equally spaced on the ERB-rate scale between fmin and fmax (ascending), the centre frequencies of a
 * gammatone filter bank.
 *
 * @param fmin - Lowest frequency in Hz.
 * @param fmax - Highest frequency in Hz.
 * @param n - Number of frequencies.
 * @returns The frequencies `[n]`.
 * @concept erb-scale
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(audio.erbSpace(100, 8000, 3)).map((v) => Math.round(v)) // => [100, 1416, 8000]
 * ```
 */
export function erbSpace(fmin: number, fmax: number, n: number): Tensor {
  if (!Number.isInteger(n) || n < 1) throw new OmaiError('value', 'erbSpace: n must be a positive integer')
  const lo = 21.4 * Math.log10(1 + 4.37e-3 * fmin)
  const hi = 21.4 * Math.log10(1 + 4.37e-3 * fmax)
  return erbToHz(
    constant(Float64Array.from({ length: n }, (_, i) => (n === 1 ? lo : lo + ((hi - lo) * i) / (n - 1)))),
  ) as Tensor
}
