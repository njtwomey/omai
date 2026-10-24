import { WhiteboxError } from '../../base/index.ts'
import { imdct, mdct, rfft, sineWindow } from '../../numerics/fourier/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { samples } from '../_shared.ts'
import { getWindow } from '../windows/index.ts'
import { hzToBark } from './scales.ts'

// Perceptual audio coding: the absolute threshold of hearing (Terhardt 1979), Schroeder's spreading function, a
// simultaneous-masking threshold in the Bark domain (in the manner of MPEG psychoacoustic model 1, simplified: every
// bin masks with a single offset), and MDCT transform coding whose quantiser step follows the masking threshold.

/**
 * The absolute threshold of hearing in dB SPL (Terhardt 1979): 3.64 f^-0.8 - 6.5 exp(-0.6 (f - 3.3)^2) + 1e-3 f^4
 * with f in kHz.
 *
 * @param f - Frequencies in Hz (positive).
 * @returns Thresholds in dB SPL.
 * @concept psychoacoustics
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * Array.from(audio.absoluteThreshold([1000])).map((v) => Math.round(v * 100) / 100) // => [3.37]
 * ```
 */
export function absoluteThreshold(f: ArrayLike<number>): Float64Array {
  return Float64Array.from(f, (hz) => {
    const k = Math.max(hz, 1) / 1000
    return 3.64 * k ** -0.8 - 6.5 * Math.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k ** 4
  })
}

/**
 * Schroeder's spreading function in dB for a Bark distance dz (masked minus masker): 15.81 + 7.5 (dz + 0.474) -
 * 17.5 sqrt(1 + (dz + 0.474)^2).
 *
 * @param dz - Bark distances.
 * @returns Attenuations in dB (0 near dz = 0, negative away from it).
 * @concept psychoacoustics
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * Array.from(audio.spreadingFunction([0, 1, -1])).map((v) => Math.round(v * 10) / 10) // => [0, -4.3, -7.9]
 * ```
 */
export function spreadingFunction(dz: ArrayLike<number>): Float64Array {
  return Float64Array.from(dz, (d) => 15.81 + 7.5 * (d + 0.474) - 17.5 * Math.sqrt(1 + (d + 0.474) ** 2))
}

/** Options of {@link maskingThreshold}. */
export interface MaskingOptions {
  /** Masking offset below the spread masker in dB (default 10). */
  readonly offsetDb?: number
  /** Level in dB SPL assigned to a full-scale sinusoid (default 96, as 16-bit audio). */
  readonly fullScaleDb?: number
}

/**
 * The global masking threshold of one frame: the frame's power spectrum (Hann window, dB SPL relative to full scale)
 * spread across Bark by Schroeder's function, lowered by the offset, power-added over maskers and combined with the
 * absolute threshold of hearing.
 *
 * @param frame - One frame of samples (full scale 1).
 * @param sr - Sampling rate.
 * @param options - Offset and calibration.
 * @returns Thresholds in dB SPL for every rFFT bin and the frame's spectrum in dB SPL.
 * @concept psychoacoustics
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * const x = Array.from({ length: 512 }, (_, i) => 0.5 * Math.sin((2 * Math.PI * 1000 * i) / 16000))
 * audio.maskingThreshold(x, 16000).threshold.length // => 257
 * ```
 */
export function maskingThreshold(
  frame: TensorLike,
  sr: number,
  options: MaskingOptions = {},
): { readonly threshold: Float64Array; readonly spectrum: Float64Array } {
  const x = samples(frame, 'maskingThreshold')
  const n = x.length
  const w = toFlat(getWindow('hann', n))
  let wsum = 0
  for (const v of w) wsum += v
  const spec = rfft(Float64Array.from(x, (v, i) => v * (w[i] ?? 0)))
  const re = toFlat(spec.re)
  const im = toFlat(spec.im)
  const bins = re.length
  const full = options.fullScaleDb ?? 96
  // A full-scale sinusoid has peak amplitude wsum / 2 in the windowed spectrum.
  const level = Float64Array.from(re, (r, k) => {
    const amp = Math.hypot(r, im[k] ?? 0) / (wsum / 2)
    return full + 20 * Math.log10(amp + 1e-12)
  })
  const freqs = Float64Array.from({ length: bins }, (_, k) => (k * sr) / n)
  const z = toFlat(hzToBark(freqs))
  const ath = absoluteThreshold(freqs)
  const offset = options.offsetDb ?? 10
  const threshold = new Float64Array(bins)
  for (let k = 0; k < bins; k++) {
    let p = 10 ** ((ath[k] ?? 0) / 10)
    for (let j = 0; j < bins; j++) {
      const sf =
        15.81 +
        7.5 * ((z[k] ?? 0) - (z[j] ?? 0) + 0.474) -
        17.5 * Math.sqrt(1 + ((z[k] ?? 0) - (z[j] ?? 0) + 0.474) ** 2)
      p += 10 ** (((level[j] ?? 0) + sf - offset) / 10)
    }
    threshold[k] = 10 * Math.log10(p)
  }
  return { threshold, spectrum: level }
}

/** The result of {@link mdctCode}. */
export interface MdctCode {
  /** Quantised coefficients (integers) `[frames, N]`. */
  readonly indices: Tensor
  /** Quantiser steps `[frames, N]`. */
  readonly steps: Tensor
  /** The decoded signal, the length of the input. */
  readonly decoded: Float64Array
  /** Bits by a first-order entropy estimate of the indices. */
  readonly bits: number
}

/**
 * Perceptual MDCT transform coding: sine-windowed MDCT frames of N coefficients, each coefficient quantised
 * uniformly with a step set by the frame's masking threshold (noise at the threshold is inaudible), then decoded by
 * the inverse MDCT with time-domain aliasing cancellation.
 *
 * @param x - Signal (full scale 1).
 * @param sr - Sampling rate.
 * @param options - Coefficients per frame (default 256), masking options and a step scale (default 1).
 * @returns The indices, steps, decoded signal and an entropy estimate of the bits.
 * @concept transform-coding
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * const x = Array.from({ length: 2048 }, (_, i) => 0.5 * Math.sin((2 * Math.PI * 440 * i) / 16000))
 * audio.mdctCode(x, 16000).decoded.length // => 2048
 * ```
 */
export function mdctCode(
  x: TensorLike,
  sr: number,
  options: MaskingOptions & { readonly n?: number; readonly stepScale?: number } = {},
): MdctCode {
  const s0 = samples(x, 'mdctCode')
  const n = options.n ?? 256
  if (!Number.isInteger(n) || n < 2) throw new WhiteboxError('value', 'mdctCode: n must be an integer >= 2')
  const len = Math.ceil(s0.length / n) * n
  const s = new Float64Array(len)
  s.set(s0)
  const coeffs = primalOf(mdct(s, n))
  const frames = coeffs.shape[0] ?? 0
  const c = Float64Array.from(coeffs.data, Number)
  const steps = new Float64Array(frames * n)
  const idx = new Float64Array(frames * n)
  const full = options.fullScaleDb ?? 96
  const scale = options.stepScale ?? 1
  const padded = new Float64Array(len + 2 * n)
  padded.set(s, n)
  for (let f = 0; f < frames; f++) {
    const seg = padded.slice(f * n, f * n + 2 * n)
    const thr = maskingThreshold(seg, sr, options).threshold
    for (let k = 0; k < n; k++) {
      // The threshold at MDCT bin k (frequency (k + 1/2) sr / 2N) as an amplitude: uniform quantisation noise of step
      // D has power D^2 / 12, matched to the masked power.
      const bin = Math.min(thr.length - 1, Math.round(k + 0.5))
      const amp = 10 ** (((thr[bin] ?? 0) - full) / 20) * n
      const step = Math.max(1e-9, (scale * amp * Math.sqrt(12)) / Math.sqrt(2 * n))
      steps[f * n + k] = step
      idx[f * n + k] = Math.round((c[f * n + k] ?? 0) / step)
    }
  }
  const deq = Float64Array.from(idx, (q, i) => q * (steps[i] ?? 0))
  const decodedAll = toFlat(primalOf(imdct(fromStorage(deq, [frames, n], 'float64'), sineWindow(2 * n))))
  const counts = new Map<number, number>()
  for (const q of idx) counts.set(q, (counts.get(q) ?? 0) + 1)
  let bits = 0
  for (const v of counts.values()) bits -= v * Math.log2(v / idx.length)
  return {
    indices: fromStorage(idx, [frames, n], 'float64'),
    steps: fromStorage(steps, [frames, n], 'float64'),
    decoded: Float64Array.from(decodedAll.slice(0, s0.length)),
    bits,
  }
}
