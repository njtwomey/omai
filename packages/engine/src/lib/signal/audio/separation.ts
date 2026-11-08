/**
 * Mask-based source separation in the short-time Fourier domain: the ideal binary, ideal ratio and phase-sensitive
 * masks computed from known sources, `applyMask` to filter a mixture by a mask, and `oracleSeparation`, the Wiener-mask
 * upper bound on what any mask-based separator can achieve.
 */

import { OmaiError } from '../../base/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { float, samples } from '../_shared.ts'
import { librosaIstft, librosaStft } from './_stft.ts'

// Time-frequency masking for source separation: ideal binary, ratio and phase-sensitive masks from known sources,
// their application to a mixture, and oracle separation by masks. Separation quality (SDR, SI-SDR, permutation-
// invariant scores) is in `metrics/audio.ts`; NMF and ICA separation are in `methods/audio`.

/** Options of the STFT used by the masks. */
export interface MaskOptions {
  /** FFT length (default 512). */
  readonly nFft?: number
  /** Hop (default nFft / 4). */
  readonly hopLength?: number
}

interface Spectrum {
  readonly re: Float64Array
  readonly im: Float64Array
  readonly bins: number
  readonly frames: number
}

function spectrumOf(x: TensorLike, options: MaskOptions, name: string): Spectrum {
  const nFft = options.nFft ?? 512
  const z = librosaStft(samples(x, name), nFft, options.hopLength ?? Math.floor(nFft / 4))
  const re = primalOf(z.re)
  return {
    re: Float64Array.from(re.data, Number),
    im: Float64Array.from(primalOf(z.im).data, Number),
    bins: re.shape[0] ?? 0,
    frames: re.shape[1] ?? 0,
  }
}

const power = (s: Spectrum, i: number): number => (s.re[i] ?? 0) ** 2 + (s.im[i] ?? 0) ** 2

/**
 * The ideal binary mask: 1 where the target's local SNR over the interference exceeds thresholdDb.
 *
 * @param target - Target signal.
 * @param interference - Interference (noise or the other sources), the same length.
 * @param options - STFT and threshold (default 0 dB).
 * @returns The mask `[bins, frames]`.
 * @concept time-frequency-masking
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * const s = Array.from({ length: 2048 }, (_, i) => Math.sin(i / 3))
 * const n = Array.from({ length: 2048 }, (_, i) => 0.1 * Math.sin(i * 2.5))
 * audio.idealBinaryMask(s, n).shape // => [257, 17]
 * ```
 */
export function idealBinaryMask(
  target: TensorLike,
  interference: TensorLike,
  options: MaskOptions & { readonly thresholdDb?: number } = {},
): Tensor {
  const s = spectrumOf(target, options, 'idealBinaryMask')
  const n = spectrumOf(interference, options, 'idealBinaryMask')
  if (s.frames !== n.frames) throw new OmaiError('shape', 'idealBinaryMask: signals differ in length')
  const thr = 10 ** ((options.thresholdDb ?? 0) / 10)
  const m = Float64Array.from({ length: s.bins * s.frames }, (_, i) => (power(s, i) > thr * power(n, i) ? 1 : 0))
  return fromStorage(m, [s.bins, s.frames], 'float64')
}

/**
 * The ideal ratio mask (|S|^2 / (|S|^2 + |N|^2))^beta; beta = 1 is the oracle Wiener filter.
 *
 * @param target - Target signal.
 * @param interference - Interference.
 * @param options - STFT and exponent (default 0.5).
 * @returns The mask in [0, 1] `[bins, frames]`.
 * @concept time-frequency-masking
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * const s = Array.from({ length: 1024 }, (_, i) => Math.sin(i / 3))
 * toFlat(audio.idealRatioMask(s, s)).every((v) => Math.abs(v - Math.SQRT1_2) < 1e-9 || v === 0) // => true
 * ```
 */
export function idealRatioMask(
  target: TensorLike,
  interference: TensorLike,
  options: MaskOptions & { readonly beta?: number } = {},
): Tensor {
  const s = spectrumOf(target, options, 'idealRatioMask')
  const n = spectrumOf(interference, options, 'idealRatioMask')
  const beta = options.beta ?? 0.5
  const m = Float64Array.from({ length: s.bins * s.frames }, (_, i) => {
    const ps = power(s, i)
    const total = ps + power(n, i)
    return total > 0 ? (ps / total) ** beta : 0
  })
  return fromStorage(m, [s.bins, s.frames], 'float64')
}

/**
 * The phase-sensitive mask |S| / |Y| cos(angle S - angle Y), truncated to [0, 1] (Erdogan et al. 2015).
 *
 * @param target - Target signal.
 * @param mixture - The mixture.
 * @param options - STFT.
 * @returns The mask `[bins, frames]`.
 * @concept time-frequency-masking
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * const s = Array.from({ length: 1024 }, (_, i) => Math.sin(i / 3))
 * audio.phaseSensitiveMask(s, s).shape // => [257, 9]
 * ```
 */
export function phaseSensitiveMask(target: TensorLike, mixture: TensorLike, options: MaskOptions = {}): Tensor {
  const s = spectrumOf(target, options, 'phaseSensitiveMask')
  const y = spectrumOf(mixture, options, 'phaseSensitiveMask')
  const m = Float64Array.from({ length: s.bins * s.frames }, (_, i) => {
    const py = power(y, i)
    if (py === 0) return 0
    // |S| |Y| cos(dphi) / |Y|^2 = Re(S conj(Y)) / |Y|^2.
    const v = ((s.re[i] ?? 0) * (y.re[i] ?? 0) + (s.im[i] ?? 0) * (y.im[i] ?? 0)) / py
    return Math.min(1, Math.max(0, v))
  })
  return fromStorage(m, [s.bins, s.frames], 'float64')
}

/**
 * Apply a time-frequency mask to a mixture: STFT, multiply, inverse STFT.
 *
 * @param mixture - The mixture.
 * @param mask - Mask `[bins, frames]` for the same STFT.
 * @param options - STFT.
 * @returns The masked signal, the length of the mixture.
 * @concept time-frequency-masking
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * const y = Array.from({ length: 1024 }, (_, i) => Math.sin(i / 3))
 * const r = audio.applyMask(y, audio.idealBinaryMask(y, new Array(1024).fill(0)))
 * Math.abs((r[500] ?? 0) - (y[500] ?? 0)) < 1e-9 // => true
 * ```
 */
export function applyMask(mixture: TensorLike, mask: TensorLike, options: MaskOptions = {}): Float64Array {
  const x = samples(mixture, 'applyMask')
  const nFft = options.nFft ?? 512
  const hop = options.hopLength ?? Math.floor(nFft / 4)
  const y = spectrumOf(x, options, 'applyMask')
  const m = samplesMatrix(mask)
  if ((m.shape[0] ?? 0) !== y.bins || (m.shape[1] ?? 0) !== y.frames)
    throw new OmaiError('shape', 'applyMask: the mask does not match the STFT')
  const re = Float64Array.from(y.re, (v, i) => v * Number(m.data[i] ?? 0))
  const im = Float64Array.from(y.im, (v, i) => v * Number(m.data[i] ?? 0))
  return librosaIstft(
    { re: fromStorage(re, [y.bins, y.frames], 'float64'), im: fromStorage(im, [y.bins, y.frames], 'float64') },
    nFft,
    hop,
    x.length,
  )
}

function samplesMatrix(x: TensorLike): Tensor {
  const v = primalOf(float(x))
  if (v.shape.length !== 2) throw new OmaiError('shape', 'applyMask: masks are [bins, frames]')
  return v
}

/**
 * Oracle separation by masks: each source's ideal ratio mask (beta 1, the Wiener masks, which sum to one) applied
 * to the mixture of all sources, the upper bound of mask-based separation.
 *
 * @param sources - The sources, the same length.
 * @param options - STFT and exponent (default 1).
 * @returns The separated estimates, one per source.
 * @concept time-frequency-masking
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * const a = Array.from({ length: 2048 }, (_, i) => Math.sin(i / 3))
 * const b = Array.from({ length: 2048 }, (_, i) => 0.5 * Math.sin(i * 2.4))
 * audio.oracleSeparation([a, b]).length // => 2
 * ```
 */
export function oracleSeparation(
  sources: readonly TensorLike[],
  options: MaskOptions & { readonly beta?: number } = {},
): Float64Array[] {
  if (sources.length < 2) throw new OmaiError('value', 'oracleSeparation: need at least two sources')
  const xs = sources.map((s) => samples(s, 'oracleSeparation'))
  const n = xs[0]?.length ?? 0
  if (xs.some((x) => x.length !== n)) throw new OmaiError('shape', 'oracleSeparation: sources differ in length')
  const mixture = Float64Array.from({ length: n }, (_, i) => xs.reduce((s, x) => s + (x[i] ?? 0), 0))
  return xs.map((x) => {
    const rest = Float64Array.from(mixture, (v, i) => v - (x[i] ?? 0))
    return applyMask(mixture, idealRatioMask(x, rest, { ...options, beta: options.beta ?? 1 }), options)
  })
}
