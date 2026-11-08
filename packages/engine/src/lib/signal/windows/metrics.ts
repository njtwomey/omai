/**
 * Figures of merit of a window function (Harris 1978): equivalent noise bandwidth, coherent gain, main-lobe and
 * half-power widths, highest sidelobe level and scalloping loss, the numbers that say how a window trades resolution
 * against leakage.
 */

import { OmaiError } from '../../base/index.ts'
import { rfft } from '../../numerics/fourier/index.ts'
import { type TensorLike, toFlat } from '../../tensor/index.ts'
import { samples } from '../_shared.ts'

/**
 * The equivalent noise bandwidth of a window in bins: n sum w^2 / (sum w)^2 (1 for the rectangular window, 1.5 for
 * Hann); multiply by fs / n for hertz.
 *
 * @param window - The window samples.
 * @returns The ENBW in bins.
 * @concept window-functions
 * @example
 * ```ts
 * import { enbw, hann } from 'omai-engine/signal'
 * enbw(hann(64, { sym: false })) // => 1.5
 * ```
 */
export function enbw(window: TensorLike): number {
  const w = samples(window, 'enbw')
  let s1 = 0
  let s2 = 0
  for (const v of w) {
    s1 += v
    s2 += v * v
  }
  if (s1 === 0) throw new OmaiError('value', 'enbw: the window sums to zero')
  return (w.length * s2) / (s1 * s1)
}

/**
 * The coherent gain of a window: its mean, the amplitude of a bin-centred sinusoid relative to a rectangular window.
 *
 * @param window - The window samples.
 * @returns sum w / n.
 * @example
 * ```ts
 * import { coherentGain, hann } from 'omai-engine/signal'
 * coherentGain(hann(4, { sym: false })) // => 0.5
 * ```
 */
export function coherentGain(window: TensorLike): number {
  const w = samples(window, 'coherentGain')
  let s = 0
  for (const v of w) s += v
  return s / w.length
}

/** Figures of merit of a window, from its spectrum on a fine grid. Widths are in bins of the window length. */
export interface WindowMetrics {
  /** Equivalent noise bandwidth (bins). */
  readonly enbw: number
  /** Mean of the window. */
  readonly coherentGain: number
  /** Width between the first nulls either side of the main lobe (bins). */
  readonly mainLobeWidth: number
  /** Width where the main lobe falls 3.01 dB below its peak (bins). */
  readonly halfPowerWidth: number
  /** Level of the highest sidelobe relative to the main lobe (dB, negative). */
  readonly sidelobeLevel: number
  /** Loss for a sinusoid half-way between bins (dB, positive). */
  readonly scallopLoss: number
}

/**
 * The classical figures of merit of a window (Harris, 1978): equivalent noise bandwidth, coherent gain, main-lobe and
 * half-power widths, highest sidelobe level and scalloping loss, from its magnitude response zero-padded to 64 times
 * its length (or `oversample` times).
 *
 * @param window - The window samples (length at least 2).
 * @param oversample - The zero-padding factor of the response grid (default 64).
 * @returns The metrics.
 * @throws OmaiError with code `value` when the response has no null (a window too short to have sidelobes).
 * @concept window-functions
 * @example
 * ```ts
 * import { hann, windowMetrics } from 'omai-engine/signal'
 * const m = windowMetrics(hann(64, { sym: false }))
 * Math.round(m.sidelobeLevel * 10) / 10 // => -31.5
 * m.mainLobeWidth // => 4
 * ```
 */
export function windowMetrics(window: TensorLike, oversample = 64): WindowMetrics {
  const w = samples(window, 'windowMetrics')
  const n = w.length
  const nfft = n * oversample
  const spec = rfft(w, nfft)
  const re = toFlat(spec.re)
  const im = toFlat(spec.im)
  const mag = Float64Array.from(re, (r, k) => Math.hypot(r, im[k] ?? 0))
  const peak = mag[0] ?? 0
  if (peak === 0) throw new OmaiError('value', 'windowMetrics: the window sums to zero')
  let nullAt = -1
  for (let k = 1; k + 1 < mag.length; k++) {
    if ((mag[k] ?? 0) <= (mag[k - 1] ?? 0) && (mag[k] ?? 0) < (mag[k + 1] ?? 0)) {
      nullAt = k
      break
    }
  }
  if (nullAt < 0) throw new OmaiError('value', 'windowMetrics: no main-lobe null on the grid')
  let side = 0
  for (let k = nullAt; k < mag.length; k++) side = Math.max(side, mag[k] ?? 0)
  // Half power: interpolate linearly in magnitude between grid points.
  const target = peak / Math.SQRT2
  let half = 0
  for (let k = 1; k <= nullAt; k++) {
    const a = mag[k - 1] ?? 0
    const b = mag[k] ?? 0
    if (b <= target) {
      half = k - 1 + (a - target) / (a - b)
      break
    }
  }
  const halfBin = oversample / 2
  const scallop = mag[Math.round(halfBin)] ?? peak
  return {
    enbw: enbw(w),
    coherentGain: coherentGain(w),
    mainLobeWidth: (2 * nullAt) / oversample,
    halfPowerWidth: (2 * half) / oversample,
    sidelobeLevel: 20 * Math.log10(side / peak),
    scallopLoss: -20 * Math.log10(scallop / peak),
  }
}
