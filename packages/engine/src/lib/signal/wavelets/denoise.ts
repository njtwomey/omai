/**
 * Wavelet denoising: soft, hard and garrote thresholding of coefficients (pywt.threshold), the robust median estimate
 * of the noise level from the finest details, and `waveletDenoise`, which decomposes a signal or image, shrinks its
 * details by VisuShrink or BayesShrink and reconstructs it, as skimage.restoration.denoise_wavelet.
 */

import { WhiteboxError } from '../../base/index.ts'
import {
  abs,
  div,
  greater,
  less,
  maximum,
  mul,
  primalOf,
  sign,
  sliceAxis,
  sub,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { float } from '../_shared.ts'
import { wavedec, wavedec2, waverec, waverec2 } from './dwt.ts'
import { dwtMaxLevel, type WaveletSpec, wavelet } from './families.ts'

// Wavelet shrinkage: pywt.threshold's rules, the robust noise estimate from the finest details, and
// skimage.restoration.denoise_wavelet's VisuShrink (universal threshold) and BayesShrink (per-subband) denoisers for
// signals and images.

/** A thresholding rule (pywt.threshold). */
export type ThresholdMode = 'soft' | 'hard' | 'garrote' | 'greater' | 'less'

/**
 * Threshold coefficients (pywt.threshold): soft (shrink towards zero by t), hard (zero below t), non-negative garrote
 * (x - t^2 / x), or keep values greater or less than t; the others become `substitute`. Differentiable almost
 * everywhere.
 *
 * @param x - Coefficients.
 * @param value - The threshold t.
 * @param mode - The rule (default 'soft').
 * @param substitute - Value for removed coefficients (default 0).
 * @returns The thresholded coefficients.
 * @concept wavelet-denoising
 * @example
 * ```ts
 * import { threshold } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(threshold([-3, -1, 0.5, 2], 1)) // => [-2, 0, 0, 1]
 * toArray(threshold([-3, -1, 0.5, 2], 1, 'hard')) // => [-3, -1, 0, 2]
 * ```
 */
export function threshold(x: TensorLike, value: number, mode: ThresholdMode = 'soft', substitute = 0): Value {
  const v = float(x)
  const mag = abs(v)
  switch (mode) {
    case 'soft': {
      const shrunk = mul(sign(v), maximum(sub(mag, value), 0))
      return substitute === 0 ? shrunk : where(less(mag, value), substitute, shrunk)
    }
    case 'hard':
      return where(less(mag, value), substitute, v)
    case 'garrote': {
      const safe = where(greater(mag, 0), v, 1)
      return where(less(mag, value), substitute, sub(v, div(value * value, safe)))
    }
    case 'greater':
      return where(less(v, value), substitute, v)
    case 'less':
      return where(greater(v, value), substitute, v)
  }
}

/**
 * A robust estimate of Gaussian noise from detail coefficients (skimage's _sigma_est_dwt): median(|d|) / 0.6745 over
 * the non-zero coefficients.
 *
 * @param detail - Finest-level detail coefficients.
 * @returns The noise standard deviation.
 * @concept wavelet-denoising
 * @example
 * ```ts
 * import { estimateNoiseSigma } from 'whitebox-engine/signal'
 * Math.round(estimateNoiseSigma([1, -1, 1, -1]) * 1e6) / 1e6 // => 1.482602
 * ```
 */
export function estimateNoiseSigma(detail: TensorLike): number {
  const d = toFlat(primalOf(float(detail)))
    .filter((v) => v !== 0)
    .map(Math.abs)
    .sort((a, b) => a - b)
  if (d.length === 0) return 0
  const m =
    d.length % 2 === 1 ? (d[(d.length - 1) / 2] ?? 0) : ((d[d.length / 2 - 1] ?? 0) + (d[d.length / 2] ?? 0)) / 2
  return m / 0.6744897501960817
}

/** Options for {@link waveletDenoise}. */
export interface WaveletDenoiseOptions {
  /** The noise level (default estimated from the finest details). */
  readonly sigma?: number
  /** The wavelet (default 'db1'). */
  readonly wavelet?: WaveletSpec
  /** Thresholding rule (default 'soft'). */
  readonly mode?: 'soft' | 'hard'
  /** Decomposition levels (default max(maximum level - 3, 1)). */
  readonly levels?: number
  /** 'BayesShrink' (default, a threshold per subband) or 'VisuShrink' (the universal sigma sqrt(2 log n)). */
  readonly method?: 'BayesShrink' | 'VisuShrink'
}

function bayes(d: Value, variance: number): number {
  const v = toFlat(primalOf(d))
  let s = 0
  for (const x of v) s += x * x
  return variance / Math.sqrt(Math.max(s / v.length - variance, Number.EPSILON))
}

/**
 * Wavelet denoising of a signal `[n]` or an image `[h, w]` (skimage.restoration.denoise_wavelet for float data):
 * decompose, threshold the details by VisuShrink or BayesShrink with the noise estimated from the finest diagonal
 * details, reconstruct and crop.
 *
 * @param x - Noisy signal or image.
 * @param options - Noise level, wavelet, rule, levels and method.
 * @returns The denoised data, the shape of x.
 * @concept wavelet-denoising
 * @example
 * ```ts
 * import { waveletDenoise } from 'whitebox-engine/signal'
 * waveletDenoise(new Array(64).fill(0).map((_, i) => Math.sin(i / 5) + 0.1 * Math.cos(37 * i))).shape // => [64]
 * ```
 */
export function waveletDenoise(x: TensorLike, options: WaveletDenoiseOptions = {}): Value {
  const v = float(x)
  const w = wavelet(options.wavelet ?? 'db1')
  const mode = options.mode ?? 'soft'
  const method = options.method ?? 'BayesShrink'
  const size = v.shape.reduce((a, b) => a * b, 1)
  if (v.shape.length === 1) {
    const n = v.shape[0] ?? 0
    const levels = options.levels ?? Math.max(dwtMaxLevel(n, w) - 3, 1)
    const coeffs = wavedec(v, w, { level: levels })
    const finest = coeffs[coeffs.length - 1] as Value
    const sigma = options.sigma ?? estimateNoiseSigma(finest)
    const universal = sigma * Math.sqrt(2 * Math.log(size))
    const out = coeffs.map((c, i) =>
      i === 0 ? c : threshold(c, method === 'VisuShrink' ? universal : bayes(c, sigma * sigma), mode),
    )
    return sliceTo(waverec(out, w), [n])
  }
  if (v.shape.length !== 2) throw new WhiteboxError('shape', 'waveletDenoise: expected a signal or a grey image')
  const [h = 0, wd = 0] = v.shape
  const levels = options.levels ?? Math.max(Math.min(dwtMaxLevel(h, w), dwtMaxLevel(wd, w)) - 3, 1)
  const c = wavedec2(v, w, { level: levels })
  const finest = c.details[c.details.length - 1]
  if (finest === undefined) throw new WhiteboxError('value', 'waveletDenoise: no detail coefficients')
  const sigma = options.sigma ?? estimateNoiseSigma(finest.cD)
  const universal = sigma * Math.sqrt(2 * Math.log(size))
  const t = (d: Value): Value => threshold(d, method === 'VisuShrink' ? universal : bayes(d, sigma * sigma), mode)
  const details = c.details.map((d) => ({ cH: t(d.cH), cV: t(d.cV), cD: t(d.cD) }))
  return sliceTo(waverec2({ cA: c.cA, details }, w), [h, wd])
}

function sliceTo(v: Value, shape: readonly number[]): Value {
  let out = v
  for (const [ax, s] of shape.entries()) if ((out.shape[ax] ?? 0) > s) out = sliceAxis(out, ax, 0, s)
  return out
}
