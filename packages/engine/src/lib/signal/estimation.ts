/**
 * Optimal linear estimation and prediction of signals: the non-causal frequency-domain Wiener filter, the causal FIR
 * Wiener filter from the Wiener-Hopf equations, scipy's local adaptive `wiener` denoiser, linear prediction
 * coefficients by Burg's method (librosa.lpc) with the residual and one-step predictor, and the matched filter that
 * maximises output SNR for a known template.
 */

import { MdmlError } from '../base/index.ts'
import { correlate, fftConvolve } from '../numerics/convolution/index.ts'
import { irfft, rfft } from '../numerics/fourier/index.ts'
import { solveToeplitz } from '../numerics/linalg/index.ts'
import {
  add,
  concat,
  cScale,
  div,
  less,
  mean,
  mul,
  neg,
  ones,
  primalOf,
  reshape,
  sliceAxis,
  stack,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../tensor/index.ts'
import { constant, float, vector } from './_shared.ts'
import { lfilter } from './filters/apply.ts'
import { burg } from './parametric.ts'

// Optimal linear estimation: Wiener filtering (the non-causal frequency-domain filter, the causal FIR Wiener-Hopf
// solution and scipy's local adaptive Wiener filter), linear prediction (LPC by Burg's method, as librosa, and its
// residual), and the matched filter.

/**
 * The non-causal Wiener filter in the frequency domain: X = S_ss / (S_ss + S_nn) Y, the minimum mean-square estimate
 * of a stationary signal in additive noise with known power spectra (given at the rfft bins of the record).
 * Differentiable in y and the spectra.
 *
 * @param y - Noisy observation `[n]`.
 * @param signalPsd - Signal power spectrum `[n / 2 + 1]`.
 * @param noisePsd - Noise power spectrum `[n / 2 + 1]` (or a constant).
 * @returns The estimate `[n]`.
 * @concept wiener-filter
 * @example
 * ```ts
 * import { nonCausalWiener } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(nonCausalWiener([2, 2, 2, 2], [1, 1, 1], 1)).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 1, 1, 1]
 * ```
 */
export function nonCausalWiener(y: TensorLike, signalPsd: TensorLike, noisePsd: TensorLike): Value {
  const v = vector(y, 'nonCausalWiener')
  const n = v.shape[0] ?? 0
  const s = reshape(float(signalPsd), [-1])
  if ((s.shape[0] ?? 0) !== Math.floor(n / 2) + 1)
    throw new MdmlError('shape', 'nonCausalWiener: the spectra must have n / 2 + 1 bins')
  const gain = div(s, add(s, float(noisePsd)))
  return irfft(cScale(rfft(v), gain), n)
}

/**
 * The causal FIR Wiener filter (the Wiener-Hopf normal equations R w = p): the taps minimising E (d - w * x)^2, from
 * the biased autocorrelation of x and its cross-correlation with the desired signal. Differentiable.
 *
 * @param x - Input signal.
 * @param d - Desired signal (same length).
 * @param order - Number of taps.
 * @returns The taps `[order]` and the filtered input.
 * @concept wiener-filter
 * @example
 * ```ts
 * import { firWiener } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 2000 }, (_, i) => Math.sin(1.3 * i) + Math.cos(0.4 * i))
 * const d = x.map((v, i) => 2 * v + 0.5 * (x[i - 1] ?? 0))
 * toFlat(firWiener(x, d, 2).taps).map((v) => Math.round(v * 10) / 10) // => [2, 0.5]
 * ```
 */
export function firWiener(
  x: TensorLike,
  d: TensorLike,
  order: number,
): { readonly taps: Value; readonly output: Value } {
  const xv = vector(x, 'firWiener')
  const dv = vector(d, 'firWiener')
  const n = xv.shape[0] ?? 0
  if ((dv.shape[0] ?? 0) !== n) throw new MdmlError('shape', 'firWiener: x and d must have the same length')
  if (!Number.isInteger(order) || order < 1 || order > n) throw new MdmlError('value', 'firWiener: 1 <= order <= n')
  const r: Value[] = []
  const p: Value[] = []
  for (let k = 0; k < order; k++) {
    r.push(div(sum(mul(sliceAxis(xv, 0, 0, n - k), sliceAxis(xv, 0, k, n))), n))
    p.push(div(sum(mul(sliceAxis(xv, 0, 0, n - k), sliceAxis(dv, 0, k, n))), n))
  }
  const taps = solveToeplitz(stack(r), undefined, stack(p))
  return { taps, output: lfilter(taps, [1], xv) }
}

/**
 * The local adaptive Wiener filter (scipy.signal.wiener): out = m + (1 - noise / v)(x - m) with the local mean m and
 * variance v over a window (zero-padded), the local mean where v is below the noise power (default the mean local
 * variance).
 *
 * @param x - Signal or image (any rank).
 * @param size - Window size, per axis or shared (default 3).
 * @param noise - Noise power (default the mean local variance).
 * @returns The filtered data.
 * @concept wiener-filter
 * @example
 * ```ts
 * import { wiener } from 'mdml-engine/signal'
 * wiener([1, 5, 2, 8, 3, 7, 4]).shape // => [7]
 * ```
 */
export function wiener(x: TensorLike, size: number | readonly number[] = 3, noise?: number): Value {
  const v = float(x)
  const rank = v.shape.length
  const k = typeof size === 'number' ? new Array<number>(rank).fill(size) : [...size]
  if (k.length !== rank) throw new MdmlError('shape', 'wiener: one window size per axis')
  const count = k.reduce((a, b) => a * b, 1)
  const box = ones(k)
  const m = div(fftConvolve(v, box, 'same'), count)
  const variance = sub(div(fftConvolve(mul(v, v), box, 'same'), count), mul(m, m))
  let nz: Value | number = noise ?? 0
  if (noise === undefined) nz = mean(variance)
  const res = add(mul(sub(v, m), sub(1, div(nz, variance))), m)
  return where(less(variance, nz), m, res)
}

/**
 * Linear prediction coefficients by Burg's method (librosa.lpc): the polynomial A(z) = 1 + a_1 z^-1 + ... + a_p z^-p
 * whose all-pole model 1 / A best predicts y (no mean removal).
 *
 * @param y - Signal.
 * @param order - Prediction order.
 * @returns [1, a_1, ..., a_p].
 * @concept linear-prediction
 * @example
 * ```ts
 * import { lpc } from 'mdml-engine/signal'
 * lpc([1, 0.5, 0.25, 0.125, 0.0625, 0.03125], 1).shape // => [2]
 * ```
 */
export function lpc(y: TensorLike, order: number): Value {
  const m = burg(y, order, false)
  return concat([constant([1]), neg(m.ar)])
}

/**
 * The prediction residual (innovation) of a linear predictor: e = A(z) y, the output of the FIR whitening filter.
 *
 * @param y - Signal.
 * @param a - Prediction polynomial [1, a_1, ..., a_p] (from {@link lpc}).
 * @returns The residual.
 * @concept linear-prediction
 * @example
 * ```ts
 * import { predictionResidual } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(predictionResidual([1, 0.5, 0.25], [1, -0.5])) // => [1, 0, 0]
 * ```
 */
export function predictionResidual(y: TensorLike, a: TensorLike): Value {
  return lfilter(a, [1], y)
}

/**
 * One-step linear prediction yhat[n] = -sum_k a_k y[n - k] (the predictor of {@link lpc}).
 *
 * @param y - Signal.
 * @param a - Prediction polynomial [1, a_1, ..., a_p].
 * @returns The predictions (y minus the residual).
 * @concept linear-prediction
 * @example
 * ```ts
 * import { linearPredict } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(linearPredict([1, 0.5, 0.25], [1, -0.5])) // => [0, 0.5, 0.25]
 * ```
 */
export function linearPredict(y: TensorLike, a: TensorLike): Value {
  const v = float(y)
  return sub(v, predictionResidual(v, a))
}

/**
 * The matched filter: correlation of the observation with the template (the time-reversed conjugate template as an
 * FIR filter), which maximises the output SNR for a known signal in white noise. Differentiable.
 *
 * @param x - Observation.
 * @param template - The known signal.
 * @param mode - 'valid' (default, template fully inside), 'same' or 'full'.
 * @returns The correlation output.
 * @concept matched-filter
 * @example
 * ```ts
 * import { matchedFilter } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(matchedFilter([0, 1, 2, 1, 0, 0], [1, 2, 1])) // => [4, 6, 4, 1]
 * ```
 */
export function matchedFilter(x: TensorLike, template: TensorLike, mode: 'valid' | 'same' | 'full' = 'valid'): Value {
  return correlate(vector(x, 'matchedFilter'), vector(template, 'matchedFilter'), mode)
}

/**
 * The output SNR of a matched filter for a template in white noise: E / sigma^2 with E the template energy.
 *
 * @param template - The known signal.
 * @param noiseVariance - Noise variance per sample.
 * @returns The SNR (linear).
 * @concept matched-filter
 * @example
 * ```ts
 * import { matchedFilterSnr } from 'mdml-engine/signal'
 * matchedFilterSnr([1, 2, 1], 0.5) // => 12
 * ```
 */
export function matchedFilterSnr(template: TensorLike, noiseVariance: number): number {
  let e = 0
  for (const v of toFlat(primalOf(float(template)))) e += v * v
  return e / noiseVariance
}
