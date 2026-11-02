/**
 * The continuous wavelet transform (pywt.cwt): the correlation of a signal with scaled copies of a wavelet such as the
 * Morlet or Mexican hat, giving a scalogram of coefficients by scale and time, plus the wavelet's centre frequency that
 * converts scales to frequencies.
 */

import { MdmlError } from '../../base/index.ts'
import { convolve } from '../../numerics/convolution/index.ts'
import { fft } from '../../numerics/fourier/index.ts'
import {
  type ComplexValue,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  stack,
  sub,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { constant, vector } from '../_shared.ts'
import { continuousWavelet } from './families.ts'

// The continuous wavelet transform as pywt.cwt (method 'conv'): the signal convolved with the integrated wavelet
// resampled at each scale, then differentiated, which equals correlation with the scaled wavelet.

/**
 * The centre frequency of a continuous wavelet in cycles per unit of its support (pywt.central_frequency): the
 * strongest non-DC bin of its DFT over the sampled support.
 *
 * @param name - The continuous wavelet.
 * @param precision - log2 of the number of samples (default 8).
 * @returns The centre frequency.
 * @concept continuous-wavelet-transform
 * @example
 * ```ts
 * import { centralFrequency } from 'mdml-engine/signal'
 * Math.round(centralFrequency('morl') * 1e4) / 1e4 // => 0.8125
 * ```
 */
export function centralFrequency(name: string, precision = 8): number {
  const w = continuousWavelet(name, precision)
  const x = toFlat(w.x)
  const spec = fft({ re: w.re, im: w.im })
  const re = toFlat(spec.re)
  const im = toFlat(spec.im)
  const n = re.length
  let best = 1
  for (let k = 1; k < n; k++)
    if (Math.hypot(re[k] ?? 0, im[k] ?? 0) > Math.hypot(re[best] ?? 0, im[best] ?? 0)) best = k
  let index = best + 1
  if (index > n / 2) index = n - index + 2
  const domain = (x[n - 1] ?? 0) - (x[0] ?? 0)
  return 1 / (domain / (index - 1))
}

/** The result of a continuous wavelet transform: coefficients `[scales, n]` and the frequency of each scale. */
export interface CwtResult {
  readonly coefficients: ComplexValue
  readonly frequencies: number[]
}

/**
 * The continuous wavelet transform (pywt.cwt): for each scale a, -sqrt(a) d/dt (x * Psi_a), where Psi_a is the
 * integral of the wavelet (on 2^12 points, as pywt) sampled at the scale; frequencies are centralFrequency / (a samplingPeriod).
 * Differentiable in x.
 *
 * @param x - Real signal `[n]`.
 * @param scales - Scales (positive).
 * @param name - Continuous wavelet ('morl', 'mexh', 'gausP', 'cmorB-C', 'shanB-C').
 * @param samplingPeriod - Sampling period (default 1).
 * @returns The coefficients (imaginary part zero for real wavelets) and frequencies.
 * @throws MdmlError with code `value` for a scale too small for the signal.
 * @concept continuous-wavelet-transform
 * @index function Continuous wavelet transform.
 * @example
 * ```ts
 * import { cwt } from 'mdml-engine/signal'
 * cwt(new Array(32).fill(0).map((_, i) => Math.sin(i / 2)), [1, 2, 4], 'mexh').coefficients.re.shape // => [3, 32]
 * ```
 */
export function cwt(x: TensorLike, scales: readonly number[], name: string, samplingPeriod = 1): CwtResult {
  const data = vector(x, 'cwt')
  const n = data.shape[0] ?? 0
  const precision = 12
  const w = continuousWavelet(name, precision)
  const grid = toFlat(w.x)
  const step = (grid[1] ?? 0) - (grid[0] ?? 0)
  const psiRe = toFlat(primalOf(w.re))
  const psiIm = toFlat(primalOf(w.im))
  // Integrated wavelet (conjugated for complex wavelets).
  const intRe = new Float64Array(psiRe.length)
  const intIm = new Float64Array(psiRe.length)
  let cr = 0
  let ci = 0
  for (let i = 0; i < psiRe.length; i++) {
    cr += psiRe[i] ?? 0
    ci += psiIm[i] ?? 0
    intRe[i] = cr * step
    intIm[i] = -ci * step
  }
  const width = (grid[grid.length - 1] ?? 0) - (grid[0] ?? 0)
  const rows: Value[] = []
  const irows: Value[] = []
  for (const a of scales) {
    if (!(a > 0)) throw new MdmlError('value', 'cwt: scales must be positive')
    const count = Math.ceil(a * width + 1)
    const j: number[] = []
    for (let i = 0; i < count; i++) {
      const v = Math.trunc(i / (a * step))
      if (v < intRe.length) j.push(v)
    }
    const kRe = constant(j.map((v) => intRe[v] ?? 0).reverse())
    const kIm = constant(j.map((v) => intIm[v] ?? 0).reverse())
    const transform = (k: Value): Value => {
      const conv = convolve(data, k)
      const len = conv.shape[0] ?? 0
      const coef = mul(sub(sliceAxis(conv, 0, 1, len), sliceAxis(conv, 0, 0, len - 1)), -Math.sqrt(a))
      const d = (len - 1 - n) / 2
      if (d < 0) throw new MdmlError('value', 'cwt: the selected scale is too small')
      return sliceAxis(coef, 0, Math.floor(d), len - 1 - Math.ceil(d))
    }
    rows.push(transform(kRe))
    irows.push(w.complex ? transform(kIm) : constant(new Float64Array(n)))
  }
  const fc = centralFrequency(name, precision)
  return {
    coefficients: { re: reshape(stack(rows), [scales.length, n]), im: reshape(stack(irows), [scales.length, n]) },
    frequencies: scales.map((a) => fc / a / samplingPeriod),
  }
}
