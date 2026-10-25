/**
 * The wavelets themselves: the discrete families (Haar, Daubechies, symlets, coiflets, biorthogonal and others) with
 * their four filters by name as pywt.Wavelet, the deepest useful decomposition level, and continuous wavelets (Morlet,
 * Mexican hat, Gaussian derivatives) sampled on their support.
 */

import { MdmlError } from '../../base/index.ts'
import type { Tensor } from '../../tensor/index.ts'
import { constant } from '../_shared.ts'
import { FILTERS } from './_filters.ts'

// Wavelet families as PyWavelets: the discrete wavelets (Haar, Daubechies, Symlets, Coiflets, biorthogonal and
// reverse biorthogonal, discrete Meyer) by their filter banks, and the continuous wavelets of the CWT by formula.

/** A discrete wavelet: its four filters (pywt's filter_bank order) and whether it is orthogonal. */
export interface DiscreteWavelet {
  readonly name: string
  readonly decLo: Float64Array
  readonly decHi: Float64Array
  readonly recLo: Float64Array
  readonly recHi: Float64Array
  readonly orthogonal: boolean
}

/** A wavelet given by name, or by its decomposition and reconstruction lowpass filters. */
export type WaveletSpec = string | DiscreteWavelet

/**
 * The names of the discrete wavelets available (pywt.wavelist(kind='discrete')).
 *
 * @param family - Restrict to a family prefix such as 'db', 'sym', 'coif', 'bior', 'rbio' (default all).
 * @returns The names.
 * @concept wavelets
 * @example
 * ```ts
 * import { waveletList } from 'mdml-engine/signal'
 * waveletList('coif').slice(0, 3) // => ['coif1', 'coif2', 'coif3']
 * ```
 */
export function waveletList(family?: string): string[] {
  const names = Object.keys(FILTERS)
  return family === undefined
    ? names
    : names.filter((n) => n.startsWith(family) && /\d/.test(n.slice(family.length, family.length + 1)))
}

/**
 * A discrete wavelet by name (pywt.Wavelet): 'haar', 'db1' .. 'db20', 'sym2' .. 'sym20', 'coif1' .. 'coif17',
 * 'biorN.M', 'rbioN.M' and 'dmey'. The highpass filters follow from the lowpass ones: dec_hi[k] = (-1)^(k+1)
 * rec_lo[k] and rec_hi[k] = (-1)^k dec_lo[k].
 *
 * @param name - The wavelet's name, or a wavelet (returned as is).
 * @returns The filter bank.
 * @throws MdmlError with code `value` for an unknown name.
 * @concept wavelets
 * @example
 * ```ts
 * import { wavelet } from 'mdml-engine/signal'
 * const w = wavelet('db2')
 * [w.decLo.length, w.orthogonal] // => [4, true]
 * ```
 */
export function wavelet(name: WaveletSpec): DiscreteWavelet {
  if (typeof name !== 'string') return name
  const f = FILTERS[name]
  if (f === undefined) throw new MdmlError('value', `wavelet: unknown wavelet '${name}'`)
  const decLo = Float64Array.from(f[0])
  const recLo = Float64Array.from(f[1])
  const decHi = Float64Array.from(recLo, (v, k) => (k % 2 === 0 ? -v : v))
  const recHi = Float64Array.from(decLo, (v, k) => (k % 2 === 0 ? v : -v))
  const orthogonal = decLo.every((v, k) => Math.abs(v - (recLo[decLo.length - 1 - k] ?? 0)) < 1e-14)
  return { name, decLo, decHi, recLo, recHi, orthogonal }
}

/**
 * The deepest useful decomposition level (pywt.dwt_max_level): floor(log2(n / (F - 1))) for a signal of n samples
 * and a filter of F taps, or 0 when the signal is shorter than the filter.
 *
 * @param n - Signal length.
 * @param w - Wavelet, or its filter length.
 * @returns The level.
 * @concept wavelets
 * @example
 * ```ts
 * import { dwtMaxLevel } from 'mdml-engine/signal'
 * dwtMaxLevel(1000, 'db4') // => 7
 * ```
 */
export function dwtMaxLevel(n: number, w: WaveletSpec | number): number {
  const f = typeof w === 'number' ? w : wavelet(w).decLo.length
  if (f < 2 || n < f - 1) return 0
  return Math.floor(Math.log2(n / (f - 1)))
}

/** A continuous wavelet sampled on its support. */
export interface WaveletFunction {
  readonly x: Tensor
  readonly re: Tensor
  /** The imaginary part (zeros for real wavelets). */
  readonly im: Tensor
  readonly complex: boolean
}

function doubleFactorial(n: number): number {
  let v = 1
  for (let k = n; k > 1; k -= 2) v *= k
  return v
}

/** The physicists' Hermite polynomial H_p(x) by the three-term recurrence. */
function hermite(p: number, x: number): number {
  let h0 = 1
  if (p === 0) return h0
  let h1 = 2 * x
  for (let k = 1; k < p; k++) {
    const h2 = 2 * x * h1 - 2 * k * h0
    h0 = h1
    h1 = h2
  }
  return h1
}

/**
 * Sample a continuous wavelet on 2^precision points of its support (pywt.ContinuousWavelet(name).wavefun): 'morl'
 * (Morlet, exp(-x^2/2) cos 5x), 'mexh' (Mexican hat), 'gaus1' .. 'gaus8' (normalised derivatives of a Gaussian),
 * 'cmorB-C' (complex Morlet with bandwidth B and centre frequency C) and 'shanB-C' (Shannon).
 *
 * @param name - The continuous wavelet.
 * @param precision - log2 of the number of samples (default 8).
 * @returns The grid and the real and imaginary parts.
 * @throws MdmlError with code `value` for an unknown wavelet.
 * @concept continuous-wavelet-transform
 * @example
 * ```ts
 * import { continuousWavelet } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * const w = continuousWavelet('mexh', 4)
 * [toFlat(w.x)[0], w.complex] // => [-8, false]
 * ```
 */
export function continuousWavelet(name: string, precision = 8): WaveletFunction {
  const n = 2 ** precision
  const grid = (lo: number, hi: number): Float64Array =>
    Float64Array.from({ length: n }, (_, i) => (n === 1 ? lo : lo + ((hi - lo) * i) / (n - 1)))
  let x: Float64Array
  let re: Float64Array
  let im = new Float64Array(n)
  let complex = false
  if (name === 'morl') {
    x = grid(-8, 8)
    re = Float64Array.from(x, (v) => Math.exp(-(v * v) / 2) * Math.cos(5 * v))
  } else if (name === 'mexh') {
    x = grid(-8, 8)
    const c = 2 / (Math.sqrt(3) * Math.PI ** 0.25)
    re = Float64Array.from(x, (v) => c * (1 - v * v) * Math.exp(-(v * v) / 2))
  } else if (/^gaus[1-8]$/.test(name)) {
    const p = Number(name.slice(4))
    x = grid(-5, 5)
    const c = ((Math.floor(p / 2) % 2 === 0 ? 1 : -1) * (2 / Math.PI) ** 0.25) / Math.sqrt(doubleFactorial(2 * p - 1))
    re = Float64Array.from(x, (v) => c * (p % 2 === 0 ? 1 : -1) * hermite(p, v) * Math.exp(-v * v))
  } else {
    const m = /^(cmor|shan)(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$/.exec(name)
    if (m === null) throw new MdmlError('value', `continuousWavelet: unknown wavelet '${name}'`)
    const b = Number(m[2])
    const fc = Number(m[3])
    complex = true
    if (m[1] === 'cmor') {
      x = grid(-8, 8)
      const amp = Float64Array.from(x, (v) => Math.exp(-(v * v) / b) / Math.sqrt(Math.PI * b))
      re = Float64Array.from(x, (v, i) => (amp[i] ?? 0) * Math.cos(2 * Math.PI * fc * v))
      im = Float64Array.from(x, (v, i) => (amp[i] ?? 0) * Math.sin(2 * Math.PI * fc * v))
    } else {
      x = grid(-20, 20)
      const amp = Float64Array.from(x, (v) => {
        const t = Math.PI * b * v
        return Math.sqrt(b) * (t === 0 ? 1 : Math.sin(t) / t)
      })
      re = Float64Array.from(x, (v, i) => (amp[i] ?? 0) * Math.cos(2 * Math.PI * fc * v))
      im = Float64Array.from(x, (v, i) => (amp[i] ?? 0) * Math.sin(2 * Math.PI * fc * v))
    }
  }
  return { x: constant(x), re: constant(re), im: constant(im), complex }
}
