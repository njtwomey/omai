/**
 * Finite impulse response (FIR) filter design, scipy.signal style: the window method (`firwin`) with Kaiser's order and
 * beta estimates, frequency sampling (`firwin2`), the Parks-McClellan equiripple design by the Remez exchange algorithm
 * (`remez`), and conversion of a linear-phase filter to minimum phase.
 */

import { OmaiError } from '../../base/index.ts'
import { fft, ifft, irfft } from '../../numerics/fourier/index.ts'
import { interp } from '../../numerics/interpolate/index.ts'
import { type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { constant, samples, sinc } from '../_shared.ts'
import { getWindow, type WindowSpec } from '../windows/index.ts'
import { type RemezType, remezKernel } from './_remez.ts'

// FIR filter design as scipy.signal: the window method (firwin), frequency sampling (firwin2), the Parks-McClellan
// minimax design (remez), Kaiser's order estimates, and conversion of a linear-phase filter to minimum phase.

/**
 * The Kaiser window beta for a stopband attenuation of a dB (scipy.signal.kaiser_beta).
 *
 * @param a - Attenuation in dB (positive).
 * @returns beta.
 * @concept fir-design
 * @example
 * ```ts
 * import { kaiserBeta } from 'omai-engine/signal'
 * Math.round(kaiserBeta(65) * 1e5) / 1e5 // => 6.20426
 * ```
 */
export function kaiserBeta(a: number): number {
  if (a > 50) return 0.1102 * (a - 8.7)
  if (a > 21) return 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
  return 0
}

/**
 * The attenuation of a Kaiser-window FIR filter with numtaps taps and a transition width (as a fraction of Nyquist)
 * (scipy.signal.kaiser_atten): 2.285 (numtaps - 1) pi width + 7.95.
 *
 * @param numtaps - Number of taps.
 * @param width - Transition width, as a fraction of the Nyquist frequency.
 * @returns The attenuation in dB.
 * @concept fir-design
 * @example
 * ```ts
 * import { kaiserAtten } from 'omai-engine/signal'
 * Math.round(kaiserAtten(211, 0.0375) * 1e4) / 1e4 // => 64.481
 * ```
 */
export function kaiserAtten(numtaps: number, width: number): number {
  return 2.285 * (numtaps - 1) * Math.PI * width + 7.95
}

/**
 * Kaiser's estimate of the taps and beta for a window-method FIR filter (scipy.signal.kaiserord).
 *
 * @param ripple - Ripple bound |H - D| in dB (the stopband attenuation), at least 8.
 * @param width - Transition width as a fraction of the Nyquist frequency.
 * @returns The number of taps and the Kaiser beta.
 * @throws OmaiError with code `value` for a ripple below 8 dB.
 * @concept fir-design
 * @example
 * ```ts
 * import { kaiserord } from 'omai-engine/signal'
 * const { numtaps, beta } = kaiserord(65, 0.0375)
 * [numtaps, Math.round(beta * 1e5) / 1e5] // => [213, 6.20426]
 * ```
 */
export function kaiserord(ripple: number, width: number): { readonly numtaps: number; readonly beta: number } {
  const a = Math.abs(ripple)
  if (a < 8) throw new OmaiError('value', `kaiserord: the ripple ${a} is too small for the Kaiser formula`)
  return { numtaps: Math.ceil((a - 7.95) / 2.285 / (Math.PI * width) + 1), beta: kaiserBeta(a) }
}

/** Options for {@link firwin}. */
export interface FirwinOptions {
  /** Transition width (in the units of fs); when given, a Kaiser window is designed for it. */
  readonly width?: number
  /** Window (default 'hamming'). */
  readonly window?: WindowSpec
  /** Whether DC is in a passband (default true), or a band type. */
  readonly passZero?: boolean | 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
  /** Scale for unit gain at the centre of the first passband (default true). */
  readonly scale?: boolean
  /** Sampling rate (default 2, so cutoffs are fractions of Nyquist). */
  readonly fs?: number
}

/**
 * A linear-phase FIR filter by the window method (scipy.signal.firwin): the ideal (sinc) response of the passbands
 * between the cutoffs, times a window, optionally scaled to unit gain.
 *
 * @param numtaps - Number of taps (odd when the response at Nyquist is non-zero).
 * @param cutoff - Cutoff frequency or band edges (in the units of fs, strictly increasing, in (0, fs / 2)).
 * @param options - Width, window, pass-zero, scaling and sampling rate.
 * @returns The taps `[numtaps]`.
 * @throws OmaiError with code `value` for invalid cutoffs or an even length passing Nyquist.
 * @concept fir-design
 * @example
 * ```ts
 * import { firwin } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(firwin(3, 0.5)).map((v) => Math.round(v * 1e8) / 1e8) // => [0.0462215, 0.907557, 0.0462215]
 * ```
 */
export function firwin(numtaps: number, cutoff: number | readonly number[], options: FirwinOptions = {}): Tensor {
  const nyq = (options.fs ?? 2) / 2
  const c = (typeof cutoff === 'number' ? [cutoff] : [...cutoff]).map((v) => v / nyq)
  if (c.length === 0) throw new OmaiError('value', 'firwin: at least one cutoff frequency is needed')
  if (c.some((v) => v <= 0 || v >= 1)) throw new OmaiError('value', 'firwin: cutoffs must lie in (0, fs / 2)')
  for (let i = 1; i < c.length; i++)
    if ((c[i] ?? 0) <= (c[i - 1] ?? 0)) throw new OmaiError('value', 'firwin: cutoffs must increase')
  let window: WindowSpec = options.window ?? 'hamming'
  if (options.width !== undefined) window = ['kaiser', kaiserBeta(kaiserAtten(numtaps, options.width / nyq))]
  let passZero = options.passZero ?? true
  if (passZero === 'lowpass' || passZero === 'bandstop') {
    if (passZero === 'lowpass' ? c.length !== 1 : c.length <= 1)
      throw new OmaiError('value', `firwin: wrong number of cutoffs for '${passZero}'`)
    passZero = true
  } else if (passZero === 'highpass' || passZero === 'bandpass') {
    if (passZero === 'highpass' ? c.length !== 1 : c.length <= 1)
      throw new OmaiError('value', `firwin: wrong number of cutoffs for '${passZero}'`)
    passZero = false
  }
  const passNyquist = (c.length % 2 === 0) === passZero
  if (passNyquist && numtaps % 2 === 0)
    throw new OmaiError('value', 'firwin: an even number of taps must have zero response at Nyquist')
  const edges = [...(passZero ? [0] : []), ...c, ...(passNyquist ? [1] : [])]
  const alpha = 0.5 * (numtaps - 1)
  const m = Array.from({ length: numtaps }, (_, i) => i - alpha)
  const h = new Float64Array(numtaps)
  for (let j = 0; j < edges.length; j += 2) {
    const left = edges[j] ?? 0
    const right = edges[j + 1] ?? 0
    for (const [i, mi] of m.entries()) h[i] = (h[i] ?? 0) + right * sinc(right * mi) - left * sinc(left * mi)
  }
  const win = toFlat(getWindow(window, numtaps, false))
  for (let i = 0; i < numtaps; i++) h[i] = (h[i] ?? 0) * (win[i] ?? 0)
  if (options.scale ?? true) {
    const left = edges[0] ?? 0
    const right = edges[1] ?? 0
    const f = left === 0 ? 0 : right === 1 ? 1 : 0.5 * (left + right)
    let s = 0
    for (const [i, mi] of m.entries()) s += (h[i] ?? 0) * Math.cos(Math.PI * mi * f)
    for (let i = 0; i < numtaps; i++) h[i] = (h[i] ?? 0) / s
  }
  return constant(h)
}

/** Options for {@link firwin2}. */
export interface Firwin2Options {
  /** Size of the interpolation grid (default 1 + 2^ceil(log2 numtaps)). */
  readonly nfreqs?: number
  /** Window (default 'hamming'; `null` for none). */
  readonly window?: WindowSpec | null
  /** Odd symmetry (types III and IV) instead of even (default false). */
  readonly antisymmetric?: boolean
  /** Sampling rate (default 2). */
  readonly fs?: number
}

/**
 * A linear-phase FIR filter by frequency sampling (scipy.signal.firwin2): interpolate the desired gains on a uniform
 * grid, inverse-FFT with the linear phase of the filter's delay, keep numtaps samples and window them.
 *
 * @param numtaps - Number of taps.
 * @param freq - Frequencies from 0 to fs / 2, non-decreasing (a value may repeat once for a step).
 * @param gain - Gains at those frequencies.
 * @param options - Grid size, window, symmetry and sampling rate.
 * @returns The taps `[numtaps]`.
 * @throws OmaiError with code `value` for invalid frequencies or gains impossible for the filter type.
 * @concept fir-design
 * @example
 * ```ts
 * import { firwin2 } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(firwin2(3, [0, 0.5, 1], [1, 1, 0])).map((v) => Math.round(v * 1e8) / 1e8) // => [0.01707107, 0.75, 0.01707107]
 * ```
 */
export function firwin2(
  numtaps: number,
  freq: readonly number[],
  gain: readonly number[],
  options: Firwin2Options = {},
): Tensor {
  const nyq = (options.fs ?? 2) / 2
  if (freq.length !== gain.length) throw new OmaiError('value', 'firwin2: freq and gain must have the same length')
  const nfreqs = options.nfreqs ?? 1 + 2 ** Math.ceil(Math.log2(numtaps))
  if (numtaps >= nfreqs) throw new OmaiError('value', 'firwin2: numtaps must be less than nfreqs')
  if (freq[0] !== 0 || freq[freq.length - 1] !== nyq)
    throw new OmaiError('value', 'firwin2: freq must run from 0 to fs / 2')
  const f = [...freq]
  for (let k = 1; k < f.length; k++) {
    if ((f[k] ?? 0) < (f[k - 1] ?? 0)) throw new OmaiError('value', 'firwin2: freq must be non-decreasing')
    if (k >= 2 && f[k] === f[k - 2]) throw new OmaiError('value', 'firwin2: a frequency may occur at most twice')
  }
  if (f[1] === 0 || f[f.length - 2] === nyq) throw new OmaiError('value', 'firwin2: 0 and fs / 2 must not repeat')
  const anti = options.antisymmetric ?? false
  const ftype = anti ? (numtaps % 2 === 0 ? 4 : 3) : numtaps % 2 === 0 ? 2 : 1
  const g0 = gain[0] ?? 0
  const gl = gain[gain.length - 1] ?? 0
  if ((ftype === 2 && gl !== 0) || (ftype === 3 && (g0 !== 0 || gl !== 0)) || (ftype === 4 && g0 !== 0))
    throw new OmaiError('value', `firwin2: a type ${ftype} filter cannot have this gain at 0 or Nyquist`)
  const eps = Number.EPSILON * nyq
  for (let k = 0; k + 1 < f.length; k++) {
    if (f[k] === f[k + 1]) {
      f[k] = (f[k] ?? 0) - eps
      f[k + 1] = (f[k + 1] ?? 0) + eps
    }
  }
  const x = Array.from({ length: nfreqs }, (_, i) => (nfreqs === 1 ? 0 : (i * nyq) / (nfreqs - 1)))
  const fx = toFlat(interp(constant(x), constant(f), constant([...gain])))
  const re = new Float64Array(nfreqs)
  const im = new Float64Array(nfreqs)
  for (let i = 0; i < nfreqs; i++) {
    let t = (-(numtaps - 1) / 2) * Math.PI * ((x[i] ?? 0) / nyq)
    if (ftype > 2) t += Math.PI / 2
    re[i] = (fx[i] ?? 0) * Math.cos(t)
    im[i] = (fx[i] ?? 0) * Math.sin(t)
  }
  const full = toFlat(irfft({ re: constant(re), im: constant(im) }))
  const win = options.window === null ? null : toFlat(getWindow(options.window ?? 'hamming', numtaps, false))
  const out = Float64Array.from({ length: numtaps }, (_, i) => (full[i] ?? 0) * (win === null ? 1 : (win[i] ?? 0)))
  if (ftype === 3) out[Math.floor(numtaps / 2)] = 0
  return constant(out)
}

/** Options for {@link remez}. */
export interface RemezOptions {
  /** Relative weight of each band (default ones). */
  readonly weight?: readonly number[]
  /** 'bandpass' (default), 'differentiator' or 'hilbert'. */
  readonly type?: RemezType
  /** Sampling rate (default 1, so band edges are in [0, 0.5]). */
  readonly fs?: number
  /** Maximum exchange iterations (default 25). */
  readonly maxiter?: number
  /** Grid density (default 16). */
  readonly gridDensity?: number
}

/**
 * The equiripple minimax FIR filter by the Parks-McClellan (Remez exchange) algorithm (scipy.signal.remez): the
 * linear-phase filter minimising the maximum weighted deviation from the desired gain in each band. It is the exchange
 * of Janovetz's remez.c on the filter-design grid (cosine-basis types I to IV, band weights); `remez` of
 * `numerics/approximation` is the same equioscillation idea for a general function, polynomial or rational, on an
 * interval, and the two do not share code.
 *
 * @param numtaps - Number of taps.
 * @param bands - Band edges, pairs from 0 to fs / 2.
 * @param desired - Desired gain in each band.
 * @param options - Weights, type, sampling rate, iterations and grid density.
 * @returns The taps `[numtaps]`.
 * @throws OmaiError with code `convergence` when the exchange does not converge.
 * @concept fir-design
 * @example
 * ```ts
 * import { remez } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * const h = toFlat(remez(11, [0, 0.1, 0.2, 0.5], [1, 0]))
 * Math.abs(h.reduce((a, v) => a + v, 0) - 1) < 0.05 // => true
 * ```
 */
export function remez(
  numtaps: number,
  bands: readonly number[],
  desired: readonly number[],
  options: RemezOptions = {},
): Tensor {
  const fs = options.fs ?? 1
  if (bands.length !== 2 * desired.length)
    throw new OmaiError('value', 'remez: bands must be pairs, one per desired gain')
  const weight = options.weight ?? desired.map(() => 1)
  if (weight.length !== desired.length) throw new OmaiError('value', 'remez: one weight per band')
  for (const b of bands)
    if (b < 0 || b > fs / 2) throw new OmaiError('value', 'remez: band edges must lie in [0, fs / 2]')
  return constant(
    remezKernel(
      numtaps,
      bands.map((b) => b / fs),
      desired,
      weight,
      options.type ?? 'bandpass',
      options.maxiter ?? 25,
      options.gridDensity ?? 16,
    ),
  )
}

/** Options for {@link minimumPhase}. */
export interface MinimumPhaseOptions {
  /** 'homomorphic' (default, by the cepstrum) or 'hilbert' (for equiripple filters). */
  readonly method?: 'homomorphic' | 'hilbert'
  /** FFT length (default a power of two near 2 (len - 1) / 0.01). */
  readonly nFft?: number
  /** Halve the magnitude (default true, so the result has about half the taps and the square root response). */
  readonly half?: boolean
}

function realIfftOf(re: Float64Array, im: Float64Array): { re: Float64Array; im: Float64Array } {
  const z = ifft({ re: constant(re), im: constant(im) })
  return { re: Float64Array.from(toFlat(z.re)), im: Float64Array.from(toFlat(z.im)) }
}

function fftOf(re: Float64Array, im: Float64Array): { re: Float64Array; im: Float64Array } {
  const z = fft({ re: constant(re), im: constant(im) })
  return { re: Float64Array.from(toFlat(z.re)), im: Float64Array.from(toFlat(z.im)) }
}

/** exp of a complex array. */
function cexp(z: { re: Float64Array; im: Float64Array }): { re: Float64Array; im: Float64Array } {
  return {
    re: Float64Array.from(z.re, (r, i) => Math.exp(r) * Math.cos(z.im[i] ?? 0)),
    im: Float64Array.from(z.re, (r, i) => Math.exp(r) * Math.sin(z.im[i] ?? 0)),
  }
}

/**
 * Convert a linear-phase FIR filter to minimum phase (scipy.signal.minimum_phase), by the homomorphic (cepstral)
 * method or the Hilbert method of Damera-Venkata, Evans and McCaslin. With `half` the magnitude response is the
 * square root of the original's.
 *
 * @param h - Linear-phase (symmetric) taps.
 * @param options - Method, FFT length and halving.
 * @returns The minimum-phase taps.
 * @throws OmaiError with code `value` for an FFT shorter than h or 'hilbert' without halving.
 * @concept minimum-phase
 * @example
 * ```ts
 * import { minimumPhase } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(minimumPhase([0.25, 0.5, 0.25])).map((v) => Math.round(v * 1e4) / 1e4) // => [0.494, 0.5058]
 * ```
 */
export function minimumPhase(h: TensorLike, options: MinimumPhaseOptions = {}): Tensor {
  const taps = samples(h, 'minimumPhase')
  const n = taps.length
  const nHalf = Math.floor(n / 2)
  const method = options.method ?? 'homomorphic'
  const half = options.half ?? true
  if (method === 'hilbert' && !half) throw new OmaiError('value', "minimumPhase: half = false needs 'homomorphic'")
  const nFft = options.nFft ?? 2 ** Math.ceil(Math.log2((2 * (n - 1)) / 0.01))
  if (nFft < n) throw new OmaiError('value', `minimumPhase: nFft must be at least ${n}`)
  const padded = new Float64Array(nFft)
  padded.set(taps)
  const spec = fftOf(padded, new Float64Array(nFft))
  let result: Float64Array
  if (method === 'hilbert') {
    const hr = Float64Array.from({ length: nFft }, (_, k) => {
      const w = ((2 * Math.PI) / nFft) * nHalf * k
      return (spec.re[k] ?? 0) * Math.cos(w) - (spec.im[k] ?? 0) * Math.sin(w)
    })
    let hi = Number.NEGATIVE_INFINITY
    let lo = Number.POSITIVE_INFINITY
    for (const v of hr) {
      hi = Math.max(hi, v)
      lo = Math.min(lo, v)
    }
    const dp = hi - 1
    const ds = -lo
    const s = 4 / (Math.sqrt(1 + dp + ds) + Math.sqrt(1 - dp + ds)) ** 2
    const mag = Float64Array.from(hr, (v) => Math.sqrt((v + ds) * s) + 1e-10)
    // The modified discrete Hilbert transform of the log magnitude.
    const sig = new Float64Array(nFft)
    const mid = Math.floor(nFft / 2)
    for (let k = 1; k < mid; k++) sig[k] = 1
    for (let k = mid + 1; k < nFft; k++) sig[k] = -1
    const lm = realIfftOf(Float64Array.from(mag, Math.log), new Float64Array(nFft))
    const prod = fftOf(
      Float64Array.from(lm.re, (v, k) => v * (sig[k] ?? 0)),
      Float64Array.from(lm.im, (v, k) => v * (sig[k] ?? 0)),
    )
    const e = cexp(prod)
    result = realIfftOf(
      Float64Array.from(e.re, (v, k) => v * (mag[k] ?? 0)),
      Float64Array.from(e.im, (v, k) => v * (mag[k] ?? 0)),
    ).re
  } else {
    const mag = Float64Array.from(spec.re, (r, k) => Math.hypot(r, spec.im[k] ?? 0))
    let floor = Number.POSITIVE_INFINITY
    for (const v of mag) if (v > 0) floor = Math.min(floor, v)
    const logMag = Float64Array.from(mag, (v) => Math.log(v + 1e-7 * floor) * (half ? 0.5 : 1))
    const cep = realIfftOf(logMag, new Float64Array(nFft)).re
    const stop = Math.floor(nFft / 2)
    for (let k = 1; k < stop; k++) cep[k] = (cep[k] ?? 0) * 2
    cep[stop] = (cep[stop] ?? 0) * (1 + (nFft % 2))
    for (let k = stop + 1; k < nFft; k++) cep[k] = 0
    const e = cexp(fftOf(cep, new Float64Array(nFft)))
    result = realIfftOf(e.re, e.im).re
  }
  const nOut = half ? nHalf + (n % 2) : n
  return constant(result.slice(0, nOut))
}
