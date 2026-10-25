/**
 * Infinite impulse response (IIR) filter design, scipy.signal style: `iirfilter` for any family and band, the named
 * designs (Butterworth, Chebyshev I and II, elliptic, Bessel) returned as transfer-function coefficients,
 * zeros-poles-gain or second-order sections, and the second-order notch, peak and comb filters.
 */

import { MdmlError } from '../../base/index.ts'
import {
  type SecondOrderSections,
  sos,
  type TransferFunction,
  tf,
  type ZerosPolesGain,
  type ZerosPolesGainData,
  zpk,
  zpk2sos,
  zpk2tf,
} from '../../systems/index.ts'
import {
  type BesselNorm,
  besselap,
  bilinearZpk,
  buttap,
  cheb1ap,
  cheb2ap,
  ellipap,
  lp2bpZpk,
  lp2bsZpk,
  lp2hpZpk,
  lp2lpZpk,
} from './prototypes.ts'

// IIR filter design as scipy.signal: prototypes transformed to the band and digitised by the bilinear transform with
// pre-warping (iirfilter), the named families (butter, cheby1, cheby2, ellip, bessel), notch, peak and comb filters.
// Designs are systems-domain values: a transfer function ('ba', coefficients in powers of z^-1 for digital filters),
// zeros-poles-gain ('zpk') or second-order sections ('sos'). Digital designs carry dt = 1 / fs, with fs = 2 when
// frequencies are given as fractions of the Nyquist frequency; analog designs carry dt = 0.

/** The band of a filter. */
export type BandType = 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'

/** The family of an IIR design. */
export type IirFamily = 'butter' | 'cheby1' | 'cheby2' | 'ellip' | 'bessel'

/** Options shared by the IIR designs. */
export interface IirOptions {
  /** Band type (default lowpass). */
  readonly btype?: BandType
  /** Analog design (frequencies in rad/s, no bilinear transform). Default false. */
  readonly analog?: boolean
  /** Sampling rate of a digital design, the units of the frequencies (default: fractions of Nyquist). */
  readonly fs?: number
}

/** A design's output form. */
export type FilterOutput = 'ba' | 'zpk' | 'sos'

/** A design in one of the three output forms. */
export type FilterDesign = TransferFunction | ZerosPolesGain | SecondOrderSections

/** The design type of an output form: 'ba' a transfer function, 'zpk' zeros-poles-gain, 'sos' sections. */
export type DesignOf<O extends FilterOutput> = O extends 'zpk'
  ? ZerosPolesGain
  : O extends 'sos'
    ? SecondOrderSections
    : TransferFunction

/** Options of {@link iirfilter}: the family, its ripples and the output form. */
export interface IirFilterOptions<O extends FilterOutput = 'ba'> extends IirOptions {
  readonly ftype?: IirFamily
  /** Passband ripple in dB (Chebyshev I, elliptic). */
  readonly rp?: number
  /** Stopband attenuation in dB (Chebyshev II, elliptic). */
  readonly rs?: number
  /** Bessel normalisation (default 'phase'). */
  readonly norm?: BesselNorm
  readonly output?: O
}

function finish(s: ZerosPolesGainData, analog: boolean, fs: number | undefined, output: FilterOutput): FilterDesign {
  const dt = analog ? 0 : 1 / (fs ?? 2)
  if (output === 'zpk') return zpk(s.zeros, s.poles, s.gain, { dt })
  if (output === 'sos') return sos(zpk2sos(s.zeros, s.poles, s.gain, { analog }), { dt })
  const ba = zpk2tf(s.zeros, s.poles, s.gain)
  return tf(ba.num, ba.den, { dt })
}

/**
 * Design an IIR filter of any family and band (scipy.signal.iirfilter): the analog prototype, transformed to the band
 * at the pre-warped frequencies, then (for digital filters) the bilinear transform.
 *
 * @param n - Order.
 * @param wn - Critical frequency, or two band edges.
 * @param options - Family, ripples, band, analog or digital, sampling rate, output form.
 * @returns The design as a transfer function, zeros-poles-gain or second-order sections.
 * @throws MdmlError with code `value` for invalid frequencies or missing ripples.
 * @concept iir-design
 * @example
 * ```ts
 * import { iirfilter } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(iirfilter(1, 0.5).num).map((v) => Math.round(v * 1e12) / 1e12) // => [0.5, 0.5]
 * ```
 */
export function iirfilter<O extends FilterOutput = 'ba'>(
  n: number,
  wn: number | readonly number[],
  options: IirFilterOptions<O> = {},
): DesignOf<O> {
  const analog = options.analog ?? false
  if (analog && options.fs !== undefined) throw new MdmlError('value', 'iirfilter: fs is for digital filters only')
  let w = typeof wn === 'number' ? [wn] : [...wn]
  if (options.fs !== undefined) w = w.map((v) => v / ((options.fs as number) / 2))
  if (w.some((v) => v <= 0)) throw new MdmlError('value', 'iirfilter: critical frequencies must be positive')
  if (w.length > 1 && !((w[0] ?? 0) < (w[1] ?? 0)))
    throw new MdmlError('value', 'iirfilter: Wn[0] must be less than Wn[1]')
  const ftype = options.ftype ?? 'butter'
  const { rp, rs } = options
  if (rp !== undefined && rp < 0) throw new MdmlError('value', 'iirfilter: rp must be positive')
  if (rs !== undefined && rs < 0) throw new MdmlError('value', 'iirfilter: rs must be positive')
  let s: ZerosPolesGainData
  switch (ftype) {
    case 'butter':
      s = buttap(n)
      break
    case 'bessel':
      s = besselap(n, options.norm ?? 'phase')
      break
    case 'cheby1':
      if (rp === undefined) throw new MdmlError('value', 'iirfilter: a Chebyshev I filter needs rp')
      s = cheb1ap(n, rp)
      break
    case 'cheby2':
      if (rs === undefined) throw new MdmlError('value', 'iirfilter: a Chebyshev II filter needs rs')
      s = cheb2ap(n, rs)
      break
    case 'ellip':
      if (rp === undefined || rs === undefined)
        throw new MdmlError('value', 'iirfilter: an elliptic filter needs rp and rs')
      s = ellipap(n, rp, rs)
      break
  }
  let warped = w
  if (!analog) {
    if (w.some((v) => v >= 1))
      throw new MdmlError('value', 'iirfilter: digital critical frequencies must be below Nyquist')
    warped = w.map((v) => 4 * Math.tan((Math.PI * v) / 2))
  }
  const btype = options.btype ?? 'lowpass'
  if (btype === 'lowpass' || btype === 'highpass') {
    if (warped.length !== 1) throw new MdmlError('value', `iirfilter: a ${btype} filter needs one critical frequency`)
    s = btype === 'lowpass' ? lp2lpZpk(s, warped[0]) : lp2hpZpk(s, warped[0])
  } else {
    if (warped.length !== 2) throw new MdmlError('value', `iirfilter: a ${btype} filter needs two band edges`)
    const bw = (warped[1] ?? 0) - (warped[0] ?? 0)
    const wo = Math.sqrt((warped[0] ?? 0) * (warped[1] ?? 0))
    s = btype === 'bandpass' ? lp2bpZpk(s, wo, bw) : lp2bsZpk(s, wo, bw)
  }
  if (!analog) s = bilinearZpk(s, 2)
  return finish(s, analog, options.fs, options.output ?? 'ba') as DesignOf<O>
}

/** Options of the named designs: band, analog, sampling rate and output. */
export interface DesignOptions<O extends FilterOutput = 'ba'> extends IirOptions {
  readonly output?: O
}

/**
 * A Butterworth filter (scipy.signal.butter): maximally flat passband.
 *
 * @param n - Order.
 * @param wn - Cutoff (-3 dB), or band edges.
 * @param options - Band, analog, sampling rate and output.
 * @returns The design.
 * @concept iir-design
 * @example
 * ```ts
 * import { butter } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(butter(2, 0.5).den).map((v) => Math.round(v * 1e9) / 1e9) // => [1, 0, 0.171572875]
 * ```
 */
export function butter<O extends FilterOutput = 'ba'>(
  n: number,
  wn: number | readonly number[],
  options: DesignOptions<O> = {},
): DesignOf<O> {
  return iirfilter<O>(n, wn, { ...options, ftype: 'butter' })
}

/**
 * A Chebyshev type I filter (scipy.signal.cheby1): equiripple passband of rp dB, steeper roll-off.
 *
 * @param n - Order.
 * @param rp - Passband ripple in dB.
 * @param wn - Passband edge (where the gain first drops below -rp dB), or band edges.
 * @param options - Band, analog, sampling rate and output.
 * @returns The design.
 * @concept iir-design
 * @example
 * ```ts
 * import { cheby1 } from 'mdml-engine/signal'
 * cheby1(4, 1, 0.3, { output: 'zpk' }).kind // => 'zpk'
 * ```
 */
export function cheby1<O extends FilterOutput = 'ba'>(
  n: number,
  rp: number,
  wn: number | readonly number[],
  options: DesignOptions<O> = {},
): DesignOf<O> {
  return iirfilter<O>(n, wn, { ...options, rp, ftype: 'cheby1' })
}

/**
 * A Chebyshev type II filter (scipy.signal.cheby2): monotonic passband, equiripple stopband rs dB down.
 *
 * @param n - Order.
 * @param rs - Stopband attenuation in dB.
 * @param wn - Stopband edge (where the gain first reaches -rs dB), or band edges.
 * @param options - Band, analog, sampling rate and output.
 * @returns The design.
 * @concept iir-design
 * @example
 * ```ts
 * import { cheby2 } from 'mdml-engine/signal'
 * cheby2(4, 40, 0.3, { output: 'sos' }).kind // => 'sos'
 * ```
 */
export function cheby2<O extends FilterOutput = 'ba'>(
  n: number,
  rs: number,
  wn: number | readonly number[],
  options: DesignOptions<O> = {},
): DesignOf<O> {
  return iirfilter<O>(n, wn, { ...options, rs, ftype: 'cheby2' })
}

/**
 * An elliptic (Cauer) filter (scipy.signal.ellip): equiripple in both bands, the steepest transition for its order.
 *
 * @param n - Order.
 * @param rp - Passband ripple in dB.
 * @param rs - Stopband attenuation in dB.
 * @param wn - Passband edge, or band edges.
 * @param options - Band, analog, sampling rate and output.
 * @returns The design.
 * @concept iir-design
 * @example
 * ```ts
 * import { ellip } from 'mdml-engine/signal'
 * ellip(4, 1, 40, [0.2, 0.4], { btype: 'bandpass', output: 'zpk' }).kind // => 'zpk'
 * ```
 */
export function ellip<O extends FilterOutput = 'ba'>(
  n: number,
  rp: number,
  rs: number,
  wn: number | readonly number[],
  options: DesignOptions<O> = {},
): DesignOf<O> {
  return iirfilter<O>(n, wn, { ...options, rp, rs, ftype: 'ellip' })
}

/**
 * A Bessel (Thomson) filter (scipy.signal.bessel): maximally flat group delay, so waveforms keep their shape.
 *
 * @param n - Order.
 * @param wn - Critical frequency (its meaning set by the normalisation), or band edges.
 * @param options - Band, analog, sampling rate, output and normalisation.
 * @returns The design.
 * @concept iir-design
 * @example
 * ```ts
 * import { bessel } from 'mdml-engine/signal'
 * bessel(3, 0.2, { norm: 'mag' }).kind // => 'tf'
 * ```
 */
export function bessel<O extends FilterOutput = 'ba'>(
  n: number,
  wn: number | readonly number[],
  options: DesignOptions<O> & { readonly norm?: BesselNorm } = {},
): DesignOf<O> {
  return iirfilter<O>(n, wn, { ...options, ftype: 'bessel' })
}

function notchPeak(w0: number, q: number, fs: number, kind: 'notch' | 'peak'): TransferFunction {
  const w = (2 * w0) / fs
  if (w > 1 || w < 0) throw new MdmlError('value', 'iirnotch: w0 must lie in (0, fs / 2)')
  const bw = (w / q) * Math.PI
  const beta = Math.tan(bw / 2)
  const gain = 1 / (1 + beta)
  const c = Math.cos(w * Math.PI)
  const b = kind === 'notch' ? [gain, -2 * gain * c, gain] : [1 - gain, 0, -(1 - gain)]
  return tf(b, [1, -2 * gain * c, 2 * gain - 1], { dt: 1 / fs })
}

/**
 * A second-order notch (band-reject) filter (scipy.signal.iirnotch) removing frequency w0 with quality factor Q
 * (bandwidth w0 / Q at -3 dB).
 *
 * @param w0 - Frequency to remove (in the units of fs).
 * @param q - Quality factor.
 * @param fs - Sampling rate (default 2).
 * @returns The transfer function.
 * @concept iir-design
 * @example
 * ```ts
 * import { iirnotch } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(iirnotch(0.5, 1).num).map((v) => Math.round(v * 1e9) / 1e9) // => [0.5, 0, 0.5]
 * ```
 */
export function iirnotch(w0: number, q: number, fs = 2): TransferFunction {
  return notchPeak(w0, q, fs, 'notch')
}

/**
 * A second-order peak (resonator) filter (scipy.signal.iirpeak) passing frequency w0 with quality factor Q.
 *
 * @param w0 - Frequency to keep.
 * @param q - Quality factor.
 * @param fs - Sampling rate (default 2).
 * @returns The transfer function.
 * @concept iir-design
 * @example
 * ```ts
 * import { iirpeak } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(iirpeak(0.5, 1).num).map((v) => Math.round(v * 1e9) / 1e9) // => [0.5, 0, -0.5]
 * ```
 */
export function iirpeak(w0: number, q: number, fs = 2): TransferFunction {
  return notchPeak(w0, q, fs, 'peak')
}

/**
 * A comb filter (scipy.signal.iircomb): notches (or peaks) at every multiple of w0, which must divide fs.
 *
 * @param w0 - Fundamental frequency.
 * @param q - Quality factor.
 * @param options - 'notch' (default) or 'peak', sampling rate (default 2), and whether DC passes.
 * @returns The transfer function of order fs / w0.
 * @throws MdmlError with code `value` when w0 does not divide fs.
 * @concept iir-design
 * @example
 * ```ts
 * import { iircomb } from 'mdml-engine/signal'
 * iircomb(250, 30, { fs: 1000 }).num.shape // => [5]
 * ```
 */
export function iircomb(
  w0: number,
  q: number,
  options: { readonly ftype?: 'notch' | 'peak'; readonly fs?: number; readonly passZero?: boolean } = {},
): TransferFunction {
  const fs = options.fs ?? 2
  const ftype = options.ftype ?? 'notch'
  if (!(w0 > 0 && w0 < fs / 2)) throw new MdmlError('value', `iircomb: w0 must lie in (0, ${fs / 2})`)
  const n = Math.round(fs / w0)
  if (Math.abs(w0 - fs / n) / fs > 1e-14) throw new MdmlError('value', 'iircomb: w0 must divide fs')
  const w = (2 * Math.PI * w0) / fs
  const delta = w / q
  const [g0, g] = ftype === 'notch' ? [1, 0] : [0, 1]
  const beta = Math.tan((n * delta) / 4)
  const ax = (1 - beta) / (1 + beta)
  const bx = (g0 + g * beta) / (1 + beta)
  const cx = (g0 - g * beta) / (1 + beta)
  const passZero = options.passZero ?? false
  const negative = (ftype === 'peak' && passZero) || (ftype === 'notch' && !passZero)
  const sgn = negative ? -1 : 1
  const b = new Array<number>(n + 1).fill(0)
  const a = new Array<number>(n + 1).fill(0)
  b[0] = bx
  b[n] = sgn * cx
  a[0] = 1
  a[n] = sgn * ax
  return tf(b, a, { dt: 1 / fs })
}
