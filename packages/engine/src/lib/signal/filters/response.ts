/**
 * What a digital filter does to each frequency: the complex frequency response of second-order sections,
 * zeros-poles-gain or any LTI system, the group delay (computed exactly from the polynomials) and the phase delay, and
 * `checkSpecification`, which measures a design's worst passband loss and stopband attenuation against what was asked
 * for.
 */

import { MdmlError } from '../../base/index.ts'
import { complexOps as C } from '../../numerics/differentiate/index.ts'
import {
  type FrequencyResponse,
  type FreqzOptions,
  freqresp,
  freqs,
  freqz,
  type Lti,
  toSos,
  type ZerosPolesGainData,
} from '../../systems/index.ts'
import { angle, cMul, primalOf, reshape, sliceAxis, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { constant, float, isLti } from '../_shared.ts'
import { unwrap } from '../signal.ts'
import type { FilterSpec } from './order.ts'

// Frequency responses of digital filters beyond systems' freqz and freqs: cascades of second-order sections and
// zeros-poles-gain forms, group and phase delay, and checks of a design against its specification.

/**
 * The frequency response of a cascade of second-order sections (scipy.signal.sosfreqz / freqz_sos): the product of
 * each section's response. Differentiable in the sections.
 *
 * @param sos - Sections `[n, 6]`, or any LTI system (converted with toSos).
 * @param worN - Frequencies (in the units of fs), or a count (default 512).
 * @param options - Whole circle, sampling rate (default 2 pi), Nyquist inclusion.
 * @returns The frequencies and the complex response.
 * @concept frequency-response
 * @example
 * ```ts
 * import { sosfreqz } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(sosfreqz([[0.5, 0.5, 0, 1, 0, 0]], 2).h.re) // => [1, 0.5]
 * ```
 */
export function sosfreqz(
  sos: TensorLike | Lti,
  worN: number | TensorLike = 512,
  options: FreqzOptions = {},
): FrequencyResponse {
  const s = isLti(sos) ? toSos(sos).sections : float(sos)
  const count = s.shape[0] ?? 0
  if (s.shape.length !== 2 || s.shape[1] !== 6 || count === 0)
    throw new MdmlError('shape', 'sosfreqz: sections must be [n, 6] with n > 0')
  let out: FrequencyResponse | null = null
  for (let i = 0; i < count; i++) {
    const row = reshape(sliceAxis(s, 0, i, i + 1), [6])
    const r = freqz(sliceAxis(row, 0, 0, 3), sliceAxis(row, 0, 3, 6), worN, options)
    out = out === null ? r : { w: r.w, h: cMul(out.h, r.h) }
  }
  return out as FrequencyResponse
}

function gridOf(worN: number | TensorLike, options: FreqzOptions): { w: number[]; rad: number[] } {
  const fs = options.fs ?? 2 * Math.PI
  if (typeof worN === 'number') {
    const last = options.whole === true ? 2 * Math.PI : Math.PI
    const rad = Array.from({ length: worN }, (_, k) => (k * last) / worN)
    return { w: rad.map((v) => (v * fs) / (2 * Math.PI)), rad }
  }
  const w = toFlat(primalOf(float(worN)))
  return { w, rad: w.map((v) => (2 * Math.PI * v) / fs) }
}

/**
 * The frequency response of a digital filter given by zeros, poles and gain (scipy.signal.freqz_zpk):
 * H = k prod (e^{jw} - z) / prod (e^{jw} - p).
 *
 * @param s - Zeros, poles and gain.
 * @param worN - Frequencies (in the units of fs), or a count (default 512).
 * @param options - Whole circle and sampling rate (default 2 pi).
 * @returns The frequencies and the response.
 * @concept frequency-response
 * @example
 * ```ts
 * import { freqzZpk } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(freqzZpk({ zeros: { re: [-1], im: [0] }, poles: { re: [0], im: [0] }, gain: 0.5 }, 2).h.re) // => [1, 0.5]
 * ```
 */
export function freqzZpk(
  s: ZerosPolesGainData,
  worN: number | TensorLike = 512,
  options: FreqzOptions = {},
): FrequencyResponse {
  const { w, rad } = gridOf(worN, options)
  const re: number[] = []
  const im: number[] = []
  for (const t of rad) {
    const z = C.of(Math.cos(t), Math.sin(t))
    let h = C.of(s.gain)
    for (const [i, zr] of s.zeros.re.entries()) h = C.mul(h, C.sub(z, C.of(zr, s.zeros.im[i] ?? 0)))
    for (const [i, pr] of s.poles.re.entries()) h = C.div(h, C.sub(z, C.of(pr, s.poles.im[i] ?? 0)))
    re.push(h.re)
    im.push(h.im)
  }
  return { w: constant(w), h: { re: constant(re), im: constant(im) } }
}

/** A group or phase delay: frequencies and the delay there, in samples. */
export interface DelayResponse {
  readonly w: Tensor
  readonly delay: Tensor
}

/**
 * The group delay -d phase / d omega of a digital filter in samples (scipy.signal.group_delay), by the derivative
 * of the polynomial B(z) A*(1/z) (no numerical differentiation). Singular frequencies get 0.
 *
 * @param b - Numerator.
 * @param a - Denominator (default 1).
 * @param worN - Frequencies (in the units of fs), or a count (default 512).
 * @param options - Whole circle and sampling rate (default 2 pi).
 * @returns The frequencies and the delays.
 * @concept group-delay
 * @example
 * ```ts
 * import { groupDelay } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(groupDelay([1, 2, 1], [1], 3).delay).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 1, 1]
 * ```
 */
export function groupDelay(
  b: TensorLike,
  a: TensorLike = [1],
  worN: number | TensorLike = 512,
  options: FreqzOptions = {},
): DelayResponse {
  const bv = toFlat(primalOf(float(b)))
  const av = toFlat(primalOf(float(a)))
  const { w, rad } = gridOf(worN, options)
  // c = b convolved with reversed a; cr = c * n.
  const c = new Array<number>(bv.length + av.length - 1).fill(0)
  for (const [i, bi] of bv.entries())
    for (const [j, aj] of [...av].reverse().entries()) c[i + j] = (c[i + j] ?? 0) + bi * aj
  const delay = rad.map((t) => {
    const z = C.of(Math.cos(t), -Math.sin(t))
    // polyval of the reversed coefficients at z: sum_n c_n z^n.
    let num = C.of(0)
    let den = C.of(0)
    for (let n = c.length - 1; n >= 0; n--) {
      num = C.add(C.mul(num, z), n * (c[n] ?? 0))
      den = C.add(C.mul(den, z), c[n] ?? 0)
    }
    const gd = C.div(num, den).re - av.length + 1
    return Number.isFinite(gd) ? gd : 0
  })
  return { w: constant(w), delay: constant(delay) }
}

/**
 * The phase delay -phase(omega) / omega of a digital filter in samples, from the unwrapped phase of its response (at
 * omega = 0 the limit, the group delay there). A linear-phase filter has equal phase and group delays.
 *
 * @param b - Numerator.
 * @param a - Denominator (default 1).
 * @param worN - Frequencies (in the units of fs), or a count (default 512).
 * @param options - Whole circle and sampling rate (default 2 pi).
 * @returns The frequencies and the delays.
 * @concept group-delay
 * @example
 * ```ts
 * import { phaseDelay } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(phaseDelay([1, 2, 1], [1], 3).delay).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 1, 1]
 * ```
 */
export function phaseDelay(
  b: TensorLike,
  a: TensorLike = [1],
  worN: number | TensorLike = 512,
  options: FreqzOptions = {},
): DelayResponse {
  const { w, rad } = gridOf(worN, options)
  const fs = options.fs ?? 2 * Math.PI
  // Unwrap on a dense grid that contains the requested frequencies, so jumps between them are resolved.
  let top = 0
  for (const t of rad) top = Math.max(top, t)
  const dense = Array.from({ length: 2049 }, (_, k) => (k * top) / 2048)
  const all = [...new Set([...dense, ...rad])].sort((x, y) => x - y)
  const r = freqz(b, a, constant(all), { fs: 2 * Math.PI })
  const phase = toFlat(unwrap(angle(r.h)))
  const at = new Map(all.map((t, i) => [t, phase[i] ?? 0]))
  const at0 = toFlat(groupDelay(b, a, [0], { fs }).delay)[0] ?? 0
  return { w: constant(w), delay: constant(rad.map((t) => (t === 0 ? at0 : -(at.get(t) ?? 0) / t))) }
}

/**
 * The frequency response of any filter: a systems-domain LTI value (digital or analog), or coefficients b and a.
 *
 * @param filter - An LTI system, or `{ b, a }` coefficients of a digital filter.
 * @param worN - Frequencies, or a count (default 512).
 * @param options - Whole circle and sampling rate for digital filters given by coefficients.
 * @returns The frequencies and the response.
 * @concept frequency-response
 * @example
 * ```ts
 * import { butter, frequencyResponse } from 'mdml-engine/signal'
 * import { absComplex, toFlat } from 'mdml-engine/tensor'
 * Math.round((toFlat(absComplex(frequencyResponse(butter(4, 0.3), [0]).h))[0] ?? 0) * 1e12) / 1e12 // => 1
 * ```
 */
export function frequencyResponse(
  filter: Lti | { readonly b: TensorLike; readonly a?: TensorLike },
  worN: number | TensorLike = 512,
  options: FreqzOptions = {},
): FrequencyResponse {
  if ('b' in filter) return freqz(filter.b, filter.a ?? [1], worN, options)
  if (filter.dt === 0) {
    if (filter.kind === 'tf' && filter.num.shape.length === 1) return freqs(filter.num, filter.den, worN)
    return freqresp(filter, typeof worN === 'number' ? undefined : worN, typeof worN === 'number' ? { n: worN } : {})
  }
  // Digital: frequencies in the units of fs = 1 / dt.
  const fs = 1 / filter.dt
  return sosfreqz(toSos(filter).sections, worN, { ...options, fs })
}

/** How a design compares with its specification. */
export interface SpecificationCheck {
  /** The largest loss in the passband (dB, positive). */
  readonly passbandLoss: number
  /** The smallest attenuation in the stopband (dB). */
  readonly stopbandAttenuation: number
  /** Whether passbandLoss <= gpass and stopbandAttenuation >= gstop (to 1e-6 dB). */
  readonly meets: boolean
}

/**
 * Check a design against a specification: the worst passband loss and stopband attenuation on a dense grid, relative
 * to the peak passband gain, compared with gpass and gstop. The band type follows from the edges, as iirdesign.
 *
 * @param filter - The design (an LTI value or `{ b, a }`).
 * @param spec - The specification (digital, frequencies in the units of fs or fractions of Nyquist).
 * @param points - Grid size over [0, Nyquist) (default 4096).
 * @returns The worst losses and whether the design meets the specification.
 * @concept filter-specification
 * @example
 * ```ts
 * import { checkSpecification, iirdesign } from 'mdml-engine/signal'
 * const spec = { wp: 0.2, ws: 0.3, gpass: 1, gstop: 40 }
 * checkSpecification(iirdesign(spec, { ftype: 'butter' }), spec).meets // => true
 * ```
 */
export function checkSpecification(
  filter: Lti | { readonly b: TensorLike; readonly a?: TensorLike },
  spec: FilterSpec,
  points = 4096,
): SpecificationCheck {
  const nyq = spec.fs === undefined ? 1 : spec.fs / 2
  const toUnit = (v: number): number => v / nyq
  const wp = (typeof spec.wp === 'number' ? [spec.wp] : [...spec.wp]).map(toUnit)
  const ws = (typeof spec.ws === 'number' ? [spec.ws] : [...spec.ws]).map(toUnit)
  const f = Array.from({ length: points }, (_, k) => k / points)
  const r =
    'b' in filter
      ? freqz(filter.b, filter.a ?? [1], constant(f), { fs: 2 })
      : sosfreqz(toSos(filter).sections, constant(f), { fs: 2 })
  const re = toFlat(r.h.re)
  const im = toFlat(r.h.im)
  const mag = re.map((v, i) => Math.hypot(v, im[i] ?? 0))
  const lowpass = (wp[0] ?? 0) < (ws[0] ?? 0)
  const inPass = (x: number): boolean => {
    if (wp.length === 1) return lowpass ? x <= (wp[0] ?? 0) : x >= (wp[0] ?? 0)
    return (wp[0] ?? 0) < (ws[0] ?? 0) ? x <= (wp[0] ?? 0) || x >= (wp[1] ?? 0) : x >= (wp[0] ?? 0) && x <= (wp[1] ?? 0)
  }
  const inStop = (x: number): boolean => {
    if (ws.length === 1) return lowpass ? x >= (ws[0] ?? 0) : x <= (ws[0] ?? 0)
    return (wp[0] ?? 0) < (ws[0] ?? 0) ? x >= (ws[0] ?? 0) && x <= (ws[1] ?? 0) : x <= (ws[0] ?? 0) || x >= (ws[1] ?? 0)
  }
  let peak = 0
  for (const [i, x] of f.entries()) if (inPass(x)) peak = Math.max(peak, mag[i] ?? 0)
  if (peak === 0) throw new MdmlError('value', 'checkSpecification: no passband gain on the grid')
  let loss = 0
  let atten = Number.POSITIVE_INFINITY
  for (const [i, x] of f.entries()) {
    const db = -20 * Math.log10((mag[i] ?? 0) / peak)
    if (inPass(x)) loss = Math.max(loss, db)
    if (inStop(x)) atten = Math.min(atten, db)
  }
  return {
    passbandLoss: loss,
    stopbandAttenuation: atten,
    meets: loss <= spec.gpass + 1e-6 && atten >= spec.gstop - 1e-6,
  }
}
