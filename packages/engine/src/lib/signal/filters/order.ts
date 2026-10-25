/**
 * Minimum-order IIR design from a specification (passband and stopband edges, allowed passband loss, required stopband
 * attenuation): the order functions of the Butterworth, Chebyshev and elliptic families (`buttord`, `cheb1ord`,
 * `cheb2ord`, `ellipord`) and `iirdesign`, which picks the band type and order and designs the filter, as scipy.signal.
 */

import { MdmlError } from '../../base/index.ts'
import { ellipk } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'
import { ellipKm1, fminbound } from './_numeric.ts'
import { type DesignOf, type FilterOutput, type IirFamily, iirfilter } from './iir.ts'

// Minimum-order selection for IIR designs, as scipy.signal's buttord, cheb1ord, cheb2ord and ellipord, and iirdesign,
// which designs the minimum-order filter meeting a specification.

/** A filter specification: passband and stopband edges, maximum passband loss and minimum stopband attenuation. */
export interface FilterSpec {
  /** Passband edge(s), in the units of fs (fractions of Nyquist by default, rad/s when analog). */
  readonly wp: number | readonly [number, number]
  /** Stopband edge(s). */
  readonly ws: number | readonly [number, number]
  /** Maximum loss in the passband (dB). */
  readonly gpass: number
  /** Minimum attenuation in the stopband (dB). */
  readonly gstop: number
  readonly analog?: boolean
  readonly fs?: number
}

/** The order and the natural (critical) frequency or frequencies of a minimum-order design. */
export interface FilterOrder {
  readonly n: number
  readonly wn: number | readonly [number, number]
}

type Kind = 'butter' | 'cheby' | 'ellip'

interface Prepared {
  readonly passb: number[]
  readonly stopb: number[]
  readonly type: number
}

function prepare(spec: FilterSpec): Prepared {
  const { gpass, gstop } = spec
  if (gpass <= 0) throw new MdmlError('value', 'gpass should be larger than 0')
  if (gstop <= 0) throw new MdmlError('value', 'gstop should be larger than 0')
  if (gpass > gstop) throw new MdmlError('value', 'gpass should be smaller than gstop')
  const analog = spec.analog ?? false
  if (analog && spec.fs !== undefined) throw new MdmlError('value', 'fs cannot be given for an analog filter')
  let wp = typeof spec.wp === 'number' ? [spec.wp] : [...spec.wp]
  let ws = typeof spec.ws === 'number' ? [spec.ws] : [...spec.ws]
  if (spec.fs !== undefined) {
    const fs = spec.fs
    wp = wp.map((v) => (2 * v) / fs)
    ws = ws.map((v) => (2 * v) / fs)
  }
  let type = 2 * (wp.length - 1) + 1
  if ((wp[0] ?? 0) >= (ws[0] ?? 0)) type += 1
  const warp = (v: number): number => (analog ? v : Math.tan((Math.PI * v) / 2))
  return { passb: wp.map(warp), stopb: ws.map(warp), type }
}

function orderOf(kind: Kind, nat: number, gpass: number, gstop: number): number {
  const GS = 10 ** (0.1 * Math.abs(gstop))
  const GP = 10 ** (0.1 * Math.abs(gpass))
  if (kind === 'butter') return Math.log10((GS - 1) / (GP - 1)) / (2 * Math.log10(nat))
  if (kind === 'cheby') return Math.acosh(Math.sqrt((GS - 1) / (GP - 1))) / Math.acosh(nat)
  const arg1 = Math.sqrt((10 ** (0.1 * gpass) - 1) / (10 ** (0.1 * gstop) - 1))
  const arg0 = 1 / nat
  return (
    (item(ellipk(arg0 * arg0)) * item(ellipk(1 - arg1 * arg1))) /
    (item(ellipk(1 - arg0 * arg0)) * item(ellipk(arg1 * arg1)))
  )
}

function naturalFrequency(p: Prepared, kind: Kind, gpass: number, gstop: number): { nat: number; passb: number[] } {
  const [p0 = 0, p1 = 0] = p.passb
  let passb = p.passb
  let nats: number[]
  if (p.type === 1) nats = p.stopb.map((s, i) => s / (p.passb[i] ?? 1))
  else if (p.type === 2) nats = p.stopb.map((s, i) => (p.passb[i] ?? 1) / s)
  else if (p.type === 3) {
    const [s0 = 0, s1 = 0] = p.stopb
    const obj =
      (ind: 0 | 1) =>
      (wp: number): number => {
        const pc = [p0, p1]
        pc[ind] = wp
        const a = pc[0] ?? 0
        const b = pc[1] ?? 0
        let nat = Number.POSITIVE_INFINITY
        for (const s of p.stopb) nat = Math.min(nat, Math.abs((s * (a - b)) / (s * s - a * b)))
        return orderOf(kind, nat, gpass, gstop)
      }
    const wp0 = fminbound(obj(0), p0, s0 - 1e-12)
    const wp1 = fminbound(obj(1), s1 + 1e-12, p1)
    passb = [wp0, wp1]
    nats = p.stopb.map((s) => (s * (wp0 - wp1)) / (s * s - wp0 * wp1))
  } else nats = p.stopb.map((s) => (s * s - p0 * p1) / (s * (p0 - p1)))
  let nat = Number.POSITIVE_INFINITY
  for (const v of nats) nat = Math.min(nat, Math.abs(v))
  return { nat, passb }
}

function post(wn: readonly number[], spec: FilterSpec): number | readonly [number, number] {
  const analog = spec.analog ?? false
  const out = wn.map((v) => {
    const w = analog ? v : (Math.atan(v) * 2) / Math.PI
    return spec.fs === undefined ? w : (w * spec.fs) / 2
  })
  return out.length === 1 ? (out[0] ?? 0) : [out[0] ?? 0, out[1] ?? 0]
}

/**
 * The minimum Butterworth order meeting a specification and its natural frequency (scipy.signal.buttord).
 *
 * @param spec - Band edges, losses, analog flag and sampling rate.
 * @returns The order and the -3 dB frequency (or band edges).
 * @throws MdmlError with code `value` for inconsistent losses.
 * @concept filter-specification
 * @example
 * ```ts
 * import { buttord } from 'mdml-engine/signal'
 * buttord({ wp: 0.2, ws: 0.3, gpass: 3, gstop: 40 }).n // => 11
 * ```
 */
export function buttord(spec: FilterSpec): FilterOrder {
  const p = prepare(spec)
  const { nat, passb } = naturalFrequency(p, 'butter', spec.gpass, spec.gstop)
  const GS = 10 ** (0.1 * Math.abs(spec.gstop))
  const GP = 10 ** (0.1 * Math.abs(spec.gpass))
  const n = Math.ceil(Math.log10((GS - 1) / (GP - 1)) / (2 * Math.log10(nat)))
  const w0 = n === 0 ? 1 : (GP - 1) ** (-1 / (2 * n))
  const [a = 0, b = 0] = passb
  let wn: number[]
  if (p.type === 1) wn = passb.map((v) => w0 * v)
  else if (p.type === 2) wn = passb.map((v) => v / w0)
  else if (p.type === 3) {
    const discr = Math.sqrt((b - a) ** 2 + 4 * w0 * w0 * a * b)
    wn = [Math.abs((b - a + discr) / (2 * w0)), Math.abs((b - a - discr) / (2 * w0))].sort((x, y) => x - y)
  } else {
    wn = [-w0, w0]
      .map((w) => Math.abs((-w * (b - a)) / 2 + Math.sqrt(((w * w) / 4) * (b - a) ** 2 + a * b)))
      .sort((x, y) => x - y)
  }
  return { n, wn: post(wn, spec) }
}

/**
 * The minimum Chebyshev type I order meeting a specification (scipy.signal.cheb1ord).
 *
 * @param spec - Band edges, losses, analog flag and sampling rate.
 * @returns The order and the passband edge(s) to design with.
 * @concept filter-specification
 * @example
 * ```ts
 * import { cheb1ord } from 'mdml-engine/signal'
 * cheb1ord({ wp: 0.2, ws: 0.3, gpass: 3, gstop: 40 }).n // => 6
 * ```
 */
export function cheb1ord(spec: FilterSpec): FilterOrder {
  const p = prepare(spec)
  const { nat, passb } = naturalFrequency(p, 'cheby', spec.gpass, spec.gstop)
  return { n: Math.ceil(orderOf('cheby', nat, spec.gpass, spec.gstop)), wn: post(passb, spec) }
}

/**
 * The minimum Chebyshev type II order meeting a specification (scipy.signal.cheb2ord), with the stopband edge(s)
 * to design with.
 *
 * @param spec - Band edges, losses, analog flag and sampling rate.
 * @returns The order and the critical frequency or frequencies.
 * @concept filter-specification
 * @example
 * ```ts
 * import { cheb2ord } from 'mdml-engine/signal'
 * cheb2ord({ wp: 0.2, ws: 0.3, gpass: 3, gstop: 40 }).n // => 6
 * ```
 */
export function cheb2ord(spec: FilterSpec): FilterOrder {
  const p = prepare(spec)
  const { nat, passb } = naturalFrequency(p, 'cheby', spec.gpass, spec.gstop)
  const GS = 10 ** (0.1 * Math.abs(spec.gstop))
  const GP = 10 ** (0.1 * Math.abs(spec.gpass))
  const v = Math.acosh(Math.sqrt((GS - 1) / (GP - 1)))
  const n = Math.ceil(v / Math.acosh(nat))
  const f = 1 / Math.cosh(v / n)
  const [a = 0, b = 0] = passb
  let wn: number[]
  if (p.type === 1) wn = passb.map((x) => x / f)
  else if (p.type === 2) wn = passb.map((x) => x * f)
  else if (p.type === 3) {
    const nat0 = (f / 2) * (a - b) + Math.sqrt((f * f * (b - a) ** 2) / 4 + b * a)
    wn = [nat0, (b * a) / nat0]
  } else {
    const nat0 = (1 / (2 * f)) * (a - b) + Math.sqrt((b - a) ** 2 / (4 * f * f) + b * a)
    wn = [nat0, (a * b) / nat0]
  }
  return { n, wn: post(wn, spec) }
}

/**
 * The minimum elliptic order meeting a specification (scipy.signal.ellipord).
 *
 * @param spec - Band edges, losses, analog flag and sampling rate.
 * @returns The order and the passband edge(s) to design with.
 * @concept filter-specification
 * @example
 * ```ts
 * import { ellipord } from 'mdml-engine/signal'
 * ellipord({ wp: 0.2, ws: 0.3, gpass: 3, gstop: 40 }).n // => 4
 * ```
 */
export function ellipord(spec: FilterSpec): FilterOrder {
  const p = prepare(spec)
  const { nat, passb } = naturalFrequency(p, 'ellip', spec.gpass, spec.gstop)
  const arg1Sq = Math.expm1(Math.LN10 * 0.1 * spec.gpass) / Math.expm1(Math.LN10 * 0.1 * spec.gstop)
  const arg0 = 1 / nat
  const d0 = [item(ellipk(arg0 * arg0)), ellipKm1(arg0 * arg0)]
  const d1 = [item(ellipk(arg1Sq)), ellipKm1(arg1Sq)]
  const n = Math.ceil(((d0[0] ?? 0) * (d1[1] ?? 0)) / ((d0[1] ?? 1) * (d1[0] ?? 1)))
  return { n, wn: post(passb, spec) }
}

/**
 * Design the minimum-order IIR filter meeting a specification (scipy.signal.iirdesign): the band type follows from
 * the edges, the order from the family's order function.
 *
 * @param spec - Band edges, losses, analog flag and sampling rate.
 * @param options - Family (default 'ellip'; 'bessel' has no order selection) and output form.
 * @returns The design.
 * @throws MdmlError with code `value` for an invalid specification.
 * @concept filter-specification
 * @example
 * ```ts
 * import { iirdesign } from 'mdml-engine/signal'
 * iirdesign({ wp: 0.2, ws: 0.3, gpass: 1, gstop: 40 }, { ftype: 'butter', output: 'sos' }).kind // => 'sos'
 * ```
 */
export function iirdesign<O extends FilterOutput = 'ba'>(
  spec: FilterSpec,
  options: { readonly ftype?: Exclude<IirFamily, 'bessel'>; readonly output?: O } = {},
): DesignOf<O> {
  const wp = typeof spec.wp === 'number' ? [spec.wp] : [...spec.wp]
  const ws = typeof spec.ws === 'number' ? [spec.ws] : [...spec.ws]
  if (wp.length !== ws.length) throw new MdmlError('value', 'iirdesign: wp and ws must have the same length')
  if (wp.some((v) => v <= 0) || ws.some((v) => v <= 0))
    throw new MdmlError('value', 'iirdesign: edges must be positive')
  const nyq = spec.fs === undefined ? 1 : spec.fs / 2
  if (!(spec.analog ?? false) && [...wp, ...ws].some((v) => v >= nyq))
    throw new MdmlError('value', 'iirdesign: edges must be below the Nyquist frequency')
  if (wp.length === 2) {
    const [a = 0, b = 0] = wp
    const [c = 0, d = 0] = ws
    if (!((c < a && b < d) || (a < c && d < b)))
      throw new MdmlError('value', 'iirdesign: the passband must lie strictly inside the stopband or vice versa')
  }
  let type = 2 * (wp.length - 1) + 1
  if ((wp[0] ?? 0) >= (ws[0] ?? 0)) type += 1
  const btype = (['lowpass', 'highpass', 'bandstop', 'bandpass'] as const)[type - 1] ?? 'lowpass'
  const ftype = options.ftype ?? 'ellip'
  const ord =
    ftype === 'butter'
      ? buttord(spec)
      : ftype === 'cheby1'
        ? cheb1ord(spec)
        : ftype === 'cheby2'
          ? cheb2ord(spec)
          : ellipord(spec)
  return iirfilter<O>(ord.n, ord.wn, {
    rp: spec.gpass,
    rs: spec.gstop,
    btype,
    ftype,
    ...(spec.analog === undefined ? {} : { analog: spec.analog }),
    ...(spec.fs === undefined ? {} : { fs: spec.fs }),
    ...(options.output === undefined ? {} : { output: options.output }),
  })
}
