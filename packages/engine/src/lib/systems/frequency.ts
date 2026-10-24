import { WhiteboxError } from '../base/index.ts'
import { complexOps as C, type ComplexNumber } from '../numerics/differentiate/index.ts'
import { eigvals, solve } from '../numerics/linalg/index.ts'
import type { ComplexArray } from '../numerics/polynomial/index.ts'
import {
  absComplex,
  add,
  angle,
  type ComplexValue,
  concat,
  div,
  eye,
  log10,
  logspace,
  matmul,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { constant, type Points, tfAt } from './_points.ts'
import { float, numbers, rootsOf, toList } from './_shared.ts'
import { isDiscrete, type Lti, type StateSpace, toTf, toZpk } from './representations.ts'
import { sos2tf } from './sections.ts'

// Frequency responses follow scipy.signal (freqresp, bode, freqs, freqz, findfreqs) and python-control (evalfr,
// dcgain, nyquist and nichols data). Frequencies are in radians per time unit; a discrete system with sample time dt
// is evaluated at z = exp(j w dt). Responses are complex pairs, `[nw]` for single-input single-output systems and
// `[nw, outputs, inputs]` otherwise, differentiable in the coefficients or matrices.

/** A frequency response: the frequencies and the complex response there. */
export interface FrequencyResponse {
  readonly w: Tensor
  readonly h: ComplexValue
}

function zpkAt(zeros_: ComplexArray, poles: ComplexArray, gain: number, z: Points): ComplexValue {
  const zr = toFlat(z.re)
  const zi = toFlat(z.im)
  const out = zr.map((re, i) => {
    const s = C.of(re, zi[i] ?? 0)
    let acc: ComplexNumber = C.of(gain)
    for (const q of toList(zeros_)) acc = C.mul(acc, C.sub(s, q))
    for (const q of toList(poles)) acc = C.div(acc, C.sub(s, q))
    return acc
  })
  return { re: constant(out.map((v) => v.re)), im: constant(out.map((v) => v.im)) }
}

/** C (zI - A)^{-1} B + D at every point, by one batched real solve of the 2n x 2n form of zI - A. */
function ssAt(s: StateSpace, z: Points): ComplexValue {
  const n = s.a.shape[0] ?? 0
  const m = s.b.shape[1] ?? 0
  const p = s.c.shape[0] ?? 0
  const nw = z.re.shape[0] ?? 0
  const zr = reshape(z.re, [nw, 1, 1])
  const zi = reshape(z.im, [nw, 1, 1])
  const identity = eye(n)
  const diag = add(mul(zr, identity), mul(s.a, -1))
  const off = mul(zi, identity)
  const system = concat([concat([diag, mul(off, -1)], 2), concat([off, diag], 2)], 1)
  const rhs = concat([s.b, zeros([n, m])], 0)
  const x = solve(system, add(rhs, zeros([nw, 2 * n, m])))
  const re = add(matmul(s.c, sliceAxis(x, 1, 0, n)), s.d)
  const im = matmul(s.c, sliceAxis(x, 1, n, 2 * n))
  if (p === 1 && m === 1) return { re: reshape(re, [nw]), im: reshape(im, [nw]) }
  return { re, im }
}

/** The response of any system at points of its own complex plane (s, or z for discrete systems). */
function responseAt(sys: Lti, z: Points): ComplexValue {
  switch (sys.kind) {
    case 'tf':
      return tfAt(sys.num, sys.den, z)
    case 'zpk':
      return zpkAt(sys.zeros, sys.poles, sys.gain, z)
    case 'sos': {
      const { num, den } = sos2tf(sys.sections)
      return tfAt(num, den, z)
    }
    case 'ss':
      return ssAt(sys, z)
  }
}

function pointsFor(sys: Lti, w: readonly number[]): Points {
  if (!isDiscrete(sys)) return { re: constant(w.map(() => 0)), im: constant(w) }
  return { re: constant(w.map((v) => Math.cos(v * sys.dt))), im: constant(w.map((v) => Math.sin(v * sys.dt))) }
}

/** numpy.round: halves to even. */
function roundHalfEven(x: number): number {
  const r = Math.round(x)
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r
}

/**
 * Frequencies covering the interesting part of a response (scipy.signal.findfreqs): a log-spaced grid from a decade
 * below the slowest pole or zero to a decade above the fastest.
 *
 * @param num - Numerator (or the zeros, for `kind` 'zp').
 * @param den - Denominator (or the poles).
 * @param n - Number of frequencies.
 * @param kind - 'ba' for coefficients (default), 'zp' for zeros and poles.
 * @returns The frequencies.
 * @concept frequency-response
 * @example
 * ```ts
 * import { findfreqs } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(findfreqs([1], [1, 1], 3)) // => [0.01, 0.31622776601683794, 10]
 * ```
 */
export function findfreqs(
  num: TensorLike | ComplexArray,
  den: TensorLike | ComplexArray,
  n: number,
  kind: 'ba' | 'zp' = 'ba',
): Tensor {
  const asList = (x: TensorLike | ComplexArray, coefficients: boolean): ComplexNumber[] => {
    if (typeof x === 'object' && x !== null && 're' in x && !Array.isArray(x)) return toList(x as ComplexArray)
    const v = numbers(float(x as TensorLike))
    return coefficients ? toList(rootsOf(v)) : v.map((r) => C.of(r))
  }
  let ep = asList(den, kind === 'ba')
  const tz = asList(num, kind === 'ba')
  if (ep.length === 0) ep = [C.of(-1000)]
  const ez = [...ep.filter((z) => z.im >= 0), ...tz.filter((z) => Math.hypot(z.re, z.im) < 1e5 && z.im >= 0)]
  const integ = (z: ComplexNumber): number => (Math.hypot(z.re, z.im) < 1e-10 ? 1 : 0)
  let hi = Number.NEGATIVE_INFINITY
  let lo = Number.POSITIVE_INFINITY
  for (const z of ez) {
    hi = Math.max(hi, 3 * Math.abs(z.re + integ(z)) + 1.5 * z.im)
    lo = Math.min(lo, Math.abs(z.re + integ(z)) + 2 * z.im)
  }
  const hfreq = roundHalfEven(Math.log10(hi) + 0.5)
  const lfreq = roundHalfEven(Math.log10(0.1 * lo) - 0.5)
  return logspace(lfreq, hfreq, n) as Tensor
}

/** The default frequencies of a system: findfreqs of its zeros and poles, or [0, pi / dt) in discrete time. */
function defaultFrequencies(sys: Lti, n: number): Tensor {
  if (isDiscrete(sys)) return constant(Array.from({ length: n }, (_, k) => (k * Math.PI) / (n * sys.dt)))
  if (sys.kind === 'ss' && ((sys.b.shape[1] ?? 0) !== 1 || (sys.c.shape[0] ?? 0) !== 1)) {
    const { re, im } = eigvals(primalOf(sys.a))
    return findfreqs({ re: [], im: [] }, { re: toFlat(re), im: toFlat(im) }, n, 'zp')
  }
  const z = toZpk(sys)
  return findfreqs(z.zeros, z.poles, n, 'zp')
}

function frequencies(sys: Lti, w: TensorLike | undefined, n: number): Tensor {
  return w === undefined ? defaultFrequencies(sys, n) : (float(w) as Tensor)
}

/**
 * The frequency response of a system (scipy.signal.freqresp; dfreqresp in discrete time, with frequencies in radians
 * per time unit). State-space systems use one batched solve of (jwI - A) X = B per frequency, so multi-input
 * multi-output systems give `[nw, outputs, inputs]`. Differentiable in the system's coefficients or matrices.
 *
 * @param sys - The system.
 * @param w - Frequencies (default: n chosen by findfreqs, or n in [0, pi / dt) in discrete time).
 * @param options - The number of default frequencies (default 10000, as scipy).
 * @returns The frequencies and the response.
 * @concept frequency-response
 * @example
 * ```ts
 * import { freqresp, tf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(freqresp(tf([1], [1, 1]), [1]).h.im) // => [-0.5]
 * ```
 */
export function freqresp(sys: Lti, w?: TensorLike, options: { readonly n?: number } = {}): FrequencyResponse {
  const ws = frequencies(sys, w, options.n ?? 10000)
  return { w: ws, h: responseAt(sys, pointsFor(sys, toFlat(ws))) }
}

/**
 * The response at one point of the complex plane (python-control evalfr): s for continuous systems, z for discrete.
 *
 * @param sys - The system.
 * @param s - The point.
 * @returns The response: scalars for single-input single-output systems, `[outputs, inputs]` otherwise.
 * @example
 * ```ts
 * import { evalfr, tf } from 'whitebox-engine/systems'
 * import { item } from 'whitebox-engine/tensor'
 * item(evalfr(tf([1], [1, 1]), { re: 1, im: 0 }).re) // => 0.5
 * ```
 */
export function evalfr(sys: Lti, s: ComplexNumber): ComplexValue {
  const h = responseAt(sys, { re: constant([s.re]), im: constant([s.im]) })
  const shape = h.re.shape.length === 1 ? [] : (h.re.shape.slice(1) as number[])
  return { re: reshape(h.re, shape), im: reshape(h.im, shape) }
}

/**
 * The steady-state gain (python-control dcgain): the response at s = 0, or z = 1 in discrete time; infinite for an
 * integrator.
 *
 * @param sys - The system.
 * @returns The gain: a scalar, or `[outputs, inputs]`.
 * @example
 * ```ts
 * import { dcgain, tf } from 'whitebox-engine/systems'
 * import { item } from 'whitebox-engine/tensor'
 * item(dcgain(tf([3], [1, 2]))) // => 1.5
 * ```
 */
export function dcgain(sys: Lti): Value {
  const at = isDiscrete(sys) ? C.of(1) : C.of(0)
  if (sys.kind === 'ss') {
    try {
      return evalfr(sys, at).re
    } catch {
      // A pole at the evaluation point: the gain is infinite (or undefined) channel by channel.
      const p = sys.c.shape[0] ?? 0
      const m = sys.b.shape[1] ?? 0
      return tensor(Array.from({ length: p }, () => Array.from({ length: m }, () => Number.POSITIVE_INFINITY)))
    }
  }
  // Real arithmetic for transfer functions, so that an integrator gives an infinite gain, not NaN.
  const g = sys.kind === 'tf' ? sys : toTf(sys)
  const valueAt = (c: Value): Value => {
    const k = c.shape[c.shape.length - 1] ?? 0
    return isDiscrete(sys) ? sum(c, -1) : reshape(sliceAxis(c, c.shape.length - 1, k - 1, k), c.shape.slice(0, -1))
  }
  const num = valueAt(g.num)
  const den = valueAt(g.den)
  return g.num.shape.length === 2 ? reshape(div(num, den), [g.num.shape[0] ?? 0, 1]) : div(num, den)
}

/** numpy.unwrap offsets (multiples of 2 pi) for a sequence of angles. */
function unwrapOffsets(phase: readonly number[]): number[] {
  const out = [0]
  let total = 0
  for (let i = 1; i < phase.length; i++) {
    const dd = (phase[i] ?? 0) - (phase[i - 1] ?? 0)
    let mod = ((((dd + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI
    if (mod === -Math.PI && dd > 0) mod = Math.PI
    if (Math.abs(dd) >= Math.PI) total += mod - dd
    out.push(total)
  }
  return out
}

/** Magnitude in dB and unwrapped phase in degrees. */
export interface BodeResponse {
  readonly w: Tensor
  /** 20 log10 |H|. */
  readonly mag: Value
  /** The unwrapped phase in degrees. */
  readonly phase: Value
}

function siso(h: ComplexValue, name: string): void {
  if (h.re.shape.length !== 1) throw new WhiteboxError('shape', `${name}: one input and one output are supported`)
}

function bodeOf(w: Tensor, h: ComplexValue): BodeResponse {
  const mag = mul(log10(absComplex(h)), 20)
  const raw = angle(h)
  const offsets = constant(unwrapOffsets(toFlat(primalOf(raw))))
  return { w, mag, phase: mul(add(raw, offsets), 180 / Math.PI) }
}

/**
 * Bode magnitude and phase (scipy.signal.bode and dbode): 20 log10 |H(jw)| and the unwrapped phase in degrees,
 * differentiable in the system.
 *
 * @param sys - A single-input single-output system.
 * @param w - Frequencies (default: 100 chosen by findfreqs, or in [0, pi / dt)).
 * @param options - The number of default frequencies (default 100).
 * @returns Frequencies, magnitude (dB) and phase (degrees).
 * @concept bode-plot
 * @example
 * ```ts
 * import { bode, tf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(bode(tf([1], [1, 1]), [1]).phase) // => [-45]
 * ```
 */
export function bode(sys: Lti, w?: TensorLike, options: { readonly n?: number } = {}): BodeResponse {
  const r = freqresp(sys, w, { n: options.n ?? 100 })
  siso(r.h, 'bode')
  return bodeOf(r.w, r.h)
}

/**
 * Nichols chart data (python-control nichols_plot): the unwrapped phase in degrees against the gain in dB.
 *
 * @param sys - A single-input single-output system.
 * @param w - Frequencies (default as {@link bode}).
 * @param options - The number of default frequencies.
 * @returns Frequencies, magnitude (dB) and phase (degrees), as {@link bode}.
 * @concept nichols-chart
 * @example
 * ```ts
 * import { nichols, tf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(nichols(tf([1], [1, 0, 0]), [1]).phase) // => [-180]
 * ```
 */
export function nichols(sys: Lti, w?: TensorLike, options: { readonly n?: number } = {}): BodeResponse {
  return bode(sys, w, options)
}

/** Nyquist curve data. */
export interface NyquistResponse {
  /** The contour's frequencies (from 0, indented around imaginary-axis poles). */
  readonly w: Tensor
  /** The response along the contour. */
  readonly h: ComplexValue
  /** Clockwise encirclements of -1 by the full curve (python-control nyquist_response count). */
  readonly count: number
}

/** Options for {@link nyquist}. */
export interface NyquistOptions {
  /** Number of default frequencies (default 1000). */
  readonly n?: number
  /** Radius of the indentation around poles on the imaginary axis (default 1e-4, as python-control). */
  readonly indentRadius?: number
}

/**
 * Nyquist curve data and the encirclement count (python-control nyquist_response): the response along the positive
 * imaginary axis (the unit circle up to the Nyquist frequency in discrete time), the contour pushed right around
 * poles on the axis, and the number of clockwise encirclements of -1, -Delta arg(1 + L) / pi over the half contour.
 * With P open-loop unstable poles, the closed loop u = -L y has Z = N + P unstable poles.
 *
 * @param sys - A single-input single-output loop transfer function.
 * @param w - Contour frequencies (default: 0 and a findfreqs grid, extended two decades each way).
 * @param options - Default frequency count and indentation radius.
 * @returns The contour frequencies, response and encirclement count.
 * @concept nyquist-criterion
 * @example
 * ```ts
 * import { nyquist, tf } from 'whitebox-engine/systems'
 * nyquist(tf([2], [1, 1])).count // => 0
 * ```
 */
export function nyquist(sys: Lti, w?: TensorLike, options: NyquistOptions = {}): NyquistResponse {
  const radius = options.indentRadius ?? 1e-4
  let ws: number[]
  if (w !== undefined) ws = numbers(float(w))
  else {
    const base = toFlat(defaultFrequencies(sys, options.n ?? 1000))
    if (isDiscrete(sys)) ws = [...base.filter((v) => v < Math.PI / sys.dt), Math.PI / sys.dt]
    else {
      const lo = Math.log10(base[0] ?? 1e-2) - 2
      const hi = Math.log10(base[base.length - 1] ?? 1e2) + 2
      ws = [0, ...toFlat(logspace(lo, hi, options.n ?? 1000))]
    }
  }
  // s-plane contour points, pushed right around poles within the radius (for discrete systems, poles mapped by log).
  const zp = toZpk(sys)
  const poles = toList(zp.poles)
    .filter((p) => !isDiscrete(sys) || Math.hypot(p.re, p.im) > 1e-12)
    .map((p) => (isDiscrete(sys) ? C.div(C.log(p), sys.dt) : p))
  let contour: ComplexNumber[] = ws.map((v) => C.of(0, v))
  if (w === undefined) {
    for (const p of poles) {
      if (p.im < 0 || Math.abs(p.re) > radius) continue
      const below = contour.filter((s) => s.im - Math.abs(p.im) < -radius)
      const above = contour.filter((s) => s.im - Math.abs(p.im) > radius)
      const start = below.length > 0 ? p.im - radius : 0
      const extra = Array.from({ length: 50 }, (_, k) => C.of(0, start + ((p.im + radius - start) * k) / 49))
      contour = [...below, ...extra, ...above]
    }
  }
  contour = contour.map((s) => {
    let best: ComplexNumber | undefined
    for (const p of poles)
      if (best === undefined || Math.hypot(s.re - p.re, s.im - p.im) < Math.hypot(s.re - best.re, s.im - best.im))
        best = p
    if (best === undefined || Math.hypot(s.re - best.re, s.im - best.im) >= radius) return s
    const offset = Math.sqrt(radius * radius - (s.im - best.im) ** 2) - (s.re - best.re)
    // Poles on the axis (to rounding: roots and logarithms leave them a hair off it) are passed on the right.
    return best.re <= 1e-10 * Math.max(1, Math.hypot(best.re, best.im)) ? C.add(s, offset) : C.sub(s, offset)
  })
  const points: Points = isDiscrete(sys)
    ? (() => {
        const zs = contour.map((s) => C.exp(C.mul(s, sys.dt)))
        return { re: constant(zs.map((z) => z.re)), im: constant(zs.map((z) => z.im)) }
      })()
    : { re: constant(contour.map((s) => s.re)), im: constant(contour.map((s) => s.im)) }
  const h = responseAt(sys, points)
  siso(h, 'nyquist')
  const hr = toFlat(primalOf(h.re))
  const hi = toFlat(primalOf(h.im))
  const phase = hr.map((r, i) => Math.atan2(hi[i] ?? 0, r + 1))
  const offsets = unwrapOffsets(phase)
  const first = (phase[0] ?? 0) + (offsets[0] ?? 0)
  const last = (phase[phase.length - 1] ?? 0) + (offsets[offsets.length - 1] ?? 0)
  return { w: constant(contour.map((s) => s.im)), h, count: roundHalfEven(-(last - first) / Math.PI) }
}
