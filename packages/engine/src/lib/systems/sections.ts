/**
 * Second-order sections, the cascade of biquad filters that is the numerically safe way to realise a high-order filter:
 * `zpk2sos` with scipy's pairing of poles and zeros, and the conversions from sections back to a single transfer
 * function or to zeros, poles and gain, and from a transfer function to sections.
 */

import { WhiteboxError } from '../base/index.ts'
import { convolve } from '../numerics/convolution/index.ts'
import { complexOps as C, type ComplexNumber } from '../numerics/differentiate/index.ts'
import type { ComplexArray } from '../numerics/polynomial/index.ts'
import { reshape, sliceAxis, type TensorLike, tensor, type Value } from '../tensor/index.ts'
import { asComplexArray, float, fromList, numbers, polyOf, toList } from './_shared.ts'
import { type Polynomials, tf2zpk, type ZerosPolesGainData } from './conversions.ts'

// Second-order sections as scipy.signal: an `[n, 6]` array of rows [b0, b1, b2, a0, a1, a2].

/** How zpk2sos pairs poles with zeros (scipy.signal.zpk2sos). */
export type SosPairing = 'nearest' | 'keep_odd' | 'minimal'

/** Options for {@link zpk2sos}. */
export interface Zpk2SosOptions {
  /** The pairing rule (default 'nearest' for digital, 'minimal' for analog systems). */
  readonly pairing?: SosPairing
  /** Whether the system is analog (worst poles are then the closest to the imaginary axis). */
  readonly analog?: boolean
}

const isReal = (z: ComplexNumber): boolean => z.im === 0

/** One of each conjugate pair (positive imaginary part) then the reals, in scipy's _cplxreal order. */
function cplxreal(zs: readonly ComplexNumber[]): ComplexNumber[] {
  const tol = 100 * Number.EPSILON
  const sorted = [...zs].sort((a, b) => a.re - b.re || Math.abs(a.im) - Math.abs(b.im))
  const real = sorted.filter((z) => Math.abs(z.im) <= tol * Math.hypot(z.re, z.im)).map((z) => C.of(z.re))
  const rest = sorted.filter((z) => Math.abs(z.im) > tol * Math.hypot(z.re, z.im))
  const pos = rest.filter((z) => z.im > 0)
  const neg = rest.filter((z) => z.im < 0)
  if (pos.length !== neg.length) throw new WhiteboxError('value', 'zpk2sos: a complex value has no conjugate')
  // Within runs of (nearly) equal real parts, order by the size of the imaginary part.
  const byRuns = (xs: ComplexNumber[]): ComplexNumber[] => {
    const out: ComplexNumber[] = []
    let start = 0
    for (let i = 1; i <= xs.length; i++) {
      const prev = xs[i - 1] as ComplexNumber
      const cur = xs[i]
      const same = cur !== undefined && cur.re - prev.re <= tol * Math.hypot(prev.re, prev.im)
      if (!same) {
        out.push(...xs.slice(start, i).sort((a, b) => Math.abs(a.im) - Math.abs(b.im)))
        start = i
      }
    }
    return out
  }
  const p = byRuns(pos)
  const n = byRuns(neg)
  const paired = p.map((z, i) => {
    const w = n[i] as ComplexNumber
    if (Math.hypot(z.re - w.re, z.im + w.im) > tol * Math.hypot(w.re, w.im)) {
      throw new WhiteboxError('value', 'zpk2sos: a complex value has no conjugate')
    }
    return C.of((z.re + w.re) / 2, (z.im - w.im) / 2)
  })
  return [...paired, ...real]
}

function nearest(from: readonly ComplexNumber[], to: ComplexNumber, which: 'real' | 'complex' | 'any'): number {
  const order = from.map((_, i) => i).sort((i, j) => dist(from[i], to) - dist(from[j], to) || i - j)
  const ok = order.filter((i) => which === 'any' || isReal(from[i] as ComplexNumber) === (which === 'real'))
  const found = ok[0]
  // As scipy: minimal analog pairing cannot finish a first-order section when only complex zeros remain.
  if (found === undefined) throw new WhiteboxError('value', `zpk2sos: no ${which} zero remains to pair with a pole`)
  return found
}

const dist = (a: ComplexNumber | undefined, b: ComplexNumber): number =>
  a === undefined ? Number.POSITIVE_INFINITY : Math.hypot(a.re - b.re, a.im - b.im)

function section(zs: readonly ComplexNumber[], ps: readonly ComplexNumber[]): number[] {
  const b = polyOf(fromList(zs))
  const a = polyOf(fromList(ps))
  const out = [0, 0, 0, 0, 0, 0]
  b.forEach((v, i) => {
    out[3 - b.length + i] = v
  })
  a.forEach((v, i) => {
    out[6 - a.length + i] = v
  })
  return out
}

const take = (xs: ComplexNumber[], i: number): ComplexNumber => xs.splice(i, 1)[0] as ComplexNumber

/**
 * Second-order sections from zeros, poles and gain (scipy.signal.zpk2sos): poles are taken from the worst (closest to
 * the unit circle, or to the imaginary axis for analog systems) and each is paired with its nearest zero; the gain
 * goes in the first section.
 *
 * @param zeros - Zeros (conjugate pairs).
 * @param poles - Poles (conjugate pairs).
 * @param gain - The gain.
 * @param options - Pairing rule and whether the system is analog.
 * @returns The sections `[n, 6]`.
 * Pairings 'nearest' and 'keep_odd' first pad the zeros (or poles) with values at the origin to equal counts, as
 * scipy, which reads the zeros and poles as factors (1 - z_i z^-1): a discrete system with fewer zeros than poles is
 * then advanced in time. {@link toSos} pairs such systems minimally. Where scipy fails (minimal pairing with only
 * complex zeros left for a real pole or a real zero), the real values get a section without them.
 *
 * @throws WhiteboxError with code `value` for an unpaired complex value or an invalid pairing.
 * @concept second-order-sections
 * @example
 * ```ts
 * import { zpk2sos } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(zpk2sos([-1], [0.5], 2)) // => [[2, 2, 0, 1, -0.5, 0]]
 * ```
 */
export function zpk2sos(
  zeros: readonly number[] | ComplexArray,
  poles: readonly number[] | ComplexArray,
  gain: number,
  options: Zpk2SosOptions = {},
): Value {
  const analog = options.analog ?? false
  const pairing = options.pairing ?? (analog ? 'minimal' : 'nearest')
  if (analog && pairing !== 'minimal') {
    throw new WhiteboxError('value', "zpk2sos: analog conversion needs pairing 'minimal'")
  }
  let z = toList(asComplexArray(zeros))
  let p = toList(asComplexArray(poles))
  if (z.length === 0 && p.length === 0) return tensor([analog ? [0, 0, gain, 0, 0, 1] : [gain, 0, 0, 1, 0, 0]])
  let nSections: number
  if (pairing !== 'minimal') {
    while (p.length < z.length) p.push(C.of(0))
    while (z.length < p.length) z.push(C.of(0))
    nSections = Math.floor((p.length + 1) / 2)
    if (p.length % 2 === 1 && pairing === 'nearest') {
      p.push(C.of(0))
      z.push(C.of(0))
    }
  } else {
    if (p.length < z.length) throw new WhiteboxError('value', 'zpk2sos: minimal pairing needs at least as many poles')
    nSections = Math.floor((p.length + 1) / 2)
  }
  z = cplxreal(z)
  p = cplxreal(p)
  const badness = (x: ComplexNumber): number => (analog ? Math.abs(x.re) : Math.abs(1 - Math.hypot(x.re, x.im)))
  const worst = (xs: readonly ComplexNumber[]): number => {
    let best = 0
    xs.forEach((x, i) => {
      if (badness(x) < badness(xs[best] as ComplexNumber)) best = i
    })
    return best
  }
  const sos: number[][] = Array.from({ length: nSections }, () => [0, 0, 0, 0, 0, 0])
  const realCount = (xs: readonly ComplexNumber[]): number => xs.filter(isReal).length
  for (let si = nSections - 1; si >= 0; si--) {
    const p1 = take(p, worst(p))
    if (isReal(p1) && realCount(p) === 0) {
      // The last remaining real pole: a first-order section.
      if (pairing !== 'minimal') {
        const z1 = take(z, nearest(z, p1, 'real'))
        sos[si] = section([z1, C.of(0)], [p1, C.of(0)])
      } else if (z.some(isReal)) {
        sos[si] = section([take(z, nearest(z, p1, 'real'))], [p1])
      } else {
        // scipy raises here when only complex zeros remain; a pole-only first-order section keeps them for the
        // complex poles still to come.
        sos[si] = section([], [p1])
      }
    } else if (p.length + 1 === z.length && !isReal(p1) && realCount(p) === 1 && realCount(z) === 1) {
      // One real pole and one real zero remain: pair this complex pole with a complex zero.
      const z1 = take(z, nearest(z, p1, 'complex'))
      sos[si] = section([z1, C.of(z1.re, -z1.im)], [p1, C.of(p1.re, -p1.im)])
    } else {
      let p2: ComplexNumber
      if (isReal(p1)) {
        const reals = p.map((x, i) => [x, i] as const).filter(([x]) => isReal(x))
        const pick = reals[worst(reals.map(([x]) => x))]
        p2 = take(p, pick?.[1] ?? 0)
      } else {
        p2 = C.of(p1.re, -p1.im)
      }
      if (z.length > 0) {
        const z1 = take(z, nearest(z, p1, 'any'))
        if (!isReal(z1)) {
          sos[si] = section([z1, C.of(z1.re, -z1.im)], [p1, p2])
        } else if (z.some(isReal)) {
          sos[si] = section([z1, take(z, nearest(z, p1, 'real'))], [p1, p2])
        } else {
          // Only complex zeros left (minimal pairing with fewer zeros than poles, where scipy fails): the real zero
          // goes alone, the complex pairs to later sections.
          sos[si] = section([z1], [p1, p2])
        }
      } else {
        sos[si] = section([], [p1, p2])
      }
    }
  }
  const first = sos[0] as number[]
  for (let i = 0; i < 3; i++) first[i] = (first[i] ?? 0) * gain
  return tensor(sos)
}

function checkSections(sos: Value, name: string): number {
  if (sos.shape.length !== 2 || sos.shape[1] !== 6)
    throw new WhiteboxError('shape', `${name}: expected [n, 6] sections`)
  return sos.shape[0] ?? 0
}

/**
 * The single transfer function of a cascade of second-order sections (scipy.signal.sos2tf), by polynomial products,
 * differentiable in the coefficients.
 *
 * @param sos - Sections `[n, 6]`.
 * @returns The numerator and denominator (each of length 2n + 1).
 * @example
 * ```ts
 * import { sos2tf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(sos2tf([[1, 1, 0, 1, -0.5, 0], [1, 0, 0, 1, 0, 0.25]]).den) // => [1, -0.5, 0.25, -0.125, 0]
 * ```
 */
export function sos2tf(sos: TensorLike): Polynomials {
  const s = float(sos)
  const n = checkSections(s, 'sos2tf')
  let num: Value = tensor([1])
  let den: Value = tensor([1])
  for (let i = 0; i < n; i++) {
    const row = sliceAxis(s, 0, i, i + 1)
    num = convolve(num, reshape(sliceAxis(row, 1, 0, 3), [3]))
    den = convolve(den, reshape(sliceAxis(row, 1, 3, 6), [3]))
  }
  return { num, den }
}

/**
 * Zeros, poles and gain of a cascade of second-order sections (scipy.signal.sos2zpk): by default two zeros and two
 * poles per section, effectively-absent ones at the origin, as scipy. A section with leading zero coefficients (a
 * first-order section written [0, b1, b2, 0, a1, a2]) then gains a zero at the origin that it does not have; with
 * `pad` false only the actual roots are returned, which is exact.
 *
 * @param sos - Sections `[n, 6]`.
 * @param options - Whether to pad each section to two zeros and two poles (default true).
 * @returns Zeros, poles and gain.
 * @example
 * ```ts
 * import { sos2zpk } from 'whitebox-engine/systems'
 * sos2zpk([[2, 2, 0, 1, -0.5, 0]]).poles.re // => [0, 0.5]
 * ```
 */
export function sos2zpk(sos: TensorLike, options: { readonly pad?: boolean } = {}): ZerosPolesGainData {
  const pad = options.pad ?? true
  const s = float(sos)
  const n = checkSections(s, 'sos2zpk')
  const rowsOf = numbers(s)
  const zeros: ComplexNumber[] = []
  const poles: ComplexNumber[] = []
  let gain = 1
  for (let i = 0; i < n; i++) {
    const r = rowsOf.slice(6 * i, 6 * i + 6)
    const part = tf2zpk(r.slice(0, 3), r.slice(3, 6))
    const zs = toList(part.zeros)
    const ps = toList(part.poles)
    if (pad) {
      for (let j = 0; j < 2; j++) {
        zeros.push(zs[j] ?? C.of(0))
        poles.push(ps[j] ?? C.of(0))
      }
    } else {
      zeros.push(...zs)
      poles.push(...ps)
    }
    gain *= part.gain
  }
  return { zeros: fromList(zeros), poles: fromList(poles), gain }
}

/**
 * Second-order sections of a transfer function (scipy.signal.tf2sos), through its zeros, poles and gain.
 *
 * @param num - Numerator.
 * @param den - Denominator.
 * @param options - Pairing rule and whether the system is analog.
 * @returns The sections `[n, 6]`.
 * @example
 * ```ts
 * import { tf2sos } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(tf2sos([2, 2], [1, -0.5])) // => [[2, 2, 0, 1, -0.5, 0]]
 * ```
 */
export function tf2sos(num: TensorLike, den: TensorLike, options: Zpk2SosOptions = {}): Value {
  const { zeros, poles, gain } = tf2zpk(num, den)
  return zpk2sos(zeros, poles, gain, options)
}
