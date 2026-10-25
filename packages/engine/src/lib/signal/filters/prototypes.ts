/**
 * The analog lowpass prototypes that IIR design starts from (Butterworth, Chebyshev I and II, elliptic from Jacobi
 * elliptic functions, and Bessel), the frequency transformations that turn a prototype into a lowpass, highpass,
 * bandpass or bandstop filter, and the bilinear transform to a digital filter, all on zeros, poles and gain as in
 * scipy.signal.
 */

import { MdmlError } from '../../base/index.ts'
import { complexOps as C, type ComplexNumber } from '../../numerics/differentiate/index.ts'
import { ellipj, ellipk } from '../../numerics/special/index.ts'
import type { ZerosPolesGainData } from '../../systems/index.ts'
import { item, toFlat } from '../../tensor/index.ts'
import { arcJacSc1, ellipDeg, secant } from './_numeric.ts'

// Analog lowpass prototypes (cutoff 1 rad/s) and the frequency transformations of scipy.signal: lp2lp, lp2hp, lp2bp,
// lp2bs and the bilinear transform, all on zeros, poles and gain. Zeros and poles are complex arrays (parallel real
// and imaginary parts), as in the systems domain.

/** Zeros, poles and gain as lists of complex numbers. */
interface Zpk {
  readonly z: ComplexNumber[]
  readonly p: ComplexNumber[]
  readonly k: number
}

function pack(s: Zpk): ZerosPolesGainData {
  return {
    zeros: { re: s.z.map((v) => v.re), im: s.z.map((v) => v.im) },
    poles: { re: s.p.map((v) => v.re), im: s.p.map((v) => v.im) },
    gain: s.k,
  }
}

function unpack(s: ZerosPolesGainData): Zpk {
  return {
    z: s.zeros.re.map((re, i) => C.of(re, s.zeros.im[i] ?? 0)),
    p: s.poles.re.map((re, i) => C.of(re, s.poles.im[i] ?? 0)),
    k: s.gain,
  }
}

function order(n: number): number {
  if (!Number.isInteger(n) || n < 0) throw new MdmlError('value', 'the filter order must be a non-negative integer')
  return n
}

const prod = (zs: readonly ComplexNumber[]): ComplexNumber => zs.reduce((a, b) => C.mul(a, b), C.of(1))
const neg = (z: ComplexNumber): ComplexNumber => C.of(-z.re, -z.im)

/**
 * The analog Butterworth prototype (scipy.signal.buttap): N poles evenly spaced on the left half of the unit circle,
 * no zeros, unit gain.
 *
 * @param n - Order.
 * @returns Zeros, poles and gain.
 * @concept iir-design
 * @example
 * ```ts
 * import { buttap } from 'mdml-engine/signal'
 * buttap(1) // => { zeros: { re: [], im: [] }, poles: { re: [-1], im: [0] }, gain: 1 }
 * ```
 */
export function buttap(n: number): ZerosPolesGainData {
  const p: ComplexNumber[] = []
  for (let m = -order(n) + 1; m < n; m += 2) {
    const t = (Math.PI * m) / (2 * n)
    p.push(C.of(-Math.cos(t), m === 0 ? 0 : -Math.sin(t)))
  }
  return pack({ z: [], p, k: 1 })
}

/**
 * The analog Chebyshev type I prototype (scipy.signal.cheb1ap): equiripple passband with rp dB of ripple, poles on
 * an ellipse.
 *
 * @param n - Order.
 * @param rp - Passband ripple in dB.
 * @returns Zeros, poles and gain.
 * @concept iir-design
 * @example
 * ```ts
 * import { cheb1ap } from 'mdml-engine/signal'
 * Math.round(cheb1ap(1, 3).poles.re[0] * 1e6) / 1e6 // => -1.002377
 * ```
 */
export function cheb1ap(n: number, rp: number): ZerosPolesGainData {
  if (order(n) === 0) return pack({ z: [], p: [], k: 10 ** (-rp / 20) })
  const eps = Math.sqrt(10 ** (0.1 * rp) - 1)
  const mu = Math.asinh(1 / eps) / n
  const p: ComplexNumber[] = []
  for (let m = -n + 1; m < n; m += 2) {
    const theta = (Math.PI * m) / (2 * n)
    // -sinh(mu + i theta)
    p.push(C.of(-Math.sinh(mu) * Math.cos(theta), -Math.cosh(mu) * Math.sin(theta)))
  }
  let k = prod(p.map(neg)).re
  if (n % 2 === 0) k /= Math.sqrt(1 + eps * eps)
  return pack({ z: [], p, k })
}

/**
 * The analog Chebyshev type II prototype (scipy.signal.cheb2ap): monotonic passband and an equiripple stopband rs dB
 * down, with zeros on the imaginary axis.
 *
 * @param n - Order.
 * @param rs - Stopband attenuation in dB.
 * @returns Zeros, poles and gain.
 * @concept iir-design
 * @example
 * ```ts
 * import { cheb2ap } from 'mdml-engine/signal'
 * cheb2ap(2, 20).zeros.re.length // => 2
 * ```
 */
export function cheb2ap(n: number, rs: number): ZerosPolesGainData {
  if (order(n) === 0) return pack({ z: [], p: [], k: 1 })
  const de = 1 / Math.sqrt(10 ** (0.1 * rs) - 1)
  const mu = Math.asinh(1 / de) / n
  const ms: number[] = []
  if (n % 2 === 1) {
    for (let m = -n + 1; m < 0; m += 2) ms.push(m)
    for (let m = 2; m < n; m += 2) ms.push(m)
  } else for (let m = -n + 1; m < n; m += 2) ms.push(m)
  const z = ms.map((m) => C.of(0, 1 / Math.sin((m * Math.PI) / (2 * n))))
  const p: ComplexNumber[] = []
  for (let m = -n + 1; m < n; m += 2) {
    const theta = (Math.PI * m) / (2 * n)
    p.push(C.div(C.of(-1), C.of(Math.sinh(mu) * Math.cos(theta), Math.cosh(mu) * Math.sin(theta))))
  }
  const k = C.div(prod(p.map(neg)), prod(z.map(neg))).re
  return pack({ z, p, k })
}

/**
 * The analog elliptic (Cauer) prototype (scipy.signal.ellipap): equiripple in both bands, rp dB passband ripple and
 * rs dB stopband attenuation, from Jacobi elliptic functions and the degree equation.
 *
 * @param n - Order.
 * @param rp - Passband ripple in dB.
 * @param rs - Stopband attenuation in dB.
 * @returns Zeros, poles and gain.
 * @throws MdmlError with code `value` for impossible specifications.
 * @concept iir-design
 * @example
 * ```ts
 * import { ellipap } from 'mdml-engine/signal'
 * ellipap(3, 1, 40).poles.re.length // => 3
 * ```
 */
export function ellipap(n: number, rp: number, rs: number): ZerosPolesGainData {
  if (order(n) === 0) return pack({ z: [], p: [], k: 10 ** (-rp / 20) })
  const pow10m1 = (x: number): number => Math.expm1(Math.LN10 * x)
  if (n === 1) {
    const p = -Math.sqrt(1 / pow10m1(0.1 * rp))
    return pack({ z: [], p: [C.of(p)], k: -p })
  }
  const epsSq = pow10m1(0.1 * rp)
  const eps = Math.sqrt(epsSq)
  const ck1Sq = epsSq / pow10m1(0.1 * rs)
  if (ck1Sq === 0) throw new MdmlError('value', 'ellipap: cannot design a filter with these rp and rs')
  const val0 = item(ellipk(ck1Sq))
  const m = ellipDeg(n, ck1Sq)
  const capk = item(ellipk(m))
  const js: number[] = []
  for (let j = 1 - (n % 2); j < n; j += 2) js.push(j)
  const ej = ellipj(
    js.map((j) => (j * capk) / n),
    js.map(() => m),
  )
  const s = toFlat(ej.sn)
  const c = toFlat(ej.cn)
  const d = toFlat(ej.dn)
  const zs = s.filter((v) => Math.abs(v) > 2e-16).map((v) => C.of(0, 1 / (Math.sqrt(m) * v)))
  const z = [...zs, ...zs.map((v) => C.of(v.re, -v.im))]
  const r = arcJacSc1(1 / eps, ck1Sq)
  const v0 = (capk * r) / (n * val0)
  const ev = ellipj([v0], [1 - m])
  const sv = toFlat(ev.sn)[0] ?? 0
  const cv = toFlat(ev.cn)[0] ?? 0
  const dv = toFlat(ev.dn)[0] ?? 0
  const p0 = s.map((si, i) => {
    const den = 1 - ((d[i] ?? 0) * sv) ** 2
    return C.of((-(c[i] ?? 0) * (d[i] ?? 0) * sv * cv) / den, (-si * dv) / den)
  })
  let p: ComplexNumber[]
  if (n % 2 === 1) {
    let norm2 = 0
    for (const v of p0) norm2 += v.re * v.re + v.im * v.im
    const keep = p0.filter((v) => Math.abs(v.im) > 2e-16 * Math.sqrt(norm2))
    p = [...p0, ...keep.map((v) => C.of(v.re, -v.im))]
  } else p = [...p0, ...p0.map((v) => C.of(v.re, -v.im))]
  let k = C.div(prod(p.map(neg)), prod(z.map(neg))).re
  if (n % 2 === 0) k /= Math.sqrt(1 + epsSq)
  return pack({ z, p, k })
}

function fallingFactorial(x: number, n: number): number {
  let v = 1
  for (let k = x - n + 1; k <= x; k++) v *= k
  return v
}

/** Zeros of the Bessel polynomial y_N by Aberth iteration from Campos and Calderon's estimates, then symmetrised. */
function besselZeros(n: number): ComplexNumber[] {
  if (n === 1) return [C.of(-1)]
  const pv = (x: number, c: readonly number[]): number => c.reduce((acc, ci, i) => acc + ci * x ** i, 0)
  const s = pv(n, [0, 0, 2, 0, -3, 1])
  const b3 = pv(n, [16, -8]) / s
  const b2 = pv(n, [-24, -12, 12]) / s
  const b1 = pv(n, [8, 24, -12, -2]) / s
  const b0 = pv(n, [0, -6, 0, 5, -1]) / s
  const r = pv(n, [0, 0, 2, 1])
  const a1 = pv(n, [-6, -6]) / r
  const a2 = 6 / r
  let x = Array.from({ length: n }, (_, i) => C.of(pv(i + 1, [0, a1, a2]), pv(i + 1, [b0, b1, b2, b3])))
  // y_N(x) = sum_k (N + k)! / ((N - k)! k! 2^k) x^k, lowest degree first.
  const coef = Array.from(
    { length: n + 1 },
    (_, k) => fallingFactorial(n + k, 2 * k) / (fallingFactorial(k, k) * 2 ** k),
  )
  const f = (z: ComplexNumber): ComplexNumber => {
    let acc = C.of(0)
    for (let k = n; k >= 0; k--) acc = C.add(C.mul(acc, z), coef[k] ?? 0)
    return acc
  }
  const fp = (z: ComplexNumber): ComplexNumber => {
    let acc = C.of(0)
    for (let k = n; k >= 1; k--) acc = C.add(C.mul(acc, z), k * (coef[k] ?? 0))
    return acc
  }
  for (let it = 0; it < 100; it++) {
    const alpha = x.map((z) => C.div(neg(f(z)), fp(z)))
    const next = x.map((z, k) => {
      let beta = C.of(0)
      for (const [j, w] of x.entries()) if (j !== k) beta = C.add(beta, C.div(C.of(1), C.sub(z, w)))
      const a = alpha[k] ?? C.of(0)
      return C.add(z, C.div(a, C.add(C.of(1), C.mul(a, beta))))
    })
    x = next
    if (alpha.every((a) => Math.hypot(a.re, a.im) <= 1e-15 * Math.max(1, Math.hypot(x[0]?.re ?? 0, x[0]?.im ?? 0))))
      break
  }
  // Newton polish, then average with the conjugate reversal as scipy does.
  x = x.map((z0) => {
    let z = z0
    for (let it = 0; it < 5; it++) z = C.sub(z, C.div(f(z), fp(z)))
    return z
  })
  return x.map((z, i) => {
    const w = x[n - 1 - i] ?? z
    return C.of(0.5 * (z.re + w.re), 0.5 * (z.im - w.im))
  })
}

/** Normalisation of the Bessel prototype. */
export type BesselNorm = 'phase' | 'delay' | 'mag'

/**
 * The analog Bessel (Thomson) prototype (scipy.signal.besselap): maximally flat group delay; normalised so the phase
 * response is -N pi / 4 at 1 rad/s ('phase', default), the group delay at DC is 1 ('delay'), or the gain is -3 dB at
 * 1 rad/s ('mag').
 *
 * @param n - Order.
 * @param norm - Normalisation.
 * @returns Zeros, poles and gain.
 * @concept iir-design
 * @example
 * ```ts
 * import { besselap } from 'mdml-engine/signal'
 * besselap(1, 'delay') // => { zeros: { re: [], im: [] }, poles: { re: [-1], im: [0] }, gain: 1 }
 * ```
 */
export function besselap(n: number, norm: BesselNorm = 'phase'): ZerosPolesGainData {
  if (order(n) === 0) return pack({ z: [], p: [], k: 1 })
  let p = besselZeros(n).map((z) => C.div(C.of(1), z))
  const aLast = Math.floor(fallingFactorial(2 * n, n) / 2 ** n)
  let k = 1
  if (norm === 'delay' || norm === 'mag') {
    k = aLast
    if (norm === 'mag') {
      const g = (w: number): number => {
        let d = C.of(1)
        for (const q of p) d = C.mul(d, C.sub(C.of(0, w), q))
        return Math.abs(k) / Math.hypot(d.re, d.im) - 1 / Math.SQRT2
      }
      const nf = secant(g, 1.5)
      p = p.map((q) => C.div(q, nf))
      k = nf ** -n * aLast
    }
  } else if (norm === 'phase') {
    const s = 10 ** (-Math.log10(aLast) / n)
    p = p.map((q) => C.mul(q, s))
  } else throw new MdmlError('value', `besselap: unknown normalisation '${norm as string}'`)
  return pack({ z: [], p, k })
}

function degree(s: Zpk): number {
  const d = s.p.length - s.z.length
  if (d < 0) throw new MdmlError('value', 'improper transfer function: more zeros than poles')
  return d
}

/**
 * Lowpass to lowpass (scipy.signal.lp2lp_zpk): scale the cutoff to wo.
 *
 * @param s - Zeros, poles and gain.
 * @param wo - New cutoff (rad/s, default 1).
 * @returns The transformed system.
 * @concept frequency-transformations
 * @example
 * ```ts
 * import { buttap, lp2lpZpk } from 'mdml-engine/signal'
 * lp2lpZpk(buttap(1), 3).poles.re // => [-3]
 * ```
 */
export function lp2lpZpk(s: ZerosPolesGainData, wo = 1): ZerosPolesGainData {
  const u = unpack(s)
  return pack({ z: u.z.map((v) => C.mul(v, wo)), p: u.p.map((v) => C.mul(v, wo)), k: u.k * wo ** degree(u) })
}

/**
 * Lowpass to highpass (scipy.signal.lp2hp_zpk): s -> wo / s, adding zeros at the origin.
 *
 * @param s - Zeros, poles and gain.
 * @param wo - Cutoff (default 1).
 * @returns The transformed system.
 * @concept frequency-transformations
 * @example
 * ```ts
 * import { buttap, lp2hpZpk } from 'mdml-engine/signal'
 * lp2hpZpk(buttap(1), 2).zeros.re // => [0]
 * ```
 */
export function lp2hpZpk(s: ZerosPolesGainData, wo = 1): ZerosPolesGainData {
  const u = unpack(s)
  const d = degree(u)
  const z = [...u.z.map((v) => C.div(C.of(wo), v)), ...Array.from({ length: d }, () => C.of(0))]
  const k = u.k * C.div(prod(u.z.map(neg)), prod(u.p.map(neg))).re
  return pack({ z, p: u.p.map((v) => C.div(C.of(wo), v)), k })
}

function bandSplit(v: ComplexNumber, wo: number): [ComplexNumber, ComplexNumber] {
  const root = C.sqrt(C.sub(C.mul(v, v), wo * wo))
  return [C.add(v, root), C.sub(v, root)]
}

/**
 * Lowpass to bandpass (scipy.signal.lp2bp_zpk): centre wo, bandwidth bw; each zero and pole splits in two and the
 * relative degree adds zeros at the origin.
 *
 * @param s - Zeros, poles and gain.
 * @param wo - Centre frequency (default 1).
 * @param bw - Bandwidth (default 1).
 * @returns The transformed system.
 * @concept frequency-transformations
 * @example
 * ```ts
 * import { buttap, lp2bpZpk } from 'mdml-engine/signal'
 * lp2bpZpk(buttap(1), 1, 1).poles.re.length // => 2
 * ```
 */
export function lp2bpZpk(s: ZerosPolesGainData, wo = 1, bw = 1): ZerosPolesGainData {
  const u = unpack(s)
  const d = degree(u)
  const zl = u.z.map((v) => C.mul(v, bw / 2))
  const pl = u.p.map((v) => C.mul(v, bw / 2))
  const zs = zl.map((v) => bandSplit(v, wo))
  const ps = pl.map((v) => bandSplit(v, wo))
  return pack({
    z: [...zs.map((q) => q[0]), ...zs.map((q) => q[1]), ...Array.from({ length: d }, () => C.of(0))],
    p: [...ps.map((q) => q[0]), ...ps.map((q) => q[1])],
    k: u.k * bw ** d,
  })
}

/**
 * Lowpass to bandstop (scipy.signal.lp2bs_zpk): centre wo, bandwidth bw; the relative degree adds zeros at +-j wo.
 *
 * @param s - Zeros, poles and gain.
 * @param wo - Centre frequency (default 1).
 * @param bw - Bandwidth (default 1).
 * @returns The transformed system.
 * @concept frequency-transformations
 * @example
 * ```ts
 * import { buttap, lp2bsZpk } from 'mdml-engine/signal'
 * lp2bsZpk(buttap(1), 2, 1).zeros.im // => [2, -2]
 * ```
 */
export function lp2bsZpk(s: ZerosPolesGainData, wo = 1, bw = 1): ZerosPolesGainData {
  const u = unpack(s)
  const d = degree(u)
  const zh = u.z.map((v) => C.div(C.of(bw / 2), v))
  const ph = u.p.map((v) => C.div(C.of(bw / 2), v))
  const zs = zh.map((v) => bandSplit(v, wo))
  const ps = ph.map((v) => bandSplit(v, wo))
  const k = u.k * C.div(prod(u.z.map(neg)), prod(u.p.map(neg))).re
  return pack({
    z: [
      ...zs.map((q) => q[0]),
      ...zs.map((q) => q[1]),
      ...Array.from({ length: d }, () => C.of(0, wo)),
      ...Array.from({ length: d }, () => C.of(0, -wo)),
    ],
    p: [...ps.map((q) => q[0]), ...ps.map((q) => q[1])],
    k,
  })
}

/**
 * The bilinear transform of zeros, poles and gain (scipy.signal.bilinear_zpk): s = 2 fs (z - 1) / (z + 1), mapping
 * the left half-plane into the unit disc and adding zeros at z = -1 for the relative degree.
 *
 * @param s - Analog zeros, poles and gain.
 * @param fs - Sampling rate.
 * @returns The digital system.
 * @concept bilinear-transform
 * @example
 * ```ts
 * import { bilinearZpk, buttap } from 'mdml-engine/signal'
 * bilinearZpk(buttap(1), 0.5) // => { zeros: { re: [-1], im: [0] }, poles: { re: [0], im: [0] }, gain: 0.5 }
 * ```
 */
export function bilinearZpk(s: ZerosPolesGainData, fs: number): ZerosPolesGainData {
  if (!(fs > 0)) throw new MdmlError('value', 'bilinearZpk: fs must be positive')
  const u = unpack(s)
  const d = degree(u)
  const fs2 = 2 * fs
  const map = (v: ComplexNumber): ComplexNumber => C.div(C.add(C.of(fs2), v), C.sub(C.of(fs2), v))
  const k = u.k * C.div(prod(u.z.map((v) => C.sub(C.of(fs2), v))), prod(u.p.map((v) => C.sub(C.of(fs2), v)))).re
  return pack({ z: [...u.z.map(map), ...Array.from({ length: d }, () => C.of(-1))], p: u.p.map(map), k })
}
