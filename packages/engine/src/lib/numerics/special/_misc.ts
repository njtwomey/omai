// Sine and cosine integrals, Fresnel integrals and Dawson's integral (series for small arguments, continued fractions
// or asymptotic series for large ones, after Numerical Recipes cisi and frenel); Owen's T function by Gauss-Legendre
// quadrature; the Kolmogorov and Smirnov distributions of the Kolmogorov-Smirnov statistic.

import { EULER_GAMMA } from '../../base/index.ts'
import { EPS, lbinom, ndtr, SQRT_2PI } from './_kernels.ts'

const FPMIN = 1e-300

interface Complex {
  re: number
  im: number
}
const cmul = (a: Complex, b: Complex): Complex => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re })
const cdiv = (a: Complex, b: Complex): Complex => {
  const den = b.re * b.re + b.im * b.im
  return { re: (a.re * b.re + a.im * b.im) / den, im: (a.im * b.re - a.re * b.im) / den }
}
const cadd = (a: Complex, b: Complex): Complex => ({ re: a.re + b.re, im: a.im + b.im })
const ONE: Complex = { re: 1, im: 0 }

/** Si(x) and Ci(|x|) (the real part of Ci for negative x, as scipy returns). */
export function sici(x: number): { si: number; ci: number } {
  if (Number.isNaN(x)) return { si: Number.NaN, ci: Number.NaN }
  const t = Math.abs(x)
  if (t === 0) return { si: 0, ci: -Infinity }
  if (t === Infinity) return { si: Math.sign(x) * (Math.PI / 2), ci: 0 }
  let si: number
  let ci: number
  if (t > 2) {
    // Continued fraction for E1(i t).
    let b: Complex = { re: 1, im: t }
    let c: Complex = { re: 1 / FPMIN, im: 0 }
    let d = cdiv(ONE, b)
    let h = d
    for (let i = 2; i <= 1000; i++) {
      const a = -(i - 1) * (i - 1)
      b = cadd(b, { re: 2, im: 0 })
      d = cdiv(ONE, cadd({ re: a * d.re, im: a * d.im }, b))
      c = cadd(b, cdiv({ re: a, im: 0 }, c))
      const del = cmul(c, d)
      h = cmul(h, del)
      if (Math.abs(del.re - 1) + Math.abs(del.im) < EPS) break
    }
    h = cmul({ re: Math.cos(t), im: -Math.sin(t) }, h)
    ci = -h.re
    si = Math.PI / 2 + h.im
  } else {
    let sum = 0
    let sums = 0
    let sumc = 0
    let sign = 1
    let fact = 1
    let odd = true
    for (let k = 1; k <= 200; k++) {
      fact *= t / k
      const term = fact / k
      sum += sign * term
      const err = term / Math.abs(sum)
      if (odd) {
        sign = -sign
        sums = sum
        sum = sumc
      } else {
        sumc = sum
        sum = sums
      }
      if (err < EPS) break
      odd = !odd
    }
    si = sums
    ci = sumc + Math.log(t) + EULER_GAMMA
  }
  return { si: x < 0 ? -si : si, ci }
}

/** The Fresnel integrals S(x) = int_0^x sin(pi t^2 / 2) dt and C(x) = int_0^x cos(pi t^2 / 2) dt. */
export function fresnel(x: number): { s: number; c: number } {
  if (Number.isNaN(x)) return { s: Number.NaN, c: Number.NaN }
  const ax = Math.abs(x)
  let s: number
  let c: number
  if (ax === Infinity) {
    s = 0.5
    c = 0.5
  } else if (ax < 1e-150) {
    s = 0
    c = ax
  } else if (ax <= 1.5) {
    let sum = 0
    let sums = 0
    let sumc = ax
    let sign = 1
    const fact = (Math.PI / 2) * ax * ax
    let odd = true
    let term = ax
    let n = 3
    for (let k = 1; k <= 200; k++) {
      term *= fact / k
      sum += (sign * term) / n
      const test = Math.abs(sum) * EPS
      if (odd) {
        sign = -sign
        sums = sum
        sum = sumc
      } else {
        sumc = sum
        sum = sums
      }
      if (term < test) break
      odd = !odd
      n += 2
    }
    s = sums
    c = sumc
  } else {
    const pix2 = Math.PI * ax * ax
    let b: Complex = { re: 1, im: -pix2 }
    let cc: Complex = { re: 1 / FPMIN, im: 0 }
    let d = cdiv(ONE, b)
    let h = d
    let n = -1
    for (let k = 2; k <= 1000; k++) {
      n += 2
      const a = -n * (n + 1)
      b = cadd(b, { re: 4, im: 0 })
      d = cdiv(ONE, cadd({ re: a * d.re, im: a * d.im }, b))
      cc = cadd(b, cdiv({ re: a, im: 0 }, cc))
      const del = cmul(cc, d)
      h = cmul(h, del)
      if (Math.abs(del.re - 1) + Math.abs(del.im) < EPS) break
    }
    h = cmul({ re: ax, im: -ax }, h)
    // cos and sin of pi x^2 / 2, reduced exactly: x^2 / 2 mod 2 in units of pi.
    const half = (ax * ax) / 2
    const r = half - 2 * Math.floor(half / 2)
    const phase: Complex = { re: Math.cos(Math.PI * r), im: Math.sin(Math.PI * r) }
    const t = cmul(phase, h)
    const cs = cmul({ re: 0.5, im: 0.5 }, { re: 1 - t.re, im: -t.im })
    c = cs.re
    s = cs.im
  }
  return x < 0 ? { s: -s, c: -c } : { s, c }
}

/** Dawson's integral D(x) = e^{-x^2} int_0^x e^{t^2} dt. */
export function dawsn(x: number): number {
  if (Number.isNaN(x)) return Number.NaN
  const ax = Math.abs(x)
  let value: number
  if (ax < 6) {
    // e^{-x^2} sum_n x^{2n+1} / (n! (2n + 1)): all terms positive.
    const x2 = ax * ax
    let term = ax
    let sum = ax
    for (let n = 1; n < 1000; n++) {
      term *= x2 / n
      const add = term / (2 * n + 1)
      sum += add
      if (add < EPS * sum) break
    }
    value = Math.exp(-x2) * sum
  } else {
    // D(x) ~ 1 / (2x) sum_n (2n - 1)!! / (2 x^2)^n.
    const inv = 1 / (2 * ax * ax)
    let term = 1
    let sum = 1
    for (let n = 1; n < 100; n++) {
      const next = term * (2 * n - 1) * inv
      if (next > term) break
      term = next
      sum += term
      if (term < EPS * sum) break
    }
    value = sum / (2 * ax)
  }
  return x < 0 ? -value : value
}

// Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration on P_n), computed on first use.
let gl: { x: number[]; w: number[] } | undefined
function gaussLegendre(): { x: number[]; w: number[] } {
  if (gl !== undefined) return gl
  const n = 48
  const x: number[] = []
  const w: number[] = []
  for (let i = 1; i <= n; i++) {
    let z = Math.cos((Math.PI * (i - 0.25)) / (n + 0.5))
    let pp = 0
    for (let it = 0; it < 100; it++) {
      let p1 = 1
      let p2 = 0
      for (let j = 1; j <= n; j++) {
        const p3 = p2
        p2 = p1
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
      }
      pp = (n * (z * p1 - p2)) / (z * z - 1)
      const dz = p1 / pp
      z -= dz
      if (Math.abs(dz) < 1e-16) break
    }
    x.push(z)
    w.push(2 / ((1 - z * z) * pp * pp))
  }
  gl = { x, w }
  return gl
}

function owensTIntegral(h: number, a: number): number {
  // (1 / 2 pi) int_0^a exp(-h^2 (1 + t^2) / 2) / (1 + t^2) dt, truncated where the integrand is below 1e-17 of its peak.
  const upper = Math.min(a, h === 0 ? a : 9 / Math.abs(h))
  const { x, w } = gaussLegendre()
  const half = upper / 2
  const e = Math.exp(-0.5 * h * h)
  let sum = 0
  for (let i = 0; i < x.length; i++) {
    const t = half * ((x[i] ?? 0) + 1)
    const q = 1 + t * t
    sum += ((w[i] ?? 0) * Math.exp(-0.5 * h * h * t * t)) / q
  }
  return ((e * half) / (2 * Math.PI)) * sum
}

/** Owen's T function T(h, a), the bivariate normal probability behind skew-normal CDFs. */
export function owensT(h: number, a: number): number {
  if (Number.isNaN(h) || Number.isNaN(a)) return Number.NaN
  if (a < 0) return -owensT(h, -a)
  if (a === 0) return 0
  const ah = Math.abs(h)
  if (a <= 1) return owensTIntegral(ah, a)
  // T(h, a) = (Phi(h) + Phi(ah)) / 2 - Phi(h) Phi(ah) - [h < 0] / 2 - T(ah, 1 / a), with |h| (T is even in h).
  if (a === Infinity) return ah === 0 ? 0.25 : 0.5 * ndtr(-ah)
  // Written with upper tails to keep relative accuracy for large h.
  const pt = ndtr(-ah)
  const qt = ndtr(-a * ah)
  return 0.5 * (pt + qt) - pt * qt - owensTIntegral(a * ah, 1 / a)
}

/** Survival function of the Kolmogorov distribution (scipy `kolmogorov`). */
export function kolmogorov(x: number): number {
  if (Number.isNaN(x)) return Number.NaN
  if (x <= 0) return 1
  if (x < 1) {
    // 1 - (sqrt(2 pi) / x) sum_k exp(-(2k - 1)^2 pi^2 / (8 x^2)) (Jacobi theta transformation).
    let s = 0
    for (let k = 1; k <= 50; k++) {
      const term = Math.exp((-((2 * k - 1) ** 2) * Math.PI * Math.PI) / (8 * x * x))
      s += term
      if (term < EPS * s) break
    }
    return 1 - (SQRT_2PI / x) * s
  }
  let s = 0
  for (let k = 1; k <= 100; k++) {
    const term = Math.exp(-2 * k * k * x * x)
    s += (k % 2 === 1 ? 1 : -1) * term
    if (term < EPS * Math.abs(s)) break
  }
  return 2 * s
}

/** The Kolmogorov density (minus the derivative of {@link kolmogorov}). */
export function kolmogorovPdf(x: number): number {
  if (Number.isNaN(x)) return Number.NaN
  if (x <= 0) return 0
  if (x < 1) {
    // d/dx of 1 - (sqrt(2 pi) / x) sum_k exp(-c_k / x^2), c_k = (2k - 1)^2 pi^2 / 8.
    let s = 0
    for (let k = 1; k <= 50; k++) {
      const ck = ((2 * k - 1) ** 2 * Math.PI * Math.PI) / 8
      const term = Math.exp(-ck / (x * x)) * ((2 * ck) / (x * x) - 1)
      s += term
      if (Math.abs(term) < EPS * Math.abs(s)) break
    }
    return (SQRT_2PI / (x * x)) * s
  }
  let s = 0
  for (let k = 1; k <= 100; k++) {
    const term = 4 * k * k * x * Math.exp(-2 * k * k * x * x)
    s += (k % 2 === 1 ? 1 : -1) * term
    if (term < EPS * Math.abs(s)) break
  }
  return 2 * s
}

/** Exact survival function of the one-sided Kolmogorov-Smirnov statistic, P(D_n^+ >= d) (Birnbaum-Tingey). */
export function smirnov(n: number, d: number): number {
  if (Number.isNaN(d) || !(n >= 1) || !Number.isInteger(n)) return Number.NaN
  if (d <= 0) return 1
  if (d >= 1) return 0
  let sum = 0
  const jmax = Math.floor(n * (1 - d))
  for (let j = 0; j <= jmax; j++) {
    const base = 1 - d - j / n
    if (base <= 0) continue
    const lt = lbinom(n, j) + (n - j) * Math.log(base) + (j - 1) * Math.log(d + j / n)
    sum += Math.exp(lt)
  }
  return d * sum
}
