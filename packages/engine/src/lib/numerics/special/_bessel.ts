// Bessel functions of real order: Temme's series for small arguments, Steed's continued fractions otherwise
// (Numerical Recipes bessjy and bessik), and the large-argument asymptotic expansions. Every routine returns the
// order-v value together with the order-(v+1) value, which the derivative rules need.

import { EULER_GAMMA } from '../../base/index.ts'
import { cospi, EPS, gamma, hurwitzZeta, sinpi } from './_kernels.ts'

const MAXIT = 1_000_000
const FPMIN = 1e-300
const XMIN = 2
// zeta(k) for odd k, for the odd part of lgamma(1 + mu) used in gam1 (computed on first use).
let oddZeta: number[] | undefined

/**
 * Temme's gamma combinations for |mu| <= 1/2: gam1 = (1/Gamma(1-mu) - 1/Gamma(1+mu)) / (2 mu), gam2 = (1/Gamma(1-mu) +
 * 1/Gamma(1+mu)) / 2, gampl = 1/Gamma(1+mu), gammi = 1/Gamma(1-mu). gam1 is computed without cancellation from the odd
 * part of lgamma(1 + mu): lgamma(1+mu) - lgamma(1-mu) = -2 (gamma mu + sum_{k odd >= 3} zeta(k) mu^k / k).
 */
function temmeGammas(mu: number): { gam1: number; gam2: number; gampl: number; gammi: number } {
  const gampl = 1 / gamma(1 + mu)
  const gammi = 1 / gamma(1 - mu)
  oddZeta ??= Array.from({ length: 60 }, (_, k) => (k >= 3 && k % 2 === 1 ? hurwitzZeta(k, 1) : 0))
  let odd = EULER_GAMMA * mu
  let pow = mu
  for (let k = 3; k < 60; k += 2) {
    pow *= mu * mu
    const term = ((oddZeta[k] ?? 0) * pow) / k
    odd += term
    if (Math.abs(term) <= EPS * Math.abs(odd)) break
  }
  // gammi / gampl = exp(lgamma(1+mu) - lgamma(1-mu)) = exp(-2 odd).
  const ratio = mu === 0 ? -EULER_GAMMA : (gampl * Math.expm1(-2 * odd)) / (2 * mu)
  return { gam1: ratio, gam2: (gammi + gampl) / 2, gampl, gammi }
}

/** Result of the Bessel routines: values of order v and v + 1 (exponentially scaled where stated). */
export interface BesselPair {
  readonly v: number
  readonly v1: number
}

// ---------------------------------------------------------------------------------------------------------------------
// Modified Bessel functions I and K, both exponentially scaled: e^{-x} I_v(x) and e^{x} K_v(x), for v >= 0 and x > 0.

/** e^{x} K_v(x) ~ (pi / 2x)^{1/2} sum_k a_k(v) / x^k, for x well above v^2. */
function asymptoticScaledK(v: number, x: number): number {
  const mu = 4 * v * v
  let term = 1
  let sum = 1
  for (let k = 1; k < 200; k++) {
    const next = (term * (mu - (2 * k - 1) ** 2)) / (k * 8 * x)
    if (Math.abs(next) > Math.abs(term) && k > 1) break
    term = next
    sum += term
    if (Math.abs(term) < EPS * Math.abs(sum)) break
  }
  return Math.sqrt(Math.PI / (2 * x)) * sum
}

function useAsymptotic(v: number, x: number): boolean {
  return x > 40 + v * v
}

/** Scaled I_v, I_{v+1}, K_v and K_{v+1} for v >= 0, x > 0. */
export function besselIK(v: number, x: number): { i: BesselPair; k: BesselPair } {
  if (useAsymptotic(v + 1, x)) {
    const iv = asymptoticScaledI(v, x)
    const iv1 = asymptoticScaledI(v + 1, x)
    return { i: { v: iv, v1: iv1 }, k: { v: asymptoticScaledK(v, x), v1: asymptoticScaledK(v + 1, x) } }
  }
  const nl = Math.floor(v + 0.5)
  const xmu = v - nl
  const xmu2 = xmu * xmu
  const xi = 1 / x
  const xi2 = 2 * xi
  // CF1: f_v = I'_v / I_v.
  let h = Math.max(v * xi, FPMIN)
  let b = xi2 * v
  let d = 0
  let c = h
  for (let i = 1; i <= MAXIT; i++) {
    b += xi2
    d = 1 / (b + d)
    c = b + 1 / c
    const del = c * d
    h *= del
    if (Math.abs(del - 1) < EPS) break
  }
  // Downward recurrence from order v to order mu (unnormalised), rescaling to avoid overflow.
  let ril = FPMIN
  let ripl = h * ril
  let ril1 = ril
  let rip1 = ripl
  let fact = v * xi
  for (let l = nl; l >= 1; l--) {
    const ritemp = fact * ril + ripl
    fact -= xi
    ripl = fact * ritemp + ril
    ril = ritemp
    if (Math.abs(ril) > 1e250) {
      ril *= 1e-250
      ripl *= 1e-250
      ril1 *= 1e-250
      rip1 *= 1e-250
    }
  }
  const f = ripl / ril
  let rkmu: number
  let rk1: number
  if (x < XMIN) {
    const x2 = 0.5 * x
    const pimu = Math.PI * xmu
    const factor = Math.abs(pimu) < EPS ? 1 : pimu / Math.sin(pimu)
    const dl = -Math.log(x2)
    const e = xmu * dl
    const fact2 = Math.abs(e) < EPS ? 1 : Math.sinh(e) / e
    const { gam1, gam2, gampl, gammi } = temmeGammas(xmu)
    let ff = factor * (gam1 * Math.cosh(e) + gam2 * fact2 * dl)
    let sum = ff
    const ee = Math.exp(e)
    let p = (0.5 * ee) / gampl
    let q = 0.5 / (ee * gammi)
    let cc = 1
    const dd = x2 * x2
    let sum1 = p
    for (let i = 1; i <= MAXIT; i++) {
      ff = (i * ff + p + q) / (i * i - xmu2)
      cc *= dd / i
      p /= i - xmu
      q /= i + xmu
      const del = cc * ff
      sum += del
      sum1 += cc * (p - i * ff)
      if (Math.abs(del) < Math.abs(sum) * EPS) break
    }
    // Scale by e^{x} so that K is e^{x} K and the Wronskian below gives e^{-x} I.
    const scale = Math.exp(x)
    rkmu = sum * scale
    rk1 = sum1 * xi2 * scale
  } else {
    // CF2 (Steed), with the e^{-x} factor of K omitted: K is computed scaled.
    let bb = 2 * (1 + x)
    let dd = 1 / bb
    let hh = dd
    let delh = dd
    let q1 = 0
    let q2 = 1
    const a1 = 0.25 - xmu2
    let q = a1
    let cc = a1
    let a = -a1
    let s = 1 + q * delh
    for (let i = 2; i <= MAXIT; i++) {
      a -= 2 * (i - 1)
      cc = (-a * cc) / i
      const qnew = (q1 - bb * q2) / a
      q1 = q2
      q2 = qnew
      q += cc * qnew
      bb += 2
      dd = 1 / (bb + a * dd)
      delh = (bb * dd - 1) * delh
      hh += delh
      const dels = q * delh
      s += dels
      if (Math.abs(dels / s) < EPS) break
    }
    hh = a1 * hh
    rkmu = Math.sqrt(Math.PI / (2 * x)) / s
    rk1 = rkmu * (xmu + x + 0.5 - hh) * xi
  }
  const rkmup = xmu * xi * rkmu - rk1
  const rimu = xi / (f * rkmu - rkmup)
  const iv = (rimu * ril1) / ril
  const ripv = (rimu * rip1) / ril
  for (let i = 1; i <= nl; i++) {
    const rktemp = (xmu + i) * xi2 * rk1 + rkmu
    rkmu = rk1
    rk1 = rktemp
  }
  // I_{v+1} = I'_v - (v / x) I_v (the scaling is unaffected since both are scaled by the same factor).
  return { i: { v: iv, v1: ripv - v * xi * iv }, k: { v: rkmu, v1: rk1 } }
}

/** e^{-x} I_v(x) ~ (2 pi x)^{-1/2} sum_k (-1)^k a_k(v) / x^k, for x well above v^2. */
function asymptoticScaledI(v: number, x: number): number {
  const mu = 4 * v * v
  let term = 1
  let sum = 1
  for (let k = 1; k < 200; k++) {
    const next = (-term * (mu - (2 * k - 1) ** 2)) / (k * 8 * x)
    if (Math.abs(next) > Math.abs(term) && k > 1) break
    term = next
    sum += term
    if (Math.abs(term) < EPS * Math.abs(sum)) break
  }
  return sum / Math.sqrt(2 * Math.PI * x)
}

/** e^{-|x|} I_v(x) for real v and x (negative x only for integer v; negative v by reflection). */
export function ive(v: number, x: number): number {
  if (Number.isNaN(v) || Number.isNaN(x)) return Number.NaN
  if (x < 0) {
    if (!Number.isInteger(v)) return Number.NaN
    const value = ive(v, -x)
    return Math.abs(v) % 2 === 1 ? -value : value
  }
  if (x === 0) return v === 0 ? 1 : v > 0 || Number.isInteger(v) ? 0 : Infinity
  if (v < 0) {
    if (Number.isInteger(v)) return ive(-v, x)
    // I_{-v} = I_v + (2 / pi) sin(v pi) K_v.
    const { i, k } = besselIK(-v, x)
    return i.v + (2 / Math.PI) * sinpi(-v) * k.v * Math.exp(-2 * x)
  }
  return besselIK(v, x).i.v
}

/** I_v(x). */
export function iv(v: number, x: number): number {
  const s = ive(v, x)
  if (s === 0 || !Number.isFinite(s)) return s
  const ax = Math.abs(x)
  return ax > 700 ? Math.sign(s) * Math.exp(Math.log(Math.abs(s)) + ax) : s * Math.exp(ax)
}

/** log I_v(x) for x > 0 (no overflow for large x). */
export function logIv(v: number, x: number): number {
  if (!(x > 0)) return x === 0 ? (v === 0 ? 0 : -Infinity) : Number.NaN
  return Math.log(ive(v, x)) + x
}

/** e^{x} K_v(x) for x > 0 (K is even in v). */
export function kve(v: number, x: number): number {
  if (Number.isNaN(v) || Number.isNaN(x) || x < 0) return Number.NaN
  if (x === 0) return Infinity
  return besselIK(Math.abs(v), x).k.v
}

/** K_v(x). */
export function kv(v: number, x: number): number {
  if (x > 700) return Math.exp(Math.log(kve(v, x)) - x)
  return kve(v, x) * Math.exp(-x)
}

// ---------------------------------------------------------------------------------------------------------------------
// Bessel functions J and Y for v >= 0 and x > 0.

function hankelAsymptotic(v: number, x: number): { j: number; y: number } {
  // J_v = sqrt(2 / (pi x)) (P cos chi - Q sin chi), Y_v = sqrt(2 / (pi x)) (P sin chi + Q cos chi).
  const mu = 4 * v * v
  let p = 1
  let q = 0
  let term = 1
  for (let k = 1; k < 200; k++) {
    const next = (term * (mu - (2 * k - 1) ** 2)) / (k * 8 * x)
    if (Math.abs(next) > Math.abs(term) && k > 1) break
    term = next
    // Terms alternate between Q (odd k) and P (even k), with signs (-1)^{floor(k/2)}.
    const signed = Math.floor(k / 2) % 2 === 0 ? term : -term
    if (k % 2 === 1) q += signed
    else p += signed
    if (Math.abs(term) < EPS * (Math.abs(p) + Math.abs(q))) break
  }
  // chi = x - (v / 2 + 1 / 4) pi, reduced exactly in the multiple of pi.
  const shift = v / 2 + 0.25
  const c = Math.cos(x) * cospi(shift) + Math.sin(x) * sinpi(shift)
  const s = Math.sin(x) * cospi(shift) - Math.cos(x) * sinpi(shift)
  const amp = Math.sqrt(2 / (Math.PI * x))
  return { j: amp * (p * c - q * s), y: amp * (p * s + q * c) }
}

/** J_v, J_{v+1}, Y_v, Y_{v+1} for v >= 0, x > 0. */
export function besselJY(v: number, x: number): { j: BesselPair; y: BesselPair } {
  if (x > 30 + (v + 1) * (v + 1)) {
    const a = hankelAsymptotic(v, x)
    const b = hankelAsymptotic(v + 1, x)
    return { j: { v: a.j, v1: b.j }, y: { v: a.y, v1: b.y } }
  }
  const nl = x < XMIN ? Math.floor(v + 0.5) : Math.max(0, Math.floor(v - x + 1.5))
  const xmu = v - nl
  const xmu2 = xmu * xmu
  const xi = 1 / x
  const xi2 = 2 * xi
  const w = xi2 / Math.PI
  let isign = 1
  let h = Math.max(v * xi, FPMIN)
  let b = xi2 * v
  let d = 0
  let c = h
  for (let i = 1; i <= MAXIT; i++) {
    b += xi2
    d = b - d
    if (Math.abs(d) < FPMIN) d = FPMIN
    c = b - 1 / c
    if (Math.abs(c) < FPMIN) c = FPMIN
    d = 1 / d
    const del = c * d
    h *= del
    if (d < 0) isign = -isign
    if (Math.abs(del - 1) < EPS) break
  }
  let rjl = isign * FPMIN
  let rjpl = h * rjl
  let rjl1 = rjl
  let rjp1 = rjpl
  let fact = v * xi
  for (let l = nl; l >= 1; l--) {
    const rjtemp = fact * rjl + rjpl
    fact -= xi
    rjpl = fact * rjtemp - rjl
    rjl = rjtemp
    if (Math.abs(rjl) > 1e250) {
      rjl *= 1e-250
      rjpl *= 1e-250
      rjl1 *= 1e-250
      rjp1 *= 1e-250
    }
  }
  if (rjl === 0) rjl = EPS
  const f = rjpl / rjl
  let rjmu: number
  let rymu: number
  let ry1: number
  if (x < XMIN) {
    const x2 = 0.5 * x
    const pimu = Math.PI * xmu
    const factor = Math.abs(pimu) < EPS ? 1 : pimu / Math.sin(pimu)
    const dl = -Math.log(x2)
    const e = xmu * dl
    const fact2 = Math.abs(e) < EPS ? 1 : Math.sinh(e) / e
    const { gam1, gam2, gampl, gammi } = temmeGammas(xmu)
    let ff = (2 / Math.PI) * factor * (gam1 * Math.cosh(e) + gam2 * fact2 * dl)
    const ee = Math.exp(e)
    let p = ee / (gampl * Math.PI)
    let q = 1 / (ee * Math.PI * gammi)
    const pimu2 = 0.5 * pimu
    const fact3 = Math.abs(pimu2) < EPS ? 1 : Math.sin(pimu2) / pimu2
    const r = Math.PI * pimu2 * fact3 * fact3
    let cc = 1
    const dd = -x2 * x2
    let sum = ff + r * q
    let sum1 = p
    for (let i = 1; i <= MAXIT; i++) {
      ff = (i * ff + p + q) / (i * i - xmu2)
      cc *= dd / i
      p /= i - xmu
      q /= i + xmu
      const del = cc * (ff + r * q)
      sum += del
      sum1 += cc * p - i * del
      if (Math.abs(del) < (1 + Math.abs(sum)) * EPS) break
    }
    rymu = -sum
    ry1 = -sum1 * xi2
    const rymup = xmu * xi * rymu - ry1
    rjmu = w / (rymup - f * rymu)
  } else {
    let a = 0.25 - xmu2
    let p = -0.5 * xi
    let q = 1
    const br = 2 * x
    let bi = 2
    let fct = (a * xi) / (p * p + q * q)
    let cr = br + q * fct
    let ci = bi + p * fct
    let den = br * br + bi * bi
    let dr = br / den
    let di = -bi / den
    let dlr = cr * dr - ci * di
    let dli = cr * di + ci * dr
    let temp = p * dlr - q * dli
    q = p * dli + q * dlr
    p = temp
    for (let i = 2; i <= MAXIT; i++) {
      a += 2 * (i - 1)
      bi += 2
      dr = a * dr + br
      di = a * di + bi
      if (Math.abs(dr) + Math.abs(di) < FPMIN) dr = FPMIN
      fct = a / (cr * cr + ci * ci)
      cr = br + cr * fct
      ci = bi - ci * fct
      if (Math.abs(cr) + Math.abs(ci) < FPMIN) cr = FPMIN
      den = dr * dr + di * di
      dr /= den
      di /= -den
      dlr = cr * dr - ci * di
      dli = cr * di + ci * dr
      temp = p * dlr - q * dli
      q = p * dli + q * dlr
      p = temp
      if (Math.abs(dlr - 1) + Math.abs(dli) < EPS) break
    }
    const gam = (p - f) / q
    rjmu = Math.sqrt(w / ((p - f) * gam + q))
    rjmu = rjl < 0 ? -rjmu : rjmu
    rymu = rjmu * gam
    const rymup = rymu * (p + q / gam)
    ry1 = xmu * xi * rymu - rymup
  }
  const scale = rjmu / rjl
  const jv = rjl1 * scale
  const jpv = rjp1 * scale
  for (let i = 1; i <= nl; i++) {
    const rytemp = (xmu + i) * xi2 * ry1 - rymu
    rymu = ry1
    ry1 = rytemp
  }
  // J_{v+1} = (v / x) J_v - J'_v.
  return { j: { v: jv, v1: v * xi * jv - jpv }, y: { v: rymu, v1: ry1 } }
}

/** J_v(x) for real v and x (negative x only for integer v; negative v by reflection). */
export function jv(v: number, x: number): number {
  if (Number.isNaN(v) || Number.isNaN(x)) return Number.NaN
  if (x < 0) {
    if (!Number.isInteger(v)) return Number.NaN
    const value = jv(v, -x)
    return Math.abs(v) % 2 === 1 ? -value : value
  }
  if (x === 0) return v === 0 ? 1 : v > 0 || Number.isInteger(v) ? 0 : Infinity
  if (v < 0) {
    if (Number.isInteger(v)) return (-v % 2 === 1 ? -1 : 1) * jv(-v, x)
    // J_{-v} = cos(v pi) J_v - sin(v pi) Y_v.
    const { j, y } = besselJY(-v, x)
    return cospi(-v) * j.v - sinpi(-v) * y.v
  }
  return besselJY(v, x).j.v
}

/** Y_v(x) for x > 0. */
export function yv(v: number, x: number): number {
  if (Number.isNaN(v) || Number.isNaN(x) || x < 0) return Number.NaN
  if (x === 0) return -Infinity
  if (v < 0) {
    if (Number.isInteger(v)) return (-v % 2 === 1 ? -1 : 1) * yv(-v, x)
    // Y_{-v} = sin(v pi) J_v + cos(v pi) Y_v.
    const { j, y } = besselJY(-v, x)
    return sinpi(-v) * j.v + cospi(-v) * y.v
  }
  return besselJY(v, x).y.v
}
