// Elliptic integrals (parameter m = k^2, as in scipy): complete integrals by the arithmetic-geometric mean, incomplete
// integrals by Carlson's symmetric forms R_F and R_D (Carlson 1995), and the Jacobi elliptic functions by the descending
// Landen (AGM) transformation (Cephes ellipj).
import { EPS } from './_kernels.ts'

export function ellipk(m: number): number {
  if (m > 1 || Number.isNaN(m)) return Number.NaN
  if (m === 1) return Infinity
  if (m === -Infinity) return 0
  let a = 1
  let g = Math.sqrt(1 - m)
  for (let i = 0; i < 60 && Math.abs(a - g) > EPS * a; i++) [a, g] = [(a + g) / 2, Math.sqrt(a * g)]
  return Math.PI / (2 * a)
}

export function ellipe(m: number): number {
  if (m > 1 || Number.isNaN(m)) return Number.NaN
  if (m === 1) return 1
  if (m === -Infinity) return Infinity
  let a = 1
  let g = Math.sqrt(1 - m)
  let c = m
  // sum 2^{n-1} c_n^2 with c_0^2 = m.
  let sum = 0.5 * c
  let pow2 = 0.5
  for (let i = 0; i < 60; i++) {
    const an = (a + g) / 2
    const cn = (a - g) / 2
    g = Math.sqrt(a * g)
    a = an
    pow2 *= 2
    c = cn * cn
    sum += pow2 * c
    if (Math.abs(cn) <= EPS * a) break
  }
  return (Math.PI / (2 * a)) * (1 - sum)
}

/** Carlson's R_F(x, y, z), at most one argument zero. */
export function carlsonRF(x: number, y: number, z: number): number {
  if (x < 0 || y < 0 || z < 0 || Number.isNaN(x + y + z)) return Number.NaN
  let xt = x
  let yt = y
  let zt = z
  for (let i = 0; i < 100; i++) {
    const sx = Math.sqrt(xt)
    const sy = Math.sqrt(yt)
    const sz = Math.sqrt(zt)
    const lambda = sx * (sy + sz) + sy * sz
    xt = 0.25 * (xt + lambda)
    yt = 0.25 * (yt + lambda)
    zt = 0.25 * (zt + lambda)
    const ave = (xt + yt + zt) / 3
    const dx = (ave - xt) / ave
    const dy = (ave - yt) / ave
    const dz = (ave - zt) / ave
    if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) < 1e-3) {
      const e2 = dx * dy - dz * dz
      const e3 = dx * dy * dz
      return (1 + (e2 / 24 - 0.1 - (3 / 44) * e3) * e2 + e3 / 14) / Math.sqrt(ave)
    }
  }
  return Number.NaN
}

/** Carlson's R_D(x, y, z), z > 0 and at most one of x, y zero. */
export function carlsonRD(x: number, y: number, z: number): number {
  if (x < 0 || y < 0 || !(z > 0)) return Number.NaN
  let xt = x
  let yt = y
  let zt = z
  let sum = 0
  let fac = 1
  for (let i = 0; i < 100; i++) {
    const sx = Math.sqrt(xt)
    const sy = Math.sqrt(yt)
    const sz = Math.sqrt(zt)
    const lambda = sx * (sy + sz) + sy * sz
    sum += fac / (sz * (zt + lambda))
    fac *= 0.25
    xt = 0.25 * (xt + lambda)
    yt = 0.25 * (yt + lambda)
    zt = 0.25 * (zt + lambda)
    const ave = 0.2 * (xt + yt + 3 * zt)
    const dx = (ave - xt) / ave
    const dy = (ave - yt) / ave
    const dz = (ave - zt) / ave
    if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) < 1e-3) {
      const c1 = 3 / 14
      const c2 = 1 / 6
      const c3 = 9 / 22
      const c4 = 3 / 26
      const ea = dx * dy
      const eb = dz * dz
      const ec = ea - eb
      const ed = ea - 6 * eb
      const ee = ed + ec + ec
      return (
        3 * sum +
        (fac *
          (1 + ed * (-c1 + 0.25 * c3 * ed - 1.5 * c4 * dz * ee) + dz * (c2 * ee + dz * (-c3 * ec + dz * c4 * ea)))) /
          (ave * Math.sqrt(ave))
      )
    }
  }
  return Number.NaN
}

// phi = n pi + psi with |psi| <= pi / 2, so that sin^2 psi and cos^2 psi are exact inputs to the Carlson forms.
function reduce(phi: number): { n: number; s: number; c: number } {
  const n = Math.round(phi / Math.PI)
  const psi = phi - n * Math.PI
  return { n, s: Math.sin(psi), c: Math.cos(psi) }
}

/** The incomplete elliptic integral of the first kind F(phi | m). */
export function ellipkinc(phi: number, m: number): number {
  if (Number.isNaN(phi) || Number.isNaN(m) || m > 1) return Number.NaN
  if (m === 0) return phi
  const { n, s, c } = reduce(phi)
  const f = s * carlsonRF(c * c, 1 - m * s * s, 1)
  return n === 0 ? f : 2 * n * ellipk(m) + f
}

/** The incomplete elliptic integral of the second kind E(phi | m). */
export function ellipeinc(phi: number, m: number): number {
  if (Number.isNaN(phi) || Number.isNaN(m) || m > 1) return Number.NaN
  if (m === 0) return phi
  const { n, s, c } = reduce(phi)
  const d = 1 - m * s * s
  const e = s * carlsonRF(c * c, d, 1) - ((m * s * s * s) / 3) * carlsonRD(c * c, d, 1)
  return n === 0 ? e : 2 * n * ellipe(m) + e
}

/** The Jacobi elliptic functions sn, cn, dn and the amplitude phi at (u, m), 0 <= m <= 1. */
export function ellipj(u: number, m: number): { sn: number; cn: number; dn: number; ph: number } {
  if (Number.isNaN(u) || Number.isNaN(m) || m < 0 || m > 1) {
    return { sn: Number.NaN, cn: Number.NaN, dn: Number.NaN, ph: Number.NaN }
  }
  if (m < 1e-9) {
    const t = Math.sin(u)
    const b = Math.cos(u)
    const ai = 0.25 * m * (u - t * b)
    return { sn: t - ai * b, cn: b + ai * t, dn: 1 - 0.5 * m * t * t, ph: u - ai }
  }
  if (m >= 0.9999999999) {
    let ai = 0.25 * (1 - m)
    const b = Math.cosh(u)
    const t = Math.tanh(u)
    const phi = 1 / b
    const twon = b * Math.sinh(u)
    const sn = t + (ai * (twon - u)) / (b * b)
    const ph = 2 * Math.atan(Math.exp(u)) - Math.PI / 2 + (ai * (twon - u)) / b
    ai *= t * phi
    return { sn, cn: phi - ai * (twon - u), dn: phi + ai * (twon + u), ph }
  }
  const a: number[] = [1]
  const c: number[] = [Math.sqrt(m)]
  let b = Math.sqrt(1 - m)
  let twon = 1
  let i = 0
  while (Math.abs((c[i] ?? 0) / (a[i] ?? 1)) > EPS && i < 40) {
    const ai = a[i] ?? 1
    i += 1
    c[i] = (ai - b) / 2
    const t = Math.sqrt(ai * b)
    a[i] = (ai + b) / 2
    b = t
    twon *= 2
  }
  let phi = twon * (a[i] ?? 1) * u
  let prev = phi
  for (; i > 0; i--) {
    const t = ((c[i] ?? 0) * Math.sin(phi)) / (a[i] ?? 1)
    prev = phi
    phi = (Math.asin(t) + phi) / 2
  }
  const cn = Math.cos(phi)
  return { sn: Math.sin(phi), cn, dn: cn / Math.cos(phi - prev), ph: phi }
}
