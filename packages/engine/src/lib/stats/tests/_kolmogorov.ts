// The null distributions of Kolmogorov-Smirnov statistics, following scipy.stats (Simard and L'Ecuyer's selection of
// methods): the exact two-sided one-sample distribution by Marsaglia-Tsang-Wang's form of Durbin's matrix, the
// Ruben-Gambino and Smirnov tails, Pelz and Good's asymptotic series for large n, and the exact two-sample
// distribution by counting lattice paths through the band the statistic allows.

import { smirnov } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'

const E128 = 2 ** 128
const EM128 = 2 ** -128
const SQRT2PI = Math.sqrt(2 * Math.PI)

const smirnovSf = (n: number, d: number): number => item(smirnov(n, d))

function matmul(a: Float64Array, b: Float64Array, m: number): Float64Array {
  const c = new Float64Array(m * m)
  for (let i = 0; i < m; i++)
    for (let k = 0; k < m; k++) {
      const v = a[i * m + k] ?? 0
      if (v === 0) continue
      for (let j = 0; j < m; j++) c[i * m + j] = (c[i * m + j] ?? 0) + v * (b[k * m + j] ?? 0)
    }
  return c
}

// P(D_n <= d) by Durbin's matrix (Marsaglia, Tsang and Wang 2003), with exponent bookkeeping against overflow.
function durbin(n: number, d: number): number {
  const nd = n * d
  if (d >= 1) return 1
  if (nd <= 0.5) return 0
  const k = Math.ceil(nd)
  const h = k - nd
  const m = 2 * k - 1
  const v = new Float64Array(m)
  const w = new Float64Array(m)
  let fac = 1
  for (let j = 1; j <= m; j++) {
    w[j - 1] = fac
    fac /= j
    v[j - 1] = (1 - h ** j) * fac
  }
  const tt = Math.max(2 * h - 1, 0) ** m - 2 * h ** m
  v[m - 1] = (1 + tt) * fac
  let H: Float64Array = new Float64Array(m * m)
  for (let i = 1; i < m; i++) for (let r = i - 1; r < m; r++) H[r * m + i] = w[r - i + 1] ?? 0
  for (let r = 0; r < m; r++) H[r * m] = v[r] ?? 0
  for (let c = 0; c < m; c++) H[(m - 1) * m + c] = v[m - 1 - c] ?? 0
  let P: Float64Array = new Float64Array(m * m)
  for (let i = 0; i < m; i++) P[i * m + i] = 1
  let nn = n
  let expnt = 0
  let hExp = 0
  while (nn > 0) {
    if (nn % 2) {
      P = matmul(P, H, m)
      expnt += hExp
    }
    H = matmul(H, H, m)
    hExp *= 2
    if (Math.abs(H[(k - 1) * m + (k - 1)] ?? 0) > E128) {
      for (let i = 0; i < H.length; i++) H[i] = (H[i] ?? 0) / E128
      hExp += 128
    }
    nn = Math.floor(nn / 2)
  }
  let p = P[(k - 1) * m + (k - 1)] ?? 0
  for (let i = 1; i <= n; i++) {
    p = (i * p) / n
    if (Math.abs(p) < EM128) {
      p *= E128
      expnt -= 128
    }
  }
  return p * 2 ** expnt
}

// Pelz and Good's (1976) asymptotic series for P(D_n <= x).
function pelzGood(n: number, x: number): number {
  if (x <= 0) return 0
  if (x >= 1) return 1
  const z = Math.sqrt(n) * x
  const z2 = z * z
  const z4 = z2 * z2
  const z6 = z4 * z2
  const qlog = -(Math.PI ** 2) / 8 / z2
  if (qlog < -708) return 0
  let q = Math.exp(qlog)
  const pi2 = Math.PI ** 2
  const pi4 = pi2 * pi2
  const pi6 = pi4 * pi2
  const k1a = -z2
  const k1b = pi2 / 4
  const k2a = 6 * z6 + 2 * z4
  const k2b = ((2 * z4 - 5 * z2) * pi2) / 4
  const k2c = (pi4 * (1 - 2 * z2)) / 16
  const k3d = (pi6 * (5 - 30 * z2)) / 64
  const k3c = (pi4 * (-60 * z2 + 212 * z4)) / 16
  const k3b = (pi2 * (135 * z4 - 96 * z6)) / 4
  const k3a = -30 * z6 - 90 * z ** 8
  const K = [0, 0, 0, 0]
  const maxk = Math.ceil((16 * z) / Math.PI)
  for (let k = maxk; k > 0; k--) {
    const m = 2 * k - 1
    const m2 = m * m
    const qp = q ** (8 * k)
    const coeffs = [1, k1a + k1b * m2, k2a + k2b * m2 + k2c * m2 * m2, k3a + k3b * m2 + k3c * m2 * m2 + k3d * m2 ** 3]
    for (let i = 0; i < 4; i++) K[i] = (K[i] ?? 0) * qp + (coeffs[i] ?? 0)
  }
  const dens = [z, 6 * z4, 72 * z ** 7, 6480 * z ** 10]
  for (let i = 0; i < 4; i++) K[i] = ((K[i] ?? 0) * q * SQRT2PI) / (dens[i] ?? 1)
  q = Math.exp(-pi2 / 2 / z2)
  let k2extra = 0
  let k3extra = 0
  const sqrt3z = Math.sqrt(3) * z
  for (let k = maxk; k > 0; k--) {
    const qp = q ** (k * k)
    k2extra += k * k * qp
    k3extra += (sqrt3z + Math.PI * k) * (sqrt3z - Math.PI * k) * k * k * qp
  }
  K[2] = (K[2] ?? 0) + (k2extra * pi2 * SQRT2PI) / (-36 * z ** 3)
  K[3] = (K[3] ?? 0) + (k3extra * pi2 * SQRT2PI) / (216 * z6)
  let s = 0
  for (let i = 0; i < 4; i++) s += (K[i] ?? 0) / n ** (i / 2)
  return s
}

const clip = (p: number): number => Math.min(1, Math.max(0, p))

/**
 * The two-sided one-sample Kolmogorov distribution: P(D_n >= x) (`sf`) or P(D_n <= x), choosing among exact and
 * asymptotic methods as scipy.stats.kstwo does.
 */
export function kolmogn(n: number, x: number, sf: boolean): number {
  const pick = (cdf: number, tail: number): number => clip(sf ? tail : cdf)
  if (x >= 1) return pick(1, 0)
  if (x <= 0) return pick(0, 1)
  const t = n * x
  if (t <= 1) {
    if (t <= 0.5) return pick(0, 1)
    let prob = 1
    for (let i = 1; i <= n; i++) prob *= (i / n) * (2 * t - 1)
    return pick(prob, 1 - prob)
  }
  if (t >= n - 1) {
    const prob = 2 * (1 - x) ** n
    return pick(1 - prob, prob)
  }
  if (x >= 0.5) {
    const prob = 2 * smirnovSf(n, x)
    return pick(1 - prob, prob)
  }
  const nx2 = t * x
  if (n <= 140) {
    if (nx2 <= 4) {
      const prob = durbin(n, x)
      return pick(prob, 1 - prob)
    }
    const prob = 2 * smirnovSf(n, x)
    return pick(1 - prob, prob)
  }
  if (sf) {
    if (nx2 >= 370) return 0
    if (nx2 >= 2.2) return clip(2 * smirnovSf(n, x))
  }
  let cdf: number
  if (nx2 >= 18) cdf = 1
  else if (n <= 100000 && n * x ** 1.5 <= 1.4) cdf = durbin(n, x)
  else cdf = pelzGood(n, x)
  return pick(cdf, 1 - cdf)
}

function gcd(a: number, b: number): number {
  let x = a
  let y = b
  while (y) [x, y] = [y, x % y]
  return x
}

/**
 * The exact null tail of the two-sample Kolmogorov-Smirnov statistic for sizes n1, n2: P(D >= d) (two-sided) or P(D+ >=
 * d) (one-sided), by the probability mass of a random lattice path from (0, 0) to (n1, n2) that leaves the band the
 * statistic allows. Returns the rounded statistic h / lcm as scipy does.
 */
export function ks2Exact(n1: number, n2: number, d: number, twoSided: boolean): { d: number; p: number } {
  const g = gcd(n1, n2)
  const lcm = (n1 / g) * n2
  const h = Math.round(d * lcm)
  const dr = h / lcm
  if (h === 0) return { d: dr, p: 1 }
  const a = n2 / g
  const b = n1 / g
  // Probability of reaching (x, y) while staying inside: |x a - y b| < h (two-sided) or x a - y b < h (one-sided).
  const inside = (x: number, y: number): boolean => {
    const s = x * a - y * b
    return twoSided ? Math.abs(s) < h : s < h
  }
  // Mass that first leaves the band is accumulated directly, so small tail probabilities keep their accuracy.
  let out = 0
  let prev = new Float64Array(n2 + 1)
  prev[0] = 1
  for (let y = 1; y <= n2; y++) {
    const incoming = (prev[y - 1] ?? 0) * ((n2 - y + 1) / (n1 + n2 - y + 1))
    if (inside(0, y)) prev[y] = incoming
    else out += incoming
  }
  for (let x = 1; x <= n1; x++) {
    const cur = new Float64Array(n2 + 1)
    for (let y = 0; y <= n2; y++) {
      const rest = n1 + n2 - x - y + 1
      let incoming = (prev[y] ?? 0) * ((n1 - x + 1) / rest)
      if (y > 0) incoming += (cur[y - 1] ?? 0) * ((n2 - y + 1) / rest)
      if (inside(x, y)) cur[y] = incoming
      else out += incoming
    }
    prev = cur
  }
  return { d: dr, p: clip(out) }
}
