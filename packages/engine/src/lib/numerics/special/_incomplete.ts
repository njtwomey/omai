// Regularised incomplete gamma and beta functions with their derivatives with respect to every argument, and their
// inverses. The shape-parameter derivatives are computed by differentiating the series and continued fractions term by
// term (forward mode through the recurrences, as in Moore 1982 and Boik and Robison-Cox 1998), so they are as accurate
// as the values themselves, including in the tails; implicit reparameterisation of gamma, beta and Dirichlet samples
// depends on them.
import { digamma, digammaDiff, EPS, lbeta, lgamma, stirlingCorrection, TINY } from './_kernels.ts'

const MAXIT = 100000
const BIG = 4.503599627370496e15
const BIGINV = 2.220446049250313e-16

const LOG_SQRT_2PI = 0.9189385332046727

/** ln(1 + u) - u, without the cancellation of the two terms for small |u| (the series -u^2/2 + u^3/3 - ...). */
function log1pmx(u: number): number {
  if (Math.abs(u) >= 0.25) return Math.log1p(u) - u
  let power = u
  let sum = 0
  for (let k = 2; k < 60; k++) {
    power *= -u
    const term = power / k
    sum += term
    if (Math.abs(term) <= 1e-18 * Math.abs(sum)) break
  }
  return sum
}

/** The exact product of two numbers as an unevaluated sum p + e (Dekker's splitting; there is no fused multiply-add). */
function twoProd(a: number, b: number): [number, number] {
  const p = a * b
  const ca = 134217729 * a
  const ah = ca - (ca - a)
  const al = a - ah
  const cb = 134217729 * b
  const bh = cb - (cb - b)
  const bl = b - bh
  return [p, ah * bh - p + ah * bl + al * bh + al * bl]
}

// Above this size of a shape parameter the log of the prefix of the series and fractions, a ln x - x - lgamma(a) or
// a ln x + b ln(1 - x) - lbeta(a, b), is a difference of terms of order a that agree to the last digits: their rounding
// (a eps, 1e-9 at a = 1e7) is the error of the value. The prefix is then formed from the Stirling series in the
// variable u = x / mu - 1 around the mode, where it is a sum of terms of order one.
const STIRLING_GAMMA = 1000
const STIRLING_BETA = 10000

/** log (x^a e^-x / Gamma(a)), the prefix of the incomplete gamma function, accurate for large a. */
function gammaPrefix(a: number, x: number): number {
  if (a < STIRLING_GAMMA) return a * Math.log(x) - x - lgamma(a)
  // Gamma(a) = sqrt(2 pi) a^(a - 1/2) e^-a e^corr, so the prefix is a (ln(1 + u) - u) + (1/2) ln(a / 2 pi) - corr.
  return a * log1pmx((x - a) / a) + 0.5 * Math.log(a) - LOG_SQRT_2PI - stirlingCorrection(a)
}

/** log (x^a y^b / B(a, b)) with y = 1 - x given accurately (the caller may know it better than 1 - x), for large a or b. */
function betaPrefix(a: number, b: number, x: number, y: number): number {
  if (Math.max(a, b) < STIRLING_BETA) return a * Math.log(x) + b * Math.log1p(-x) - lbeta(a, b)
  if (Math.min(a, b) >= 10) {
    // With mu = a / (a + b): x^a y^b / B = sqrt(a b / (a + b) / 2 pi) e^-corr exp(a (ln(1 + u) - u) + b (ln(1 + v) - v)),
    // u = x / mu - 1, v = y / (1 - mu) - 1, (the linear terms cancel since a u + b v = 0 up to the rounding of y).
    const s = a + b
    const [px, ex] = twoProd(x, s)
    const [py, ey] = twoProd(y, s)
    const u = (px - a + ex) / a
    const v = (py - b + ey) / b
    const corr = stirlingCorrection(a) + stirlingCorrection(b) - stirlingCorrection(s)
    return a * log1pmx(u) + b * log1pmx(v) + 0.5 * Math.log((a * b) / s) - LOG_SQRT_2PI - corr
  }
  // One parameter small: the terms are of order one except a ln x or b ln y, formed from the better of x and y.
  const lx = x < 0.5 ? Math.log(x) : Math.log1p(-y)
  const ly = y < 0.5 ? Math.log(y) : Math.log1p(-x)
  return a * lx + b * ly - lbeta(a, b)
}

/** A value with its derivatives with respect to the first and second shape parameters. */
interface Dual {
  readonly v: number
  readonly da: number
  readonly db: number
}

// ---------------------------------------------------------------------------------------------------------------------
// Incomplete gamma: series for P where x <= max(1, a), continued fraction for Q elsewhere (Cephes igam / igamc).

/** P(a, x) and dP/da from the series; `logScale` returns log P instead of P (for underflowing values). */
function gammaPSeries(a: number, x: number): { logP: number; dlogP: number } {
  let r = a
  let c = 1
  let dc = 0
  let ans = 1
  let dans = 0
  for (let n = 0; n < MAXIT; n++) {
    r += 1
    dc = (x / r) * (dc - c / r)
    c *= x / r
    ans += c
    dans += dc
    if (c <= EPS * ans && Math.abs(dc) <= EPS * Math.abs(dans) + EPS * EPS) break
  }
  const logax = gammaPrefix(a, x)
  // P = (ax / a) ans; d log P / da = ln x - psi(a) - 1/a + dans / ans.
  return { logP: logax - Math.log(a) + Math.log(ans), dlogP: Math.log(x) - digamma(a) - 1 / a + dans / ans }
}

/** Q(a, x) and dQ/da from the continued fraction (with derivatives carried through the recurrences). */
function gammaQFraction(a: number, x: number): { logQ: number; dlogQ: number } {
  let y = 1 - a
  let z = x + y + 1
  let c = 0
  let pkm2 = 1
  let qkm2 = x
  let pkm1 = x + 1
  let qkm1 = z * x
  let dpkm2 = 0
  let dqkm2 = 0
  let dpkm1 = 0
  let dqkm1 = -x
  let ans = pkm1 / qkm1
  let dans = (dpkm1 - ans * dqkm1) / qkm1
  for (let n = 0; n < MAXIT; n++) {
    c += 1
    y += 1
    z += 2
    const yc = y * c
    const dyc = -c
    const pk = pkm1 * z - pkm2 * yc
    const qk = qkm1 * z - qkm2 * yc
    const dpk = dpkm1 * z - pkm1 - dpkm2 * yc - pkm2 * dyc
    const dqk = dqkm1 * z - qkm1 - dqkm2 * yc - qkm2 * dyc
    let t = 1
    let dt = 1
    if (qk !== 0) {
      const r = pk / qk
      const dr = (dpk - r * dqk) / qk
      t = Math.abs((ans - r) / r)
      dt = Math.abs(dans - dr) / Math.max(Math.abs(dr), TINY)
      ans = r
      dans = dr
    }
    pkm2 = pkm1
    pkm1 = pk
    qkm2 = qkm1
    qkm1 = qk
    dpkm2 = dpkm1
    dpkm1 = dpk
    dqkm2 = dqkm1
    dqkm1 = dqk
    if (Math.abs(pk) > BIG) {
      pkm2 *= BIGINV
      pkm1 *= BIGINV
      qkm2 *= BIGINV
      qkm1 *= BIGINV
      dpkm2 *= BIGINV
      dpkm1 *= BIGINV
      dqkm2 *= BIGINV
      dqkm1 *= BIGINV
    }
    if (t <= EPS && dt <= 1e-15) break
  }
  const logax = gammaPrefix(a, x)
  return { logQ: logax + Math.log(ans), dlogQ: Math.log(x) - digamma(a) + dans / ans }
}

function invalidGamma(a: number, x: number): boolean {
  return Number.isNaN(a) || Number.isNaN(x) || a <= 0 || x < 0
}

/** P(a, x), the regularised lower incomplete gamma function. */
export function gammainc(a: number, x: number): number {
  if (a === 0 && x > 0) return 1
  if (invalidGamma(a, x)) return Number.NaN
  if (x === 0) return 0
  if (x === Infinity) return 1
  if (x <= 1 || x <= a) return Math.exp(gammaPSeries(a, x).logP)
  return -Math.expm1(gammaQFraction(a, x).logQ)
}

/** Q(a, x) = 1 - P(a, x). */
export function gammaincc(a: number, x: number): number {
  if (a === 0 && x > 0) return 0
  if (invalidGamma(a, x)) return Number.NaN
  if (x === 0) return 1
  if (x === Infinity) return 0
  if (x <= 1 || x <= a) return -Math.expm1(gammaPSeries(a, x).logP)
  return Math.exp(gammaQFraction(a, x).logQ)
}

/** log P(a, x), accurate where P underflows. */
export function logGammainc(a: number, x: number): number {
  if (a === 0 && x > 0) return 0
  if (invalidGamma(a, x)) return Number.NaN
  if (x === 0) return -Infinity
  if (x <= 1 || x <= a) return gammaPSeries(a, x).logP
  return Math.log1p(-Math.exp(gammaQFraction(a, x).logQ))
}

/** log Q(a, x), accurate where Q underflows. */
export function logGammaincc(a: number, x: number): number {
  if (invalidGamma(a, x)) return Number.NaN
  if (x === 0) return 0
  if (x <= 1 || x <= a) return Math.log1p(-Math.exp(gammaPSeries(a, x).logP))
  return gammaQFraction(a, x).logQ
}

/** dP(a, x)/da (equal to -dQ/da), accurate in both tails. */
export function gammaincDa(a: number, x: number): number {
  if (invalidGamma(a, x)) return Number.NaN
  if (x === 0 || x === Infinity) return 0
  if (x <= 1 || x <= a) {
    const { logP, dlogP } = gammaPSeries(a, x)
    return Math.exp(logP) * dlogP
  }
  const { logQ, dlogQ } = gammaQFraction(a, x)
  return -Math.exp(logQ) * dlogQ
}

/** d log P(a, x) / da, finite where P underflows. */
export function logGammaincDa(a: number, x: number): number {
  if (invalidGamma(a, x)) return Number.NaN
  if (x === 0) return Number.NaN
  if (x <= 1 || x <= a) return gammaPSeries(a, x).dlogP
  // d log P / da = -(dQ / da) / P = -Q (d log Q / da) / (1 - Q).
  const { logQ, dlogQ } = gammaQFraction(a, x)
  return (-Math.exp(logQ) * dlogQ) / -Math.expm1(logQ)
}

/** d log Q(a, x) / da, finite where Q underflows. */
export function logGammainccDa(a: number, x: number): number {
  if (invalidGamma(a, x)) return Number.NaN
  if (x === 0) return 0
  if (x <= 1 || x <= a) {
    const { logP, dlogP } = gammaPSeries(a, x)
    return (-Math.exp(logP) * dlogP) / -Math.expm1(logP)
  }
  return gammaQFraction(a, x).dlogQ
}

/** dP(a, x)/dx, the gamma density with unit scale. */
export function gammaincDx(a: number, x: number): number {
  if (invalidGamma(a, x)) return Number.NaN
  if (x === 0) return a === 1 ? 1 : a < 1 ? Infinity : 0
  return Math.exp((a - 1) * Math.log(x) - x - lgamma(a))
}

/** The x with P(a, x) = p (Numerical Recipes starting point, then Halley steps on the accurate P or Q). */
export function gammaincinv(a: number, p: number): number {
  if (!(a > 0) || Number.isNaN(p) || p < 0 || p > 1) return Number.NaN
  if (p === 0) return 0
  if (p === 1) return Infinity
  return gammaInverse(a, p, 1 - p)
}

/** The x with Q(a, x) = q. */
export function gammainccinv(a: number, q: number): number {
  if (!(a > 0) || Number.isNaN(q) || q < 0 || q > 1) return Number.NaN
  if (q === 0) return Infinity
  if (q === 1) return 0
  return gammaInverse(a, 1 - q, q)
}

function gammaInverse(a: number, p: number, q: number): number {
  const a1 = a - 1
  const gln = lgamma(a)
  let x: number
  if (a > 1) {
    const pp = p < 0.5 ? p : q
    const t = Math.sqrt(-2 * Math.log(pp))
    let z = (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)) - t
    if (p < 0.5) z = -z
    x = Math.max(1e-3, a * (1 - 1 / (9 * a) - z / (3 * Math.sqrt(a))) ** 3)
  } else {
    const t = 1 - a * (0.253 + a * 0.12)
    // Written with q = 1 - p so that tiny upper-tail probabilities keep their precision.
    x = p < t ? (p / t) ** (1 / a) : 1 - Math.log(q / (1 - t))
  }
  // Newton-Halley on whichever of P and Q is smaller, so the residual keeps relative accuracy in both tails. Deep in
  // a tail the probability is exponential in x, so Newton runs on its logarithm instead, which is nearly linear.
  const useQ = q < p
  if (Math.min(p, q) < 1e-6) return logTailNewton(a, x, useQ, Math.log(useQ ? q : p))
  for (let j = 0; j < 200; j++) {
    if (!(x > 0)) x = Number.MIN_VALUE
    const density = Math.exp(a1 * Math.log(x) - x - gln)
    if (density === 0 || !Number.isFinite(density)) break
    const err = useQ ? -(gammaincc(a, x) - q) : gammainc(a, x) - p
    const u = err / density
    const dx = u / (1 - 0.5 * Math.min(1, u * (a1 / x - 1)))
    const next = x - dx
    x = next <= 0 ? x / 2 : next
    if (Math.abs(dx) <= 4 * EPS * x) break
  }
  return x
}

function logTailNewton(a: number, start: number, upper: boolean, target: number): number {
  let x = start
  for (let j = 0; j < 200; j++) {
    if (!(x > 0)) x = Number.MIN_VALUE
    const logF = upper ? logGammaincc(a, x) : logGammainc(a, x)
    // d log P / dx = density / P and d log Q / dx = -density / Q.
    const slope = (upper ? -1 : 1) * Math.exp((a - 1) * Math.log(x) - x - lgamma(a) - logF)
    if (!Number.isFinite(slope) || slope === 0) break
    const dx = (logF - target) / slope
    const next = x - dx
    x = next <= 0 ? x / 2 : next
    if (Math.abs(dx) <= 4 * EPS * x) break
  }
  return x
}

// ---------------------------------------------------------------------------------------------------------------------
// Incomplete beta: the Numerical Recipes continued fraction with derivatives in a and b carried through modified Lentz.

// `y` is 1 - x, known accurately by the caller. Near x = 1 with a large and b small the first terms of the fraction are
// 1 + (a quantity close to -1): the first denominator 1 - (a + b) x / (a + 1) and the odd-step terms 1 + d_m, with
// d_m = -(a + m)(a + b + m) x / ((a + 2m)(a + 1 + 2m)), each lose their digits to the cancellation (1e-9 relative at
// a = 5e7). They are formed instead from y (1 - b + (a + b) y) / (a + 1), and (1 + d_m) = (a (1 + 2m - b) + m (2 + 3m - b) +
// (a + m)(a + b + m) y) / ((a + 2m)(a + 1 + 2m)), and the deviations c - 1 and d - 1 are carried beside c and d so
// that 1 + aa / c and 1 + aa d are (1 + aa) plus a correction, not a difference of numbers that agree.
function betaFraction(a: number, b: number, x: number, y: number): Dual {
  const qab = a + b
  const qap = a + 1
  const qam = a - 1
  let c = 1
  let dca = 0
  let dcb = 0
  const nearOne = x > 0.5
  let d = nearOne ? (1 - b + qab * y) / qap : 1 - (qab * x) / qap
  let dda = (-x * (1 - b)) / (qap * qap)
  let ddb = -x / qap
  if (Math.abs(d) < TINY) {
    d = TINY
    dda = 0
    ddb = 0
  }
  // d - 1 for the reciprocal below: (1 - d0) / d0 with 1 - d0 = (a + b) x / (a + 1).
  let dm1 = nearOne ? (qab * x) / qap / d : 1 / d - 1
  let cm1 = 0
  d = 1 / d
  dda = -dda * d * d
  ddb = -ddb * d * d
  let h = d
  let dha = dda
  let dhb = ddb
  // Returns whether the value and both derivatives have stopped changing. The value alone is not enough: when b is a
  // small integer the fraction terminates (a numerator m (b - m) x vanishes) while its derivative in b does not.
  const step = (aa: number, daa: number, dab: number, onePlus: number = 1 + aa): boolean => {
    // d <- 1 / (1 + aa d), c <- 1 + aa / c, h <- h d c, with derivatives; 1 + aa d = (1 + aa) + aa (d - 1) and
    // 1 + aa / c = (1 + aa) - aa (c - 1) / c.
    const aad = aa * d
    let nd = nearOne ? onePlus + aa * dm1 : 1 + aad
    let ndA = daa * d + aa * dda
    let ndB = dab * d + aa * ddb
    if (Math.abs(nd) < TINY) {
      nd = TINY
      ndA = 0
      ndB = 0
    }
    let nc = nearOne ? onePlus - (aa * cm1) / c : 1 + aa / c
    const aac = aa / c
    let ncA = daa / c - (aa * dca) / (c * c)
    let ncB = dab / c - (aa * dcb) / (c * c)
    if (Math.abs(nc) < TINY) {
      nc = TINY
      ncA = 0
      ncB = 0
    }
    dm1 = -aad / nd
    cm1 = aac
    d = 1 / nd
    dda = -ndA * d * d
    ddb = -ndB * d * d
    c = nc
    dca = ncA
    dcb = ncB
    const del = d * c
    const delA = dda * c + d * dca
    const delB = ddb * c + d * dcb
    dha = dha * del + h * delA
    dhb = dhb * del + h * delB
    h *= del
    return (
      Math.abs(del - 1) < EPS &&
      Math.abs(h * delA) <= EPS * Math.abs(dha) + TINY &&
      Math.abs(h * delB) <= EPS * Math.abs(dhb) + TINY
    )
  }
  for (let m = 1; m < MAXIT; m++) {
    const m2 = 2 * m
    const ev = (m * (b - m) * x) / ((qam + m2) * (a + m2))
    const evenDone = step(ev, -ev * (1 / (qam + m2) + 1 / (a + m2)), (m * x) / ((qam + m2) * (a + m2)))
    const od = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2))
    const onePlusOd =
      nearOne && 1 + m2 - b > 0 && 2 + 3 * m - b > 0
        ? (a * (1 + m2 - b) + m * (2 + 3 * m - b) + (a + m) * (qab + m) * y) / ((a + m2) * (qap + m2))
        : 1 + od
    const oddDone = step(
      od,
      od * (1 / (a + m) + 1 / (qab + m) - 1 / (a + m2) - 1 / (qap + m2)),
      od / (qab + m),
      onePlusOd,
    )
    if (evenDone && oddDone && m > 1) break
  }
  return { v: h, da: dha, db: dhb }
}

/** I_x(a, b) through the fraction for the side where it converges, with d/da and d/db. */
function betaDual(a: number, b: number, x: number, y: number = 1 - x): Dual {
  if (x < (a + 1) / (a + b + 2)) {
    const f = betaFraction(a, b, x, y)
    const front = Math.exp(betaPrefix(a, b, x, y)) / a
    const v = front * f.v
    return {
      v,
      da: v * (Math.log(x) + digammaDiff(a, b) - 1 / a) + front * f.da,
      db: v * (Math.log1p(-x) + digammaDiff(b, a)) + front * f.db,
    }
  }
  // I_x(a, b) = 1 - I_{1-x}(b, a).
  const f = betaFraction(b, a, y, x)
  const front = Math.exp(betaPrefix(a, b, x, y)) / b
  const w = front * f.v
  // f.da is d/d(first parameter of the swapped fraction) = d/db here, and f.db = d/da.
  return {
    v: 1 - w,
    da: -(w * (Math.log(x) + digammaDiff(a, b)) + front * f.db),
    db: -(w * (Math.log1p(-x) + digammaDiff(b, a) - 1 / b) + front * f.da),
  }
}

function invalidBeta(a: number, b: number, x: number): boolean {
  return Number.isNaN(a) || Number.isNaN(b) || Number.isNaN(x) || a <= 0 || b <= 0 || x < 0 || x > 1
}

/** The regularised incomplete beta function I_x(a, b). */
export function betainc(a: number, b: number, x: number): number {
  if (invalidBeta(a, b, x)) return Number.NaN
  if (x === 0 || x === 1) return x
  return betaDual(a, b, x).v
}

/** I_x(a, b) for a caller that knows both x and y = 1 - x more accurately than the subtraction gives (x near 1). */
function betaincXY(a: number, b: number, x: number, y: number): number {
  if (x === 0 || y === 0) return x === 0 ? 0 : 1
  return betaDual(a, b, x, y).v
}

/** dI_x(a, b)/da. */
export function betaincDa(a: number, b: number, x: number): number {
  if (invalidBeta(a, b, x)) return Number.NaN
  if (x === 0 || x === 1) return 0
  return betaDual(a, b, x).da
}

/** dI_x(a, b)/db. */
export function betaincDb(a: number, b: number, x: number): number {
  if (invalidBeta(a, b, x)) return Number.NaN
  if (x === 0 || x === 1) return 0
  return betaDual(a, b, x).db
}

/** dI_x(a, b)/dx, the beta density. */
export function betaincDx(a: number, b: number, x: number): number {
  if (invalidBeta(a, b, x)) return Number.NaN
  if (x === 0) return a === 1 ? b : a < 1 ? Infinity : 0
  if (x === 1) return b === 1 ? a : b < 1 ? Infinity : 0
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log1p(-x) - lbeta(a, b))
}

/** The x with I_x(a, b) = p (Numerical Recipes starting point and Halley iteration). */
export function betaincinv(a: number, b: number, p: number): number {
  if (!(a > 0 && b > 0) || Number.isNaN(p) || p < 0 || p > 1) return Number.NaN
  if (p === 0 || p === 1) return p
  // Solve with whichever of p and 1 - p is smaller (the given probability keeps its relative precision); when that
  // puts the answer below 1/2 after all, solve again directly so that it is not formed as 1 - (a number near 1).
  if (p <= 0.5) return betaInverseLower(a, b, p)
  const x = 1 - betaInverseLower(b, a, 1 - p)
  return x < 0.5 ? betaInverseLower(a, b, p) : x
}

function betaInverseLower(a: number, b: number, p: number): number {
  const a1 = a - 1
  const b1 = b - 1
  let x: number
  if (a >= 1 && b >= 1) {
    const t = Math.sqrt(-2 * Math.log(p))
    const z = -((2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)) - t)
    const al = (z * z - 3) / 6
    const h = 2 / (1 / (2 * a - 1) + 1 / (2 * b - 1))
    const w = (z * Math.sqrt(al + h)) / h - (1 / (2 * b - 1) - 1 / (2 * a - 1)) * (al + 5 / 6 - 2 / (3 * h))
    x = a / (a + b * Math.exp(2 * w))
  } else {
    const lna = Math.log(a / (a + b))
    const lnb = Math.log(b / (a + b))
    const t = Math.exp(a * lna) / a
    const u = Math.exp(b * lnb) / b
    const w = t + u
    x = p < t / w ? (a * w * p) ** (1 / a) : 1 - (b * w * (1 - p)) ** (1 / b)
  }
  const afac = -lbeta(a, b)
  for (let j = 0; j < 200; j++) {
    if (!(x > 0)) x = Number.MIN_VALUE
    if (x >= 1) x = 1 - EPS
    const err = betainc(a, b, x) - p
    const density = Math.exp(a1 * Math.log(x) + b1 * Math.log1p(-x) + afac)
    if (density === 0 || !Number.isFinite(density)) break
    const u = err / density
    const dx = u / (1 - 0.5 * Math.min(1, u * (a1 / x - b1 / (1 - x))))
    let next = x - dx
    if (next <= 0) next = x / 2
    if (next >= 1) next = (x + 1) / 2
    x = next
    if (Math.abs(dx) <= 4 * EPS * x) break
  }
  return x
}

/** Inverse of the Student t CDF with `df` degrees of freedom. */
export function stdtrit(df: number, p: number): number {
  if (!(df > 0) || Number.isNaN(p) || p < 0 || p > 1) return Number.NaN
  if (p === 0) return -Infinity
  if (p === 1) return Infinity
  if (p === 0.5) return 0
  const sign = p < 0.5 ? -1 : 1
  if (Math.abs(p - 0.5) < 0.25) {
    // Near the centre: t^2 / (df + t^2) = z with I_z(1/2, df/2) = |2p - 1|.
    const z = betaincinv(0.5, df / 2, Math.abs(2 * p - 1))
    return sign * Math.sqrt((df * z) / (1 - z))
  }
  const x = betaincinv(df / 2, 0.5, 2 * Math.min(p, 1 - p))
  return sign * Math.sqrt((df * (1 - x)) / x)
}

/** The Student t CDF, from whichever incomplete beta form avoids cancellation. */
export function stdtr(df: number, t: number): number {
  if (!(df > 0) || Number.isNaN(t)) return Number.NaN
  if (t === Infinity) return 1
  if (t === -Infinity) return 0
  const t2 = t * t
  // tail = P(T < -|t|) = I_x(df / 2, 1 / 2) / 2 with x = df / (df + t^2): accurate when small.
  const tail = 0.5 * betaincXY(df / 2, 0.5, df / (df + t2), t2 / (df + t2))
  if (tail < 0.25) return t > 0 ? 1 - tail : tail
  // Near the centre: 1/2 +- I_z(1/2, df / 2) / 2 with z = t^2 / (df + t^2), formed directly (not as 1 - x).
  const centre = 0.5 * betaincXY(0.5, df / 2, t2 / (df + t2), df / (df + t2))
  return t > 0 ? 0.5 + centre : 0.5 - centre
}
