// Scalar kernels for the special functions. Each is a standard published method; accuracy targets are about 1e-14
// relative unless noted, and every kernel is checked against scipy fixtures.

export const SQRT_PI = 1.7724538509055159
export const SQRT_2PI = 2.5066282746310002
export const LOG_SQRT_2PI = 0.9189385332046728
export const EPS = 2.220446049250313e-16
export const TINY = 1e-300

// ---------------------------------------------------------------------------------------------------------------------
// Error function family

/** erf by the everywhere-positive series 2/sqrt(pi) e^{-x^2} sum 2^n x^{2n+1} / (2n+1)!! (no cancellation). */
function erfSeries(x: number): number {
  const x2 = x * x
  let term = x
  let total = x
  for (let n = 1; n < 500; n++) {
    term *= (2 * x2) / (2 * n + 1)
    total += term
    if (term < EPS * total) break
  }
  return (2 / SQRT_PI) * Math.exp(-x2) * total
}

/**
 * The Laplace continued fraction K(x) = x + (1/2) / (x + 1 / (x + (3/2) / (x + ...))) with erfc(x) = e^{-x^2} /
 * (sqrt(pi) K(x)), evaluated by modified Lentz. `start` skips the first terms: the fraction from the term with
 * numerator start / 2 onwards. Converges for x > 0, quickly for x above about 1.5.
 */
function erfcFraction(x: number, start = 1): number {
  let f = x
  let c = f
  let d = 0
  for (let k = start; k < 5000 + start; k++) {
    const a = k / 2
    d = x + a * d
    if (Math.abs(d) < TINY) d = TINY
    c = x + a / c
    if (Math.abs(c) < TINY) c = TINY
    d = 1 / d
    const delta = c * d
    f *= delta
    if (Math.abs(delta - 1) < EPS) break
  }
  return f
}

/**
 * exp(-x^2 / 2^e) with x^2 formed exactly: x = m + f with m a multiple of 1/128, so m^2 is exact and only the small
 * part 2 m f + f^2 is rounded (Cephes expx2). Rounding x^2 directly would cost a relative error of x^2 eps.
 */
export function expMinusSquare(x: number, halve = false): number {
  const ax = Math.abs(x)
  const m = Math.floor(128 * ax + 0.5) / 128
  const f = ax - m
  const u = m * m
  const u1 = 2 * m * f + f * f
  return halve ? Math.exp(-u / 2) * Math.exp(-u1 / 2) : Math.exp(-u) * Math.exp(-u1)
}

const CF_THRESHOLD = 1.5

export function erf(x: number): number {
  if (Number.isNaN(x)) return Number.NaN
  const a = Math.abs(x)
  const v = a < CF_THRESHOLD ? erfSeries(a) : 1 - expMinusSquare(a) / (SQRT_PI * erfcFraction(a))
  return x < 0 ? -v : v
}

// erfc switches to the continued fraction earlier than erf, so 1 - erf(x) never cancels.
const ERFC_THRESHOLD = 0.5

export function erfc(x: number): number {
  if (Number.isNaN(x)) return Number.NaN
  if (x === Number.POSITIVE_INFINITY) return 0
  if (x === Number.NEGATIVE_INFINITY) return 2
  if (x >= ERFC_THRESHOLD) return expMinusSquare(x) / (SQRT_PI * erfcFraction(x))
  if (x <= -CF_THRESHOLD) return 2 - expMinusSquare(x) / (SQRT_PI * erfcFraction(-x))
  return 1 - (x < 0 ? -erfSeries(-x) : erfSeries(x))
}

/** Scaled complementary error function e^{x^2} erfc(x), stable for large positive x. */
export function erfcx(x: number): number {
  if (Number.isNaN(x)) return Number.NaN
  if (x >= ERFC_THRESHOLD) return 1 / (SQRT_PI * erfcFraction(x))
  if (x < -26.7) return Infinity
  return Math.exp(x * x) * erfc(x)
}

/** d/dx erfcx(x) = 2 x erfcx(x) - 2 / sqrt(pi), without the cancellation of that formula for large x. */
export function erfcxPrime(x: number): number {
  if (Number.isNaN(x)) return Number.NaN
  if (x >= ERFC_THRESHOLD) {
    // With K = x + (1/2) / K2: 2 x / (sqrt(pi) K) - 2 / sqrt(pi) = -(2 / sqrt(pi)) (1/2) / (K2 K).
    const k2 = erfcFraction(x, 2)
    const k = x + 0.5 / k2
    return -(1 / SQRT_PI) / (k2 * k)
  }
  return 2 * x * erfcx(x) - 2 / SQRT_PI
}

export function ndtr(x: number): number {
  if (x === Number.NEGATIVE_INFINITY) return 0
  // In the lower tail, exp(-x^2 / 2) is formed from x itself (not from x / sqrt 2, whose rounding the exponential
  // would amplify by x^2); the continued fraction is insensitive to that rounding.
  if (x <= -ERFC_THRESHOLD * Math.SQRT2)
    return (0.5 * expMinusSquare(x, true)) / (SQRT_PI * erfcFraction(-x / Math.SQRT2))
  return 0.5 * erfc(-x / Math.SQRT2)
}

/** log of the normal CDF, accurate in the far lower tail through erfcx. */
export function logNdtr(x: number): number {
  if (x > 0) return Math.log1p(-ndtr(-x))
  if (x > -5) return Math.log(ndtr(x))
  if (x === Number.NEGATIVE_INFINITY) return Number.NEGATIVE_INFINITY
  const z = -x / Math.SQRT2
  return Math.log(0.5) + Math.log(erfcx(z)) - z * z
}

// Acklam's rational approximation of the normal quantile, refined by one Halley step with the accurate erfc.
const QA = [
  -39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239,
] as const
const QB = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572] as const
const QC = [
  -0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968,
  2.938163982698783,
] as const
const QD = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416] as const

function acklamTail(q: number): number {
  const [c0, c1, c2, c3, c4, c5] = QC
  const [d0, d1, d2, d3] = QD
  return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1)
}

export function ndtri(p: number): number {
  if (Number.isNaN(p) || p < 0 || p > 1) return Number.NaN
  if (p === 0) return -Infinity
  if (p === 1) return Infinity
  let x: number
  if (p < 0.02425) x = acklamTail(Math.sqrt(-2 * Math.log(p)))
  else if (p > 1 - 0.02425) x = -acklamTail(Math.sqrt(-2 * Math.log(1 - p)))
  else {
    const [a0, a1, a2, a3, a4, a5] = QA
    const [b0, b1, b2, b3, b4] = QB
    const q = p - 0.5
    const r = q * q
    x =
      ((((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q) /
      (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1)
  }
  // Halley steps on the residual in the tail that is computed accurately.
  for (let i = 0; i < 2; i++) {
    const e = p < 0.5 ? ndtr(x) - p : -(ndtr(-x) - (1 - p))
    const u = e * SQRT_2PI * Math.exp((x * x) / 2)
    x -= u / (1 + (x * u) / 2)
  }
  return x
}

export function erfinv(y: number): number {
  if (y <= -1) return y === -1 ? -Infinity : Number.NaN
  if (y >= 1) return y === 1 ? Infinity : Number.NaN
  // (1 - y) / 2 is exact for y > 1/2, so the tail keeps its precision.
  if (y > 0.5) return -ndtri((1 - y) / 2) / Math.SQRT2
  if (y < -0.5) return ndtri((1 + y) / 2) / Math.SQRT2
  return ndtri((y + 1) / 2) / Math.SQRT2
}

export function erfcinv(y: number): number {
  if (y <= 0) return y === 0 ? Infinity : Number.NaN
  if (y >= 2) return y === 2 ? -Infinity : Number.NaN
  return -ndtri(y / 2) / Math.SQRT2
}

// ---------------------------------------------------------------------------------------------------------------------
// Gamma family

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
] as const

function lanczosSum(z: number): number {
  let a = LANCZOS[0]
  for (let i = 1; i < 9; i++) a += (LANCZOS[i] ?? 0) / (z + i)
  return a
}

/** sin(pi x) without the rounding of pi x (exact near the integers). */
export function sinpi(x: number): number {
  const r = x - 2 * Math.round(x / 2)
  if (r === 0 || Math.abs(r) === 1) return 0
  return Math.sin(Math.PI * r)
}

/** cos(pi x) without the rounding of pi x. */
export function cospi(x: number): number {
  const r = x - 2 * Math.round(x / 2)
  if (Math.abs(r) === 0.5) return 0
  return Math.cos(Math.PI * r)
}

// zeta(k) for k = 2, 3, ..., for the Taylor series of lgamma about 1 (computed on first use).
let zetaValues: number[] | undefined
function zetaInt(k: number): number {
  zetaValues ??= Array.from({ length: 64 }, (_, i) => (i < 2 ? Number.NaN : hurwitzZeta(i, 1)))
  return zetaValues[k] ?? 1
}

/** log Gamma(1 + z) for |z| <= 0.25 by its Taylor series (keeps relative accuracy at the zeros 1 and 2). */
function lgamma1p(z: number): number {
  let sum = -0.5772156649015329 * z
  let pow = -z
  for (let k = 2; k < 64; k++) {
    pow *= -z
    const term = (zetaInt(k) * pow) / k
    sum += term
    if (Math.abs(term) <= EPS * Math.abs(sum)) break
  }
  return sum
}

/** log |Gamma(x)| (Lanczos, g = 7, Taylor series near 1 and 2, reflection below 0.5). */
export function lgamma(x: number): number {
  if (Number.isNaN(x)) return Number.NaN
  if (x <= 0 && Number.isInteger(x)) return Infinity
  if (x < 0.5) return Math.log(Math.PI / Math.abs(sinpi(x))) - lgamma(1 - x)
  if (x === 1 || x === 2) return 0
  if (Math.abs(x - 1) <= 0.25) return lgamma1p(x - 1)
  if (Math.abs(x - 2) <= 0.25) return lgamma1p(x - 2) + Math.log1p(x - 2)
  if (x >= 10) return stirling(x)
  const z = x - 1
  const t = z + 7.5
  return LOG_SQRT_2PI + (z + 0.5) * Math.log(t) - t + Math.log(lanczosSum(z))
}

const FACTORIALS: number[] = [1]
for (let i = 1; i <= 170; i++) FACTORIALS.push((FACTORIALS[i - 1] ?? 1) * i)

export function factorial(n: number): number {
  if (!Number.isInteger(n) || n < 0) return Number.NaN
  return n > 170 ? Infinity : (FACTORIALS[n] ?? Infinity)
}

export function gamma(x: number): number {
  if (Number.isNaN(x)) return Number.NaN
  if (Number.isInteger(x)) {
    if (x <= 0) return Number.NaN
    return x <= 171 ? factorial(x - 1) : Infinity
  }
  if (x < 0.5) return Math.PI / (sinpi(x) * gamma(1 - x))
  if (x > 171.62) return Infinity
  const z = x - 1
  const t = z + 7.5
  // Split the power so t^(z + 1/2) never overflows before the exponential brings it back.
  const half = t ** ((z + 0.5) / 2)
  return SQRT_2PI * half * (Math.exp(-t) * half) * lanczosSum(z)
}

/** Reciprocal gamma 1 / Gamma(x), zero at the poles. */
export function rgamma(x: number): number {
  if (x <= 0 && Number.isInteger(x)) return 0
  return 1 / gamma(x)
}

// Bernoulli numbers B_2, B_4, ..., B_24.
const BERNOULLI = [
  1 / 6,
  -1 / 30,
  1 / 42,
  -1 / 30,
  5 / 66,
  -691 / 2730,
  7 / 6,
  -3617 / 510,
  43867 / 798,
  -174611 / 330,
  854513 / 138,
  -236364091 / 2730,
] as const

export function digamma(x: number): number {
  if (Number.isNaN(x)) return Number.NaN
  if (x <= 0 && Number.isInteger(x)) return Number.NaN
  if (x < 0) return digamma(1 - x) - (Math.PI * cospi(x)) / sinpi(x)
  let acc = 0
  let y = x
  while (y < 10) {
    acc -= 1 / y
    y += 1
  }
  // Asymptotic series psi(y) ~ ln y - 1/(2y) - sum B_2k / (2k y^2k).
  const inv2 = 1 / (y * y)
  let term = inv2
  let series = 0
  for (let k = 1; k <= 8; k++) {
    series += ((BERNOULLI[k - 1] ?? 0) / (2 * k)) * term
    term *= inv2
  }
  return acc + Math.log(y) - 0.5 / y - series
}

// (2k)! / B_2k for the Euler-Maclaurin tail of the Hurwitz zeta function (Cephes zeta.c).
const ZETA_A: number[] = BERNOULLI.map((b, k) => factorial(2 * k + 2) / b)

/** Hurwitz zeta function zeta(s, q) = sum_{k >= 0} (k + q)^{-s}, for s > 1 and q > 0 (Euler-Maclaurin). */
export function hurwitzZeta(s: number, q: number): number {
  if (s === 1) return Infinity
  if (s < 1 || q <= 0) return Number.NaN
  let sum = q ** -s
  let a = q
  let b = 0
  let i = 0
  while (i < 9 || a <= 9) {
    i += 1
    a += 1
    b = a ** -s
    sum += b
    if (Math.abs(b / sum) < EPS) return sum
  }
  const w = a
  sum += (b * w) / (s - 1) - 0.5 * b
  let fac = 1
  let k = 0
  for (let j = 0; j < 12; j++) {
    fac *= s + k
    b /= w
    const t = (fac * b) / (ZETA_A[j] ?? Infinity)
    sum += t
    if (Math.abs(t / sum) < EPS) break
    k += 1
    fac *= s + k
    b /= w
    k += 1
  }
  return sum
}

/** d/ds of the Hurwitz zeta function, by differentiating its Euler-Maclaurin evaluation term by term. */
export function hurwitzZetaDs(s: number, q: number): number {
  if (!(s > 1) || !(q > 0)) return Number.NaN
  let sum = q ** -s
  let dsum = -Math.log(q) * sum
  let a = q
  let b = 0
  let db = 0
  let i = 0
  while (i < 9 || a <= 9) {
    i += 1
    a += 1
    b = a ** -s
    db = -Math.log(a) * b
    sum += b
    dsum += db
    if (Math.abs(b / sum) < EPS && Math.abs(db) < EPS * Math.abs(dsum)) return dsum
  }
  const w = a
  dsum += (db * w) / (s - 1) - (b * w) / ((s - 1) * (s - 1)) - 0.5 * db
  sum += (b * w) / (s - 1) - 0.5 * b
  let fac = 1
  let dfac = 0
  let k = 0
  for (let j = 0; j < 12; j++) {
    dfac = dfac * (s + k) + fac
    fac *= s + k
    b /= w
    db /= w
    const t = (fac * b) / (ZETA_A[j] ?? Infinity)
    const dt = (dfac * b + fac * db) / (ZETA_A[j] ?? Infinity)
    sum += t
    dsum += dt
    if (Math.abs(t / sum) < EPS && Math.abs(dt) < EPS * Math.abs(dsum)) break
    k += 1
    dfac = dfac * (s + k) + fac
    fac *= s + k
    b /= w
    db /= w
    k += 1
  }
  return dsum
}

export function polygamma(n: number, x: number): number {
  if (!Number.isInteger(n) || n < 0) return Number.NaN
  if (n === 0) return digamma(x)
  if (x <= 0 && Number.isInteger(x)) return Number.NaN
  // psi^(n)(x) = psi^(n)(x + 1) - (-1)^n n! / x^(n+1), applied until the argument is positive.
  let shift = 0
  let y = x
  while (y < 1) {
    shift -= ((n % 2 === 0 ? 1 : -1) * factorial(n)) / y ** (n + 1)
    y += 1
  }
  if (y !== x) return polygamma(n, y) + shift
  return (n % 2 === 1 ? 1 : -1) * factorial(n) * hurwitzZeta(n + 1, x)
}

// Stirling series: lgamma(x) = (x - 1/2) ln x - x + ln sqrt(2 pi) + correction(x), for x >= 10.
function stirlingCorrection(x: number): number {
  const inv2 = 1 / (x * x)
  let pow = 1 / x
  let sum = 0
  for (let k = 1; k <= 8; k++) {
    sum += ((BERNOULLI[k - 1] ?? 0) / (2 * k * (2 * k - 1))) * pow
    pow *= inv2
  }
  return sum
}

function stirling(x: number): number {
  return (x - 0.5) * Math.log(x) - x + LOG_SQRT_2PI + stirlingCorrection(x)
}

/** log |B(a, b)|, without the cancellation of three large log-gammas when an argument is large. */
/** psi(x + h) - psi(x), without cancellation when h is small relative to x. */
export function digammaDiff(x: number, h: number): number {
  if (Number.isNaN(x) || Number.isNaN(h)) return Number.NaN
  if (h === 0) return 0
  if (!(x > 0 && x + h > 0)) return digamma(x + h) - digamma(x)
  // Shift both arguments to at least 10: psi(y + n) = psi(y) + sum_{j<n} 1 / (y + j).
  let shift = 0
  let y = x
  while (Math.min(y, y + h) < 10) {
    shift += h / ((y + h) * y)
    y += 1
  }
  // Asymptotic series, differenced term by term: ln, 1/(2y) and y^{-2k} differences without cancellation.
  const r = Math.log1p(h / y)
  let series = r + h / (2 * y * (y + h))
  let pow = 1 / (y * y)
  for (let k = 1; k <= 8; k++) {
    series -= ((BERNOULLI[k - 1] ?? 0) / (2 * k)) * pow * Math.expm1(-2 * k * r)
    pow /= y * y
  }
  return series + shift
}

export function lbeta(a: number, b: number): number {
  if (a <= 0 || b <= 0) return lgamma(a) + lgamma(b) - lgamma(a + b)
  const lo = Math.min(a, b)
  const hi = Math.max(a, b)
  if (lo >= 10) {
    const corr = stirlingCorrection(lo) + stirlingCorrection(hi) - stirlingCorrection(lo + hi)
    return LOG_SQRT_2PI - 0.5 * Math.log(hi) + corr + (lo - 0.5) * Math.log(lo / (lo + hi)) - hi * Math.log1p(lo / hi)
  }
  if (hi >= 10) {
    // lgamma(hi) - lgamma(hi + lo) by the Stirling series.
    const corr = stirlingCorrection(hi) - stirlingCorrection(hi + lo)
    return lgamma(lo) + corr - (hi - 0.5) * Math.log1p(lo / hi) - lo * Math.log(hi + lo) + lo
  }
  return lgamma(a) + lgamma(b) - lgamma(a + b)
}

export function beta(a: number, b: number): number {
  if (a > 0 && b > 0 && a + b < 171) return (gamma(a) * gamma(b)) / gamma(a + b)
  return Math.exp(lbeta(a, b))
}

export function lbinom(n: number, k: number): number {
  // C(n, k) = 1 / ((n + 1) B(n - k + 1, k + 1)), with the accurate lbeta.
  if (n > -1 && k > -1 && n - k > -1) return -Math.log1p(n) - lbeta(n - k + 1, k + 1)
  return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
}

export function binom(n: number, k: number): number {
  if (Number.isInteger(n) && Number.isInteger(k) && n >= 0) {
    if (k < 0 || k > n) return 0
    const m = Math.min(k, n - k)
    if (m < 60 && n < 1e6) {
      let r = 1
      for (let i = 1; i <= m; i++) r = (r * (n - m + i)) / i
      return Math.round(r)
    }
  }
  return 1 / ((n + 1) * beta(n - k + 1, k + 1))
}

// ---------------------------------------------------------------------------------------------------------------------
// Zeta, exponential integrals, Lambert W

export function zeta(s: number): number {
  return hurwitzZeta(s, 1)
}

/** Lambert W on the principal branch (W >= -1), by Halley iteration. */
export function lambertW(x: number): number {
  if (x < -1 / Math.E) return Number.NaN
  if (x === -1 / Math.E) return -1
  if (x === 0) return 0
  let w = x < 1 ? (x < -0.25 ? -1 + Math.sqrt(2 * (1 + Math.E * x)) : x) : Math.log(x) - Math.log(Math.log(x) + 1)
  for (let i = 0; i < 100; i++) {
    const ew = Math.exp(w)
    const f = w * ew - x
    const step = f / (ew * (w + 1) - ((w + 2) * f) / (2 * w + 2))
    w -= step
    if (Math.abs(step) <= EPS * (1 + Math.abs(w))) break
  }
  return w
}

/** Exponential integral E_n(x) (Numerical Recipes expint). */
export function expn(n: number, x: number): number {
  if (n < 0 || x < 0 || (x === 0 && n <= 1)) return n === 1 && x === 0 ? Infinity : Number.NaN
  if (n === 0) return Math.exp(-x) / x
  if (x === 0) return 1 / (n - 1)
  const nm1 = n - 1
  if (x > 1) {
    let b = x + n
    let c = 1 / TINY
    let d = 1 / b
    let h = d
    for (let i = 1; i < 100000; i++) {
      const a = -i * (nm1 + i)
      b += 2
      d = 1 / (a * d + b)
      c = b + a / c
      const del = c * d
      h *= del
      if (Math.abs(del - 1) < EPS) break
    }
    return h * Math.exp(-x)
  }
  let ans = nm1 !== 0 ? 1 / nm1 : -Math.log(x) - 0.5772156649015329
  let fact = 1
  for (let i = 1; i < 100000; i++) {
    fact *= -x / i
    let del: number
    if (i !== nm1) del = -fact / (i - nm1)
    else {
      let psi = -0.5772156649015329
      for (let ii = 1; ii <= nm1; ii++) psi += 1 / ii
      del = fact * (-Math.log(x) + psi)
    }
    ans += del
    if (Math.abs(del) < Math.abs(ans) * EPS) break
  }
  return ans
}

/** Exponential integral Ei(x) (series for small |x|, asymptotic for large x; E_1 for negative x). */
export function expi(x: number): number {
  if (x === 0) return -Infinity
  if (x < 0) return -expn(1, -x)
  if (x < -Math.log(EPS)) {
    let sum = 0
    let fact = 1
    for (let k = 1; k < 1000; k++) {
      fact *= x / k
      const term = fact / k
      sum += term
      if (term < EPS * sum) break
    }
    return sum + Math.log(x) + 0.5772156649015329
  }
  let sum = 0
  let term = 1
  for (let k = 1; k < 1000; k++) {
    const prev = term
    term *= k / x
    if (term < EPS) break
    if (term >= prev) {
      term = prev
      break
    }
    sum += term
  }
  return (Math.exp(x) * (1 + sum)) / x
}
