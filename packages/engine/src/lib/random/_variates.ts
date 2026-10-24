// Scalar samplers over a uniform stream (truncated normal, gamma, Poisson, binomial), shared by continuous.ts and
// discrete.ts. Private to random/.
import { logGammaSmall, normalFrom } from './_unit.ts'

// One standard normal variate on [a, b] with a >= 0 (Robert 1995): a uniform proposal when the interval is short,
// otherwise a shifted exponential proposal with the optimal rate, rejecting draws beyond b.
function truncatedTail(a: number, b: number, next: () => number): number {
  const root = Math.sqrt(a * a + 4)
  const short = ((2 * Math.sqrt(Math.E)) / (a + root)) * Math.exp((a * a - a * root) / 4)
  if (b - a < short) {
    for (;;) {
      const z = a + (b - a) * next()
      if (next() <= Math.exp((a * a - z * z) / 2)) return z
    }
  }
  const alpha = (a + root) / 2
  for (;;) {
    const z = a - Math.log(1 - next()) / alpha
    if (z <= b && next() <= Math.exp(-((z - alpha) ** 2) / 2)) return z
  }
}

/** One standard normal variate on [a, b]. */
export function truncatedOne(a: number, b: number, next: () => number): number {
  if (a >= 0) return truncatedTail(a, b, next)
  if (b <= 0) return -truncatedTail(-b, -a, next)
  // The interval straddles zero: normal rejection when it is wide (it then holds at least half the mass of one side),
  // a uniform proposal with acceptance exp(-z^2 / 2) when it is narrow.
  if (b - a >= Math.sqrt(2 * Math.PI)) {
    for (;;) {
      const z = normalFrom(next)
      if (z >= a && z <= b) return z
    }
  }
  for (;;) {
    const z = a + (b - a) * next()
    if (next() <= Math.exp((-z * z) / 2)) return z
  }
}

/** One gamma variate with shape alpha and scale 1 (Marsaglia and Tsang). */
export function gammaOne(alpha: number, next: () => number): number {
  // Marsaglia and Tsang (2000); for alpha < 1 boost with U^(1/alpha), done in log space to avoid underflow.
  if (alpha < 1) return Math.exp(Math.log(gammaOne(alpha + 1, next)) + Math.log(1 - next()) / alpha)
  const d = alpha - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  for (;;) {
    let x: number
    let v: number
    do {
      x = normalFrom(next)
      v = 1 + c * x
    } while (v <= 0)
    v = v * v * v
    const u = next()
    if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v
  }
}

/** The logarithm of one gamma variate; for alpha < 1 the boost U^(1/alpha) is applied in log space, so tiny shapes
 * do not underflow to zero before a ratio is formed. */
export function logGammaOne(alpha: number, next: () => number): number {
  if (alpha >= 1) return Math.log(gammaOne(alpha, next))
  return Math.log(gammaOne(alpha + 1, next)) + Math.log(1 - next()) / alpha
}

/** One Poisson variate: Knuth's product of uniforms below rate 30, Hormann's PTRS (transformed rejection) above. */
export function poissonOne(lambda: number, next: () => number): number {
  if (lambda === 0) return 0
  if (lambda < 30) {
    const limit = Math.exp(-lambda)
    let p = 1
    let count = -1
    do {
      count += 1
      p *= next()
    } while (p > limit)
    return count
  }
  const slam = Math.sqrt(lambda)
  const loglam = Math.log(lambda)
  const b = 0.931 + 2.53 * slam
  const a = -0.059 + 0.02483 * b
  const invalpha = 1.1239 + 1.1328 / (b - 3.4)
  const vr = 0.9277 - 3.6224 / (b - 2)
  for (;;) {
    const u = next() - 0.5
    const v = next()
    const us = 0.5 - Math.abs(u)
    const kk = Math.floor(((2 * a) / us + b) * u + lambda + 0.43)
    if (us >= 0.07 && v <= vr) return kk
    if (kk < 0 || (us < 0.013 && v > us)) continue
    if (
      Math.log(v) + Math.log(invalpha) - Math.log(a / (us * us) + b) <=
      -lambda + kk * loglam - logGammaSmall(kk + 1)
    ) {
      return kk
    }
  }
}

/** Binomial by geometric waiting times between successes, for `n min(p, 1 - p) < 30` (expected cost below 31). */
function binomialWaiting(n: number, q: number, next: () => number): number {
  if (q === 0) return 0
  const logq = Math.log(1 - q)
  let count = 0
  let position = 0
  for (;;) {
    position += Math.floor(Math.log(1 - next()) / logq) + 1
    if (position > n) return count
    count += 1
  }
}

// Stirling-series correction terms of the BTPE acceptance test.
function stirlingTail(x: number): number {
  const x2 = x * x
  return (13680 - (462 - (132 - (99 - 140 / x2) / x2) / x2) / x2) / x / 166320
}

/**
 * Kachitvichyanukul and Schmeiser's BTPE (triangle, parallelogram and exponential tails, with a squeeze) for
 * `n r >= 30`, `r = min(p, 1 - p)`; a step-for-step port of numpy's `random_binomial_btpe`.
 */
function binomialBtpe(n: number, r: number, next: () => number): number {
  const q = 1 - r
  const fm = n * r + r
  const m = Math.floor(fm)
  const p1 = Math.floor(2.195 * Math.sqrt(n * r * q) - 4.6 * q) + 0.5
  const xm = m + 0.5
  const xl = xm - p1
  const xr = xm + p1
  const c = 0.134 + 20.5 / (15.3 + m)
  let a = (fm - xl) / (fm - xl * r)
  const laml = a * (1 + a / 2)
  a = (xr - fm) / (xr * q)
  const lamr = a * (1 + a / 2)
  const p2 = p1 * (1 + 2 * c)
  const p3 = p2 + c / laml
  const p4 = p3 + c / lamr
  const nrq = n * r * q
  for (;;) {
    const u = next() * p4
    let v = next()
    let y: number
    if (u <= p1) return Math.floor(xm - p1 * v + u) // the triangle: accepted outright
    if (u <= p2) {
      const x = xl + (u - p1) / c
      v = v * c + 1 - Math.abs(m - x + 0.5) / p1
      if (v > 1) continue
      y = Math.floor(x)
    } else if (u <= p3) {
      y = Math.floor(xl + Math.log(v) / laml)
      if (y < 0 || v === 0) continue
      v = v * (u - p2) * laml
    } else {
      y = Math.floor(xr - Math.log(v) / lamr)
      if (y > n || v === 0) continue
      v = v * (u - p3) * lamr
    }
    const k = Math.abs(y - m)
    if (k <= 20 || k >= nrq / 2 - 1) {
      // Explicit ratio of probabilities f(y) / f(m) by recursion.
      const s = r / q
      const aa = s * (n + 1)
      let f = 1
      if (m < y) for (let i = m + 1; i <= y; i++) f *= aa / i - s
      else if (m > y) for (let i = y + 1; i <= m; i++) f /= aa / i - s
      if (v <= f) return y
      continue
    }
    // Squeeze on log v, then the exact test with Stirling's formula.
    const rho = (k / nrq) * ((k * (k / 3 + 0.625) + 0.16666666666666666) / nrq + 0.5)
    const t = (-k * k) / (2 * nrq)
    const logV = Math.log(v)
    if (logV < t - rho) return y
    if (logV > t + rho) continue
    const x1 = y + 1
    const f1 = m + 1
    const z = n + 1 - m
    const w = n - y + 1
    const bound =
      xm * Math.log(f1 / x1) +
      (n - m + 0.5) * Math.log(z / w) +
      (y - m) * Math.log((w * r) / (x1 * q)) +
      stirlingTail(f1) +
      stirlingTail(z) +
      stirlingTail(x1) +
      stirlingTail(w)
    if (logV <= bound) return y
  }
}

/** One binomial variate, exact for any `n`: waiting times when `n min(p, 1 - p) < 30`, BTPE otherwise. */
export function binomialOne(n: number, p: number, next: () => number): number {
  const r = Math.min(p, 1 - p)
  const y = n * r < 30 ? binomialWaiting(n, r, next) : binomialBtpe(n, r, next)
  return p <= 0.5 ? y : n - y
}
