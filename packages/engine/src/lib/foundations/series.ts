/**
 * Sequences and series: partial sums and sum estimates, the classical convergence tests (n-th term, ratio, root,
 * integral, p-series, limit comparison, alternating series) with the statistic behind each verdict, radii of
 * convergence, Fourier coefficients and partial sums with the Gibbs overshoot at a jump, and power series of rational
 * generating functions, Cauchy products and linear recurrences.
 */

import { kahanSum, MdmlError } from '../base/index.ts'
import { wynnEpsilon } from '../numerics/approximation/index.ts'
import { polymul } from '../numerics/polynomial/index.ts'
import { quad } from '../numerics/quadrature/index.ts'
import {
  add,
  cos,
  item,
  mul,
  promoteFloat,
  sin,
  type TensorLike,
  toFlat,
  type Value,
  zerosLike,
} from '../tensor/index.ts'
import { extrapolateToZero } from './_extrapolate.ts'

/** A sequence: its n-th term for n = 0, 1, 2, ... (or from whatever index the caller uses). */
export type Sequence = (n: number) => number

/**
 * The terms `a(start), ..., a(start + count - 1)` of a sequence.
 *
 * @param a - The sequence.
 * @param count - How many terms.
 * @param start - The first index (default 0).
 * @returns The terms.
 * @concept sequences
 * @example
 * ```ts
 * import { sequenceTerms } from 'mdml-engine/foundations'
 * sequenceTerms((n) => 1 / n, 4, 1) // => [1, 0.5, 0.3333333333333333, 0.25]
 * ```
 */
export function sequenceTerms(a: Sequence, count: number, start = 0): number[] {
  return Array.from({ length: count }, (_, i) => a(start + i))
}

/**
 * The partial sums `s_k = a_0 + ... + a_k` of a list of terms, accumulated with Kahan's compensated summation.
 *
 * @param terms - The terms.
 * @returns The running sums.
 * @concept partial-sums
 * @example
 * ```ts
 * import { partialSums } from 'mdml-engine/foundations'
 * partialSums([1, 0.5, 0.25, 0.125]) // => [1, 1.5, 1.75, 1.875]
 * ```
 */
export function partialSums(terms: readonly number[]): number[] {
  const out: number[] = []
  let s = 0
  let c = 0
  for (const t of terms) {
    const y = t - c
    const next = s + y
    c = next - s - y
    s = next
    out.push(s)
  }
  return out
}

/**
 * Estimate the sum of a convergent series from its terms. Series whose terms shrink geometrically or alternate in sign
 * are accelerated by Wynn's epsilon algorithm on the partial sums; slowly (logarithmically) convergent series of
 * one sign, whose tail behaves like a power of 1/N (such as `sum 1/n^2`), by Richardson extrapolation of the partial
 * sums at N, 2N, 4N, ... in powers of 1/N.
 *
 * @param a - The terms.
 * @param count - How many terms the first estimate uses (default 30).
 * @param start - The first index (default 0).
 * @returns The plain partial sum of `count` terms, the accelerated estimate and the method used.
 * @concept series
 * @example
 * ```ts
 * import { seriesSum } from 'mdml-engine/foundations'
 * // 1 - 1/2 + 1/3 - ... = ln 2
 * Math.abs(seriesSum((n) => (-1) ** n / (n + 1), 20).accelerated - Math.LN2) < 1e-12 // => true
 * seriesSum((n) => 1 / (n + 1) ** 2).method // => 'richardson'
 * ```
 */
export function seriesSum(
  a: Sequence,
  count = 30,
  start = 0,
): { readonly partial: number; readonly accelerated: number; readonly method: 'wynn' | 'richardson' } {
  const terms = sequenceTerms(a, count, start)
  const s = partialSums(terms)
  const partial = s[s.length - 1] ?? 0
  const last = terms[terms.length - 1] ?? 0
  const before = terms[terms.length - 2] ?? 0
  const alternating = last * before < 0
  const slow = !alternating && before !== 0 && Math.abs(last / before) > 0.9
  if (!slow) return { partial, accelerated: wynnEpsilon(s).value, method: 'wynn' }
  // Partial sums at N, 2N, ..., 64N, extrapolated to N -> infinity in powers of 1/N.
  const sums: number[] = []
  let total = 0
  let next = start
  for (let m = count; m <= count * 64; m *= 2) {
    const chunk: number[] = []
    for (; next < start + m; next++) chunk.push(a(next))
    total += kahanSum(chunk)
    sums.push(total)
  }
  return { partial, accelerated: extrapolateToZero(sums).value, method: 'richardson' }
}

/** The verdict of a convergence test. */
export type Verdict = 'converges' | 'diverges' | 'inconclusive'

/** The outcome of a convergence test, with the statistic it rests on. */
export interface ConvergenceTest {
  readonly test: string
  readonly verdict: Verdict
  /** The limit estimated by the test (ratio or root limit, tail integral, ...). */
  readonly statistic: number
  /** The sequence the statistic was read from, for figures. */
  readonly evidence: readonly number[]
}

/** Indices k0, 2 k0, 4 k0, ... up to about n: terms expanding in powers of 1/k extrapolate cleanly along them. */
function doublings(n: number): number[] {
  const out: number[] = []
  for (let k = Math.max(1, Math.floor(n / 64)); k <= n; k *= 2) out.push(k)
  return out
}

/** The limit of g(k) as k -> infinity from samples along doubling k (Richardson in h = 1/k). */
function settle(ks: readonly number[], g: (k: number) => number): { value: number; evidence: number[] } {
  const evidence = ks.map(g)
  return { value: extrapolateToZero(evidence).value, evidence }
}

/**
 * The n-th term (divergence) test: if `a_n` does not tend to zero the series diverges; if it does, the test says
 * nothing.
 *
 * @param a - The terms.
 * @param options - Indices to sample (default 1 to 2000) and the zero tolerance (1e-6).
 * @returns 'diverges' or 'inconclusive', with the estimated limit of the terms.
 * @concept convergence-tests
 * @example
 * ```ts
 * import { nthTermTest } from 'mdml-engine/foundations'
 * nthTermTest((n) => n / (n + 1)).verdict // => 'diverges'
 * nthTermTest((n) => 1 / n).verdict // => 'inconclusive'
 * ```
 */
export function nthTermTest(
  a: Sequence,
  options: { readonly n?: number; readonly tol?: number } = {},
): ConvergenceTest {
  const { value: statistic, evidence } = settle(doublings(options.n ?? 2048), (k) => Math.abs(a(k)))
  return {
    test: 'nth term',
    verdict: Math.abs(statistic) > (options.tol ?? 1e-6) ? 'diverges' : 'inconclusive',
    statistic,
    evidence,
  }
}

/**
 * The ratio test: with `L = lim |a_{n+1} / a_n|`, the series converges absolutely if L < 1 and diverges if L > 1;
 * L = 1 is inconclusive. L is estimated from the ratios at growing n, accelerated.
 *
 * @param a - The terms (nonzero for large n).
 * @param options - The largest index sampled (default 400) and the band around 1 treated as inconclusive (1e-3).
 * @returns The verdict and the estimated L.
 * @concept ratio-test
 * @example
 * ```ts
 * import { ratioTest } from 'mdml-engine/foundations'
 * ratioTest((n) => 2 ** n / (n + 1) ** 3).verdict // => 'diverges'
 * Math.round(ratioTest((n) => 1 / 3 ** n).statistic * 1e9) / 1e9 // => 0.333333333
 * ```
 */
export function ratioTest(a: Sequence, options: { readonly n?: number; readonly band?: number } = {}): ConvergenceTest {
  const { value: statistic, evidence } = settle(doublings(options.n ?? 512), (k) => Math.abs(a(k + 1) / a(k)))
  const band = options.band ?? 1e-3
  const verdict: Verdict = statistic < 1 - band ? 'converges' : statistic > 1 + band ? 'diverges' : 'inconclusive'
  return { test: 'ratio', verdict, statistic, evidence }
}

/**
 * The root test: with `L = limsup |a_n|^(1/n)`, convergence if L < 1, divergence if L > 1, inconclusive at 1. The
 * n-th roots are taken through logarithms (`exp(log|a_n| / n)`), so huge or tiny terms are fine when `logAbs` is
 * given.
 *
 * @param a - The terms, or `logAbs: true` and a function returning `log |a_n|`.
 * @param options - Largest index (default 400), inconclusive band (1e-3), and whether `a` returns `log |a_n|`.
 * @returns The verdict and the estimated L.
 * @concept root-test
 * @example
 * ```ts
 * import { rootTest } from 'mdml-engine/foundations'
 * rootTest((n) => (n / (2 * n + 1)) ** n).verdict // => 'converges'
 * ```
 */
export function rootTest(
  a: Sequence,
  options: { readonly n?: number; readonly band?: number; readonly logAbs?: boolean } = {},
): ConvergenceTest {
  // log|a_2k| - log|a_k| = k log L + O(1) terms that cancel, so (log|a_2k| - log|a_k|) / k -> log L in powers of 1/k.
  const logOf = (k: number): number => (options.logAbs === true ? a(k) : Math.log(Math.abs(a(k))))
  const { value: logL, evidence: logs } = settle(doublings(options.n ?? 512), (k) => (logOf(2 * k) - logOf(k)) / k)
  const statistic = Math.exp(logL)
  const evidence = logs.map(Math.exp)
  const band = options.band ?? 1e-3
  const verdict: Verdict = statistic < 1 - band ? 'converges' : statistic > 1 + band ? 'diverges' : 'inconclusive'
  return { test: 'root', verdict, statistic, evidence }
}

/**
 * The integral test for a positive decreasing f with `a_n = f(n)`: the series and `int_N^inf f` converge together.
 * The tail integral is computed by adaptive quadrature; a non-finite or non-converged result means divergence.
 *
 * @param f - The function, vectorised over a value of points.
 * @param from - Where the integral starts (default 1).
 * @returns The verdict and the tail integral.
 * @concept integral-test
 * @example
 * ```ts
 * import { integralTest } from 'mdml-engine/foundations'
 * import { div, mul, type Value } from 'mdml-engine/tensor'
 * integralTest((x: Value) => div(1, mul(x, x))).statistic // => 1
 * integralTest((x: Value) => div(1, x)).verdict // => 'diverges'
 * ```
 */
export function integralTest(f: (x: Value) => Value, from = 1): ConvergenceTest {
  const r = quad(f, from, Infinity, { limit: 200 })
  const statistic = item(r.value)
  const converges = r.converged && Number.isFinite(statistic) && r.error <= 1e-6 * Math.max(1, Math.abs(statistic))
  return {
    test: 'integral',
    verdict: converges ? 'converges' : 'diverges',
    statistic: converges ? statistic : Infinity,
    evidence: [],
  }
}

/**
 * The p-series test: `sum 1 / n^p` converges exactly when p > 1 (its sum is the Riemann zeta function).
 *
 * @param p - The exponent.
 * @returns The verdict.
 * @concept p-series
 * @example
 * ```ts
 * import { pSeriesTest } from 'mdml-engine/foundations'
 * pSeriesTest(2).verdict // => 'converges'
 * pSeriesTest(1).verdict // => 'diverges'
 * ```
 */
export function pSeriesTest(p: number): ConvergenceTest {
  return { test: 'p-series', verdict: p > 1 ? 'converges' : 'diverges', statistic: p, evidence: [] }
}

/**
 * The limit comparison test against a series of known behaviour: if `a_n / b_n -> c` with `0 < c < infinity`, both
 * series do the same thing.
 *
 * @param a - Positive terms under test.
 * @param b - Positive comparison terms.
 * @param bConverges - Whether the series of b converges.
 * @param n - The largest index sampled (default 2000).
 * @returns The verdict and the estimated limit c.
 * @concept comparison-test
 * @example
 * ```ts
 * import { limitComparisonTest } from 'mdml-engine/foundations'
 * limitComparisonTest((n) => 1 / (n * n + n + 1), (n) => 1 / (n * n), true).verdict // => 'converges'
 * ```
 */
export function limitComparisonTest(a: Sequence, b: Sequence, bConverges: boolean, n = 2048): ConvergenceTest {
  const { value: statistic, evidence } = settle(doublings(n), (k) => a(k) / b(k))
  const finitePositive = Number.isFinite(statistic) && statistic > 1e-9
  const verdict: Verdict = finitePositive ? (bConverges ? 'converges' : 'diverges') : 'inconclusive'
  return { test: 'limit comparison', verdict, statistic, evidence }
}

/**
 * The alternating series (Leibniz) test: if the terms alternate in sign and `|a_n|` decreases to zero, the series
 * converges, and the error of the partial sum `s_n` is at most `|a_{n+1}|`.
 *
 * @param a - The terms.
 * @param n - How many terms to check (default 200).
 * @returns The verdict, with the error bound after n terms as the statistic.
 * @concept alternating-series
 * @example
 * ```ts
 * import { alternatingSeriesTest } from 'mdml-engine/foundations'
 * alternatingSeriesTest((n) => (-1) ** n / (n + 1)).verdict // => 'converges'
 * ```
 */
export function alternatingSeriesTest(a: Sequence, n = 200): ConvergenceTest {
  const terms = sequenceTerms(a, n + 1)
  let alternates = true
  let decreasing = true
  for (let i = 0; i + 1 < n; i++) {
    const x = terms[i] ?? 0
    const y = terms[i + 1] ?? 0
    if (x * y >= 0) alternates = false
    if (Math.abs(y) > Math.abs(x)) decreasing = false
  }
  const tail = Math.abs(terms[n] ?? Number.NaN)
  const toZero = nthTermTest(a, { n: Math.max(8, n * 10) }).verdict === 'inconclusive'
  return {
    test: 'alternating series',
    verdict: alternates && decreasing && toZero ? 'converges' : 'inconclusive',
    statistic: tail,
    evidence: terms,
  }
}

/**
 * The radius of convergence of a power series `sum c_n (x - x0)^n` by the Cauchy-Hadamard formula
 * `R = 1 / limsup |c_n|^(1/n)`, with the n-th roots read from growing n (and the ratio `|c_n / c_{n+1}|` as a second
 * estimate). When the coefficients fall faster than any geometric sequence (R infinite) the estimate is large and grows
 * with `n`.
 *
 * @param c - The coefficients, or `log |c_n|` when `options.logAbs` is set.
 * @param options - Largest index (default 400) and whether c returns `log |c_n|`.
 * @returns R by the root formula and by the ratio formula.
 * @concept radius-of-convergence
 * @example
 * ```ts
 * import { radiusOfConvergence } from 'mdml-engine/foundations'
 * Math.round(radiusOfConvergence((n) => 2 ** -n * (n + 1)).ratio * 1e6) / 1e6 // => 2
 * ```
 */
export function radiusOfConvergence(
  c: Sequence,
  options: { readonly n?: number; readonly logAbs?: boolean } = {},
): { readonly root: number; readonly ratio: number } {
  const root = rootTest(c, { ...options })
  const logOf = (k: number): number => (options.logAbs === true ? c(k) : Math.log(Math.abs(c(k))))
  const ratio = Math.exp(settle(doublings(options.n ?? 512), (k) => logOf(k) - logOf(k + 1)).value)
  return { root: root.statistic === 0 ? Infinity : 1 / root.statistic, ratio }
}

/** Real Fourier coefficients of a periodic function: `f ~ a0/2 + sum a_k cos(k w x) + b_k sin(k w x)`, w = 2 pi / T. */
export interface FourierCoefficients {
  readonly period: number
  /** `a[0]` is a0 (the series uses a0 / 2); `a[k]`, `b[k]` for k >= 1 (`b[0]` is 0). */
  readonly a: number[]
  readonly b: number[]
}

/**
 * The real Fourier coefficients of a function on one period `[x0, x0 + T]`,
 * `a_k = (2/T) int f cos(k w x)`, `b_k = (2/T) int f sin(k w x)`, by adaptive quadrature (breakpoints may be given to
 * split at discontinuities).
 *
 * @param f - The function, vectorised over a value of points.
 * @param period - The period T (default 2 pi).
 * @param terms - The highest harmonic N.
 * @param options - Start of the period (default `-T/2`) and interior breakpoints.
 * @returns The coefficients up to N.
 * @concept fourier-series
 * @example
 * ```ts
 * import { fourierCoefficients } from 'mdml-engine/foundations'
 * import { sign } from 'mdml-engine/tensor'
 * fourierCoefficients(sign, 2 * Math.PI, 3, { points: [0] }).b.map((v) => Math.round(v * 1e10) / 1e10) // => [0, 1.2732395447, 0, 0.4244131816]
 * ```
 */
export function fourierCoefficients(
  f: (x: Value) => Value,
  period: number = 2 * Math.PI,
  terms = 10,
  options: { readonly start?: number; readonly points?: readonly number[] } = {},
): FourierCoefficients {
  const x0 = options.start ?? -period / 2
  const w = (2 * Math.PI) / period
  const integrate = (g: (x: Value) => Value): number =>
    item(quad(g, x0, x0 + period, { epsabs: 1e-13, epsrel: 1e-12, limit: 200, points: options.points ?? [] }).value)
  const clean = (v: number): number => (Math.abs(v) < 1e-14 ? 0 : v)
  const a: number[] = []
  const b: number[] = []
  for (let k = 0; k <= terms; k++) {
    a.push(clean((2 / period) * integrate((x) => mul(f(x), cos(mul(x, k * w))))))
    b.push(k === 0 ? 0 : clean((2 / period) * integrate((x) => mul(f(x), sin(mul(x, k * w))))))
  }
  return { period, a, b }
}

/**
 * Evaluate a Fourier partial sum `a0/2 + sum_{k=1}^{N} a_k cos(k w x) + b_k sin(k w x)` (N defaults to all the
 * coefficients). Differentiable in x.
 *
 * @param coefficients - Fourier coefficients.
 * @param x - Points.
 * @param terms - How many harmonics to include.
 * @returns The partial sum at the points.
 * @concept fourier-series
 * @example
 * ```ts
 * import { fourierPartialSum } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(fourierPartialSum({ period: 2 * Math.PI, a: [2, 0], b: [0, 1] }, [Math.PI / 2])) // => [2]
 * ```
 */
export function fourierPartialSum(coefficients: FourierCoefficients, x: TensorLike, terms?: number): Value {
  const [xv] = promoteFloat(x) as [Value]
  const n = Math.min(terms ?? Infinity, coefficients.a.length - 1)
  const w = (2 * Math.PI) / coefficients.period
  let out: Value = add(zerosLike(xv), (coefficients.a[0] ?? 0) / 2)
  for (let k = 1; k <= n; k++) {
    const ak = coefficients.a[k] ?? 0
    const bk = coefficients.b[k] ?? 0
    if (ak !== 0) out = add(out, mul(cos(mul(xv, k * w)), ak))
    if (bk !== 0) out = add(out, mul(sin(mul(xv, k * w)), bk))
  }
  return out
}

/**
 * The Fourier coefficients of the square wave `sign(sin x)` in closed form: `b_k = 4 / (pi k)` for odd k, the classic
 * example of slow (1/k) decay at a jump.
 *
 * @param terms - The highest harmonic.
 * @returns The coefficients (period 2 pi).
 * @concept gibbs-phenomenon
 * @example
 * ```ts
 * import { squareWaveCoefficients } from 'mdml-engine/foundations'
 * squareWaveCoefficients(3).b // => [0, 1.2732395447351628, 0, 0.4244131815783876]
 * ```
 */
export function squareWaveCoefficients(terms: number): FourierCoefficients {
  const a = Array.from({ length: terms + 1 }, () => 0)
  const b = Array.from({ length: terms + 1 }, (_, k) => (k % 2 === 1 ? 4 / (Math.PI * k) : 0))
  return { period: 2 * Math.PI, a, b }
}

/** The Gibbs overshoot of a Fourier partial sum next to a jump. */
export interface GibbsOvershoot {
  /** Where the first peak after the jump is. */
  readonly x: number
  /** The partial sum there. */
  readonly peak: number
  /** The overshoot past the limit on that side, as a fraction of the jump. */
  readonly fraction: number
}

/**
 * Measure the Gibbs phenomenon: the first peak of a Fourier partial sum just past a jump discontinuity, and how far it
 * overshoots. As N grows the peak moves towards the jump but the overshoot tends to about 8.95% of the jump
 * (`Si(pi) / pi - 1/2`), instead of vanishing.
 *
 * @param coefficients - Fourier coefficients of a function with a jump at `at`.
 * @param terms - The partial sum's order N.
 * @param jump - `{ at, left, right }`: the location and the one-sided limits.
 * @returns The peak position, value and overshoot fraction.
 * @concept gibbs-phenomenon
 * @example
 * ```ts
 * import { gibbsOvershoot, squareWaveCoefficients } from 'mdml-engine/foundations'
 * Math.round(gibbsOvershoot(squareWaveCoefficients(401), 401, { at: 0, left: -1, right: 1 }).fraction * 1e4) / 1e4 // => 0.0895
 * ```
 */
export function gibbsOvershoot(
  coefficients: FourierCoefficients,
  terms: number,
  jump: { readonly at: number; readonly left: number; readonly right: number },
): GibbsOvershoot {
  if (!(terms >= 1)) throw new MdmlError('value', 'gibbsOvershoot: need at least one term')
  const size = jump.right - jump.left
  const sign = size >= 0 ? 1 : -1
  // The first peak lies within about one period of the highest harmonic past the jump.
  const width = (2 * coefficients.period) / (2 * terms + 1)
  const n = 400
  const xs = Array.from({ length: n + 1 }, (_, i) => jump.at + (width * i) / n)
  const values = fourierValues(coefficients, xs, terms)
  let best = 0
  for (let i = 1; i <= n; i++) if (sign * (values[i] ?? 0) > sign * (values[best] ?? 0)) best = i
  // Refine the peak by golden-section search on the bracketing samples.
  let lo = xs[Math.max(0, best - 1)] ?? jump.at
  let hi = xs[Math.min(n, best + 1)] ?? jump.at
  const g = (x: number): number => -sign * (fourierValues(coefficients, [x], terms)[0] ?? 0)
  const phi = (Math.sqrt(5) - 1) / 2
  let c = hi - phi * (hi - lo)
  let d = lo + phi * (hi - lo)
  for (let it = 0; it < 80 && hi - lo > 1e-14 * Math.max(1, Math.abs(hi)); it++) {
    if (g(c) < g(d)) hi = d
    else lo = c
    c = hi - phi * (hi - lo)
    d = lo + phi * (hi - lo)
  }
  const x = (lo + hi) / 2
  const peak = fourierValues(coefficients, [x], terms)[0] ?? Number.NaN
  return { x, peak, fraction: (sign * (peak - jump.right)) / Math.abs(size) }
}

function fourierValues(coefficients: FourierCoefficients, xs: readonly number[], terms: number): number[] {
  return toFlat(fourierPartialSum(coefficients, [...xs], terms))
}

/**
 * The first n coefficients of the power series of a rational generating function `P(x) / Q(x)` (with `Q(0) != 0`), by
 * long division of power series: `c_k = (p_k - sum_{j>=1} q_j c_{k-j}) / q_0`. Every sequence satisfying a linear
 * recurrence with constant coefficients has such a generating function.
 *
 * @param numerator - P, lowest degree first.
 * @param denominator - Q, lowest degree first.
 * @param n - How many coefficients.
 * @returns `c_0, ..., c_{n-1}`.
 * @throws MdmlError with code `value` when Q(0) = 0.
 * @concept generating-functions
 * @example
 * ```ts
 * import { rationalSeries } from 'mdml-engine/foundations'
 * // x / (1 - x - x^2) generates the Fibonacci numbers.
 * rationalSeries([0, 1], [1, -1, -1], 8) // => [0, 1, 1, 2, 3, 5, 8, 13]
 * ```
 */
export function rationalSeries(numerator: readonly number[], denominator: readonly number[], n: number): number[] {
  const q0 = denominator[0] ?? 0
  if (q0 === 0) throw new MdmlError('value', 'rationalSeries: the denominator must not vanish at 0')
  const c: number[] = []
  for (let k = 0; k < n; k++) {
    let s = numerator[k] ?? 0
    for (let j = 1; j <= Math.min(k, denominator.length - 1); j++) s -= (denominator[j] ?? 0) * (c[k - j] ?? 0)
    c.push(s / q0)
  }
  return c
}

/**
 * The Cauchy product of two power series truncated to n terms: the coefficients of `A(x) B(x)`, the convolution
 * `c_k = sum_j a_j b_{k-j}`.
 *
 * @param a - Coefficients of A.
 * @param b - Coefficients of B.
 * @param n - How many coefficients to keep (default all).
 * @returns The product's coefficients.
 * @concept generating-functions
 * @example
 * ```ts
 * import { seriesProduct } from 'mdml-engine/foundations'
 * // (1 + x + x^2 + ...)^2 = 1 + 2x + 3x^2 + ...
 * seriesProduct([1, 1, 1, 1], [1, 1, 1, 1], 4) // => [1, 2, 3, 4]
 * ```
 */
export function seriesProduct(a: readonly number[], b: readonly number[], n?: number): number[] {
  const full = polymul(a, b)
  return full.slice(0, n ?? full.length)
}

/**
 * The terms of a sequence defined by a linear recurrence with constant coefficients,
 * `a_k = c_1 a_{k-1} + ... + c_d a_{k-d}`, from its first d terms.
 *
 * @param coefficients - `[c_1, ..., c_d]`.
 * @param initial - `[a_0, ..., a_{d-1}]`.
 * @param n - How many terms.
 * @returns `a_0, ..., a_{n-1}`.
 * @concept recurrences
 * @example
 * ```ts
 * import { linearRecurrence } from 'mdml-engine/foundations'
 * linearRecurrence([1, 1], [2, 1], 6) // => [2, 1, 3, 4, 7, 11]
 * ```
 */
export function linearRecurrence(coefficients: readonly number[], initial: readonly number[], n: number): number[] {
  const d = coefficients.length
  if (initial.length !== d) throw new MdmlError('value', 'linearRecurrence: give one initial term per coefficient')
  const out = initial.slice(0, n)
  for (let k = d; k < n; k++) {
    out.push(kahanSum(coefficients.map((c, j) => c * (out[k - 1 - j] ?? 0))))
  }
  return out
}

/**
 * The rational generating function of a linear recurrence: for `a_k = sum c_j a_{k-j}`, `A(x) = P(x) / Q(x)` with
 * `Q = 1 - c_1 x - ... - c_d x^d` and P the first d terms of `Q A` (degree below d).
 *
 * @param coefficients - `[c_1, ..., c_d]`.
 * @param initial - `[a_0, ..., a_{d-1}]`.
 * @returns The numerator and denominator, lowest degree first.
 * @concept generating-functions
 * @example
 * ```ts
 * import { recurrenceGeneratingFunction } from 'mdml-engine/foundations'
 * recurrenceGeneratingFunction([1, 1], [0, 1]) // => { numerator: [0, 1], denominator: [1, -1, -1] }
 * ```
 */
export function recurrenceGeneratingFunction(
  coefficients: readonly number[],
  initial: readonly number[],
): { readonly numerator: number[]; readonly denominator: number[] } {
  const denominator = [1, ...coefficients.map((c) => -c)]
  const numerator = seriesProduct(denominator, initial, initial.length)
  return { numerator, denominator }
}
