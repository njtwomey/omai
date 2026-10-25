/**
 * Adaptive quadrature of a function on an interval: globally adaptive Gauss-Kronrod (QUADPACK's qag, the core of
 * `scipy.integrate.quad`), which keeps bisecting the subinterval with the largest error estimate, adaptive Simpson with
 * Richardson correction, and tanh-sinh (double-exponential) quadrature, which copes with endpoint singularities.
 * Subintervals are reported for figures.
 */

import {
  add,
  div,
  fromStorage,
  item,
  mul,
  neg,
  primalOf,
  promoteFloat,
  sub,
  sum,
  toFlat,
  type Value,
} from '../../tensor/index.ts'

// Gauss-Kronrod 7-15 nodes (non-negative half) and weights (QUADPACK qk15).
const XGK = [
  0.9914553711208126, 0.9491079123427585, 0.8648644233597691, 0.7415311855993945, 0.5860872354676911,
  0.4058451513773972, 0.20778495500789848, 0,
]
const WGK = [
  0.022935322010529224, 0.06309209262997856, 0.10479001032225019, 0.14065325971552592, 0.1690047266392679,
  0.19035057806478542, 0.20443294007529889, 0.20948214108472782,
]
const WG = [0.1294849661688697, 0.27970539148927664, 0.3818300505051189, 0.4179591836734694]

const K15_NODES: readonly number[] = [...XGK.slice(0, 7).map((x) => -x), 0, ...XGK.slice(0, 7).reverse()]
const K15_WEIGHTS: readonly number[] = [...WGK.slice(0, 7), WGK[7] ?? 0, ...WGK.slice(0, 7).reverse()]
// The 7-point Gauss rule uses every other Kronrod node (odd indices in XGK order 1, 3, 5, and the centre).
const G7_WEIGHTS: readonly number[] = K15_NODES.map((_, i) => {
  const k = i < 7 ? i : i === 7 ? 7 : 14 - i
  if (k === 7) return WG[3] ?? 0
  return k % 2 === 1 ? (WG[(k - 1) / 2] ?? 0) : 0
})

/** One subinterval of an adaptive quadrature, for figures. */
export interface QuadInterval {
  readonly a: number
  readonly b: number
  readonly estimate: number
  readonly error: number
}

/** Result of {@link quad}. */
export interface QuadResult {
  /** The integral, differentiable in the integrand's parameters. */
  readonly value: Value
  /** The estimated absolute error. */
  readonly error: number
  readonly intervals: readonly QuadInterval[]
  readonly evaluations: number
  readonly converged: boolean
}

/** Options of {@link quad}. */
export interface QuadOptions {
  readonly epsabs?: number
  readonly epsrel?: number
  /** The maximum number of subintervals (default 50, as scipy). */
  readonly limit?: number
  /** Points inside (a, b) where the integrand is irregular: the initial subdivision. */
  readonly points?: readonly number[]
}

/** f on [0, 1] after the substitution for infinite limits (QUADPACK qagi): returns the transformed integrand. */
function transform(f: (x: Value) => Value, a: number, b: number): { g: (t: Value) => Value; lo: number; hi: number } {
  if (Number.isFinite(a) && Number.isFinite(b)) return { g: f, lo: a, hi: b }
  if (Number.isFinite(a)) {
    // x = a + (1 - t) / t on (0, 1], dx = dt / t^2.
    return { g: (t) => div(f(add(a, div(sub(1, t), t))), mul(t, t)), lo: 0, hi: 1 }
  }
  if (Number.isFinite(b)) return { g: (t) => div(f(sub(b, div(sub(1, t), t))), mul(t, t)), lo: 0, hi: 1 }
  // Both infinite: f(x) + f(-x) on [0, inf).
  return { g: (t) => div(add(f(div(sub(1, t), t)), f(neg(div(sub(1, t), t)))), mul(t, t)), lo: 0, hi: 1 }
}

/**
 * Globally adaptive Gauss-Kronrod quadrature (QUADPACK's qag with the 7-15 pair, the core of scipy.integrate.quad):
 * repeatedly bisect the interval with the largest error estimate until the total error is below max(epsabs, epsrel
 * |I|). Infinite limits are mapped to [0, 1]. Intervals are chosen on concrete values; the returned value is assembled
 * from traced evaluations, so it is differentiable in anything the integrand closes over.
 *
 * @param f - The integrand, vectorised over a 1-D value of points.
 * @param a - Lower limit (may be -Infinity).
 * @param b - Upper limit (may be Infinity).
 * @param options - Tolerances (default 1.49e-8), subinterval limit and breakpoints.
 * @returns The value, error estimate and the intervals used.
 * @concept quadrature
 * @example
 * ```ts
 * import { quad } from 'mdml-engine/numerics/quadrature'
 * import { exp, item, neg, square } from 'mdml-engine/tensor'
 * Math.abs(item(quad((x) => exp(neg(square(x))), -Infinity, Infinity).value) - Math.sqrt(Math.PI)) < 1e-10 // => true
 * ```
 */
export function quad(f: (x: Value) => Value, a: number, b: number, options: QuadOptions = {}): QuadResult {
  if (a === b)
    return {
      value: mul(f(promoteFloat([a])[0] as Value), 0) as Value,
      error: 0,
      intervals: [],
      evaluations: 0,
      converged: true,
    }
  if (a > b) {
    const r = quad(f, b, a, options)
    return { ...r, value: neg(r.value) }
  }
  const { g, lo, hi } = transform(f, a, b)
  const epsabs = options.epsabs ?? 1.49e-8
  const epsrel = options.epsrel ?? 1.49e-8
  const limit = options.limit ?? 50
  let evaluations = 0
  const rule = (l: number, r: number): { value: Value; estimate: number; error: number } => {
    const half = (r - l) / 2
    const mid = (l + r) / 2
    const xs = K15_NODES.map((x) => mid + half * x)
    const fx = g(fromStorage(Float64Array.from(xs), [15], 'float64'))
    evaluations += 15
    const kronrod = mul(sum(mul(fx, fromStorage(Float64Array.from(K15_WEIGHTS), [15], 'float64'))), half)
    const values = toFlat(primalOf(fx))
    const gauss = half * values.reduce((s, v, i) => s + v * (G7_WEIGHTS[i] ?? 0), 0)
    const estimate = item(primalOf(kronrod))
    return { value: kronrod, estimate, error: Math.abs(estimate - gauss) }
  }
  const cuts = [lo, ...(options.points ?? []).filter((p) => p > lo && p < hi).sort((x, y) => x - y), hi]
  const intervals = cuts
    .slice(0, -1)
    .map((l, i) => ({ a: l, b: cuts[i + 1] as number, ...rule(l, cuts[i + 1] as number) }))
  const total = (): number => intervals.reduce((s, iv) => s + iv.estimate, 0)
  const errorSum = (): number => intervals.reduce((s, iv) => s + iv.error, 0)
  let converged = errorSum() <= Math.max(epsabs, epsrel * Math.abs(total()))
  while (!converged && intervals.length < limit) {
    let worst = 0
    intervals.forEach((iv, i) => {
      if (iv.error > (intervals[worst]?.error ?? 0)) worst = i
    })
    const iv = intervals[worst] as (typeof intervals)[number]
    const mid = (iv.a + iv.b) / 2
    if (mid <= iv.a || mid >= iv.b) break
    intervals.splice(worst, 1, { a: iv.a, b: mid, ...rule(iv.a, mid) }, { a: mid, b: iv.b, ...rule(mid, iv.b) })
    converged = errorSum() <= Math.max(epsabs, epsrel * Math.abs(total()))
  }
  const value = intervals.map((iv) => iv.value).reduce((x, y) => add(x, y))
  return {
    value,
    error: errorSum(),
    intervals: intervals.map(({ a: l, b: r, estimate, error }) => ({ a: l, b: r, estimate, error })),
    evaluations,
    converged,
  }
}

/**
 * Adaptive Simpson quadrature with Richardson correction (Lyness): recursively halve intervals whose two-panel and
 * one-panel Simpson estimates disagree by more than 15 tol.
 *
 * @param f - The integrand (vectorised over a 1-D value).
 * @param a - Lower limit.
 * @param b - Upper limit.
 * @param options - Tolerance (default 1e-10) and maximum depth (default 50).
 * @returns The estimate and the number of evaluations.
 * @example
 * ```ts
 * import { adaptiveSimpson } from 'mdml-engine/numerics/quadrature'
 * import { sin } from 'mdml-engine/tensor'
 * Math.abs(adaptiveSimpson((x) => sin(x), 0, Math.PI).value - 2) < 1e-9 // => true
 * ```
 */
export function adaptiveSimpson(
  f: (x: Value) => Value,
  a: number,
  b: number,
  options: { readonly tol?: number; readonly maxDepth?: number } = {},
): { readonly value: number; readonly evaluations: number } {
  let evaluations = 0
  const at = (x: number): number => {
    evaluations += 1
    return toFlat(primalOf(f(promoteFloat([x])[0] as Value)))[0] ?? Number.NaN
  }
  const simpsonPanel = (l: number, r: number, fl: number, fm: number, fr: number): number =>
    ((r - l) / 6) * (fl + 4 * fm + fr)
  const recurse = (
    l: number,
    r: number,
    fl: number,
    fm: number,
    fr: number,
    whole: number,
    tol: number,
    depth: number,
  ): number => {
    const m = (l + r) / 2
    const lm = (l + m) / 2
    const rm = (m + r) / 2
    const flm = at(lm)
    const frm = at(rm)
    const left = simpsonPanel(l, m, fl, flm, fm)
    const right = simpsonPanel(m, r, fm, frm, fr)
    const delta = left + right - whole
    if (depth <= 0 || Math.abs(delta) <= 15 * tol) return left + right + delta / 15
    return recurse(l, m, fl, flm, fm, left, tol / 2, depth - 1) + recurse(m, r, fm, frm, fr, right, tol / 2, depth - 1)
  }
  const fa = at(a)
  const fb = at(b)
  const fm = at((a + b) / 2)
  const value = recurse(a, b, fa, fm, fb, simpsonPanel(a, b, fa, fm, fb), options.tol ?? 1e-10, options.maxDepth ?? 50)
  return { value, evaluations }
}

/**
 * Tanh-sinh (double-exponential) quadrature (Takahasi and Mori): the substitution x = tanh((pi/2) sinh t) makes the
 * trapezoidal rule converge double-exponentially, even with endpoint singularities. Levels halve the step until two
 * estimates agree.
 *
 * @param f - The integrand (vectorised over a 1-D value).
 * @param a - Lower limit.
 * @param b - Upper limit.
 * @param options - Tolerance (default 1e-12) and maximum level (default 8).
 * @returns The value (differentiable in f's parameters) and the levels used.
 * @example
 * ```ts
 * import { tanhSinh } from 'mdml-engine/numerics/quadrature'
 * import { item, log } from 'mdml-engine/tensor'
 * Math.abs(item(tanhSinh((x) => log(x), 0, 1).value) + 1) < 1e-10 // => true
 * ```
 */
export function tanhSinh(
  f: (x: Value) => Value,
  a: number,
  b: number,
  options: { readonly tol?: number; readonly maxLevel?: number } = {},
): { readonly value: Value; readonly levels: number } {
  const half = (b - a) / 2
  const mid = (a + b) / 2
  const estimate = (h: number): Value => {
    const ts: number[] = []
    for (let t = -4; t <= 4 + 1e-12; t += h) ts.push(t)
    const xs: number[] = []
    const ws: number[] = []
    for (const t of ts) {
      const u = (Math.PI / 2) * Math.sinh(t)
      const x = Math.tanh(u)
      const w = (h * (Math.PI / 2) * Math.cosh(t)) / Math.cosh(u) ** 2
      // Skip nodes that round to the endpoints (the integrand may be singular there).
      if (Math.abs(x) >= 1 || w === 0) continue
      xs.push(mid + half * x)
      ws.push(half * w)
    }
    const fx = f(fromStorage(Float64Array.from(xs), [xs.length], 'float64'))
    return sum(mul(fx, fromStorage(Float64Array.from(ws), [ws.length], 'float64')))
  }
  let h = 0.5
  let previous = estimate(h)
  for (let level = 1; level <= (options.maxLevel ?? 8); level++) {
    h /= 2
    const next = estimate(h)
    if (
      Math.abs(item(primalOf(next)) - item(primalOf(previous))) <=
      (options.tol ?? 1e-12) * Math.max(1, Math.abs(item(primalOf(next))))
    ) {
      return { value: next, levels: level }
    }
    previous = next
  }
  return { value: previous, levels: options.maxLevel ?? 8 }
}
