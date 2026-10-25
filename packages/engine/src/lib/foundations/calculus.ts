/**
 * Single-variable calculus computed and shown step by step: numerical limits with Richardson extrapolation, difference
 * quotients and secant tables converging to the derivative, Riemann sums with their rectangles, the fundamental theorem
 * checked numerically, Taylor polynomials with the Lagrange error bound, Newton's method with every step, critical
 * points and the second-derivative test, mean value points and tangent lines, with derivatives from automatic
 * differentiation.
 */

import { taylorCoefficients, taylorPolynomial } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import { quad } from '../numerics/quadrature/index.ts'
import { brentq } from '../numerics/roots/index.ts'
import { item, linspace, onesLike, promoteFloat, type TensorLike, toFlat, type Value } from '../tensor/index.ts'
import { extrapolateToZero } from './_extrapolate.ts'

/**
 * A real function of one variable, written with tensor operations and applied elementwise (vectorised over `[n]`), so
 * that it can be evaluated on grids and differentiated.
 */
export type RealFunction = (x: Value) => Value

const evalAt = (f: RealFunction, xs: readonly number[]): number[] => toFlat(f(promoteFloat([...xs])[0] as Value))
const at = (f: RealFunction, x: number): number => evalAt(f, [x])[0] ?? Number.NaN

/**
 * The derivatives `f(x), f'(x), ..., f^(order)(x)` at points, from one Taylor-mode pass.
 *
 * @param f - The function.
 * @param x - Points `[n]` (or a scalar).
 * @param order - The highest derivative.
 * @returns `order + 1` arrays of values at the points.
 * @concept derivative
 * @example
 * ```ts
 * import { derivatives } from 'mdml-engine/foundations'
 * import { mul, type Value } from 'mdml-engine/tensor'
 * derivatives((x: Value) => mul(x, mul(x, x)), [2], 3) // => [[8], [12], [12], [6]]
 * ```
 */
export function derivatives(f: RealFunction, x: TensorLike, order: number): number[][] {
  const [xv0] = promoteFloat(x) as [Value]
  const xv = xv0.shape.length === 0 ? (promoteFloat([item(xv0)])[0] as Value) : xv0
  const cs = taylorCoefficients(f, xv, onesLike(xv), order)
  let factorial = 1
  return cs.map((c, k) => {
    if (k > 0) factorial *= k
    return toFlat(c).map((v) => v * factorial)
  })
}

/** One estimate in a sequence that approaches a limit. */
export interface LimitStep {
  readonly h: number
  readonly value: number
}

/** Result of {@link limit}. */
export interface LimitResult {
  /** The estimate (NaN when the one-sided limits disagree). */
  readonly value: number
  /** The one-sided estimates (from below and from above), when both were taken. */
  readonly left: number
  readonly right: number
  /** The raw values f(a +- h) as h shrinks, from the right (or the requested side). */
  readonly steps: readonly LimitStep[]
  /** Whether the extrapolated estimates settled (and the sides agree). */
  readonly exists: boolean
}

function oneSided(
  f: RealFunction,
  a: number,
  sign: 1 | -1,
  h0: number,
  levels: number,
): { value: number; steps: LimitStep[]; settled: boolean } {
  const hs = Array.from({ length: levels }, (_, k) => h0 / 2 ** k)
  const xs = hs.map((h) => (Number.isFinite(a) ? a + sign * h : sign / h))
  const values = evalAt(f, xs)
  const steps = hs.map((h, k) => ({ h, value: values[k] ?? Number.NaN }))
  // Richardson extrapolation in h, assuming f(a + h) = L + c1 h + c2 h^2 + ...
  const { value: best, change: bestErr } = extrapolateToZero(values)
  const scale = Math.max(1, Math.abs(best))
  return { value: best, steps, settled: Number.isFinite(best) && bestErr <= 1e-6 * scale }
}

/**
 * A numerical limit `lim_{x -> a} f(x)` (or `x -> +-infinity`), from function values at `a +- h` with h halving,
 * sharpened by Richardson extrapolation. Two-sided limits compare the sides: a jump gives `exists = false` and NaN.
 * Numerical limits are evidence, not proof: a function can fool any finite sample.
 *
 * @param f - The function.
 * @param a - The point (may be +-Infinity).
 * @param options - Side ('both' default, 'left', 'right'), starting step (0.1) and number of halvings (12).
 * @returns The estimate, the sides, the raw sequence and whether the limit appears to exist.
 * @concept limits
 * @example
 * ```ts
 * import { limit } from 'mdml-engine/foundations'
 * import { div, sin, sign, type Value } from 'mdml-engine/tensor'
 * Math.round(limit((x: Value) => div(sin(x), x), 0).value * 1e12) / 1e12 // => 1
 * limit((x: Value) => sign(x), 0).exists // => false
 * ```
 */
export function limit(
  f: RealFunction,
  a: number,
  options: { readonly side?: 'both' | 'left' | 'right'; readonly h?: number; readonly levels?: number } = {},
): LimitResult {
  const h0 = options.h ?? 0.1
  const levels = options.levels ?? 12
  if (!Number.isFinite(a)) {
    // At +-infinity approach along x = +-1/h as h -> 0+.
    const r = oneSided(f, a, a > 0 ? 1 : -1, h0, levels)
    return {
      value: r.value,
      left: a > 0 ? r.value : Number.NaN,
      right: a < 0 ? r.value : Number.NaN,
      steps: r.steps,
      exists: r.settled,
    }
  }
  const side = options.side ?? 'both'
  const fromAbove = side !== 'left' ? oneSided(f, a, 1, h0, levels) : null
  const fromBelow = side !== 'right' ? oneSided(f, a, -1, h0, levels) : null
  const right = fromAbove?.value ?? Number.NaN
  const left = fromBelow?.value ?? Number.NaN
  if (side === 'right')
    return { value: right, left, right, steps: fromAbove?.steps ?? [], exists: fromAbove?.settled ?? false }
  if (side === 'left')
    return { value: left, left, right, steps: fromBelow?.steps ?? [], exists: fromBelow?.settled ?? false }
  const agree = Math.abs(left - right) <= 1e-6 * Math.max(1, Math.abs(left), Math.abs(right))
  const exists = agree && (fromAbove?.settled ?? false) && (fromBelow?.settled ?? false)
  return { value: agree ? (left + right) / 2 : Number.NaN, left, right, steps: fromAbove?.steps ?? [], exists }
}

/** The kinds of difference quotient. */
export type DifferenceScheme = 'forward' | 'backward' | 'central'

/**
 * A difference quotient: forward `(f(x + h) - f(x)) / h`, backward `(f(x) - f(x - h)) / h` or central
 * `(f(x + h) - f(x - h)) / 2h`, the secant slopes whose limit is the derivative.
 *
 * @param f - The function.
 * @param x - The point.
 * @param h - The step (positive).
 * @param scheme - 'forward' (default), 'backward' or 'central'.
 * @returns The quotient.
 * @concept difference-quotient
 * @example
 * ```ts
 * import { differenceQuotient } from 'mdml-engine/foundations'
 * import { mul, type Value } from 'mdml-engine/tensor'
 * differenceQuotient((x: Value) => mul(x, x), 3, 0.5) // => 6.5
 * differenceQuotient((x: Value) => mul(x, x), 3, 0.5, 'central') // => 6
 * ```
 */
export function differenceQuotient(
  f: RealFunction,
  x: number,
  h: number,
  scheme: DifferenceScheme = 'forward',
): number {
  if (scheme === 'central') {
    const [a = 0, b = 0] = evalAt(f, [x + h, x - h])
    return (a - b) / (2 * h)
  }
  const [a = 0, b = 0] = evalAt(f, scheme === 'forward' ? [x + h, x] : [x, x - h])
  return (a - b) / h
}

/**
 * Difference quotients for a sequence of shrinking steps next to the exact derivative (by automatic differentiation),
 * showing first-order (one-sided) or second-order (central) convergence and, for tiny steps, rounding error taking over.
 *
 * @param f - The function.
 * @param x - The point.
 * @param options - Scheme (default 'forward') and steps (default 10^-1 to 10^-12).
 * @returns Per step: h, the quotient and its error against the derivative.
 * @concept difference-quotient
 * @example
 * ```ts
 * import { secantTable } from 'mdml-engine/foundations'
 * import { exp } from 'mdml-engine/tensor'
 * secantTable(exp, 0, { steps: [0.1, 0.01] }).map((r) => r.error < 0.06) // => [true, true]
 * ```
 */
export function secantTable(
  f: RealFunction,
  x: number,
  options: { readonly scheme?: DifferenceScheme; readonly steps?: readonly number[] } = {},
): { readonly h: number; readonly quotient: number; readonly error: number }[] {
  const exact = derivatives(f, [x], 1)[1]?.[0] ?? Number.NaN
  const steps = options.steps ?? Array.from({ length: 12 }, (_, k) => 10 ** -(k + 1))
  return steps.map((h) => {
    const quotient = differenceQuotient(f, x, h, options.scheme ?? 'forward')
    return { h, quotient, error: Math.abs(quotient - exact) }
  })
}

/** The Riemann-sum rules. */
export type RiemannRule = 'left' | 'right' | 'midpoint' | 'trapezoid'

/** Result of {@link riemannSum}: the value and the shapes it adds up, for drawing. */
export interface RiemannSum {
  readonly value: number
  /** One piece per subinterval: its ends and the heights used (equal for rectangles, the two ends for trapezoids). */
  readonly pieces: readonly { readonly x0: number; readonly x1: number; readonly h0: number; readonly h1: number }[]
}

/**
 * A Riemann sum of f over `[a, b]` with n equal subintervals: left, right or midpoint rectangles, or trapezoids.
 *
 * @param f - The function.
 * @param a - Lower limit.
 * @param b - Upper limit.
 * @param n - Number of subintervals.
 * @param rule - 'left', 'right', 'midpoint' (default) or 'trapezoid'.
 * @returns The sum and its pieces.
 * @concept riemann-sums
 * @example
 * ```ts
 * import { riemannSum } from 'mdml-engine/foundations'
 * import { mul, type Value } from 'mdml-engine/tensor'
 * riemannSum((x: Value) => mul(x, x), 0, 1, 2, 'left').value // => 0.125
 * riemannSum((x: Value) => mul(x, x), 0, 1, 2, 'trapezoid').value // => 0.375
 * ```
 */
export function riemannSum(
  f: RealFunction,
  a: number,
  b: number,
  n: number,
  rule: RiemannRule = 'midpoint',
): RiemannSum {
  if (!(Number.isInteger(n) && n >= 1)) throw new MdmlError('value', 'riemannSum: n must be a positive integer')
  const dx = (b - a) / n
  const edges = Array.from({ length: n + 1 }, (_, i) => (i === n ? b : a + i * dx))
  const mids = Array.from({ length: n }, (_, i) => a + (i + 0.5) * dx)
  const fe = rule === 'midpoint' ? [] : evalAt(f, edges)
  const fm = rule === 'midpoint' ? evalAt(f, mids) : []
  const pieces = Array.from({ length: n }, (_, i) => {
    const x0 = edges[i] ?? 0
    const x1 = edges[i + 1] ?? 0
    if (rule === 'trapezoid') return { x0, x1, h0: fe[i] ?? 0, h1: fe[i + 1] ?? 0 }
    const h = rule === 'left' ? (fe[i] ?? 0) : rule === 'right' ? (fe[i + 1] ?? 0) : (fm[i] ?? 0)
    return { x0, x1, h0: h, h1: h }
  })
  let value = 0
  for (const p of pieces) value += ((p.h0 + p.h1) / 2) * (p.x1 - p.x0)
  return { value, pieces }
}

/** Result of {@link fundamentalTheorem}. */
export interface FundamentalTheoremCheck {
  readonly x: number[]
  /** The accumulation function `F(x) = int_a^x f`. */
  readonly accumulated: number[]
  /** The central-difference derivative of F at x. */
  readonly derivative: number[]
  /** f(x), which the derivative should match. */
  readonly integrand: number[]
  readonly maxError: number
}

/**
 * Check the fundamental theorem of calculus numerically: build `F(x) = int_a^x f(t) dt` by adaptive quadrature,
 * differentiate it by central differences, and compare with f.
 *
 * @param f - A continuous function.
 * @param a - The lower limit of the accumulation.
 * @param xs - Points where F' and f are compared.
 * @param h - The difference step (default 1e-4).
 * @returns The accumulation, its derivative, f, and the largest disagreement.
 * @concept fundamental-theorem-of-calculus
 * @example
 * ```ts
 * import { fundamentalTheorem } from 'mdml-engine/foundations'
 * import { cos } from 'mdml-engine/tensor'
 * fundamentalTheorem(cos, 0, [0.5, 1, 2]).maxError < 1e-7 // => true
 * ```
 */
export function fundamentalTheorem(
  f: RealFunction,
  a: number,
  xs: readonly number[],
  h = 1e-4,
): FundamentalTheoremCheck {
  const F = (x: number): number => item(quad(f, a, x, { epsabs: 1e-13, epsrel: 1e-13 }).value)
  const accumulated = xs.map(F)
  const derivative = xs.map((x) => (F(x + h) - F(x - h)) / (2 * h))
  const integrand = evalAt(f, xs)
  let maxError = 0
  for (let i = 0; i < xs.length; i++)
    maxError = Math.max(maxError, Math.abs((derivative[i] ?? 0) - (integrand[i] ?? 0)))
  return { x: [...xs], accumulated, derivative, integrand, maxError }
}

/** A Taylor polynomial with the Lagrange bound on its error over an interval. */
export interface TaylorApproximation {
  readonly center: number
  readonly order: number
  /** `coefficients[k]` multiplies `(x - center)^k`. */
  readonly coefficients: readonly number[]
  /** The largest |f^(order + 1)| found on the interval (by sampling). */
  readonly derivativeBound: number
  readonly interval: readonly [number, number]
  /** Evaluate the polynomial (differentiable). */
  evaluate(x: TensorLike): Value
  /** The Lagrange remainder bound `M |x - c|^(n+1) / (n+1)!` at points of the interval. */
  remainderBound(x: readonly number[]): number[]
}

/**
 * A Taylor polynomial of degree `order` about `center` (through Taylor-mode automatic differentiation) and the Lagrange
 * form of its remainder: `|f(x) - P_n(x)| <= M |x - c|^(n+1) / (n+1)!` where M bounds `|f^(n+1)|` on the interval (here
 * the maximum over a fine sample, so a sharp but not rigorous bound).
 *
 * @param f - The function (smooth on the interval).
 * @param center - The expansion point.
 * @param order - The degree.
 * @param interval - Where the bound should hold (default `[center - 1, center + 1]`).
 * @param samples - Points used to bound the next derivative (default 201).
 * @returns The polynomial, its coefficients and the remainder bound.
 * @concept taylor-series
 * @example
 * ```ts
 * import { taylorApproximation } from 'mdml-engine/foundations'
 * import { exp } from 'mdml-engine/tensor'
 * const t = taylorApproximation(exp, 0, 2, [-1, 1])
 * t.coefficients // => [1, 1, 0.5]
 * Math.round(t.remainderBound([1])[0] * 1e12) / 1e12 // => 0.45304697141
 * ```
 */
export function taylorApproximation(
  f: RealFunction,
  center: number,
  order: number,
  interval: readonly [number, number] = [center - 1, center + 1],
  samples = 201,
): TaylorApproximation {
  if (!(Number.isInteger(order) && order >= 0))
    throw new MdmlError('value', 'taylorApproximation: order must be a non-negative integer')
  const p = taylorPolynomial(f, center, order)
  const grid = toFlat(linspace(interval[0], interval[1], samples))
  const next = derivatives(f, grid, order + 1)[order + 1] ?? []
  let derivativeBound = 0
  for (const v of next) derivativeBound = Math.max(derivativeBound, Math.abs(v))
  let factorial = 1
  for (let k = 2; k <= order + 1; k++) factorial *= k
  return {
    center,
    order,
    coefficients: p.coefficients,
    derivativeBound,
    interval,
    evaluate: p.evaluate,
    remainderBound: (xs) => xs.map((x) => (derivativeBound * Math.abs(x - center) ** (order + 1)) / factorial),
  }
}

/** One step of Newton's method. */
export interface NewtonStep {
  readonly x: number
  readonly fx: number
  readonly slope: number
  /** Where the tangent line at x crosses zero: the next iterate. */
  readonly next: number
}

/** Result of {@link newtonSteps}. */
export interface NewtonRun {
  readonly root: number
  readonly converged: boolean
  readonly steps: readonly NewtonStep[]
}

/**
 * Newton's method `x <- x - f(x) / f'(x)` with every step recorded (the tangent line at each iterate crosses zero at
 * the next), the derivative by automatic differentiation. Quadratic convergence near a simple root: the number of
 * correct digits roughly doubles each step.
 *
 * @param f - The function.
 * @param x0 - The starting point.
 * @param options - Tolerance on the step (default 1e-12, relative) and iteration cap (default 50).
 * @returns The root, whether it converged, and the steps.
 * @concept newtons-method
 * @example
 * ```ts
 * import { newtonSteps } from 'mdml-engine/foundations'
 * import { mul, sub, type Value } from 'mdml-engine/tensor'
 * const r = newtonSteps((x: Value) => sub(mul(x, x), 2), 1)
 * r.root // => 1.4142135623730951
 * r.steps[0].next // => 1.5
 * ```
 */
export function newtonSteps(
  f: RealFunction,
  x0: number,
  options: { readonly tol?: number; readonly maxIter?: number } = {},
): NewtonRun {
  const tol = options.tol ?? 1e-12
  const steps: NewtonStep[] = []
  let x = x0
  for (let i = 0; i < (options.maxIter ?? 50); i++) {
    const [fx = Number.NaN, slope = Number.NaN] = derivatives(f, [x], 1).map((d) => d[0] ?? Number.NaN)
    if (fx === 0) return { root: x, converged: true, steps }
    if (slope === 0 || !Number.isFinite(slope)) return { root: x, converged: false, steps }
    const next = x - fx / slope
    steps.push({ x, fx, slope, next })
    if (Math.abs(next - x) <= tol * Math.max(1, Math.abs(next))) return { root: next, converged: true, steps }
    x = next
  }
  return { root: x, converged: false, steps }
}

/** The verdict of the second-derivative test. */
export type CriticalPointKind = 'minimum' | 'maximum' | 'inconclusive'

/** A critical point of a function of one variable. */
export interface CriticalPoint {
  readonly x: number
  readonly value: number
  readonly secondDerivative: number
  readonly kind: CriticalPointKind
}

/**
 * The second-derivative test at a critical point: `f'' > 0` a local minimum, `f'' < 0` a local maximum, `f'' = 0`
 * inconclusive (the higher derivatives decide: an inflection like x^3 or a flat extremum like x^4).
 *
 * @param f - The function.
 * @param x - A critical point (f'(x) = 0).
 * @param tol - Below this |f''| counts as zero (default 1e-10).
 * @returns The classification.
 * @concept second-derivative-test
 * @example
 * ```ts
 * import { secondDerivativeTest } from 'mdml-engine/foundations'
 * import { cos } from 'mdml-engine/tensor'
 * secondDerivativeTest(cos, 0) // => 'maximum'
 * ```
 */
export function secondDerivativeTest(f: RealFunction, x: number, tol = 1e-10): CriticalPointKind {
  const second = derivatives(f, [x], 2)[2]?.[0] ?? Number.NaN
  return second > tol ? 'minimum' : second < -tol ? 'maximum' : 'inconclusive'
}

/** Roots of g on [a, b]: sign changes on a grid, refined by Brent's method (exact zeros on the grid kept). */
function rootsOn(g: (xs: readonly number[]) => number[], a: number, b: number, n: number): number[] {
  const xs = toFlat(linspace(a, b, n + 1))
  const gs = g(xs)
  const roots: number[] = []
  for (let i = 0; i < n; i++) {
    const x0 = xs[i] ?? 0
    const x1 = xs[i + 1] ?? 0
    const g0 = gs[i] ?? Number.NaN
    const g1 = gs[i + 1] ?? Number.NaN
    if (g0 === 0) roots.push(x0)
    else if (g0 * g1 < 0) roots.push(brentq((x) => g([x])[0] ?? Number.NaN, x0, x1, { xtol: 1e-14, rtol: 1e-15 }).root)
  }
  if ((gs[n] ?? 1) === 0) roots.push(xs[n] ?? b)
  return roots
}

/**
 * The critical points of f in `[a, b]` (where f' changes sign or vanishes on a fine grid, refined by Brent's method),
 * each classified by the second-derivative test. Double roots of f' that do not change sign (such as the inflection of
 * x^3 between grid points) can be missed: refine `n` or test candidates with {@link secondDerivativeTest}.
 *
 * @param f - The function.
 * @param a - Left end.
 * @param b - Right end.
 * @param n - Grid intervals (default 400).
 * @returns The critical points in increasing order.
 * @concept critical-points
 * @example
 * ```ts
 * import { criticalPoints } from 'mdml-engine/foundations'
 * import { mul, sub, type Value } from 'mdml-engine/tensor'
 * criticalPoints((x: Value) => sub(mul(x, mul(x, x)), mul(3, x)), -2, 2.1).map((c) => [c.x, c.kind]) // => [[-1, 'maximum'], [1, 'minimum']]
 * ```
 */
export function criticalPoints(f: RealFunction, a: number, b: number, n = 400): CriticalPoint[] {
  const slope = (xs: readonly number[]): number[] => derivatives(f, xs, 1)[1] ?? []
  return rootsOn(slope, a, b, n).map((x) => {
    const [value = Number.NaN, , second = Number.NaN] = derivatives(f, [x], 2).map((d) => d[0] ?? Number.NaN)
    return { x, value, secondDerivative: second, kind: secondDerivativeTest(f, x) }
  })
}

/**
 * The points promised by the mean value theorem: every c in `(a, b)` where the tangent is parallel to the secant,
 * `f'(c) = (f(b) - f(a)) / (b - a)`.
 *
 * @param f - A function differentiable on `(a, b)`.
 * @param a - Left end.
 * @param b - Right end.
 * @param n - Grid intervals for the search (default 400).
 * @returns The secant slope and the points c.
 * @concept mean-value-theorem
 * @example
 * ```ts
 * import { meanValuePoints } from 'mdml-engine/foundations'
 * import { mul, type Value } from 'mdml-engine/tensor'
 * meanValuePoints((x: Value) => mul(x, x), 0, 2) // => { slope: 2, points: [1] }
 * ```
 */
export function meanValuePoints(
  f: RealFunction,
  a: number,
  b: number,
  n = 400,
): { readonly slope: number; readonly points: number[] } {
  const slope = (at(f, b) - at(f, a)) / (b - a)
  const g = (xs: readonly number[]): number[] => (derivatives(f, xs, 1)[1] ?? []).map((d) => d - slope)
  const points = rootsOn(g, a, b, n).filter((c) => c > a && c < b)
  return { slope, points }
}

/**
 * The tangent line to f at a point, as slope and intercept (`y = slope x + intercept`), from automatic
 * differentiation: the best linear approximation.
 *
 * @param f - The function.
 * @param x - The point of tangency.
 * @returns `{ slope, intercept }`.
 * @concept derivative
 * @example
 * ```ts
 * import { tangentLine } from 'mdml-engine/foundations'
 * import { mul, type Value } from 'mdml-engine/tensor'
 * tangentLine((x: Value) => mul(x, x), 3) // => { slope: 6, intercept: -9 }
 * ```
 */
export function tangentLine(f: RealFunction, x: number): { readonly slope: number; readonly intercept: number } {
  const [fx = Number.NaN, slope = Number.NaN] = derivatives(f, [x], 1).map((d) => d[0] ?? Number.NaN)
  return { slope, intercept: fx - slope * x }
}

/**
 * A definite integral by adaptive quadrature next to the antiderivative difference `F(b) - F(a)` (the fundamental
 * theorem in its second form), when an antiderivative is known.
 *
 * @param f - The integrand.
 * @param antiderivative - Any F with F' = f.
 * @param a - Lower limit.
 * @param b - Upper limit.
 * @returns Both values and their difference.
 * @concept fundamental-theorem-of-calculus
 * @example
 * ```ts
 * import { definiteIntegral } from 'mdml-engine/foundations'
 * import { cos, sin } from 'mdml-engine/tensor'
 * definiteIntegral(cos, sin, 0, Math.PI / 2).byAntiderivative // => 1
 * ```
 */
export function definiteIntegral(
  f: RealFunction,
  antiderivative: RealFunction,
  a: number,
  b: number,
): { readonly byQuadrature: number; readonly byAntiderivative: number; readonly difference: number } {
  const byQuadrature = item(quad(f, a, b).value)
  const [fb = 0, fa = 0] = evalAt(antiderivative, [b, a])
  const byAntiderivative = fb - fa
  return { byQuadrature, byAntiderivative, difference: Math.abs(byQuadrature - byAntiderivative) }
}
