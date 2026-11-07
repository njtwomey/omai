/**
 * Numerical differentiation: Ridders' extrapolated central differences for a scalar derivative with an error estimate,
 * numpy.gradient on sampled values, finite-difference Jacobians and Hessians (as scipy.optimize's `approx_derivative`),
 * Fornberg's weights for any stencil, complex-step derivatives accurate to machine precision, and dense difference
 * matrices.
 */

import { MdmlError } from '../../base/index.ts'
import {
  add,
  concat,
  concreteOrThrow,
  div,
  fromStorage,
  moveaxis,
  mul,
  promoteFloat,
  sliceAxis,
  sub,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'

const EPS = Number.EPSILON

/** Result of {@link derivative}. */
export interface DerivativeResult {
  readonly df: number
  /** The error estimate: the difference between the last two extrapolations. */
  readonly error: number
  /** The number of step halvings used. */
  readonly levels: number
  /** The Richardson tableau, one row per step size. */
  readonly table: readonly (readonly number[])[]
}

/**
 * The derivative of a scalar function by central differences with Richardson extrapolation (Ridders' method): central
 * differences at steps h, h / c, h / c^2, ... are extrapolated in a tableau, stopping when the error estimate stops
 * improving. The error is typically near machine precision for smooth functions.
 *
 * @param f - The function.
 * @param x - The point.
 * @param options - Initial step (default 0.5, or |x| / 256 where 0.5 is lost in the rounding of x, beyond about 4e15),
 * step factor (default 2) and maximum levels (default 12).
 * @returns The derivative, its error estimate and the tableau.
 * @concept finite-differences
 * @index function Derivative of a scalar function by Richardson-extrapolated differences.
 * @example
 * ```ts
 * import { derivative } from 'mdml-engine/numerics/differentiate'
 * Math.abs(derivative(Math.exp, 1).df - Math.E) < 1e-12 // => true
 * ```
 */
export function derivative(
  f: (x: number) => number,
  x: number,
  options: { readonly initialStep?: number; readonly stepFactor?: number; readonly maxLevels?: number } = {},
): DerivativeResult {
  const c = options.stepFactor ?? 2
  const c2 = c * c
  let h = options.initialStep ?? (x + 0.5 === x - 0.5 ? Math.abs(x) / 256 : 0.5)
  const levels = options.maxLevels ?? 12
  const table: number[][] = [[(f(x + h) - f(x - h)) / (2 * h)]]
  let best = table[0]?.[0] ?? Number.NaN
  let error = Number.POSITIVE_INFINITY
  let used = 0
  for (let i = 1; i < levels; i++) {
    h /= c
    const row = [(f(x + h) - f(x - h)) / (2 * h)]
    let factor = c2
    for (let j = 1; j <= i; j++) {
      const prev = table[i - 1]?.[j - 1] ?? 0
      row.push(((row[j - 1] ?? 0) * factor - prev) / (factor - 1))
      factor *= c2
      const e = Math.max(Math.abs((row[j] ?? 0) - (row[j - 1] ?? 0)), Math.abs((row[j] ?? 0) - prev))
      if (e <= error) {
        error = e
        best = row[j] ?? best
        used = i
      }
    }
    table.push(row)
    // Stop once higher order makes things worse by a wide margin (rounding has taken over).
    if (Math.abs((row[i] ?? 0) - (table[i - 1]?.[i - 1] ?? 0)) >= 2 * error) break
  }
  return { df: best, error, levels: used, table }
}

/** Options of {@link gradient}. */
export interface GradientOptions {
  /** Constant spacing (default 1) or the sample coordinates along the axis. */
  readonly spacing?: number | TensorLike
  /** The axis (default 0). */
  readonly axis?: number
  /** One-sided differences of first or second order at the boundaries (default 1, as numpy). */
  readonly edgeOrder?: 1 | 2
}

/**
 * The gradient of sampled values along one axis (numpy.gradient): second-order central differences inside (exact
 * for quadratics also on uneven grids) and first- or second-order one-sided differences at the edges. Linear in the
 * samples, so differentiable in them.
 *
 * @param y - Samples.
 * @param options - Spacing or coordinates, axis and edge order.
 * @returns The derivative estimates, the shape of y.
 * @example
 * ```ts
 * import { gradient } from 'mdml-engine/numerics/differentiate'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(gradient([1, 2, 4, 7, 11])) // => [1, 1.5, 2.5, 3.5, 4]
 * ```
 */
export function gradient(y: TensorLike, options: GradientOptions = {}): Value {
  const [yv] = promoteFloat(y) as [Value]
  const rank = yv.shape.length
  const axis = (options.axis ?? 0) < 0 ? rank + (options.axis ?? 0) : (options.axis ?? 0)
  const n = yv.shape[axis] ?? 0
  const edge = options.edgeOrder ?? 1
  if (n < edge + 1) throw new MdmlError('shape', `gradient: at least ${edge + 1} samples are needed along the axis`)
  // Work on the last axis, then move it back.
  const z = moveaxis(yv, axis, -1)
  const at = (start: number, stop: number): Value => sliceAxis(z, -1, start, stop)
  const spacing = options.spacing ?? 1
  const xs =
    typeof spacing === 'number'
      ? Array.from({ length: n }, (_, i) => i * spacing)
      : toFlat(concreteOrThrow(promoteFloat(spacing)[0] as Tensor, 'spacing'))
  if (xs.length !== n) throw new MdmlError('shape', 'gradient: coordinates must match the axis length')
  const vector = (values: readonly number[]): Tensor =>
    fromStorage(Float64Array.from(values), [values.length], 'float64')
  const d = (i: number): number => (xs[i + 1] ?? 0) - (xs[i] ?? 0)
  const pieces: Value[] = []
  if (edge === 1) {
    pieces.push(div(sub(at(1, 2), at(0, 1)), d(0)))
  } else {
    const dx1 = d(0)
    const dx2 = d(1)
    const a = -(2 * dx1 + dx2) / (dx1 * (dx1 + dx2))
    const b = (dx1 + dx2) / (dx1 * dx2)
    const c = -dx1 / (dx2 * (dx1 + dx2))
    pieces.push(add(add(mul(at(0, 1), a), mul(at(1, 2), b)), mul(at(2, 3), c)))
  }
  if (n > 2) {
    const ca: number[] = []
    const cb: number[] = []
    const cc: number[] = []
    for (let i = 1; i < n - 1; i++) {
      const hs = d(i - 1)
      const hd = d(i)
      ca.push(-hd / (hs * (hd + hs)))
      cb.push((hd - hs) / (hd * hs))
      cc.push(hs / (hd * (hd + hs)))
    }
    pieces.push(add(add(mul(at(0, n - 2), vector(ca)), mul(at(1, n - 1), vector(cb))), mul(at(2, n), vector(cc))))
  }
  if (edge === 1) {
    pieces.push(div(sub(at(n - 1, n), at(n - 2, n - 1)), d(n - 2)))
  } else {
    const dx1 = d(n - 3)
    const dx2 = d(n - 2)
    const a = dx2 / (dx1 * (dx1 + dx2))
    const b = -(dx2 + dx1) / (dx1 * dx2)
    const c = (2 * dx2 + dx1) / (dx2 * (dx1 + dx2))
    pieces.push(add(add(mul(at(n - 3, n - 2), a), mul(at(n - 2, n - 1), b)), mul(at(n - 1, n), c)))
  }
  return moveaxis(concat(pieces, -1), -1, axis)
}

/** Options of the finite-difference Jacobian and Hessian. */
export interface FdOptions {
  /** `forward` (2-point) or `central` (3-point, default). */
  readonly method?: 'forward' | 'central'
  /** Relative step; by default eps^(1/2) forward and eps^(1/3) central (as scipy.optimize.approx_fprime). */
  readonly relativeStep?: number
}

function steps(x: readonly number[], relative: number): number[] {
  return x.map((v) => {
    const h = relative * (v >= 0 ? 1 : -1) * Math.max(1, Math.abs(v))
    // The step actually taken, after rounding x + h.
    return v + h - v
  })
}

/**
 * The Jacobian of f: R^n -> R^m by finite differences (scipy.optimize's approx_derivative): forward or central
 * differences with steps scaled to |x|. For checking autodiff and for black-box functions.
 *
 * @param f - The function on flat number arrays.
 * @param x - The point.
 * @param options - Method and relative step.
 * @returns The Jacobian `[m, n]`.
 * @example
 * ```ts
 * import { jacobianFd } from 'mdml-engine/numerics/differentiate'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(jacobianFd(([a, b]) => [a * b, a + b], [2, 3])).flat().map((v) => Math.round(v * 1e8) / 1e8) // => [3, 2, 1, 1]
 * ```
 */
export function jacobianFd(
  f: (x: number[]) => readonly number[],
  x: readonly number[],
  options: FdOptions = {},
): Tensor {
  const central = (options.method ?? 'central') === 'central'
  const h = steps(x, options.relativeStep ?? (central ? EPS ** (1 / 3) : EPS ** 0.5))
  const f0 = central ? [] : f([...x])
  const columns: number[][] = x.map((_, j) => {
    const plus = [...x]
    plus[j] = (plus[j] ?? 0) + (h[j] ?? 0)
    const fp = f(plus)
    if (!central) return fp.map((v, i) => (v - (f0[i] ?? 0)) / (h[j] ?? 1))
    const minus = [...x]
    minus[j] = (minus[j] ?? 0) - (h[j] ?? 0)
    const fm = f(minus)
    return fp.map((v, i) => (v - (fm[i] ?? 0)) / (2 * (h[j] ?? 1)))
  })
  const m = columns[0]?.length ?? 0
  const n = x.length
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[i * n + j] = columns[j]?.[i] ?? 0
  return fromStorage(out, [m, n], 'float64')
}

/**
 * The Hessian of a scalar function by central second differences, with steps eps^(1/4) max(1, |x|), symmetrised.
 *
 * @param f - The function on flat number arrays.
 * @param x - The point.
 * @param options - Relative step.
 * @returns The Hessian `[n, n]`.
 * @example
 * ```ts
 * import { hessianFd } from 'mdml-engine/numerics/differentiate'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(hessianFd(([a, b]) => a * a * b, [1, 2])).flat().map((v) => Math.round(v * 1e5) / 1e5) // => [4, 2, 2, 0]
 * ```
 */
export function hessianFd(
  f: (x: number[]) => number,
  x: readonly number[],
  options: { readonly relativeStep?: number } = {},
): Tensor {
  const n = x.length
  const h = steps(x, options.relativeStep ?? EPS ** 0.25)
  const at = (di: number, si: number, dj: number, sj: number): number => {
    const p = [...x]
    p[di] = (p[di] ?? 0) + si * (h[di] ?? 0)
    p[dj] = (p[dj] ?? 0) + sj * (h[dj] ?? 0)
    return f(p)
  }
  const out = new Float64Array(n * n)
  const f0 = f([...x])
  for (let i = 0; i < n; i++) {
    const hi = h[i] ?? 1
    // Diagonal: (f(x + 2h) - 2 f(x) + f(x - 2h)) / (4 h^2), consistent with the off-diagonal stencil.
    out[i * n + i] = (at(i, 1, i, 1) - 2 * f0 + at(i, -1, i, -1)) / (4 * hi * hi)
    for (let j = i + 1; j < n; j++) {
      const hj = h[j] ?? 1
      const v = (at(i, 1, j, 1) - at(i, 1, j, -1) - at(i, -1, j, 1) + at(i, -1, j, -1)) / (4 * hi * hj)
      out[i * n + j] = v
      out[j * n + i] = v
    }
  }
  return fromStorage(out, [n, n], 'float64')
}

/**
 * Finite-difference weights for the m-th derivative at x0 from values at arbitrary stencil points (Fornberg 1988):
 * the weights c with f^(m)(x0) ~ sum_k c_k f(x_k), exact for polynomials of degree below the stencil size.
 *
 * @param order - The derivative order m (< number of points).
 * @param stencil - The sample points (distinct).
 * @param x0 - Where the derivative is wanted (default 0).
 * @returns The weights, one per stencil point.
 * @throws MdmlError with code `value` when the order is not below the number of points or a point is repeated.
 * @concept finite-differences
 * @example
 * ```ts
 * import { finiteDifferenceWeights } from 'mdml-engine/numerics/differentiate'
 * finiteDifferenceWeights(2, [-1, 0, 1]) // => [1, -2, 1]
 * ```
 */
export function finiteDifferenceWeights(order: number, stencil: readonly number[], x0 = 0): number[] {
  const n = stencil.length
  if (order >= n) throw new MdmlError('value', 'finiteDifferenceWeights: need more points than the derivative order')
  if (new Set(stencil).size !== n)
    throw new MdmlError('value', 'finiteDifferenceWeights: the stencil points must be distinct')
  // c[j][k]: weight of point j for derivative k, built point by point.
  const c: number[][] = Array.from({ length: n }, () => new Array<number>(order + 1).fill(0))
  let c1 = 1
  let c4 = (stencil[0] ?? 0) - x0
  ;(c[0] as number[])[0] = 1
  for (let i = 1; i < n; i++) {
    const mn = Math.min(i, order)
    let c2 = 1
    const c5 = c4
    c4 = (stencil[i] ?? 0) - x0
    for (let j = 0; j < i; j++) {
      const c3 = (stencil[i] ?? 0) - (stencil[j] ?? 0)
      c2 *= c3
      const row = c[i] as number[]
      const prev = c[j] as number[]
      if (j === i - 1) {
        for (let k = mn; k >= 1; k--)
          row[k] = (c1 * (k * ((c[i - 1] as number[])[k - 1] ?? 0) - c5 * ((c[i - 1] as number[])[k] ?? 0))) / c2
        row[0] = (-c1 * c5 * ((c[i - 1] as number[])[0] ?? 0)) / c2
      }
      for (let k = mn; k >= 1; k--) prev[k] = (c4 * (prev[k] ?? 0) - k * (prev[k - 1] ?? 0)) / c3
      prev[0] = (c4 * (prev[0] ?? 0)) / c3
    }
    c1 = c2
  }
  return c.map((row) => {
    const w = row[order] ?? 0
    return Math.abs(w) < 1e-14 * n ? 0 : w
  })
}

/** A complex number for complex-step differentiation. */
export interface ComplexNumber {
  readonly re: number
  readonly im: number
}

/** Elementary complex arithmetic for writing functions differentiated by {@link complexStep}. */
export const complexOps: {
  readonly of: (re: number, im?: number) => ComplexNumber
  readonly add: (a: ComplexNumber, b: ComplexNumber | number) => ComplexNumber
  readonly sub: (a: ComplexNumber, b: ComplexNumber | number) => ComplexNumber
  readonly mul: (a: ComplexNumber, b: ComplexNumber | number) => ComplexNumber
  readonly div: (a: ComplexNumber, b: ComplexNumber | number) => ComplexNumber
  readonly exp: (a: ComplexNumber) => ComplexNumber
  readonly log: (a: ComplexNumber) => ComplexNumber
  readonly sin: (a: ComplexNumber) => ComplexNumber
  readonly cos: (a: ComplexNumber) => ComplexNumber
  readonly sqrt: (a: ComplexNumber) => ComplexNumber
  readonly pow: (a: ComplexNumber, p: number) => ComplexNumber
} = (() => {
  const of = (re: number, im = 0): ComplexNumber => ({ re, im })
  const lift = (b: ComplexNumber | number): ComplexNumber => (typeof b === 'number' ? of(b) : b)
  const exp = (a: ComplexNumber): ComplexNumber => {
    const r = Math.exp(a.re)
    return of(r * Math.cos(a.im), r * Math.sin(a.im))
  }
  const log = (a: ComplexNumber): ComplexNumber => of(Math.log(Math.hypot(a.re, a.im)), Math.atan2(a.im, a.re))
  const mul = (a: ComplexNumber, b0: ComplexNumber | number): ComplexNumber => {
    const b = lift(b0)
    return of(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
  }
  return {
    of,
    add: (a, b0) => {
      const b = lift(b0)
      return of(a.re + b.re, a.im + b.im)
    },
    sub: (a, b0) => {
      const b = lift(b0)
      return of(a.re - b.re, a.im - b.im)
    },
    mul,
    div: (a, b0) => {
      const b = lift(b0)
      const d = b.re * b.re + b.im * b.im
      return of((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d)
    },
    exp,
    log,
    sin: (a) => of(Math.sin(a.re) * Math.cosh(a.im), Math.cos(a.re) * Math.sinh(a.im)),
    cos: (a) => of(Math.cos(a.re) * Math.cosh(a.im), -Math.sin(a.re) * Math.sinh(a.im)),
    sqrt: (a) => {
      // The stable form: never subtract r and |re| (the imaginary part of a complex step is far below their rounding).
      const r = Math.hypot(a.re, a.im)
      if (r === 0) return of(0)
      if (a.re >= 0) {
        const t = Math.sqrt((r + a.re) / 2)
        return of(t, a.im / (2 * t))
      }
      const t = Math.sqrt((r - a.re) / 2)
      return of(Math.abs(a.im) / (2 * t), a.im < 0 ? -t : t)
    },
    pow: (a, p) => (a.re === 0 && a.im === 0 ? of(0) : exp(mul(log(a), p))),
  }
})()

/**
 * The complex-step derivative Im f(x + i h) / h (Squire and Trapp): no subtractive cancellation, so h can be tiny
 * (default 1e-20) and the derivative is accurate to machine precision. f must be real-analytic and written with
 * {@link complexOps}.
 *
 * @param f - The function on complex numbers.
 * @param x - The point.
 * @param h - The imaginary step.
 * @returns The derivative.
 * @example
 * ```ts
 * import { complexStep, complexOps as C } from 'mdml-engine/numerics/differentiate'
 * complexStep((z) => C.mul(z, C.exp(z)), 0) // => 1
 * ```
 */
export function complexStep(f: (z: ComplexNumber) => ComplexNumber, x: number, h = 1e-20): number {
  return f({ re: x, im: h }).im / h
}

/**
 * A dense finite-difference matrix D on n equally spaced points: D y approximates the first derivative (as
 * {@link gradient} with second-order edges) or the second derivative (central [1, -2, 1] / h^2 inside, second-order
 * one-sided [2, -5, 4, -1] / h^2 at the ends).
 *
 * @param n - Number of points.
 * @param options - Derivative order (1 or 2) and spacing h (default 1).
 * @returns The matrix `[n, n]`.
 * @example
 * ```ts
 * import { differenceMatrix } from 'mdml-engine/numerics/differentiate'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(differenceMatrix(3))[1] // => [-0.5, 0, 0.5]
 * ```
 */
export function differenceMatrix(n: number, options: { readonly order?: 1 | 2; readonly h?: number } = {}): Tensor {
  const order = options.order ?? 1
  const h = options.h ?? 1
  if (n < (order === 1 ? 3 : 4)) throw new MdmlError('value', 'differenceMatrix: too few points')
  const d = new Float64Array(n * n)
  const set = (r: number, c: number, v: number): void => {
    d[r * n + c] = v
  }
  if (order === 1) {
    set(0, 0, -1.5 / h)
    set(0, 1, 2 / h)
    set(0, 2, -0.5 / h)
    for (let i = 1; i < n - 1; i++) {
      set(i, i - 1, -0.5 / h)
      set(i, i + 1, 0.5 / h)
    }
    set(n - 1, n - 3, 0.5 / h)
    set(n - 1, n - 2, -2 / h)
    set(n - 1, n - 1, 1.5 / h)
  } else {
    const h2 = h * h
    ;[2, -5, 4, -1].forEach((v, k) => {
      set(0, k, v / h2)
      set(n - 1, n - 1 - k, v / h2)
    })
    for (let i = 1; i < n - 1; i++) {
      set(i, i - 1, 1 / h2)
      set(i, i, -2 / h2)
      set(i, i + 1, 1 / h2)
    }
  }
  return fromStorage(d, [n, n], 'float64')
}
