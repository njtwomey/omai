import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  concat,
  concreteOrThrow,
  cumsum,
  div,
  fromStorage,
  matmul,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  sliceAxis,
  sub,
  swapaxes,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { solve } from '../linalg/index.ts'
import { type Extrapolate, type PPoly, ppoly } from './piecewise.ts'

function knots(x: TensorLike, name: string): number[] {
  const xs = toFlat(concreteOrThrow(promoteFloat(x)[0] as Tensor, name))
  for (let i = 1; i < xs.length; i++) {
    if (!((xs[i] ?? 0) > (xs[i - 1] ?? 0))) throw new WhiteboxError('value', `${name}: x must be strictly increasing`)
  }
  return xs
}

/** Broadcast a per-interval vector `[m]` against values `[m, ...]`. */
function column(v: readonly number[], rank: number): Tensor {
  const t = fromStorage(Float64Array.from(v), [v.length], 'float64')
  return rank <= 1 ? t : (reshape(t, [v.length, ...new Array<number>(rank - 1).fill(1)]) as Tensor)
}

/**
 * The cubic Hermite interpolant with given values and first derivatives at the knots
 * (scipy.interpolate.CubicHermiteSpline), differentiable in both.
 *
 * @param x - Knots `[n]`, increasing.
 * @param y - Values `[n, ...]`.
 * @param dydx - Derivatives `[n, ...]`.
 * @param extrapolate - Extend the end pieces (default true).
 * @returns The piecewise cubic.
 * @example
 * ```ts
 * import { cubicHermiteSpline } from 'whitebox-engine/numerics/interpolate'
 * import { item } from 'whitebox-engine/tensor'
 * item(cubicHermiteSpline([0, 1], [0, 1], [0, 0]).evaluate(0.5)) // => 0.5
 * ```
 */
export function cubicHermiteSpline(
  x: TensorLike,
  y: TensorLike,
  dydx: TensorLike,
  extrapolate: Extrapolate = true,
): PPoly {
  const xs = knots(x, 'cubicHermiteSpline')
  const [yv, sv] = promoteFloat(y, dydx) as [Value, Value]
  const n = xs.length
  if (yv.shape[0] !== n || sv.shape[0] !== n)
    throw new WhiteboxError('shape', 'cubicHermiteSpline: x, y and dydx must have the same length')
  const h = column(
    xs.slice(1).map((v, i) => v - (xs[i] ?? 0)),
    yv.shape.length,
  )
  const y0 = sliceAxis(yv, 0, 0, n - 1)
  const y1 = sliceAxis(yv, 0, 1, n)
  const s0 = sliceAxis(sv, 0, 0, n - 1)
  const s1 = sliceAxis(sv, 0, 1, n)
  const slope = div(sub(y1, y0), h)
  const t = div(sub(add(s0, s1), mul(slope, 2)), h)
  const c3 = div(t, h)
  const c2 = sub(div(sub(slope, s0), h), t)
  const stack = (vs: Value[]): Value =>
    concat(
      vs.map((v) => reshape(v, [1, ...v.shape])),
      0,
    )
  return ppoly(stack([c3, c2, s0, y0]), xs, extrapolate)
}

/** A boundary condition: the order (1 or 2) of the derivative fixed and its value. */
export type SplineCondition = readonly [1 | 2, number]

/** Boundary conditions of {@link cubicSpline}. */
export type SplineBoundary =
  | 'not-a-knot'
  | 'natural'
  | 'clamped'
  | 'periodic'
  | readonly [SplineCondition, SplineCondition]

/**
 * The cubic interpolating spline (scipy.interpolate.CubicSpline): twice continuously differentiable, with not-a-knot
 * (default), natural (zero second derivatives), clamped (zero first derivatives), periodic, or explicit end
 * conditions. The knot derivatives solve a tridiagonal system that is linear in y, so the spline is differentiable in
 * the data and the query points.
 *
 * @param x - Knots `[n]`, increasing.
 * @param y - Values `[n, ...]`.
 * @param boundary - The end conditions.
 * @param extrapolate - Extend the end pieces (default true; 'periodic' for periodic splines, as scipy).
 * @returns The piecewise cubic.
 * @concept splines
 * @example
 * ```ts
 * import { cubicSpline } from 'whitebox-engine/numerics/interpolate'
 * import { item } from 'whitebox-engine/tensor'
 * // Not-a-knot reproduces cubics exactly.
 * item(cubicSpline([0, 1, 2, 3, 4], [0, 1, 8, 27, 64]).evaluate(2.5)) // => 15.625
 * ```
 */
export function cubicSpline(
  x: TensorLike,
  y: TensorLike,
  boundary: SplineBoundary = 'not-a-knot',
  extrapolate: Extrapolate = boundary === 'periodic' ? 'periodic' : true,
): PPoly {
  const xs = knots(x, 'cubicSpline')
  const [yv] = promoteFloat(y) as [Value]
  const n = xs.length
  if (n < 2) throw new WhiteboxError('value', 'cubicSpline: at least two points are needed')
  if (yv.shape[0] !== n) throw new WhiteboxError('shape', 'cubicSpline: x and y must have the same length')
  const dx = xs.slice(1).map((v, i) => v - (xs[i] ?? 0))
  if (boundary === 'periodic') {
    const yp = toFlat(primalOf(yv))
    const width = yp.length / n
    for (let j = 0; j < width; j++) {
      if (Math.abs((yp[j] ?? 0) - (yp[(n - 1) * width + j] ?? 0)) > 1e-12 * Math.max(1, Math.abs(yp[j] ?? 0))) {
        throw new WhiteboxError('value', 'cubicSpline: periodic boundary needs y[0] = y[n - 1]')
      }
    }
  }
  // The knot slopes s solve A s = B y + e; A and B depend only on x.
  const a = new Float64Array(n * n)
  const b = new Float64Array(n * n)
  const e = new Float64Array(n)
  const A = (r: number, c: number, v: number): void => {
    a[r * n + c] = (a[r * n + c] ?? 0) + v
  }
  // B y adds coef * slope_i, slope_i = (y_{i+1} - y_i) / dx_i.
  const S = (r: number, i: number, coef: number): void => {
    const w = coef / (dx[i] ?? 1)
    b[r * n + i + 1] = (b[r * n + i + 1] ?? 0) + w
    b[r * n + i] = (b[r * n + i] ?? 0) - w
  }
  const conditions: readonly [SplineCondition, SplineCondition] | 'not-a-knot' | 'periodic' =
    boundary === 'natural'
      ? [
          [2, 0],
          [2, 0],
        ]
      : boundary === 'clamped'
        ? [
            [1, 0],
            [1, 0],
          ]
        : boundary
  if (n === 2 || (n === 3 && conditions === 'not-a-knot')) {
    if (n === 2) {
      // A straight line (scipy's choice for two points with not-a-knot or periodic conditions).
      if (conditions === 'not-a-knot' || conditions === 'periodic') {
        S(0, 0, 1)
        S(1, 0, 1)
        A(0, 0, 1)
        A(1, 1, 1)
      } else fillEnds(conditions)
    } else {
      // The parabola through three points.
      A(0, 0, 1)
      A(0, 1, 1)
      S(0, 0, 2)
      A(1, 0, dx[1] ?? 0)
      A(1, 1, 2 * ((dx[0] ?? 0) + (dx[1] ?? 0)))
      A(1, 2, dx[0] ?? 0)
      S(1, 0, 3 * (dx[1] ?? 0))
      S(1, 1, 3 * (dx[0] ?? 0))
      A(2, 1, 1)
      A(2, 2, 1)
      S(2, 1, 2)
    }
  } else if (conditions === 'periodic') {
    // Unknowns s_0..s_{n-2}, with s_{n-1} = s_0; row n-1 states that equality.
    const p = n - 1
    for (let i = 0; i < p; i++) {
      const left = (i - 1 + p) % p
      const hl = dx[left] ?? 0
      const hr = dx[i] ?? 0
      A(i, left, hr)
      A(i, i, 2 * (hl + hr))
      A(i, (i + 1) % p, hl)
      S(i, left, 3 * hr)
      S(i, i, 3 * hl)
    }
    A(p, p, 1)
    A(p, 0, -1)
  } else {
    for (let i = 1; i < n - 1; i++) {
      A(i, i - 1, dx[i] ?? 0)
      A(i, i, 2 * ((dx[i - 1] ?? 0) + (dx[i] ?? 0)))
      A(i, i + 1, dx[i - 1] ?? 0)
      S(i, i - 1, 3 * (dx[i] ?? 0))
      S(i, i, 3 * (dx[i - 1] ?? 0))
    }
    if (conditions === 'not-a-knot') {
      const d0 = (xs[2] ?? 0) - (xs[0] ?? 0)
      const h0 = dx[0] ?? 0
      const h1 = dx[1] ?? 0
      A(0, 0, h1)
      A(0, 1, d0)
      S(0, 0, ((h0 + 2 * d0) * h1) / d0)
      S(0, 1, (h0 * h0) / d0)
      const d1 = (xs[n - 1] ?? 0) - (xs[n - 3] ?? 0)
      const g0 = dx[n - 3] ?? 0
      const g1 = dx[n - 2] ?? 0
      A(n - 1, n - 1, g0)
      A(n - 1, n - 2, d1)
      S(n - 1, n - 3, (g1 * g1) / d1)
      S(n - 1, n - 2, ((2 * d1 + g1) * g0) / d1)
    } else fillEnds(conditions)
  }
  function fillEnds([start, end]: readonly [SplineCondition, SplineCondition]): void {
    const h0 = dx[0] ?? 0
    const hn = dx[n - 2] ?? 0
    if (start[0] === 1) {
      A(0, 0, 1)
      e[0] = start[1]
    } else {
      A(0, 0, 2 * h0)
      A(0, 1, h0)
      S(0, 0, 3 * h0)
      e[0] = -0.5 * start[1] * h0 * h0
    }
    if (end[0] === 1) {
      A(n - 1, n - 1, 1)
      e[n - 1] = end[1]
    } else {
      A(n - 1, n - 1, 2 * hn)
      A(n - 1, n - 2, hn)
      S(n - 1, n - 2, 3 * hn)
      e[n - 1] = 0.5 * end[1] * hn * hn
    }
  }
  const am = fromStorage(a, [n, n], 'float64')
  const map = solve(am, fromStorage(b, [n, n], 'float64'))
  const offset = toFlat(solve(am, fromStorage(e, [n], 'float64')))
  let slopes = matmul(map, yv)
  if (offset.some((v) => v !== 0)) slopes = add(slopes, column(offset, yv.shape.length))
  return cubicHermiteSpline(xs, yv, slopes, extrapolate)
}

/**
 * A B-spline (scipy.interpolate.BSpline): sum_i c_i B_{i,k}(x) on knots t. The basis is built by the Cox-de Boor
 * recursion with operations, so the spline is differentiable in its coefficients and the query points.
 */
export interface BSpline {
  readonly knots: Tensor
  /** Coefficients `[n, ...]`, n = len(t) - k - 1. */
  readonly coefficients: Value
  readonly degree: number
  readonly extrapolate: boolean
  readonly evaluate: (x: TensorLike, nu?: number) => Value
  /** The basis values `[q, n]` at the query points. */
  readonly designMatrix: (x: TensorLike) => Value
  readonly derivative: (nu?: number) => BSpline
  readonly antiderivative: (nu?: number) => BSpline
  readonly integrate: (a: number, b: number) => Value
}

/**
 * A B-spline from knots, coefficients and degree.
 *
 * @param t - Knots, non-decreasing, length n + k + 1.
 * @param c - Coefficients `[n, ...]`.
 * @param k - Degree.
 * @param extrapolate - Extend the end pieces (default true; otherwise NaN outside [t_k, t_n]).
 * @returns The spline.
 * @concept splines
 * @example
 * ```ts
 * import { bspline } from 'whitebox-engine/numerics/interpolate'
 * import { item } from 'whitebox-engine/tensor'
 * // The quadratic B-spline basis function on knots 0..3 peaks at 0.75.
 * item(bspline([0, 1, 2, 3], [1], 2).evaluate(1.5)) // => 0.75
 * ```
 */
export function bspline(t: TensorLike, c: TensorLike, k: number, extrapolate = true): BSpline {
  const ts = toFlat(concreteOrThrow(promoteFloat(t)[0] as Tensor, 'bspline knots'))
  const [cv] = promoteFloat(c) as [Value]
  const n = ts.length - k - 1
  if (n < 1 || cv.shape[0] !== n)
    throw new WhiteboxError('shape', `bspline: ${ts.length} knots and degree ${k} need ${Math.max(n, 0)} coefficients`)
  for (let i = 1; i < ts.length; i++) {
    if ((ts[i] ?? 0) < (ts[i - 1] ?? 0)) throw new WhiteboxError('value', 'bspline: knots must be non-decreasing')
  }
  const lo = ts[k] ?? 0
  const hi = ts[n] ?? 0

  const designMatrix = (query: TensorLike): Value => {
    const [q] = promoteFloat(query) as [Value]
    const qf = reshape(q, [-1])
    const qp = toFlat(primalOf(qf))
    const L = ts.length - 1
    // Degree 0: the indicator of the knot span holding x (the end spans for points outside, so the end pieces
    // extend).
    const zero = new Float64Array(qp.length * L)
    qp.forEach((v, r) => {
      let mu = k
      while (mu < n - 1 && v >= (ts[mu + 1] ?? 0)) mu++
      zero[r * L + mu] = 1
    })
    let basis: Value = fromStorage(zero, [qp.length, L], 'float64')
    const xcol = reshape(qf, [qp.length, 1])
    for (let p = 1; p <= k; p++) {
      const count = L - p
      const left = new Float64Array(count)
      const leftInv = new Float64Array(count)
      const right = new Float64Array(count)
      const rightInv = new Float64Array(count)
      for (let i = 0; i < count; i++) {
        const d1 = (ts[i + p] ?? 0) - (ts[i] ?? 0)
        const d2 = (ts[i + p + 1] ?? 0) - (ts[i + 1] ?? 0)
        left[i] = ts[i] ?? 0
        leftInv[i] = d1 === 0 ? 0 : 1 / d1
        right[i] = ts[i + p + 1] ?? 0
        rightInv[i] = d2 === 0 ? 0 : 1 / d2
      }
      const row = (v: Float64Array): Tensor => fromStorage(v, [1, count], 'float64')
      const omegaLeft = mul(sub(xcol, row(left)), row(leftInv))
      const omegaRight = mul(sub(row(right), xcol), row(rightInv))
      basis = add(mul(omegaLeft, sliceAxis(basis, 1, 0, count)), mul(omegaRight, sliceAxis(basis, 1, 1, count + 1)))
    }
    return sliceAxis(basis, 1, 0, n)
  }

  const evaluate = (query: TensorLike, nu = 0): Value => {
    if (nu > 0) return derivative(nu).evaluate(query)
    if (nu < 0) return antiderivative(-nu).evaluate(query)
    const [q] = promoteFloat(query) as [Value]
    const flat = cv.shape.length === 1 ? reshape(cv, [n, 1]) : reshape(cv, [n, -1])
    let out: Value = matmul(designMatrix(q), flat)
    if (!extrapolate) {
      const qp = toFlat(primalOf(reshape(q, [-1])))
      const outside = qp.map((v) => v < lo || v > hi)
      if (outside.some(Boolean))
        out = where(fromStorage(Uint8Array.from(outside, Number), [qp.length, 1], 'bool'), Number.NaN, out)
    }
    return reshape(out, [...q.shape, ...cv.shape.slice(1)])
  }

  const derivative = (nu = 1): BSpline => {
    if (nu === 0) return self
    if (k === 0) throw new WhiteboxError('value', 'bspline derivative: degree 0 has no derivative')
    // c'_i = k (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1}) on knots t[1:-1], degree k - 1.
    const spans = Array.from({ length: n - 1 }, (_, i) => {
      const d = (ts[i + k + 1] ?? 0) - (ts[i + 1] ?? 0)
      return d === 0 ? 0 : k / d
    })
    const dc = mul(sub(sliceAxis(cv, 0, 1, n), sliceAxis(cv, 0, 0, n - 1)), column(spans, cv.shape.length))
    return bspline(ts.slice(1, -1), dc, k - 1, extrapolate).derivative(nu - 1)
  }

  const antiderivative = (nu = 1): BSpline => {
    if (nu === 0) return self
    // scipy: t -> [t_0, t, t_last], c -> cumsum(c_i (t_{i+k+1} - t_i)) / (k + 1).
    const widths = Array.from({ length: n }, (_, i) => ((ts[i + k + 1] ?? 0) - (ts[i] ?? 0)) / (k + 1))
    const ci = cumsum(mul(cv, column(widths, cv.shape.length)), 0)
    const padded = concat([mul(sliceAxis(ci, 0, 0, 1), 0), ci], 0)
    return bspline([ts[0] ?? 0, ...ts, ts[ts.length - 1] ?? 0], padded, k + 1, extrapolate).antiderivative(nu - 1)
  }

  const integrate = (a: number, b: number): Value => {
    const anti = antiderivative(1)
    const ends = anti.evaluate([a, b])
    return sub(
      reshape(sliceAxis(ends, 0, 1, 2), ends.shape.slice(1)),
      reshape(sliceAxis(ends, 0, 0, 1), ends.shape.slice(1)),
    )
  }

  const self: BSpline = {
    knots: fromStorage(Float64Array.from(ts), [ts.length], 'float64'),
    coefficients: cv,
    degree: k,
    extrapolate,
    evaluate,
    designMatrix,
    derivative,
    antiderivative,
    integrate,
  }
  return self
}

/** Not-a-knot knots for interpolation of odd degree k at x (scipy's default for make_interp_spline). */
export function notAKnot(x: readonly number[], k: number): number[] {
  if (k % 2 !== 1) throw new WhiteboxError('value', 'notAKnot: odd degrees only')
  const m = (k - 1) / 2
  const inner = x.slice(m + 1, x.length - m - 1)
  return [...new Array<number>(k + 1).fill(x[0] ?? 0), ...inner, ...new Array<number>(k + 1).fill(x[x.length - 1] ?? 0)]
}

/**
 * The interpolating B-spline of odd degree k with not-a-knot conditions (scipy.interpolate.make_interp_spline with its
 * defaults; k = 3 is the not-a-knot cubic spline). The coefficients solve the collocation system, linear in y.
 *
 * @param x - Points `[n]`, increasing.
 * @param y - Values `[n, ...]`.
 * @param k - Odd degree (default 3).
 * @param knots - Optional knots (default not-a-knot).
 * @returns The spline.
 * @example
 * ```ts
 * import { makeInterpSpline } from 'whitebox-engine/numerics/interpolate'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(makeInterpSpline([0, 1, 2, 3, 4], [0, 1, 8, 27, 64]).evaluate(2.5)) - 15.625) < 1e-12 // => true
 * ```
 */
export function makeInterpSpline(x: TensorLike, y: TensorLike, k = 3, knotVector?: readonly number[]): BSpline {
  const xs = knots(x, 'makeInterpSpline')
  const [yv] = promoteFloat(y) as [Value]
  if (xs.length < k + 1)
    throw new WhiteboxError('value', `makeInterpSpline: degree ${k} needs at least ${k + 1} points`)
  const t = knotVector ?? notAKnot(xs, k)
  const n = t.length - k - 1
  const probe = bspline(t, fromStorage(new Float64Array(n), [n], 'float64'), k)
  const collocation = probe.designMatrix(xs)
  return bspline(t, solve(collocation, yv), k)
}

/**
 * The least-squares spline with given knots (scipy.interpolate.make_lsq_spline): coefficients minimising
 * sum (w_i (y_i - s(x_i)))^2 (weights enter squared, as scipy and FITPACK), by the normal equations; linear (so
 * differentiable) in y.
 *
 * @param x - Points `[m]`.
 * @param y - Values `[m, ...]`.
 * @param t - Knots (with k + 1 repeated end knots, satisfying the Schoenberg-Whitney conditions).
 * @param k - Degree (default 3).
 * @param w - Optional weights `[m]`.
 * @returns The spline.
 * @example
 * ```ts
 * import { makeLsqSpline } from 'whitebox-engine/numerics/interpolate'
 * import { item } from 'whitebox-engine/tensor'
 * const xs = [0, 0.5, 1, 1.5, 2, 2.5, 3]
 * Math.abs(item(makeLsqSpline(xs, xs.map((v) => v * v), [0, 0, 0, 0, 1.5, 3, 3, 3, 3]).evaluate(1.2)) - 1.44) < 1e-12 // => true
 * ```
 */
export function makeLsqSpline(x: TensorLike, y: TensorLike, t: readonly number[], k = 3, w?: TensorLike): BSpline {
  const [xv] = promoteFloat(x) as [Value]
  const [yv] = promoteFloat(y) as [Value]
  const n = t.length - k - 1
  const probe = bspline(t, fromStorage(new Float64Array(n), [n], 'float64'), k)
  const basis = probe.designMatrix(primalOf(xv))
  const weights = w === undefined ? undefined : (promoteFloat(w)[0] as Value)
  const weighted = weights === undefined ? basis : mul(basis, reshape(mul(weights, weights), [-1, 1]))
  const bt = swapaxes(weighted, 0, 1)
  const normal = matmul(bt, basis)
  return bspline(t, solve(normal, matmul(bt, yv)), k)
}
