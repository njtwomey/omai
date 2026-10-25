/**
 * Interpolation on grids and of scattered data, numpy and scipy.interpolate style: `interp` (piecewise linear),
 * `interp1d` with nearest, previous, next and spline kinds, `regularGridInterpolator` for rectilinear grids in any
 * dimension, and `griddata` for scattered sites (nearest, Delaunay-linear and natural-neighbour), all linear in the
 * data values.
 */

import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  concat,
  concreteOrThrow,
  fromStorage,
  matmul,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  sliceAxis,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { barycentricCoordinates, delaunay, type Point2 } from '../geometry/index.ts'
import { naturalNeighbourWeights } from './scattered.ts'
import { makeInterpSpline } from './splines.ts'

function axisPoints(x: TensorLike, name: string): number[] {
  const xs = toFlat(concreteOrThrow(promoteFloat(x)[0] as Tensor, name))
  for (let i = 1; i < xs.length; i++) {
    if (!((xs[i] ?? 0) > (xs[i - 1] ?? 0)))
      throw new WhiteboxError('value', `${name}: points must be strictly increasing`)
  }
  return xs
}

/** The interval index i with x_i <= v < x_{i+1}, clamped to [0, n - 2]. */
function interval(xs: readonly number[], v: number): number {
  let lo = 0
  let hi = xs.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (v >= (xs[mid] ?? 0)) lo = mid
    else hi = mid
  }
  return lo
}

/** Interpolation methods along one axis: weights `[q, n]` that are linear in the data. */
export type GridMethod =
  | 'linear'
  | 'nearest'
  | 'previous'
  | 'next'
  | 'zero'
  | 'slinear'
  | 'quadratic'
  | 'cubic'
  | 'quintic'

const SPLINE_DEGREE: Partial<Record<GridMethod, number>> = { zero: 0, slinear: 1, quadratic: 2, cubic: 3, quintic: 5 }

/**
 * The weights of one-dimensional interpolation at query points: row r holds the coefficients of the data in the
 * interpolant at q_r (hat functions for linear, indicators for nearest, previous and next, cardinal splines otherwise).
 * Linear and spline weights are differentiable in the query points.
 */
function axisWeights(xs: readonly number[], q: Value, method: GridMethod): Value {
  const n = xs.length
  const qp = toFlat(primalOf(q))
  const m = qp.length
  const degree = SPLINE_DEGREE[method]
  if (degree !== undefined) {
    if (n <= degree)
      throw new WhiteboxError('value', `${method} interpolation needs more than ${degree} points per axis`)
    const identity = new Float64Array(n * n)
    for (let i = 0; i < n; i++) identity[i * n + i] = 1
    return reshape(makeInterpSpline(xs, fromStorage(identity, [n, n], 'float64'), degree).evaluate(q), [m, n])
  }
  if (method === 'linear') {
    const left = new Float64Array(m * n)
    const right = new Float64Array(m * n)
    const starts = new Float64Array(m)
    const inverse = new Float64Array(m)
    qp.forEach((v, r) => {
      const i = n === 1 ? 0 : interval(xs, v)
      left[r * n + i] = 1
      if (n > 1) {
        right[r * n + i + 1] = 1
        starts[r] = xs[i] ?? 0
        inverse[r] = 1 / ((xs[i + 1] ?? 0) - (xs[i] ?? 0))
      }
    })
    if (n === 1) return fromStorage(left, [m, n], 'float64')
    const t = reshape(
      mul(sub(reshape(q, [m]), fromStorage(starts, [m], 'float64')), fromStorage(inverse, [m], 'float64')),
      [m, 1],
    )
    return add(mul(fromStorage(left, [m, n], 'float64'), sub(1, t)), mul(fromStorage(right, [m, n], 'float64'), t))
  }
  const w = new Float64Array(m * n)
  qp.forEach((v, r) => {
    let j: number
    if (method === 'nearest') {
      const i = n === 1 ? 0 : interval(xs, v)
      // Halfway points round down, as scipy's interp1d 'nearest'.
      j = n === 1 ? 0 : v - (xs[i] ?? 0) <= (xs[i + 1] ?? 0) - v ? i : i + 1
    } else if (method === 'previous') {
      j = 0
      while (j + 1 < n && (xs[j + 1] ?? 0) <= v) j++
    } else {
      j = n - 1
      while (j > 0 && (xs[j - 1] ?? 0) >= v) j--
    }
    w[r * n + j] = 1
  })
  return fromStorage(w, [m, n], 'float64')
}

/**
 * Piecewise-linear interpolation of 1-D data (numpy.interp): values outside the range are fp[0] and fp[-1] (or
 * `left` and `right`), and with a period the coordinates wrap. Differentiable in x and fp.
 *
 * @param x - Query points.
 * @param xp - Increasing data coordinates.
 * @param fp - Data values.
 * @param options - Values beyond the ends and an optional period.
 * @returns Interpolated values, the shape of x.
 * @example
 * ```ts
 * import { interp } from 'whitebox-engine/numerics/interpolate'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(interp([-1, 0.5, 1.5, 9], [0, 1, 2], [10, 20, 0])) // => [10, 15, 10, 0]
 * ```
 */
export function interp(
  x: TensorLike,
  xp: TensorLike,
  fp: TensorLike,
  options: { readonly left?: number; readonly right?: number; readonly period?: number } = {},
): Value {
  const [q] = promoteFloat(x) as [Value]
  const [f] = promoteFloat(fp) as [Value]
  let xs = toFlat(concreteOrThrow(promoteFloat(xp)[0] as Tensor, 'interp'))
  let fv = f
  let qf = reshape(q, [-1])
  if (options.period !== undefined) {
    // Wrap the data into [0, period), sort it, and pad one period-shifted point at each end.
    const period = options.period
    const wrapped = xs.map((v) => ((v % period) + period) % period)
    const order = wrapped.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v)
    const idx = order.map((o) => o.i)
    const take = (i: number): Value => sliceAxis(f, 0, i, i + 1)
    const last = idx[idx.length - 1] ?? 0
    const first = idx[0] ?? 0
    xs = [(order[order.length - 1]?.v ?? 0) - period, ...order.map((o) => o.v), (order[0]?.v ?? 0) + period]
    fv = reshape(concat([take(last), ...idx.map(take), take(first)], 0), [-1])
    const qp = toFlat(primalOf(qf))
    qf = add(
      qf,
      fromStorage(
        Float64Array.from(qp, (v) => -Math.floor(v / period) * period),
        [qp.length],
        'float64',
      ),
    )
  }
  const w = axisWeights(xs, qf, 'linear')
  let out: Value = matmul(w, fv)
  if (options.period === undefined) {
    const qp = toFlat(primalOf(qf))
    const lo = xs[0] ?? 0
    const hi = xs[xs.length - 1] ?? 0
    const below = qp.map((v) => v < lo)
    const above = qp.map((v) => v > hi)
    const at = (i: number): Value => reshape(sliceAxis(fv, 0, i, i + 1), [])
    if (below.some(Boolean))
      out = where(fromStorage(Uint8Array.from(below, Number), [qp.length], 'bool'), options.left ?? at(0), out)
    if (above.some(Boolean))
      out = where(
        fromStorage(Uint8Array.from(above, Number), [qp.length], 'bool'),
        options.right ?? at(xs.length - 1),
        out,
      )
  }
  return reshape(out, q.shape)
}

/** Options of {@link interp1d}. */
export interface Interp1dOptions {
  readonly kind?: GridMethod
  /** Raise outside the data range (default true unless a fill value is given). */
  readonly boundsError?: boolean
  /** The value outside the range (default NaN), or `extrapolate`. */
  readonly fillValue?: number | 'extrapolate'
}

/**
 * One-dimensional interpolation of several kinds (scipy.interpolate.interp1d): linear, nearest, previous, next, and
 * splines of degree 0 (`zero`), 1 (`slinear`), 2 (`quadratic`), 3 (`cubic`) and 5. Values are linear in y along axis 0.
 *
 * @param x - Increasing coordinates `[n]`.
 * @param y - Values `[n, ...]`.
 * @param options - Kind, bounds handling and fill value.
 * @returns The interpolating function.
 * @example
 * ```ts
 * import { interp1d } from 'whitebox-engine/numerics/interpolate'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(interp1d([0, 1, 2], [0, 10, 20], { kind: 'previous' })([0.5, 1.5])) // => [0, 10]
 * ```
 */
export function interp1d(x: TensorLike, y: TensorLike, options: Interp1dOptions = {}): (query: TensorLike) => Value {
  const xs = axisPoints(x, 'interp1d')
  const [yv] = promoteFloat(y) as [Value]
  const kind = options.kind ?? 'linear'
  const boundsError = options.boundsError ?? options.fillValue === undefined
  const lo = xs[0] ?? 0
  const hi = xs[xs.length - 1] ?? 0
  const trailing = yv.shape.slice(1)
  const flat = reshape(yv, [xs.length, -1])
  return (query) => {
    const [q] = promoteFloat(query) as [Value]
    const qf = reshape(q, [-1])
    const qp = toFlat(primalOf(qf))
    const outside = qp.map((v) => v < lo || v > hi)
    if (boundsError && outside.some(Boolean))
      throw new WhiteboxError('value', 'interp1d: a query point is outside the data range')
    let out: Value = matmul(axisWeights(xs, qf, kind), flat)
    if (options.fillValue !== 'extrapolate' && outside.some(Boolean)) {
      out = where(
        fromStorage(Uint8Array.from(outside, Number), [qp.length, 1], 'bool'),
        options.fillValue ?? Number.NaN,
        out,
      )
    }
    return reshape(out, [...q.shape, ...trailing])
  }
}

/** Options of {@link regularGridInterpolator}. */
export interface RegularGridOptions {
  readonly method?: Exclude<GridMethod, 'previous' | 'next' | 'zero' | 'quadratic'>
  /** Raise outside the grid (default true). */
  readonly boundsError?: boolean
  /** The value outside the grid when not raising (default NaN); null extrapolates. */
  readonly fillValue?: number | null
}

/**
 * Interpolation on a rectilinear grid in any dimension (scipy.interpolate.RegularGridInterpolator): multilinear,
 * nearest-neighbour, or tensor-product splines (slinear, cubic, quintic), all linear in the grid values. Each axis
 * contributes weights `[q, n_k]`, and the values are contracted with them axis by axis, so the result is
 * differentiable in the values and (for linear and spline methods) the query points.
 *
 * @param points - The grid coordinates of each axis.
 * @param values - Grid values `[n_1, ..., n_d, ...]`.
 * @param options - Method, bounds handling and fill value.
 * @returns The interpolating function on query points `[q, d]` (or `[d]`).
 * @concept grid-interpolation
 * @example
 * ```ts
 * import { regularGridInterpolator } from 'whitebox-engine/numerics/interpolate'
 * import { toArray } from 'whitebox-engine/tensor'
 * // f(x, y) = x + 10 y is reproduced exactly by multilinear interpolation.
 * const f = regularGridInterpolator([[0, 1, 2], [0, 1]], [[0, 10], [1, 11], [2, 12]])
 * toArray(f([[0.5, 0.5], [1.5, 0.25]])) // => [5.5, 4]
 * ```
 */
export function regularGridInterpolator(
  points: readonly TensorLike[],
  values: TensorLike,
  options: RegularGridOptions = {},
): (query: TensorLike) => Value {
  const axes = points.map((p, k) => axisPoints(p, `regularGridInterpolator axis ${k}`))
  const [v] = promoteFloat(values) as [Value]
  const d = axes.length
  axes.forEach((xs, k) => {
    if (v.shape[k] !== xs.length)
      throw new WhiteboxError(
        'shape',
        `regularGridInterpolator: axis ${k} has ${xs.length} points but the values have ${v.shape[k]}`,
      )
  })
  const method = options.method ?? 'linear'
  const boundsError = options.boundsError ?? true
  const trailing = v.shape.slice(d)
  return (query) => {
    const [q0] = promoteFloat(query) as [Value]
    const single = q0.shape.length === 1
    const q = single ? reshape(q0, [1, d]) : q0
    if (q.shape[q.shape.length - 1] !== d)
      throw new WhiteboxError('shape', `regularGridInterpolator: query points must have ${d} coordinates`)
    const m = q.shape[0] ?? 0
    const qp = toFlat(primalOf(q))
    const outside = Array.from({ length: m }, (_, r) =>
      axes.some((xs, k) => (qp[r * d + k] ?? 0) < (xs[0] ?? 0) || (qp[r * d + k] ?? 0) > (xs[xs.length - 1] ?? 0)),
    )
    if (boundsError && outside.some(Boolean))
      throw new WhiteboxError('value', 'regularGridInterpolator: a query point is outside the grid')
    // Contract axis 0 by matrix product, then each further axis by a per-point weighted sum.
    let acc: Value = matmul(
      axisWeights(axes[0] as number[], reshape(sliceAxis(q, 1, 0, 1), [m]), method),
      reshape(v, [axes[0]?.length ?? 0, -1]),
    )
    for (let k = 1; k < d; k++) {
      const nk = axes[k]?.length ?? 0
      const w = axisWeights(axes[k] as number[], reshape(sliceAxis(q, 1, k, k + 1), [m]), method)
      acc = sum(mul(reshape(acc, [m, nk, -1]), reshape(w, [m, nk, 1])), 1)
    }
    let out: Value = acc
    if (options.fillValue !== null && outside.some(Boolean)) {
      out = where(fromStorage(Uint8Array.from(outside, Number), [m, 1], 'bool'), options.fillValue ?? Number.NaN, out)
    }
    const shaped = reshape(out, [m, ...trailing])
    return single ? reshape(shaped, trailing) : shaped
  }
}

/**
 * Interpolation of scattered data at query points (scipy.interpolate.griddata): `nearest` (the closest data site, any
 * dimension), `linear` (barycentric interpolation on the Delaunay triangulation, planar data, NaN outside the convex
 * hull) or `natural` (Sibson's natural-neighbour interpolation, planar). All are linear in the values.
 *
 * @param points - Data sites `[n, d]`.
 * @param values - Values `[n]`.
 * @param query - Query points `[q, d]`.
 * @param method - The method (default linear).
 * @param fillValue - The value outside the hull (default NaN).
 * @returns Interpolated values `[q]`.
 * @example
 * ```ts
 * import { griddata } from 'whitebox-engine/numerics/interpolate'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(griddata([[0, 0], [1, 0], [0, 1], [1, 1]], [0, 1, 2, 3], [[0.5, 0.5], [0.25, 0.5]])) // => [1.5, 1.25]
 * ```
 */
export function griddata(
  points: TensorLike,
  values: TensorLike,
  query: TensorLike,
  method: 'nearest' | 'linear' | 'natural' = 'linear',
  fillValue: number = Number.NaN,
): Value {
  const sites = concreteOrThrow(promoteFloat(points)[0] as Tensor, 'griddata')
  const [v] = promoteFloat(values) as [Value]
  const q = concreteOrThrow(promoteFloat(query)[0] as Tensor, 'griddata query')
  const n = sites.shape[0] ?? 0
  const d = sites.shape[1] ?? 1
  const m = q.shape[0] ?? 0
  const sf = toFlat(sites)
  const qf = toFlat(q)
  if (method === 'natural') {
    const w = naturalNeighbourWeights(sites, q)
    return where(
      fromStorage(
        Uint8Array.from({ length: m }, (_, r) => (Number.isNaN(w.data[r * n] ?? 0) ? 1 : 0)),
        [m],
        'bool',
      ),
      fillValue,
      matmul(
        where(
          fromStorage(
            Uint8Array.from(w.data, (x) => (Number.isNaN(x) ? 1 : 0)),
            [m, n],
            'bool',
          ),
          0,
          w,
        ),
        v,
      ),
    )
  }
  const weights = new Float64Array(m * n)
  const outside = new Uint8Array(m)
  if (method === 'nearest') {
    for (let r = 0; r < m; r++) {
      let best = 0
      let bestD = Number.POSITIVE_INFINITY
      for (let i = 0; i < n; i++) {
        let dist = 0
        for (let k = 0; k < d; k++) dist += ((qf[r * d + k] ?? 0) - (sf[i * d + k] ?? 0)) ** 2
        if (dist < bestD) {
          bestD = dist
          best = i
        }
      }
      weights[r * n + best] = 1
    }
  } else {
    if (d !== 2) throw new WhiteboxError('value', 'griddata: linear interpolation needs planar points')
    const tri = delaunay(sites)
    for (let r = 0; r < m; r++) {
      const p: Point2 = [qf[2 * r] ?? 0, qf[2 * r + 1] ?? 0]
      const s = tri.findSimplex(p)
      if (s === -1) {
        outside[r] = 1
        continue
      }
      const t = tri.simplices[s] as readonly [number, number, number]
      const corner = (i: number): Point2 => tri.points[i] as Point2
      const l = barycentricCoordinates([corner(t[0]), corner(t[1]), corner(t[2])], p)
      t.forEach((vertex, j) => {
        weights[r * n + vertex] = l[j] ?? 0
      })
    }
  }
  const out = matmul(fromStorage(weights, [m, n], 'float64'), v)
  return outside.some((o) => o === 1) ? where(fromStorage(outside, [m], 'bool'), fillValue, out) : out
}
