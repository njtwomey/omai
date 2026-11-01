/**
 * The B-spline basis as differentiable tensor operations: the Cox-de Boor recurrence (any degree) and its derivatives
 * written over `tensor` ops, so gradients flow to the evaluation points and to the knot positions alike. A function of
 * `x` and `knots` only through comparisons (which interval a point falls in) and smooth arithmetic, the basis is
 * differentiable in the knots wherever the spline is (free-knot splines, learnable grids). Also the knot vector
 * helpers: extending interior knots with repeated boundary knots, the Greville abscissae, and the monotone
 * parameterisation of interior knots through a softmax of gaps that free-knot layers train. Everything is batched over
 * a leading feature axis: knots `[f, m]` with points `[n, f]` give a basis `[n, f, K]`, one knot vector per feature.
 */

import { MdmlError } from '../../base/index.ts'
import {
  add,
  asTensor,
  broadcastTo,
  clip,
  concat,
  cumsum,
  div,
  equal,
  full,
  greater,
  greaterEqual,
  less,
  logicalAnd,
  logicalOr,
  mul,
  notEqual,
  reshape,
  sliceAxis,
  softmax,
  sub,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'

/** What a basis does for points outside the knot range: hold the boundary value, or continue linearly. */
export type Outside = 'clamp' | 'linear'

/** Options of {@link bsplineBasis} and {@link bsplineBasisDerivative}. */
export interface BasisOptions {
  /** Points beyond the boundary knots: `clamp` evaluates at the nearest boundary (default), `linear` continues with the boundary tangent. */
  readonly outside?: Outside
}

const lastAxis = (v: Value): number => v.shape.length - 1

/** The points as a column the knots broadcast against: `[n, 1]` for one knot vector, `[n, f, 1]` for `f` of them. */
const column = (x: TensorLike, batched: boolean): Value => {
  const v = asTensor(x)
  if (!batched) {
    if (v.shape.length === 1) return reshape(v, [v.shape[0] ?? 0, 1])
    if (v.shape.length === 2 && v.shape[1] === 1) return v
    throw new MdmlError('shape', `bspline: expected points [n] or [n, 1], got [${v.shape.join(', ')}]`)
  }
  if (v.shape.length !== 2) throw new MdmlError('shape', `bspline: expected points [n, f], got [${v.shape.join(', ')}]`)
  return reshape(v, [v.shape[0] ?? 0, v.shape[1] ?? 0, 1])
}

const checkKnots = (who: string, t: Value, degree: number): number => {
  if (!(Number.isInteger(degree) && degree >= 0)) throw new MdmlError('value', `${who}: degree must be an integer >= 0`)
  if (t.shape.length < 1 || t.shape.length > 2 || (t.shape[lastAxis(t)] ?? 0) < degree + 2)
    throw new MdmlError('shape', `${who}: need a knot vector [m] or [f, m] with m >= degree + 2`)
  return t.shape[lastAxis(t)] ?? 0
}

/** The knots `[from, to)` along the last axis. */
const knotsAt = (t: Value, from: number, to: number): Value => sliceAxis(t, lastAxis(t), from, to)

/** Division that returns 0 where the denominator is not positive (the 0/0 := 0 convention of repeated knots). */
const safeRatio = (numerator: Value, denominator: Value): Value => {
  const ok = greater(denominator, 0)
  return where(ok, div(numerator, where(ok, denominator, 1)), 0)
}

/** The degree-0 basis `[n, m - 1]`: the indicator of each knot span, the last non-empty span closed on the right. */
function indicator(x: Value, t: Value, m: number): Value {
  const left = knotsAt(t, 0, m - 1)
  const right = knotsAt(t, 1, m)
  const last = knotsAt(t, m - 1, m)
  const inside = logicalAnd(greaterEqual(x, left), less(x, right))
  const closing = logicalAnd(logicalAnd(equal(right, last), less(left, right)), equal(x, last))
  return where(logicalOr(inside, closing), 1, 0)
}

/**
 * The B-spline basis functions of a knot vector at points, by the Cox-de Boor recurrence over tensor operations. The
 * knots may be traced: the gradient with respect to a knot is the exact derivative of the spline basis in that knot
 * (the indicator of the span is locally constant). Repeated knots follow the usual convention that a ratio with a zero
 * denominator is zero, and the last non-empty span is closed on the right, so the basis sums to one on the closed
 * range between the first and last non-repeated knot. With knots `[f, m]` (one vector per feature) the points are
 * `[n, f]` and the basis `[n, f, K]`.
 *
 * @param x - The points `[n]` (or `[n, 1]`), or `[n, f]` for batched knots.
 * @param knots - The knot vector `[m]` (or `[f, m]`), non-decreasing; for a clamped spline the boundary knots repeat `degree + 1` times.
 * @param degree - The degree (3 is cubic).
 * @param options - How to continue outside the knot range.
 * @returns The basis `[n, m - degree - 1]` (or `[n, f, m - degree - 1]`).
 * @throws MdmlError `shape` for a knot vector that is too short or points of the wrong shape; `value` for a bad degree.
 * @concept splines
 * @example
 * ```ts
 * import { bsplineBasis } from 'mdml-engine/nn'
 * import { sum, toFlat } from 'mdml-engine/tensor'
 * const basis = bsplineBasis([0.3], [0, 0, 0, 0, 0.5, 1, 1, 1, 1], 3)
 * basis.shape // => [1, 5]
 * Math.abs(toFlat(sum(basis))[0] - 1) < 1e-12 // => true
 * ```
 */
export function bsplineBasis(x: TensorLike, knots: TensorLike, degree: number, options: BasisOptions = {}): Value {
  const t = asTensor(knots)
  const m = checkKnots('bsplineBasis', t, degree)
  const xs = column(x, t.shape.length === 2)
  const lo = knotsAt(t, 0, 1)
  const hi = knotsAt(t, m - 1, m)
  const xc = clip(xs, lo, hi)
  const at = rawBasis(xc, t, m, degree)
  if (options.outside !== 'linear' || degree === 0) return at
  return add(at, mul(sub(xs, xc), rawDerivative(xc, t, m, degree, 1)))
}

function rawBasis(x: Value, t: Value, m: number, degree: number): Value {
  let basis = indicator(x, t, m)
  for (let r = 1; r <= degree; r++) {
    const count = m - 1 - r
    const a = safeRatio(sub(x, knotsAt(t, 0, count)), sub(knotsAt(t, r, r + count), knotsAt(t, 0, count)))
    const b = safeRatio(
      sub(knotsAt(t, r + 1, r + 1 + count), x),
      sub(knotsAt(t, r + 1, r + 1 + count), knotsAt(t, 1, 1 + count)),
    )
    basis = add(
      mul(a, sliceAxis(basis, lastAxis(basis), 0, count)),
      mul(b, sliceAxis(basis, lastAxis(basis), 1, count + 1)),
    )
  }
  return basis
}

/** The `order`th derivative of the degree-`degree` basis, from the degree - 1 derivative of order - 1. */
function rawDerivative(x: Value, t: Value, m: number, degree: number, order: number): Value {
  if (order === 0) return rawBasis(x, t, m, degree)
  const count = m - degree - 1
  if (degree === 0) return mul(rawBasis(x, t, m, 0), 0)
  const lower = rawDerivative(x, t, m, degree - 1, order - 1)
  const ax = lastAxis(lower)
  const first = safeRatio(sliceAxis(lower, ax, 0, count), sub(knotsAt(t, degree, degree + count), knotsAt(t, 0, count)))
  const second = safeRatio(
    sliceAxis(lower, ax, 1, count + 1),
    sub(knotsAt(t, degree + 1, degree + 1 + count), knotsAt(t, 1, 1 + count)),
  )
  return mul(sub(first, second), degree)
}

/**
 * The `order`th derivative of the B-spline basis functions with respect to the point, by the same recurrence (so it
 * is differentiable in the knots and the points too). At the knots of a spline of degree `d` the derivatives of order
 * up to `d - 1` are continuous; the one-sided value from the right is returned where they jump.
 *
 * @param x - The points `[n]` (or `[n, 1]`), or `[n, f]` for batched knots.
 * @param knots - The knot vector `[m]` (or `[f, m]`).
 * @param degree - The degree.
 * @param order - The derivative order (default 1; 0 is the basis).
 * @param options - How to continue outside the knot range (`linear` keeps the boundary tangent beyond it).
 * @returns The derivatives `[n, m - degree - 1]` (or `[n, f, m - degree - 1]`).
 * @concept splines
 * @example
 * ```ts
 * import { bsplineBasisDerivative } from 'mdml-engine/nn'
 * import { sum, toFlat } from 'mdml-engine/tensor'
 * // The basis sums to one, so its derivatives sum to zero.
 * const d = bsplineBasisDerivative([0.3], [0, 0, 0, 0, 0.5, 1, 1, 1, 1], 3, 1)
 * Math.abs(toFlat(sum(d))[0]) < 1e-12 // => true
 * ```
 */
export function bsplineBasisDerivative(
  x: TensorLike,
  knots: TensorLike,
  degree: number,
  order = 1,
  options: BasisOptions = {},
): Value {
  const t = asTensor(knots)
  const m = checkKnots('bsplineBasisDerivative', t, degree)
  if (!(Number.isInteger(order) && order >= 0))
    throw new MdmlError('value', 'bsplineBasisDerivative: order must be an integer >= 0')
  if (order === 0) return bsplineBasis(x, knots, degree, options)
  const xs = column(x, t.shape.length === 2)
  const xc = clip(xs, knotsAt(t, 0, 1), knotsAt(t, m - 1, m))
  const d = rawDerivative(xc, t, m, degree, order)
  // Beyond the boundary the clamped basis is constant (derivative 0); the linear continuation keeps the boundary
  // tangent, whose own derivatives of higher order vanish.
  const outside = notEqual(xs, xc)
  const kept = options.outside === 'linear' && order === 1 ? d : mul(d, where(outside, 0, 1))
  return kept
}

const bounds = (v: number | ArrayLike<number>): Value =>
  typeof v === 'number' ? asTensor(v) : reshape(asTensor(Array.from(v)), [v.length, 1])

/**
 * The knot vector of a clamped spline: the boundary knots repeated `degree + 1` times around the interior knots.
 *
 * @param interior - The interior knots `[k]` (or `[f, k]`), strictly between the boundaries (may be traced).
 * @param lo - The lower boundary (one per feature for batched knots).
 * @param hi - The upper boundary.
 * @param degree - The degree.
 * @returns The knot vector `[k + 2 (degree + 1)]` (or `[f, k + 2 (degree + 1)]`).
 * @concept splines
 * @example
 * ```ts
 * import { extendKnots } from 'mdml-engine/nn'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(extendKnots([0.5], 0, 1, 2)) // => [0, 0, 0, 0.5, 1, 1, 1]
 * ```
 */
export function extendKnots(
  interior: TensorLike,
  lo: number | ArrayLike<number>,
  hi: number | ArrayLike<number>,
  degree: number,
): Value {
  const inner = asTensor(interior)
  const edge = (v: number | ArrayLike<number>): Value =>
    inner.shape.length === 1
      ? full([degree + 1], typeof v === 'number' ? v : (v[0] ?? 0))
      : broadcastTo(bounds(v), [inner.shape[0] ?? 0, degree + 1])
  return concat([edge(lo), inner, edge(hi)], lastAxis(inner))
}

/**
 * The Greville abscissae of a spline: the knot averages `(t_{i+1} + ... + t_{i+d}) / d`, the point on the axis where
 * the `i`th coefficient acts, so that the coefficients against them form the spline's control polygon. A coefficient
 * sequence with non-decreasing slopes against the abscissae gives a convex spline, which is how the shape-constrained
 * layers read their constraints.
 *
 * @param knots - The knot vector `[m]` (or `[f, m]`).
 * @param degree - The degree (at least 1).
 * @returns The abscissae `[m - degree - 1]` (or `[f, m - degree - 1]`).
 * @concept splines
 * @example
 * ```ts
 * import { grevilleAbscissae } from 'mdml-engine/nn'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(grevilleAbscissae([0, 0, 0, 1, 1, 1], 2)) // => [0, 0.5, 1]
 * ```
 */
export function grevilleAbscissae(knots: TensorLike, degree: number): Value {
  const t = asTensor(knots)
  const m = checkKnots('grevilleAbscissae', t, degree)
  if (degree < 1) throw new MdmlError('value', 'grevilleAbscissae: degree must be at least 1')
  const count = m - degree - 1
  let total: Value = knotsAt(t, 1, 1 + count)
  for (let j = 2; j <= degree; j++) total = add(total, knotsAt(t, j, j + count))
  return div(total, degree)
}

/**
 * Interior knots from unconstrained logits: the `k + 1` gaps between the boundaries (and the `k` knots) are a softmax,
 * so the knots are strictly increasing inside `(lo, hi)` for any logits, and gradients reach every logit. A floor
 * `minGap` (a share of the range) keeps adjacent knots from colliding.
 *
 * @param logits - The logits `[k + 1]` (or `[f, k + 1]`).
 * @param lo - The lower boundary (one per feature for batched logits).
 * @param hi - The upper boundary.
 * @param minGap - The least gap as a share of the range (default 1e-3; at most `1 / (k + 1)`).
 * @returns The interior knots `[k]` (or `[f, k]`).
 * @concept splines
 * @example
 * ```ts
 * import { knotsFromLogits } from 'mdml-engine/nn'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(knotsFromLogits([0, 0, 0], 0, 3, 0)) // => [1, 2]
 * ```
 */
export function knotsFromLogits(
  logits: TensorLike,
  lo: number | ArrayLike<number>,
  hi: number | ArrayLike<number>,
  minGap = 1e-3,
): Value {
  const g = asTensor(logits)
  const ax = lastAxis(g)
  const parts = g.shape[ax] ?? 0
  if (g.shape.length < 1 || g.shape.length > 2 || parts < 2)
    throw new MdmlError('shape', 'knotsFromLogits: need logits [k + 1] or [f, k + 1], k >= 1')
  if (!(minGap >= 0 && minGap * parts <= 1))
    throw new MdmlError('value', 'knotsFromLogits: minGap must lie in [0, 1 / (k + 1)]')
  const gaps = add(mul(softmax(g, ax), 1 - minGap * parts), minGap)
  const inner = sliceAxis(cumsum(gaps, ax), ax, 0, parts - 1)
  const low = bounds(lo)
  return add(mul(inner, sub(bounds(hi), low)), low)
}

/**
 * The logits whose {@link knotsFromLogits} give the interior knots (the initialisation of a free-knot layer from
 * uniform or quantile knots).
 *
 * @param interior - The interior knots, strictly increasing inside `(lo, hi)`.
 * @param lo - The lower boundary.
 * @param hi - The upper boundary.
 * @param minGap - The same floor as {@link knotsFromLogits}.
 * @returns The logits `[k + 1]` (centred to mean zero).
 * @throws MdmlError `value` when the knots are not strictly increasing inside the boundaries.
 * @concept splines
 * @example
 * ```ts
 * import { knotLogits, knotsFromLogits } from 'mdml-engine/nn'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(knotsFromLogits(knotLogits([0.2, 0.5], 0, 1), 0, 1)).map((v) => Math.round(v * 1e9) / 1e9) // => [0.2, 0.5]
 * ```
 */
export function knotLogits(interior: TensorLike, lo: number, hi: number, minGap = 1e-3): Value {
  const inner = toFlat(asTensor(interior))
  const edges = [lo, ...inner, hi]
  const parts = edges.length - 1
  const share = edges.slice(1).map((e, i) => (e - (edges[i] as number)) / (hi - lo))
  if (share.some((v) => !(v > minGap)))
    throw new MdmlError(
      'value',
      'knotLogits: interior knots must be strictly increasing inside (lo, hi), further apart than minGap',
    )
  const logits = share.map((v) => Math.log((v - minGap) / (1 - minGap * parts)))
  const centre = logits.reduce((a, b) => a + b, 0) / parts
  return asTensor(logits.map((v) => v - centre))
}
