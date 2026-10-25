/**
 * Vectors and the geometry of lines and planes: dot, cross and triple products, lengths and unit vectors, projections
 * and rejections, angles between vectors by Kahan's stable formula, Gram-Schmidt orthonormalisation, linear
 * independence, span dimension and span membership, lines and planes with their equations, distances, closest points
 * and intersections.
 */

import { MdmlError } from '../base/index.ts'
import { daxpy, ddot, dnrm2, dscal } from '../numerics/linalg/blas.ts'
import { lstsq, matrixRank } from '../numerics/linalg/index.ts'
import {
  add,
  arctan2,
  div,
  matmul,
  mul,
  promoteFloat,
  sliceAxis,
  sqrt,
  squeeze,
  stack,
  sub,
  sum,
  type TensorLike,
  toArray,
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'

const component = (v: Value, i: number): Value => squeeze(sliceAxis(v, -1, i, i + 1), -1)
const dotLast = (a: Value, b: Value): Value => sum(mul(a, b), -1)
const keep = (x: Value): Value => stack([x], -1)

function floats(...xs: readonly TensorLike[]): Value[] {
  return promoteFloat(...xs)
}

/**
 * The dot product along the last axis, broadcasting over the rest: `a . b = sum_i a_i b_i`. Differentiable.
 *
 * @param a - Vectors `[..., n]`.
 * @param b - Vectors `[..., n]`.
 * @returns The products `[...]`.
 * @concept dot-product
 * @example
 * ```ts
 * import { dotProduct } from 'mdml-engine/foundations'
 * import { item } from 'mdml-engine/tensor'
 * item(dotProduct([1, 2, 3], [4, 5, 6])) // => 32
 * ```
 */
export function dotProduct(a: TensorLike, b: TensorLike): Value {
  const [x, y] = floats(a, b) as [Value, Value]
  return dotLast(x, y)
}

/**
 * The length (Euclidean norm) of vectors along the last axis. Differentiable away from zero.
 *
 * @param a - Vectors `[..., n]`.
 * @returns The lengths `[...]`.
 * @concept vector-norm
 * @example
 * ```ts
 * import { magnitude } from 'mdml-engine/foundations'
 * import { item } from 'mdml-engine/tensor'
 * item(magnitude([3, 4])) // => 5
 * ```
 */
export function magnitude(a: TensorLike): Value {
  const [x] = floats(a) as [Value]
  return sqrt(dotLast(x, x))
}

/**
 * The unit vector in the direction of each vector (NaN for the zero vector).
 *
 * @param a - Vectors `[..., n]`.
 * @returns `a / |a|`.
 * @concept vector-norm
 * @example
 * ```ts
 * import { unitVector } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(unitVector([3, 4])) // => [0.6, 0.8]
 * ```
 */
export function unitVector(a: TensorLike): Value {
  const [x] = floats(a) as [Value]
  return div(x, keep(sqrt(dotLast(x, x))))
}

/**
 * The cross product of 3-D vectors (numpy `cross`), or for 2-D vectors the scalar z-component `a_x b_y - a_y b_x`
 * (the signed area of the parallelogram). Differentiable.
 *
 * @param a - Vectors `[..., 3]` or `[..., 2]`.
 * @param b - Vectors of the same length.
 * @returns `a x b` `[..., 3]`, or `[...]` for 2-D input.
 * @throws MdmlError with code `shape` for other lengths.
 * @concept cross-product
 * @example
 * ```ts
 * import { crossProduct } from 'mdml-engine/foundations'
 * import { item, toArray } from 'mdml-engine/tensor'
 * toArray(crossProduct([1, 0, 0], [0, 1, 0])) // => [0, 0, 1]
 * item(crossProduct([2, 0], [1, 3])) // => 6
 * ```
 */
export function crossProduct(a: TensorLike, b: TensorLike): Value {
  const [x, y] = floats(a, b) as [Value, Value]
  const n = x.shape[x.shape.length - 1]
  if (n !== y.shape[y.shape.length - 1] || (n !== 2 && n !== 3)) {
    throw new MdmlError('shape', 'crossProduct: vectors must both have length 2 or 3')
  }
  const [x0, x1, y0, y1] = [component(x, 0), component(x, 1), component(y, 0), component(y, 1)]
  if (n === 2) return sub(mul(x0, y1), mul(x1, y0))
  const [x2, y2] = [component(x, 2), component(y, 2)]
  return stack([sub(mul(x1, y2), mul(x2, y1)), sub(mul(x2, y0), mul(x0, y2)), sub(mul(x0, y1), mul(x1, y0))], -1)
}

/**
 * The scalar triple product `a . (b x c)`: the signed volume of the parallelepiped on three 3-D vectors (the
 * determinant of the matrix with rows a, b, c). Differentiable.
 *
 * @param a - Vectors `[..., 3]`.
 * @param b - Vectors `[..., 3]`.
 * @param c - Vectors `[..., 3]`.
 * @returns The volumes `[...]`.
 * @concept triple-product
 * @example
 * ```ts
 * import { tripleProduct } from 'mdml-engine/foundations'
 * import { item } from 'mdml-engine/tensor'
 * item(tripleProduct([1, 0, 0], [0, 2, 0], [0, 0, 3])) // => 6
 * ```
 */
export function tripleProduct(a: TensorLike, b: TensorLike, c: TensorLike): Value {
  const [x] = floats(a) as [Value]
  return dotLast(x, crossProduct(b, c))
}

/**
 * The vector triple product `a x (b x c) = b (a . c) - c (a . b)` (the BAC-CAB rule). Differentiable.
 *
 * @param a - Vectors `[..., 3]`.
 * @param b - Vectors `[..., 3]`.
 * @param c - Vectors `[..., 3]`.
 * @returns The products `[..., 3]`.
 * @concept triple-product
 * @example
 * ```ts
 * import { vectorTripleProduct } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(vectorTripleProduct([1, 0, 0], [1, 0, 0], [0, 1, 0])) // => [0, -1, 0]
 * ```
 */
export function vectorTripleProduct(a: TensorLike, b: TensorLike, c: TensorLike): Value {
  return crossProduct(a, crossProduct(b, c))
}

/**
 * The vector projection of `a` onto the direction of `onto`: `(a . b / b . b) b`. Differentiable.
 *
 * @param a - Vectors `[..., n]`.
 * @param onto - Directions `[..., n]` (nonzero).
 * @returns The projections `[..., n]`.
 * @concept vector-projection
 * @example
 * ```ts
 * import { projection } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(projection([2, 3], [1, 0])) // => [2, 0]
 * ```
 */
export function projection(a: TensorLike, onto: TensorLike): Value {
  const [x, y] = floats(a, onto) as [Value, Value]
  return mul(keep(div(dotLast(x, y), dotLast(y, y))), y)
}

/**
 * The scalar projection (component) of `a` along `onto`: `a . b / |b|`, signed. Differentiable.
 *
 * @param a - Vectors `[..., n]`.
 * @param onto - Directions `[..., n]` (nonzero).
 * @returns The components `[...]`.
 * @concept vector-projection
 * @example
 * ```ts
 * import { scalarProjection } from 'mdml-engine/foundations'
 * import { item } from 'mdml-engine/tensor'
 * item(scalarProjection([3, 4], [0, -2])) // => -4
 * ```
 */
export function scalarProjection(a: TensorLike, onto: TensorLike): Value {
  const [x, y] = floats(a, onto) as [Value, Value]
  return div(dotLast(x, y), sqrt(dotLast(y, y)))
}

/**
 * The rejection of `a` from `onto`: the part of `a` perpendicular to it, `a - proj(a, b)`. Differentiable.
 *
 * @param a - Vectors `[..., n]`.
 * @param onto - Directions `[..., n]` (nonzero).
 * @returns The rejections `[..., n]`.
 * @concept vector-projection
 * @example
 * ```ts
 * import { rejection } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(rejection([2, 3], [1, 0])) // => [0, 3]
 * ```
 */
export function rejection(a: TensorLike, onto: TensorLike): Value {
  const [x] = floats(a) as [Value]
  return sub(x, projection(x, onto))
}

/**
 * The angle between vectors in `[0, pi]`, by Kahan's formula `2 atan2(|a/|a| - b/|b||, |a/|a| + b/|b||)`, which keeps
 * its digits for nearly parallel and nearly opposite vectors (unlike `acos` of the cosine). Differentiable.
 *
 * @param a - Vectors `[..., n]` (nonzero).
 * @param b - Vectors `[..., n]` (nonzero).
 * @returns The angles `[...]` in radians.
 * @concept angle-between-vectors
 * @example
 * ```ts
 * import { angleBetween } from 'mdml-engine/foundations'
 * import { item } from 'mdml-engine/tensor'
 * item(angleBetween([1, 0], [0, 2])) // => 1.5707963267948966
 * ```
 */
export function angleBetween(a: TensorLike, b: TensorLike): Value {
  const ua = unitVector(a)
  const ub = unitVector(b)
  return mul(2, arctan2(magnitude(sub(ua, ub)), magnitude(add(ua, ub))))
}

/** Result of {@link orthonormalise}. */
export interface Orthonormalised {
  /** Orthonormal vectors spanning the same space, one per independent input. */
  readonly basis: number[][]
  /** Indices of the input vectors that contributed a new direction. */
  readonly independent: number[]
  /** For each input vector, the length of its component orthogonal to the previous ones (0 for dependent ones). */
  readonly residuals: number[]
}

const rows = (vectors: TensorLike): number[][] => {
  const [v] = floats(vectors) as [Value]
  if (v.shape.length !== 2) throw new MdmlError('shape', 'expected vectors as rows of a matrix [k, n]')
  return toArray(v) as number[][]
}

/**
 * Orthonormalise a list of vectors by modified Gram-Schmidt with one reorthogonalisation pass, dropping vectors that
 * are (numerically) in the span of the earlier ones: the basis of their span, in order.
 *
 * @param vectors - Vectors as rows `[k, n]`.
 * @param tol - Relative threshold below which a residual counts as zero (default 1e-10).
 * @returns The basis, which inputs were independent, and the residual lengths.
 * @concept orthonormalisation
 * @example
 * ```ts
 * import { orthonormalise } from 'mdml-engine/foundations'
 * const r = orthonormalise([[3, 4], [6, 8], [1, 0]])
 * r.independent // => [0, 2]
 * r.basis // => [[0.6, 0.8], [0.8, -0.6]]
 * ```
 */
export function orthonormalise(vectors: TensorLike, tol = 1e-10): Orthonormalised {
  const vs = rows(vectors)
  const basis: Float64Array[] = []
  const independent: number[] = []
  const residuals: number[] = []
  for (const [i, v] of vs.entries()) {
    let w: Float64Array<ArrayBufferLike> = Float64Array.from(v)
    const size = dnrm2(w)
    for (let pass = 0; pass < 2; pass++) {
      for (const q of basis) w = daxpy(-ddot(q, w), q, w)
    }
    const r = dnrm2(w)
    if (r > tol * Math.max(size, 1e-300) && r > 0) {
      basis.push(dscal(1 / r, w))
      independent.push(i)
      residuals.push(r)
    } else residuals.push(0)
  }
  return { basis: basis.map((q) => Array.from(q, (x) => (Object.is(x, -0) ? 0 : x))), independent, residuals }
}

/**
 * Whether vectors are linearly independent: the matrix with them as rows has full row rank (numpy `matrix_rank`).
 *
 * @param vectors - Vectors as rows `[k, n]`.
 * @param tol - Singular-value threshold (default numpy's).
 * @returns `true` when no vector is a combination of the others.
 * @concept linear-independence
 * @example
 * ```ts
 * import { isLinearlyIndependent } from 'mdml-engine/foundations'
 * isLinearlyIndependent([[1, 0, 0], [0, 1, 0]]) // => true
 * isLinearlyIndependent([[1, 2], [2, 4]]) // => false
 * ```
 */
export function isLinearlyIndependent(vectors: TensorLike, tol?: number): boolean {
  const vs = rows(vectors)
  return vs.length <= (vs[0]?.length ?? 0) && matrixRank(vs, tol) === vs.length
}

/**
 * The dimension of the span of a set of vectors (the rank of the matrix with them as rows).
 *
 * @param vectors - Vectors as rows `[k, n]`.
 * @param tol - Singular-value threshold (default numpy's).
 * @returns The dimension.
 * @concept span
 * @example
 * ```ts
 * import { spanDimension } from 'mdml-engine/foundations'
 * spanDimension([[1, 0, 1], [0, 1, 1], [1, 1, 2]]) // => 2
 * ```
 */
export function spanDimension(vectors: TensorLike, tol?: number): number {
  return matrixRank(rows(vectors), tol)
}

/** Result of {@link inSpan}. */
export interface SpanMembership {
  readonly inSpan: boolean
  /** Least-squares coefficients c with sum_i c_i v_i closest to the target (exact when it is in the span). */
  readonly coefficients: number[]
  /** The distance from the target to the span. */
  readonly residual: number
}

/**
 * Whether a vector lies in the span of others, with the coefficients that combine them (least squares on the
 * independent vectors, so the coefficients are unique) and the distance to the span.
 *
 * @param vectors - Spanning vectors as rows `[k, n]`.
 * @param target - The vector `[n]`.
 * @param tol - Relative distance below which it counts as in the span (default 1e-10).
 * @returns Membership, coefficients (zero for dependent vectors) and the residual distance.
 * @concept span
 * @example
 * ```ts
 * import { inSpan } from 'mdml-engine/foundations'
 * inSpan([[1, 0, 0], [0, 1, 0]], [2, 3, 0]).coefficients // => [2, 3]
 * inSpan([[1, 0, 0], [0, 1, 0]], [0, 0, 1]).inSpan // => false
 * ```
 */
export function inSpan(vectors: TensorLike, target: TensorLike, tol = 1e-10): SpanMembership {
  const vs = rows(vectors)
  const t = toFlat(floats(target)[0] as Value)
  const { independent } = orthonormalise(vs)
  const coefficients = vs.map(() => 0)
  let fitted: Float64Array<ArrayBufferLike> = new Float64Array(t.length)
  if (independent.length > 0) {
    const columns = transpose(independent.map((i) => vs[i] as number[]))
    const c = toFlat(lstsq(columns, t))
    for (const [j, idx] of independent.entries()) {
      coefficients[idx] = c[j] ?? 0
      fitted = daxpy(c[j] ?? 0, vs[idx] as number[], fitted)
    }
  }
  const residual = dnrm2(daxpy(-1, fitted, t))
  return { inSpan: residual <= tol * Math.max(1, dnrm2(t)), coefficients, residual }
}

/** A line through `point` with direction `direction` (in 2-D or 3-D, or any dimension for distances). */
export interface Line {
  readonly point: readonly number[]
  readonly direction: readonly number[]
}

/** A plane through `point` with normal `normal`: the points x with `normal . (x - point) = 0`. */
export interface Plane {
  readonly point: readonly number[]
  readonly normal: readonly number[]
}

/**
 * The line through two distinct points, directed from the first to the second.
 *
 * @param p - A point.
 * @param q - Another point.
 * @returns The line.
 * @throws MdmlError with code `value` when the points coincide.
 * @concept lines-and-planes
 * @example
 * ```ts
 * import { lineThrough } from 'mdml-engine/foundations'
 * lineThrough([1, 1, 0], [3, 1, 0]).direction // => [2, 0, 0]
 * ```
 */
export function lineThrough(p: readonly number[], q: readonly number[]): Line {
  const direction = Array.from(daxpy(-1, p, q))
  if (dnrm2(direction) === 0) throw new MdmlError('value', 'lineThrough: the points coincide')
  return { point: [...p], direction }
}

/**
 * The plane through three non-collinear points, with normal `(q - p) x (r - p)`.
 *
 * @param p - A point `[3]`.
 * @param q - A point `[3]`.
 * @param r - A point `[3]`.
 * @returns The plane.
 * @throws MdmlError with code `value` when the points are collinear.
 * @concept lines-and-planes
 * @example
 * ```ts
 * import { planeThrough } from 'mdml-engine/foundations'
 * planeThrough([0, 0, 0], [1, 0, 0], [0, 1, 0]).normal // => [0, 0, 1]
 * ```
 */
export function planeThrough(p: readonly number[], q: readonly number[], r: readonly number[]): Plane {
  const normal = toFlat(crossProduct(Array.from(daxpy(-1, p, q)), Array.from(daxpy(-1, p, r))))
  if (dnrm2(normal) === 0) throw new MdmlError('value', 'planeThrough: the points are collinear')
  return { point: [...p], normal }
}

/**
 * The plane `a x + b y + c z = d` from its coefficients (through the point of it nearest the origin).
 *
 * @param coefficients - `[a, b, c, d]`.
 * @returns The plane.
 * @throws MdmlError with code `value` when the normal is zero.
 * @concept lines-and-planes
 * @example
 * ```ts
 * import { planeFromEquation } from 'mdml-engine/foundations'
 * planeFromEquation([0, 0, 2, 4]).point // => [0, 0, 2]
 * ```
 */
export function planeFromEquation(coefficients: readonly [number, number, number, number]): Plane {
  const [a, b, c, d] = coefficients
  const nn = a * a + b * b + c * c
  if (nn === 0) throw new MdmlError('value', 'planeFromEquation: the normal is zero')
  return { point: [(a * d) / nn, (b * d) / nn, (c * d) / nn], normal: [a, b, c] }
}

/**
 * The coefficients `[a, b, c, d]` of a plane's equation `a x + b y + c z = d`, with the normal as given.
 *
 * @param plane - The plane.
 * @returns `[a, b, c, d]`.
 * @concept lines-and-planes
 * @example
 * ```ts
 * import { planeEquation, planeThrough } from 'mdml-engine/foundations'
 * planeEquation(planeThrough([0, 0, 1], [1, 0, 1], [0, 1, 1])) // => [0, 0, 1, 1]
 * ```
 */
export function planeEquation(plane: Plane): [number, number, number, number] {
  const [a = 0, b = 0, c = 0] = plane.normal
  return [a, b, c, ddot(plane.normal, plane.point)]
}

/**
 * The distance from points to a line (any dimension): the length of the rejection of `x - point` from the direction.
 * Differentiable in the points.
 *
 * @param x - Points `[..., n]`.
 * @param line - The line.
 * @returns Distances `[...]`.
 * @concept lines-and-planes
 * @example
 * ```ts
 * import { pointLineDistance } from 'mdml-engine/foundations'
 * import { item } from 'mdml-engine/tensor'
 * item(pointLineDistance([0, 3, 4], { point: [0, 0, 0], direction: [1, 0, 0] })) // => 5
 * ```
 */
export function pointLineDistance(x: TensorLike, line: Line): Value {
  const [xv] = floats(x) as [Value]
  return magnitude(rejection(sub(xv, [...line.point]), [...line.direction]))
}

/**
 * The signed distance from points to a plane, positive on the side the normal points to. Differentiable in the points.
 *
 * @param x - Points `[..., 3]`.
 * @param plane - The plane.
 * @returns Signed distances `[...]`.
 * @concept lines-and-planes
 * @example
 * ```ts
 * import { pointPlaneDistance } from 'mdml-engine/foundations'
 * import { item } from 'mdml-engine/tensor'
 * item(pointPlaneDistance([1, 1, -2], { point: [0, 0, 1], normal: [0, 0, 5] })) // => -3
 * ```
 */
export function pointPlaneDistance(x: TensorLike, plane: Plane): Value {
  const [xv] = floats(x) as [Value]
  return scalarProjection(sub(xv, [...plane.point]), [...plane.normal])
}

/** Result of {@link closestPoints}. */
export interface ClosestPoints {
  /** Parameters s, t of the closest points `p1 + s d1` and `p2 + t d2`. */
  readonly s: number
  readonly t: number
  readonly onFirst: number[]
  readonly onSecond: number[]
  readonly distance: number
  /** Whether the lines are parallel (then s = 0 and t is the foot of p1 on the second line). */
  readonly parallel: boolean
}

/**
 * The closest points of two lines in any dimension and the distance between them (for skew lines in 3-D, the length
 * of the common perpendicular), from the 2x2 normal equations.
 *
 * @param first - A line.
 * @param second - Another line of the same dimension.
 * @param tol - Relative threshold for parallel directions (default 1e-12).
 * @returns The parameters, the points and the distance.
 * @concept lines-and-planes
 * @example
 * ```ts
 * import { closestPoints } from 'mdml-engine/foundations'
 * closestPoints({ point: [0, 0, 0], direction: [1, 0, 0] }, { point: [0, 1, 2], direction: [0, 1, 0] }).distance // => 2
 * ```
 */
export function closestPoints(first: Line, second: Line, tol = 1e-12): ClosestPoints {
  const { point: p1, direction: d1 } = first
  const { point: p2, direction: d2 } = second
  const w = daxpy(-1, p2, p1)
  const a = ddot(d1, d1)
  const b = ddot(d1, d2)
  const c = ddot(d2, d2)
  const d = ddot(d1, w)
  const e = ddot(d2, w)
  const den = a * c - b * b
  const parallel = den <= tol * a * c
  const s = parallel ? 0 : (b * e - c * d) / den
  const t = parallel ? e / c : (a * e - b * d) / den
  const onFirst = Array.from(daxpy(s, d1, p1))
  const onSecond = Array.from(daxpy(t, d2, p2))
  return { s, t, onFirst, onSecond, distance: dnrm2(daxpy(-1, onSecond, onFirst)), parallel }
}

/**
 * The intersection point of two lines, or null when they are parallel or skew (closer than `tol` counts as meeting).
 *
 * @param first - A line.
 * @param second - Another line.
 * @param tol - Distance tolerance, relative to the size of the points (default 1e-10).
 * @returns The point, or null.
 * @concept lines-and-planes
 * @example
 * ```ts
 * import { lineIntersection } from 'mdml-engine/foundations'
 * lineIntersection({ point: [0, 0], direction: [1, 1] }, { point: [2, 0], direction: [0, 1] }) // => [2, 2]
 * ```
 */
export function lineIntersection(first: Line, second: Line, tol = 1e-10): number[] | null {
  const r = closestPoints(first, second)
  if (r.parallel) return null
  const scale = Math.max(1, dnrm2(r.onFirst))
  return r.distance <= tol * scale ? r.onFirst.map((v, i) => (v + (r.onSecond[i] ?? v)) / 2) : null
}

/**
 * Where a line meets a plane: the parameter `t` with `point + t direction` on the plane, or null when the line is
 * parallel to the plane (whether or not it lies in it).
 *
 * @param line - A line in 3-D.
 * @param plane - A plane.
 * @param tol - Relative threshold for a parallel line (default 1e-12).
 * @returns `{ point, t }`, or null.
 * @concept lines-and-planes
 * @example
 * ```ts
 * import { linePlaneIntersection } from 'mdml-engine/foundations'
 * linePlaneIntersection({ point: [0, 0, 0], direction: [1, 1, 1] }, { point: [0, 0, 3], normal: [0, 0, 1] })?.point // => [3, 3, 3]
 * ```
 */
export function linePlaneIntersection(
  line: Line,
  plane: Plane,
  tol = 1e-12,
): { readonly point: number[]; readonly t: number } | null {
  const den = ddot(plane.normal, line.direction)
  if (Math.abs(den) <= tol * dnrm2(plane.normal) * dnrm2(line.direction)) return null
  const t = ddot(plane.normal, daxpy(-1, line.point, plane.point)) / den
  return { point: Array.from(daxpy(t, line.direction, line.point)), t }
}

/**
 * The line where two planes meet (direction `n1 x n2`, through the point of it nearest the origin), or null for
 * parallel planes.
 *
 * @param first - A plane.
 * @param second - Another plane.
 * @param tol - Relative threshold for parallel normals (default 1e-12).
 * @returns The line, or null.
 * @concept lines-and-planes
 * @example
 * ```ts
 * import { planeIntersection } from 'mdml-engine/foundations'
 * planeIntersection({ point: [0, 0, 1], normal: [0, 0, 1] }, { point: [2, 0, 0], normal: [1, 0, 0] }) // => { point: [2, 0, 1], direction: [0, 1, 0] }
 * ```
 */
export function planeIntersection(first: Plane, second: Plane, tol = 1e-12): Line | null {
  const n1 = first.normal
  const n2 = second.normal
  const direction = toFlat(crossProduct([...n1], [...n2]))
  if (dnrm2(direction) <= tol * dnrm2(n1) * dnrm2(n2)) return null
  // The point x = alpha n1 + beta n2 on both planes is nearest the origin.
  const d1 = ddot(n1, first.point)
  const d2 = ddot(n2, second.point)
  const a = ddot(n1, n1)
  const b = ddot(n1, n2)
  const c = ddot(n2, n2)
  const den = a * c - b * b
  const alpha = (d1 * c - d2 * b) / den
  const beta = (d2 * a - d1 * b) / den
  const point = Array.from(daxpy(beta, n2, dscal(alpha, n1)), (v) => (Object.is(v, -0) ? 0 : v))
  return { point, direction: direction.map((v) => (Object.is(v, -0) ? 0 : v)) }
}

/**
 * Apply a list of 2-D or 3-D vectors as columns to coefficients: `sum_i c_i v_i`, the linear combination behind span
 * figures. Differentiable in both.
 *
 * @param vectors - Vectors as rows `[k, n]`.
 * @param coefficients - Coefficients `[..., k]`.
 * @returns Combinations `[..., n]`.
 * @concept span
 * @example
 * ```ts
 * import { linearCombination } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(linearCombination([[1, 0], [1, 1]], [2, 3])) // => [5, 3]
 * ```
 */
export function linearCombination(vectors: TensorLike, coefficients: TensorLike): Value {
  const [v, c] = floats(vectors, coefficients) as [Value, Value]
  return matmul(c, v)
}
