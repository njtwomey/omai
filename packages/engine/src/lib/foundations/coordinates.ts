/**
 * Coordinate systems and changes of coordinates: conversions between Cartesian, polar, cylindrical, spherical (physics
 * convention) and geographic coordinates; homogeneous coordinates for affine and projective maps; barycentric
 * coordinates in simplices; coordinates in a basis, change-of-basis matrices and similarity transforms; and the
 * Jacobian of a coordinate map by automatic differentiation.
 */

import { jacobian } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import { det, solve } from '../numerics/linalg/index.ts'
import {
  add,
  arctan2,
  concat,
  cos,
  div,
  item,
  matmul,
  mul,
  onesLike,
  promoteFloat,
  reshape,
  sin,
  sliceAxis,
  sqrt,
  squeeze,
  stack,
  sub,
  sum,
  swapaxes,
  type TensorLike,
  toArray,
  type Value,
} from '../tensor/index.ts'

const part = (v: Value, i: number): Value => squeeze(sliceAxis(v, -1, i, i + 1), -1)

function lastAxis(x: TensorLike, n: number, name: string): Value {
  const [v] = promoteFloat(x) as [Value]
  if (v.shape[v.shape.length - 1] !== n) throw new MdmlError('shape', `${name}: expected points [..., ${n}]`)
  return v
}

/**
 * Cartesian to polar coordinates: `(x, y) -> (r, theta)` with `theta = atan2(y, x)` in `(-pi, pi]`. Differentiable away
 * from the origin.
 *
 * @param points - Points `[..., 2]`.
 * @returns `[..., 2]` as (r, theta).
 * @concept polar-coordinates
 * @example
 * ```ts
 * import { cartesianToPolar } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(cartesianToPolar([0, 2])) // => [2, 1.5707963267948966]
 * ```
 */
export function cartesianToPolar(points: TensorLike): Value {
  const p = lastAxis(points, 2, 'cartesianToPolar')
  const x = part(p, 0)
  const y = part(p, 1)
  return stack([sqrt(add(mul(x, x), mul(y, y))), arctan2(y, x)], -1)
}

/**
 * Polar to Cartesian coordinates: `(r, theta) -> (r cos theta, r sin theta)`. Differentiable.
 *
 * @param polar - `[..., 2]` as (r, theta).
 * @returns Points `[..., 2]`.
 * @concept polar-coordinates
 * @example
 * ```ts
 * import { polarToCartesian } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(polarToCartesian([2, 0])) // => [2, 0]
 * ```
 */
export function polarToCartesian(polar: TensorLike): Value {
  const p = lastAxis(polar, 2, 'polarToCartesian')
  const r = part(p, 0)
  const t = part(p, 1)
  return stack([mul(r, cos(t)), mul(r, sin(t))], -1)
}

/**
 * Cartesian to cylindrical coordinates: `(x, y, z) -> (rho, phi, z)`. Differentiable away from the axis.
 *
 * @param points - Points `[..., 3]`.
 * @returns `[..., 3]` as (rho, phi, z).
 * @concept cylindrical-coordinates
 * @example
 * ```ts
 * import { cartesianToCylindrical } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(cartesianToCylindrical([3, 4, 5])) // => [5, 0.9272952180016122, 5]
 * ```
 */
export function cartesianToCylindrical(points: TensorLike): Value {
  const p = lastAxis(points, 3, 'cartesianToCylindrical')
  const x = part(p, 0)
  const y = part(p, 1)
  return stack([sqrt(add(mul(x, x), mul(y, y))), arctan2(y, x), part(p, 2)], -1)
}

/**
 * Cylindrical to Cartesian coordinates: `(rho, phi, z) -> (rho cos phi, rho sin phi, z)`. Differentiable.
 *
 * @param cylindrical - `[..., 3]` as (rho, phi, z).
 * @returns Points `[..., 3]`.
 * @concept cylindrical-coordinates
 * @example
 * ```ts
 * import { cylindricalToCartesian } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(cylindricalToCartesian([1, 0, 7])) // => [1, 0, 7]
 * ```
 */
export function cylindricalToCartesian(cylindrical: TensorLike): Value {
  const p = lastAxis(cylindrical, 3, 'cylindricalToCartesian')
  const r = part(p, 0)
  const t = part(p, 1)
  return stack([mul(r, cos(t)), mul(r, sin(t)), part(p, 2)], -1)
}

/**
 * Cartesian to spherical coordinates in the physics (ISO 80000-2) convention: `(x, y, z) -> (r, theta, phi)` with
 * `theta` the polar angle from the z-axis in `[0, pi]` and `phi` the azimuth in `(-pi, pi]`. Differentiable away from
 * the z-axis.
 *
 * @param points - Points `[..., 3]`.
 * @returns `[..., 3]` as (r, theta, phi).
 * @concept spherical-coordinates
 * @example
 * ```ts
 * import { cartesianToSpherical } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(cartesianToSpherical([0, 0, 2])) // => [2, 0, 0]
 * ```
 */
export function cartesianToSpherical(points: TensorLike): Value {
  const p = lastAxis(points, 3, 'cartesianToSpherical')
  const x = part(p, 0)
  const y = part(p, 1)
  const z = part(p, 2)
  const rho = sqrt(add(mul(x, x), mul(y, y)))
  return stack([sqrt(add(mul(rho, rho), mul(z, z))), arctan2(rho, z), arctan2(y, x)], -1)
}

/**
 * Spherical (physics convention: radius, polar angle from z, azimuth) to Cartesian coordinates. Differentiable.
 *
 * @param spherical - `[..., 3]` as (r, theta, phi).
 * @returns Points `[..., 3]`.
 * @concept spherical-coordinates
 * @example
 * ```ts
 * import { sphericalToCartesian } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(sphericalToCartesian([2, 0, 1])) // => [0, 0, 2]
 * ```
 */
export function sphericalToCartesian(spherical: TensorLike): Value {
  const p = lastAxis(spherical, 3, 'sphericalToCartesian')
  const r = part(p, 0)
  const t = part(p, 1)
  const f = part(p, 2)
  const s = mul(r, sin(t))
  return stack([mul(s, cos(f)), mul(s, sin(f)), mul(r, cos(t))], -1)
}

/**
 * Geographic latitude and longitude (radians) on a sphere of radius r to Cartesian coordinates: latitude is measured
 * from the equator, so it is `pi/2 - theta` of {@link sphericalToCartesian}. Differentiable.
 *
 * @param latLon - `[..., 2]` as (latitude, longitude) in radians.
 * @param radius - The radius (default 1).
 * @returns Points `[..., 3]`.
 * @concept spherical-coordinates
 * @example
 * ```ts
 * import { geographicToCartesian } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(geographicToCartesian([0, 0])) // => [1, 0, 0]
 * ```
 */
export function geographicToCartesian(latLon: TensorLike, radius = 1): Value {
  const p = lastAxis(latLon, 2, 'geographicToCartesian')
  const lat = part(p, 0)
  const lon = part(p, 1)
  const c = mul(cos(lat), radius)
  return stack([mul(c, cos(lon)), mul(c, sin(lon)), mul(sin(lat), radius)], -1)
}

/**
 * Homogeneous coordinates: append a 1 to each point, so that affine maps (and projective ones) act by matrix
 * multiplication. Differentiable.
 *
 * @param points - Points `[..., d]`.
 * @returns `[..., d + 1]`.
 * @concept homogeneous-coordinates
 * @example
 * ```ts
 * import { toHomogeneous } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(toHomogeneous([[1, 2]])) // => [[1, 2, 1]]
 * ```
 */
export function toHomogeneous(points: TensorLike): Value {
  const [p] = promoteFloat(points) as [Value]
  return concat([p, onesLike(sliceAxis(p, -1, 0, 1))], -1)
}

/**
 * Back from homogeneous coordinates: divide by the last coordinate and drop it (points at infinity give infinities).
 * Differentiable.
 *
 * @param points - `[..., d + 1]`.
 * @returns Points `[..., d]`.
 * @concept homogeneous-coordinates
 * @example
 * ```ts
 * import { fromHomogeneous } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(fromHomogeneous([4, 6, 2])) // => [2, 3]
 * ```
 */
export function fromHomogeneous(points: TensorLike): Value {
  const [p] = promoteFloat(points) as [Value]
  const n = p.shape[p.shape.length - 1] ?? 0
  return div(sliceAxis(p, -1, 0, n - 1), sliceAxis(p, -1, n - 1, n))
}

/**
 * The `(d + 1) x (d + 1)` homogeneous matrix of the affine map `x -> A x + t`.
 *
 * @param linear - The linear part `[d, d]`.
 * @param translation - The translation `[d]` (default zero).
 * @returns The matrix `[[A, t], [0, 1]]`.
 * @concept homogeneous-coordinates
 * @example
 * ```ts
 * import { homogeneousMatrix } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(homogeneousMatrix([[2, 0], [0, 2]], [1, -1])) // => [[2, 0, 1], [0, 2, -1], [0, 0, 1]]
 * ```
 */
export function homogeneousMatrix(linear: TensorLike, translation?: TensorLike): Value {
  const [a] = promoteFloat(linear) as [Value]
  const d = a.shape[0] ?? 0
  if (a.shape.length !== 2 || a.shape[1] !== d) throw new MdmlError('shape', 'homogeneousMatrix: linear part [d, d]')
  const [t] = promoteFloat(translation ?? Array.from({ length: d }, () => 0)) as [Value]
  const top = concat([a, stack([t], -1)], -1)
  const bottom = stack([concat([mul(sliceAxis(t, 0, 0, d), 0), [1]], 0)], 0)
  return concat([top, bottom], 0)
}

/**
 * Apply a homogeneous `(d + 1) x (d + 1)` matrix to points `[..., d]`: lift, multiply, and divide back (affine and
 * projective maps alike). Differentiable.
 *
 * @param matrix - The matrix `[d + 1, d + 1]`.
 * @param points - Points `[..., d]`.
 * @returns The mapped points `[..., d]`.
 * @concept homogeneous-coordinates
 * @example
 * ```ts
 * import { applyHomogeneous, homogeneousMatrix } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(applyHomogeneous(homogeneousMatrix([[0, -1], [1, 0]], [1, 0]), [1, 0])) // => [1, 1]
 * ```
 */
export function applyHomogeneous(matrix: TensorLike, points: TensorLike): Value {
  const [m] = promoteFloat(matrix) as [Value]
  return fromHomogeneous(matmul(toHomogeneous(points), swapaxes(m, 0, 1)))
}

/**
 * Barycentric coordinates of points in a simplex of any dimension (a segment, triangle, tetrahedron, ...): weights
 * summing to one with `p = sum_i l_i v_i`, all non-negative exactly inside. Differentiable in the points and vertices.
 *
 * @param points - Points `[..., d]`.
 * @param simplex - The `d + 1` vertices `[d + 1, d]`.
 * @returns Weights `[..., d + 1]`.
 * @throws MdmlError with code `shape` unless the simplex has one more vertex than dimensions.
 * @concept barycentric-coordinates
 * @example
 * ```ts
 * import { barycentric } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(barycentric([0.25, 0.25], [[0, 0], [1, 0], [0, 1]])) // => [0.5, 0.25, 0.25]
 * ```
 */
export function barycentric(points: TensorLike, simplex: TensorLike): Value {
  const [p, s] = promoteFloat(points, simplex) as [Value, Value]
  const d = s.shape[1] ?? 0
  if (s.shape.length !== 2 || s.shape[0] !== d + 1) {
    throw new MdmlError('shape', 'barycentric: a simplex in d dimensions has d + 1 vertices [d + 1, d]')
  }
  const v0 = squeeze(sliceAxis(s, 0, 0, 1), 0)
  // Columns v_i - v_0; solve for the last d weights, the first is what remains.
  const edges = swapaxes(sub(sliceAxis(s, 0, 1, d + 1), v0), 0, 1)
  const rel = reshape(sub(p, v0), [-1, d])
  const lam = swapaxes(solve(edges, swapaxes(rel, 0, 1)), 0, 1)
  const tail = reshape(lam, [...p.shape.slice(0, -1), d])
  return concat([stack([sub(1, sum(tail, -1))], -1), tail], -1)
}

/**
 * Points from barycentric weights: `sum_i l_i v_i`. Differentiable.
 *
 * @param weights - Weights `[..., k]`.
 * @param vertices - Vertices `[k, d]`.
 * @returns Points `[..., d]`.
 * @concept barycentric-coordinates
 * @example
 * ```ts
 * import { fromBarycentric } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(fromBarycentric([0.5, 0.25, 0.25], [[0, 0], [1, 0], [0, 1]])) // => [0.25, 0.25]
 * ```
 */
export function fromBarycentric(weights: TensorLike, vertices: TensorLike): Value {
  const [w, v] = promoteFloat(weights, vertices) as [Value, Value]
  return matmul(w, v)
}

/**
 * The coordinates of vectors in a basis: the `c` with `B c = v`, where the basis vectors are the columns of `B`.
 * Differentiable.
 *
 * @param vectors - Vectors `[n]` or `[n, k]` (as columns).
 * @param basis - Basis vectors as columns `[n, n]`.
 * @returns The coordinates, the shape of `vectors`.
 * @concept change-of-basis
 * @example
 * ```ts
 * import { coordinatesInBasis } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(coordinatesInBasis([3, 1], [[1, 1], [1, -1]])) // => [2, 1]
 * ```
 */
export function coordinatesInBasis(vectors: TensorLike, basis: TensorLike): Value {
  const [v, b] = promoteFloat(vectors, basis) as [Value, Value]
  return solve(b, v)
}

/**
 * The change-of-basis matrix from one basis to another: `P = B_to^-1 B_from`, so that `[v]_to = P [v]_from`.
 * Differentiable.
 *
 * @param from - The old basis as columns `[n, n]`.
 * @param to - The new basis as columns `[n, n]`.
 * @returns `P` `[n, n]`.
 * @concept change-of-basis
 * @example
 * ```ts
 * import { changeOfBasis } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(changeOfBasis([[2, 0], [0, 2]], [[1, 0], [0, 1]])) // => [[2, 0], [0, 2]]
 * ```
 */
export function changeOfBasis(from: TensorLike, to: TensorLike): Value {
  const [f, t] = promoteFloat(from, to) as [Value, Value]
  return solve(t, f)
}

/**
 * The matrix of a linear map in another basis: `B^-1 A B` (a similarity transform), with the basis as columns.
 * Differentiable.
 *
 * @param matrix - The map in standard coordinates `[n, n]`.
 * @param basis - The new basis as columns `[n, n]`.
 * @returns The matrix in the new basis.
 * @concept change-of-basis
 * @example
 * ```ts
 * import { matrixInBasis } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(matrixInBasis([[2, 1], [1, 2]], [[1, 1], [1, -1]])) // => [[3, 0], [0, 1]]
 * ```
 */
export function matrixInBasis(matrix: TensorLike, basis: TensorLike): Value {
  const [a, b] = promoteFloat(matrix, basis) as [Value, Value]
  return solve(b, matmul(a, b))
}

/** A named coordinate map to Cartesian coordinates. */
export type CoordinateSystem = 'polar' | 'cylindrical' | 'spherical'

const TO_CARTESIAN: Readonly<Record<CoordinateSystem, (x: Value) => Value>> = {
  polar: polarToCartesian,
  cylindrical: cylindricalToCartesian,
  spherical: sphericalToCartesian,
}

/**
 * The Jacobian determinant of a coordinate map at a point, by automatic differentiation: the factor by which the map
 * scales small volumes (`dx dy = |J| dr dtheta`). Either a named system (to Cartesian: `r` for polar and cylindrical,
 * `r^2 sin theta` for spherical) or any map from `R^n` to `R^n`.
 *
 * @param map - 'polar', 'cylindrical', 'spherical', or a function of one point `[n] -> [n]`.
 * @param point - The point `[n]` in the map's input coordinates.
 * @returns The determinant.
 * @concept jacobian-determinant
 * @example
 * ```ts
 * import { coordinateJacobianDeterminant } from 'mdml-engine/foundations'
 * coordinateJacobianDeterminant('polar', [2, 0.3]) // => 2
 * coordinateJacobianDeterminant('spherical', [2, Math.PI / 2, 0]) // => 4
 * ```
 */
export function coordinateJacobianDeterminant(
  map: CoordinateSystem | ((x: Value) => Value),
  point: TensorLike,
): number {
  const f = typeof map === 'string' ? TO_CARTESIAN[map] : map
  const j = jacobian(f)(point)
  const n = j.shape[0] ?? 0
  if (j.shape.length !== 2 || j.shape[1] !== n) {
    throw new MdmlError('shape', 'coordinateJacobianDeterminant: the map must send R^n to R^n')
  }
  return item(det(j))
}

/**
 * The Jacobian matrix of a coordinate map at a point (rows: outputs, columns: inputs), by automatic differentiation.
 *
 * @param map - A named system or a function `[n] -> [m]`.
 * @param point - The point `[n]`.
 * @returns The Jacobian `[m, n]` as nested arrays.
 * @concept jacobian-determinant
 * @example
 * ```ts
 * import { coordinateJacobian } from 'mdml-engine/foundations'
 * coordinateJacobian('polar', [2, 0]) // => [[1, 0], [0, 2]]
 * ```
 */
export function coordinateJacobian(map: CoordinateSystem | ((x: Value) => Value), point: TensorLike): number[][] {
  const f = typeof map === 'string' ? TO_CARTESIAN[map] : map
  return (toArray(jacobian(f)(point)) as number[][]).map((r) => r.map((v) => (Object.is(v, -0) ? 0 : v)))
}
