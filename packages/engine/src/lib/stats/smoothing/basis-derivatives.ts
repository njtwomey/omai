/**
 * Exact derivatives of the B-spline basis functions: the basis is the identity-coefficient B-spline of `numerics/
 * interpolate` on the knots of the smooth, whose piecewise polynomials differentiate exactly, so the first and second
 * derivative of every basis function at the query points are the de Boor recursion's, not finite differences.
 */

import { bspline } from '../../numerics/interpolate/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat, toVector } from '../../tensor/index.ts'

/**
 * The `order`th derivative of all B-spline basis functions of a knot vector and degree at the points. Takes and returns
 * tensors (migration, D-009: `knots` and `x` were `readonly number[]` and `ArrayLike<number>`, and the result a record
 * `{ data, cols }` of flat storage; the matrix now carries its own shape).
 *
 * @param knots - The knot vector (boundary knots repeated), length `k + degree + 1`: a vector.
 * @param degree - The degree.
 * @param x - The points: a vector.
 * @param order - The derivative order, 1 or 2.
 * @returns A float64 tensor of shape `[x.length, k]`: the derivatives of the `k` basis functions at the points.
 * @throws OmaiError with code `shape` unless `knots` and `x` are vectors.
 * @concept splines
 * @example
 * ```ts
 * import { bsplineDerivativeDesign } from 'omai-engine/stats/smoothing'
 * // On [0, 1] the cubic basis sums to one, so its derivatives sum to zero.
 * const knots = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]
 * const D = bsplineDerivativeDesign(knots, 3, [0.3], 1)
 * Math.abs(Array.from(D.data).reduce((a, b) => a + b, 0)) < 1e-12 && D.shape[1] === 5 // => true
 * ```
 */
export function bsplineDerivativeDesign(
  knots: TensorLike,
  degree: number,
  x: TensorLike,
  order: 1 | 2,
): Tensor<'float64'> {
  const t = toVector(knots, 'bsplineDerivativeDesign')
  const points = toVector(x, 'bsplineDerivativeDesign')
  const cols = t.length - degree - 1
  const identity = Float64Array.from({ length: cols * cols }, (_, i) => (Math.floor(i / cols) === i % cols ? 1 : 0))
  const spline = bspline(t, fromStorage(identity, [cols, cols], 'float64'), degree)
  const values = Float64Array.from(toFlat(spline.evaluate(Array.from(points), order) as Tensor))
  return fromStorage(values, [points.length, cols], 'float64')
}
