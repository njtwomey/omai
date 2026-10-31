/**
 * Exact derivatives of the B-spline basis functions: the basis is the identity-coefficient B-spline of `numerics/
 * interpolate` on the knots of the smooth, whose piecewise polynomials differentiate exactly, so the first and second
 * derivative of every basis function at the query points are the de Boor recursion's, not finite differences.
 */

import { bspline } from '../../numerics/interpolate/index.ts'
import { fromStorage, type Tensor, toFlat } from '../../tensor/index.ts'

/**
 * The `order`th derivative of all B-spline basis functions of a knot vector and degree at the points.
 *
 * @param knots - The knot vector (boundary knots repeated), length `k + degree + 1`.
 * @param degree - The degree.
 * @param x - The points.
 * @param order - The derivative order, 1 or 2.
 * @returns The derivatives `[x.length, k]`, row-major, and `k`.
 * @concept splines
 * @example
 * ```ts
 * import { bsplineDerivativeDesign } from 'mdml-engine/stats/smoothing'
 * // On [0, 1] the cubic basis sums to one, so its derivatives sum to zero.
 * const knots = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]
 * const { data, cols } = bsplineDerivativeDesign(knots, 3, [0.3], 1)
 * Math.abs(Array.from(data).reduce((a, b) => a + b, 0)) < 1e-12 && cols === 5 // => true
 * ```
 */
export function bsplineDerivativeDesign(
  knots: readonly number[],
  degree: number,
  x: ArrayLike<number>,
  order: 1 | 2,
): { readonly data: Float64Array; readonly cols: number } {
  const cols = knots.length - degree - 1
  const identity = Float64Array.from({ length: cols * cols }, (_, i) => (Math.floor(i / cols) === i % cols ? 1 : 0))
  const spline = bspline(knots, fromStorage(identity, [cols, cols], 'float64'), degree)
  return { data: Float64Array.from(toFlat(spline.evaluate(Array.from(x), order) as Tensor)), cols }
}
