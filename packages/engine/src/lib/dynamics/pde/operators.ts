/**
 * Differential operators of fields defined pointwise on batches of points, computed by automatic differentiation rather
 * than grids: partial derivatives, gradient, divergence, Laplacian, curl and time derivative, plus a PDE residual
 * helper for the losses of physics-informed neural networks.
 */

import { jvp } from '../../autodiff/index.ts'
import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  mul,
  reshape,
  sliceAxis,
  stack,
  sub,
  sum,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'

/** A field evaluated pointwise on a batch of points `[m, d]`, returning `[m]` (scalar) or `[m, k]` (vector). */
export type PointField = (x: Value) => Value

function basis(x: Value, i: number): Value {
  const m = x.shape[0] ?? 0
  const d = x.shape[1] ?? 0
  const e = new Float64Array(m * d)
  for (let r = 0; r < m; r++) e[r * d + i] = 1
  return fromStorage(e, [m, d], 'float64')
}

function points(x: TensorLike): Value {
  const v = asTensor(x)
  if (v.shape.length !== 2) throw new WhiteboxError('shape', 'differential operators take points as [m, d]')
  return v
}

/**
 * The partial derivative d^k u / dx_i^k of a pointwise field at a batch of points, by k nested forward-mode passes
 * along the i-th coordinate of every point at once (valid because the field acts on each point separately). Built
 * from autodiff transforms, so it is itself differentiable in the parameters the field closes over (the residual
 * machinery of physics-informed networks).
 *
 * @param u - The field `[m, d] -> [m]` (or `[m, k]`).
 * @param i - The coordinate.
 * @param order - The derivative order (default 1).
 * @returns A function of the points.
 * @throws WhiteboxError with code `value` for a negative order.
 * @concept physics-informed-networks
 * @example
 * ```ts
 * import { partialDerivative } from 'whitebox-engine/dynamics/pde'
 * import { mul, sliceAxis, reshape, tensor, toFlat } from 'whitebox-engine/tensor'
 * // u = x^2 y at (1, 2) and (3, 1): u_xx = 2 y.
 * const u = (p) => { const x = reshape(sliceAxis(p, 1, 0, 1), [-1]); const y = reshape(sliceAxis(p, 1, 1, 2), [-1]); return mul(mul(x, x), y) }
 * toFlat(partialDerivative(u, 0, 2)(tensor([[1, 2], [3, 1]]))) // => [4, 2]
 * ```
 */
export function partialDerivative(u: PointField, i: number, order = 1): (x: TensorLike) => Value {
  if (!Number.isInteger(order) || order < 0) throw new WhiteboxError('value', 'partialDerivative: order >= 0')
  return (x) => {
    const p = points(x)
    let f: PointField = u
    for (let k = 0; k < order; k++) {
      const g = f
      f = (q) => jvp(g, q, basis(q, i)).tangent
    }
    return f(p)
  }
}

/**
 * The gradient of a scalar pointwise field: `[m, d]` with row r the gradient at point r.
 *
 * @param u - The field `[m, d] -> [m]`.
 * @returns A function of the points.
 * @concept physics-informed-networks
 * @example
 * ```ts
 * import { gradientOf } from 'whitebox-engine/dynamics/pde'
 * import { mul, sum, tensor, toArray } from 'whitebox-engine/tensor'
 * // u = |x|^2 / 2 per point: the gradient is x.
 * toArray(gradientOf((p) => mul(sum(mul(p, p), 1), 0.5))(tensor([[1, 2]]))) // => [[1, 2]]
 * ```
 */
export function gradientOf(u: PointField): (x: TensorLike) => Value {
  return (x) => {
    const p = points(x)
    const d = p.shape[1] ?? 0
    return stack(
      Array.from({ length: d }, (_, i) => partialDerivative(u, i)(p)),
      1,
    )
  }
}

/**
 * The divergence of a vector pointwise field F: `[m, d] -> [m, d]`, sum_i dF_i/dx_i, as `[m]`.
 *
 * @param F - The field.
 * @returns A function of the points.
 * @concept physics-informed-networks
 * @example
 * ```ts
 * import { divergenceOf } from 'whitebox-engine/dynamics/pde'
 * import { mul, tensor, toFlat } from 'whitebox-engine/tensor'
 * // F = 3 x: divergence 3 d.
 * toFlat(divergenceOf((p) => mul(p, 3))(tensor([[1, 2], [0, 5]]))) // => [6, 6]
 * ```
 */
export function divergenceOf(F: PointField): (x: TensorLike) => Value {
  return (x) => {
    const p = points(x)
    const d = p.shape[1] ?? 0
    let acc: Value | undefined
    for (let i = 0; i < d; i++) {
      const di = reshape(sliceAxis(jvp(F, p, basis(p, i)).tangent, 1, i, i + 1), [-1])
      acc = acc === undefined ? di : add(acc, di)
    }
    return acc ?? mul(sum(p, 1), 0)
  }
}

/**
 * The Laplacian of a scalar pointwise field, sum_i d^2 u / dx_i^2, as `[m]` (forward over forward mode).
 *
 * @param u - The field `[m, d] -> [m]`.
 * @returns A function of the points.
 * @concept physics-informed-networks
 * @example
 * ```ts
 * import { laplacianOf } from 'whitebox-engine/dynamics/pde'
 * import { mul, sum, tensor, toFlat } from 'whitebox-engine/tensor'
 * toFlat(laplacianOf((p) => sum(mul(p, p), 1))(tensor([[1, 2, 3]]))) // => [6]
 * ```
 */
export function laplacianOf(u: PointField): (x: TensorLike) => Value {
  return (x) => {
    const p = points(x)
    const d = p.shape[1] ?? 0
    let acc: Value | undefined
    for (let i = 0; i < d; i++) {
      const di = partialDerivative(u, i, 2)(p)
      acc = acc === undefined ? di : add(acc, di)
    }
    return acc ?? mul(sum(p, 1), 0)
  }
}

/**
 * The curl of a 2-D or 3-D vector pointwise field: the scalar dF_y/dx - dF_x/dy `[m]` in 2-D, the vector `[m, 3]` in
 * 3-D.
 *
 * @param F - The field `[m, d] -> [m, d]` with d = 2 or 3.
 * @returns A function of the points.
 * @throws WhiteboxError with code `shape` for other dimensions.
 * @concept physics-informed-networks
 * @example
 * ```ts
 * import { curlOf } from 'whitebox-engine/dynamics/pde'
 * import { neg, sliceAxis, concat, tensor, toFlat } from 'whitebox-engine/tensor'
 * // F = (-y, x): curl 2.
 * toFlat(curlOf((p) => concat([neg(sliceAxis(p, 1, 1, 2)), sliceAxis(p, 1, 0, 1)], 1))(tensor([[1, 1]]))) // => [2]
 * ```
 */
export function curlOf(F: PointField): (x: TensorLike) => Value {
  return (x) => {
    const p = points(x)
    const d = p.shape[1] ?? 0
    const J = (j: number, i: number): Value => reshape(sliceAxis(jvp(F, p, basis(p, i)).tangent, 1, j, j + 1), [-1])
    if (d === 2) return sub(J(1, 0), J(0, 1))
    if (d === 3) return stack([sub(J(2, 1), J(1, 2)), sub(J(0, 2), J(2, 0)), sub(J(1, 0), J(0, 1))], 1)
    throw new WhiteboxError('shape', 'curlOf: the field must be 2-D or 3-D')
  }
}

/**
 * The time derivative du/dt of a space-time field u(t, x) at a batch of (t, x) pairs, by forward mode in t: the
 * residual term u_t of evolution equations in physics-informed networks.
 *
 * @param u - The field (t `[m]`, x `[m, d]`) -> `[m]`.
 * @returns A function of (t, x).
 * @concept physics-informed-networks
 * @example
 * ```ts
 * import { timeDerivativeOf } from 'whitebox-engine/dynamics/pde'
 * import { mul, sum, tensor, toFlat } from 'whitebox-engine/tensor'
 * // u = t x: u_t = x.
 * toFlat(timeDerivativeOf((t, x) => mul(t, sum(x, 1)))(tensor([0.5, 2]), tensor([[3], [4]]))) // => [3, 4]
 * ```
 */
export function timeDerivativeOf(u: (t: Value, x: Value) => Value): (t: TensorLike, x: TensorLike) => Value {
  return (t, x) => {
    const tv = asTensor(t)
    const xv = points(x)
    const ones = fromStorage(new Float64Array(tv.shape[0] ?? 0).fill(1), [tv.shape[0] ?? 0], 'float64')
    return jvp((s: Value) => u(s, xv), tv, ones).tangent
  }
}

/** The derivatives a PDE residual may use, all at the collocation points. */
export interface FieldDerivatives {
  readonly u: Value
  /** `[m, d]` */
  readonly grad: Value
  /** `[m]` */
  readonly laplacian: Value
  /** d^2 u / dx_i^2 per coordinate, `[m, d]`. */
  readonly second: Value
}

/**
 * Evaluate a PDE residual of a pointwise field at collocation points from its value, gradient, Laplacian and second
 * derivatives (for a physics-informed loss, the mean square of the result).
 *
 * @param u - The field `[m, d] -> [m]`.
 * @param residual - The residual from the derivatives and the points (for Poisson: laplacian - f(x)).
 * @param x - Collocation points `[m, d]`.
 * @returns The residual `[m]`.
 * @concept physics-informed-networks
 * @example
 * ```ts
 * import { pdeResidual } from 'whitebox-engine/dynamics/pde'
 * import { mul, sub, sum, tensor, toFlat } from 'whitebox-engine/tensor'
 * // u = x^2 + y^2 solves Delta u = 4.
 * toFlat(pdeResidual((p) => sum(mul(p, p), 1), (d) => sub(d.laplacian, 4), tensor([[1, 2], [0.5, 0.1]]))) // => [0, 0]
 * ```
 */
export function pdeResidual(u: PointField, residual: (d: FieldDerivatives, x: Value) => Value, x: TensorLike): Value {
  const p = points(x)
  const d = p.shape[1] ?? 0
  const second = stack(
    Array.from({ length: d }, (_, i) => partialDerivative(u, i, 2)(p)),
    1,
  )
  return residual({ u: u(p), grad: gradientOf(u)(p), laplacian: sum(second, 1), second }, p)
}
