/**
 * Proximal operators and projections, the building blocks of nonsmooth optimisation (the proximal operator of g maps x
 * to the point that balances a small g against staying near x): soft thresholding and the L2, elastic-net, group-lasso,
 * nuclear-norm, Huber and L-infinity proxes, the conjugate prox by Moreau decomposition, and projections onto boxes,
 * simplices, norm balls, affine sets, half-spaces and the PSD cone.
 */

import { MdmlError } from '../base/index.ts'
import { eigh, solve, svd } from '../numerics/linalg/index.ts'
import {
  abs,
  add,
  asTensor,
  clip,
  div,
  expandDims,
  fromStorage,
  lessEqual,
  matmul,
  maximum,
  minimum,
  mul,
  neg,
  norm,
  primalOf,
  relu,
  reshape,
  segmentSum,
  sign,
  sqrt,
  square,
  sub,
  sum,
  swapaxes,
  type TensorLike,
  take,
  toFlat,
  type Value,
  where,
} from '../tensor/index.ts'

/**
 * A proximal operator: prox_{t g}(x) = argmin_u g(u) + |u - x|^2 / (2 t). Operators act on values, so they are
 * differentiable wherever the proximal map is.
 */
export type Prox = (x: Value, t: number) => Value

const flatPrimal = (x: Value): Float64Array => Float64Array.from(toFlat(primalOf(x)))

/**
 * The proximal operator of t |x|_1, soft thresholding: sign(x) max(|x| - t, 0), computed as x - clip(x, -t, t).
 *
 * @param x - The point.
 * @param t - The threshold (a number, or one per entry for a weighted L1 norm).
 * @returns The thresholded value.
 * @concept proximal-operators
 * @example
 * ```ts
 * import { proxL1 } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(proxL1([3, -0.5, -2], 1)) // => [2, 0, -1]
 * ```
 */
export function proxL1(x: TensorLike, t: TensorLike): Value {
  const v = asTensor(x)
  return sub(v, clip(v, neg(t), t))
}

/**
 * The proximal operator of t |x|_2 (block soft thresholding): x max(1 - t / |x|_2, 0).
 *
 * @param x - The point (the whole value is one block).
 * @param t - The threshold.
 * @returns The shrunk value (zero when |x|_2 <= t).
 * @example
 * ```ts
 * import { proxL2 } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(proxL2([3, 4], 2.5)) // => [1.5, 2]
 * ```
 */
export function proxL2(x: TensorLike, t: number): Value {
  const v = asTensor(x)
  if (t === 0) return v
  return mul(v, relu(sub(1, div(t, maximum(norm(v), t)))))
}

/**
 * The proximal operator of (t / 2) |x|_2^2: x / (1 + t).
 *
 * @param x - The point.
 * @param t - The step.
 * @returns The scaled value.
 * @example
 * ```ts
 * import { proxSquaredL2 } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(proxSquaredL2([2, -4], 1)) // => [1, -2]
 * ```
 */
export function proxSquaredL2(x: TensorLike, t: number): Value {
  return div(asTensor(x), 1 + t)
}

/**
 * The proximal operator of the elastic net t (l1 |x|_1 + (l2 / 2) |x|_2^2): soft thresholding at t l1, then
 * shrinkage by 1 / (1 + t l2).
 *
 * @param x - The point.
 * @param t - The step.
 * @param weights - The L1 and L2 weights.
 * @returns The proximal point.
 * @example
 * ```ts
 * import { proxElasticNet } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(proxElasticNet([3, -1], 1, { l1: 1, l2: 1 })) // => [1, 0]
 * ```
 */
export function proxElasticNet(x: TensorLike, t: number, weights: { readonly l1: number; readonly l2: number }): Value {
  return div(proxL1(x, t * weights.l1), 1 + t * weights.l2)
}

/**
 * The proximal operator of the group lasso t sum_g w_g |x_g|_2: block soft thresholding of each group.
 *
 * @param x - The point, a vector.
 * @param t - The step.
 * @param groups - The group index (0, 1, ...) of each entry.
 * @param weights - One weight per group (default 1; `sqrt` of the group sizes is a common choice).
 * @returns The proximal point.
 * @example
 * ```ts
 * import { proxGroupLasso } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(proxGroupLasso([3, 4, 0.1, 0.1], 2.5, [0, 0, 1, 1])) // => [1.5, 2, 0, 0]
 * ```
 */
export function proxGroupLasso(
  x: TensorLike,
  t: number,
  groups: readonly number[],
  weights?: readonly number[],
): Value {
  const v = reshape(asTensor(x), [-1])
  if (groups.length !== v.shape[0]) throw new MdmlError('shape', 'proxGroupLasso: one group index per entry')
  const g = groups.reduce((m, i) => Math.max(m, i + 1), 0)
  const w = Float64Array.from({ length: g }, (_, i) => t * (weights?.[i] ?? 1))
  const ids = fromStorage(Int32Array.from(groups), [groups.length], 'int32')
  const norms = sqrt(segmentSum(square(v), ids, g))
  const thresholds = fromStorage(w, [g], 'float64')
  const factor = relu(sub(1, div(thresholds, maximum(norms, thresholds))))
  return reshape(mul(v, take(factor, ids)), asTensor(x).shape)
}

/**
 * The proximal operator of the nuclear norm t |X|_* (singular value thresholding): U max(s - t, 0) V^T.
 *
 * @param x - A matrix.
 * @param t - The threshold.
 * @returns The proximal point (differentiable where singular values are distinct).
 * @example
 * ```ts
 * import { proxNuclear } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(proxNuclear([[3, 0], [0, 1]], 2)) // => [[1, 0], [0, 0]]
 * ```
 */
export function proxNuclear(x: TensorLike, t: number): Value {
  const { u, s, vh } = svd(x)
  return matmul(mul(u, expandDims(relu(sub(s, t)), -2)), vh)
}

/**
 * The proximal operator of the Huber function t h_delta (quadratic within delta, linear outside), elementwise.
 *
 * @param x - The point.
 * @param t - The step.
 * @param delta - The Huber threshold.
 * @returns The proximal point.
 * @example
 * ```ts
 * import { proxHuber } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(proxHuber([1, 10], 1, 1)) // => [0.5, 9]
 * ```
 */
export function proxHuber(x: TensorLike, t: number, delta = 1): Value {
  const v = asTensor(x)
  return where(lessEqual(abs(v), delta * (1 + t)), div(v, 1 + t), sub(v, mul(sign(v), t * delta)))
}

/**
 * The proximal operator of t |x|_inf, by the Moreau decomposition: x - P_{t B_1}(x).
 *
 * @param x - The point.
 * @param t - The step.
 * @returns The proximal point.
 * @example
 * ```ts
 * import { proxLinf } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(proxLinf([3, 1], 1)) // => [2, 1]
 * ```
 */
export function proxLinf(x: TensorLike, t: number): Value {
  const v = asTensor(x)
  return sub(v, projectL1Ball(v, t))
}

/**
 * The proximal operator of the convex conjugate g*, from that of g by the Moreau decomposition:
 * prox_{t g*}(x) = x - t prox_{g / t}(x / t).
 *
 * @param prox - The proximal operator of g.
 * @returns The proximal operator of g*.
 * @example
 * ```ts
 * import { proxConjugate, proxL1 } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * // The conjugate of |x|_1 is the indicator of the unit infinity ball: its prox clips to [-1, 1].
 * toArray(proxConjugate((x, t) => proxL1(x, t))(tensor([3, -0.5]), 2)) // => [1, -0.5]
 * ```
 */
export function proxConjugate(prox: Prox): Prox {
  return (x, t) => sub(x, mul(prox(div(x, t), 1 / t), t))
}

/**
 * Euclidean projection onto the box lo <= x <= hi (the proximal operator of its indicator).
 *
 * @param x - The point.
 * @param lo - Lower bounds (a number or one per entry; -Infinity for none).
 * @param hi - Upper bounds.
 * @returns The projection.
 * @concept projections
 * @example
 * ```ts
 * import { projectBox } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(projectBox([-2, 0.5, 3], 0, 1)) // => [0, 0.5, 1]
 * ```
 */
export function projectBox(x: TensorLike, lo: TensorLike, hi: TensorLike): Value {
  return clip(asTensor(x), lo, hi)
}

/**
 * Euclidean projection onto the nonnegative orthant: max(x, 0).
 *
 * @param x - The point.
 * @returns The projection.
 * @example
 * ```ts
 * import { projectNonnegative } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(projectNonnegative([-1, 2])) // => [0, 2]
 * ```
 */
export function projectNonnegative(x: TensorLike): Value {
  return maximum(asTensor(x), 0)
}

/** The simplex threshold tau and the support mask for projecting `a` onto {x >= 0, sum x = z}. */
function simplexThreshold(a: Float64Array, z: number): { readonly mask: Float64Array; readonly k: number } {
  const sorted = Float64Array.from(a).sort().reverse()
  let cumulative = 0
  let k = 0
  let total = 0
  for (let i = 0; i < sorted.length; i++) {
    cumulative += sorted[i] ?? 0
    if ((sorted[i] ?? 0) - (cumulative - z) / (i + 1) > 0) {
      k = i + 1
      total = cumulative
    }
  }
  const tau = (total - z) / Math.max(k, 1)
  return { mask: Float64Array.from(a, (v) => (v - tau > 0 ? 1 : 0)), k: Math.max(k, 1) }
}

/**
 * Euclidean projection onto the simplex {x >= 0, sum x = z} of the whole value (the sort-based algorithm of Held,
 * Wolfe and Crowder, as in Duchi et al. 2008), differentiable: on the support the derivative is the centring map.
 *
 * @param x - The point (flattened for the sum).
 * @param z - The simplex's total (default 1).
 * @returns The projection.
 * @concept projections
 * @example
 * ```ts
 * import { projectSimplex } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(projectSimplex([0.5, 1, -1])) // => [0.25, 0.75, 0]
 * ```
 */
export function projectSimplex(x: TensorLike, z = 1): Value {
  if (!(z > 0)) throw new MdmlError('value', `projectSimplex: the total must be positive, got ${z}`)
  const v = asTensor(x)
  const { mask, k } = simplexThreshold(flatPrimal(v), z)
  const m = fromStorage(mask, v.shape, 'float64')
  const tau = div(sub(sum(mul(v, m)), z), k)
  return mul(relu(sub(v, tau)), m)
}

/**
 * Euclidean projection onto the L1 ball {|x|_1 <= r} (Duchi et al. 2008): x itself inside the ball, otherwise
 * sign(x) times the projection of |x| onto the simplex of total r.
 *
 * @param x - The point.
 * @param r - The radius.
 * @returns The projection.
 * @concept projections
 * @example
 * ```ts
 * import { projectL1Ball } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(projectL1Ball([2, -1], 1)) // => [1, 0]
 * ```
 */
export function projectL1Ball(x: TensorLike, r = 1): Value {
  const v = asTensor(x)
  const a = flatPrimal(v)
  let l1 = 0
  for (const e of a) l1 += Math.abs(e)
  if (l1 <= r) return v
  return mul(sign(v), projectSimplex(abs(v), r))
}

/**
 * Euclidean projection onto the L2 ball {|x - c|_2 <= r}.
 *
 * @param x - The point.
 * @param r - The radius.
 * @param center - The centre (default 0).
 * @returns The projection.
 * @concept projections
 * @example
 * ```ts
 * import { projectL2Ball } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(projectL2Ball([6, 8], 5)) // => [3, 4]
 * ```
 */
export function projectL2Ball(x: TensorLike, r = 1, center: TensorLike = 0): Value {
  const d = sub(asTensor(x), center)
  return add(center, mul(d, minimum(1, div(r, maximum(norm(d), r)))))
}

/**
 * Euclidean projection onto the affine set {x : A x = b}: x - A^T (A A^T)^{-1} (A x - b), A of full row rank.
 *
 * @param x - The point, a vector `[n]`.
 * @param a - The matrix `[m, n]`.
 * @param b - The right-hand side `[m]`.
 * @returns The projection.
 * @concept projections
 * @example
 * ```ts
 * import { projectAffine } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(projectAffine([1, 1], [[1, 1]], [0])) // => [0, 0]
 * ```
 */
export function projectAffine(x: TensorLike, a: TensorLike, b: TensorLike): Value {
  const av = asTensor(a)
  const xv = reshape(asTensor(x), [-1, 1])
  const residual = sub(matmul(av, xv), reshape(asTensor(b), [-1, 1]))
  const at = swapaxes(av, 0, 1)
  return reshape(sub(xv, matmul(at, solve(matmul(av, at), residual))), [-1])
}

/**
 * Euclidean projection onto the half-space {x : a . x <= b}.
 *
 * @param x - The point, a vector.
 * @param a - The normal.
 * @param b - The offset.
 * @returns The projection.
 * @concept projections
 * @example
 * ```ts
 * import { projectHalfspace } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(projectHalfspace([2, 2], [1, 1], 2)) // => [1, 1]
 * ```
 */
export function projectHalfspace(x: TensorLike, a: TensorLike, b: number): Value {
  const xv = asTensor(x)
  const av = asTensor(a)
  const excess = relu(sub(sum(mul(av, xv)), b))
  return sub(xv, mul(av, div(excess, sum(square(av)))))
}

/**
 * Euclidean (Frobenius) projection onto the nuclear-norm ball {|X|_* <= r}: the singular values are projected onto
 * the L1 ball.
 *
 * @param x - A matrix.
 * @param r - The radius.
 * @returns The projection.
 * @concept projections
 * @example
 * ```ts
 * import { projectNuclearBall } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(projectNuclearBall([[3, 0], [0, 1]], 2)) // => [[2, 0], [0, 0]]
 * ```
 */
export function projectNuclearBall(x: TensorLike, r = 1): Value {
  const { u, s, vh } = svd(x)
  return matmul(mul(u, expandDims(projectL1Ball(s, r), -2)), vh)
}

/**
 * Projection of a symmetric matrix onto the positive semidefinite cone: eigenvalues clipped at zero.
 *
 * @param x - A symmetric matrix (its symmetric part is used).
 * @returns The nearest positive semidefinite matrix in the Frobenius norm.
 * @concept projections
 * @example
 * ```ts
 * import { projectPsd } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(projectPsd([[1, 0], [0, -1]])) // => [[1, 0], [0, 0]]
 * ```
 */
export function projectPsd(x: TensorLike): Value {
  const v = asTensor(x)
  const symmetric = mul(add(v, swapaxes(v, -1, -2)), 0.5)
  const { eigenvalues, eigenvectors } = eigh(symmetric)
  return matmul(mul(eigenvectors, expandDims(relu(eigenvalues), -2)), swapaxes(eigenvectors, -1, -2))
}
