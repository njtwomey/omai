/**
 * The penalties and constraints that smooth terms are built from: difference penalties (P-splines) and integrated
 * squared derivative penalties of B-spline bases, the null space of a penalty and shrinkage that penalises it too, the
 * sum-to-zero identifiability constraint and the reparameterisation that absorbs linear constraints, and Pya and Wood's
 * monotone and convex shape constraints.
 */

import { MdmlError } from '../../base/index.ts'
import { bspline } from '../../numerics/interpolate/index.ts'
import { dsyev } from '../../numerics/linalg/index.ts'
import { gaussLegendre } from '../../numerics/quadrature/index.ts'
import {
  asTensor,
  concreteOrThrow,
  exp,
  matmul,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { mat, rowsOf } from '../_util.ts'

/**
 * The difference penalty D^T D of order `order` on k coefficients (Eilers and Marx's P-spline penalty): beta^T S beta
 * is the sum of squared order-th differences of adjacent coefficients.
 *
 * @param k - The number of coefficients.
 * @param order - The difference order (default 2).
 * @returns The k x k penalty.
 * @example
 * ```ts
 * import { differencePenalty } from 'mdml-engine/stats/smoothing'
 * differencePenalty(3, 1) // => [[1, -1, 0], [-1, 2, -1], [0, -1, 1]]
 * ```
 */
export function differencePenalty(k: number, order = 2): Tensor {
  if (!(Number.isInteger(order) && order >= 0 && order < k))
    throw new MdmlError('value', 'differencePenalty: order must be in [0, k)')
  // Rows of the difference matrix by repeated differencing of the identity.
  let D: number[][] = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => (i === j ? 1 : 0)))
  for (let o = 0; o < order; o++) D = D.slice(1).map((row, i) => row.map((v, j) => v - (D[i]?.[j] ?? 0)))
  const S = new Float64Array(k * k)
  for (const row of D)
    for (let a = 0; a < k; a++)
      for (let b = 0; b < k; b++) S[a * k + b] = (S[a * k + b] ?? 0) + (row[a] ?? 0) * (row[b] ?? 0)
  return mat(S, k, k)
}

/**
 * The derivative penalty S_ij = integral B_i^(m)(x) B_j^(m)(x) dx of a B-spline basis over its domain, exact by
 * Gauss-Legendre quadrature on each knot span (O'Sullivan's penalty; m = 2 gives the integrated squared second
 * derivative of cubic smoothing splines).
 *
 * @param knots - The full knot vector (with the boundary knots repeated degree + 1 times, as for `bspline`).
 * @param degree - The spline degree.
 * @param order - The derivative order m (default 2).
 * @returns The penalty over the n = len(knots) - degree - 1 basis functions.
 * @example
 * ```ts
 * import { derivativePenalty } from 'mdml-engine/stats/smoothing'
 * derivativePenalty([0, 0, 1, 1], 1, 1) // => [[1, -1], [-1, 1]]
 * ```
 */
export function derivativePenalty(knots: TensorLike, degree: number, order = 2): Tensor {
  const t = toFlat(asTensor(knots))
  const n = t.length - degree - 1
  if (n < 1) throw new MdmlError('value', 'derivativePenalty: too few knots for the degree')
  const identity = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  const spline = bspline(t, identity, degree)
  const d = order === 0 ? spline : spline.derivative(order)
  const q = Math.max(1, degree - order + 1)
  const rule = gaussLegendre(q)
  const nodes = toFlat(rule.nodes)
  const weights = toFlat(rule.weights)
  const S = new Float64Array(n * n)
  const lo = t[degree] ?? 0
  const hi = t[n] ?? 0
  for (let s = degree; s < n; s++) {
    const a = Math.max(lo, t[s] ?? 0)
    const b = Math.min(hi, t[s + 1] ?? 0)
    if (!(b > a)) continue
    const xs = nodes.map((u) => ((b - a) * u + (a + b)) / 2)
    const vals = toFlat(d.evaluate(xs) as Tensor)
    for (let p = 0; p < xs.length; p++) {
      const w = ((weights[p] ?? 0) * (b - a)) / 2
      for (let i = 0; i < n; i++) {
        const vi = vals[p * n + i] ?? 0
        if (vi === 0) continue
        for (let j = 0; j < n; j++) S[i * n + j] = (S[i * n + j] ?? 0) + w * vi * (vals[p * n + j] ?? 0)
      }
    }
  }
  return mat(S, n, n)
}

/** The null space of a penalty. */
export interface PenaltyNullSpace {
  /** An orthonormal basis of the null space, k x M. */
  readonly basis: Tensor
  readonly dimension: number
  /** The penalty's eigenvalues, ascending. */
  readonly eigenvalues: Tensor
}

/**
 * The null space of a symmetric penalty: the unpenalised functions (eigenvalues below tol times the largest).
 *
 * @param S - The k x k penalty.
 * @param options - tol (default 1e-9).
 * @returns An orthonormal basis of the null space and its dimension.
 * @example
 * ```ts
 * import { differencePenalty, nullSpace } from 'mdml-engine/stats/smoothing'
 * nullSpace(differencePenalty(5, 2)).dimension // => 2
 * ```
 */
export function nullSpace(S: TensorLike, options: { readonly tol?: number } = {}): PenaltyNullSpace {
  const m = rowsOf('nullSpace', S)
  const k = m.rows
  const { w, v } = dsyev(m.data, k)
  let top = 0
  for (const e of w) top = Math.max(top, Math.abs(e))
  const tol = (options.tol ?? 1e-9) * top
  const cols: number[] = []
  for (let c = 0; c < k; c++) if (Math.abs(w[c] ?? 0) <= tol) cols.push(c)
  const basis = new Float64Array(k * cols.length)
  for (let i = 0; i < k; i++)
    cols.forEach((c, j) => {
      basis[i * cols.length + j] = v[i * k + c] ?? 0
    })
  return { basis: mat(basis, k, cols.length), dimension: cols.length, eigenvalues: asTensor(Array.from(w)) as Tensor }
}

/**
 * Add shrinkage to a penalty (mgcv's "ts" and "cs" smooths): the null space is penalised by epsilon times the smallest
 * positive eigenvalue, so that the smoothing parameter can shrink the whole term to zero.
 *
 * @param S - The penalty.
 * @param options - epsilon (default 0.1) and the null-space tol.
 * @returns The shrunk penalty, of full rank.
 * @example
 * ```ts
 * import { differencePenalty, nullSpace, shrinkagePenalty } from 'mdml-engine/stats/smoothing'
 * nullSpace(shrinkagePenalty(differencePenalty(5, 2))).dimension // => 0
 * ```
 */
export function shrinkagePenalty(
  S: TensorLike,
  options: { readonly epsilon?: number; readonly tol?: number } = {},
): Tensor {
  const m = rowsOf('shrinkagePenalty', S)
  const k = m.rows
  const { w, v } = dsyev(m.data, k)
  let top = 0
  for (const e of w) top = Math.max(top, Math.abs(e))
  const tol = (options.tol ?? 1e-9) * top
  let smallest = Number.POSITIVE_INFINITY
  for (const e of w) if (e > tol) smallest = Math.min(smallest, e)
  const add = (options.epsilon ?? 0.1) * (Number.isFinite(smallest) ? smallest : 1)
  const out = Float64Array.from(m.data)
  for (let c = 0; c < k; c++) {
    if (Math.abs(w[c] ?? 0) > tol) continue
    for (let i = 0; i < k; i++)
      for (let j = 0; j < k; j++)
        out[i * k + j] = (out[i * k + j] ?? 0) + add * (v[i * k + c] ?? 0) * (v[j * k + c] ?? 0)
  }
  return mat(out, k, k)
}

/**
 * The sum-to-zero (identifiability) constraint of a smooth (mgcv's and patsy's centring): the constraint c^T beta = 0
 * with c the column means of the model matrix, absorbed by the orthogonal complement Z of c from a Householder QR
 * (LAPACK's convention, as scipy.linalg.qr), so the constrained basis is X Z.
 *
 * @param X - The n x k model matrix.
 * @returns Z, k x (k - 1).
 * @example
 * ```ts
 * import { sumToZeroConstraint } from 'mdml-engine/stats/smoothing'
 * sumToZeroConstraint([[1, 0], [0, 1]]).shape // => [2, 1]
 * ```
 */
export function sumToZeroConstraint(X: TensorLike): Tensor {
  const m = rowsOf('sumToZeroConstraint', X)
  const c = new Float64Array(m.cols)
  for (let i = 0; i < m.rows; i++)
    for (let j = 0; j < m.cols; j++) c[j] = (c[j] ?? 0) + (m.data[i * m.cols + j] ?? 0) / m.rows
  return nullSpaceOf(c, m.cols)
}

/**
 * The orthogonal complement of the row space of linear constraints C beta = 0, from the full Householder QR of C^T
 * (LAPACK's reflector convention): the last k - r columns of Q.
 *
 * Takes tensors (migration, D-009: `C` was a flat `ArrayLike<number>` with the width `k` as a second argument; the width
 * is now the matrix's own, and a flat array of numbers is one constraint).
 *
 * @param C - The constraints, `[r, k]` (one row per constraint), or a vector of length `k` for a single constraint.
 * @returns Z, k x (k - r), with C Z = 0 and orthonormal columns.
 * @throws MdmlError with code `shape` unless `C` has rank 1 or 2.
 * @example
 * ```ts
 * import { constraintNullSpace } from 'mdml-engine/stats/smoothing'
 * constraintNullSpace([1, 1]) // => [[-0.707107], [0.707107]]
 * constraintNullSpace([[1, 1]]) // => [[-0.707107], [0.707107]]
 * ```
 */
export function constraintNullSpace(C: TensorLike): Tensor {
  const t = concreteOrThrow(asTensor(C), 'constraintNullSpace')
  if (t.shape.length !== 1 && t.shape.length !== 2)
    throw new MdmlError('shape', `constraintNullSpace: expected a [r, k] matrix or a vector, got rank ${t.shape.length}`)
  const k = t.shape.length === 1 ? (t.shape[0] ?? 0) : (t.shape[1] ?? 0)
  return nullSpaceOf(t.data, k)
}

/** The kernel of {@link constraintNullSpace} on a flat row-major `r x k` matrix. */
function nullSpaceOf(C: ArrayLike<number>, k: number): Tensor {
  const r = Math.round(C.length / k)
  // A = C^T (k x r); apply Householder reflections H_1..H_r and accumulate Q = H_1 ... H_r.
  const A = Float64Array.from({ length: k * r }, (_, idx) => C[(idx % r) * k + Math.floor(idx / r)] ?? 0)
  const Q = new Float64Array(k * k)
  for (let i = 0; i < k; i++) Q[i * k + i] = 1
  for (let j = 0; j < r; j++) {
    let norm = 0
    for (let i = j; i < k; i++) norm += (A[i * r + j] ?? 0) ** 2
    norm = Math.sqrt(norm)
    const alpha = A[j * r + j] ?? 0
    if (norm === 0) continue
    const beta = alpha >= 0 ? -norm : norm
    const tau = (beta - alpha) / beta
    const v = new Float64Array(k)
    v[j] = 1
    for (let i = j + 1; i < k; i++) v[i] = (A[i * r + j] ?? 0) / (alpha - beta)
    // A <- H A, Q <- Q H, with H = I - tau v v^T.
    for (let c = 0; c < r; c++) {
      let s = 0
      for (let i = j; i < k; i++) s += (v[i] ?? 0) * (A[i * r + c] ?? 0)
      for (let i = j; i < k; i++) A[i * r + c] = (A[i * r + c] ?? 0) - tau * (v[i] ?? 0) * s
    }
    for (let row = 0; row < k; row++) {
      let s = 0
      for (let i = j; i < k; i++) s += (Q[row * k + i] ?? 0) * (v[i] ?? 0)
      for (let i = j; i < k; i++) Q[row * k + i] = (Q[row * k + i] ?? 0) - tau * s * (v[i] ?? 0)
    }
  }
  const out = new Float64Array(k * (k - r))
  for (let i = 0; i < k; i++) for (let j = r; j < k; j++) out[i * (k - r) + (j - r)] = Q[i * k + j] ?? 0
  return mat(out, k, k - r)
}

/** A shape-constrained reparameterisation beta = Sigma gamma~ (Pya and Wood's shape-constrained P-splines). */
export interface ShapeConstraint {
  /** The k x k matrix Sigma. */
  readonly sigma: Tensor
  /** Map unconstrained gamma to coefficients beta satisfying the constraint (differentiable). */
  readonly transform: (gamma: Value) => Value
}

/**
 * A shape-constraint reparameterisation of B-spline coefficients (Pya and Wood 2015): monotone (increasing or
 * decreasing coefficients, hence a monotone spline) or convex/concave (increasing first differences), by beta =
 * Sigma gamma~ with gamma~_1 = gamma_1 (and gamma~_2 = gamma_2 for convexity) and the remaining entries exp(gamma_j) >
 * 0, so any unconstrained gamma gives a constrained spline.
 *
 * @param k - The number of coefficients.
 * @param shape - `'increasing'`, `'decreasing'`, `'convex'` or `'concave'`.
 * @returns Sigma and the transform.
 * @example
 * ```ts
 * import { shapeConstraint } from 'mdml-engine/stats/smoothing'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(shapeConstraint(3, 'increasing').transform(tensor([1, 0, 0]))) // => [1, 2, 3]
 * toFlat(shapeConstraint(4, 'convex').transform(tensor([0, 0, 0, 0]))) // => [0, 0, 1, 3]
 * ```
 */
export function shapeConstraint(k: number, shape: 'increasing' | 'decreasing' | 'convex' | 'concave'): ShapeConstraint {
  const convex = shape === 'convex' || shape === 'concave'
  const sign = shape === 'decreasing' || shape === 'concave' ? -1 : 1
  const free = convex ? 2 : 1
  // Monotone: beta_i = gamma_1 + sum_{2 <= j <= i} gamma~_j. Convex: beta_i = gamma_1 + gamma_2 i + sum_{j >= 3}
  // (i - j + 2)_+ gamma~_j, so second differences are the positive gamma~_j.
  const M = new Float64Array(k * k)
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k; j++) {
      let v: number
      if (convex) v = j === 0 ? 1 : j === 1 ? i : Math.max(0, i - j + 1)
      else v = j <= i ? 1 : 0
      M[i * k + j] = j >= free ? sign * v : v
    }
  const sigma = mat(M, k, k)
  const mask = asTensor(
    Array.from({ length: k }, (_, i) => i < free),
    'bool',
  )
  const transform = (gamma: Value): Value => matmul(sigma, where(mask, gamma, exp(gamma)))
  return { sigma, transform }
}
