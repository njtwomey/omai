import { maxOf, WhiteboxError } from '../../base/index.ts'
import {
  abs,
  allclose,
  arg,
  batchLeading,
  bind,
  definePrimitive,
  div,
  exp,
  expandDims,
  type FloatDType,
  greater,
  matmul,
  max,
  min,
  mul,
  type Primitive,
  pow,
  promoteFloat,
  sqrt,
  square,
  sum,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { slogdetKernel } from './_kernels.ts'
import { assemble, identity, matrices, matrixShape, mT, squareSize, sym } from './_matrix.ts'
import { cholesky } from './cholesky.ts'
import { inv } from './lu.ts'
import { svdvals } from './svd.ts'

const slogdetPrimitive: Primitive<null> = definePrimitive<null>({
  name: 'slogdet',
  abstractEval: (xs) => {
    const a = arg(xs, 0)
    const out = { shape: a.shape.slice(0, -2), dtype: a.dtype }
    return [out, out]
  },
  impl: ([a]) => {
    const av = a as NonNullable<typeof a>
    const n = squareSize(av.shape, 'slogdet')
    const results = matrices(av).map((m) => slogdetKernel(m, n))
    const batch = av.shape.slice(0, -2)
    const dtype = av.dtype as FloatDType
    return [
      assemble(
        av,
        dtype,
        batch,
        [],
        results.map((r) => [r.sign]),
      ),
      assemble(
        av,
        dtype,
        batch,
        [],
        results.map((r) => [r.logabsdet]),
      ),
    ]
  },
  // d log|det A| = tr(A^{-1} dA); the sign is piecewise constant.
  jvp: ([a], [ta]) => {
    if (ta === null || ta === undefined) return [null, null]
    return [null, sum(sum(mul(mT(inv(a as Value)), ta), -1), -1)]
  },
  vjp: ([, g], [a]) => {
    if (g === null || g === undefined) return [null]
    return [mul(expandDims(expandDims(g, -1), -1), mT(inv(a as Value)))]
  },
  // Factorisations and solves already map over leading batch axes.
  batch: (xs, ds, p, n) => batchLeading(slogdetPrimitive, xs, ds, p, n),
})

/** Result of {@link slogdet}. */
export interface SlogdetResult {
  /** +1, -1, or 0 for a singular matrix. */
  readonly sign: Value
  /** log |det A| (-inf for a singular matrix). */
  readonly logabsdet: Value
}

/**
 * The sign and log of the absolute determinant (numpy.linalg.slogdet), without the overflow of {@link det}.
 *
 * @param a - Square matrix `[..., n, n]`.
 * @returns `{ sign, logabsdet }`; logabsdet is differentiable (derivative A^{-T}).
 * @concept linear-algebra
 * @example
 * ```ts
 * import { slogdet } from 'whitebox-engine/numerics/linalg'
 * import { item } from 'whitebox-engine/tensor'
 * item(slogdet([[0, 2], [3, 0]]).sign) // => -1
 * ```
 */
export function slogdet(a: TensorLike): SlogdetResult {
  const [av] = promoteFloat(a) as [Value]
  const [sign, logabsdet] = bind(slogdetPrimitive, [av], null) as [Value, Value]
  return { sign, logabsdet }
}

/**
 * The determinant (numpy.linalg.det), as sign exp(log |det|).
 *
 * @param a - Square matrix `[..., n, n]`.
 * @returns det A, differentiable for non-singular A.
 * @example
 * ```ts
 * import { det } from 'whitebox-engine/numerics/linalg'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(det([[1, 2], [3, 4]])) + 2) < 1e-14 // => true
 * ```
 */
export function det(a: TensorLike): Value {
  const { sign, logabsdet } = slogdet(a)
  return mul(sign, exp(logabsdet))
}

/** Matrix and vector norm orders (numpy.linalg.norm `ord`). */
export type NormOrder = 'fro' | 'nuc' | number

/**
 * Vector or matrix norm (numpy.linalg.norm). A 1-D input is a vector (`ord`: any p, Infinity, -Infinity, 0 for the
 * count of non-zeros; default 2); otherwise the last two axes are a matrix (`ord`: 'fro' (default), 'nuc', 1, -1, 2,
 * -2, Infinity, -Infinity).
 *
 * @param x - Vector or matrix (or batch of matrices).
 * @param ord - The norm.
 * @returns The norm, differentiable where the norm is.
 * @concept linear-algebra
 * @example
 * ```ts
 * import { norm } from 'whitebox-engine/numerics/linalg'
 * import { item } from 'whitebox-engine/tensor'
 * item(norm([3, 4])) // => 5
 * item(norm([[1, 2], [3, 4]], 1)) // => 6
 * ```
 */
export function norm(x: TensorLike, ord?: NormOrder): Value {
  const [v] = promoteFloat(x) as [Value]
  if (v.shape.length === 1) return vectorNorm(v, ord === undefined ? 2 : ord)
  matrixShape(v.shape, 'norm')
  const o = ord ?? 'fro'
  if (o === 'fro') return sqrt(sum(sum(square(v), -1), -1))
  if (o === 'nuc') return sum(svdvals(v), -1)
  if (o === 2) return max(svdvals(v), -1)
  if (o === -2) return min(svdvals(v), -1)
  if (o === 1) return max(sum(abs(v), -2), -1)
  if (o === -1) return min(sum(abs(v), -2), -1)
  if (o === Number.POSITIVE_INFINITY) return max(sum(abs(v), -1), -1)
  if (o === Number.NEGATIVE_INFINITY) return min(sum(abs(v), -1), -1)
  throw new WhiteboxError('value', `norm: unsupported matrix norm ${String(o)}`)
}

function vectorNorm(v: Value, ord: NormOrder): Value {
  if (ord === 'fro' || ord === 2) return sqrt(sum(square(v), -1))
  if (ord === Number.POSITIVE_INFINITY) return max(abs(v), -1)
  if (ord === Number.NEGATIVE_INFINITY) return min(abs(v), -1)
  if (ord === 0) return sum(greater(abs(v), 0), -1)
  if (ord === 1) return sum(abs(v), -1)
  if (typeof ord === 'number') return powNorm(abs(v), ord)
  throw new WhiteboxError('value', `norm: unsupported vector norm ${String(ord)}`)
}

function powNorm(a: Value, p: number): Value {
  return pow(sum(pow(a, p), -1), 1 / p)
}

/**
 * The condition number (numpy.linalg.cond): the ratio of extreme singular values for `p` = 2 (default) or -2,
 * otherwise norm(A, p) norm(A^{-1}, p).
 *
 * @param a - Matrix `[..., m, n]` (square unless p is 2 or -2).
 * @param p - The norm.
 * @returns The condition number, differentiable.
 * @example
 * ```ts
 * import { cond } from 'whitebox-engine/numerics/linalg'
 * import { item } from 'whitebox-engine/tensor'
 * item(cond([[2, 0], [0, 0.5]])) // => 4
 * ```
 */
export function cond(a: TensorLike, p: NormOrder = 2): Value {
  const [v] = promoteFloat(a) as [Value]
  if (p === 2 || p === -2) {
    const s = svdvals(v)
    return p === 2 ? div(max(s, -1), min(s, -1)) : div(min(s, -1), max(s, -1))
  }
  return mul(norm(v, p), norm(inv(v), p))
}

/**
 * The numerical rank (numpy.linalg.matrix_rank): the number of singular values above `tol` (default
 * max(s) max(m, n) eps).
 *
 * @param a - Matrix `[m, n]`.
 * @param tol - Threshold.
 * @returns The rank.
 * @example
 * ```ts
 * import { matrixRank } from 'whitebox-engine/numerics/linalg'
 * matrixRank([[1, 2], [2, 4]]) // => 1
 * ```
 */
export function matrixRank(a: TensorLike, tol?: number): number {
  const [v] = promoteFloat(a) as [Value]
  const { m, n } = matrixShape(v.shape, 'matrixRank')
  const s = toFlat(svdvals(v))
  const largest = s.length === 0 ? 0 : Math.max(0, maxOf(s))
  const threshold = tol ?? largest * Math.max(m, n) * 2.220446049250313e-16
  return s.filter((x) => x > threshold).length
}

/**
 * Integer power of a square matrix (numpy.linalg.matrix_power) by repeated squaring; negative powers invert first.
 *
 * @param a - Square matrix `[..., n, n]`.
 * @param k - The power (an integer).
 * @returns A^k, differentiable.
 * @example
 * ```ts
 * import { matrixPower } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(matrixPower([[1, 1], [0, 1]], 3)) // => [[1, 3], [0, 1]]
 * ```
 */
export function matrixPower(a: TensorLike, k: number): Value {
  if (!Number.isInteger(k)) throw new WhiteboxError('value', 'matrixPower: the power must be an integer')
  const [v] = promoteFloat(a) as [Value]
  const n = squareSize(v.shape, 'matrixPower')
  let base = k < 0 ? inv(v) : v
  let e = Math.abs(k)
  let result: Value | undefined
  while (e > 0) {
    if (e % 2 === 1) result = result === undefined ? base : matmul(result, base)
    e = Math.floor(e / 2)
    if (e > 0) base = matmul(base, base)
  }
  return result ?? mul(identity(n, v), 1)
}

/**
 * Whether a matrix equals its transpose within a tolerance.
 *
 * @param a - Matrix `[n, n]`.
 * @param tol - Absolute tolerance (default 1e-10).
 * @returns `true` when symmetric.
 * @example
 * ```ts
 * import { isSymmetric } from 'whitebox-engine/numerics/linalg'
 * isSymmetric([[1, 2], [2, 1]]) // => true
 * ```
 */
export function isSymmetric(a: TensorLike, tol = 1e-10): boolean {
  const [v] = promoteFloat(a) as [Value]
  const { m, n } = matrixShape(v.shape, 'isSymmetric')
  return m === n && allclose(v, mT(v), 0, tol)
}

/**
 * The symmetric part `(A + A^T) / 2` of a square matrix (or a batch of them, over the last two axes): the projection
 * onto symmetric matrices, used to keep Riccati and Hessian recursions symmetric under rounding. Differentiable.
 *
 * @param a - Matrix `[..., n, n]`.
 * @returns The symmetric part, same shape.
 * @throws WhiteboxError with code `shape` when the matrices are not square.
 * @example
 * ```ts
 * import { symmetricPart } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(symmetricPart([[1, 2], [4, 3]])) // => [[1, 3], [3, 3]]
 * ```
 */
export function symmetricPart(a: TensorLike): Value {
  const [v] = promoteFloat(a) as [Value]
  squareSize(v.shape, 'symmetricPart')
  return sym(v)
}

/**
 * Whether a symmetric matrix is positive definite (its Cholesky factorisation succeeds).
 *
 * @param a - Symmetric matrix `[n, n]`.
 * @returns `true` when positive definite.
 * @example
 * ```ts
 * import { isPositiveDefinite } from 'whitebox-engine/numerics/linalg'
 * isPositiveDefinite([[2, 1], [1, 2]]) // => true
 * isPositiveDefinite([[1, 2], [2, 1]]) // => false
 * ```
 */
export function isPositiveDefinite(a: TensorLike): boolean {
  if (!isSymmetric(a)) return false
  try {
    cholesky(a)
    return true
  } catch {
    return false
  }
}
