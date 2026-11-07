/**
 * The QR factorisation by Householder reflections (numpy.linalg `qr`, reduced) and the least-squares and minimum-norm
 * solutions built on it (`lstsq`), plus the elementary building blocks: Householder reflectors and Givens rotations.
 */

import { MdmlError } from '../../base/index.ts'
import {
  add,
  arg,
  batchLeading,
  bind,
  definePrimitive,
  type FloatDType,
  matmul,
  type Primitive,
  primalOf,
  promoteFloat,
  squeeze,
  sub,
  type TensorLike,
  toFlat,
  tril,
  type Value,
} from '../../tensor/index.ts'
import { qrKernel } from './_kernels.ts'
import { asColumns, assemble, broadcastBatch, matrices, matrixShape, mT } from './_matrix.ts'
import { pinv } from './svd.ts'
import { triangularSolve } from './triangular.ts'

const upper = (transpose: boolean) => ({ lower: false, transpose, unitDiagonal: false })

// Y R^{-1} for upper-triangular R, as (R^{-T} Y^T)^T.
const rightSolve = (y: Value, r: Value): Value => mT(triangularSolve(r, mT(y), upper(true)))
// Y R^{-T}, as (R^{-1} Y^T)^T.
const rightSolveTransposed = (y: Value, r: Value): Value => mT(triangularSolve(r, mT(y), upper(false)))

function requireTall(shape: readonly number[]): void {
  const { m, n } = matrixShape(shape, 'qr')
  if (m < n) throw new MdmlError('not-differentiable', 'qr: derivatives need m >= n (full column rank)')
}

const qrPrimitive: Primitive<null> = definePrimitive<null>({
  name: 'qr',
  abstractEval: (xs) => {
    const a = arg(xs, 0)
    const { batch, m, n } = matrixShape(a.shape, 'qr')
    const k = Math.min(m, n)
    return [
      { shape: [...batch, m, k], dtype: a.dtype },
      { shape: [...batch, k, n], dtype: a.dtype },
    ]
  },
  impl: ([a]) => {
    const av = a as NonNullable<typeof a>
    const { batch, m, n } = matrixShape(av.shape, 'qr')
    const k = Math.min(m, n)
    const results = matrices(av).map((x) => qrKernel(x, m, n))
    const dtype = av.dtype as FloatDType
    return [
      assemble(
        av,
        dtype,
        batch,
        [m, k],
        results.map((r) => r.q),
      ),
      assemble(
        av,
        dtype,
        batch,
        [k, n],
        results.map((r) => r.r),
      ),
    ]
  },
  // With Y = dA R^{-1}, C = Q^T Y and Omega = tril(C, -1) - tril(C, -1)^T: dR = (C - Omega) R, dQ = Q Omega + Y - Q C.
  jvp: ([a], [ta], _p, [q, r]) => {
    if (ta === null || ta === undefined) return [null, null]
    requireTall((a as Value).shape)
    const qv = q as Value
    const rv = r as Value
    const y = rightSolve(ta, rv)
    const c = matmul(mT(qv), y)
    const low = tril(c, -1)
    const omega = sub(low, mT(low))
    return [add(matmul(qv, omega), sub(y, matmul(qv, c))), matmul(sub(c, omega), rv)]
  },
  // The adjoint of the rule above: with G = Q^T dQ - dR R^T and K = tril(G - G^T, -1) - G, dA = (dQ + Q K) R^{-T}.
  vjp: ([gq, gr], [a], _p, [q, r]) => {
    requireTall((a as Value).shape)
    const qv = q as Value
    const rv = r as Value
    const hasQ = gq !== null && gq !== undefined
    const hasR = gr !== null && gr !== undefined
    if (!hasQ && !hasR) return [null]
    const qtg = hasQ ? matmul(mT(qv), gq) : undefined
    const rrt = hasR ? matmul(gr, mT(rv)) : undefined
    const g = qtg === undefined ? sub(0, rrt as Value) : rrt === undefined ? qtg : sub(qtg, rrt)
    const k = sub(tril(sub(g, mT(g)), -1), g)
    const ybar = hasQ ? add(gq, matmul(qv, k)) : matmul(qv, k)
    return [rightSolveTransposed(ybar, rv)]
  },
  // Factorisations and solves already map over leading batch axes.
  batch: (xs, ds, p, n) => batchLeading(qrPrimitive, xs, ds, p, n),
})

/** Result of {@link qr}. */
export interface QrResult {
  /** Orthonormal columns `[..., m, k]`. */
  readonly q: Value
  /** Upper-triangular `[..., k, n]`. */
  readonly r: Value
}

/**
 * Reduced QR factorisation by Householder reflections, A = Q R (numpy.linalg.qr, mode 'reduced'), k = min(m, n). The
 * signs follow LAPACK's convention, as numpy's do.
 *
 * @param a - Matrix `[..., m, n]`.
 * @returns `{ q, r }`, differentiable when m >= n and A has full column rank.
 * @concept linear-algebra
 * @index function Reduced QR factorisation by Householder reflections.
 * @example
 * ```ts
 * import { qr } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(qr([[3, 0], [4, 1]]).r)[0] // => [-5, -0.8]
 * ```
 */
export function qr(a: TensorLike): QrResult {
  const [av] = promoteFloat(a) as [Value]
  const [q, r] = bind(qrPrimitive, [av], null) as [Value, Value]
  return { q, r }
}

/**
 * Whether any matrix of the batch has a triangular factor `r` `[..., k, k]` whose diagonal is rank-deficient: its smallest
 * magnitude at most max(m, n) eps times its largest (numpy.linalg.lstsq's default `rcond`, applied to the diagonal).
 */
function rankDeficient(r: Value, m: number, n: number): boolean {
  const k = Math.min(m, n)
  if (k === 0) return false
  const flat = toFlat(primalOf(r))
  const cut = Math.max(m, n) * Number.EPSILON
  for (let offset = 0; offset < flat.length; offset += k * k) {
    let smallest = Number.POSITIVE_INFINITY
    let largest = 0
    for (let i = 0; i < k; i++) {
      const d = Math.abs(flat[offset + i * k + i] ?? 0)
      smallest = Math.min(smallest, d)
      largest = Math.max(largest, d)
    }
    if (!(smallest > cut * largest)) return true
  }
  return false
}

/**
 * Least-squares solution of A x = b (numpy.linalg.lstsq's solution): the minimiser of ||A x - b|| when m >= n, and the
 * minimum-norm solution when m < n, computed from a QR factorisation. A matrix whose R factor has a diagonal entry below
 * max(m, n) eps times the largest (a collinear column, a constant feature, a repeated abscissa) is rank-deficient: the
 * solution is then the minimum-norm least-squares one, `pinv(A) b`, as numpy's (a batch with any such matrix is solved
 * that way throughout).
 *
 * @param a - Matrix `[..., m, n]`.
 * @param b - Right-hand side `[..., m, k]` or `[m]`.
 * @returns The solution `[..., n, k]` or `[n]`, differentiable in both arguments.
 * @index function Least-squares solution of A x = b.
 * @example
 * ```ts
 * import { lstsq } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(lstsq([[1, 0], [0, 1], [0, 0]], [1, 2, 3])) // => [1, 2]
 * ```
 */
export function lstsq(a: TensorLike, b: TensorLike): Value {
  const [av, bv] = promoteFloat(a, b) as [Value, Value]
  const cols = asColumns(av, bv)
  const [ab, bb] = broadcastBatch(av, cols.b)
  const { m, n } = matrixShape(ab.shape, 'lstsq')
  let x: Value
  if (m >= n) {
    const { q, r } = qr(ab)
    x = rankDeficient(r, m, n)
      ? matmul(pinv(ab), bb)
      : triangularSolve(r, matmul(mT(q), bb), upper(false))
  } else {
    // A^T = Q R, so A = R^T Q^T and the minimum-norm solution is Q R^{-T} b.
    const { q, r } = qr(mT(ab))
    x = rankDeficient(r, m, n) ? matmul(pinv(ab), bb) : matmul(q, triangularSolve(r, bb, upper(true)))
  }
  return cols.vector ? squeeze(x, -1) : x
}

/**
 * An elementary reflector (LAPACK's dlarfg): H = I - tau v v^T with v[0] = 1 maps x to alpha e1, alpha = -sign(x0) |x|
 * (tau = 0 and H = I when x needs no reflection).
 *
 * @param x - The vector.
 * @returns `{ v, tau, alpha }`.
 * @concept orthogonal-transformations
 * @example
 * ```ts
 * import { householderReflector } from 'mdml-engine/numerics/linalg'
 * householderReflector([3, 4]).alpha // => -5
 * ```
 */
export function householderReflector(x: readonly number[]): {
  readonly v: number[]
  readonly tau: number
  readonly alpha: number
} {
  const x0 = x[0] ?? 0
  const tail = x.slice(1)
  const tailNorm = Math.sqrt(tail.reduce((s, v) => s + v * v, 0))
  if (tailNorm === 0) return { v: [1, ...tail.map(() => 0)], tau: 0, alpha: x0 }
  const norm = Math.hypot(x0, tailNorm)
  const alpha = x0 >= 0 ? -norm : norm
  const scale = 1 / (x0 - alpha)
  return { v: [1, ...tail.map((t) => t * scale)], tau: (alpha - x0) / alpha, alpha }
}

/**
 * A Givens rotation: c and s with [[c, s], [-s, c]] [a, b] = [r, 0] (r = hypot(a, b) >= 0).
 *
 * @param a - First component.
 * @param b - Second component (to be zeroed).
 * @returns `{ c, s, r }`.
 * @example
 * ```ts
 * import { givensRotation } from 'mdml-engine/numerics/linalg'
 * givensRotation(3, 4) // => { c: 0.6, s: 0.8, r: 5 }
 * ```
 */
export function givensRotation(a: number, b: number): { readonly c: number; readonly s: number; readonly r: number } {
  if (b === 0) return { c: a >= 0 ? 1 : -1, s: 0, r: Math.abs(a) }
  const r = Math.hypot(a, b)
  return { c: a / r, s: b / r, r }
}
