import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  arg,
  bind,
  definePrimitive,
  type FloatDType,
  matmul,
  type Primitive,
  promoteFloat,
  squeeze,
  sub,
  type TensorLike,
  tril,
  type Value,
} from '../../tensor/index.ts'
import { qrKernel } from './_kernels.ts'
import { asColumns, assemble, broadcastBatch, matrices, matrixShape, mT } from './_matrix.ts'
import { triangularSolve } from './triangular.ts'

const upper = (transpose: boolean) => ({ lower: false, transpose, unitDiagonal: false })

// Y R^{-1} for upper-triangular R, as (R^{-T} Y^T)^T.
const rightSolve = (y: Value, r: Value): Value => mT(triangularSolve(r, mT(y), upper(true)))
// Y R^{-T}, as (R^{-1} Y^T)^T.
const rightSolveTransposed = (y: Value, r: Value): Value => mT(triangularSolve(r, mT(y), upper(false)))

function requireTall(shape: readonly number[]): void {
  const { m, n } = matrixShape(shape, 'qr')
  if (m < n) throw new WhiteboxError('not-differentiable', 'qr: derivatives need m >= n (full column rank)')
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
 * @example
 * ```ts
 * import { qr } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(qr([[3, 0], [4, 1]]).r)[0] // => [-5, -0.8]
 * ```
 */
export function qr(a: TensorLike): QrResult {
  const [av] = promoteFloat(a) as [Value]
  const [q, r] = bind(qrPrimitive, [av], null) as [Value, Value]
  return { q, r }
}

/**
 * Least-squares solution of A x = b (numpy.linalg.lstsq's solution) for A of full rank: the minimiser of ||A x - b|| when
 * m >= n, and the minimum-norm solution when m < n; computed from a QR factorisation.
 *
 * @param a - Matrix `[..., m, n]` of full rank.
 * @param b - Right-hand side `[..., m, k]` or `[m]`.
 * @returns The solution `[..., n, k]` or `[n]`, differentiable in both arguments.
 * @example
 * ```ts
 * import { lstsq } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
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
    x = triangularSolve(r, matmul(mT(q), bb), upper(false))
  } else {
    // A^T = Q R, so A = R^T Q^T and the minimum-norm solution is Q R^{-T} b.
    const { q, r } = qr(mT(ab))
    x = matmul(q, triangularSolve(r, bb, upper(true)))
  }
  return cols.vector ? squeeze(x, -1) : x
}
