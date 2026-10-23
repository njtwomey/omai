import {
  add,
  arg,
  bind,
  bind1,
  definePrimitive,
  type FloatDType,
  type MaybeZero,
  matmul,
  neg,
  type Primitive,
  promoteFloat,
  squeeze,
  sub,
  type TensorLike,
  tril,
  triu,
  type Value,
} from '../../tensor/index.ts'
import { luKernel, solveKernel } from './_kernels.ts'
import { asColumns, assemble, broadcastBatch, identity, matrices, matrixShape, mT, squareSize } from './_matrix.ts'
import { triangularSolve } from './triangular.ts'

// ---------------------------------------------------------------------------------------------------------------------
// solve

const solvePrimitive: Primitive<null> = definePrimitive<null>({
  name: 'solve',
  abstractEval: (xs) => [arg(xs, 1)],
  impl: ([a, b]) => {
    const av = a as NonNullable<typeof a>
    const bv = b as NonNullable<typeof b>
    const n = squareSize(av.shape, 'solve')
    const { batch, n: k } = matrixShape(bv.shape, 'solve')
    const bs = matrices(bv)
    const parts = matrices(av).map((m, i) => solveKernel(m, bs[i] as Float64Array, n, k))
    return [assemble(bv, bv.dtype as FloatDType, batch, [n, k], parts)]
  },
  // X = A^{-1} B: dX = A^{-1} (dB - dA X).
  jvp: ([a], [ta, tb], _p, [x]) => {
    let rhs: MaybeZero = tb ?? null
    if (ta !== null && ta !== undefined) {
      const term = matmul(ta, x as Value)
      rhs = rhs === null ? neg(term) : sub(rhs, term)
    }
    return [rhs === null ? null : bind1(solvePrimitive, [a as Value, rhs], null)]
  },
  // dB = A^{-T} dX, dA = -dB X^T.
  vjp: ([g], [a], _p, [x]) => {
    if (g === null || g === undefined) return [null, null]
    const gb = bind1(solvePrimitive, [mT(a as Value), g], null)
    return [neg(matmul(gb, mT(x as Value))), gb]
  },
})

/**
 * Solve a square linear system A x = b by LU with partial pivoting (numpy.linalg.solve). Batches broadcast; a
 * 1-D right-hand side is a vector.
 *
 * @param a - Square matrix `[..., n, n]`.
 * @param b - Right-hand side `[..., n, k]` or `[n]`.
 * @returns The solution, differentiable in both arguments.
 * @throws WhiteboxError with code `value` when A is exactly singular.
 * @concept linear-algebra
 * @example
 * ```ts
 * import { solve } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(solve([[3, 1], [1, 2]], [9, 8])) // => [2, 3]
 * ```
 */
export function solve(a: TensorLike, b: TensorLike): Value {
  const [av, bv] = promoteFloat(a, b) as [Value, Value]
  const cols = asColumns(av, bv)
  const [ab, bb] = broadcastBatch(av, cols.b)
  const x = bind1(solvePrimitive, [ab, bb], null)
  return cols.vector ? squeeze(x, -1) : x
}

/**
 * The inverse of a square matrix (numpy.linalg.inv), as the solution of A X = I.
 *
 * @param a - Square matrix `[..., n, n]`.
 * @returns A^{-1}, differentiable.
 * @example
 * ```ts
 * import { inv } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(inv([[2, 0], [0, 4]])) // => [[0.5, 0], [0, 0.25]]
 * ```
 */
export function inv(a: TensorLike): Value {
  const [av] = promoteFloat(a) as [Value]
  const n = squareSize(av.shape, 'inv')
  return solve(av, identity(n, av))
}

// ---------------------------------------------------------------------------------------------------------------------
// LU factorisation

function permutationMatrix(perm: Int32Array, n: number): Float64Array {
  // Rows of P A are rows perm[i] of A; scipy returns P with A = P L U, i.e. the transpose: P[perm[i], i] = 1.
  const p = new Float64Array(n * n)
  perm.forEach((src, i) => {
    p[src * n + i] = 1
  })
  return p
}

const luPrimitive: Primitive<null> = definePrimitive<null>({
  name: 'lu',
  abstractEval: (xs) => [arg(xs, 0), arg(xs, 0), arg(xs, 0)],
  impl: ([a]) => {
    const av = a as NonNullable<typeof a>
    const n = squareSize(av.shape, 'lu')
    const batch = av.shape.slice(0, -2)
    const ps: Float64Array[] = []
    const ls: Float64Array[] = []
    const us: Float64Array[] = []
    for (const m of matrices(av)) {
      const { lu, perm } = luKernel(m, n)
      const l = new Float64Array(n * n)
      const u = new Float64Array(n * n)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (j < i) l[i * n + j] = lu[i * n + j] ?? 0
          else u[i * n + j] = lu[i * n + j] ?? 0
        }
        l[i * n + i] = 1
      }
      ps.push(permutationMatrix(perm, n))
      ls.push(l)
      us.push(u)
    }
    const dtype = av.dtype as FloatDType
    return [ps, ls, us].map((parts) => assemble(av, dtype, batch, [n, n], parts))
  },
  // A = P L U: with X = L^{-1} P^T dA U^{-1}, dL = L tril(X, -1) and dU = triu(X) U (P is locally constant).
  jvp: (_xs, [ta], _p, [p, l, u]) => {
    if (ta === null || ta === undefined) return [null, null, null]
    const x = sandwich(l as Value, u as Value, matmul(mT(p as Value), ta))
    return [null, matmul(l as Value, tril(x, -1)), matmul(triu(x), u as Value)]
  },
  // The adjoint: dA = P L^{-T} (tril(L^T dL, -1) + triu(dU U^T)) U^{-T}.
  vjp: ([, gl, gu], _xs, _p, [p, l, u]) => {
    let xbar: MaybeZero = null
    if (gl !== null && gl !== undefined) xbar = tril(matmul(mT(l as Value), gl), -1)
    if (gu !== null && gu !== undefined) {
      const t = triu(matmul(gu, mT(u as Value)))
      xbar = xbar === null ? t : add(xbar, t)
    }
    if (xbar === null) return [null]
    return [matmul(p as Value, adjointSandwich(l as Value, u as Value, xbar))]
  },
})

// L^{-1} Y U^{-1}, with L unit lower and U upper.
function sandwich(l: Value, u: Value, y: Value): Value {
  const y1 = triangularSolve(l, y, { lower: true, transpose: false, unitDiagonal: true })
  return mT(triangularSolve(u, mT(y1), { lower: false, transpose: true, unitDiagonal: false }))
}

// L^{-T} Y U^{-T}.
function adjointSandwich(l: Value, u: Value, y: Value): Value {
  const y1 = triangularSolve(l, y, { lower: true, transpose: true, unitDiagonal: true })
  return mT(triangularSolve(u, mT(y1), { lower: false, transpose: false, unitDiagonal: false }))
}

/** Result of {@link lu}. */
export interface LuResult {
  /** Permutation matrix. */
  readonly p: Value
  /** Unit lower-triangular factor. */
  readonly l: Value
  /** Upper-triangular factor. */
  readonly u: Value
}

/**
 * LU factorisation with partial pivoting, A = P L U (scipy.linalg.lu), for square matrices.
 *
 * @param a - Square matrix `[..., n, n]`.
 * @returns `{ p, l, u }`; L and U are differentiable (P is piecewise constant).
 * @concept linear-algebra
 * @example
 * ```ts
 * import { lu } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(lu([[1, 2], [3, 4]]).p) // => [[0, 1], [1, 0]]
 * ```
 */
export function lu(a: TensorLike): LuResult {
  const [av] = promoteFloat(a) as [Value]
  const [p, l, u] = bind(luPrimitive, [av], null) as [Value, Value, Value]
  return { p, l, u }
}
