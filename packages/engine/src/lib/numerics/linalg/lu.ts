/**
 * LU factorisation with partial pivoting and what is built on it: `solve` and `inv` (numpy.linalg), `lu` with explicit
 * permutation, and the packed `luFactor` and `luSolve` pair (scipy.linalg) for solving many right-hand sides with one
 * factorisation.
 */

import {
  add,
  arg,
  batchLeading,
  bind,
  bind1,
  concreteOrThrow,
  definePrimitive,
  type FloatDType,
  fromStorage,
  type MaybeZero,
  matmul,
  neg,
  type Primitive,
  promoteFloat,
  squeeze,
  sub,
  type Tensor,
  type TensorLike,
  tril,
  triu,
  type Value,
} from '../../tensor/index.ts'
import { luKernel, solveKernel, triangularKernel } from './_kernels.ts'
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
  // Factorisations and solves already map over leading batch axes.
  batch: (xs, ds, p, n) => batchLeading(solvePrimitive, xs, ds, p, n),
})

/**
 * Solve a square linear system A x = b by LU with partial pivoting (numpy.linalg.solve). Batches broadcast; a
 * 1-D right-hand side is a vector.
 *
 * @param a - Square matrix `[..., n, n]`.
 * @param b - Right-hand side `[..., n, k]` or `[n]`.
 * @returns The solution, differentiable in both arguments.
 * @throws MdmlError with code `value` when A is exactly singular.
 * @concept linear-algebra
 * @index function Solve a square linear system by LU, like numpy.linalg.solve.
 * @example
 * ```ts
 * import { solve } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
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
 * @index function Matrix inverse, batched and differentiable.
 * @example
 * ```ts
 * import { inv } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
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
  // Factorisations and solves already map over leading batch axes.
  batch: (xs, ds, p, n) => batchLeading(luPrimitive, xs, ds, p, n),
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
 * import { lu } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(lu([[1, 2], [3, 4]]).p) // => [[0, 1], [1, 0]]
 * ```
 */
export function lu(a: TensorLike): LuResult {
  const [av] = promoteFloat(a) as [Value]
  const [p, l, u] = bind(luPrimitive, [av], null) as [Value, Value, Value]
  return { p, l, u }
}

/** A packed LU factorisation with LAPACK pivot indices (scipy.linalg.lu_factor's output). */
export interface LuFactorisation {
  /** L (unit, below the diagonal) and U (on and above) in one matrix. */
  readonly lu: Tensor
  /** Row j was interchanged with row piv[j] at step j. */
  readonly piv: Tensor<'int32'>
}

/**
 * The packed LU factorisation with partial pivoting (scipy.linalg.lu_factor), to solve many right-hand sides with
 * {@link luSolve}.
 *
 * @param a - Square matrix `[n, n]`.
 * @returns The packed factors and pivot indices.
 * @example
 * ```ts
 * import { luFactor } from 'mdml-engine/numerics/linalg'
 * Array.from(luFactor([[1, 2], [3, 4]]).piv.data) // => [1, 1]
 * ```
 */
export function luFactor(a: TensorLike): LuFactorisation {
  const t = concreteOrThrow(promoteFloat(a)[0] as Tensor, 'luFactor')
  const n = squareSize(t.shape, 'luFactor')
  const { lu: packed, piv } = luKernel(Float64Array.from(t.data), n)
  return { lu: fromStorage(packed, [n, n], 'float64'), piv: fromStorage(piv, [n], 'int32') }
}

/**
 * Solve A x = b from a packed LU factorisation (scipy.linalg.lu_solve).
 *
 * @param factorisation - From {@link luFactor}.
 * @param b - Right-hand side `[n]` or `[n, k]`.
 * @param options - `transpose` to solve A^T x = b.
 * @returns The solution.
 * @example
 * ```ts
 * import { luFactor, luSolve } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(luSolve(luFactor([[3, 1], [1, 2]]), [9, 8])) // => [2, 3]
 * ```
 */
export function luSolve(
  factorisation: LuFactorisation,
  b: TensorLike,
  options: { readonly transpose?: boolean } = {},
): Tensor {
  const packed = factorisation.lu
  const n = packed.shape[0] ?? 0
  const rhs = concreteOrThrow(promoteFloat(b)[0] as Tensor, 'luSolve')
  const k = rhs.shape.length === 1 ? 1 : (rhs.shape[1] ?? 1)
  const x = Float64Array.from(rhs.data)
  const swap = (j: number, p: number): void => {
    for (let c = 0; c < k; c++) {
      const tmp = x[j * k + c] ?? 0
      x[j * k + c] = x[p * k + c] ?? 0
      x[p * k + c] = tmp
    }
  }
  const piv = factorisation.piv.data
  const lu = Float64Array.from(packed.data)
  let out: Float64Array
  if (options.transpose === true) {
    // A^T = U^T L^T P: solve U^T y = b, L^T z = y, then undo the interchanges in reverse.
    const y = triangularKernel(lu, x, n, k, { lower: false, transpose: true, unitDiagonal: false })
    out = triangularKernel(lu, y, n, k, { lower: true, transpose: true, unitDiagonal: true })
    for (let j = n - 1; j >= 0; j--) {
      const p = piv[j] ?? j
      if (p !== j) {
        for (let c = 0; c < k; c++) {
          const tmp = out[j * k + c] ?? 0
          out[j * k + c] = out[p * k + c] ?? 0
          out[p * k + c] = tmp
        }
      }
    }
  } else {
    for (let j = 0; j < n; j++) if ((piv[j] ?? j) !== j) swap(j, piv[j] ?? j)
    const y = triangularKernel(lu, x, n, k, { lower: true, transpose: false, unitDiagonal: true })
    out = triangularKernel(lu, y, n, k, { lower: false, transpose: false, unitDiagonal: false })
  }
  return fromStorage(out, rhs.shape, 'float64')
}
