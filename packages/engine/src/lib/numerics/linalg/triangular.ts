/**
 * Triangular solves by substitution as a differentiable primitive (scipy.linalg `solve_triangular`, upper or lower,
 * transposed, unit diagonal), and inverses of triangular matrices. The Cholesky, LU and QR solves are built on it.
 */

import {
  arg,
  batchLeading,
  bind1,
  cast,
  definePrimitive,
  eye,
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
import { type TriangularOptions, triangularKernel } from './_kernels.ts'
import { asColumns, assemble, broadcastBatch, matrices, matrixShape, mT, squareSize } from './_matrix.ts'

/** The part of A a triangular solve reads: its triangle, without the diagonal when the diagonal is taken as ones. */
function usedTriangle(x: Value, o: TriangularOptions): Value {
  const k = o.unitDiagonal ? (o.lower ? -1 : 1) : 0
  return o.lower ? tril(x, k) : triu(x, k)
}

const triangularPrimitive: Primitive<TriangularOptions> = definePrimitive<TriangularOptions>({
  name: 'solveTriangular',
  abstractEval: (xs) => [arg(xs, 1)],
  impl: ([a, b], o) => {
    const t = a as NonNullable<typeof a>
    const rhs = b as NonNullable<typeof b>
    const n = squareSize(t.shape, 'solveTriangular')
    const { batch, n: k } = matrixShape(rhs.shape, 'solveTriangular')
    const bs = matrices(rhs)
    const parts = matrices(t).map((m, i) => triangularKernel(m, bs[i] as Float64Array, n, k, o))
    return [assemble(rhs, rhs.dtype as FloatDType, batch, [n, k], parts)]
  },
  // X = op(T)^{-1} B: dX = op(T)^{-1} (dB - op(dT) X).
  jvp: ([a, ,], [ta, tb], o, [x]) => {
    const out = x as Value
    let rhs: MaybeZero = tb ?? null
    if (ta !== null && ta !== undefined) {
      const dt = usedTriangle(ta, o)
      const term = matmul(o.transpose ? mT(dt) : dt, out)
      rhs = rhs === null ? neg(term) : sub(rhs, term)
    }
    return [rhs === null ? null : bind1(triangularPrimitive, [a as Value, rhs], o)]
  },
  // dB = op(T)^{-T} dX; dA = the used triangle of -(dB X^T) (or of its transpose when solving with A^T).
  vjp: ([g], [a], o, [x]) => {
    if (g === null || g === undefined) return [null, null]
    const gb = bind1(triangularPrimitive, [a as Value, g], { ...o, transpose: !o.transpose })
    const outer = neg(matmul(gb, mT(x as Value)))
    return [usedTriangle(o.transpose ? mT(outer) : outer, o), gb]
  },
  // Factorisations and solves already map over leading batch axes.
  batch: (xs, ds, p, n) => batchLeading(triangularPrimitive, xs, ds, p, n),
  describe: (o) =>
    `${o.lower ? 'lower' : 'upper'}${o.transpose ? ', transposed' : ''}${o.unitDiagonal ? ', unit' : ''}`,
})

/** Options of {@link solveTriangular} (scipy's `solve_triangular` keywords). */
export interface SolveTriangularOptions {
  /** Use the lower triangle of `a` (default false: the upper triangle, as scipy). */
  readonly lower?: boolean
  /** Solve `a^T x = b` instead of `a x = b`. */
  readonly transpose?: boolean
  /** Take the diagonal of `a` to be ones (it is not read). */
  readonly unitDiagonal?: boolean
}

/** Apply the triangular-solve primitive (no batch broadcasting or vector handling). */
export function triangularSolve(a: Value, b: Value, options: TriangularOptions): Value {
  return bind1(triangularPrimitive, [a, b], options)
}

/**
 * Solve a triangular system `a x = b` by substitution (scipy.linalg.solve_triangular). Only the chosen triangle of `a`
 * is read. Batches broadcast; a 1-D right-hand side is a vector.
 *
 * @param a - Triangular matrix `[..., n, n]`.
 * @param b - Right-hand side `[..., n, k]` or `[n]`.
 * @param options - Triangle, transpose and unit diagonal.
 * @returns The solution, with the shape of `b`; differentiable in both arguments.
 * @concept linear-algebra
 * @example
 * ```ts
 * import { solveTriangular } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(solveTriangular([[2, 0], [1, 1]], [2, 3], { lower: true })) // => [1, 2]
 * ```
 */
export function solveTriangular(a: TensorLike, b: TensorLike, options: SolveTriangularOptions = {}): Value {
  const [av, bv] = promoteFloat(a, b) as [Value, Value]
  const cols = asColumns(av, bv)
  const [ab, bb] = broadcastBatch(av, cols.b)
  const x = triangularSolve(ab, bb, {
    lower: options.lower ?? false,
    transpose: options.transpose ?? false,
    unitDiagonal: options.unitDiagonal ?? false,
  })
  return cols.vector ? squeeze(x, -1) : x
}

/**
 * Inverse of a triangular matrix (scipy.linalg.solve_triangular against the identity).
 *
 * @param a - Triangular matrix `[..., n, n]`.
 * @param options - Triangle and unit diagonal.
 * @returns The inverse (triangular of the same kind), differentiable.
 * @example
 * ```ts
 * import { invTriangular } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(invTriangular([[2, 0], [0, 4]], { lower: true })) // => [[0.5, 0], [0, 0.25]]
 * ```
 */
export function invTriangular(a: TensorLike, options: Omit<SolveTriangularOptions, 'transpose'> = {}): Value {
  const [av] = promoteFloat(a) as [Value]
  const n = squareSize(av.shape, 'invTriangular')
  return solveTriangular(av, cast(eye(n), av.dtype), options)
}
