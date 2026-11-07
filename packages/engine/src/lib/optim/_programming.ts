// Shared helpers of the mathematical-programming files (linear, quadratic, conic, integer, SMO): converting inputs to
// row-major number arrays and dense solves on them through the public linear-algebra routines. Not exported from the
// barrel.
import { MdmlError } from '../base/index.ts'
import { choleskyJitter, choSolve, dgemv, luFactor, luSolve } from '../numerics/linalg/index.ts'
import { asTensor, fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'

/** A dense row-major matrix. */
export interface Dense {
  readonly data: Float64Array
  readonly rows: number
  readonly cols: number
}

/** A vector from anything tensor-like (empty when undefined). */
export function vectorFrom(x: TensorLike | undefined): Float64Array {
  if (x === undefined) return new Float64Array(0)
  return Float64Array.from(toFlat(primalOf(asTensor(x))))
}

/** A matrix with `cols` columns from anything tensor-like (0 rows when undefined); a 1-D input is one row. */
export function matrixFrom(x: TensorLike | undefined, cols: number, name: string): Dense {
  if (x === undefined) return { data: new Float64Array(0), rows: 0, cols }
  const t = primalOf(asTensor(x))
  const data = Float64Array.from(toFlat(t))
  if (data.length === 0) return { data, rows: 0, cols }
  const c = t.shape.length === 1 ? (t.shape[0] ?? 0) : (t.shape[t.shape.length - 1] ?? 0)
  if (c !== cols || t.shape.length > 2)
    throw new MdmlError('shape', `${name} must have ${cols} columns, got shape [${t.shape.join(', ')}]`)
  return { data, rows: data.length / cols, cols }
}

/** Check that a vector has the expected length. */
export function expectLength(x: Float64Array, n: number, name: string): void {
  if (x.length !== n) throw new MdmlError('shape', `${name} must have length ${n}, got ${x.length}`)
}

/** Solve A X = B for a square n x n A and B with k columns (row-major), by LU with partial pivoting. */
export function solveDense(a: Float64Array, b: Float64Array, n: number, k = 1): Float64Array {
  if (n === 0) return new Float64Array(0)
  const f = luFactor(fromStorage(Float64Array.from(a), [n, n], 'float64'))
  const rhs =
    k === 1 ? fromStorage(Float64Array.from(b), [n], 'float64') : fromStorage(Float64Array.from(b), [n, k], 'float64')
  return singularAsNaN(n * k, () => Float64Array.from(luSolve(f, rhs).data))
}

/** The result of a dense solve, or NaN throughout when the matrix is singular (the solvers read non-finite as singular). */
function singularAsNaN(length: number, solve: () => Float64Array): Float64Array {
  try {
    return solve()
  } catch (error) {
    if (error instanceof MdmlError && error.code === 'value') return new Float64Array(length).fill(Number.NaN)
    throw error
  }
}

/** Solve A^T x = b for a square n x n A by LU. */
export function solveDenseTransposed(a: Float64Array, b: Float64Array, n: number): Float64Array {
  if (n === 0) return new Float64Array(0)
  const f = luFactor(fromStorage(Float64Array.from(a), [n, n], 'float64'))
  return singularAsNaN(n, () =>
    Float64Array.from(luSolve(f, fromStorage(Float64Array.from(b), [n], 'float64'), { transpose: true }).data),
  )
}

/** Solve S x = b for a symmetric positive (semi)definite S, by Cholesky with a jitter ladder for near-singular S. */
export function solveSpd(s: Float64Array, b: Float64Array, n: number): Float64Array {
  if (n === 0) return new Float64Array(0)
  const { l } = choleskyJitter(fromStorage(Float64Array.from(s), [n, n], 'float64'), 1e-2)
  return Float64Array.from(toFlat(primalOf(choSolve(l, fromStorage(Float64Array.from(b), [n], 'float64')))))
}

/** y = A x (or A^T x) for a dense matrix, by BLAS dgemv. */
export function matVec(a: Dense, x: ArrayLike<number>, transpose = false): Float64Array {
  return dgemv(a.data, a.rows, a.cols, x, transpose)
}

/** A 1-D float64 tensor from numbers. */
export function vectorTensor(x: ArrayLike<number>): Tensor<'float64'> {
  return fromStorage(Float64Array.from(x), [x.length], 'float64')
}

/** The quadratic 1/2 x^T P x + q^T x for an n x n row-major P. */
export function quadraticValue(P: Float64Array, q: Float64Array, x: ArrayLike<number>): number {
  const n = q.length
  let s = 0
  for (let i = 0; i < n; i++) {
    let row = 0
    for (let j = 0; j < n; j++) row += (P[i * n + j] ?? 0) * (x[j] ?? 0)
    s += (x[i] ?? 0) * (0.5 * row + (q[i] ?? 0))
  }
  return s
}
