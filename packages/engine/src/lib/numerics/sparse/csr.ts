/**
 * Sparse matrices in compressed sparse row form (as `scipy.sparse.csr_array`): construction from arrays, dense matrices
 * or coordinates, conversion to dense, matrix-vector and matrix-matrix products, transposes, diagonals and sums, all
 * differentiable in the stored values, plus the `LinearOperator` interface that lets solvers take CSR, dense or
 * implicit matrices alike.
 */

import { MdmlError } from '../../base/index.ts'
import {
  concat,
  concreteOrThrow,
  fromStorage,
  matmul,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  segmentSum,
  swapaxes,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
} from '../../tensor/index.ts'

/**
 * A sparse matrix in compressed sparse row form (scipy.sparse.csr_array): row i holds the entries
 * values[indptr[i] .. indptr[i + 1]] in columns indices[...]. The structure is fixed; the values are a Value, so products
 * are differentiable in them.
 */
export interface CsrMatrix {
  readonly shape: readonly [number, number]
  readonly indptr: Int32Array
  readonly indices: Int32Array
  readonly values: Value
  /** The row of each stored entry (expanded from indptr). */
  readonly rows: Int32Array
}

function rowsOf(indptr: Int32Array): Int32Array {
  const n = indptr.length - 1
  const rows = new Int32Array(indptr[n] ?? 0)
  for (let i = 0; i < n; i++) rows.fill(i, indptr[i] ?? 0, indptr[i + 1] ?? 0)
  return rows
}

/**
 * A CSR matrix from its arrays, or from a dense matrix (keeping the non-zero entries).
 *
 * @param source - A dense matrix, or `{ indptr, indices, values, shape }`.
 * @returns The matrix.
 * @concept sparse-matrices
 * @index structure Compressed sparse row matrix from a dense matrix or its arrays.
 * @example
 * ```ts
 * import { csr } from 'mdml-engine/numerics/sparse'
 * Array.from(csr([[1, 0], [0, 2]]).indices) // => [0, 1]
 * ```
 */
export function csr(
  source:
    | TensorLike
    | {
        readonly indptr: ArrayLike<number>
        readonly indices: ArrayLike<number>
        readonly values: TensorLike
        readonly shape: readonly [number, number]
      },
): CsrMatrix {
  if (typeof source === 'object' && source !== null && 'indptr' in source) {
    const indptr = Int32Array.from(source.indptr)
    const indices = Int32Array.from(source.indices)
    const [values] = promoteFloat(source.values) as [Value]
    if (
      indptr.length !== source.shape[0] + 1 ||
      (indptr[indptr.length - 1] ?? 0) !== indices.length ||
      values.shape[0] !== indices.length
    ) {
      throw new MdmlError('shape', 'csr: inconsistent indptr, indices and values')
    }
    return { shape: source.shape, indptr, indices, values, rows: rowsOf(indptr) }
  }
  const dense = concreteOrThrow(promoteFloat(source as TensorLike)[0] as Tensor, 'csr')
  if (dense.shape.length !== 2) throw new MdmlError('shape', 'csr: a dense source must be a matrix')
  const [m = 0, n = 0] = dense.shape
  const indptr = new Int32Array(m + 1)
  const indices: number[] = []
  const values: number[] = []
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const v = dense.data[i * n + j] ?? 0
      if (v !== 0) {
        indices.push(j)
        values.push(v)
      }
    }
    indptr[i + 1] = indices.length
  }
  return csr({
    indptr,
    indices,
    values: fromStorage(Float64Array.from(values), [values.length], 'float64'),
    shape: [m, n],
  })
}

/**
 * A CSR matrix from coordinates (scipy's coo_array(...).tocsr()): entries sorted by row then column, duplicates summed
 * (the summation is a segment sum, so the values stay differentiable).
 *
 * @param rows - Row of each entry.
 * @param cols - Column of each entry.
 * @param values - Values.
 * @param shape - Matrix shape.
 * @returns The matrix.
 * @example
 * ```ts
 * import { fromCoo, toDense } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(toDense(fromCoo([0, 1, 0], [1, 0, 1], [1, 2, 3], [2, 2]))) // => [[0, 4], [2, 0]]
 * ```
 */
export function fromCoo(
  rows: ArrayLike<number>,
  cols: ArrayLike<number>,
  values: TensorLike,
  shape: readonly [number, number],
): CsrMatrix {
  const [v] = promoteFloat(values) as [Value]
  const order = Array.from({ length: rows.length }, (_, i) => i).sort(
    (a, b) => (rows[a] ?? 0) - (rows[b] ?? 0) || (cols[a] ?? 0) - (cols[b] ?? 0),
  )
  const segment = new Int32Array(order.length)
  const keys: [number, number][] = []
  order.forEach((k, i) => {
    const r = rows[k] ?? 0
    const c = cols[k] ?? 0
    if (r < 0 || r >= shape[0] || c < 0 || c >= shape[1])
      throw new MdmlError('value', 'fromCoo: an index is out of range')
    const prev = keys[keys.length - 1]
    if (prev === undefined || prev[0] !== r || prev[1] !== c) keys.push([r, c])
    segment[i] = keys.length - 1
  })
  const sorted = take(v, fromStorage(Int32Array.from(order), [order.length], 'int32'), 0)
  const summed = segmentSum(sorted, fromStorage(segment, [segment.length], 'int32'), keys.length)
  const indptr = new Int32Array(shape[0] + 1)
  for (const [r] of keys) indptr[r + 1] = (indptr[r + 1] ?? 0) + 1
  for (let i = 0; i < shape[0]; i++) indptr[i + 1] = (indptr[i + 1] ?? 0) + (indptr[i] ?? 0)
  return { shape, indptr, indices: Int32Array.from(keys, (k) => k[1]), values: summed, rows: rowsOf(indptr) }
}

/**
 * The dense matrix of a CSR matrix (differentiable in the values).
 *
 * @param a - The matrix.
 * @returns `[m, n]`.
 * @example
 * ```ts
 * import { csr, toDense } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(toDense(csr([[0, 3], [4, 0]]))) // => [[0, 3], [4, 0]]
 * ```
 */
export function toDense(a: CsrMatrix): Value {
  const [m, n] = a.shape
  const flat = Int32Array.from(a.rows, (r, k) => r * n + (a.indices[k] ?? 0))
  const dense = segmentSum(a.values, fromStorage(flat, [flat.length], 'int32'), m * n)
  return reshape(dense, [m, n])
}

/**
 * The sparse matrix-vector product y = A x (differentiable in A's values and in x): gather x at the stored columns,
 * multiply, and sum each row.
 *
 * @param a - The matrix `[m, n]`.
 * @param x - Vector `[n]`.
 * @returns `[m]`.
 * @example
 * ```ts
 * import { csr, spmv } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(spmv(csr([[1, 0, 2], [0, 3, 0]]), [1, 1, 1])) // => [3, 3]
 * ```
 */
export function spmv(a: CsrMatrix, x: TensorLike): Value {
  const [xv] = promoteFloat(x) as [Value]
  if (xv.shape.length !== 1 || xv.shape[0] !== a.shape[1])
    throw new MdmlError('shape', 'spmv: x must have one entry per column')
  const gathered = take(xv, fromStorage(a.indices, [a.indices.length], 'int32'), 0)
  return segmentSum(mul(a.values, gathered), fromStorage(a.rows, [a.rows.length], 'int32'), a.shape[0])
}

/**
 * The sparse-dense product A B (differentiable in A's values and in B).
 *
 * @param a - The matrix `[m, n]`.
 * @param b - Dense `[n, k]`.
 * @returns `[m, k]`.
 * @example
 * ```ts
 * import { csr, spmm } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(spmm(csr([[1, 0], [0, 2]]), [[1, 2], [3, 4]])) // => [[1, 2], [6, 8]]
 * ```
 */
export function spmm(a: CsrMatrix, b: TensorLike): Value {
  const [bv] = promoteFloat(b) as [Value]
  if (bv.shape.length !== 2 || bv.shape[0] !== a.shape[1]) throw new MdmlError('shape', 'spmm: b must be [n, k]')
  const gathered = take(bv, fromStorage(a.indices, [a.indices.length], 'int32'), 0)
  return segmentSum(
    mul(reshape(a.values, [-1, 1]), gathered),
    fromStorage(a.rows, [a.rows.length], 'int32'),
    a.shape[0],
  )
}

/**
 * The transpose, as a CSR matrix (its values a permutation of A's, so still differentiable).
 *
 * @param a - The matrix.
 * @returns A^T.
 * @example
 * ```ts
 * import { csr, toDense, transposeCsr } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(toDense(transposeCsr(csr([[1, 2], [0, 3]])))) // => [[1, 0], [2, 3]]
 * ```
 */
export function transposeCsr(a: CsrMatrix): CsrMatrix {
  const [m, n] = a.shape
  return fromCoo(a.indices, a.rows, a.values, [n, m])
}

/**
 * A diagonal (or offset-diagonal) CSR matrix (scipy.sparse.diags with one diagonal).
 *
 * @param diagonal - The entries.
 * @param offset - 0 for the main diagonal, k above, -k below (default 0).
 * @param size - Matrix size (default length + |offset|).
 * @returns The matrix.
 * @example
 * ```ts
 * import { diagCsr, toDense } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(toDense(diagCsr([1, 2]))) // => [[1, 0], [0, 2]]
 * ```
 */
export function diagCsr(diagonal: TensorLike, offset = 0, size?: number): CsrMatrix {
  const [d] = promoteFloat(diagonal) as [Value]
  const len = d.shape[0] ?? 0
  const n = size ?? len + Math.abs(offset)
  const rows = Array.from({ length: len }, (_, i) => (offset >= 0 ? i : i - offset))
  const cols = Array.from({ length: len }, (_, i) => (offset >= 0 ? i + offset : i))
  return fromCoo(rows, cols, d, [n, n])
}

/**
 * The sum of CSR matrices of the same shape (a CSR matrix of the union pattern).
 *
 * @param a - First matrix.
 * @param b - Second matrix.
 * @param beta - Scale of b (default 1): a + beta b.
 * @returns The sum.
 * @example
 * ```ts
 * import { csrAdd, diagCsr, toDense } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(toDense(csrAdd(diagCsr([1, 1]), diagCsr([1], 1, 2), 2))) // => [[1, 2], [0, 1]]
 * ```
 */
export function csrAdd(a: CsrMatrix, b: CsrMatrix, beta = 1): CsrMatrix {
  if (a.shape[0] !== b.shape[0] || a.shape[1] !== b.shape[1]) throw new MdmlError('shape', 'csrAdd: shapes differ')
  const rows = [...a.rows, ...b.rows]
  const cols = [...a.indices, ...b.indices]
  return fromCoo(rows, cols, concat([a.values, mul(b.values, beta)], 0), a.shape)
}

/** The diagonal of a CSR matrix (zeros where nothing is stored). */
export function csrDiagonal(a: CsrMatrix): Value {
  const n = Math.min(a.shape[0], a.shape[1])
  const picks: number[] = []
  const slots: number[] = []
  a.rows.forEach((r, k) => {
    if (r === a.indices[k] && r < n) {
      picks.push(k)
      slots.push(r)
    }
  })
  const values = take(a.values, fromStorage(Int32Array.from(picks), [picks.length], 'int32'), 0)
  return segmentSum(values, fromStorage(Int32Array.from(slots), [slots.length], 'int32'), n)
}

/** A linear operator: anything with a shape and a matrix-vector product (CSR matrices, dense matrices, implicit maps). */
export interface LinearOperator {
  readonly shape: readonly [number, number]
  readonly matvec: (x: Value) => Value
  /** The transpose product A^T x, when available. */
  readonly rmatvec?: (x: Value) => Value
}

/**
 * A linear operator from a CSR matrix, a dense matrix or an operator.
 *
 * @param a - The matrix or operator.
 * @returns The operator.
 * @example
 * ```ts
 * import { asOperator, csr } from 'mdml-engine/numerics/sparse'
 * asOperator(csr([[2, 0], [0, 3]])).shape // => [2, 2]
 * ```
 */
export function asOperator(a: CsrMatrix | TensorLike | LinearOperator): LinearOperator {
  if (typeof a === 'object' && a !== null && 'matvec' in a) return a as LinearOperator
  if (typeof a === 'object' && a !== null && 'indptr' in a) {
    const m = a as CsrMatrix
    let t: CsrMatrix | undefined
    const transposed = (): CsrMatrix => {
      if (t === undefined) t = transposeCsr(m)
      return t
    }
    return { shape: m.shape, matvec: (x) => spmv(m, x), rmatvec: (x) => spmv(transposed(), x) }
  }
  const [d] = promoteFloat(a as TensorLike) as [Value]
  if (d.shape.length !== 2) throw new MdmlError('shape', 'asOperator: a dense operator must be a matrix')
  const shape: [number, number] = [d.shape[0] ?? 0, d.shape[1] ?? 0]
  return {
    shape,
    matvec: (x) => reshape(matmul(d, x), [shape[0]]),
    rmatvec: (x) => reshape(matmul(swapaxes(d, 0, 1), x), [shape[1]]),
  }
}

/** The number of stored entries. */
export function nnz(a: CsrMatrix): number {
  return a.indices.length
}

/** The stored values as numbers (primal), for kernels that do not need derivatives. */
export function primalValues(a: CsrMatrix): number[] {
  return toFlat(primalOf(a.values))
}
