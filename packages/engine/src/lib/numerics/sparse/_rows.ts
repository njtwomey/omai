// Row-wise numerics shared by the stationary iterations and multigrid: a CSR matrix as plain arrays with its
// diagonal, the residual norm, and one Jacobi / Gauss-Seidel / SOR sweep.
import { OmaiError } from '../../base/index.ts'
import type { TensorLike } from '../../tensor/index.ts'
import { dnrm2 as norm } from '../linalg/blas.ts'
import { type CsrMatrix, csr, primalValues } from './csr.ts'

/** The splitting of a sweep. */
export type SweepMethod = 'jacobi' | 'gauss-seidel' | 'sor'

/** A square CSR matrix as plain arrays, with its diagonal. */
export interface Rows {
  readonly n: number
  readonly indptr: Int32Array
  readonly indices: Int32Array
  readonly values: Float64Array
  readonly diagonal: Float64Array
}

export function rowsOf(a: CsrMatrix | TensorLike): Rows {
  const m = csrOf(a)
  const [rows, cols] = m.shape
  if (rows !== cols) throw new OmaiError('shape', 'stationary iterations need a square matrix')
  const values = Float64Array.from(primalValues(m))
  const diagonal = new Float64Array(rows)
  for (let i = 0; i < rows; i++)
    for (let k = m.indptr[i] ?? 0; k < (m.indptr[i + 1] ?? 0); k++)
      if (m.indices[k] === i) diagonal[i] = (diagonal[i] ?? 0) + (values[k] ?? 0)
  const zero = diagonal.indexOf(0)
  if (zero >= 0) throw new OmaiError('value', `stationary iterations need a nonzero diagonal (row ${zero} has none)`)
  return { n: rows, indptr: m.indptr, indices: m.indices, values, diagonal }
}

/** Sum over the stored entries of row i, other than the diagonal, of a_ij x_j. */
export function offDiagonal(r: Rows, i: number, x: ArrayLike<number>): number {
  let s = 0
  for (let k = r.indptr[i] ?? 0; k < (r.indptr[i + 1] ?? 0); k++) {
    const j = r.indices[k] ?? 0
    if (j !== i) s += (r.values[k] ?? 0) * (x[j] ?? 0)
  }
  return s
}

export function residualNormOf(r: Rows, x: ArrayLike<number>, b: ArrayLike<number>): number {
  const res = new Float64Array(r.n)
  for (let i = 0; i < r.n; i++) res[i] = (b[i] ?? 0) - offDiagonal(r, i, x) - (r.diagonal[i] ?? 0) * (x[i] ?? 0)
  return norm(res)
}

/** One sweep of the method from `x`, returning the new iterate. */
export function sweep(
  r: Rows,
  b: ArrayLike<number>,
  x: ArrayLike<number>,
  method: SweepMethod,
  omega: number,
): number[] {
  const out = Array.from(x)
  if (method === 'jacobi') {
    for (let i = 0; i < r.n; i++) {
      const d = r.diagonal[i] ?? 1
      out[i] = (1 - omega) * (x[i] ?? 0) + (omega * ((b[i] ?? 0) - offDiagonal(r, i, x))) / d
    }
    return out
  }
  const w = method === 'gauss-seidel' ? 1 : omega
  for (let i = 0; i < r.n; i++) {
    const d = r.diagonal[i] ?? 1
    out[i] = (1 - w) * (out[i] ?? 0) + (w * ((b[i] ?? 0) - offDiagonal(r, i, out))) / d
  }
  return out
}

/** A CSR matrix from a CSR matrix or a dense one. */
export function csrOf(a: CsrMatrix | TensorLike): CsrMatrix {
  return typeof a === 'object' && a !== null && 'indptr' in a && 'rows' in a ? (a as CsrMatrix) : csr(a as TensorLike)
}
