// Linear-algebra glue for the PDE schemes: operators as triplets, (alpha I + beta A) factorised once.
import { type CsrMatrix, spmv } from '../../numerics/sparse/index.ts'
import { fromStorage, primalOf, type Tensor } from '../../tensor/index.ts'
import { type SparseLu, sparseLu, sparseLuSolve } from '../ode/_sparse-lu.ts'

/** y = A x on numbers (spmv on concrete values). */
export function csrApply(a: CsrMatrix, x: ArrayLike<number>): Float64Array {
  return Float64Array.from(primalOf(spmv(a, fromStorage(Float64Array.from(x), [x.length], 'float64'))).data)
}

/** Factorise alpha I + beta A, with identity rows at the given (Dirichlet) nodes. */
export function shiftedLu(a: CsrMatrix, alpha: number, beta: number, identityRows: ReadonlySet<number>): SparseLu {
  const n = a.shape[0]
  const vals = (a.values as Tensor).data
  const rows: number[] = []
  const cols: number[] = []
  const v: number[] = []
  for (let r = 0; r < n; r++) {
    rows.push(r)
    cols.push(r)
    v.push(identityRows.has(r) ? 1 : alpha)
    if (identityRows.has(r)) continue
    for (let e = a.indptr[r] ?? 0; e < (a.indptr[r + 1] ?? 0); e++) {
      rows.push(r)
      cols.push(a.indices[e] ?? 0)
      v.push(beta * (vals[e] ?? 0))
    }
  }
  return sparseLu(n, rows, cols, v)
}

export { sparseLuSolve }

/** Every entry of a CSR row except the diagonal, and the diagonal. */
export function rowOf(a: CsrMatrix, r: number): { diag: number; off: { c: number; v: number }[] } {
  const vals = (a.values as Tensor).data
  let diag = 0
  const off: { c: number; v: number }[] = []
  for (let e = a.indptr[r] ?? 0; e < (a.indptr[r + 1] ?? 0); e++) {
    const c = a.indices[e] ?? 0
    const v = vals[e] ?? 0
    if (c === r) diag += v
    else off.push({ c, v })
  }
  return { diag, off }
}
