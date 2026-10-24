// Sparse Gaussian elimination with partial pivoting for the almost-block-diagonal systems of collocation (scipy's
// solve_bvp uses splu). Rows are kept as maps, so the fill stays within the blocks and the boundary-condition rows.
// TODO(numerics/sparse): a general sparse direct solver belongs in numerics/sparse; this one serves solveBvp.

/** A sparse factorisation: the row interchanges and eliminations, and the upper-triangular rows. */
export interface SparseLu {
  readonly n: number
  readonly pivotRow: Int32Array
  readonly ops: readonly { readonly target: number; readonly source: number; readonly factor: number }[]
  readonly upper: readonly Map<number, number>[]
  readonly singular: boolean
}

/** Factorise an n x n matrix given as triplets (duplicates are summed). */
export function sparseLu(
  n: number,
  rows: readonly number[],
  cols: readonly number[],
  values: readonly number[],
): SparseLu {
  const a: Map<number, number>[] = Array.from({ length: n }, () => new Map<number, number>())
  const colRows: Set<number>[] = Array.from({ length: n }, () => new Set<number>())
  for (let e = 0; e < values.length; e++) {
    const r = rows[e] ?? 0
    const c = cols[e] ?? 0
    const row = a[r] as Map<number, number>
    row.set(c, (row.get(c) ?? 0) + (values[e] ?? 0))
    colRows[c]?.add(r)
  }
  const active = new Uint8Array(n).fill(1)
  const pivotRow = new Int32Array(n)
  const ops: { target: number; source: number; factor: number }[] = []
  const upper: Map<number, number>[] = new Array(n)
  let singular = false
  for (let k = 0; k < n; k++) {
    let best = -1
    let bestAbs = 0
    for (const r of colRows[k] ?? []) {
      if (active[r] === 0) continue
      const v = Math.abs(a[r]?.get(k) ?? 0)
      if (v > bestAbs) {
        bestAbs = v
        best = r
      }
    }
    if (best < 0 || bestAbs === 0) {
      singular = true
      upper[k] = new Map([[k, 0]])
      continue
    }
    active[best] = 0
    pivotRow[k] = best
    const prow = a[best] as Map<number, number>
    const pv = prow.get(k) ?? 1
    upper[k] = prow
    for (const r of [...(colRows[k] ?? [])]) {
      if (active[r] === 0 || r === best) continue
      const row = a[r] as Map<number, number>
      const v = row.get(k) ?? 0
      if (v === 0) continue
      const factor = v / pv
      ops.push({ target: r, source: best, factor })
      row.delete(k)
      for (const [c, w] of prow) {
        if (c <= k) continue
        const nv = (row.get(c) ?? 0) - factor * w
        row.set(c, nv)
        colRows[c]?.add(r)
      }
    }
  }
  return { n, pivotRow, ops, upper, singular }
}

/** Solve A x = b with a factorisation from {@link sparseLu}. */
export function sparseLuSolve(lu: SparseLu, b: ArrayLike<number>): Float64Array {
  const rhs = Float64Array.from(b)
  for (const op of lu.ops) rhs[op.target] = (rhs[op.target] ?? 0) - op.factor * (rhs[op.source] ?? 0)
  const x = new Float64Array(lu.n)
  for (let k = lu.n - 1; k >= 0; k--) {
    const row = lu.upper[k] as Map<number, number>
    let s = rhs[lu.pivotRow[k] ?? 0] ?? 0
    for (const [c, w] of row) if (c > k) s -= w * (x[c] ?? 0)
    x[k] = s / (row.get(k) ?? 1)
  }
  return x
}
