// Linear algebra over GF(2) on 0/1 arrays, shared by the block codes and LDPC codes. Not exported.
import { WhiteboxError } from '../base/index.ts'

/** A binary matrix as rows of 0/1. */
export type BitMatrix = readonly (readonly number[])[]

/** Check a binary matrix and return a mutable copy (entries reduced mod 2). */
export function bitMatrix(m: BitMatrix, name: string): number[][] {
  const cols = m[0]?.length ?? 0
  if (m.length === 0 || cols === 0 || m.some((r) => r.length !== cols))
    throw new WhiteboxError('shape', `${name}: expected a non-empty rectangular 0/1 matrix`)
  return m.map((r) =>
    r.map((v) => {
      if (v !== 0 && v !== 1) throw new WhiteboxError('value', `${name}: entries must be 0 or 1`)
      return v
    }),
  )
}

/** Reduced row echelon form over GF(2): the reduced rows (zero rows dropped) and the pivot column of each. */
export function rref(m: BitMatrix): { readonly rows: number[][]; readonly pivots: number[] } {
  const a = m.map((r) => [...r])
  const cols = a[0]?.length ?? 0
  const pivots: number[] = []
  let r = 0
  for (let c = 0; c < cols && r < a.length; c++) {
    let p = r
    while (p < a.length && a[p]?.[c] !== 1) p++
    if (p === a.length) continue
    ;[a[r], a[p]] = [a[p] as number[], a[r] as number[]]
    const pivotRow = a[r] as number[]
    for (let i = 0; i < a.length; i++) {
      const row = a[i] as number[]
      if (i !== r && row[c] === 1) for (let j = c; j < cols; j++) row[j] = (row[j] ?? 0) ^ (pivotRow[j] ?? 0)
    }
    pivots.push(c)
    r++
  }
  return { rows: a.slice(0, r), pivots }
}

/** A basis of the null space { x : M x = 0 } as rows. */
export function nullSpace(m: BitMatrix): number[][] {
  const cols = m[0]?.length ?? 0
  const { rows, pivots } = rref(m)
  const free = Array.from({ length: cols }, (_, c) => c).filter((c) => !pivots.includes(c))
  return free.map((f) => {
    const x = new Array<number>(cols).fill(0)
    x[f] = 1
    pivots.forEach((p, i) => {
      x[p] = rows[i]?.[f] ?? 0
    })
    return x
  })
}

/** M x over GF(2). */
export function matVec(m: BitMatrix, x: readonly number[]): number[] {
  return m.map((row) => row.reduce((s, v, j) => s ^ (v & (x[j] ?? 0)), 0))
}

/** x M over GF(2) (a row vector times a matrix). */
export function vecMat(x: readonly number[], m: BitMatrix): number[] {
  const cols = m[0]?.length ?? 0
  const out = new Array<number>(cols).fill(0)
  m.forEach((row, i) => {
    if ((x[i] ?? 0) & 1) for (let j = 0; j < cols; j++) out[j] = (out[j] ?? 0) ^ (row[j] ?? 0)
  })
  return out
}

/** Solve x M = c for x (M has full row rank); throws when c is not in the row space. */
export function solveRow(m: BitMatrix, c: readonly number[], name: string): number[] {
  const k = m.length
  const n = m[0]?.length ?? 0
  // Eliminate on [M^T | c^T] so that the pivots pick an information set.
  const augmented = Array.from({ length: n }, (_, j) => [
    ...Array.from({ length: k }, (_, i) => m[i]?.[j] ?? 0),
    c[j] ?? 0,
  ])
  const { rows, pivots } = rref(augmented)
  if (pivots.includes(k)) throw new WhiteboxError('value', `${name}: the word is not a codeword`)
  const x = new Array<number>(k).fill(0)
  pivots.forEach((p, i) => {
    x[p] = rows[i]?.[k] ?? 0
  })
  return x
}
