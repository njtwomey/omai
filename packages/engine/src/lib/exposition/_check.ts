// Shape checks shared by the record constructors (private to the domain).
import { WhiteboxError } from '../base/index.ts'

/** Fail unless `n` equals `expected`, naming the record and the field. */
export function sameLength(kind: string, field: string, n: number, expected: number): void {
  if (n !== expected) throw new WhiteboxError('shape', `${kind}: ${field} has ${n} entries, expected ${expected}`)
}

/** Fail unless every row of a matrix has `width` entries. */
export function rowsOf(kind: string, field: string, rows: readonly (readonly number[])[], width: number): void {
  for (const [i, row] of rows.entries()) {
    if (row.length !== width)
      throw new WhiteboxError('shape', `${kind}: ${field}[${i}] has ${row.length} entries, expected ${width}`)
  }
}
