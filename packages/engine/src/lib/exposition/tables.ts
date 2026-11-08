/**
 * The exposition record of a small table (`markedTable`): rows and columns of cells with marks on them (a status colour per
 * cell or row, a highlighted decision), for a profile, a test and its decision rule, a balance table, a set of metrics
 * with their direction, or data with its missing and outlying cells flagged.
 */

import { OmaiError } from '../base/index.ts'
import { sameLength, seal } from './_check.ts'

/**
 * What a mark says about a cell or a row: `ok`, `warn` and `bad` colour it as such, `info` and `muted` quieter,
 * `decision` outlines the cell or row the rule chose.
 */
export type CellStatus = 'ok' | 'warn' | 'bad' | 'info' | 'muted' | 'decision'

/** A mark on a cell (`row` and `col`), or on a whole row (no `col`), or on a whole column (no `row`). */
export interface CellMark {
  readonly row?: number
  readonly col?: number
  readonly status: CellStatus
  /** A hint shown on hover. */
  readonly note?: string
}

/** A table: column names, the cells by row (text, numbers or missing), optional row names and cell marks. */
export interface MarkedTable {
  readonly kind: 'markedTable'
  readonly columns: readonly string[]
  /** cells[r][c]: the cell of row r and column c. */
  readonly cells: readonly (readonly (string | number | null)[])[]
  /** A name for each row, shown in a leading column. */
  readonly rowNames?: readonly string[]
  readonly marks?: readonly CellMark[]
  /** Significant digits of the numbers shown (default 3). */
  readonly digits?: number
  /** What the table is, shown as its caption. */
  readonly caption?: string
}

/**
 * A table, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `shape` when a row or the row names do not match the columns; `value` when a mark names a row or
 *   a column the table lacks, or names neither.
 * @example
 * ```ts
 * import { markedTable } from 'omai-engine/exposition'
 * const r = markedTable({
 *   columns: ['test', 'p'],
 *   cells: [['t', 0.03], ['z', 0.4]],
 *   marks: [{ row: 0, status: 'decision', note: 'p < 0.05' }],
 * })
 * r.marks?.[0]?.status // => 'decision'
 * ```
 */
export function markedTable(record: Omit<MarkedTable, 'kind'>): MarkedTable {
  for (const [r, row] of record.cells.entries())
    sameLength('markedTable', `cells[${r}]`, row.length, record.columns.length)
  if (record.rowNames !== undefined) sameLength('markedTable', 'rowNames', record.rowNames.length, record.cells.length)
  for (const [k, m] of (record.marks ?? []).entries()) {
    if (m.row === undefined && m.col === undefined)
      throw new OmaiError('value', `markedTable: marks[${k}] names neither a row nor a column`)
    if (m.row !== undefined && !(Number.isInteger(m.row) && m.row >= 0 && m.row < record.cells.length))
      throw new OmaiError('value', `markedTable: marks[${k}] names the row ${m.row}, which the table lacks`)
    if (m.col !== undefined && !(Number.isInteger(m.col) && m.col >= 0 && m.col < record.columns.length))
      throw new OmaiError('value', `markedTable: marks[${k}] names the column ${m.col}, which the table lacks`)
  }
  return seal({ kind: 'markedTable', ...record })
}
