import { WhiteboxError } from '../base/index.ts'
import { rowsOf, sameLength } from './_check.ts'

/**
 * The input rows behind a record, as a view draws them: two coordinates per row (the two features a surface spans,
 * the first two features, or a 2-D projection) with their names, and the rows' classes where known. Methods include
 * them only for small data (a few thousand rows), so records stay small; views draw them unless given rows of their
 * own.
 */
export interface InputRows {
  /** points[i]: row i's two drawn coordinates. */
  readonly points: readonly (readonly number[])[]
  /** The names of the two coordinates. */
  readonly features: readonly [string, string]
  /** labels[i]: row i's class index (-1 when unlabelled), for classifiers and labelled data. */
  readonly labels?: readonly number[]
  /** Class names, by class index. */
  readonly classes?: readonly string[]
}

/**
 * Input rows, with their shapes checked.
 *
 * @param rows - The rows.
 * @returns The rows.
 * @throws WhiteboxError `shape` when a point does not have two coordinates or the labels do not match the points.
 * @example
 * ```ts
 * import { inputRows } from 'whitebox-engine/exposition'
 * inputRows({ points: [[0, 1], [2, 3]], features: ['a', 'b'], labels: [0, 1] }).labels // => [0, 1]
 * ```
 */
export function inputRows(rows: InputRows): InputRows {
  rowsOf('inputRows', 'points', rows.points, 2)
  if (rows.labels !== undefined) sameLength('inputRows', 'labels', rows.labels.length, rows.points.length)
  if (rows.labels?.some((l) => !Number.isInteger(l) || l < -1))
    throw new WhiteboxError('value', 'inputRows: labels must be class indices (or -1 for unlabelled)')
  return rows
}
