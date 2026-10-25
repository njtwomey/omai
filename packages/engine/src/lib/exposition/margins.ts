/**
 * The exposition record of a margin method (support vector machines, SVDD): the support vectors with their dual
 * coefficients, how the dual was solved, and the decision surface with its margins.
 */

import { rowsOf, sameLength } from './_check.ts'
import type { InputRows } from './rows.ts'
import type { DecisionSurface } from './surfaces.ts'

/** A margin classifier or regressor's support and how its dual was solved (SVMs, SVDD). */
export interface MarginRecord {
  readonly kind: 'margin'
  /** Indices of the support vectors among the training rows. */
  readonly support: readonly number[]
  /** The dual coefficient of each support vector. */
  readonly dualCoefficients: readonly number[]
  readonly intercept: number
  /** The functional margin (or distance to the boundary) of each training row. */
  readonly margins: readonly number[]
  /** The dual objective after each solver step (SMO or coordinate descent). */
  readonly dualObjective?: readonly number[]
  /** The working set chosen at each step: the pair of indices SMO updated. */
  readonly workingSets?: readonly (readonly number[])[]
  /** The solver step of each `dualObjective` entry (when the trace was thinned; default 0, 1, 2, ...). */
  readonly steps?: readonly number[]
  /** The training rows (two coordinates each) that `support`, `margins` and `workingSets` index, when the data is small. */
  readonly rows?: InputRows
  /** The decision surface over the rows' two features, with its steps through training (two features only). */
  readonly surface?: DecisionSurface
}

/**
 * A margin record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when the support and its coefficients differ in length.
 * @example
 * ```ts
 * import { marginRecord } from 'whitebox-engine/exposition'
 * marginRecord({ support: [0, 3], dualCoefficients: [0.5, -0.5], intercept: 0.1, margins: [1, 2, 3, 1] }).support // => [0, 3]
 * ```
 */
export function marginRecord(record: Omit<MarginRecord, 'kind'>): MarginRecord {
  sameLength('margin', 'dualCoefficients', record.dualCoefficients.length, record.support.length)
  if (record.workingSets !== undefined && record.dualObjective !== undefined)
    sameLength('margin', 'workingSets', record.workingSets.length, record.dualObjective.length)
  if (record.workingSets !== undefined) rowsOf('margin', 'workingSets', record.workingSets, 2)
  if (record.steps !== undefined && record.dualObjective !== undefined)
    sameLength('margin', 'steps', record.steps.length, record.dualObjective.length)
  return { kind: 'margin', ...record }
}
