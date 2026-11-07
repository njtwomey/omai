/**
 * The exposition record for values drawn as cells rather than smooth fields: convolution feature maps and filters,
 * images, a recurrent network's hidden states over time, attention or weight matrices. Grids can be recorded at steps
 * of training so a view can play them.
 */

import { MdmlError } from '../base/index.ts'
import { rowsOf, sameLength, seal } from './_check.ts'

/** One grid of values drawn as cells: values[i][j] in row i (row 0 at the top, as an image) and column j. */
export interface ValueGrid {
  readonly name: string
  readonly values: readonly (readonly number[])[]
}

/** The grids at one recorded step of training (an epoch, a minibatch step), in the record's grid order. */
export interface ValueGridStep {
  /** The iteration (step, epoch) the snapshot was taken after; 0 is the initial model. */
  readonly iteration: number
  /** grids[g]: grid g's values at this step. */
  readonly grids: readonly (readonly (readonly number[])[])[]
  /** The training objective at this step. */
  readonly objective?: number
}

/**
 * Grids of values shown as cells, not smooth fields: a convolution's feature maps and filters, an image, a recurrent
 * network's hidden states over time (time down, units across), an attention or weight matrix. Optionally the grids at
 * recorded steps of training, so a view can play them forming.
 */
export interface ValueGrids {
  readonly kind: 'valueGrids'
  /** The grids, as fitted (the last step's values, when there are steps). */
  readonly grids: readonly ValueGrid[]
  /**
   * How values map to colours: one range shared by every grid (default; magnitudes compare across them), or each grid
   * on its own range (weak and strong channels both show).
   */
  readonly scale?: 'shared' | 'each'
  /** Centre the colours on zero on a diverging ramp (weights, filters, tanh states); default sequential. */
  readonly diverging?: boolean
  /** The names of the rows and the columns of a grid (`time`, `unit`), when they mean something. */
  readonly axes?: readonly [string, string]
  /** What a value is ('activation', 'weight', 'pixel'). */
  readonly valueName?: string
  /** Grids across (default about the square root of their count). */
  readonly columns?: number
  readonly steps?: readonly ValueGridStep[]
  /** What a step is (default 'step'). */
  readonly stepName?: string
}

/**
 * Value grids, with their shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when a grid is ragged, or a step does not give every grid at its size.
 * @example
 * ```ts
 * import { valueGrids } from 'mdml-engine/exposition'
 * const r = valueGrids({ grids: [{ name: 'filter 1', values: [[1, 0], [0, -1]] }], diverging: true })
 * r.grids[0]?.values // => [[1, 0], [0, -1]]
 * ```
 */
export function valueGrids(record: Omit<ValueGrids, 'kind'>): ValueGrids {
  for (const g of record.grids) rowsOf('valueGrids', g.name, g.values, g.values[0]?.length ?? 0)
  for (const [t, s] of (record.steps ?? []).entries()) {
    sameLength('valueGrids', `steps[${t}].grids`, s.grids.length, record.grids.length)
    for (const [i, g] of s.grids.entries()) {
      const ref = record.grids[i]?.values ?? []
      sameLength('valueGrids', `steps[${t}].grids[${i}]`, g.length, ref.length)
      rowsOf('valueGrids', `steps[${t}].grids[${i}]`, g, ref[0]?.length ?? 0)
    }
  }
  if (record.columns !== undefined && !(Number.isInteger(record.columns) && record.columns >= 1))
    throw new MdmlError('value', 'valueGrids: columns must be a positive integer')
  return seal({ kind: 'valueGrids', ...record })
}
