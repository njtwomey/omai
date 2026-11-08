/**
 * The slot of GPTQ that varies as an option table: the order in which the columns of a layer are quantised, either as
 * they come (`natural`) or by decreasing diagonal of the Hessian, so the columns the calibration inputs excite most are
 * quantised first while there is most room to compensate (`activation`, GPTQ's act-order). The table is one `union`
 * space for controls, grids and tuners, and builds the ordering from a configuration.
 */

import { dict, type Option, type OptionTable, optionTable } from '../space/index.ts'

/** The quantisation order of the columns, from the diagonal of the Hessian. */
export type ColumnOrder = (hessianDiagonal: readonly number[]) => number[]

/** What an order does, for validity rules and selectors. */
export interface ColumnOrderProperties {
  /** Whether the order depends on the calibration inputs. */
  readonly dataDependent: boolean
}

/**
 * The column orders of GPTQ as one slot: `natural` (left to right) and `activation` (decreasing Hessian diagonal, ties in
 * column order).
 *
 * @concept quantisation
 * @example
 * ```ts
 * import { gptqOrders } from 'omai-engine/nn'
 * gptqOrders.make({ kind: 'activation' })([1, 5, 3]) // => [1, 2, 0]
 * ```
 */
export const gptqOrders: OptionTable<ColumnOrder, ColumnOrderProperties> = optionTable(
  {
    natural: {
      description: 'quantise the columns left to right',
      properties: { dataDependent: false },
      space: dict({}),
      make: (): ColumnOrder => (h) => h.map((_, j) => j),
    } satisfies Option<ColumnOrder, ColumnOrderProperties>,
    activation: {
      description: 'quantise the columns with the largest Hessian diagonal first (act-order)',
      properties: { dataDependent: true },
      space: dict({}),
      make: (): ColumnOrder => (h) => h.map((_, j) => j).sort((a, b) => (h[b] ?? 0) - (h[a] ?? 0) || a - b),
    } satisfies Option<ColumnOrder, ColumnOrderProperties>,
  },
  { default: 'natural', label: 'column order', description: 'The order the columns of a layer are quantised in.' },
)
