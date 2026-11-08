/**
 * The slot of the pool-adjacent-violators algorithm that varies as an option table: the direction of the monotone fit
 * (non-decreasing or non-increasing), each with the sign it applies to the data. The table is one `union` space for
 * controls, grids and tuners, and builds the sign from a configuration.
 */

import { dict, type Option, type OptionTable, optionTable } from '../space/index.ts'

/** What a direction of the fit means, for validity rules and selectors. */
export interface IsotonicDirectionProperties {
  /** 1 when the fit may only rise along the sequence, -1 when it may only fall. */
  readonly sign: 1 | -1
}

function direction(description: string, sign: 1 | -1): Option<1 | -1, IsotonicDirectionProperties> {
  return { description, properties: { sign }, space: dict({}), make: () => sign }
}

/**
 * The directions of an isotonic fit as one slot: `increasing` (the fitted values never fall along the sequence, the
 * default) and `decreasing` (they never rise). A table's `make` gives the sign applied to the data before the
 * increasing algorithm runs and undone on the fit.
 *
 * @concept isotonic-regression
 * @example
 * ```ts
 * import { isotonicDirections } from 'omai-engine/learn'
 * isotonicDirections.make({ kind: 'decreasing' }) // => -1
 * ```
 */
export const isotonicDirections: OptionTable<1 | -1, IsotonicDirectionProperties> = optionTable(
  {
    increasing: direction('the fit is non-decreasing along the sequence', 1),
    decreasing: direction('the fit is non-increasing along the sequence', -1),
  },
  {
    default: 'increasing',
    label: 'direction',
    description: 'Whether the monotone fit rises or falls along the sequence.',
  },
)
