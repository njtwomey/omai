/**
 * The operators of this folder as an array, for the registry (`registry.ts`), which orders them by `operatorNames`; the order here is
 * alphabetical. The lifts of a scalar link to a regression row and to a plate of rows.
 */

import type { FactorOperator } from '../../core/types.ts'
import { linearRegressionPlate, logisticPlate, probitPlate } from './plates.ts'
import { logisticRow, probitRow, spinRow } from './rows.ts'

/** The operators of this folder. */
export const liftOperatorList: readonly FactorOperator[] = [
  linearRegressionPlate,
  logisticPlate,
  logisticRow,
  probitPlate,
  probitRow,
  spinRow,
]
