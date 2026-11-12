/**
 * The operators of this folder as an array, for the registry (`registry.ts`), which orders them by `operatorNames`; the order here is
 * alphabetical. The link operators: the probit and truncation links, the logistic, the spin and the multiclass site.
 */

import type { FactorOperator } from '../../core/types.ts'
import { logistic } from './links.ts'
import { multiclassSite } from './multiclass.ts'
import { spin } from './spin.ts'
import { greaterThan, isBetween, probit } from './truncation.ts'

/** The operators of this folder. */
export const linkOperatorList: readonly FactorOperator[] = [
  greaterThan,
  isBetween,
  logistic,
  multiclassSite,
  probit,
  spin,
]
