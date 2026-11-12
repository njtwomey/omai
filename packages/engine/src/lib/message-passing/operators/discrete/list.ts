/**
 * The operators of this folder as an array, for the registry (`registry.ts`), which orders them by `operatorNames`; the order here is
 * alphabetical. The discrete and conjugate-probability operators: the conjugate likelihoods of counts and categories, the table, the select and the Gaussian mixture. The gate is not in it: it is built over the whole registry.
 */

import type { FactorOperator } from '../../core/types.ts'
import {
  bernoulliBeta,
  binomial,
  discreteDirichlet,
  gammaObserved,
  mixtureWeight,
  negativeBinomialBeta,
  poisson,
} from './conjugate.ts'
import { gaussianMixture, select, table } from './discrete.ts'

/** The operators of this folder. */
export const discreteOperatorList: readonly FactorOperator[] = [
  bernoulliBeta,
  binomial,
  discreteDirichlet,
  gammaObserved,
  gaussianMixture,
  mixtureWeight,
  negativeBinomialBeta,
  poisson,
  select,
  table,
]
