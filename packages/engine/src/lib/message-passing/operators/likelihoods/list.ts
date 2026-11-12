/**
 * The operators of this folder as an array, for the registry (`registry.ts`), which orders them by `operatorNames`; the order here is
 * alphabetical. The numeric likelihood sites and their rows, the robust sites, the prior site, the unary functions and the two-variable likelihoods.
 */

import type { FactorOperator } from '../../core/types.ts'
import { apply, applyProductOperator } from './functions.ts'
import { likelihoodRowOperator, likelihoodSiteOperator, negativeBinomialDispersion } from './likelihoods.ts'
import { likelihoodPairOperator } from './pairs.ts'
import { priorSite } from './prior-sites.ts'
import { laplaceRow, laplaceSite, mixtureRow, mixtureSite, studentTRow, studentTSite } from './robust.ts'

/** The operators of this folder. */
export const likelihoodOperatorList: readonly FactorOperator[] = [
  apply,
  applyProductOperator,
  laplaceRow,
  laplaceSite,
  likelihoodPairOperator,
  likelihoodRowOperator,
  likelihoodSiteOperator,
  mixtureRow,
  mixtureSite,
  negativeBinomialDispersion,
  priorSite,
  studentTRow,
  studentTSite,
]
