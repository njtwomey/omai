/**
 * The operators of this folder as an array, for the registry (`registry.ts`), which orders them by `operatorNames`; the order here is
 * alphabetical. The covariance-matrix operators: the Wishart precision, its normal-Wishart conjugate form and the standard-deviation-and-correlation site.
 */

import type { FactorOperator } from '../../core/types.ts'
import { mvnSdCorr } from './correlation.ts'
import { mvnNormalWishart, mvnWishart } from './covariance.ts'
import { mvnWishartGroup } from './hierarchy.ts'

/** The operators of this folder. */
export const covarianceOperatorList: readonly FactorOperator[] = [
  mvnNormalWishart,
  mvnSdCorr,
  mvnWishart,
  mvnWishartGroup,
]
