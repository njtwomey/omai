/**
 * The operators of this folder as an array, for the registry (`registry.ts`), which orders them by `operatorNames`; the order here is
 * alphabetical. The Gaussian operators: the prior and the Gaussian factors, the linear-Gaussian family, the extremum, the random log scale and the transition.
 */

import type { FactorOperator } from '../../core/types.ts'
import { gaussianChain } from './chain.ts'
import { maximum, minimum } from './extremum.ts'
import {
  equal,
  gaussianGammaPrecision,
  gaussianLogScale,
  gaussianNormalGamma,
  gaussianOperator,
  prior,
  product,
} from './gaussian.ts'
import {
  difference,
  dotConstant,
  gaussianJoint,
  gaussianMvStack,
  linear,
  linearRegressionRow,
  matmulConstant,
  plus,
  sum,
  timesConstant,
} from './linear.ts'
import { gaussianScale } from './log-scale.ts'
import { gaussianTransition } from './transition.ts'

/** The operators of this folder. */
export const gaussianOperatorList: readonly FactorOperator[] = [
  difference,
  dotConstant,
  equal,
  gaussianChain,
  gaussianGammaPrecision,
  gaussianJoint,
  gaussianLogScale,
  gaussianMvStack,
  gaussianNormalGamma,
  gaussianOperator,
  gaussianScale,
  gaussianTransition,
  linear,
  linearRegressionRow,
  matmulConstant,
  maximum,
  minimum,
  plus,
  prior,
  product,
  sum,
  timesConstant,
]
