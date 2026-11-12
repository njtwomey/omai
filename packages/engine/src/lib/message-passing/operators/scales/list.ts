/**
 * The operators of this folder as an array, for the registry (`registry.ts`), which orders them by `operatorNames`; the order here is
 * alphabetical. The scale and concentration operators: the shared smoothing parameters, the gamma scale hierarchies, the random concentration and the stick-breaking assignment.
 */

import type { FactorOperator } from '../../core/types.ts'
import { betaTwoShapes, dirichletConcentration } from './concentration.ts'
import { gaussianMvFunctionScale } from './function-scale.ts'
import { gammaProduct, gammaRate } from './gamma-scale.ts'
import { gaussianMvPrecisionScale } from './scale.ts'
import { stickBreaking } from './stick-breaking.ts'

/** The operators of this folder. */
export const scaleOperatorList: readonly FactorOperator[] = [
  betaTwoShapes,
  dirichletConcentration,
  gammaProduct,
  gammaRate,
  gaussianMvFunctionScale,
  gaussianMvPrecisionScale,
  stickBreaking,
]
