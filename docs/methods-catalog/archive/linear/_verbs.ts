/**
 * What the linear regressors of this area share: the plain JSON parameters of a linear fit, and the verbs those
 * parameters give.
 */

import { linearPredictor, vectorTensor } from '../../lib/linear/index.ts'
import type { Tensor, TensorLike } from '../../lib/tensor/index.ts'

/** The fitted parameters of a linear regressor: plain JSON. */
export type LinearParams = {
  readonly coef: number[]
  readonly intercept: number
}

/** What a fitted linear regressor offers. */
export interface LinearRegressorVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
  readonly featureImportances: number[]
}

/**
 * The verbs of linear parameters.
 *
 * @param p - The coefficients and intercept.
 * @param who - The caller, for errors.
 * @returns Prediction and facts.
 */
export function linearRegressorVerbs(p: LinearParams, who: string): LinearRegressorVerbs {
  return {
    predict: (x) => vectorTensor(linearPredictor(x, p.coef, p.intercept, who)),
    nFeaturesIn: p.coef.length,
    featureImportances: p.coef.map(Math.abs),
  }
}
