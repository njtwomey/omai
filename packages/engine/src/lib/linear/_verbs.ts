/**
 * The verbs of a fitted regression that is a plain affine map (`x . coef + intercept`): prediction, the feature count,
 * the absolute coefficients as importances and the default R-squared score. The robust, expectile, censored and
 * generalised least-squares estimators of this folder share it.
 */

import { type LearnData, withDefaultScore } from '../learn/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { linearPredictor, vectorTensor } from './design.ts'

/** What a fitted affine regressor offers. */
export interface AffineVerbs {
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  readonly featureImportances: number[]
  readonly score: (data: LearnData) => number
}

/**
 * The verbs of an affine regression.
 *
 * @param coef - The slopes.
 * @param intercept - The intercept.
 * @param who - The caller, for errors.
 * @returns The verbs.
 */
export function affineVerbs(coef: readonly number[], intercept: number, who: string): AffineVerbs {
  const base = {
    nFeaturesIn: coef.length,
    predict: (x: TensorLike): Tensor => vectorTensor(linearPredictor(x, coef, intercept, who)),
    featureImportances: coef.map(Math.abs),
  }
  return { ...base, score: withDefaultScore('regressor', base).score ?? ((): number => Number.NaN) }
}
