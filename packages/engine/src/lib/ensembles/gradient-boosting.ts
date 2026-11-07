/**
 * Gradient boosting (Friedman 2001; scikit-learn's `GradientBoostingRegressor` and `GradientBoostingClassifier`) on the
 * iterative base: an additive model built stage by stage, each stage a regression tree fitted to the negative gradient
 * of the loss and added with shrinkage, one stage per step. Squared, absolute, Huber, quantile and Poisson losses for
 * regression; the logistic (multinomial for more than two classes) and exponential losses for classification. Leaf
 * values are re-estimated for the loss as Friedman does (a median for the absolute error, a Newton step for the
 * logistic loss) or, as a mode, by Newton's method on the gradient and hessian (XGBoost's second-order boosting, with
 * an L2 penalty). Stochastic boosting, early stopping on a held-out loss and staged predictions are included.
 */

import { type FittedTree, leafCount } from '../trees/tree.ts'
import {
  boostingEstimator,
  type ClassificationLoss,
  type GradientBoostingEstimator,
  type GradientBoostingHyper,
  type GradientBoostingParams,
  type RegressionLoss,
} from './_boosting.ts'

export type {
  BoostingState,
  ClassificationLoss,
  GradientBoostingEstimator,
  GradientBoostingHyper,
  GradientBoostingParams,
  GradientBoostingVerbs,
  RegressionLoss,
} from './_boosting.ts'

const boostingDefaults = (
  hyper: Partial<GradientBoostingHyper>,
  loss: RegressionLoss | ClassificationLoss,
): GradientBoostingHyper => ({
  nEstimators: hyper.nEstimators ?? 100,
  learningRate: hyper.learningRate ?? 0.1,
  maxDepth: hyper.maxDepth ?? 3,
  minSamplesLeaf: hyper.minSamplesLeaf ?? 1,
  minSamplesSplit: hyper.minSamplesSplit ?? 2,
  minImpurityDecrease: hyper.minImpurityDecrease ?? 0,
  subsample: hyper.subsample ?? 1,
  maxFeatures: hyper.maxFeatures ?? 'all',
  ...(hyper.maxLeafNodes === undefined ? {} : { maxLeafNodes: hyper.maxLeafNodes }),
  loss: hyper.loss ?? loss,
  alpha: hyper.alpha ?? 0.9,
  leaves: hyper.leaves ?? 'friedman',
  l2: hyper.l2 ?? 0,
  validationFraction: hyper.validationFraction ?? 0,
  ...(hyper.nIterNoChange === undefined ? {} : { nIterNoChange: hyper.nIterNoChange }),
  tol: hyper.tol ?? 1e-4,
})

/**
 * Gradient boosting for regression (sklearn.ensemble.GradientBoostingRegressor), one stage per step: each stage fits a
 * Friedman-MSE tree to the negative gradient of the loss, re-estimates its leaf values for the loss (or by Newton's
 * method, `leaves: 'newton'`) and adds them scaled by the learning rate; subsampling gives stochastic gradient boosting
 * with out-of-bag losses, and `validationFraction` with `nIterNoChange` stops early on a held-out loss, firing the event
 * `boost/stopped`. The session logs `boost/round`, `boost/trainLoss`, `boost/validLoss`, `eval/loss`, `boost/residuals`
 * (the gradient the tree fits, in full) and `boost/leafValues`, and answers the probe `model/stagedPrediction`.
 *
 * @param hyper - Stages (default 100), learning rate (0.1), tree depth (3), leaf limits, subsample, loss (squared
 *   error), alpha, leaf mode, L2 penalty and early stopping.
 * @returns The estimator.
 * @concept boosting
 * @example
 * ```ts
 * import { gradientBoostingRegressor } from 'mdml-engine/ensembles'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 40])
 * const y = x.map(([a = 0]) => (a < 0.5 ? 1 : 3))
 * const fitted = gradientBoostingRegressor({ nEstimators: 30, maxDepth: 1 }).fit({ x, y })
 * toFlat(fitted.predict([[0.2], [0.8]])).map(Math.round) // => [1, 3]
 * ```
 */
export function gradientBoostingRegressor(
  hyper: Partial<GradientBoostingHyper & { readonly loss: RegressionLoss }> = {},
): GradientBoostingEstimator {
  return boostingEstimator('gradient-boosting-regressor', false, boostingDefaults(hyper, 'squared_error'))
}

/**
 * Gradient boosting for classification (sklearn.ensemble.GradientBoostingClassifier): the logistic loss (one tree per
 * stage for two classes, Newton leaf values on the log odds; one per class otherwise, a softmax) or the exponential loss
 * of AdaBoost. Logging and probes as {@link gradientBoostingRegressor}.
 *
 * @param hyper - As {@link gradientBoostingRegressor} (loss `log_loss` or `exponential`).
 * @returns The estimator.
 * @concept boosting
 * @index estimator Gradient boosting over trees; regressor twin beside it.
 * @example
 * ```ts
 * import { gradientBoostingClassifier } from 'mdml-engine/ensembles'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 40])
 * const y = x.map(([a = 0]) => (a < 0.5 ? 0 : 1))
 * toFlat(gradientBoostingClassifier({ nEstimators: 20, maxDepth: 1 }).fit({ x, y }).predict([[0.1], [0.9]])) // => [0, 1]
 * ```
 */
export function gradientBoostingClassifier(
  hyper: Partial<GradientBoostingHyper & { readonly loss: ClassificationLoss }> = {},
): GradientBoostingEstimator {
  return boostingEstimator('gradient-boosting-classifier', true, boostingDefaults(hyper, 'log_loss'))
}

/**
 * The number of leaves of each stage's first tree, a quick view of how deep the stages grew.
 *
 * @param params - Fitted parameters.
 * @returns Leaves per stage.
 * @example
 * ```ts
 * import { stageLeaves } from 'mdml-engine/ensembles'
 * stageLeaves({ init: [0], stages: [], learningRate: 0.1, loss: 'squared_error', classes: [], nFeatures: 1 }) // => []
 * ```
 */
export function stageLeaves(params: GradientBoostingParams): number[] {
  return params.stages.map((s) => leafCount(s[0] as FittedTree))
}
