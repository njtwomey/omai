/**
 * Sequential Bayesian linear regression as a particular of the iterative base: a polynomial of the features with a
 * Gaussian prior on its coefficients, absorbing one training row per step in a random order fixed by the key. The tape
 * holds the log evidence and the mean predictive standard deviation of every step, and the posterior predictive band
 * over the training rows is a probe of any step, so a page can watch the band tighten as data arrive.
 */

import { type Distribution, normal } from '../../lib/dists/index.ts'
import { type IterativeEstimator, iterative } from '../../lib/learn/index.ts'
import type { MinMaxScalerParams } from '../../lib/learn/preprocessing/index.ts'
import {
  type BayesianPosterior,
  bayesianLinearAlgorithm,
  designOf,
  polynomialBasis,
  polynomialBasisOf,
  posteriorPredictive,
  vectorTensor,
} from '../../lib/linear/index.ts'
import { permutation } from '../../lib/random/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import { type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Contract, defineContract, derived, Log } from '../../lib/trace/index.ts'

/** The hyperparameters of {@link bayesianRegression}. */
export interface BayesianRegressionHyper {
  /** The degree of the polynomial of the (scaled) features. */
  readonly degree: number
  /** The precision of the zero-mean Gaussian prior on the coefficients. */
  readonly alpha: number
  /** The precision (inverse variance) of the Gaussian noise on the targets. */
  readonly beta: number
}

/** The fitted parameters of {@link bayesianRegression}: the posterior over the coefficients, plain JSON. */
export type BayesianRegressionParams = {
  readonly degree: number
  readonly alpha: number
  readonly beta: number
  /** The min-max scaler of the features to [-1, 1], applied before the polynomial expansion. */
  readonly scaler: MinMaxScalerParams
  /** The posterior mean of the coefficients (bias first). */
  readonly mean: number[]
  /** The posterior covariance of the coefficients. */
  readonly covariance: number[][]
  /** Training rows absorbed. */
  readonly nObserved: number
  readonly logEvidence: number
}

/** What a fitted Bayesian regression offers. */
export interface BayesianRegressionVerbs {
  /** The posterior predictive mean `[m]`. */
  readonly predict: (x: TensorLike) => Tensor
  /** The posterior predictive: a normal per row, with the noise and the weights' uncertainty in its variance. */
  readonly predictDistribution: (x: TensorLike) => Distribution
  readonly nFeaturesIn: number
}

/**
 * The paths of Bayesian regression beyond the solver’s: the posterior predictive on the training rows, derived.
 *
 * @example
 * ```ts
 * import type { BayesianRegressionPath } from 'mdml-engine/methods/linear'
 * const path: BayesianRegressionPath = 'model/predictive'
 * path // => 'model/predictive'
 * ```
 */
export type BayesianRegressionPath = 'model/predictive'

/**
 * Bayesian polynomial regression with a Gaussian prior on the coefficients and known noise precision, one training
 * row per step. The fitted model is the posterior after all rows (the batch closed form); any step's posterior can be
 * probed, and `model/predictive` gives its mean and +/- 2 predictive standard deviations over the training rows.
 *
 * @example
 * ```ts
 * import { bayesianRegression } from 'mdml-engine/methods/linear'
 * const fitted = bayesianRegression.with({ degree: 1 }).fit({ x: [[0], [1], [2], [3]], y: [1, 3, 5, 7] })
 * fitted.params.nObserved // => 4
 * ```
 */
export const bayesianRegression: IterativeEstimator<
  BayesianRegressionHyper,
  BayesianPosterior,
  BayesianRegressionParams,
  BayesianRegressionVerbs,
  BayesianRegressionPath
> = iterative<
  BayesianRegressionHyper,
  BayesianPosterior,
  BayesianRegressionParams,
  BayesianRegressionVerbs,
  BayesianRegressionPath
>({
  name: 'linear/bayesian-regression',
  kind: 'regressor',
  defaults: { degree: 3, alpha: 1, beta: 25 },
  hyperSpace: (h) =>
    dict({
      degree: int(1, 6, { default: h.degree, label: 'polynomial degree' }),
      alpha: real(1e-2, 1e2, { log: true, default: h.alpha, label: 'prior precision α' }),
      beta: real(1, 1e3, { log: true, default: h.beta, label: 'noise precision β' }),
    }),
  algorithm: (data, h, key) => {
    const design = designOf(data, 'bayesianRegression')
    const x = polynomialBasis(h.degree, data.x).expand(data.x)
    const order = Array.from(toFlat(permutation(key, design.x.n)))
    return bayesianLinearAlgorithm({ x, y: design.y }, { alpha: h.alpha, beta: h.beta }, order)
  },
  // One row per step: the budget is the number of training rows.
  maxSteps: (_, data) => designOf(data, 'bayesianRegression').x.n,
  params: (s, data, h) => ({
    degree: h.degree,
    alpha: h.alpha,
    beta: h.beta,
    scaler: polynomialBasis(h.degree, data.x).scaler,
    mean: s.mean,
    covariance: s.covariance,
    nObserved: s.n,
    logEvidence: s.logEvidence,
  }),
  verbs: (p) => {
    const basis = polynomialBasisOf(p.degree, p.scaler)
    const predictive = (x: TensorLike): { readonly mean: number[]; readonly variance: number[] } =>
      posteriorPredictive(p, basis.expand(x), p.beta)
    return {
      predict: (x) => vectorTensor(predictive(x).mean),
      predictDistribution: (x) => {
        const { mean, variance } = predictive(x)
        return normal(vectorTensor(mean), vectorTensor(variance.map((v) => Math.sqrt(v))))
      },
      nFeaturesIn: p.scaler.scale.length,
    }
  },
  contract: (data, h): Contract<BayesianPosterior, BayesianRegressionPath> => {
    const x = polynomialBasis(h.degree, data.x).expand(data.x)
    return defineContract({
      'model/predictive': derived<BayesianPosterior>({
        description: 'the posterior predictive on the training rows: its mean and the mean ± 2 predictive deviations',
        compute: (s) => {
          const { mean, variance } = posteriorPredictive(s, x, h.beta)
          const width = variance.map((v) => 2 * Math.sqrt(v))
          return {
            mean,
            lower: mean.map((m, i) => m - (width[i] ?? 0)),
            upper: mean.map((m, i) => m + (width[i] ?? 0)),
          }
        },
      }),
    })
  },
  logs: { 'solver/logEvidence': Log(), 'solver/meanPredictiveStd': Log() },
})
