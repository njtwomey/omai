/**
 * Logistic regression as a particular of the iterative base: iteratively reweighted least squares of the linear
 * library for the binomial family, with the objective, the coefficients and the step halvings of every iteration on the
 * tape. Binary; with more classes, the last class is fitted against the rest.
 */

import { OmaiError } from '../../lib/base/index.ts'
import { type IterativeEstimator, iterative, type LearnData, targetVector } from '../../lib/learn/index.ts'
import {
  designOf,
  familyOf,
  type IrlsState,
  irlsAlgorithm,
  linearPredictor,
  matrixTensor,
  vectorTensor,
} from '../../lib/linear/index.ts'
import { bool, dict, real } from '../../lib/space/index.ts'
import type { Tensor, TensorLike } from '../../lib/tensor/index.ts'
import { Log } from '../../lib/trace/index.ts'

/** The hyperparameters of {@link logisticRegression}. */
export interface LogisticHyper {
  /** The L2 penalty on the per-sample deviance (scikit-learn's 1 / (C n)). */
  readonly alpha: number
  readonly fitIntercept: boolean
  readonly tol: number
  readonly maxIter: number
}

/** The fitted parameters of {@link logisticRegression}: plain JSON. */
export type LogisticParams = {
  readonly coef: number[]
  readonly intercept: number
  /** The two class labels: the negative, then the positive. */
  readonly classes: number[]
}

/** What a fitted logistic regression offers. */
export interface LogisticVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  readonly featureImportances: number[]
}

const classesOf = (data: LearnData): number[] => {
  const labels = [...new Set(targetVector(data.y ?? [], 'logisticRegression'))].sort((a, b) => a - b)
  if (labels.length < 2) throw new OmaiError('value', 'logisticRegression: needs at least two classes')
  return [labels.length === 2 ? (labels[0] as number) : -1, labels[labels.length - 1] as number]
}

const binary = (data: LearnData, classes: readonly number[]): LearnData => ({
  ...data,
  y: targetVector(data.y ?? [], 'logisticRegression').map((v) => (v === classes[1] ? 1 : 0)),
})

/**
 * Logistic regression by IRLS.
 *
 * @example
 * ```ts
 * import { logisticRegression } from 'omai-engine/methods/linear'
 * logisticRegression.fit({ x: [[0], [1], [2], [3]], y: [0, 0, 1, 1] }).classes // => [0, 1]
 * ```
 */
export const logisticRegression: IterativeEstimator<LogisticHyper, IrlsState, LogisticParams, LogisticVerbs> =
  iterative<LogisticHyper, IrlsState, LogisticParams, LogisticVerbs>({
    name: 'linear/logistic-regression',
    kind: 'classifier',
    defaults: { alpha: 0.01, fitIntercept: true, tol: 1e-8, maxIter: 100 },
    hyperSpace: (h) =>
      dict({
        alpha: real(1e-4, 10, { log: true, default: h.alpha, label: 'penalty strength' }),
        fitIntercept: bool({ default: h.fitIntercept }),
      }),
    algorithm: (data, h) => {
      const design = designOf(binary(data, classesOf(data)), 'logisticRegression')
      const { family, link } = familyOf({ family: 'binomial', power: 1.5, nbAlpha: 1 })
      return irlsAlgorithm(
        { design, family, link, alpha: h.alpha, l1Ratio: 0, fitIntercept: h.fitIntercept },
        { tol: h.tol },
      )
    },
    maxSteps: (h) => h.maxIter,
    params: (s, data) => ({ coef: Array.from(s.coef), intercept: s.intercept, classes: classesOf(data) }),
    verbs: (p) => {
      const eta = (x: TensorLike): Float64Array => linearPredictor(x, p.coef, p.intercept, 'logisticRegression')
      const positive = (x: TensorLike): Float64Array => eta(x).map((v) => 1 / (1 + Math.exp(-v)))
      return {
        predict: (x) =>
          vectorTensor(positive(x).map((q) => (q > 0.5 ? (p.classes[1] as number) : (p.classes[0] as number)))),
        predictProba: (x) => {
          const q = positive(x)
          const out = new Float64Array(q.length * 2)
          q.forEach((v, i) => {
            out[2 * i] = 1 - v
            out[2 * i + 1] = v
          })
          return matrixTensor(out, q.length, 2)
        },
        decisionFunction: (x) => vectorTensor(eta(x)),
        classes: p.classes,
        nFeaturesIn: p.coef.length,
        featureImportances: p.coef.map(Math.abs),
      }
    },
    logs: { 'solver/objective': Log(), 'solver/coef': Log({ detail: 'full' }) },
  })
