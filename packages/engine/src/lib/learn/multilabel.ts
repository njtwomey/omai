/**
 * Multilabel classification over any binary base classifier: binary relevance (one classifier per label, independently),
 * classifier chains (each label's classifier also sees the earlier labels, so dependences between labels are learnt)
 * and the label powerset (every distinct combination of labels is one class of a multiclass problem). Targets are a
 * 0/1 matrix `[n, labels]`; `predict` gives a matrix of the same shape and `score` is the subset accuracy. Each is an
 * estimator on the iterative base that solves one problem per step.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { Log } from '../trace/index.ts'
import { matrix, type StagedState, stagedAlgorithm } from './_staged.ts'
import { type IterativeEstimator, iterative } from './bases/index.ts'
import type { Estimator, LearnData } from './estimator.ts'
import {
  type ChainMethod,
  independentOutputs,
  type OutputsHyper,
  type OutputsParams,
  type OutputsPlan,
  type OutputsVerbs,
  outputChain,
} from './multioutput.ts'
import { matrixView } from './validation.ts'

/**
 * Binary relevance (`sklearn.multiclass.OneVsRestClassifier` on a label matrix, `skmultilearn`'s BinaryRelevance): one
 * base classifier per label, fitted independently on that label's column. `predictProba` gives the marginal
 * probability of each label `[m, labels]`. A label that is constant in the training data is predicted as that
 * constant.
 *
 * @param estimator - The binary base classifier.
 * @returns The estimator.
 * @example
 * ```ts
 * import { binaryRelevance, kNeighboursClassifier } from 'mdml-engine/learn'
 * import { toArray } from 'mdml-engine/tensor'
 * const x = [[0], [1], [10], [11]]
 * const f = binaryRelevance(kNeighboursClassifier({ nNeighbors: 1 })).fit({ x, y: [[1, 0], [1, 0], [0, 1], [1, 1]] })
 * toArray(f.predict([[0.4], [10.6]])) // => [[1, 0], [1, 1]]
 * ```
 */
export function binaryRelevance(
  estimator: Estimator,
): IterativeEstimator<OutputsHyper, StagedState<OutputsPlan>, OutputsParams, OutputsVerbs> {
  return independentOutputs('binary-relevance', estimator, 'classification')
}

/**
 * Classifier chains (`sklearn.multioutput.ClassifierChain`, Read et al.): the labels are fitted in `order`, each
 * classifier seeing the features and the true earlier labels; at prediction time it sees the chain's own earlier
 * outputs (hard predictions by default, or probabilities or decision values with `chainMethod`). The order
 * matters, so it is a knob: a permutation, or `'random'` from the run's key.
 *
 * @param estimator - The binary base classifier.
 * @param options - `order` and `chainMethod`.
 * @returns The estimator.
 * @reference J. Read, B. Pfahringer, G. Holmes and E. Frank, Classifier chains for multi-label classification,
 * Machine Learning 85 (2011).
 * @example
 * ```ts
 * import { classifierChain, kNeighboursClassifier } from 'mdml-engine/learn'
 * import { toArray } from 'mdml-engine/tensor'
 * const x = [[0], [1], [10], [11]]
 * const f = classifierChain(kNeighboursClassifier({ nNeighbors: 1 }), { order: [1, 0] }).fit({ x, y: [[1, 0], [1, 0], [0, 1], [1, 1]] })
 * toArray(f.predict([[0.4], [10.6]])) // => [[1, 0], [1, 1]]
 * ```
 */
export function classifierChain(
  estimator: Estimator,
  options: { readonly order?: OutputsHyper['order']; readonly chainMethod?: ChainMethod } = {},
): IterativeEstimator<OutputsHyper, StagedState<OutputsPlan>, OutputsParams, OutputsVerbs> {
  return outputChain('classifier-chain', estimator, 'classification', options)
}

/** The hyperparameters of {@link labelPowerset}. */
export interface LabelPowersetHyper {
  readonly estimator: Estimator
}

/** The plan of the label powerset: the distinct label combinations, in lexicographic order. */
export type LabelPowersetPlan = { readonly combinations: number[][] }

/** The fitted parameters of the label powerset. */
export type LabelPowersetParams = {
  readonly nLabels: number
  readonly nFeatures: number
  readonly plan: LabelPowersetPlan
  readonly models: Json[]
}

/** What a fitted label powerset offers. */
export interface LabelPowersetVerbs {
  readonly nFeaturesIn: number
  /** The combination of labels most probable for each row, `[m, labels]`. */
  readonly predict: (x: TensorLike) => Tensor
  /** The marginal probability of each label, `[m, labels]`, from the probabilities of the combinations. */
  readonly predictProba: (x: TensorLike) => Tensor
  /** The probability of each combination of the plan, `[m, combinations]`. */
  readonly predictCombinations: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
}

function labelMatrix(data: LearnData): ReturnType<typeof matrixView> {
  if (data.y === undefined) throw new MdmlError('value', 'labelPowerset: needs a label matrix y [n, labels]')
  return matrixView(data.y, 'labelPowerset')
}

/**
 * The label powerset (Tsoumakas and Katakis): each distinct combination of labels in the training data is a class of
 * one multiclass problem for the base classifier, so label dependences are captured exactly, at the price of only ever
 * predicting combinations seen in training. `predictProba` marginalises the combinations' probabilities to labels.
 *
 * @param estimator - The multiclass base classifier (with `predictProba` for the marginals).
 * @returns The estimator.
 * @reference G. Tsoumakas and I. Katakis, Multi-label classification: an overview, International Journal of Data
 * Warehousing and Mining 3 (2007).
 * @example
 * ```ts
 * import { labelPowerset, kNeighboursClassifier } from 'mdml-engine/learn'
 * import { toArray } from 'mdml-engine/tensor'
 * const x = [[0], [1], [10], [11]]
 * const f = labelPowerset(kNeighboursClassifier({ nNeighbors: 1 })).fit({ x, y: [[1, 0], [1, 0], [0, 1], [1, 1]] })
 * toArray(f.predict([[0.4], [11.6]])) // => [[1, 0], [1, 1]]
 * ```
 */
export function labelPowerset(
  estimator: Estimator,
): IterativeEstimator<LabelPowersetHyper, StagedState<LabelPowersetPlan>, LabelPowersetParams, LabelPowersetVerbs> {
  const hyper: LabelPowersetHyper = { estimator }
  return iterative<LabelPowersetHyper, StagedState<LabelPowersetPlan>, LabelPowersetParams, LabelPowersetVerbs>(
    {
      name: 'label-powerset',
      kind: 'other',
      defaults: hyper,
      algorithm: (data, h) => {
        const y = labelMatrix(data)
        const rows = Array.from({ length: y.n }, (_, i) => Array.from(y.data.subarray(i * y.d, (i + 1) * y.d)))
        const combinations = [...new Map(rows.map((r) => [r.join(','), r])).values()].sort((a, b) => {
          for (let j = 0; j < a.length; j++) if (a[j] !== b[j]) return (a[j] ?? 0) - (b[j] ?? 0)
          return 0
        })
        const index = new Map(combinations.map((c, i) => [c.join(','), i]))
        return stagedAlgorithm<LabelPowersetPlan>(h.estimator, {
          name: 'label-powerset',
          plan: () => ({ combinations }),
          total: () => 1,
          stage: () => ({
            ...data,
            y: matrix(
              rows.map((r) => index.get(r.join(',')) ?? 0),
              [y.n],
            ),
          }),
        })
      },
      maxSteps: () => 1,
      params: (s, data) => ({
        nLabels: labelMatrix(data).d,
        nFeatures: matrixView(data.x, 'labelPowerset').d,
        plan: s.plan,
        models: s.models,
      }),
      verbs: (p, h) => {
        const fitted = h.estimator.revive(p.models[0] as Json)
        const combos = p.plan.combinations
        const L = p.nLabels
        const classes = fitted.classes ?? combos.map((_, i) => i)
        const probabilities = (x: TensorLike): { readonly n: number; readonly p: Float64Array } => {
          const pr = (fitted.predictProba as (x: TensorLike) => Tensor | undefined)?.(x)
          if (pr === undefined) throw new MdmlError('value', 'labelPowerset: the base estimator has no predictProba')
          const flat = toFlat(pr)
          const k = pr.shape[1] ?? classes.length
          const out = new Float64Array((flat.length / k) * combos.length)
          for (let r = 0; r < flat.length / k; r++)
            classes.forEach((cls, c) => {
              out[r * combos.length + cls] = flat[r * k + c] ?? 0
            })
          return { n: flat.length / k, p: out }
        }
        const predict = (x: TensorLike): Tensor => {
          const labels = toFlat((fitted.predict as (x: TensorLike) => Tensor)(x))
          const out = new Float64Array(labels.length * L)
          labels.forEach((cls, r) => {
            for (let j = 0; j < L; j++) out[r * L + j] = combos[cls]?.[j] ?? 0
          })
          return matrix(out, [labels.length, L])
        }
        return {
          nFeaturesIn: p.nFeatures,
          predict,
          predictProba: (x) => {
            const { n, p: pc } = probabilities(x)
            const out = new Float64Array(n * L)
            for (let r = 0; r < n; r++)
              combos.forEach((combo, c) => {
                for (let j = 0; j < L; j++)
                  out[r * L + j] = (out[r * L + j] ?? 0) + (pc[r * combos.length + c] ?? 0) * (combo[j] ?? 0)
              })
            return matrix(out, [n, L])
          },
          predictCombinations: (x) => {
            const { n, p: pc } = probabilities(x)
            return matrix(pc, [n, combos.length])
          },
          score: (data) => {
            const pred = toFlat(predict(data.x))
            const truth = labelMatrix(data)
            let hits = 0
            for (let i = 0; i < truth.n; i++) {
              let all = true
              for (let j = 0; j < L; j++) if (pred[i * L + j] !== truth.data[i * L + j]) all = false
              if (all) hits++
            }
            return hits / truth.n
          },
        }
      },
      logs: { 'solver/problems': Log() },
    },
    hyper,
  )
}
