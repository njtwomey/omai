// What the classifiers and regressors written on the iterative base share (not exported from the domain): the verbs of
// a classifier built from per-class scores, the default score, and the algorithm that absorbs the training rows one per
// step, which closed-form and memory-based learners use to give a session a meaning (the model on the first t rows).
import { argmaxOf, logSumExpOf } from '../base/index.ts'
import { accuracyScore, r2Score } from '../metrics/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import type { LearnData } from './estimator.ts'
import { type MatrixView, matrixView, rowsOf, takeData } from './validation.ts'

/** The verbs every classifier built from class scores offers. */
export interface ScoreClassifierVerbs {
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly predictLogProba: (x: TensorLike) => Tensor
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
}

const optionalWeights = (data: LearnData): { readonly sampleWeight?: TensorLike } =>
  data.weights === undefined ? {} : { sampleWeight: data.weights }

/** The default score of a classifier: accuracy. */
export function accuracyOf(predict: (x: TensorLike) => Tensor): (data: LearnData) => number {
  return (data) => accuracyScore(data.y ?? [], predict(data.x), optionalWeights(data)).estimate.value
}

/** The default score of a regressor: R squared. */
export function r2Of(predict: (x: TensorLike) => Tensor): (data: LearnData) => number {
  return (data) => r2Score(data.y ?? [], predict(data.x), optionalWeights(data)).estimate.value
}

/**
 * The verbs of a classifier from its per-class joint log scores (`log prior + log likelihood`, up to a constant): the
 * posterior by softmax, the labels by argmax (ties to the first class), and the decision function (the difference of
 * the two scores for two classes, the scores otherwise, as scikit-learn).
 */
export function classifierFromScores(
  classes: readonly number[],
  d: number,
  scores: (m: MatrixView) => Float64Array,
  who: string,
): ScoreClassifierVerbs {
  const k = classes.length
  const read = (x: TensorLike): { readonly m: MatrixView; readonly s: Float64Array } => {
    const m = matrixView(x, who, { nFeatures: d })
    return { m, s: scores(m) }
  }
  const logProba = (x: TensorLike): Float64Array => {
    const { m, s } = read(x)
    const out = new Float64Array(m.n * k)
    for (let i = 0; i < m.n; i++) {
      const row = s.subarray(i * k, (i + 1) * k)
      const z = logSumExpOf(row)
      for (let c = 0; c < k; c++) out[i * k + c] = Number.isFinite(z) ? (row[c] ?? 0) - z : Number.NEGATIVE_INFINITY
    }
    return out
  }
  const predict = (x: TensorLike): Tensor => {
    const { m, s } = read(x)
    const out = new Float64Array(m.n)
    for (let i = 0; i < m.n; i++) out[i] = classes[argmaxOf(s.subarray(i * k, (i + 1) * k))] ?? Number.NaN
    return fromStorage(out, [m.n], 'float64')
  }
  return {
    classes,
    nFeaturesIn: d,
    predict,
    predictProba: (x) => {
      const lp = logProba(x)
      return fromStorage(
        lp.map((v) => Math.exp(v)),
        [lp.length / k, k],
        'float64',
      )
    },
    predictLogProba: (x) => {
      const lp = logProba(x)
      return fromStorage(lp, [lp.length / k, k], 'float64')
    },
    decisionFunction: (x) => {
      const { m, s } = read(x)
      if (k !== 2) return fromStorage(Float64Array.from(s), [m.n, k], 'float64')
      const out = new Float64Array(m.n)
      for (let i = 0; i < m.n; i++) out[i] = (s[i * 2 + 1] ?? 0) - (s[i * 2] ?? 0)
      return fromStorage(out, [m.n], 'float64')
    },
    score: accuracyOf(predict),
  }
}

/** The state of the algorithm that absorbs rows one per step. */
export type RowsState = { readonly n: number }

/**
 * An algorithm that reveals the training rows one per step, in order, the first already at step 0 (a model needs a row): a learner whose model is a function of the rows
 * it has seen (nearest neighbours, a closed-form fit) takes `params(state)` as that function of the first `n` rows, so
 * a session shows the model growing with the data and `fittedAt(t)` is the model on the first `t` rows.
 *
 * @param name - The algorithm's name.
 * @param total - The number of rows.
 * @param config - Its configuration, for the run header.
 * @returns The algorithm; it logs the metric `rows`.
 */
export function rowsAlgorithm(
  name: string,
  total: number,
  config: { readonly [key: string]: number | string } = {},
): Algorithm<RowsState> {
  return {
    name,
    config: { rows: total, ...config },
    schema: channels({ metrics: { rows: { description: 'training rows absorbed so far', direction: 'higher' } } }),
    init: () => ({ n: Math.min(1, total) }),
    step: (s) => ({ n: s.n + 1 }),
    done: (s) => (s.n >= total ? 'converged' : undefined),
    observe: (s) => ({ metrics: { rows: s.n } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as RowsState,
  }
}

/** The first `n` rows of a dataset (all of them once `n` reaches the end). */
export function prefixOf(data: LearnData, n: number): LearnData {
  const total = rowsOf(data.x)
  if (n >= total) return data
  return takeData(
    data,
    Array.from({ length: n }, (_, i) => i),
  )
}
