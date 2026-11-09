/**
 * Baselines that ignore the features: `dummyClassifier` (the class prior, the most frequent class, a draw from the
 * prior, a uniform draw, or a constant label) and `dummyRegressor` (the mean, the median, a quantile or a constant of
 * the targets), scikit-learn's `DummyClassifier` and `DummyRegressor`. They are what a learner is compared against: a
 * model that does not beat them has learnt nothing. The random strategies draw from a key kept in the fitted
 * parameters, so a fitted dummy predicts the same thing for the same rows every time.
 */

import { OmaiError } from '../base/index.ts'
import { categorical, type Key, randint } from '../random/index.ts'
import { choice, dict, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { keyFor } from './_shared.ts'
import { defineEstimator, type Estimator, type Fitted, requireTargets } from './estimator.ts'
import { encodeLabels, rowsOf, sampleWeightsOf, targetVector } from './validation.ts'

/** How {@link dummyClassifier} predicts: see its documentation. */
export type DummyClassifierStrategy = 'prior' | 'most_frequent' | 'stratified' | 'uniform' | 'constant'

/** Hyperparameters of {@link dummyClassifier}. */
export interface DummyClassifierHyper {
  /** The rule (default `prior`). */
  readonly strategy: DummyClassifierStrategy
  /** The label of the `constant` strategy, which must be one of the training labels. */
  readonly constant?: number
}

/** The fitted parameters of a dummy classifier: the classes, their (weighted) frequencies, and the key of the random strategies. */
export type DummyClassifierParams = {
  readonly classes: number[]
  readonly prior: number[]
  readonly strategy: DummyClassifierStrategy
  /** The index in `classes` of the constant label (0 when the strategy has none). */
  readonly constantIndex: number
  readonly keyWords: number[]
  readonly nFeaturesIn: number
}

/** The verbs of a fitted dummy classifier. */
export interface DummyClassifierVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
}

const keyOf = (words: readonly number[]): Key => ({ kind: 'key', words: [words[0] ?? 0, words[1] ?? 0] })

/** The class index of each of `n` rows, and the probability rows, under a strategy. */
function dummyProbabilities(p: DummyClassifierParams, n: number): number[] {
  const k = p.classes.length
  const out = new Array<number>(n * k).fill(0)
  const oneHot = (i: number, c: number): void => {
    out[i * k + c] = 1
  }
  const mostFrequent = p.prior.reduce((best, v, c) => (v > (p.prior[best] ?? 0) ? c : best), 0)
  switch (p.strategy) {
    case 'prior':
      for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) out[i * k + c] = p.prior[c] ?? 0
      break
    case 'most_frequent':
      for (let i = 0; i < n; i++) oneHot(i, mostFrequent)
      break
    case 'constant':
      for (let i = 0; i < n; i++) oneHot(i, p.constantIndex)
      break
    case 'uniform':
      for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) out[i * k + c] = 1 / k
      break
    case 'stratified': {
      const draws = categorical(keyOf(p.keyWords), p.prior, [n]).data
      for (let i = 0; i < n; i++) oneHot(i, draws[i] ?? 0)
      break
    }
  }
  return out
}

function dummyClassifierVerbs(p: DummyClassifierParams): DummyClassifierVerbs {
  const k = p.classes.length
  return {
    classes: p.classes,
    nFeaturesIn: p.nFeaturesIn,
    predictProba: (x) => {
      const n = rowsOf(x)
      return fromStorage(Float64Array.from(dummyProbabilities(p, n)), [n, k], 'float64')
    },
    predict: (x) => {
      const n = rowsOf(x)
      let index: ArrayLike<number>
      if (p.strategy === 'uniform') index = randint(keyOf(p.keyWords), [n], 0, k).data
      else {
        const proba = dummyProbabilities(p, n)
        // prior: the most frequent class (the first on ties), as the others' one-hot rows say.
        index = Array.from({ length: n }, (_, i) => {
          let best = 0
          for (let c = 1; c < k; c++) if ((proba[i * k + c] ?? 0) > (proba[i * k + best] ?? 0)) best = c
          return best
        })
      }
      return fromStorage(
        Float64Array.from(index, (c) => p.classes[c] ?? 0),
        [n],
        'float64',
      )
    },
  }
}

/**
 * The baseline classifier (scikit-learn's `DummyClassifier`), which never looks at the features. `prior` (default)
 * predicts the most frequent class and returns the class frequencies as probabilities; `most_frequent` the same with
 * one-hot probabilities; `stratified` draws each prediction from the class frequencies and `uniform` uniformly over
 * the classes (both from a key kept in the fitted parameters, so the same rows get the same predictions each time);
 * `constant` always predicts the given label. Frequencies are weighted by the sample weights. One target column.
 *
 * @param hyper - The `strategy` and, for `constant`, the label.
 * @returns The estimator.
 * @index estimator Baseline classifier: prior, most frequent, stratified, uniform or constant.
 * @example
 * ```ts
 * import { dummyClassifier } from 'omai-engine/learn'
 * import { toFlat } from 'omai-engine/tensor'
 * const f = dummyClassifier({ strategy: 'prior' }).fit({ x: [[0], [1], [2], [3]], y: [1, 1, 1, 0] })
 * toFlat(f.predict([[5], [6]])) // => [1, 1]
 * toFlat(f.predictProba([[5]])) // => [0.25, 0.75]
 * ```
 */
export function dummyClassifier(
  hyper: Partial<DummyClassifierHyper> = {},
): Estimator<DummyClassifierHyper, Fitted<DummyClassifierParams> & DummyClassifierVerbs> {
  const h: DummyClassifierHyper = {
    strategy: hyper.strategy ?? 'prior',
    ...(hyper.constant === undefined ? {} : { constant: hyper.constant }),
  }
  return defineEstimator<DummyClassifierHyper, DummyClassifierParams, DummyClassifierVerbs>(
    {
      name: 'dummy/classifier',
      kind: 'classifier',
      hyperSpace: (current) =>
        dict({
          strategy: choice(['prior', 'most_frequent', 'stratified', 'uniform'] as const, {
            default: current.strategy === 'constant' ? 'prior' : current.strategy,
          }),
        }),
      fit: (data, hy, options) => {
        const y = requireTargets(data, 'dummyClassifier')
        const { classes, codes } = encodeLabels(y, 'dummyClassifier')
        const w = sampleWeightsOf(data.weights, codes.length, 'dummyClassifier')
        const counts = new Array<number>(classes.length).fill(0)
        for (const [i, c] of codes.entries()) counts[c] = (counts[c] ?? 0) + (w[i] ?? 0)
        const total = counts.reduce((a, v) => a + v, 0)
        let constantIndex = 0
        if (hy.strategy === 'constant') {
          if (hy.constant === undefined)
            throw new OmaiError('value', 'dummyClassifier: the constant strategy needs a constant label')
          constantIndex = classes.indexOf(hy.constant)
          if (constantIndex < 0)
            throw new OmaiError('value', `dummyClassifier: the constant ${hy.constant} is not among the labels`)
        }
        return {
          classes: [...classes],
          prior: counts.map((v) => v / total),
          strategy: hy.strategy,
          constantIndex,
          keyWords: [...keyFor(options, 'dummyClassifier').words],
          nFeaturesIn: matrixWidth(data.x),
        }
      },
      methods: (params) => dummyClassifierVerbs(params),
    },
    h,
  )
}

const matrixWidth = (x: TensorLike): number => {
  const first = Array.isArray(x) ? x[0] : undefined
  if (Array.isArray(first)) return first.length
  if (Array.isArray(x)) return 1
  const shape = (x as { readonly shape?: readonly number[] }).shape
  return shape?.[1] ?? 1
}

/** How {@link dummyRegressor} predicts: see its documentation. */
export type DummyRegressorStrategy = 'mean' | 'median' | 'quantile' | 'constant'

/** Hyperparameters of {@link dummyRegressor}. */
export interface DummyRegressorHyper {
  /** The rule (default `mean`). */
  readonly strategy: DummyRegressorStrategy
  /** The value of the `constant` strategy. */
  readonly constant?: number
  /** The quantile in `[0, 1]` of the `quantile` strategy. */
  readonly quantile?: number
}

/** The fitted parameters of a dummy regressor: the constant it predicts. */
export type DummyRegressorParams = { readonly constant: number; readonly nFeaturesIn: number }

/** The verbs of a fitted dummy regressor. */
export interface DummyRegressorVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/**
 * The `q` quantile of values with weights: numpy's linear interpolation when the weights are equal, and otherwise the
 * smallest value whose cumulative weight reaches `q` of the total (the inverted empirical distribution function).
 *
 * @param y - The values.
 * @param w - Their weights, or undefined for equal weights.
 * @param q - The quantile in `[0, 1]`.
 * @returns The quantile.
 * @example
 * ```ts
 * import { weightedQuantile } from 'omai-engine/learn'
 * weightedQuantile([1, 2, 3, 4], undefined, 0.5) // => 2.5
 * weightedQuantile([1, 2, 3, 4], [1, 1, 1, 5], 0.5) // => 4
 * ```
 */
export function weightedQuantile(y: ArrayLike<number>, w: ArrayLike<number> | undefined, q: number): number {
  const order = Array.from({ length: y.length }, (_, i) => i).sort((a, b) => (y[a] ?? 0) - (y[b] ?? 0))
  const sorted = order.map((i) => y[i] ?? 0)
  if (w === undefined) {
    const pos = q * (sorted.length - 1)
    const lo = Math.floor(pos)
    const hi = Math.ceil(pos)
    return (sorted[lo] ?? 0) + ((sorted[hi] ?? 0) - (sorted[lo] ?? 0)) * (pos - lo)
  }
  const total = order.reduce((a, i) => a + (w[i] ?? 0), 0)
  let run = 0
  for (const [r, i] of order.entries()) {
    run += w[i] ?? 0
    if (run >= q * total - 1e-12 * total) return sorted[r] ?? 0
  }
  return sorted[sorted.length - 1] ?? 0
}

/**
 * The baseline regressor (scikit-learn's `DummyRegressor`), which never looks at the features: it predicts the mean
 * (default), the median, a quantile (`quantile` in `[0, 1]`, numpy's linear interpolation) or a constant of the
 * training targets, weighted by the sample weights when given. One target column.
 *
 * @param hyper - The `strategy`, and the `quantile` or `constant` it needs.
 * @returns The estimator.
 * @index estimator Baseline regressor: mean, median, quantile or constant.
 * @example
 * ```ts
 * import { dummyRegressor } from 'omai-engine/learn'
 * import { toFlat } from 'omai-engine/tensor'
 * const f = dummyRegressor({ strategy: 'median' }).fit({ x: [[0], [1], [2]], y: [1, 2, 10] })
 * toFlat(f.predict([[7], [8]])) // => [2, 2]
 * ```
 */
export function dummyRegressor(
  hyper: Partial<DummyRegressorHyper> = {},
): Estimator<DummyRegressorHyper, Fitted<DummyRegressorParams> & DummyRegressorVerbs> {
  const h: DummyRegressorHyper = {
    strategy: hyper.strategy ?? 'mean',
    ...(hyper.constant === undefined ? {} : { constant: hyper.constant }),
    ...(hyper.quantile === undefined ? {} : { quantile: hyper.quantile }),
  }
  return defineEstimator<DummyRegressorHyper, DummyRegressorParams, DummyRegressorVerbs>(
    {
      name: 'dummy/regressor',
      kind: 'regressor',
      hyperSpace: (current) =>
        dict({
          strategy: choice(['mean', 'median'] as const, { default: current.strategy === 'median' ? 'median' : 'mean' }),
          ...(current.strategy === 'quantile' ? { quantile: real(0, 1, { default: current.quantile ?? 0.5 }) } : {}),
        }),
      fit: (data, hy) => {
        const y = targetVector(requireTargets(data, 'dummyRegressor'), 'dummyRegressor')
        const w = data.weights === undefined ? undefined : sampleWeightsOf(data.weights, y.length, 'dummyRegressor')
        let constant: number
        switch (hy.strategy) {
          case 'mean': {
            const ws = w ?? new Float64Array(y.length).fill(1)
            let num = 0
            let den = 0
            for (const [i, v] of y.entries()) {
              num += (ws[i] ?? 0) * v
              den += ws[i] ?? 0
            }
            constant = num / den
            break
          }
          case 'median':
            constant = weightedQuantile(y, w, 0.5)
            break
          case 'quantile': {
            if (hy.quantile === undefined || !(hy.quantile >= 0 && hy.quantile <= 1))
              throw new OmaiError('value', 'dummyRegressor: the quantile strategy needs a quantile in [0, 1]')
            constant = weightedQuantile(y, w, hy.quantile)
            break
          }
          case 'constant':
            if (hy.constant === undefined)
              throw new OmaiError('value', 'dummyRegressor: the constant strategy needs a constant')
            constant = hy.constant
            break
        }
        return { constant, nFeaturesIn: matrixWidth(data.x) }
      },
      methods: (params) => ({
        nFeaturesIn: params.nFeaturesIn,
        predict: (x) => {
          const n = rowsOf(x)
          return fromStorage(new Float64Array(n).fill(params.constant), [n], 'float64')
        },
      }),
    },
    h,
  )
}
