import { meanOf, varianceOf, WhiteboxError } from '../base/index.ts'
import { type ClassificationData, type RegressionData, take } from '../data/index.ts'
import type { Metric } from '../metrics/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { Space, SpaceValue } from '../space/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { kFoldIndices, stratifiedKFoldIndices } from './_folds.ts'
import { type Objective, type ObjectiveOptions, objectiveBase, type TrialContext } from './objectives.ts'

/** A fitted model: predictions, and class probabilities for classifiers. */
export interface FittedModel {
  predict(x: TensorLike): TensorLike
  predictProba?(x: TensorLike): TensorLike
}

/**
 * The part of the estimator contract that tuning needs: a name, an optional `hyperSpace`, `fit`, and `withHyper` to
 * make the same estimator with other hyperparameters (scikit-learn's `clone(est).set_params(**params)`).
 * TODO(learn): `learn/estimator.ts` will define the full contract; it satisfies this one.
 */
export interface TunableEstimator<D = TabularData> {
  readonly name: string
  readonly hyperSpace?: Space
  fit(data: D, options?: { readonly key?: Key }): FittedModel
  withHyper?(params: SpaceValue): TunableEstimator<D>
}

/** Tabular data with features and targets (the schemas estimators train on). */
export type TabularData = RegressionData | ClassificationData

/** An estimator to re-create with each configuration, or a function making one from a configuration. */
export type EstimatorSource<D = TabularData> = TunableEstimator<D> | ((params: SpaceValue) => TunableEstimator<D>)

/** One cross-validation split: row indices to train on and to test on. */
export interface Fold {
  readonly train: readonly number[]
  readonly test: readonly number[]
}

/** A splitter: the folds of a dataset (a key for shuffling ones). */
export type Splitter<D = TabularData> = (data: D, key: Key) => readonly Fold[]

/**
 * Make the estimator for a configuration.
 *
 * @param source - An estimator with `withHyper`, or a factory.
 * @param params - The configuration.
 * @returns The estimator.
 * @throws WhiteboxError with code `value` when an estimator lacks `withHyper`.
 * @example
 * ```ts
 * import { estimatorFor } from 'whitebox-engine/tune'
 * const make = (p: unknown) => ({ name: `k=${(p as { k: number }).k}`, fit: () => ({ predict: () => [0] }) })
 * estimatorFor(make, { k: 3 }).name // => 'k=3'
 * ```
 */
export function estimatorFor<D>(source: EstimatorSource<D>, params: SpaceValue): TunableEstimator<D> {
  if (typeof source === 'function') return source(params)
  if (source.withHyper === undefined)
    throw new WhiteboxError('value', `${source.name}: tuning needs withHyper (or pass a factory)`)
  return source.withHyper(params)
}

function rows(data: TabularData): number {
  return data.y.shape[0] ?? 0
}

/**
 * The default folds for `cv: k`, as scikit-learn picks them: stratified k-fold for classification, k-fold otherwise,
 * unshuffled.
 *
 * @param data - The data.
 * @param k - The number of folds.
 * @returns The folds.
 * @throws WhiteboxError with code `value` for fewer than 2 folds or more folds than rows.
 * @example
 * ```ts
 * import { defaultFolds } from 'whitebox-engine/tune'
 * import { xy } from 'whitebox-engine/data'
 * defaultFolds(xy([[1], [2], [3], [4], [5]], [1, 2, 3, 4, 5]), 2).map((f) => f.test) // => [[0, 1, 2], [3, 4]]
 * ```
 */
export function defaultFolds(data: TabularData, k: number): Fold[] {
  return data.kind === 'classification' ? stratifiedKFoldIndices(toFlat(data.y), k) : kFoldIndices(rows(data), k)
}

/** Options of {@link estimatorObjective}. */
export interface EstimatorObjectiveOptions<D = TabularData> extends ObjectiveOptions {
  /** The metric scored on each test fold; its spec gives the direction. */
  readonly metric: Metric
  /** The splitter (the name of the worked path in the plan; takes precedence over `cv`). */
  readonly splitter?: Splitter<D>
  /** A splitter, or a number of folds (default 5: stratified for classification), as scikit-learn's `cv`. */
  readonly cv?: number | Splitter<D>
  /** Score `predictProba` instead of `predict` (default: when the metric reads probabilities or scores). */
  readonly response?: 'predict' | 'predictProba'
  /** The key of a shuffling splitter (default `key(0)`: every trial sees the same folds, as scikit-learn's searches). */
  readonly splitKey?: Key
}

function responseOf(options: Pick<EstimatorObjectiveOptions, 'metric' | 'response'>): 'predict' | 'predictProba' {
  const inputs = options.metric.spec.inputs
  return options.response ?? (inputs === 'probabilities' || inputs === 'scores' ? 'predictProba' : 'predict')
}

/** Fit on one dataset and score the metric on another. */
function fitAndScore(
  source: EstimatorSource,
  params: SpaceValue,
  train: TabularData,
  test: TabularData,
  options: Pick<EstimatorObjectiveOptions, 'metric' | 'response'>,
  key: Key,
): number {
  const estimator = estimatorFor(source, params)
  const fitted = estimator.fit(train, { key })
  let predicted: TensorLike
  if (responseOf(options) === 'predictProba') {
    if (fitted.predictProba === undefined)
      throw new WhiteboxError('value', `${estimator.name}: the metric needs predictProba`)
    predicted = fitted.predictProba(test.x)
  } else predicted = fitted.predict(test.x)
  return options.metric(test.y, asTensor(predicted)).estimate.value
}

/** The fold scores of a configuration (scikit-learn's `cross_val_score`), reporting the running mean after each fold. */
function crossValidate(
  source: EstimatorSource,
  data: TabularData,
  params: SpaceValue,
  folds: readonly Fold[],
  options: Pick<EstimatorObjectiveOptions, 'metric' | 'response'>,
  trial: TrialContext,
): number[] {
  const scores: number[] = []
  for (const [i, fold] of folds.entries()) {
    scores.push(
      fitAndScore(source, params, take(data, fold.train), take(data, fold.test), options, foldIn(trial.key, i)),
    )
    if (trial.report(i, meanOf(scores))) break
  }
  return scores
}

/**
 * Cross-validated estimator performance as an objective: fit the estimator with each configuration on every training
 * fold, score the metric on the test fold, and return the mean (direction from the metric's spec). The running mean
 * is reported after each fold, so pruners can stop hopeless configurations early. Attributes record the fold scores
 * and their standard deviation.
 *
 * @param estimator - An estimator with `withHyper`, or a factory from configurations to estimators.
 * @param data - The data.
 * @param options - The metric, the folds (`splitter`, or `cv`: a splitter or a count, default 5), the prediction to
 *   score, the splitter's key, and a default space (default the estimator's `hyperSpace`).
 * @returns The objective.
 * @concept cross-validation
 * @example
 * ```ts
 * import { estimatorObjective, tune, gridSampler } from 'whitebox-engine/tune'
 * import { xy } from 'whitebox-engine/data'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * import { dict, choice } from 'whitebox-engine/space'
 * // A constant predictor: the configuration is the constant.
 * const make = (p: unknown) => ({ name: 'constant', fit: () => ({ predict: (x: unknown) => (x as { shape: number[] }).shape.slice(0, 1).map(() => (p as { c: number }).c).flatMap((c) => Array((x as { shape: number[] }).shape[0]).fill(c)) }) })
 * const data = xy([[0], [1], [2], [3]], [2, 2, 2, 2])
 * const objective = estimatorObjective(make, data, { metric: meanSquaredError, cv: 2 })
 * tune(objective, dict({ c: choice([0, 2, 4]) }), { sampler: gridSampler() }).best?.params // => { c: 2 }
 * ```
 */
export function estimatorObjective(
  estimator: EstimatorSource,
  data: TabularData,
  options: EstimatorObjectiveOptions,
): Objective {
  const space = options.space ?? (typeof estimator === 'function' ? undefined : estimator.hyperSpace)
  const spec = options.metric.spec
  return {
    ...objectiveBase(
      { ...options, ...(space === undefined ? {} : { space }) },
      `cv:${spec.name}`,
      spec.direction === 'higher' ? 'maximize' : 'minimize',
    ),
    metricNames: options.metricNames ?? [spec.name],
    evaluate: (params, trial) => {
      const cv = options.splitter ?? options.cv ?? 5
      const folds = typeof cv === 'number' ? defaultFolds(data, cv) : cv(data, options.splitKey ?? makeKey(0))
      const scores = crossValidate(estimator, data, params, folds, options, trial)
      return {
        values: meanOf(scores),
        attrs: { foldScores: scores, std: Math.sqrt(varianceOf(scores)) },
      }
    },
  }
}

/**
 * Validation-set performance as an objective: fit on the training data, score the metric on the validation data.
 *
 * @param estimator - An estimator with `withHyper`, or a factory.
 * @param train - Training data.
 * @param validation - Validation data.
 * @param options - The metric, the prediction to score and a default space.
 * @returns The objective.
 * @concept model-selection
 * @example
 * ```ts
 * import { holdoutObjective } from 'whitebox-engine/tune'
 * import { xy } from 'whitebox-engine/data'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * const make = () => ({ name: 'zero', fit: () => ({ predict: () => [0, 0] }) })
 * holdoutObjective(make, xy([[0], [1]], [0, 1]), xy([[0], [1]], [1, 1]), { metric: meanSquaredError }).directions // => ['minimize']
 * ```
 */
export function holdoutObjective(
  estimator: EstimatorSource,
  train: TabularData,
  validation: TabularData,
  options: Omit<EstimatorObjectiveOptions, 'cv' | 'splitter' | 'splitKey'>,
): Objective {
  const space = options.space ?? (typeof estimator === 'function' ? undefined : estimator.hyperSpace)
  const spec = options.metric.spec
  return {
    ...objectiveBase(
      { ...options, ...(space === undefined ? {} : { space }) },
      `holdout:${spec.name}`,
      spec.direction === 'higher' ? 'maximize' : 'minimize',
    ),
    metricNames: options.metricNames ?? [spec.name],
    evaluate: (params, trial) => {
      const score = fitAndScore(estimator, params, train, validation, options, trial.key)
      trial.report(0, score)
      return score
    },
  }
}
