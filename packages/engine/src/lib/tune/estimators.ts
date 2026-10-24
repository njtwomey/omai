import { meanOf, varianceOf, WhiteboxError } from '../base/index.ts'
import {
  type Estimator,
  type LearnData,
  makeScorer,
  type ResponseMethod,
  type Scorer,
  type ScorerName,
  scorerFor,
  takeData,
} from '../learn/index.ts'
import { checkCv, type Fold, type Splitter, splitIndices, splitInputOf } from '../learn/model-selection/index.ts'
import type { Metric } from '../metrics/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { type Objective, type ObjectiveOptions, objectiveBase, type TrialContext } from './objectives.ts'
import type { StudyDirection } from './study.ts'

/** An estimator to re-create with each configuration (`estimator.with(params)`), or a function making one. */
export type EstimatorSource = Estimator | ((params: SpaceValue) => Estimator)

/**
 * The estimator for a configuration: `estimator.with(params)` (scikit-learn's `clone(est).set_params(**params)`), or
 * the factory's estimator.
 *
 * @param source - An estimator, or a factory from configurations to estimators.
 * @param params - The configuration (a record of hyperparameters for `with`).
 * @returns The estimator.
 * @throws WhiteboxError with code `value` when an estimator is given a configuration that is not a record.
 * @example
 * ```ts
 * import { estimatorFor } from 'whitebox-engine/tune'
 * import { standardScaler } from 'whitebox-engine/learn/preprocessing'
 * estimatorFor(standardScaler(), { withMean: false }).hyper // => { withMean: false, withStd: true }
 * ```
 */
export function estimatorFor(source: EstimatorSource, params: SpaceValue): Estimator {
  if (typeof source === 'function') return source(params)
  if (typeof params !== 'object' || params === null || Array.isArray(params))
    throw new WhiteboxError('value', `${source.name}: a configuration of an estimator is a record of hyperparameters`)
  return source.with(params as Readonly<Record<string, unknown>>)
}

/**
 * How an estimator objective scores a fold: a metric (its spec gives the direction), a `learn` scorer (maximised), or
 * neither: the estimator's own `score` (accuracy or R^2, maximised).
 */
export interface ScoringChoice {
  /** A metric scored on each test fold, minimised or maximised as its spec says. */
  readonly metric?: Metric
  /** A scorer or scikit-learn scorer name (higher is better), as `learn`'s searches take it. */
  readonly scoring?: ScorerName | Scorer
  /** The verb a metric reads (default: `predictProba` for metrics of probabilities or scores, else `predict`). */
  readonly response?: ResponseMethod
}

/** The scorer of a choice, the sign that turns its score into the objective value, and the objective's direction. */
function scorerOf(choice: ScoringChoice): {
  readonly scorer: Scorer
  readonly sign: number
  readonly direction: StudyDirection
  readonly name: string
} {
  if (choice.metric !== undefined) {
    const spec = choice.metric.spec
    const higher = spec.direction === 'higher'
    const response =
      choice.response ?? (spec.inputs === 'probabilities' || spec.inputs === 'scores' ? 'predictProba' : 'predict')
    const scorer = makeScorer(choice.metric, { response, greaterIsBetter: higher, name: spec.name })
    // makeScorer negates losses; the objective reports the metric itself and minimises it.
    return { scorer, sign: higher ? 1 : -1, direction: higher ? 'maximize' : 'minimize', name: spec.name }
  }
  // A scorer, a scorer name, or (neither given) the estimator's own `score`, as scikit-learn's searches.
  const scorer = scorerFor(choice.scoring)
  return { scorer, sign: 1, direction: 'maximize', name: scorer.name }
}

/** Options of {@link estimatorObjective}. */
export interface EstimatorObjectiveOptions extends ObjectiveOptions, ScoringChoice {
  /** The splitter (takes precedence over `cv`). */
  readonly splitter?: Splitter
  /** A splitter, or a number of folds (default 5: stratified for classifiers), as scikit-learn's `cv`. */
  readonly cv?: Splitter | number
  /** Group ids for group splitters. */
  readonly groups?: ArrayLike<number>
  /** The key of a shuffling splitter (default `key(0)`: every trial sees the same folds, as scikit-learn's searches). */
  readonly splitKey?: Key
}

/** Fit on one part and score the other, the fit a child run of the trial. */
function fitAndScore(
  estimator: Estimator,
  train: LearnData,
  test: LearnData,
  scorer: Scorer,
  sign: number,
  trial: TrialContext,
  key: Key,
): number {
  const fitted = estimator.fit(train, { key, parent: trial.run, explain: false })
  return sign * scorer.score(fitted, test)
}

/**
 * Cross-validated estimator performance as an objective: the configuration makes an estimator (`with`, or a
 * factory), which is fitted on every training fold (each fit a child run of the trial) and scored on the test fold;
 * the value is the mean score, with the direction of the metric (or maximised for a scorer). The running mean is
 * reported after each fold, so pruners can stop hopeless configurations early. Folds come from `learn`'s splitters
 * (`cv: 5` is stratified k-fold for classifiers, as scikit-learn's `check_cv`); attributes record the fold scores and
 * their standard deviation.
 *
 * @param estimator - An estimator (re-created with each configuration by `with`), or a factory.
 * @param data - The data.
 * @param options - The metric or scoring, the folds (`splitter`, or `cv`), groups, the splitter's key and a default
 *   space (default the estimator's `hyperSpace`).
 * @returns The objective.
 * @concept cross-validation
 * @example
 * ```ts
 * import { estimatorObjective, tune, gridSampler } from 'whitebox-engine/tune'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * import { dict, choice } from 'whitebox-engine/space'
 * import { asTensor, full } from 'whitebox-engine/tensor'
 * const constant = defineEstimator(
 *   { name: 'constant', kind: 'regressor', fit: () => ({}), methods: (_, h) => ({ predict: (x) => full([asTensor(x).shape[0] ?? 0], h.c) }) },
 *   { c: 0 },
 * )
 * const objective = estimatorObjective(constant, { x: [[0], [1], [2], [3]], y: [2, 2, 2, 2] }, { metric: meanSquaredError, cv: 2 })
 * tune(objective, dict({ c: choice([0, 2, 4]) }), { sampler: gridSampler() }).best?.params // => { c: 2 }
 * ```
 */
export function estimatorObjective(
  estimator: EstimatorSource,
  data: LearnData,
  options: EstimatorObjectiveOptions,
): Objective {
  const { scorer, sign, direction, name } = scorerOf(options)
  const space = options.space ?? (typeof estimator === 'function' ? undefined : estimator.hyperSpace)
  return {
    ...objectiveBase({ ...options, ...(space === undefined ? {} : { space }) }, `cv:${name}`, direction),
    metricNames: options.metricNames ?? [name],
    evaluate: (params, trial) => {
      const est = estimatorFor(estimator, params)
      const splitter = checkCv(options.splitter ?? options.cv, est)
      const folds: Fold[] = splitIndices(splitter, splitInputOf(data, options.groups), options.splitKey ?? makeKey(0))
      const scores: number[] = []
      for (const [i, fold] of folds.entries()) {
        scores.push(
          fitAndScore(
            est,
            takeData(data, fold.train),
            takeData(data, fold.test),
            scorer,
            sign,
            trial,
            foldIn(trial.key, i),
          ),
        )
        if (trial.report(i, meanOf(scores))) break
      }
      return { values: meanOf(scores), attrs: { foldScores: scores, std: Math.sqrt(varianceOf(scores)) } }
    },
  }
}

/**
 * Validation-set performance as an objective: fit on the training data, score on the validation data (reported once,
 * at step 0).
 *
 * @param estimator - An estimator (re-created by `with`), or a factory.
 * @param train - Training data.
 * @param validation - Validation data.
 * @param options - The metric or scoring and a default space.
 * @returns The objective.
 * @concept model-selection
 * @example
 * ```ts
 * import { holdoutObjective } from 'whitebox-engine/tune'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * import { asTensor, full } from 'whitebox-engine/tensor'
 * const zero = defineEstimator({ name: 'zero', kind: 'regressor', fit: () => ({}), methods: () => ({ predict: (x) => full([asTensor(x).shape[0] ?? 0], 0) }) }, {})
 * holdoutObjective(zero, { x: [[0], [1]], y: [0, 1] }, { x: [[0], [1]], y: [1, 1] }, { metric: meanSquaredError }).directions // => ['minimize']
 * ```
 */
export function holdoutObjective(
  estimator: EstimatorSource,
  train: LearnData,
  validation: LearnData,
  options: ObjectiveOptions & ScoringChoice,
): Objective {
  const { scorer, sign, direction, name } = scorerOf(options)
  const space = options.space ?? (typeof estimator === 'function' ? undefined : estimator.hyperSpace)
  return {
    ...objectiveBase({ ...options, ...(space === undefined ? {} : { space }) }, `holdout:${name}`, direction),
    metricNames: options.metricNames ?? [name],
    evaluate: (params, trial) => {
      const score = fitAndScore(estimatorFor(estimator, params), train, validation, scorer, sign, trial, trial.key)
      trial.report(0, score)
      return score
    },
  }
}
