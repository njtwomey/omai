/**
 * Cross-validation, scikit-learn style: `crossValidate` and `crossValScore` fit an estimator on each fold's training
 * rows and score it on the held-out rows, `crossValPredict` gives out-of-fold predictions, learning and validation
 * curves show scores against training size or a hyperparameter, and `permutationTestScore` tests whether a score beats
 * chance.
 */

import { MdmlError, meanOf, varianceOf } from '../../base/index.ts'
import { type Explained, learningCurve as learningCurveRecord } from '../../exposition/index.ts'
import { foldIn, type Key, permutation } from '../../random/index.ts'
import { fromStorage, type Tensor, toFlat } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  type RunHandle,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  subRun,
} from '../../trace/index.ts'
import { innerFitOptions, keyFor, runOptionsFor } from '../_shared.ts'
import { type Estimator, type FitOptions, type Fitted, type LearnData, requireVerb } from '../estimator.ts'
import { type Scorer, type Scoring, scorersOf } from '../scoring.ts'
import { encodeLabels, rowsOf, takeData, targetVector } from '../validation.ts'
import { scoreRecordFromJson, scoreRecordToJson, scoresFromJson, scoresToJson } from './_state.ts'
import { type Fold, kFold, type SplitInput, type Splitter, splitIndices, stratifiedKFold } from './splitters.ts'

/** A splitter, or a number of folds (stratified for classifiers, plain k-fold otherwise, as sklearn's `check_cv`). */
export type CvOption = Splitter | number

/**
 * The splitter an option names: a number of folds becomes stratified k-fold for classifiers and k-fold otherwise.
 *
 * @param cv - A splitter or a number of folds (default 5).
 * @param estimator - The estimator (its kind decides stratification).
 * @returns The splitter.
 * @example
 * ```ts
 * import { checkCv } from 'mdml-engine/learn/model-selection'
 * checkCv(3, { kind: 'classifier' }).name // => 'stratified-k-fold'
 * ```
 */
export function checkCv(cv: CvOption | undefined, estimator: Pick<Estimator, 'kind'>): Splitter {
  if (cv === undefined || typeof cv === 'number') {
    const k = cv ?? 5
    return estimator.kind === 'classifier' ? stratifiedKFold(k) : kFold(k)
  }
  return cv
}

/**
 * The split input of learning data: the number of rows, the labels (when `y` is a vector) and the groups.
 *
 * @param data - The data.
 * @param groups - Group ids, if any.
 * @returns The splitter input.
 * @example
 * ```ts
 * import { splitInputOf } from 'mdml-engine/learn/model-selection'
 * splitInputOf({ x: [[1], [2]], y: [0, 1] }).n // => 2
 * ```
 */
export function splitInputOf(data: LearnData, groups?: ArrayLike<number>): SplitInput {
  const n = rowsOf(data.x)
  return {
    n,
    ...(data.y === undefined || !isLabelVector(data.y, n) ? {} : { y: targetVector(data.y, 'split', n) }),
    ...(groups === undefined ? {} : { groups }),
  }
}

function isLabelVector(y: LearnData['y'], n: number): boolean {
  try {
    return targetVector(y ?? [], 'split', n).length === n
  } catch {
    return false
  }
}

/** Options of {@link crossValidate}. */
export interface CrossValidateOptions extends FitOptions {
  /** The splitter or number of folds (default 5). */
  readonly cv?: CvOption
  /** One or several scorers (default the estimator's `score`). */
  readonly scoring?: Scoring
  /** Also score the training folds (default false). */
  readonly returnTrainScore?: boolean
  /** Keep the fitted estimator of each fold (default false). */
  readonly returnEstimator?: boolean
  /** Group ids for group splitters. */
  readonly groups?: ArrayLike<number>
  /** The score of a fold whose fit throws, or `'raise'` (default) to rethrow. */
  readonly errorScore?: number | 'raise'
}

/** The result of {@link crossValidate}. */
export interface CrossValidateResult {
  /** Test scores per scorer, one per fold. */
  readonly testScores: Readonly<Record<string, readonly number[]>>
  /** Training scores per scorer, when asked. */
  readonly trainScores?: Readonly<Record<string, readonly number[]>>
  /** The fitted estimator of each fold, when asked. */
  readonly estimators?: readonly Fitted[]
  /** The folds used. */
  readonly folds: readonly Fold[]
  /** Mean and standard deviation (population) of the test scores per scorer. */
  readonly mean: Readonly<Record<string, number>>
  readonly std: Readonly<Record<string, number>>
}

interface FoldOutcome {
  readonly test: Readonly<Record<string, number>>
  readonly train?: Readonly<Record<string, number>>
  readonly fitted?: Fitted
}

function scoreAll(scorers: Readonly<Record<string, Scorer>>, fitted: Fitted, data: LearnData): Record<string, number> {
  return Object.fromEntries(Object.entries(scorers).map(([name, s]) => [name, s.score(fitted, data)]))
}

/** Fit on one fold and score it, as a one-step child run. */
function runFold(
  ctx: { readonly key: Key; readonly run: RunHandle },
  index: number,
  estimator: Estimator,
  data: LearnData,
  fold: Fold,
  scorers: Readonly<Record<string, Scorer>>,
  options: CrossValidateOptions,
): FoldOutcome {
  const foldAlgorithm: Algorithm<FoldOutcome | undefined> = {
    name: 'fold',
    config: { fold: index, train: fold.train.length, test: fold.test.length },
    init: () => undefined,
    step: (_, c) => {
      const train = takeData(data, fold.train)
      const test = takeData(data, fold.test)
      let fitted: Fitted
      try {
        fitted = estimator.fit(train, innerFitOptions(options, foldIn(c.key, 0), c.run))
      } catch (e) {
        if (options.errorScore === undefined || options.errorScore === 'raise') throw e
        const failed = Object.fromEntries(Object.keys(scorers).map((n) => [n, options.errorScore as number]))
        return { test: failed, ...(options.returnTrainScore === true ? { train: failed } : {}) }
      }
      return {
        test: scoreAll(scorers, fitted, test),
        ...(options.returnTrainScore === true ? { train: scoreAll(scorers, fitted, train) } : {}),
        ...(options.returnEstimator === true ? { fitted } : {}),
      }
    },
    done: (s) => (s === undefined ? undefined : 'completed'),
    observe: (s) => ({ metrics: s === undefined ? {} : s.test }),
  }
  const r = subRun({ t: index, key: ctx.key, run: ctx.run }, foldAlgorithm, { maxSteps: 1, index })
  if (r.state === undefined) throw new MdmlError('convergence', 'crossValidate: a fold did not run')
  return r.state
}

/** The paths {@link crossValidateAlgorithm} logs: the fold, its sizes and scores (the first scorer, then every scorer). */
export type CrossValidatePath =
  | 'cv/fold'
  | 'cv/trainSize'
  | 'cv/testSize'
  | 'cv/testScore'
  | 'cv/trainScore'
  | 'cv/test/*'
  | 'cv/train/*'

/** What a stepped cross-validation logs, by path. */
export const crossValidateContract: Contract<unknown, CrossValidatePath> = defineContract({
  'cv/fold': scalar({ description: 'the fold just fitted and scored (from 0)' }),
  'cv/trainSize': scalar({ description: 'rows the fold was fitted on' }),
  'cv/testSize': scalar({ description: 'rows the fold was scored on' }),
  'cv/testScore': scalar({ description: 'the test score of the fold under the first scorer', direction: 'higher' }),
  'cv/trainScore': scalar({ description: 'the training score of the fold under the first scorer (when asked)' }),
  'cv/test/*': scalar({ description: 'the test score of the fold under each scorer, by name' }),
  'cv/train/*': scalar({ description: 'the training score of the fold under each scorer, by name (when asked)' }),
})

/**
 * The state of {@link crossValidateAlgorithm}: the folds done and, per scorer, the test scores (and the training scores
 * when asked) of the folds so far. It serialises; the fitted estimators of `returnEstimator` are not part of it.
 */
export interface CvState {
  /** Folds fitted and scored so far. */
  readonly done: number
  readonly test: Readonly<Record<string, readonly number[]>>
  readonly train: Readonly<Record<string, readonly number[]>>
}

interface CvRun {
  readonly algorithm: Algorithm<CvState, CrossValidatePath>
  readonly folds: Fold[]
  readonly scorers: Readonly<Record<string, Scorer>>
  /** The fitted estimators of the folds so far, when `returnEstimator`. */
  readonly kept: Fitted[]
}

function crossValidateRun(estimator: Estimator, data: LearnData, options: CrossValidateOptions): CvRun {
  const splitter = checkCv(options.cv, estimator)
  const folds = splitIndices(splitter, splitInputOf(data, options.groups), foldIn(keyFor(options, 'crossValidate'), 0))
  const scorers = scorersOf(options.scoring)
  const names = Object.keys(scorers)
  const withTrain = options.returnTrainScore === true
  const kept: Fitted[] = []
  const empty = (): Record<string, number[]> => Object.fromEntries(names.map((n) => [n, []]))
  const algorithm: Algorithm<CvState, CrossValidatePath> = {
    name: 'cross-validate',
    config: { estimator: estimator.name, splitter: splitter.name, folds: folds.length, scoring: names },
    schema: channels({
      metrics: Object.fromEntries(
        names.map((n) => [n, { description: `the test ${n} of the fold`, direction: 'higher' as const }]),
      ),
      messages: {
        fold: {
          description: 'a fold was fitted and scored',
          payload: { fold: 'number', train: 'number', test: 'number' },
        },
      },
    }),
    init: () => ({ done: 0, test: empty(), train: withTrain ? empty() : {} }),
    step: (s, ctx) => {
      const fold = folds[s.done] as Fold
      const o = runFold(ctx, s.done, estimator, data, fold, scorers, options)
      if (o.fitted !== undefined) kept.push(o.fitted)
      ctx.log('cv/fold', s.done)
      ctx.log('cv/trainSize', fold.train.length)
      ctx.log('cv/testSize', fold.test.length)
      const first = names[0]
      if (first !== undefined) ctx.log('cv/testScore', o.test[first] ?? Number.NaN)
      if (first !== undefined && o.train !== undefined) ctx.log('cv/trainScore', o.train[first] ?? Number.NaN)
      for (const n of names) {
        ctx.log(`cv/test/${n}`, o.test[n] ?? Number.NaN)
        if (o.train !== undefined) ctx.log(`cv/train/${n}`, o.train[n] ?? Number.NaN)
      }
      const add = (
        prev: Readonly<Record<string, readonly number[]>>,
        got: Readonly<Record<string, number>> | undefined,
      ) =>
        got === undefined ? prev : Object.fromEntries(names.map((n) => [n, [...(prev[n] ?? []), got[n] ?? Number.NaN]]))
      return { done: s.done + 1, test: add(s.test, o.test), train: withTrain ? add(s.train, o.train) : {} }
    },
    done: (s) => (s.done >= folds.length ? 'completed' : undefined),
    observe: (s) => {
      const last = Object.fromEntries(names.map((n) => [n, s.test[n]?.at(-1) ?? Number.NaN]))
      const f = folds[s.done - 1]
      return {
        metrics: s.done === 0 ? {} : last,
        messages:
          f === undefined || s.done === 0
            ? []
            : [{ kind: 'fold', payload: { fold: s.done - 1, train: f.train.length, test: f.test.length } }],
      }
    },
    serialise: (s) => stateToJson({ done: s.done, test: scoreRecordToJson(s.test), train: scoreRecordToJson(s.train) }),
    deserialise: (j) => {
      const o = stateFromJson(j) as { readonly done: number; readonly test: never; readonly train: never }
      return { done: o.done, test: scoreRecordFromJson(o.test), train: scoreRecordFromJson(o.train) }
    },
  }
  return { algorithm, folds, scorers, kept }
}

/**
 * Cross-validation as a stepped algorithm: each step fits the estimator on one fold's training rows and scores it on
 * the held-out rows (and on the training rows when asked), the fold a child run under which the estimator's own fit
 * nests. The state (the scores so far) serialises, so a cross-validation can be paused and resumed; the fitted
 * estimators of `returnEstimator` are kept by {@link crossValidate}, not in the state.
 *
 * @param estimator - The estimator.
 * @param data - The data (features, targets, weights).
 * @param options - cv, scoring, returnTrainScore, groups, errorScore and key.
 * @returns The algorithm, one step per fold.
 * @concept cross-validation
 * @example
 * ```ts
 * import { crossValidateAlgorithm } from 'mdml-engine/learn/model-selection'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { full } from 'mdml-engine/tensor'
 * const zero = defineEstimator({ name: 'zero', kind: 'regressor', fit: () => ({}), methods: () => ({ predict: (x) => full([x.shape[0]], 0) }) }, {})
 * const alg = crossValidateAlgorithm(zero, { x: [[0], [1], [2], [3]], y: [1, 1, 1, 1] }, { cv: 2, scoring: 'neg_mean_squared_error' })
 * runAlgorithm(alg, { maxSteps: 5 }).state.test.neg_mean_squared_error // => [-1, -1]
 * ```
 */
export function crossValidateAlgorithm(
  estimator: Estimator,
  data: LearnData,
  options: CrossValidateOptions = {},
): Algorithm<CvState, CrossValidatePath> {
  return crossValidateRun(estimator, data, options).algorithm
}

/**
 * Cross-validate an estimator (sklearn.model_selection.cross_validate): for each fold, fit on the training rows and
 * score on the test rows (and optionally the training rows). This is {@link crossValidateAlgorithm} run to the end, a
 * traced algorithm with one step per fold, each fold a child run under which the estimator's own fit nests.
 *
 * @param estimator - The estimator.
 * @param data - The data (features, targets, weights).
 * @param options - cv, scoring, returnTrainScore, returnEstimator, groups, errorScore, key and tracing.
 * @returns Scores per scorer and fold, their means and deviations, the folds and optionally the fitted estimators.
 * @concept cross-validation
 * @index function Cross-validate an estimator with several scorers.
 * @example
 * ```ts
 * import { crossValidate } from 'mdml-engine/learn/model-selection'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { full, mean, item } from 'mdml-engine/tensor'
 * const meanModel = defineEstimator(
 *   { name: 'mean', kind: 'regressor', fit: (d) => ({ m: item(mean(d.y)) }), methods: (p) => ({ predict: (x) => full([x.shape[0]], p.m) }) },
 *   {},
 * )
 * const r = crossValidate(meanModel, { x: [[0], [1], [2], [3]], y: [1, 1, 1, 1] }, { cv: 2, scoring: 'neg_mean_absolute_error' })
 * r.testScores.neg_mean_absolute_error // => [-0, -0]
 * ```
 */
export function crossValidate(
  estimator: Estimator,
  data: LearnData,
  options: CrossValidateOptions = {},
): CrossValidateResult {
  const { algorithm, folds, scorers, kept } = crossValidateRun(estimator, data, options)
  const { state } = runAlgorithm(algorithm, { ...runOptionsFor(options, 'crossValidate'), maxSteps: folds.length })
  const names = Object.keys(scorers)
  const testScores = Object.fromEntries(names.map((n) => [n, [...(state.test[n] ?? [])]]))
  return {
    testScores,
    ...(options.returnTrainScore === true
      ? { trainScores: Object.fromEntries(names.map((n) => [n, [...(state.train[n] ?? [])]])) }
      : {}),
    ...(options.returnEstimator === true ? { estimators: kept } : {}),
    folds,
    mean: Object.fromEntries(names.map((n) => [n, meanOf(testScores[n] ?? [])])),
    std: Object.fromEntries(names.map((n) => [n, Math.sqrt(varianceOf(testScores[n] ?? [], 0))])),
  }
}

/**
 * The test scores of one scorer across folds (sklearn.model_selection.cross_val_score).
 *
 * @param estimator - The estimator.
 * @param data - The data.
 * @param options - As {@link crossValidate}, with a single scorer.
 * @returns One score per fold.
 * @example
 * ```ts
 * import { crossValScore } from 'mdml-engine/learn/model-selection'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { full } from 'mdml-engine/tensor'
 * const zero = defineEstimator({ name: 'zero', kind: 'regressor', fit: () => ({}), methods: () => ({ predict: (x) => full([x.shape[0]], 0) }) }, {})
 * crossValScore(zero, { x: [[0], [1]], y: [1, 3] }, { cv: 2, scoring: 'neg_mean_squared_error' }) // => [-1, -9]
 * ```
 */
export function crossValScore(estimator: Estimator, data: LearnData, options: CrossValidateOptions = {}): number[] {
  const r = crossValidate(estimator, data, options)
  return [...(Object.values(r.testScores)[0] ?? [])]
}

/** Options of {@link crossValPredict}. */
export interface CrossValPredictOptions extends FitOptions {
  readonly cv?: CvOption
  /** The verb whose out-of-fold output is collected (default `'predict'`). */
  readonly method?: 'predict' | 'predictProba' | 'decisionFunction' | 'transform'
  readonly groups?: ArrayLike<number>
}

/** The paths {@link crossValPredictAlgorithm} logs. */
export type CrossValPredictPath = 'cv/fold' | 'cv/testSize' | 'cv/predicted'

/** What a stepped out-of-fold prediction logs, by path. */
export const crossValPredictContract: Contract<unknown, CrossValPredictPath> = defineContract({
  'cv/fold': scalar({ description: 'the fold just fitted and predicted (from 0)' }),
  'cv/testSize': scalar({ description: 'rows predicted by the fold' }),
  'cv/predicted': scalar({ description: 'rows with an out-of-fold prediction so far' }),
})

/**
 * The state of {@link crossValPredictAlgorithm}: the folds done, the width of the output (0 for one value per row, -1
 * before the first fold) and the predictions so far, row-major in the original row order (zero where not yet predicted).
 */
export interface CvPredictState {
  readonly done: number
  readonly width: number
  readonly predictions: readonly number[]
}

function crossValPredictRun(
  estimator: Estimator,
  data: LearnData,
  options: CrossValPredictOptions,
): { readonly algorithm: Algorithm<CvPredictState, CrossValPredictPath>; readonly folds: Fold[]; readonly n: number } {
  const splitter = checkCv(options.cv, estimator)
  if (!splitter.partition)
    throw new MdmlError('value', `crossValPredict: ${splitter.name} does not partition the samples`)
  const n = rowsOf(data.x)
  const folds = splitIndices(
    splitter,
    splitInputOf(data, options.groups),
    foldIn(keyFor(options, 'crossValPredict'), 0),
  )
  const method = options.method ?? 'predict'
  const classes =
    method === 'predictProba' && data.y !== undefined ? encodeLabels(data.y, 'crossValPredict', n).classes : undefined
  const algorithm: Algorithm<CvPredictState, CrossValPredictPath> = {
    name: 'cross-val-predict',
    config: { estimator: estimator.name, splitter: splitter.name, method },
    init: () => ({ done: 0, width: -1, predictions: [] }),
    step: (s, ctx) => {
      const fold = folds[s.done] as Fold
      const fitted = estimator.fit(takeData(data, fold.train), innerFitOptions(options, foldIn(ctx.key, 0), ctx.run))
      const pred = requireVerb(fitted, method, 'crossValPredict')(takeData(data, fold.test).x)
      const v = toFlat(pred)
      const k = pred.shape.length === 2 ? (pred.shape[1] ?? 1) : 1
      const cols = classes !== undefined ? classes.length : k
      const width = s.width < 0 ? (pred.shape.length === 2 || classes !== undefined ? cols : 0) : s.width
      const out = s.width < 0 ? new Array<number>(n * Math.max(1, width)).fill(0) : [...s.predictions]
      const map = classes !== undefined ? (fitted.classes ?? classes).map((c) => classes.indexOf(c)) : undefined
      for (const [r, i] of fold.test.entries())
        for (let c = 0; c < k; c++) out[i * Math.max(1, width) + (map?.[c] ?? c)] = v[r * k + c] ?? 0
      ctx.log('cv/fold', s.done)
      ctx.log('cv/testSize', fold.test.length)
      ctx.log(
        'cv/predicted',
        folds.slice(0, s.done + 1).reduce((a, f) => a + f.test.length, 0),
      )
      return { done: s.done + 1, width, predictions: out }
    },
    done: (s) => (s.done >= folds.length ? 'completed' : undefined),
    serialise: (s) => stateToJson({ ...s, predictions: scoresToJson(s.predictions) }),
    deserialise: (j) => {
      const o = stateFromJson(j) as { readonly done: number; readonly width: number; readonly predictions: never }
      return { ...o, predictions: scoresFromJson(o.predictions) }
    },
  }
  return { algorithm, folds, n }
}

/**
 * Out-of-fold predictions as a stepped algorithm: each step fits the estimator on one fold's training rows and fills in
 * the predictions of that fold's test rows. The state serialises. Needs a partitioning splitter.
 *
 * @param estimator - The estimator.
 * @param data - The data.
 * @param options - cv, method, groups and key.
 * @returns The algorithm, one step per fold.
 * @throws MdmlError with code `value` when the splitter does not partition the samples.
 * @example
 * ```ts
 * import { crossValPredictAlgorithm } from 'mdml-engine/learn/model-selection'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { full, item, mean } from 'mdml-engine/tensor'
 * const meanModel = defineEstimator(
 *   { name: 'mean', kind: 'regressor', fit: (d) => ({ m: item(mean(d.y)) }), methods: (p) => ({ predict: (x) => full([x.shape[0]], p.m) }) },
 *   {},
 * )
 * const alg = crossValPredictAlgorithm(meanModel, { x: [[0], [1], [2], [3]], y: [0, 2, 4, 6] }, { cv: 2 })
 * runAlgorithm(alg, { maxSteps: 5 }).state.predictions // => [5, 5, 1, 1]
 * ```
 */
export function crossValPredictAlgorithm(
  estimator: Estimator,
  data: LearnData,
  options: CrossValPredictOptions = {},
): Algorithm<CvPredictState, CrossValPredictPath> {
  return crossValPredictRun(estimator, data, options).algorithm
}

/**
 * Out-of-fold predictions (sklearn.model_selection.cross_val_predict): every sample predicted by the model fitted
 * without its fold. Needs a partitioning splitter. Probability columns are aligned to all the classes in `y`, a class
 * missing from a training fold getting probability 0. This is {@link crossValPredictAlgorithm} run to the end.
 *
 * @param estimator - The estimator.
 * @param data - The data.
 * @param options - cv, method, groups, key and tracing.
 * @returns `[n]` or `[n, k]` predictions in the original row order.
 * @throws MdmlError with code `value` when the splitter does not partition the samples.
 * @example
 * ```ts
 * import { crossValPredict } from 'mdml-engine/learn/model-selection'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { full, item, mean, toFlat } from 'mdml-engine/tensor'
 * const meanModel = defineEstimator(
 *   { name: 'mean', kind: 'regressor', fit: (d) => ({ m: item(mean(d.y)) }), methods: (p) => ({ predict: (x) => full([x.shape[0]], p.m) }) },
 *   {},
 * )
 * toFlat(crossValPredict(meanModel, { x: [[0], [1], [2], [3]], y: [0, 2, 4, 6] }, { cv: 2 })) // => [5, 5, 1, 1]
 * ```
 */
export function crossValPredict(estimator: Estimator, data: LearnData, options: CrossValPredictOptions = {}): Tensor {
  const { algorithm, folds, n } = crossValPredictRun(estimator, data, options)
  const { state } = runAlgorithm(algorithm, { ...runOptionsFor(options, 'crossValPredict'), maxSteps: folds.length })
  const out = Float64Array.from(state.predictions)
  return state.width > 0 ? fromStorage(out, [n, state.width], 'float64') : fromStorage(out, [n], 'float64')
}

/** Options of {@link learningCurve}. */
export interface LearningCurveOptions extends FitOptions {
  /** Training sizes: fractions in (0, 1] of the largest training set, or counts (default 0.1 to 1 in 5 steps). */
  readonly trainSizes?: readonly number[]
  readonly cv?: CvOption
  readonly scoring?: Scoring
  readonly groups?: ArrayLike<number>
  /** Shuffle each training fold before taking prefixes (default false). */
  readonly shuffle?: boolean
}

/** Scores by setting and fold. */
export interface CurveResult {
  /** The settings (training sizes or parameter values). */
  readonly values: readonly number[]
  /** trainScores[setting][fold]. */
  readonly trainScores: readonly (readonly number[])[]
  /** testScores[setting][fold]. */
  readonly testScores: readonly (readonly number[])[]
}

/** The paths {@link learningCurveAlgorithm} logs. */
export type LearningCurvePath = 'curve/fold' | 'curve/size' | 'curve/trainScore' | 'curve/testScore'

/** What a stepped learning curve logs, by path. */
export const learningCurveContract: Contract<unknown, LearningCurvePath> = defineContract({
  'curve/fold': scalar({ description: 'the fold of the job just done (from 0)' }),
  'curve/size': scalar({ description: 'the number of training rows of the job just done' }),
  'curve/trainScore': scalar({ description: 'the score on the training rows used', direction: 'higher' }),
  'curve/testScore': scalar({ description: 'the score on the fold’s test rows', direction: 'higher' }),
})

/**
 * The state of {@link learningCurveAlgorithm}: jobs (a fold at a training size, fold-major) done and their scores in
 * job order.
 */
export interface LearningCurveState {
  readonly done: number
  readonly train: readonly number[]
  readonly test: readonly number[]
}

function learningCurveRun(
  estimator: Estimator,
  data: LearnData,
  options: LearningCurveOptions,
): {
  readonly algorithm: Algorithm<LearningCurveState, LearningCurvePath>
  readonly sizes: number[]
  readonly folds: Fold[]
  readonly jobs: number
  readonly scoreName: string
} {
  const splitter = checkCv(options.cv, estimator)
  const key = keyFor(options, 'learningCurve')
  const folds = splitIndices(splitter, splitInputOf(data, options.groups), foldIn(key, 0))
  const scorer = Object.values(scorersOf(options.scoring))[0] as Scorer
  const nMax = folds[0]?.train.length ?? 0
  const requested = options.trainSizes ?? [0.1, 0.325, 0.55, 0.775, 1]
  const fractions = requested.every((s) => s > 0 && s <= 1)
  const sizes = Array.from(
    new Set(requested.map((s) => Math.min(Math.max(fractions ? Math.trunc(s * nMax) : s, 1), nMax))),
  ).sort((a, b) => a - b)
  if (!fractions && requested.some((s) => !Number.isInteger(s) || s < 1 || s > nMax)) {
    throw new MdmlError('value', `learningCurve: train sizes must be fractions in (0, 1] or counts in [1, ${nMax}]`)
  }
  const jobs = folds.flatMap((fold, f) => sizes.map((size) => ({ fold, f, size })))
  const algorithm: Algorithm<LearningCurveState, LearningCurvePath> = {
    name: 'learning-curve',
    config: { estimator: estimator.name, sizes, folds: folds.length },
    schema: channels({
      metrics: {
        train: { description: 'the training score' },
        test: { description: 'the test score' },
        size: { description: 'training rows' },
      },
    }),
    init: () => ({ done: 0, train: [], test: [] }),
    step: (s, ctx) => {
      const { fold, f, size } = jobs[s.done] as (typeof jobs)[number]
      const order =
        options.shuffle === true
          ? toFlat(permutation(foldIn(key, 1 + f), fold.train.length)).map((r) => fold.train[r] ?? 0)
          : Array.from(fold.train)
      const train = takeData(data, order.slice(0, size))
      const fitted = estimator.fit(train, innerFitOptions(options, foldIn(ctx.key, 0), ctx.run))
      const trainScore = scorer.score(fitted, train)
      const testScore = scorer.score(fitted, takeData(data, fold.test))
      ctx.log('curve/fold', f)
      ctx.log('curve/size', size)
      ctx.log('curve/trainScore', trainScore)
      ctx.log('curve/testScore', testScore)
      return { done: s.done + 1, train: [...s.train, trainScore], test: [...s.test, testScore] }
    },
    done: (s) => (s.done >= jobs.length ? 'completed' : undefined),
    observe: (s) => ({
      metrics:
        s.done === 0
          ? {}
          : { train: s.train[s.done - 1] ?? 0, test: s.test[s.done - 1] ?? 0, size: jobs[s.done - 1]?.size ?? 0 },
    }),
    serialise: (s) => stateToJson({ done: s.done, train: scoresToJson(s.train), test: scoresToJson(s.test) }),
    deserialise: (j) => {
      const o = stateFromJson(j) as { readonly done: number; readonly train: never; readonly test: never }
      return { done: o.done, train: scoresFromJson(o.train), test: scoresFromJson(o.test) }
    },
  }
  return {
    algorithm,
    sizes,
    folds,
    jobs: jobs.length,
    scoreName: Object.keys(scorersOf(options.scoring))[0] ?? 'score',
  }
}

/**
 * A learning curve as a stepped algorithm: each step fits the estimator on the first `size` training rows of one fold
 * (shuffled first when asked) and scores it on those rows and on the fold's test rows. Jobs go fold by fold, the sizes
 * in increasing order within each. The state serialises.
 *
 * @param estimator - The estimator.
 * @param data - The data.
 * @param options - trainSizes, cv, scoring, groups, shuffle and key.
 * @returns The algorithm, one step per fold and size.
 * @throws MdmlError with code `value` for sizes that are neither fractions in (0, 1] nor counts in range.
 * @example
 * ```ts
 * import { learningCurveAlgorithm } from 'mdml-engine/learn/model-selection'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { full } from 'mdml-engine/tensor'
 * const zero = defineEstimator({ name: 'zero', kind: 'regressor', fit: () => ({}), methods: () => ({ predict: (x) => full([x.shape[0]], 0) }) }, {})
 * const data = { x: [[0], [1], [2], [3], [4], [5]], y: [0, 0, 0, 0, 0, 0] }
 * runAlgorithm(learningCurveAlgorithm(zero, data, { cv: 3, trainSizes: [0.5, 1] }), { maxSteps: 10 }).state.done // => 6
 * ```
 */
export function learningCurveAlgorithm(
  estimator: Estimator,
  data: LearnData,
  options: LearningCurveOptions = {},
): Algorithm<LearningCurveState, LearningCurvePath> {
  return learningCurveRun(estimator, data, options).algorithm
}

/**
 * A learning curve (sklearn.model_selection.learning_curve): for each fold and training size, fit on the first n
 * training rows of the fold (shuffled first when asked) and score on them and on the test rows. This is
 * {@link learningCurveAlgorithm} run to the end.
 *
 * @param estimator - The estimator.
 * @param data - The data.
 * @param options - trainSizes, cv, scoring, groups, shuffle, key and tracing.
 * @returns The absolute training sizes and the scores.
 * @example
 * ```ts
 * import { learningCurve } from 'mdml-engine/learn/model-selection'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { full } from 'mdml-engine/tensor'
 * const zero = defineEstimator({ name: 'zero', kind: 'regressor', fit: () => ({}), methods: () => ({ predict: (x) => full([x.shape[0]], 0) }) }, {})
 * const x = [[0], [1], [2], [3], [4], [5]]
 * learningCurve(zero, { x, y: [0, 0, 0, 0, 0, 0] }, { cv: 3, trainSizes: [0.5, 1], scoring: 'neg_mean_squared_error' }).values // => [2, 4]
 * ```
 */
export function learningCurve(
  estimator: Estimator,
  data: LearnData,
  options: LearningCurveOptions = {},
): Explained<CurveResult> {
  const { algorithm, sizes, folds, jobs, scoreName } = learningCurveRun(estimator, data, options)
  const { state } = runAlgorithm(algorithm, { ...runOptionsFor(options, 'learningCurve'), maxSteps: jobs })
  const grid = (pick: readonly number[]): number[][] =>
    sizes.map((_, si) => folds.map((_, f) => pick[f * sizes.length + si] ?? Number.NaN))
  const trainScores = grid(state.train)
  const testScores = grid(state.test)
  const sd = (v: readonly number[]): number => Math.sqrt(varianceOf(v))
  return {
    values: sizes,
    trainScores,
    testScores,
    // The mean score over folds at each size, with the folds' standard deviation as the spread.
    exposition: [
      learningCurveRecord({
        sizes,
        train: trainScores.map((f) => meanOf(f)),
        validation: testScores.map((f) => meanOf(f)),
        trainSpread: trainScores.map(sd),
        validationSpread: testScores.map(sd),
        resource: 'training rows',
        scoreName,
      }),
    ],
  }
}

/** Options of {@link validationCurve}. */
export interface ValidationCurveOptions extends FitOptions {
  readonly cv?: CvOption
  readonly scoring?: Scoring
  readonly groups?: ArrayLike<number>
}

/**
 * A validation curve (sklearn.model_selection.validation_curve): cross-validated training and test scores as one
 * hyperparameter varies.
 *
 * @param estimator - The estimator.
 * @param data - The data.
 * @param param - The hyperparameter (`step.param` for composites).
 * @param range - Its values.
 * @param options - cv, scoring, groups, key and tracing.
 * @returns The values and the scores by value and fold.
 * @example
 * ```ts
 * import { validationCurve } from 'mdml-engine/learn/model-selection'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { full } from 'mdml-engine/tensor'
 * const constant = defineEstimator({ name: 'c', kind: 'regressor', fit: () => ({}), methods: (_, h) => ({ predict: (x) => full([x.shape[0]], h.c) }) }, { c: 0 })
 * validationCurve(constant, { x: [[0], [1]], y: [1, 1] }, 'c', [0, 1], { cv: 2, scoring: 'neg_mean_absolute_error' }).testScores // => [[-1, -1], [-0, -0]]
 * ```
 */
export function validationCurve(
  estimator: Estimator,
  data: LearnData,
  param: string,
  range: readonly unknown[],
  options: ValidationCurveOptions = {},
): CurveResult & { readonly params: readonly unknown[] } {
  const results = range.map((v) =>
    crossValidate(estimator.with({ [param]: v }), data, {
      ...options,
      key: foldIn(keyFor(options, 'validationCurve'), 0),
      returnTrainScore: true,
    }),
  )
  const first = (r: CrossValidateResult, which: 'test' | 'train'): readonly number[] =>
    Object.values((which === 'test' ? r.testScores : r.trainScores) ?? {})[0] ?? []
  return {
    params: range,
    values: range.map((v) => (typeof v === 'number' ? v : Number.NaN)),
    trainScores: results.map((r) => first(r, 'train')),
    testScores: results.map((r) => first(r, 'test')),
  }
}

/** Options of {@link permutationTestScore}. */
export interface PermutationTestOptions extends FitOptions {
  readonly cv?: CvOption
  readonly scoring?: Scoring
  /** Permutations of the targets (default 100). */
  readonly nPermutations?: number
  /** Permute only within groups (also used by group splitters). */
  readonly groups?: ArrayLike<number>
}

/** The result of {@link permutationTestScore}. */
export interface PermutationTestResult {
  /** The mean cross-validated score on the real targets. */
  readonly score: number
  /** The mean scores on permuted targets. */
  readonly permutationScores: readonly number[]
  /** (count(permuted >= score) + 1) / (nPermutations + 1). */
  readonly pValue: number
}

/** The paths {@link permutationTestAlgorithm} logs. */
export type PermutationTestPath = 'perm/index' | 'perm/score' | 'perm/realScore' | 'perm/pValue'

/** What a stepped permutation test logs, by path. */
export const permutationTestContract: Contract<unknown, PermutationTestPath> = defineContract({
  'perm/index': scalar({ description: 'the permutation just scored (from 0; -1 for the real targets)' }),
  'perm/score': scalar({ description: 'the mean cross-validated score on the permuted targets' }),
  'perm/realScore': scalar({ description: 'the mean cross-validated score on the real targets', direction: 'higher' }),
  'perm/pValue': scalar({
    description: '(count(permuted >= real) + 1) / (permutations so far + 1)',
    direction: 'lower',
  }),
})

/**
 * The state of {@link permutationTestAlgorithm}: the score on the real targets (`null` before the first step) and the
 * scores on the permutations so far.
 */
export interface PermutationTestState {
  readonly real: number | null
  readonly scores: readonly number[]
}

function permutationTestRun(
  estimator: Estimator,
  data: LearnData,
  options: PermutationTestOptions,
): { readonly algorithm: Algorithm<PermutationTestState, PermutationTestPath>; readonly nPerm: number } {
  const nPerm = options.nPermutations ?? 100
  if (!(Number.isInteger(nPerm) && nPerm >= 1))
    throw new MdmlError('value', 'permutationTestScore: nPermutations must be >= 1')
  const key = keyFor(options, 'permutationTestScore')
  const n = rowsOf(data.x)
  const y = targetVector(data.y ?? [], 'permutationTestScore', n)
  const cvOptions = {
    ...options,
    key: foldIn(key, 0),
    ...(options.groups === undefined ? {} : { groups: options.groups }),
  }
  const meanScore = (d: LearnData, parent?: RunHandle): number =>
    meanOf(crossValScore(estimator, d, { ...cvOptions, ...(parent === undefined ? {} : { parent }) }))
  const permuteWithinGroups = (k: Key): Float64Array => {
    const out = Float64Array.from(y)
    const groups = options.groups
    const blocks =
      groups === undefined
        ? [Array.from({ length: n }, (_, i) => i)]
        : Array.from(new Set(Array.from(groups))).map((g) =>
            Array.from({ length: n }, (_, i) => i).filter((i) => groups[i] === g),
          )
    for (const [b, rows] of blocks.entries()) {
      const perm = toFlat(permutation(foldIn(k, b), rows.length))
      for (const [r, i] of rows.entries()) out[i] = y[rows[perm[r] ?? 0] ?? 0] ?? 0
    }
    return out
  }
  const algorithm: Algorithm<PermutationTestState, PermutationTestPath> = {
    name: 'permutation-test-score',
    config: { estimator: estimator.name, nPermutations: nPerm },
    schema: channels({ metrics: { score: { description: 'the mean CV score on permuted targets' } } }),
    init: () => ({ real: null, scores: [] }),
    step: (s, ctx) => {
      if (s.real === null) {
        const real = meanScore(data, ctx.run)
        ctx.log('perm/index', -1)
        ctx.log('perm/realScore', real)
        return { real, scores: [] }
      }
      const score = meanScore({ ...data, y: permuteWithinGroups(foldIn(key, 1 + s.scores.length)) }, ctx.run)
      const scores = [...s.scores, score]
      ctx.log('perm/index', s.scores.length)
      ctx.log('perm/score', score)
      ctx.log('perm/realScore', s.real)
      ctx.log('perm/pValue', (scores.filter((v) => v >= (s.real ?? 0)).length + 1) / (scores.length + 1))
      return { real: s.real, scores }
    },
    done: (s) => (s.real !== null && s.scores.length >= nPerm ? 'completed' : undefined),
    observe: (s) => ({ metrics: s.scores.length === 0 ? {} : { score: s.scores[s.scores.length - 1] ?? 0 } }),
    serialise: (s) =>
      stateToJson({ real: s.real === null ? null : scoresToJson([s.real]), scores: scoresToJson(s.scores) }),
    deserialise: (j) => {
      const o = stateFromJson(j) as { readonly real: never | null; readonly scores: never }
      return {
        real: o.real === null ? null : (scoresFromJson(o.real)[0] ?? Number.NaN),
        scores: scoresFromJson(o.scores),
      }
    },
  }
  return { algorithm, nPerm }
}

/**
 * A permutation test of a cross-validated score as a stepped algorithm: the first step scores the real targets, and
 * each later step scores one random permutation of the targets (within groups when given), logging the running
 * p-value. The state serialises.
 *
 * @param estimator - The estimator.
 * @param data - The data.
 * @param options - cv, scoring, nPermutations, groups and key.
 * @returns The algorithm, one step for the real score and one per permutation.
 * @throws MdmlError with code `value` for fewer than one permutation.
 * @example
 * ```ts
 * import { permutationTestAlgorithm } from 'mdml-engine/learn/model-selection'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { full } from 'mdml-engine/tensor'
 * const zero = defineEstimator({ name: 'zero', kind: 'regressor', fit: () => ({}), methods: () => ({ predict: (x) => full([x.shape[0]], 0) }) }, {})
 * const alg = permutationTestAlgorithm(zero, { x: [[0], [1], [2], [3]], y: [1, 2, 3, 4] }, { cv: 2, nPermutations: 5 })
 * runAlgorithm(alg, { maxSteps: 10 }).state.scores.length // => 5
 * ```
 */
export function permutationTestAlgorithm(
  estimator: Estimator,
  data: LearnData,
  options: PermutationTestOptions = {},
): Algorithm<PermutationTestState, PermutationTestPath> {
  return permutationTestRun(estimator, data, options).algorithm
}

/**
 * A permutation test of a cross-validated score (sklearn.model_selection.permutation_test_score; Ojala and Garriga
 * 2010): how often a model on randomly permuted targets (within groups) scores at least as well. This is
 * {@link permutationTestAlgorithm} run to the end, a traced algorithm with one step per permutation.
 *
 * @param estimator - The estimator.
 * @param data - The data.
 * @param options - cv, scoring, nPermutations, groups, key and tracing.
 * @returns The score, the permutation scores and the p-value.
 * @example
 * ```ts
 * import { permutationTestScore } from 'mdml-engine/learn/model-selection'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { full } from 'mdml-engine/tensor'
 * const zero = defineEstimator({ name: 'zero', kind: 'regressor', fit: () => ({}), methods: () => ({ predict: (x) => full([x.shape[0]], 0) }) }, {})
 * const r = permutationTestScore(zero, { x: [[0], [1], [2], [3]], y: [1, 2, 3, 4] }, { cv: 2, nPermutations: 9, scoring: 'neg_mean_absolute_error' })
 * r.pValue // => 1
 * ```
 */
export function permutationTestScore(
  estimator: Estimator,
  data: LearnData,
  options: PermutationTestOptions = {},
): PermutationTestResult {
  const { algorithm, nPerm } = permutationTestRun(estimator, data, options)
  const { state } = runAlgorithm(algorithm, { ...runOptionsFor(options, 'permutationTestScore'), maxSteps: nPerm + 1 })
  const score = state.real ?? Number.NaN
  return {
    score,
    permutationScores: [...state.scores],
    pValue: (state.scores.filter((s) => s >= score).length + 1) / (nPerm + 1),
  }
}
