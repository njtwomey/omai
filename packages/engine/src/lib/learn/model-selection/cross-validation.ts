import { meanOf, varianceOf, WhiteboxError } from '../../base/index.ts'
import { foldIn, type Key, permutation } from '../../random/index.ts'
import { fromStorage, type Tensor, toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunHandle, runAlgorithm, subRun } from '../../trace/index.ts'
import { innerFitOptions, keyFor, runOptionsFor } from '../_shared.ts'
import { type Estimator, type FitOptions, type Fitted, type LearnData, requireVerb } from '../estimator.ts'
import { type Scorer, type Scoring, scorersOf } from '../scoring.ts'
import { encodeLabels, rowsOf, takeData, targetVector } from '../validation.ts'
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
 * import { checkCv } from 'whitebox-engine/learn/model-selection'
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
 * import { splitInputOf } from 'whitebox-engine/learn/model-selection'
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
  if (r.state === undefined) throw new WhiteboxError('convergence', 'crossValidate: a fold did not run')
  return r.state
}

interface CvState {
  readonly outcomes: readonly FoldOutcome[]
}

/**
 * Cross-validate an estimator (sklearn.model_selection.cross_validate): for each fold, fit on the training rows and
 * score on the test rows (and optionally the training rows). A traced algorithm with one step per fold, each fold a
 * child run under which the estimator's own fit nests.
 *
 * @param estimator - The estimator.
 * @param data - The data (features, targets, weights).
 * @param options - cv, scoring, returnTrainScore, returnEstimator, groups, errorScore, key and tracing.
 * @returns Scores per scorer and fold, their means and deviations, the folds and optionally the fitted estimators.
 * @concept cross-validation
 * @example
 * ```ts
 * import { crossValidate } from 'whitebox-engine/learn/model-selection'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { full, mean, item } from 'whitebox-engine/tensor'
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
  const splitter = checkCv(options.cv, estimator)
  const folds = splitIndices(splitter, splitInputOf(data, options.groups), foldIn(keyFor(options, 'crossValidate'), 0))
  const scorers = scorersOf(options.scoring)
  const algorithm: Algorithm<CvState> = {
    name: 'cross-validate',
    config: { estimator: estimator.name, splitter: splitter.name, folds: folds.length, scoring: Object.keys(scorers) },
    schema: channels({
      metrics: Object.fromEntries(
        Object.keys(scorers).map((n) => [
          n,
          { description: `the test ${n} of the fold`, direction: 'higher' as const },
        ]),
      ),
      messages: {
        fold: {
          description: 'a fold was fitted and scored',
          payload: { fold: 'number', train: 'number', test: 'number' },
        },
      },
    }),
    init: () => ({ outcomes: [] }),
    step: (s, ctx) => {
      const fold = folds[s.outcomes.length] as Fold
      return { outcomes: [...s.outcomes, runFold(ctx, s.outcomes.length, estimator, data, fold, scorers, options)] }
    },
    done: (s) => (s.outcomes.length >= folds.length ? 'completed' : undefined),
    observe: (s) => {
      const last = s.outcomes[s.outcomes.length - 1]
      const f = folds[s.outcomes.length - 1]
      return {
        metrics: last?.test ?? {},
        messages:
          f === undefined
            ? []
            : [{ kind: 'fold', payload: { fold: s.outcomes.length - 1, train: f.train.length, test: f.test.length } }],
      }
    },
  }
  const { state } = runAlgorithm(algorithm, { ...runOptionsFor(options, 'crossValidate'), maxSteps: folds.length })
  const names = Object.keys(scorers)
  const collect = (pick: (o: FoldOutcome) => Readonly<Record<string, number>> | undefined): Record<string, number[]> =>
    Object.fromEntries(names.map((n) => [n, state.outcomes.map((o) => pick(o)?.[n] ?? Number.NaN)]))
  const testScores = collect((o) => o.test)
  return {
    testScores,
    ...(options.returnTrainScore === true ? { trainScores: collect((o) => o.train) } : {}),
    ...(options.returnEstimator === true
      ? { estimators: state.outcomes.flatMap((o) => (o.fitted === undefined ? [] : [o.fitted])) }
      : {}),
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
 * import { crossValScore } from 'whitebox-engine/learn/model-selection'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { full } from 'whitebox-engine/tensor'
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

/**
 * Out-of-fold predictions (sklearn.model_selection.cross_val_predict): every sample predicted by the model fitted
 * without its fold. Needs a partitioning splitter. Probability columns are aligned to all the classes in `y`, a class
 * missing from a training fold getting probability 0.
 *
 * @param estimator - The estimator.
 * @param data - The data.
 * @param options - cv, method, groups, key and tracing.
 * @returns `[n]` or `[n, k]` predictions in the original row order.
 * @throws WhiteboxError with code `value` when the splitter does not partition the samples.
 * @example
 * ```ts
 * import { crossValPredict } from 'whitebox-engine/learn/model-selection'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { full, item, mean, toFlat } from 'whitebox-engine/tensor'
 * const meanModel = defineEstimator(
 *   { name: 'mean', kind: 'regressor', fit: (d) => ({ m: item(mean(d.y)) }), methods: (p) => ({ predict: (x) => full([x.shape[0]], p.m) }) },
 *   {},
 * )
 * toFlat(crossValPredict(meanModel, { x: [[0], [1], [2], [3]], y: [0, 2, 4, 6] }, { cv: 2 })) // => [5, 5, 1, 1]
 * ```
 */
export function crossValPredict(estimator: Estimator, data: LearnData, options: CrossValPredictOptions = {}): Tensor {
  const splitter = checkCv(options.cv, estimator)
  if (!splitter.partition)
    throw new WhiteboxError('value', `crossValPredict: ${splitter.name} does not partition the samples`)
  const n = rowsOf(data.x)
  const folds = splitIndices(
    splitter,
    splitInputOf(data, options.groups),
    foldIn(keyFor(options, 'crossValPredict'), 0),
  )
  const method = options.method ?? 'predict'
  const classes =
    method === 'predictProba' && data.y !== undefined ? encodeLabels(data.y, 'crossValPredict', n).classes : undefined
  let width = -1
  let out = new Float64Array(0)
  const algorithm: Algorithm<number> = {
    name: 'cross-val-predict',
    config: { estimator: estimator.name, splitter: splitter.name, method },
    init: () => 0,
    step: (f, ctx) => {
      const fold = folds[f] as Fold
      const fitted = estimator.fit(takeData(data, fold.train), innerFitOptions(options, foldIn(ctx.key, 0), ctx.run))
      const pred = requireVerb(fitted, method, 'crossValPredict')(takeData(data, fold.test).x)
      const v = toFlat(pred)
      const k = pred.shape.length === 2 ? (pred.shape[1] ?? 1) : 1
      const cols = classes !== undefined ? classes.length : k
      if (width < 0) {
        width = pred.shape.length === 2 || classes !== undefined ? cols : 0
        out = new Float64Array(n * Math.max(1, width))
      }
      const map = classes !== undefined ? (fitted.classes ?? classes).map((c) => classes.indexOf(c)) : undefined
      fold.test.forEach((i, r) => {
        for (let c = 0; c < k; c++) out[i * Math.max(1, width) + (map?.[c] ?? c)] = v[r * k + c] ?? 0
      })
      return f + 1
    },
    done: (f) => (f >= folds.length ? 'completed' : undefined),
  }
  runAlgorithm(algorithm, { ...runOptionsFor(options, 'crossValPredict'), maxSteps: folds.length })
  return width > 0 ? fromStorage(out, [n, width], 'float64') : fromStorage(out, [n], 'float64')
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

/**
 * A learning curve (sklearn.model_selection.learning_curve): for each fold and training size, fit on the first n
 * training rows of the fold (shuffled first when asked) and score on them and on the test rows.
 *
 * @param estimator - The estimator.
 * @param data - The data.
 * @param options - trainSizes, cv, scoring, groups, shuffle, key and tracing.
 * @returns The absolute training sizes and the scores.
 * @example
 * ```ts
 * import { learningCurve } from 'whitebox-engine/learn/model-selection'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { full } from 'whitebox-engine/tensor'
 * const zero = defineEstimator({ name: 'zero', kind: 'regressor', fit: () => ({}), methods: () => ({ predict: (x) => full([x.shape[0]], 0) }) }, {})
 * const x = [[0], [1], [2], [3], [4], [5]]
 * learningCurve(zero, { x, y: [0, 0, 0, 0, 0, 0] }, { cv: 3, trainSizes: [0.5, 1], scoring: 'neg_mean_squared_error' }).values // => [2, 4]
 * ```
 */
export function learningCurve(estimator: Estimator, data: LearnData, options: LearningCurveOptions = {}): CurveResult {
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
    throw new WhiteboxError('value', `learningCurve: train sizes must be fractions in (0, 1] or counts in [1, ${nMax}]`)
  }
  const jobs = folds.flatMap((fold, f) => sizes.map((size) => ({ fold, f, size })))
  const scores: { train: number; test: number }[] = []
  const algorithm: Algorithm<number> = {
    name: 'learning-curve',
    config: { estimator: estimator.name, sizes, folds: folds.length },
    schema: channels({
      metrics: {
        train: { description: 'the training score' },
        test: { description: 'the test score' },
        size: { description: 'training rows' },
      },
    }),
    init: () => 0,
    step: (j, ctx) => {
      const { fold, f, size } = jobs[j] as (typeof jobs)[number]
      const order =
        options.shuffle === true
          ? toFlat(permutation(foldIn(key, 1 + f), fold.train.length)).map((r) => fold.train[r] ?? 0)
          : Array.from(fold.train)
      const train = takeData(data, order.slice(0, size))
      const fitted = estimator.fit(train, innerFitOptions(options, foldIn(ctx.key, 0), ctx.run))
      scores.push({ train: scorer.score(fitted, train), test: scorer.score(fitted, takeData(data, fold.test)) })
      return j + 1
    },
    done: (j) => (j >= jobs.length ? 'completed' : undefined),
    observe: (j) => {
      const s = scores[j - 1]
      return { metrics: s === undefined ? {} : { train: s.train, test: s.test, size: jobs[j - 1]?.size ?? 0 } }
    },
  }
  runAlgorithm(algorithm, { ...runOptionsFor(options, 'learningCurve'), maxSteps: jobs.length })
  const grid = (pick: (s: { train: number; test: number }) => number): number[][] =>
    sizes.map((_, si) => folds.map((_, f) => pick(scores[f * sizes.length + si] as { train: number; test: number })))
  return { values: sizes, trainScores: grid((s) => s.train), testScores: grid((s) => s.test) }
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
 * import { validationCurve } from 'whitebox-engine/learn/model-selection'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { full } from 'whitebox-engine/tensor'
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

/**
 * A permutation test of a cross-validated score (sklearn.model_selection.permutation_test_score; Ojala and Garriga
 * 2010): how often a model on randomly permuted targets (within groups) scores at least as well. A traced algorithm
 * with one step per permutation.
 *
 * @param estimator - The estimator.
 * @param data - The data.
 * @param options - cv, scoring, nPermutations, groups, key and tracing.
 * @returns The score, the permutation scores and the p-value.
 * @example
 * ```ts
 * import { permutationTestScore } from 'whitebox-engine/learn/model-selection'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { full } from 'whitebox-engine/tensor'
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
  const nPerm = options.nPermutations ?? 100
  if (!(Number.isInteger(nPerm) && nPerm >= 1))
    throw new WhiteboxError('value', 'permutationTestScore: nPermutations must be >= 1')
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
  const score = meanScore(data)
  const permuteWithinGroups = (k: Key): Float64Array => {
    const out = Float64Array.from(y)
    const groups = options.groups
    const blocks =
      groups === undefined
        ? [Array.from({ length: n }, (_, i) => i)]
        : Array.from(new Set(Array.from(groups))).map((g) =>
            Array.from({ length: n }, (_, i) => i).filter((i) => groups[i] === g),
          )
    blocks.forEach((rows, b) => {
      const perm = toFlat(permutation(foldIn(k, b), rows.length))
      rows.forEach((i, r) => {
        out[i] = y[rows[perm[r] ?? 0] ?? 0] ?? 0
      })
    })
    return out
  }
  const algorithm: Algorithm<readonly number[]> = {
    name: 'permutation-test-score',
    config: { estimator: estimator.name, nPermutations: nPerm },
    schema: channels({ metrics: { score: { description: 'the mean CV score on permuted targets' } } }),
    init: () => [],
    step: (s, ctx) => [...s, meanScore({ ...data, y: permuteWithinGroups(foldIn(key, 1 + s.length)) }, ctx.run)],
    done: (s) => (s.length >= nPerm ? 'completed' : undefined),
    observe: (s) => ({ metrics: s.length === 0 ? {} : { score: s[s.length - 1] ?? 0 } }),
  }
  const { state } = runAlgorithm(algorithm, { ...runOptionsFor(options, 'permutationTestScore'), maxSteps: nPerm })
  return { score, permutationScores: state, pValue: (state.filter((s) => s >= score).length + 1) / (nPerm + 1) }
}
