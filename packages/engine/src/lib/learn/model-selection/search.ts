import { meanOf, varianceOf, WhiteboxError } from '../../base/index.ts'
import { foldIn, permutation } from '../../random/index.ts'
import { type Space, type SpaceValue, sample } from '../../space/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import { innerFitOptions, keyFor, runOptionsFor } from '../_shared.ts'
import type { Estimator, FitOptions, Fitted, LearnData } from '../estimator.ts'
import { type Scoring, scorersOf } from '../scoring.ts'
import { takeData } from '../validation.ts'
import { type CvOption, checkCv, crossValidate, splitInputOf } from './cross-validation.ts'
import { type Fold, type Splitter, splitIndices } from './splitters.ts'

/** Hyperparameter settings: each a record of values by name (`step.param` for composites). */
export type Candidate = Readonly<Record<string, unknown>>

/** A grid: each name with the values to try (or several grids, searched one after another). */
export type ParamGrid =
  | Readonly<Record<string, readonly unknown[]>>
  | readonly Readonly<Record<string, readonly unknown[]>>[]

/** The most candidates a grid may expand to. */
export const MAX_CANDIDATES = 100000

/**
 * Every combination of a grid, in sklearn's order (ParameterGrid): names sorted, the last name varying fastest; with
 * several grids, one after another.
 *
 * @param grid - A grid or a list of grids.
 * @returns The candidates.
 * @throws WhiteboxError with code `value` beyond {@link MAX_CANDIDATES} candidates.
 * @example
 * ```ts
 * import { parameterGrid } from 'whitebox-engine/learn/model-selection'
 * parameterGrid({ b: [1, 2], a: ['x', 'y'] }) // => [{ a: 'x', b: 1 }, { a: 'x', b: 2 }, { a: 'y', b: 1 }, { a: 'y', b: 2 }]
 * ```
 */
export function parameterGrid(grid: ParamGrid): Candidate[] {
  const grids = Array.isArray(grid) ? grid : [grid as Readonly<Record<string, readonly unknown[]>>]
  const out: Candidate[] = []
  for (const g of grids) {
    const keys = Object.keys(g).sort()
    const total = keys.reduce((p, k) => p * (g[k]?.length ?? 0), 1)
    if (out.length + total > MAX_CANDIDATES)
      throw new WhiteboxError('value', `parameterGrid: more than ${MAX_CANDIDATES} candidates`)
    for (let i = 0; i < total; i++) {
      let r = i
      const c: Record<string, unknown> = {}
      for (let k = keys.length - 1; k >= 0; k--) {
        const name = keys[k] as string
        const values = g[name] ?? []
        c[name] = values[r % values.length]
        r = Math.floor(r / values.length)
      }
      out.push(Object.fromEntries(keys.map((k) => [k, c[k]])))
    }
  }
  return out
}

/** Flatten a nested space value (`{ step: { param } }`) to `step.param` keys. */
function flattenValue(v: SpaceValue, prefix = ''): Record<string, unknown> {
  if (typeof v === 'object' && v !== null && !Array.isArray(v) && !ArrayBuffer.isView(v)) {
    return Object.fromEntries(
      Object.entries(v as Readonly<Record<string, SpaceValue>>).flatMap(([k, x]) =>
        Object.entries(flattenValue(x, prefix === '' ? k : `${prefix}.${k}`)),
      ),
    )
  }
  return { [prefix]: v }
}

/**
 * Random candidates (sklearn's ParameterSampler): draws from a `Space` (such as an estimator's `hyperSpace`), or, from
 * lists of values, distinct grid points without replacement (with replacement once the grid is exhausted is never
 * needed: at most the grid's size is returned).
 *
 * @param distributions - A space, or lists of values by name.
 * @param nIter - The number of candidates.
 * @param key - The key.
 * @returns The candidates (nested space values flattened to `step.param` names).
 * @example
 * ```ts
 * import { parameterSampler } from 'whitebox-engine/learn/model-selection'
 * import { key } from 'whitebox-engine/random'
 * parameterSampler({ a: [1, 2, 3] }, 5, key(0)).length // => 3
 * ```
 */
export function parameterSampler(
  distributions: Space | Readonly<Record<string, readonly unknown[]>>,
  nIter: number,
  key: Parameters<typeof sample>[1],
): Candidate[] {
  if (!(Number.isInteger(nIter) && nIter >= 1)) throw new WhiteboxError('value', 'parameterSampler: nIter must be >= 1')
  if (typeof (distributions as Space).kind === 'string') {
    const space = distributions as Space
    return Array.from({ length: nIter }, (_, i) => flattenValue(sample(space, foldIn(key, i))))
  }
  const grid = parameterGrid(distributions as Readonly<Record<string, readonly unknown[]>>)
  const perm = toFlat(permutation(key, grid.length))
  return perm.slice(0, Math.min(nIter, grid.length)).map((p) => grid[p] as Candidate)
}

/** Options of the searches. */
export interface SearchOptions extends FitOptions {
  readonly cv?: CvOption
  /** One or several scorers (default the estimator's `score`). */
  readonly scoring?: Scoring
  /** Refit the best candidate on all the data (default true); with several scorers, the scorer to rank by. */
  readonly refit?: boolean | string
  readonly returnTrainScore?: boolean
  readonly groups?: ArrayLike<number>
  readonly errorScore?: number | 'raise'
}

/** The results of a search (sklearn's `cv_results_` and best attributes). */
export interface SearchResult {
  readonly params: readonly Candidate[]
  /** Per scorer: the mean, standard deviation and rank (1 is best, ties share the smallest rank) over candidates. */
  readonly meanTestScore: Readonly<Record<string, readonly number[]>>
  readonly stdTestScore: Readonly<Record<string, readonly number[]>>
  readonly rankTestScore: Readonly<Record<string, readonly number[]>>
  /** splitTestScores[scorer][candidate][fold]. */
  readonly splitTestScores: Readonly<Record<string, readonly (readonly number[])[]>>
  readonly meanTrainScore?: Readonly<Record<string, readonly number[]>>
  readonly bestIndex: number
  readonly bestParams: Candidate
  readonly bestScore: number
  /** The best candidate refitted on all the data (when `refit`). */
  readonly bestEstimator?: Fitted
  readonly folds: readonly Fold[]
}

function rankMin(values: readonly number[]): number[] {
  // scipy.stats.rankdata(-values, method='min'); NaN scores rank last.
  return values.map((v) => (Number.isNaN(v) ? values.length : 1 + values.filter((u) => u > v).length))
}

/**
 * Cross-validate a list of candidates (the engine of the searches): one step per candidate, each candidate's cross-
 * validation a child run; all candidates use the same folds.
 *
 * @param estimator - The estimator.
 * @param candidates - Hyperparameter settings.
 * @param data - The data.
 * @param options - cv, scoring, refit, returnTrainScore, groups, errorScore, key and tracing.
 * @returns The results.
 * @example
 * ```ts
 * import { evaluateCandidates } from 'whitebox-engine/learn/model-selection'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { full } from 'whitebox-engine/tensor'
 * const constant = defineEstimator({ name: 'c', kind: 'regressor', fit: () => ({}), methods: (_, h) => ({ predict: (x) => full([x.shape[0]], h.c) }) }, { c: 0 })
 * const r = evaluateCandidates(constant, [{ c: 0 }, { c: 2 }], { x: [[0], [1]], y: [2, 2] }, { cv: 2, scoring: 'neg_mean_absolute_error' })
 * r.bestParams // => { c: 2 }
 * ```
 */
export function evaluateCandidates(
  estimator: Estimator,
  candidates: readonly Candidate[],
  data: LearnData,
  options: SearchOptions = {},
): SearchResult {
  if (candidates.length === 0) throw new WhiteboxError('value', 'search: no candidates')
  const key = keyFor(options, 'search')
  const splitter = checkCv(options.cv, estimator)
  const folds = splitIndices(splitter, splitInputOf(data, options.groups), foldIn(key, 0))
  // Every candidate sees the same folds.
  const cv: Splitter = {
    name: splitter.name,
    randomised: false,
    partition: splitter.partition,
    nSplits: () => folds.length,
    split: () => folds,
  }
  const scorerNames = Object.keys(scorersOf(options.scoring))
  const refitBy = typeof options.refit === 'string' ? options.refit : scorerNames[0]
  if (refitBy === undefined || !scorerNames.includes(refitBy))
    throw new WhiteboxError('value', `search: refit by unknown scorer ${String(refitBy)}`)
  type Row = {
    readonly test: Readonly<Record<string, readonly number[]>>
    readonly train?: Readonly<Record<string, readonly number[]>>
  }
  const algorithm: Algorithm<readonly Row[]> = {
    name: 'search',
    config: { estimator: estimator.name, candidates: candidates.length, folds: folds.length },
    schema: channels({
      metrics: {
        mean: { description: `the mean test ${refitBy} of the candidate`, direction: 'higher' },
        best: { description: 'the best mean so far', direction: 'higher' },
      },
      messages: { candidate: { description: 'a candidate was evaluated', payload: { index: 'number' } } },
    }),
    init: () => [],
    step: (rows, ctx) => {
      const c = candidates[rows.length] as Candidate
      const r = crossValidate(estimator.with(c), data, {
        ...innerFitOptions(options, foldIn(ctx.key, 0), ctx.run),
        cv,
        ...(options.scoring === undefined ? {} : { scoring: options.scoring }),
        ...(options.returnTrainScore === true ? { returnTrainScore: true } : {}),
        ...(options.errorScore === undefined ? {} : { errorScore: options.errorScore }),
      })
      return [...rows, { test: r.testScores, ...(r.trainScores === undefined ? {} : { train: r.trainScores }) }]
    },
    done: (rows) => (rows.length >= candidates.length ? 'completed' : undefined),
    observe: (rows) => {
      const means = rows.map((r) => meanOf(r.test[refitBy] ?? []))
      return {
        metrics:
          rows.length === 0 ? {} : { mean: means[means.length - 1] ?? 0, best: means.reduce((a, b) => Math.max(a, b)) },
        messages: rows.length === 0 ? [] : [{ kind: 'candidate', payload: { index: rows.length - 1 } }],
      }
    },
  }
  const { state } = runAlgorithm(algorithm, { ...runOptionsFor(options, 'search'), maxSteps: candidates.length })
  const by = (
    pick: (r: Row) => Readonly<Record<string, readonly number[]>> | undefined,
    f: (xs: readonly number[]) => number,
  ) => Object.fromEntries(scorerNames.map((n) => [n, state.map((r) => f(pick(r)?.[n] ?? []))]))
  const meanTestScore = by((r) => r.test, meanOf)
  const stdTestScore = by(
    (r) => r.test,
    (xs) => Math.sqrt(varianceOf(xs, 0)),
  )
  const rankTestScore = Object.fromEntries(scorerNames.map((n) => [n, rankMin(meanTestScore[n] ?? [])]))
  const ranks = rankTestScore[refitBy] ?? []
  const bestIndex = ranks.indexOf(ranks.reduce((a, b) => Math.min(a, b)))
  const bestParams = candidates[bestIndex] as Candidate
  const refit = options.refit !== false
  return {
    params: candidates,
    meanTestScore,
    stdTestScore,
    rankTestScore,
    splitTestScores: Object.fromEntries(scorerNames.map((n) => [n, state.map((r) => r.test[n] ?? [])])),
    ...(options.returnTrainScore === true ? { meanTrainScore: by((r) => r.train, meanOf) } : {}),
    bestIndex,
    bestParams,
    bestScore: meanTestScore[refitBy]?.[bestIndex] ?? Number.NaN,
    ...(refit
      ? {
          bestEstimator: estimator
            .with(bestParams)
            .fit(data, innerFitOptions(options, foldIn(key, 1), options.parent, true)),
        }
      : {}),
    folds,
  }
}

/**
 * Exhaustive search over a grid with cross-validation (sklearn.model_selection.GridSearchCV): every candidate of
 * {@link parameterGrid}, ranked by mean test score, the best refitted on all the data.
 *
 * @param estimator - The estimator.
 * @param grid - The grid (names may address pipeline steps as `step.param`).
 * @param data - The data.
 * @param options - cv, scoring, refit, returnTrainScore, groups, errorScore, key and tracing.
 * @returns The search results.
 * @concept hyperparameter-search
 * @example
 * ```ts
 * import { gridSearchCV } from 'whitebox-engine/learn/model-selection'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { full } from 'whitebox-engine/tensor'
 * const constant = defineEstimator({ name: 'c', kind: 'regressor', fit: () => ({}), methods: (_, h) => ({ predict: (x) => full([x.shape[0]], h.c) }) }, { c: 0 })
 * const r = gridSearchCV(constant, { c: [0, 1, 2, 3] }, { x: [[0], [1], [2]], y: [1, 1, 1] }, { cv: 3, scoring: 'neg_mean_squared_error' })
 * r.rankTestScore.neg_mean_squared_error // => [2, 1, 2, 4]
 * ```
 */
export function gridSearchCV(
  estimator: Estimator,
  grid: ParamGrid,
  data: LearnData,
  options: SearchOptions = {},
): SearchResult {
  return evaluateCandidates(estimator, parameterGrid(grid), data, options)
}

/**
 * Random search with cross-validation (sklearn.model_selection.RandomizedSearchCV): `nIter` candidates from
 * {@link parameterSampler} (a space such as `estimator.hyperSpace`, or lists of values), evaluated like the grid
 * search.
 *
 * @param estimator - The estimator.
 * @param distributions - A space or lists of values (default the estimator's `hyperSpace`).
 * @param data - The data.
 * @param options - nIter (default 10) and the search options.
 * @returns The search results.
 * @example
 * ```ts
 * import { randomizedSearchCV } from 'whitebox-engine/learn/model-selection'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { full } from 'whitebox-engine/tensor'
 * const constant = defineEstimator({ name: 'c', kind: 'regressor', fit: () => ({}), methods: (_, h) => ({ predict: (x) => full([x.shape[0]], h.c) }) }, { c: 0 })
 * const r = randomizedSearchCV(constant, { c: [0, 1, 2] }, { x: [[0], [1]], y: [1, 1] }, { cv: 2, nIter: 3, scoring: 'neg_mean_absolute_error' })
 * r.bestParams // => { c: 1 }
 * ```
 */
export function randomizedSearchCV(
  estimator: Estimator,
  distributions: Space | Readonly<Record<string, readonly unknown[]>> | undefined,
  data: LearnData,
  options: SearchOptions & { readonly nIter?: number } = {},
): SearchResult {
  const space = distributions ?? estimator.hyperSpace
  if (space === undefined)
    throw new WhiteboxError('value', `randomizedSearchCV: ${estimator.name} declares no hyperSpace`)
  const candidates = parameterSampler(space, options.nIter ?? 10, foldIn(keyFor(options, 'randomizedSearchCV'), 0))
  return evaluateCandidates(estimator, candidates, data, options)
}

/** Options of {@link nestedCrossValidation}. */
export interface NestedOptions extends FitOptions {
  /** The outer splitter (default 5 folds). */
  readonly outerCv?: CvOption
  /** The inner splitter used by the search on each outer training set (default 3 folds). */
  readonly innerCv?: CvOption
  readonly scoring?: Scoring
}

/** The result of {@link nestedCrossValidation}. */
export interface NestedResult {
  /** The outer test score of each outer fold's selected and refitted model. */
  readonly scores: readonly number[]
  /** The candidate each outer fold's inner search chose. */
  readonly chosen: readonly Candidate[]
  readonly mean: number
  readonly std: number
}

/**
 * Nested cross-validation (Varma and Simon 2006; Cawley and Talbot 2010): an inner grid search selects the
 * hyperparameters on each outer training set, and the outer test fold scores the refitted choice, an almost unbiased
 * estimate of the whole selection procedure's performance. One step per outer fold, each inner search a child run.
 *
 * @param estimator - The estimator.
 * @param grid - The grid searched inside each outer fold.
 * @param data - The data.
 * @param options - outerCv, innerCv, scoring, key and tracing.
 * @returns The outer scores, the chosen candidates and their mean and deviation.
 * @example
 * ```ts
 * import { nestedCrossValidation } from 'whitebox-engine/learn/model-selection'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { full } from 'whitebox-engine/tensor'
 * const constant = defineEstimator({ name: 'c', kind: 'regressor', fit: () => ({}), methods: (_, h) => ({ predict: (x) => full([x.shape[0]], h.c) }) }, { c: 0 })
 * const x = [[0], [1], [2], [3], [4], [5]]
 * const r = nestedCrossValidation(constant, { c: [0, 1] }, { x, y: [1, 1, 1, 1, 1, 1] }, { outerCv: 2, innerCv: 3, scoring: 'neg_mean_absolute_error' })
 * r.chosen // => [{ c: 1 }, { c: 1 }]
 * ```
 */
export function nestedCrossValidation(
  estimator: Estimator,
  grid: ParamGrid,
  data: LearnData,
  options: NestedOptions = {},
): NestedResult {
  const key = keyFor(options, 'nestedCrossValidation')
  const outer = splitIndices(checkCv(options.outerCv ?? 5, estimator), splitInputOf(data), foldIn(key, 0))
  const candidates = parameterGrid(grid)
  const scorer = Object.values(scorersOf(options.scoring))[0]
  if (scorer === undefined) throw new WhiteboxError('value', 'nestedCrossValidation: no scorer')
  type State = readonly { readonly score: number; readonly chosen: Candidate }[]
  const algorithm: Algorithm<State> = {
    name: 'nested-cross-validation',
    config: { estimator: estimator.name, outer: outer.length, candidates: candidates.length },
    schema: channels({ metrics: { score: { description: 'the outer test score', direction: 'higher' } } }),
    init: () => [],
    step: (s, ctx) => {
      const fold = outer[s.length] as Fold
      const train = takeData(data, fold.train)
      const inner = evaluateCandidates(estimator, candidates, train, {
        ...innerFitOptions(options, foldIn(ctx.key, 0), ctx.run),
        cv: options.innerCv ?? 3,
        ...(options.scoring === undefined ? {} : { scoring: options.scoring }),
      })
      const best = inner.bestEstimator
      if (best === undefined) throw new WhiteboxError('value', 'nestedCrossValidation: the inner search did not refit')
      return [...s, { score: scorer.score(best, takeData(data, fold.test)), chosen: inner.bestParams }]
    },
    done: (s) => (s.length >= outer.length ? 'completed' : undefined),
    observe: (s) => ({ metrics: s.length === 0 ? {} : { score: s[s.length - 1]?.score ?? 0 } }),
  }
  const { state } = runAlgorithm(algorithm, {
    ...runOptionsFor(options, 'nestedCrossValidation'),
    maxSteps: outer.length,
  })
  const scores = state.map((r) => r.score)
  return { scores, chosen: state.map((r) => r.chosen), mean: meanOf(scores), std: Math.sqrt(varianceOf(scores, 0)) }
}
