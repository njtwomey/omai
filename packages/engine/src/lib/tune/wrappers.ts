import { type Json, WhiteboxError } from '../base/index.ts'
import type { Metric } from '../metrics/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../random/index.ts'
import { choice, clip, dict, grid, type Space, type SpaceValue, sample } from '../space/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { nullTracer, type RunHandle, type RunOptions, startRun } from '../trace/index.ts'
import {
  defaultFolds,
  type EstimatorSource,
  estimatorFor,
  estimatorObjective,
  type FittedModel,
  type Fold,
  type Splitter,
  type TabularData,
  type TunableEstimator,
} from './estimators.ts'
import { randomSampler } from './samplers.ts'
import { halvingPlan } from './schedules.ts'
import type { Study, Trial } from './study.ts'
import { tune } from './tuning.ts'

/** A parameter grid as scikit-learn takes it: lists of values per name (or several such grids), or a space. */
export type ParamGrid =
  | Space
  | Readonly<Record<string, readonly (string | number | boolean)[]>>
  | readonly Readonly<Record<string, readonly (string | number | boolean)[]>>[]

/** Distributions to sample as scikit-learn takes them: a space, or per name a space or a list of values. */
export type ParamDistributions = Space | Readonly<Record<string, Space | readonly (string | number | boolean)[]>>

/** Options shared by the search wrappers. */
export interface SearchOptions extends Partial<Pick<RunOptions, 'key' | 'seed' | 'tracer' | 'policy' | 'parent'>> {
  /** The metric scored on each test fold (its spec gives the direction). */
  readonly metric: Metric
  /** A splitter or a number of folds (default 5: stratified for classification). */
  readonly cv?: number | Splitter
  /** Refit the best configuration on all the data (default true). */
  readonly refit?: boolean
  /** Points per continuous dimension when a grid is a space (default 5). */
  readonly points?: number
}

/** scikit-learn's `cv_results_`, one entry per evaluated candidate. */
export interface CvResults {
  readonly params: readonly SpaceValue[]
  readonly meanTestScore: readonly number[]
  readonly stdTestScore: readonly number[]
  /** 1 for the best (ties share the smallest rank). */
  readonly rankTestScore: readonly number[]
  readonly splitTestScores: readonly (readonly number[])[]
  /** The halving iteration of each row, and its resource. */
  readonly iter?: readonly number[]
  readonly nResources?: readonly number[]
}

/** A fitted search: the best configuration, its score, every candidate's results, and the refitted best model. */
export interface FittedSearch extends FittedModel {
  readonly bestParams: SpaceValue
  readonly bestScore: number
  readonly bestIndex: number
  readonly cvResults: CvResults
  /** The studies that ran (one per halving iteration). */
  readonly studies: readonly Study[]
  /** The best configuration refitted on all the data (absent with `refit: false`). */
  readonly best?: FittedModel
  /** The halving schedule actually run. */
  readonly nCandidates?: readonly number[]
  readonly nResources?: readonly number[]
}

/** A search as an estimator: fitting it runs the search (so searches nest inside cross-validation). */
export interface SearchEstimator extends TunableEstimator {
  fit(data: TabularData, options?: { readonly key?: Key }): FittedSearch
}

function asGrids(paramGrid: ParamGrid): { readonly space: Space; readonly configs: SpaceValue[] } | undefined {
  if (typeof paramGrid === 'object' && 'kind' in paramGrid && typeof paramGrid.kind === 'string') return undefined
  const grids = (Array.isArray(paramGrid) ? paramGrid : [paramGrid]) as readonly Readonly<
    Record<string, readonly (string | number | boolean)[]>
  >[]
  const union = new Map<string, Set<string | number | boolean>>()
  const configs: SpaceValue[] = []
  for (const g of grids) {
    // scikit-learn's ParameterGrid sorts the names, and the last varies fastest.
    const names = Object.keys(g).sort()
    for (const n of names) for (const v of g[n] ?? []) (union.get(n) ?? union.set(n, new Set()).get(n))?.add(v)
    if (names.some((n) => (g[n] ?? []).length === 0)) continue
    configs.push(...grid(dict(Object.fromEntries(names.map((n) => [n, choice(g[n] ?? [])])))))
  }
  const space = dict(Object.fromEntries([...union.entries()].sort().map(([n, values]) => [n, choice([...values])])))
  return { space, configs }
}

function asDistributions(d: ParamDistributions): Space {
  if (typeof d === 'object' && 'kind' in d && typeof (d as { kind: unknown }).kind === 'string') return d as Space
  const record = d as Readonly<Record<string, Space | readonly (string | number | boolean)[]>>
  return dict(
    Object.fromEntries(
      Object.keys(record)
        .sort()
        .map((n) => {
          const v = record[n] as Space | readonly (string | number | boolean)[]
          return [n, Array.isArray(v) ? choice(v as readonly (string | number | boolean)[]) : (v as Space)]
        }),
    ),
  )
}

/** Ranks with ties sharing the smallest (scipy's rankdata method='min'), best first. */
function ranks(losses: readonly number[]): number[] {
  return losses.map((v) => 1 + losses.filter((w) => w < v || (Number.isNaN(v) && !Number.isNaN(w))).length)
}

interface SearchRun {
  readonly run: RunHandle
  readonly key: Key
}

function openRun(name: string, options: SearchOptions, config: Json, key?: Key): SearchRun {
  const k = key ?? options.key ?? makeKey(options.seed ?? 0)
  const info = {
    name,
    seed: options.seed ?? null,
    config,
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  }
  const run = options.parent !== undefined ? options.parent.child(info) : startRun(options.tracer ?? nullTracer, info)
  return { run, key: k }
}

function rowsOf(studies: readonly { readonly study: Study; readonly iter: number; readonly resource?: number }[]): {
  trials: Trial[]
  iter: number[]
  resources: number[]
} {
  const trials: Trial[] = []
  const iter: number[] = []
  const resources: number[] = []
  for (const s of studies)
    for (const t of s.study.trials) {
      trials.push(t)
      iter.push(s.iter)
      if (s.resource !== undefined) resources.push(s.resource)
    }
  return { trials, iter, resources }
}

function finish(
  source: EstimatorSource,
  data: TabularData,
  options: SearchOptions,
  studies: readonly { readonly study: Study; readonly iter: number; readonly resource?: number }[],
  key: Key,
  lastIterationOnly: boolean,
  run: RunHandle,
  extra: { readonly nCandidates?: readonly number[]; readonly nResources?: readonly number[] } = {},
): FittedSearch {
  const higher = options.metric.spec.direction === 'higher'
  const { trials, iter, resources } = rowsOf(studies)
  const mean = trials.map((t) => (t.state === 'complete' ? (t.values[0] ?? Number.NaN) : Number.NaN))
  const losses = mean.map((v) => (Number.isNaN(v) ? Number.NaN : higher ? -v : v))
  const lastIter = iter.reduce((m, v) => Math.max(m, v), 0)
  const eligible = losses.map((v, i) => (!lastIterationOnly || iter[i] === lastIter) && !Number.isNaN(v))
  let bestIndex = -1
  for (const [i, v] of losses.entries())
    if (eligible[i] === true && (bestIndex < 0 || v < (losses[bestIndex] as number))) bestIndex = i
  if (bestIndex < 0) throw new WhiteboxError('value', 'search: every candidate failed')
  const bestParams = trials[bestIndex]?.params as SpaceValue
  const cvResults: CvResults = {
    params: trials.map((t) => t.params),
    meanTestScore: mean,
    stdTestScore: trials.map((t) => Number(t.attrs?.['std'] ?? Number.NaN)),
    rankTestScore: ranks(losses),
    splitTestScores: trials.map((t) => (t.attrs?.['foldScores'] as readonly number[] | undefined) ?? []),
    ...(lastIterationOnly ? { iter, nResources: resources } : {}),
  }
  const best = options.refit === false ? undefined : estimatorFor(source, bestParams).fit(data, { key: foldIn(key, 1) })
  run.end('completed', studies.length, { bestScore: mean[bestIndex] ?? Number.NaN })
  return {
    bestParams,
    bestScore: mean[bestIndex] as number,
    bestIndex,
    cvResults,
    studies: studies.map((s) => s.study),
    ...(best === undefined ? {} : { best }),
    ...extra,
    predict: (x: TensorLike) => {
      if (best === undefined) throw new WhiteboxError('value', 'search: refit is off, so there is no model to predict')
      return best.predict(x)
    },
    predictProba: (x: TensorLike) => {
      if (best?.predictProba === undefined)
        throw new WhiteboxError('value', 'search: the best model has no predictProba')
      return best.predictProba(x)
    },
  }
}

function evaluate(
  source: EstimatorSource,
  data: TabularData,
  options: SearchOptions,
  space: Space,
  configs: readonly SpaceValue[],
  folds: Splitter,
  run: RunHandle,
  key: Key,
  name: string,
): Study {
  const objective = estimatorObjective(source, data, { metric: options.metric, cv: folds })
  return tune(objective, space, {
    enqueue: configs,
    nTrials: configs.length,
    sampler: randomSampler(),
    key,
    parent: run,
    name,
    maxFailures: Number.POSITIVE_INFINITY,
    constantTrials: Number.POSITIVE_INFINITY,
  }).study
}

function fixedFolds(data: TabularData, options: SearchOptions, key: Key): readonly Fold[] {
  const cv = options.cv ?? 5
  return typeof cv === 'number' ? defaultFolds(data, cv) : cv(data, key)
}

/**
 * Exhaustive search over a parameter grid with cross-validation (scikit-learn's `GridSearchCV`), built on a study:
 * every configuration of the grid (scikit-learn's order: names sorted, the last varying fastest; several grids one
 * after another) is a trial of the cross-validated objective, on the same folds. Fitting returns the best
 * configuration, its mean score, `cvResults`, and the best model refitted on all the data.
 *
 * @param estimator - An estimator with `withHyper`, or a factory.
 * @param paramGrid - Values per name (or a list of such grids), or a space (gridded with `points`).
 * @param options - The metric, folds, refit and the run options.
 * @returns The search, an estimator.
 * @concept grid-search
 * @example
 * ```ts
 * import { gridSearchCv } from 'whitebox-engine/tune'
 * import { xy } from 'whitebox-engine/data'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * import { asTensor } from 'whitebox-engine/tensor'
 * const constant = (p: unknown) => ({ name: 'constant', fit: () => ({ predict: (x: unknown) => new Array(asTensor(x as number[][]).shape[0]).fill((p as { c: number }).c) }) })
 * const search = gridSearchCv(constant, { c: [0, 1, 2] }, { metric: meanSquaredError, cv: 2 })
 * search.fit(xy([[0], [1], [2], [3]], [1, 1, 1, 1])).bestParams // => { c: 1 }
 * ```
 */
export function gridSearchCv(
  estimator: EstimatorSource,
  paramGrid: ParamGrid,
  options: SearchOptions,
): SearchEstimator {
  return {
    name: 'gridSearchCv',
    fit: (data, fitOptions = {}) => {
      const grids = asGrids(paramGrid)
      const space = grids?.space ?? (paramGrid as Space)
      const configs = grids?.configs ?? grid(space, options.points ?? 5).map((v) => clip(space, v))
      const { run, key } = openRun('gridSearchCv', options, { candidates: configs.length }, fitOptions.key)
      const folds = fixedFolds(data, options, foldIn(key, 0))
      const study = evaluate(estimator, data, options, space, configs, () => folds, run, foldIn(key, 2), 'gridSearchCv')
      return finish(estimator, data, options, [{ study, iter: 0 }], key, false, run)
    },
  }
}

/**
 * Random search with cross-validation (scikit-learn's `RandomizedSearchCV`): `nIter` configurations drawn from the
 * distributions (a space: log-uniform on log dimensions; lists as uniform choices), evaluated on the same folds.
 *
 * @param estimator - An estimator with `withHyper`, or a factory.
 * @param distributions - A space, or per name a space or a list of values.
 * @param options - The number of configurations (default 10), the metric, folds, refit and the run options.
 * @returns The search, an estimator.
 * @concept random-search
 * @example
 * ```ts
 * import { randomizedSearchCv } from 'whitebox-engine/tune'
 * import { xy } from 'whitebox-engine/data'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * import { real } from 'whitebox-engine/space'
 * import { asTensor } from 'whitebox-engine/tensor'
 * const constant = (p: unknown) => ({ name: 'constant', fit: () => ({ predict: (x: unknown) => new Array(asTensor(x as number[][]).shape[0]).fill((p as { c: number }).c) }) })
 * const fitted = randomizedSearchCv(constant, { c: real(0, 2) }, { metric: meanSquaredError, cv: 2, nIter: 30 }).fit(xy([[0], [1], [2], [3]], [1, 1, 1, 1]))
 * Math.abs((fitted.bestParams as { c: number }).c - 1) < 0.1 // => true
 * ```
 */
export function randomizedSearchCv(
  estimator: EstimatorSource,
  distributions: ParamDistributions,
  options: SearchOptions & { readonly nIter?: number },
): SearchEstimator {
  return {
    name: 'randomizedSearchCv',
    fit: (data, fitOptions = {}) => {
      const space = asDistributions(distributions)
      const nIter = options.nIter ?? 10
      const { run, key } = openRun('randomizedSearchCv', options, { nIter }, fitOptions.key)
      const folds = fixedFolds(data, options, foldIn(key, 0))
      const objective = estimatorObjective(estimator, data, { metric: options.metric, cv: () => folds })
      const study = tune(objective, space, {
        nTrials: nIter,
        sampler: randomSampler(),
        key: foldIn(key, 2),
        parent: run,
        name: 'randomizedSearchCv',
        maxFailures: Number.POSITIVE_INFINITY,
        constantTrials: Number.POSITIVE_INFINITY,
      }).study
      return finish(estimator, data, options, [{ study, iter: 0 }], key, false, run)
    },
  }
}

/** Options of the successive-halving searches. */
export interface HalvingOptions extends SearchOptions {
  /** Candidates kept per iteration: 1 / factor of them (default 3). */
  readonly factor?: number
  /** The resource: `'nSamples'` (default; folds are subsampled) or a hyperparameter name set to the resource. */
  readonly resource?: string
  /** Resource of the first iteration (default `'exhaust'`; `'smallest'`: 2 x folds (x classes) samples, or 1). */
  readonly minResources?: number | 'exhaust' | 'smallest'
  /** The most resource (default `'auto'`: the number of samples). */
  readonly maxResources?: number | 'auto'
  /** Iterate until one candidate is left even when resources run out (default false). */
  readonly aggressiveElimination?: boolean
}

function halving(
  name: string,
  estimator: EstimatorSource,
  space: Space,
  candidatesFor: (plan: { readonly min: number; readonly max: number }, key: Key) => SpaceValue[],
  data: TabularData,
  options: HalvingOptions,
  key: Key,
): FittedSearch {
  const factor = options.factor ?? 3
  const resource = options.resource ?? 'nSamples'
  const n = data.y.shape[0] ?? 0
  const base = fixedFolds(data, options, foldIn(key, 0))
  let min: number
  const requested = options.minResources ?? 'exhaust'
  if (typeof requested === 'number') min = requested
  else if (resource === 'nSamples') {
    min = base.length * 2
    if (data.kind === 'classification') min *= new Set(Array.from(data.y.data)).size
  } else min = 1
  const requestedMax = options.maxResources ?? 'auto'
  if (requestedMax === 'auto' && resource !== 'nSamples')
    throw new WhiteboxError('value', `${name}: maxResources must be given for the resource ${resource}`)
  const max = requestedMax === 'auto' ? n : requestedMax
  if (min > max) throw new WhiteboxError('value', `${name}: min resources ${min} exceed max resources ${max}`)
  let candidates = candidatesFor({ min, max }, foldIn(key, 3))
  const plan = halvingPlan({
    nCandidates: candidates.length,
    minResources: min,
    maxResources: max,
    factor,
    exhaust: requested === 'exhaust',
    ...(options.aggressiveElimination === undefined ? {} : { aggressiveElimination: options.aggressiveElimination }),
  })
  const { run } = openRun(name, options, { factor, resource, iterations: plan.nIterations }, key)
  const higher = options.metric.spec.direction === 'higher'
  const studies: { study: Study; iter: number; resource: number }[] = []
  for (let itr = 0; itr < plan.nIterations; itr++) {
    const r = plan.nResources[itr] as number
    const iterationKey = foldIn(key, 10 + itr)
    let folds: Splitter = () => base
    let configs = candidates
    if (resource === 'nSamples') {
      // Each fold's training and test rows subsampled to the fraction r / n (nested across iterations).
      const fraction = r / n
      const sub = base.map((f, i) => {
        const order = (rows: readonly number[], salt: number): number[] => {
          const p = permutation(foldIn(foldIn(key, 4), 2 * i + salt), rows.length).data
          return Array.from(p, (j) => rows[j] as number).slice(0, Math.trunc(fraction * rows.length))
        }
        return { train: order(f.train, 0), test: order(f.test, 1) }
      })
      folds = () => sub
    } else configs = candidates.map((c) => ({ ...(c as Readonly<Record<string, SpaceValue>>), [resource]: r }))
    const study = evaluate(estimator, data, options, space, configs, folds, run, iterationKey, `${name}:${itr}`)
    studies.push({ study, iter: itr, resource: r })
    // scikit-learn's _top_k: ascending by score with NaN first, keep the last k (the best last).
    const keep = Math.ceil(candidates.length / factor)
    const scored = study.trials.map((t, i) => ({
      i,
      score:
        t.state === 'complete' ? (higher ? (t.values[0] ?? Number.NaN) : -(t.values[0] ?? Number.NaN)) : Number.NaN,
    }))
    scored.sort((a, b) =>
      Number.isNaN(a.score)
        ? Number.isNaN(b.score)
          ? a.i - b.i
          : -1
        : Number.isNaN(b.score)
          ? 1
          : a.score - b.score || a.i - b.i,
    )
    candidates = scored.slice(-keep).map((s) => candidates[s.i] as SpaceValue)
  }
  return finish(estimator, data, options, studies, key, true, run, {
    nCandidates: plan.nCandidates,
    nResources: plan.nResources,
  })
}

/**
 * Successive-halving grid search (scikit-learn's `HalvingGridSearchCV`): every grid configuration is evaluated with a
 * small resource (a subsample of each fold, or a hyperparameter such as the number of epochs), the best 1 / factor go
 * on with factor times the resource, and so on (the iterations of `halvingPlan`). Each iteration is a study. The best
 * configuration is the best of the last iteration.
 *
 * @param estimator - An estimator with `withHyper`, or a factory.
 * @param paramGrid - Values per name (or a list of grids), or a space.
 * @param options - Factor, resource, minimum and maximum resources, aggressive elimination, the metric, folds, refit
 *   and the run options.
 * @returns The search, an estimator.
 * @concept successive-halving
 * @example
 * ```ts
 * import { halvingGridSearchCv } from 'whitebox-engine/tune'
 * import { xy } from 'whitebox-engine/data'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * import { asTensor } from 'whitebox-engine/tensor'
 * const constant = (p: unknown) => ({ name: 'constant', fit: () => ({ predict: (x: unknown) => new Array(asTensor(x as number[][]).shape[0]).fill((p as { c: number }).c) }) })
 * const data = xy(Array.from({ length: 60 }, (_, i) => [i]), new Array(60).fill(4))
 * const fitted = halvingGridSearchCv(constant, { c: [1, 2, 3, 4, 5, 6, 7, 8, 9] }, { metric: meanSquaredError, cv: 3 }).fit(data)
 * [fitted.bestParams, fitted.nCandidates] // => [{ c: 4 }, [9, 3, 1]]
 * ```
 */
export function halvingGridSearchCv(
  estimator: EstimatorSource,
  paramGrid: ParamGrid,
  options: HalvingOptions,
): SearchEstimator {
  return {
    name: 'halvingGridSearchCv',
    fit: (data, fitOptions = {}) => {
      const grids = asGrids(paramGrid)
      const space = grids?.space ?? (paramGrid as Space)
      const configs = grids?.configs ?? grid(space, options.points ?? 5).map((v) => clip(space, v))
      const key = fitOptions.key ?? options.key ?? makeKey(options.seed ?? 0)
      return halving('halvingGridSearchCv', estimator, space, () => configs, data, options, key)
    },
  }
}

/**
 * Successive-halving random search (scikit-learn's `HalvingRandomSearchCV`): `nCandidates` random configurations
 * (default `'exhaust'`: max // min resources of them, so the last iteration can use all the resources), halved as in
 * {@link halvingGridSearchCv}.
 *
 * @param estimator - An estimator with `withHyper`, or a factory.
 * @param distributions - A space, or per name a space or a list of values.
 * @param options - Candidates, factor, resource, minimum and maximum resources, aggressive elimination, the metric,
 *   folds, refit and the run options.
 * @returns The search, an estimator.
 * @concept successive-halving
 * @example
 * ```ts
 * import { halvingRandomSearchCv } from 'whitebox-engine/tune'
 * import { xy } from 'whitebox-engine/data'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * import { real } from 'whitebox-engine/space'
 * import { asTensor } from 'whitebox-engine/tensor'
 * const constant = (p: unknown) => ({ name: 'constant', fit: () => ({ predict: (x: unknown) => new Array(asTensor(x as number[][]).shape[0]).fill((p as { c: number }).c) }) })
 * const data = xy(Array.from({ length: 90 }, (_, i) => [i]), new Array(90).fill(4))
 * const fitted = halvingRandomSearchCv(constant, { c: real(0, 8) }, { metric: meanSquaredError, cv: 3, minResources: 10 }).fit(data)
 * fitted.nCandidates // => [9, 3, 1]
 * ```
 */
export function halvingRandomSearchCv(
  estimator: EstimatorSource,
  distributions: ParamDistributions,
  options: HalvingOptions & { readonly nCandidates?: number | 'exhaust' },
): SearchEstimator {
  return {
    name: 'halvingRandomSearchCv',
    fit: (data, fitOptions = {}) => {
      const space = asDistributions(distributions)
      const requested = options.nCandidates ?? 'exhaust'
      if (requested === 'exhaust' && options.minResources === 'exhaust')
        throw new WhiteboxError('value', 'halvingRandomSearchCv: nCandidates and minResources cannot both exhaust')
      const key = fitOptions.key ?? options.key ?? makeKey(options.seed ?? 0)
      const draw = (plan: { readonly min: number; readonly max: number }, k: Key): SpaceValue[] => {
        const count = requested === 'exhaust' ? Math.floor(plan.max / plan.min) : requested
        return Array.from({ length: count }, (_, i) => sample(space, foldIn(k, i)))
      }
      // scikit-learn's defaults here: as many candidates as the resources allow, starting from the smallest resource.
      const minResources = options.minResources ?? 'smallest'
      return halving('halvingRandomSearchCv', estimator, space, draw, data, { ...options, minResources }, key)
    },
  }
}
