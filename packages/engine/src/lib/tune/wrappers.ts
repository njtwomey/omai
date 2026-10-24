import { type Decoded, type Encodable, fromJson, type Json, toJson, WhiteboxError } from '../base/index.ts'
import {
  defineEstimator,
  type Estimator,
  type FitOptions,
  type Fitted,
  type LearnData,
  type Verbs,
} from '../learn/index.ts'
import {
  checkCv,
  type Fold,
  type ParamGrid as LearnParamGrid,
  parameterGrid,
  type Splitter,
  splitIndices,
  splitInputOf,
} from '../learn/model-selection/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../random/index.ts'
import { choice, clip, dict, grid, type Space, type SpaceValue, sample } from '../space/index.ts'
import { nullTracer, type RunHandle, startRun } from '../trace/index.ts'
import { type EstimatorSource, estimatorFor, estimatorObjective, type ScoringChoice } from './estimators.ts'
import { randomSampler } from './samplers.ts'
import { halvingPlan } from './schedules.ts'
import { type Study, studyFromJson, studyToJson, type Trial } from './study.ts'
import { tune } from './tuning.ts'

/** A parameter grid as scikit-learn takes it (values per name, or several such grids), or a space to grid. */
export type ParamGrid = LearnParamGrid | Space

/** Distributions to sample as scikit-learn takes them: a space, or per name a space or a list of values. */
export type ParamDistributions = Space | Readonly<Record<string, Space | readonly (string | number | boolean)[]>>

/** Options shared by the search wrappers. */
export interface SearchOptions extends ScoringChoice {
  /** A splitter or a number of folds (default 5: stratified for classifiers). */
  readonly cv?: Splitter | number
  /** Group ids for group splitters. */
  readonly groups?: ArrayLike<number>
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

/** What a fitted search knows besides the best model's verbs. */
export interface SearchFacts {
  readonly bestParams: SpaceValue
  readonly bestScore: number
  readonly bestIndex: number
  readonly cvResults: CvResults
  /** The studies that ran (one per halving iteration). */
  readonly studies: readonly Study[]
  /** The best configuration refitted on all the data (absent with `refit: false`). */
  readonly best?: Fitted
  /** The halving schedule actually run. */
  readonly nCandidates?: readonly number[]
  readonly nResources?: readonly number[]
}

/** A fitted search: a `learn` fitted value whose verbs are the refitted best model's. */
export type FittedSearch = Fitted & SearchFacts

/** A search as a `learn` estimator: fitting it runs the search, so searches nest inside cross-validation. */
export type SearchEstimator = Estimator<object, FittedSearch>

interface SearchOutcome {
  readonly studies: readonly { readonly study: Study; readonly iter: number; readonly resource?: number }[]
  readonly lastIterationOnly: boolean
  readonly nCandidates?: readonly number[]
  readonly nResources?: readonly number[]
}

function isSpace(x: unknown): x is Space {
  return typeof x === 'object' && x !== null && 'kind' in x && typeof (x as { kind: unknown }).kind === 'string'
}

/** The candidates of a grid and a space to record them in. */
function gridOf(paramGrid: ParamGrid, points: number): { readonly space: Space; readonly configs: SpaceValue[] } {
  if (isSpace(paramGrid)) return { space: paramGrid, configs: grid(paramGrid, points).map((v) => clip(paramGrid, v)) }
  const configs = parameterGrid(paramGrid) as SpaceValue[]
  const union = new Map<string, Set<string | number | boolean>>()
  for (const c of configs)
    for (const [k, v] of Object.entries(c as Readonly<Record<string, string | number | boolean>>)) {
      const values = union.get(k) ?? new Set<string | number | boolean>()
      values.add(v)
      union.set(k, values)
    }
  const space = dict(Object.fromEntries([...union.keys()].sort().map((k) => [k, choice([...(union.get(k) ?? [])])])))
  return { space, configs }
}

function asDistributions(d: ParamDistributions): Space {
  if (isSpace(d)) return d
  return dict(
    Object.fromEntries(
      Object.keys(d)
        .sort()
        .map((n) => {
          const v = d[n] as Space | readonly (string | number | boolean)[]
          return [n, Array.isArray(v) ? choice(v as readonly (string | number | boolean)[]) : (v as Space)]
        }),
    ),
  )
}

/** Ranks with ties sharing the smallest (scipy's rankdata method='min'), best (smallest loss) first. */
function ranks(losses: readonly number[]): number[] {
  return losses.map((v) => 1 + losses.filter((w) => w < v || (Number.isNaN(v) && !Number.isNaN(w))).length)
}

function openRun(name: string, options: FitOptions, config: Json): RunHandle {
  const info = { name, seed: null, config, ...(options.policy === undefined ? {} : { policy: options.policy }) }
  return options.parent !== undefined ? options.parent.child(info) : startRun(options.tracer ?? nullTracer, info)
}

/** The search's objective direction: higher-is-better scores are maximised. */
function higherIsBetter(study: Study): boolean {
  return study.directions[0] === 'maximize'
}

function results(outcome: SearchOutcome): {
  readonly cv: CvResults
  readonly bestIndex: number
  readonly trials: Trial[]
} {
  const trials: Trial[] = []
  const iter: number[] = []
  const resources: number[] = []
  for (const s of outcome.studies)
    for (const t of s.study.trials) {
      trials.push(t)
      iter.push(s.iter)
      if (s.resource !== undefined) resources.push(s.resource)
    }
  const higher = higherIsBetter(outcome.studies[0]?.study as Study)
  const mean = trials.map((t) => (t.state === 'complete' ? (t.values[0] ?? Number.NaN) : Number.NaN))
  const losses = mean.map((v) => (Number.isNaN(v) ? Number.NaN : higher ? -v : v))
  const lastIter = iter.reduce((m, v) => Math.max(m, v), 0)
  let bestIndex = -1
  for (const [i, v] of losses.entries()) {
    const eligible = (!outcome.lastIterationOnly || iter[i] === lastIter) && !Number.isNaN(v)
    if (eligible && (bestIndex < 0 || v < (losses[bestIndex] as number))) bestIndex = i
  }
  if (bestIndex < 0) throw new WhiteboxError('value', 'search: every candidate failed')
  return {
    trials,
    bestIndex,
    cv: {
      params: trials.map((t) => t.params),
      meanTestScore: mean,
      stdTestScore: trials.map((t) => Number(t.attrs?.['std'] ?? Number.NaN)),
      rankTestScore: ranks(losses),
      splitTestScores: trials.map((t) => (t.attrs?.['foldScores'] as readonly number[] | undefined) ?? []),
      ...(outcome.lastIterationOnly ? { iter, nResources: resources } : {}),
    },
  }
}

/** Build a search estimator: `run` performs the search; refit, serialisation and verbs are shared. */
function searchEstimator(
  name: string,
  source: EstimatorSource,
  probe: SpaceValue,
  options: SearchOptions,
  run: (data: LearnData, fit: FitOptions, runHandle: RunHandle, key: Key) => SearchOutcome,
): SearchEstimator {
  const kind = estimatorFor(source, probe).kind
  const estimator = defineEstimator(
    {
      name,
      kind,
      fit: (data: LearnData, _hyper: object, fit: FitOptions): Json => {
        const key = fit.key ?? makeKey(0)
        const handle = openRun(name, fit, { kind })
        const outcome = run(data, fit, handle, key)
        const { cv, bestIndex, trials } = results(outcome)
        const bestParams = trials[bestIndex]?.params as SpaceValue
        const best =
          options.refit === false
            ? null
            : estimatorFor(source, bestParams).fit(data, { key: foldIn(key, 1), parent: handle }).params
        handle.end('completed', outcome.studies.length, { bestScore: cv.meanTestScore[bestIndex] ?? Number.NaN })
        return toJson({
          bestParams,
          bestScore: cv.meanTestScore[bestIndex] ?? Number.NaN,
          bestIndex,
          cvResults: cv,
          studies: outcome.studies.map((s) => studyToJson(s.study)),
          best,
          ...(outcome.nCandidates === undefined ? {} : { nCandidates: outcome.nCandidates }),
          ...(outcome.nResources === undefined ? {} : { nResources: outcome.nResources }),
        } as unknown as Encodable)
      },
      methods: (params: Json): Verbs & SearchFacts => {
        const p = fromJson(params) as Decoded as unknown as Omit<SearchFacts, 'studies' | 'best'> & {
          readonly studies: readonly Json[]
          readonly best: Json | null
        }
        const best = p.best === null ? undefined : estimatorFor(source, p.bestParams).revive(p.best)
        return {
          bestParams: p.bestParams,
          bestScore: p.bestScore,
          bestIndex: p.bestIndex,
          cvResults: p.cvResults,
          studies: p.studies.map(studyFromJson),
          ...(best === undefined ? {} : { best }),
          ...(p.nCandidates === undefined ? {} : { nCandidates: p.nCandidates }),
          ...(p.nResources === undefined ? {} : { nResources: p.nResources }),
          ...(best?.predict === undefined ? {} : { predict: best.predict }),
          ...(best?.predictProba === undefined ? {} : { predictProba: best.predictProba }),
          ...(best?.decisionFunction === undefined ? {} : { decisionFunction: best.decisionFunction }),
          ...(best?.predictDistribution === undefined ? {} : { predictDistribution: best.predictDistribution }),
          ...(best?.transform === undefined ? {} : { transform: best.transform }),
          ...(best?.score === undefined ? {} : { score: best.score }),
          ...(best?.classes === undefined ? {} : { classes: best.classes }),
        }
      },
    },
    {},
  )
  return estimator as unknown as SearchEstimator
}

/** Evaluate configurations as a study (enqueued, one trial each) on fixed folds. */
function evaluate(
  source: EstimatorSource,
  data: LearnData,
  options: SearchOptions,
  space: Space,
  configs: readonly SpaceValue[],
  splitter: Splitter,
  run: RunHandle,
  key: Key,
  name: string,
): Study {
  const objective = estimatorObjective(source, data, {
    ...(options.metric === undefined ? {} : { metric: options.metric }),
    ...(options.scoring === undefined ? {} : { scoring: options.scoring }),
    ...(options.response === undefined ? {} : { response: options.response }),
    splitter,
    ...(options.groups === undefined ? {} : { groups: options.groups }),
  })
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

/** A splitter that always returns the same folds (the fixed folds of a search). */
function fixed(folds: readonly Fold[], name: string): Splitter {
  return { name, randomised: false, partition: false, nSplits: () => folds.length, split: () => [...folds] }
}

function baseFolds(
  source: EstimatorSource,
  probe: SpaceValue,
  data: LearnData,
  options: SearchOptions,
  key: Key,
): Fold[] {
  const splitter = checkCv(options.cv, estimatorFor(source, probe))
  return splitIndices(splitter, splitInputOf(data, options.groups), key)
}

/**
 * Exhaustive search over a parameter grid with cross-validation (scikit-learn's `GridSearchCV`), built on a study:
 * every configuration of the grid (scikit-learn's order: names sorted, the last varying fastest; several grids one
 * after another; a space is gridded with `points` per continuous dimension) is a trial of the cross-validated
 * objective, on the same folds. The search is a `learn` estimator: fitting it returns the best configuration, its mean
 * score, `cvResults`, the studies, and the verbs of the best model refitted on all the data.
 *
 * @param estimator - An estimator (re-created by `with`), or a factory.
 * @param paramGrid - Values per name (or a list of such grids), or a space.
 * @param options - The metric or scoring (default the estimator's score), folds, groups, refit and grid points.
 * @returns The search, an estimator.
 * @concept grid-search
 * @example
 * ```ts
 * import { gridSearchCv } from 'whitebox-engine/tune'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * import { asTensor, full } from 'whitebox-engine/tensor'
 * const constant = defineEstimator(
 *   { name: 'constant', kind: 'regressor', fit: () => ({}), methods: (_, h) => ({ predict: (x) => full([asTensor(x).shape[0] ?? 0], h.c) }) },
 *   { c: 0 },
 * )
 * const search = gridSearchCv(constant, { c: [0, 1, 2] }, { metric: meanSquaredError, cv: 2 })
 * search.fit({ x: [[0], [1], [2], [3]], y: [1, 1, 1, 1] }).bestParams // => { c: 1 }
 * ```
 */
export function gridSearchCv(
  estimator: EstimatorSource,
  paramGrid: ParamGrid,
  options: SearchOptions = {},
): SearchEstimator {
  const { space, configs } = gridOf(paramGrid, options.points ?? 5)
  if (configs.length === 0) throw new WhiteboxError('value', 'gridSearchCv: the grid is empty')
  return searchEstimator('gridSearchCv', estimator, configs[0] as SpaceValue, options, (data, _fit, run, key) => {
    const folds = baseFolds(estimator, configs[0] as SpaceValue, data, options, foldIn(key, 0))
    const study = evaluate(
      estimator,
      data,
      options,
      space,
      configs,
      fixed(folds, 'search folds'),
      run,
      foldIn(key, 2),
      'gridSearchCv',
    )
    return { studies: [{ study, iter: 0 }], lastIterationOnly: false }
  })
}

/**
 * Random search with cross-validation (scikit-learn's `RandomizedSearchCV`): `nIter` configurations drawn from the
 * distributions (a space: log-uniform on log dimensions; lists as uniform choices) by a study's random sampler,
 * evaluated on the same folds.
 *
 * @param estimator - An estimator (re-created by `with`), or a factory.
 * @param distributions - A space, or per name a space or a list of values.
 * @param options - The number of configurations (default 10), the metric or scoring, folds, groups and refit.
 * @returns The search, an estimator.
 * @concept random-search
 * @example
 * ```ts
 * import { randomizedSearchCv } from 'whitebox-engine/tune'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * import { real } from 'whitebox-engine/space'
 * import { asTensor, full } from 'whitebox-engine/tensor'
 * const constant = defineEstimator(
 *   { name: 'constant', kind: 'regressor', fit: () => ({}), methods: (_, h) => ({ predict: (x) => full([asTensor(x).shape[0] ?? 0], h.c) }) },
 *   { c: 0 },
 * )
 * const fitted = randomizedSearchCv(constant, { c: real(0, 2) }, { metric: meanSquaredError, cv: 2, nIter: 30 }).fit({ x: [[0], [1], [2], [3]], y: [1, 1, 1, 1] })
 * Math.abs((fitted.bestParams as { c: number }).c - 1) < 0.1 // => true
 * ```
 */
export function randomizedSearchCv(
  estimator: EstimatorSource,
  distributions: ParamDistributions,
  options: SearchOptions & { readonly nIter?: number } = {},
): SearchEstimator {
  const space = asDistributions(distributions)
  const nIter = options.nIter ?? 10
  const probe = sample(space, makeKey(0))
  return searchEstimator('randomizedSearchCv', estimator, probe, options, (data, _fit, run, key) => {
    const folds = baseFolds(estimator, probe, data, options, foldIn(key, 0))
    const objective = estimatorObjective(estimator, data, {
      ...scoringOf(options),
      splitter: fixed(folds, 'search folds'),
      ...(options.groups === undefined ? {} : { groups: options.groups }),
    })
    const study = tune(objective, space, {
      nTrials: nIter,
      sampler: randomSampler(),
      key: foldIn(key, 2),
      parent: run,
      name: 'randomizedSearchCv',
      maxFailures: Number.POSITIVE_INFINITY,
      constantTrials: Number.POSITIVE_INFINITY,
    }).study
    return { studies: [{ study, iter: 0 }], lastIterationOnly: false }
  })
}

function scoringOf(options: SearchOptions): ScoringChoice {
  return {
    ...(options.metric === undefined ? {} : { metric: options.metric }),
    ...(options.scoring === undefined ? {} : { scoring: options.scoring }),
    ...(options.response === undefined ? {} : { response: options.response }),
  }
}

/** Options of the successive-halving searches. */
export interface HalvingOptions extends SearchOptions {
  /** Candidates kept per iteration: 1 / factor of them (default 3). */
  readonly factor?: number
  /** The resource: `'nSamples'` (default; folds are subsampled) or a hyperparameter name set to the resource. */
  readonly resource?: string
  /** Resource of the first iteration (`'exhaust'`, `'smallest'`: 2 x folds (x classes) samples, or 1; or a number). */
  readonly minResources?: number | 'exhaust' | 'smallest'
  /** The most resource (default `'auto'`: the number of samples). */
  readonly maxResources?: number | 'auto'
  /** Iterate until one candidate is left even when resources run out (default false). */
  readonly aggressiveElimination?: boolean
}

function rowsOf(data: LearnData): number {
  return splitInputOf(data).n
}

function halving(
  name: string,
  source: EstimatorSource,
  space: Space,
  probe: SpaceValue,
  candidatesFor: (plan: { readonly min: number; readonly max: number }, key: Key) => SpaceValue[],
  options: HalvingOptions,
  minDefault: 'exhaust' | 'smallest',
): SearchEstimator {
  const factor = options.factor ?? 3
  const resource = options.resource ?? 'nSamples'
  return searchEstimator(name, source, probe, options, (data, _fit, run, key) => {
    const n = rowsOf(data)
    const base = baseFolds(source, probe, data, options, foldIn(key, 0))
    const requested = options.minResources ?? minDefault
    let min: number
    if (typeof requested === 'number') min = requested
    else if (resource === 'nSamples') {
      min = base.length * 2
      if (estimatorFor(source, probe).kind === 'classifier') min *= new Set(Array.from(splitInputOf(data).y ?? [])).size
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
    const studies: { study: Study; iter: number; resource: number }[] = []
    for (let itr = 0; itr < plan.nIterations; itr++) {
      const r = plan.nResources[itr] as number
      let splitter = fixed(base, 'search folds')
      let configs = candidates
      if (resource === 'nSamples') {
        // Each fold's training and test rows subsampled to the fraction r / n, nested across iterations.
        const fraction = r / n
        const pick = (rows: Int32Array, salt: number): Int32Array => {
          const p = permutation(foldIn(foldIn(key, 4), salt), rows.length).data
          return Int32Array.from(Array.from(p, (j) => rows[j] ?? 0).slice(0, Math.trunc(fraction * rows.length)))
        }
        splitter = fixed(
          base.map((f, i) => ({ train: pick(f.train, 2 * i), test: pick(f.test, 2 * i + 1) })),
          'subsampled folds',
        )
      } else configs = candidates.map((c) => ({ ...(c as Readonly<Record<string, SpaceValue>>), [resource]: r }))
      const study = evaluate(
        source,
        data,
        options,
        space,
        configs,
        splitter,
        run,
        foldIn(key, 10 + itr),
        `${name}:${itr}`,
      )
      studies.push({ study, iter: itr, resource: r })
      // scikit-learn's _top_k: ascending by score with NaN first, keep the last k (the best last).
      const keep = Math.ceil(candidates.length / factor)
      const higher = higherIsBetter(study)
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
    return { studies, lastIterationOnly: true, nCandidates: plan.nCandidates, nResources: plan.nResources }
  })
}

/**
 * Successive-halving grid search (scikit-learn's `HalvingGridSearchCV`): every grid configuration is evaluated with a
 * small resource (a subsample of each fold, or a hyperparameter such as the number of epochs), the best 1 / factor go
 * on with factor times the resource, and so on (the iterations of `halvingPlan`, minimum resource `'exhaust'` by
 * default). Each iteration is a study. The best configuration is the best of the last iteration.
 *
 * @param estimator - An estimator (re-created by `with`), or a factory.
 * @param paramGrid - Values per name (or a list of grids), or a space.
 * @param options - Factor, resource, minimum and maximum resources, aggressive elimination, the metric or scoring,
 *   folds, groups and refit.
 * @returns The search, an estimator.
 * @concept successive-halving
 * @example
 * ```ts
 * import { halvingGridSearchCv } from 'whitebox-engine/tune'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * import { asTensor, full } from 'whitebox-engine/tensor'
 * const constant = defineEstimator(
 *   { name: 'constant', kind: 'regressor', fit: () => ({}), methods: (_, h) => ({ predict: (x) => full([asTensor(x).shape[0] ?? 0], h.c) }) },
 *   { c: 0 },
 * )
 * const data = { x: Array.from({ length: 60 }, (_, i) => [i]), y: new Array(60).fill(4) }
 * const fitted = halvingGridSearchCv(constant, { c: [1, 2, 3, 4, 5, 6, 7, 8, 9] }, { metric: meanSquaredError, cv: 3 }).fit(data)
 * [fitted.bestParams, fitted.nCandidates] // => [{ c: 4 }, [9, 3, 1]]
 * ```
 */
export function halvingGridSearchCv(
  estimator: EstimatorSource,
  paramGrid: ParamGrid,
  options: HalvingOptions = {},
): SearchEstimator {
  const { space, configs } = gridOf(paramGrid, options.points ?? 5)
  if (configs.length === 0) throw new WhiteboxError('value', 'halvingGridSearchCv: the grid is empty')
  return halving('halvingGridSearchCv', estimator, space, configs[0] as SpaceValue, () => configs, options, 'exhaust')
}

/**
 * Successive-halving random search (scikit-learn's `HalvingRandomSearchCV`): `nCandidates` random configurations
 * (default `'exhaust'`: max // min resources of them, so the last iteration can use all the resources; minimum
 * resource `'smallest'` by default), halved as in {@link halvingGridSearchCv}.
 *
 * @param estimator - An estimator (re-created by `with`), or a factory.
 * @param distributions - A space, or per name a space or a list of values.
 * @param options - Candidates, factor, resource, minimum and maximum resources, aggressive elimination, the metric or
 *   scoring, folds, groups and refit.
 * @returns The search, an estimator.
 * @throws WhiteboxError with code `value` when both the candidates and the minimum resource are `'exhaust'`.
 * @concept successive-halving
 * @example
 * ```ts
 * import { halvingRandomSearchCv } from 'whitebox-engine/tune'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * import { real } from 'whitebox-engine/space'
 * import { asTensor, full } from 'whitebox-engine/tensor'
 * const constant = defineEstimator(
 *   { name: 'constant', kind: 'regressor', fit: () => ({}), methods: (_, h) => ({ predict: (x) => full([asTensor(x).shape[0] ?? 0], h.c) }) },
 *   { c: 0 },
 * )
 * const data = { x: Array.from({ length: 90 }, (_, i) => [i]), y: new Array(90).fill(4) }
 * const fitted = halvingRandomSearchCv(constant, { c: real(0, 8) }, { metric: meanSquaredError, cv: 3, minResources: 10 }).fit(data)
 * fitted.nCandidates // => [9, 3, 1]
 * ```
 */
export function halvingRandomSearchCv(
  estimator: EstimatorSource,
  distributions: ParamDistributions,
  options: HalvingOptions & { readonly nCandidates?: number | 'exhaust' } = {},
): SearchEstimator {
  const space = asDistributions(distributions)
  const requested = options.nCandidates ?? 'exhaust'
  if (requested === 'exhaust' && options.minResources === 'exhaust')
    throw new WhiteboxError('value', 'halvingRandomSearchCv: nCandidates and minResources cannot both exhaust')
  const draw = (plan: { readonly min: number; readonly max: number }, k: Key): SpaceValue[] => {
    const count = requested === 'exhaust' ? Math.floor(plan.max / plan.min) : requested
    return Array.from({ length: count }, (_, i) => sample(space, foldIn(k, i)))
  }
  return halving('halvingRandomSearchCv', estimator, space, sample(space, makeKey(0)), draw, options, 'smallest')
}
