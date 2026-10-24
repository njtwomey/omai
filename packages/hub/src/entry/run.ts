import { WhiteboxError } from 'whitebox-engine/base'
import { type Dataset, type Knobs, shuffleRows, take } from 'whitebox-engine/data'
import { datasetCollections, datasetGenerators, loadEntry } from 'whitebox-engine/datasets'
import type { Exposition } from 'whitebox-engine/exposition'
import { decisionSurfaceOf, type Estimator, type LearnData } from 'whitebox-engine/learn'
import { key } from 'whitebox-engine/random'
import { contains, type DictSpace, defaults, dict, type Space, type SpaceValue } from 'whitebox-engine/space'
import type { HubEntry, HubProgress, HubRunInput } from './entry.ts'

/** The data of a dataset that learners take: its features and, when it has them, its targets. */
export function learnDataOf(data: Dataset | undefined): LearnData {
  if (data === undefined) throw new WhiteboxError('value', 'learnDataOf: the entry runs without a dataset')
  const d = data as { readonly x?: LearnData['x']; readonly y?: LearnData['y'] }
  if (d.x === undefined) throw new WhiteboxError('value', `learnDataOf: a ${data.kind} dataset has no features`)
  return data as unknown as LearnData
}

/** Options of {@link fitEstimator}. */
export interface FitEstimatorOptions {
  /** Hyperparameters to leave out of the controls (they keep the estimator's values). */
  readonly exclude?: readonly string[]
  /** Add the fitted model's decision surface over the first two features when the method returns none. */
  readonly surface?: boolean
  /** Drop the targets before fitting (clusterers and transformers fitted on labelled data). */
  readonly unsupervised?: boolean
  /** Controls replacing the estimator's own (say a narrower range for the demo datasets). */
  readonly override?: Readonly<Record<string, Space>>
}

/**
 * An entry's `params` and `run` from an estimator: its `hyperSpace` (minus `exclude`) as the controls, with the
 * estimator's own hyperparameters as the demo defaults, and a run that fits `estimator.with(params)` to the dataset.
 *
 * @param estimator - The estimator with its demo hyperparameters.
 * @param options - Hyperparameters to leave out, and whether to add a decision surface.
 * @returns The `params` and `run` of a {@link HubEntry}.
 * @example
 * ```ts
 * import { fitEstimator } from 'whitebox-hub'
 * import { lasso } from 'whitebox-engine/methods/linear'
 * Object.keys(fitEstimator(lasso({ alpha: 0.1 })).params.spaces).includes('alpha') // => true
 * ```
 */
export function fitEstimator(
  estimator: Estimator,
  options: FitEstimatorOptions = {},
): Pick<HubEntry, 'params' | 'run'> {
  const exclude = new Set(options.exclude ?? [])
  const space = estimator.hyperSpace
  const spaces = space?.kind === 'dict' ? space.spaces : {}
  const params: DictSpace = dict(
    Object.fromEntries(Object.entries({ ...spaces, ...options.override }).filter(([k]) => !exclude.has(k))),
  )
  const run = (input: HubRunInput) => {
    const model = Object.keys(input.params).length === 0 ? estimator : estimator.with(input.params)
    const learn = learnDataOf(input.data)
    const fitted = model.fit(options.unsupervised === true ? { x: learn.x } : learn, { key: input.key })
    const exposition: readonly Exposition[] =
      (fitted as { readonly exposition?: readonly Exposition[] }).exposition ?? []
    if (options.surface === true && !exposition.some((r) => r.kind === 'decisionSurface'))
      return { ...fitted, exposition: [...exposition, decisionSurfaceOf(fitted, learn.x)] }
    return { ...fitted, exposition }
  }
  return { params, run }
}

/**
 * An entry's `params` and `run` from a classifier and a regressor of one method (a tree, a forest, k nearest
 * neighbours): the run fits the classifier to classification data and the regressor to anything else. The controls
 * are the classifier's, less the hyperparameters the regressor lacks.
 *
 * @param classifier - The classifier with its demo hyperparameters.
 * @param regressor - The regressor with its demo hyperparameters.
 * @param options - As {@link fitEstimator}.
 * @returns The `params` and `run` of a {@link HubEntry}.
 * @example
 * ```ts
 * import { fitEither } from 'whitebox-hub'
 * import { decisionTreeClassifier, decisionTreeRegressor } from 'whitebox-engine/methods/trees'
 * 'maxDepth' in fitEither(decisionTreeClassifier(), decisionTreeRegressor()).params.spaces // => true
 * ```
 */
export function fitEither(
  classifier: Estimator,
  regressor: Estimator,
  options: FitEstimatorOptions = {},
): Pick<HubEntry, 'params' | 'run'> {
  const shared = (e: Estimator): string[] => (e.hyperSpace?.kind === 'dict' ? Object.keys(e.hyperSpace.spaces) : [])
  const inRegressor = new Set(shared(regressor))
  const only = shared(classifier).filter((k) => !inRegressor.has(k))
  const exclude = [...(options.exclude ?? []), ...only]
  const c = fitEstimator(classifier, { ...options, exclude })
  const r = fitEstimator(regressor, { ...options, exclude })
  // Values the regressor's space does not hold (a classification criterion, say) keep the regressor's own.
  const pick = (params: HubRunInput['params'], from: DictSpace): HubRunInput['params'] =>
    Object.fromEntries(
      Object.entries(params).filter(([k, v]) => {
        const space = from.spaces[k]
        return space !== undefined && contains(space, v)
      }),
    )
  return {
    params: c.params,
    run: (input) =>
      input.data?.kind === 'classification' ? c.run(input) : r.run({ ...input, params: pick(input.params, r.params) }),
  }
}

/** What a hub demo runs on: a dataset of one of the entry's collections, its seed and knobs, and the parameters. */
export interface HubState {
  /** A dataset collection id among the entry's `datasets`. */
  readonly collection: string
  /** An entry name in that collection. */
  readonly dataset: string
  readonly seed: number
  /** The generator's knobs (empty for a vendored dataset). */
  readonly knobs: Knobs
  readonly params: Readonly<Record<string, SpaceValue>>
}

const NO_DATA: Pick<HubState, 'collection' | 'dataset' | 'seed' | 'knobs'> = {
  collection: '',
  dataset: '',
  seed: 0,
  knobs: {},
}

/**
 * The state a demo starts from: the entry's first dataset collection and its `dataset` (default the collection's first)
 * with demo knobs, seed 0,
 * and the parameter space's defaults.
 *
 * @param entry - The entry.
 * @returns The state.
 * @example
 * ```ts
 * import { catalogue, demoState } from 'whitebox-hub'
 * const first = catalogue.entries[0]
 * first === undefined ? 0 : demoState(first).seed // => 0
 * ```
 */
export function demoState(entry: HubEntry): HubState {
  const collection = entry.datasets[0]
  return {
    ...(collection === undefined ? NO_DATA : datasetState(collection, entry.dataset)),
    params: defaults(entry.params) as Readonly<Record<string, SpaceValue>>,
  }
}

/**
 * The dataset part of a state: a collection entry (default its first) with its demo knobs over the generator's
 * defaults.
 *
 * @param collection - A dataset collection id.
 * @param name - An entry name (default the first entry).
 * @param seed - The seed (default 0).
 * @returns The collection, the dataset name, the seed and the knobs.
 * @throws WhiteboxError `value` when the collection or the entry is unknown.
 */
export function datasetState(
  collection: string,
  name?: string,
  seed = 0,
): Pick<HubState, 'collection' | 'dataset' | 'seed' | 'knobs'> {
  const c = datasetCollections[collection]
  if (c === undefined) throw new WhiteboxError('value', `datasetState: unknown dataset collection '${collection}'`)
  const e = name === undefined ? c.entries[0] : c.entries.find((x) => x.name === name)
  if (e === undefined) throw new WhiteboxError('value', `datasetState: '${collection}' has no dataset '${name ?? ''}'`)
  const space = 'generator' in e ? datasetGenerators[e.generator]?.knobs : undefined
  const knobs: Knobs =
    space === undefined ? {} : { ...(defaults(space) as Knobs), ...('knobs' in e ? (e.knobs ?? {}) : {}) }
  return { collection, dataset: e.name, seed, knobs }
}

/**
 * Load the dataset a state names.
 *
 * @param state - The state.
 * @returns The dataset.
 * @throws WhiteboxError `value` when the collection or the entry is unknown.
 */
export function loadState(state: Pick<HubState, 'collection' | 'dataset' | 'seed' | 'knobs'>): Dataset {
  const c = datasetCollections[state.collection]
  const e = c?.entries.find((x) => x.name === state.dataset)
  if (e === undefined)
    throw new WhiteboxError('value', `loadState: no dataset '${state.dataset}' in '${state.collection}'`)
  return loadEntry(e, key(state.seed), state.knobs).data
}

/**
 * The dataset an entry runs on in a state: the state's dataset, cut to a keyed subsample of the entry's `maxRows`
 * when it has more.
 *
 * @param entry - The entry.
 * @param state - The state.
 * @returns The dataset.
 */
export function entryData(entry: HubEntry, state: HubState): Dataset | undefined {
  if (entry.datasets.length === 0) return undefined
  const data = loadState(state)
  const x = (data as { readonly x?: { readonly shape?: readonly number[] } }).x
  const n = x?.shape?.[0] ?? 0
  if (entry.maxRows === undefined || n <= entry.maxRows) return data
  return take(
    shuffleRows(data, key(state.seed + 2)),
    Array.from({ length: entry.maxRows }, (_, i) => i),
  )
}

/** The outcome of {@link runDemo}: the records and the number of rows the method saw. */
export interface DemoResult {
  readonly exposition: readonly Exposition[]
  readonly rows: number
}

/**
 * Run an entry's demo on a state (default its demo state): load the dataset, then run the method with the state's
 * parameters and a key from its seed.
 *
 * @param entry - The entry.
 * @param state - The dataset and parameters (default {@link demoState}).
 * @param progress - Where the run reports progress, if it does.
 * @returns The exposition records and the dataset's row count.
 */
export async function runDemo(
  entry: HubEntry,
  state: HubState = demoState(entry),
  progress?: (p: HubProgress) => void,
): Promise<DemoResult> {
  const data = entryData(entry, state)
  const out = await entry.run({
    data,
    params: state.params,
    key: key(state.seed + 1),
    ...(progress === undefined ? {} : { progress }),
  })
  const x = (data as { readonly x?: { readonly shape?: readonly number[] } } | undefined)?.x
  return { exposition: out.exposition, rows: x?.shape?.[0] ?? 0 }
}
