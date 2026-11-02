/**
 * The tagged dataset registry's machinery: an entry is a generator, a vendored real dataset or a pinned remote asset
 * with its tags (kind, modality, real or synthetic, size, licence, source) beside its title and description;
 * `generatorEntry`, `vendoredEntry` and `remoteEntry` build one, `whereDatasets` answers the queries a selector makes
 * (kinds, modalities, real, form, free text), `loadDataset` and `fetchDataset` make or load an entry's data, and
 * `registerDatasets` returns a new registry with an app's own entries added. The library's own registry, built from
 * the catalogue, is `datasetRegistry` in `catalogue.ts`; nothing registers itself on import.
 *
 * @index structure Registry machinery: entries, tags, queries, load and fetch.
 */

import { find, MdmlError, matchesText, type Registry, register } from '../base/index.ts'
import {
  asGenerated,
  type Dataset,
  type DatasetGenerator,
  type DatasetKind,
  type DatasetRef,
  type Generated,
  type GenerateOptions,
  generate,
  type Knobs,
  type LoadOptions,
  load,
} from '../data/index.ts'

/**
 * What a dataset's rows are: the modality a selector filters by, beside the finer task `kind`. Tabular rows (features
 * and targets), images, audio, time series, text, graphs, token sequences, events of a point process, interactions
 * between two kinds of entity (users and items, players, workers and items) and trajectories of an agent.
 */
export type DatasetModality =
  | 'tabular'
  | 'image'
  | 'audio'
  | 'timeSeries'
  | 'text'
  | 'graph'
  | 'sequence'
  | 'events'
  | 'interactions'
  | 'trajectories'

/** Every modality, in the order selectors list them. */
export const DATASET_MODALITIES: readonly DatasetModality[] = [
  'tabular',
  'image',
  'audio',
  'timeSeries',
  'text',
  'graph',
  'sequence',
  'events',
  'interactions',
  'trajectories',
]

/** How an entry's data come to be: a generator with knobs, a dataset vendored into the bundle, or a remote asset. */
export type DatasetForm = 'generator' | 'vendored' | 'remote'

/** The tags of a dataset entry. */
export interface DatasetTags {
  readonly kind: DatasetKind
  readonly modality: DatasetModality
  /** Real data (measured or collected), against synthetic data from a generator. */
  readonly real: boolean
  /** Rows, samples, documents or nodes, where known (generators take theirs as a knob). */
  readonly size?: number
  readonly licence?: string
  readonly source?: string
}

/** What every dataset entry carries besides its form. */
export interface DatasetEntryBase extends DatasetTags {
  /** The entry's name, unique in its registry (a generator's name, a loader's name). */
  readonly name: string
  readonly title: string
  readonly description: string
}

/**
 * One registered dataset: its tags, and its form: a generator (with the knobs a selector starts from), a vendored
 * loader, or a remote reference loaded by `fetchDataset`.
 *
 * @concept datasets
 */
export type DatasetEntry = DatasetEntryBase &
  (
    | { readonly form: 'generator'; readonly generator: DatasetGenerator; readonly knobs?: Knobs }
    | { readonly form: 'vendored'; readonly load: () => Generated }
    | { readonly form: 'remote'; readonly ref: DatasetRef }
  )

/** A registry of datasets. */
export type DatasetRegistry = Registry<DatasetEntry>

/** The tags and texts of a generator entry: its title, modality and whatever the generator does not say itself. */
export interface GeneratorEntryTags extends Partial<Omit<DatasetTags, 'kind' | 'modality'>> {
  readonly title: string
  readonly modality: DatasetModality
  /** A name other than the generator's. */
  readonly name?: string
  /** A description other than the generator's. */
  readonly description?: string
  /** Knob values a selector starts from (the rest at the generator's defaults). */
  readonly knobs?: Knobs
}

/**
 * A registry entry for a generator: its kind, name and description come from the generator, `real` is false unless
 * said otherwise.
 *
 * @param generator - The generator.
 * @param tags - The title, the modality and any other tags.
 * @returns The entry.
 * @concept datasets
 * @example
 * ```ts
 * import { generatorEntry, moons } from 'mdml-engine/datasets'
 * generatorEntry(moons, { title: 'Two moons', modality: 'tabular' }).kind // => 'classification'
 * ```
 */
export function generatorEntry(generator: DatasetGenerator, tags: GeneratorEntryTags): DatasetEntry {
  const { title, modality, name, description, knobs, real, size, licence, source } = tags
  return {
    form: 'generator',
    generator,
    name: name ?? generator.name,
    title,
    description: description ?? generator.description,
    kind: generator.kind,
    modality,
    real: real ?? false,
    ...(knobs === undefined ? {} : { knobs }),
    ...(size === undefined ? {} : { size }),
    ...(licence === undefined ? {} : { licence }),
    ...(source === undefined ? {} : { source }),
  }
}

/** The tags and texts of a vendored entry (everything, since nothing is loaded at registration). */
export interface VendoredEntryTags extends Omit<DatasetTags, 'real'> {
  readonly title: string
  readonly description: string
  /** Real data unless said otherwise (a vendored dataset may be a fixed synthetic one). */
  readonly real?: boolean
}

/** Whether a loader returned data with its truth (a `Generated`) rather than a bare dataset. */
function isGenerated(value: Dataset | Generated): value is Generated {
  return 'truth' in value && 'data' in value
}

/**
 * A registry entry for a dataset vendored into the bundle: a loader returning the data, or the data with its truth.
 *
 * @param name - The entry's name.
 * @param loader - The loader.
 * @param tags - The title, description, kind, modality and provenance.
 * @returns The entry.
 * @concept datasets
 * @example
 * ```ts
 * import { iris, vendoredEntry } from 'mdml-engine/datasets'
 * const entry = vendoredEntry('iris', iris, {
 *   title: 'Iris', description: 'Fisher’s irises.', kind: 'classification', modality: 'tabular', size: 150,
 * })
 * entry.real // => true
 * ```
 */
export function vendoredEntry(name: string, loader: () => Dataset | Generated, tags: VendoredEntryTags): DatasetEntry {
  const { real = true, ...rest } = tags
  return {
    form: 'vendored',
    name,
    real,
    ...rest,
    load: () => {
      const out = loader()
      return isGenerated(out) ? out : asGenerated(out)
    },
  }
}

/** The tags and texts of a remote entry; the licence defaults to the reference's. */
export interface RemoteEntryTags extends Omit<DatasetTags, 'real'> {
  readonly title: string
  readonly description: string
  readonly real?: boolean
}

/**
 * A registry entry for a remote asset pinned by its reference, loaded on demand by {@link fetchDataset}.
 *
 * @param ref - The dataset reference.
 * @param tags - The title, description, kind, modality and provenance.
 * @returns The entry.
 * @concept datasets
 * @example
 * ```ts
 * import { californiaHousing, remoteEntry } from 'mdml-engine/datasets'
 * remoteEntry(californiaHousing, {
 *   title: 'California housing', description: 'Census block groups.', kind: 'regression', modality: 'tabular',
 * }).licence // => 'Public domain (US Census data)'
 * ```
 */
export function remoteEntry(ref: DatasetRef, tags: RemoteEntryTags): DatasetEntry {
  const { real = true, licence, ...rest } = tags
  const lic = licence ?? ('licence' in ref ? ref.licence : undefined)
  return { form: 'remote', ref, name: ref.name, real, ...rest, ...(lic === undefined ? {} : { licence: lic }) }
}

// ------------------------------------------------------------------------------------------------------ queries

/** A query over a dataset registry: every given filter must hold (a list means any of its values). */
export interface DatasetQuery {
  readonly kind?: DatasetKind | readonly DatasetKind[]
  readonly modality?: DatasetModality | readonly DatasetModality[]
  readonly real?: boolean
  readonly form?: DatasetForm | readonly DatasetForm[]
  /** Search-box text: every word must appear in the name, title, description, kind, modality or source. */
  readonly text?: string
}

const oneOf = <T>(filter: T | readonly T[] | undefined, value: T): boolean =>
  filter === undefined || (Array.isArray(filter) ? (filter as readonly T[]).includes(value) : filter === value)

/**
 * Whether an entry satisfies a query (what {@link whereDatasets} tests entry by entry).
 *
 * @param entry - The entry.
 * @param query - The query.
 * @returns Whether every given filter holds.
 * @concept datasets
 * @example
 * ```ts
 * import { lookup } from 'mdml-engine/base'
 * import { datasetRegistry, matchesDatasetQuery } from 'mdml-engine/datasets'
 * const moons = lookup(datasetRegistry, 'moons')
 * matchesDatasetQuery(moons, { kind: ['classification', 'regression'], real: false }) // => true
 * matchesDatasetQuery(moons, { text: 'real' }) // => false
 * ```
 */
export function matchesDatasetQuery(entry: DatasetEntry, query: DatasetQuery): boolean {
  return (
    oneOf(query.kind, entry.kind) &&
    oneOf(query.modality, entry.modality) &&
    (query.real === undefined || query.real === entry.real) &&
    oneOf(query.form, entry.form) &&
    matchesText(query.text, [entry.name, entry.title, entry.description, entry.kind, entry.modality, entry.source])
  )
}

/**
 * The entries of a registry that satisfy a query, in registration order; no query returns every entry.
 *
 * @param reg - The registry.
 * @param query - The filters.
 * @returns The matching entries.
 * @concept datasets
 * @index function Registry entries matching kind, modality, size or free text.
 * @example
 * ```ts
 * import { datasetRegistry, whereDatasets } from 'mdml-engine/datasets'
 * whereDatasets(datasetRegistry, { kind: 'image', real: true }).map((e) => e.name) // => ['digitImages', 'fashionMnistTest']
 * whereDatasets(datasetRegistry, { modality: 'tabular', text: 'penguins' }).length // => 2
 * ```
 */
export function whereDatasets(reg: DatasetRegistry, query: DatasetQuery = {}): DatasetEntry[] {
  return reg.entries.filter((e) => matchesDatasetQuery(e, query))
}

/**
 * A new dataset registry with more entries (an app's own generators and data), checked: a generator entry's kind must
 * be its generator's. An entry with a registered name replaces it; the given registry is left as it was.
 *
 * @param reg - The registry to extend (usually `datasetRegistry`).
 * @param entries - The entries to add.
 * @returns The extended registry.
 * @throws MdmlError with code `value` for a generator entry whose kind is not its generator's, or a name added twice.
 * @concept datasets
 * @example
 * ```ts
 * import { datasetRegistry, generatorEntry, moons, registerDatasets, whereDatasets } from 'mdml-engine/datasets'
 * const mine = registerDatasets(datasetRegistry, [
 *   generatorEntry(moons, { name: 'noisyMoons', title: 'Noisy moons', modality: 'tabular', knobs: { noise: 0.4 } }),
 * ])
 * mine.entries.length - datasetRegistry.entries.length // => 1
 * whereDatasets(mine, { text: 'noisy moons' })[0]?.name // => 'noisyMoons'
 * ```
 */
export function registerDatasets(reg: DatasetRegistry, entries: readonly DatasetEntry[]): DatasetRegistry {
  for (const e of entries) {
    if (e.form === 'generator' && e.kind !== e.generator.kind) {
      throw new MdmlError(
        'value',
        `registerDatasets: '${e.name}' is tagged ${e.kind} but its generator makes ${e.generator.kind}`,
      )
    }
  }
  return register(reg, entries)
}

/**
 * The entry of a name in a dataset registry.
 *
 * @param reg - The registry.
 * @param name - The entry's name.
 * @returns The entry, or `undefined` when nothing of that name is registered.
 * @concept datasets
 * @example
 * ```ts
 * import { datasetRegistry, findDataset } from 'mdml-engine/datasets'
 * findDataset(datasetRegistry, 'wine')?.title // => 'Wine'
 * ```
 */
export function findDataset(reg: DatasetRegistry, name: string): DatasetEntry | undefined {
  return find(reg, name)
}

// ------------------------------------------------------------------------------------------------------ loading

/**
 * Make or load an entry's data, synchronously: a generator entry is generated (its knobs under the given ones, the key
 * from the options or the context, as `generate`), a vendored entry is loaded. A remote entry needs
 * {@link fetchDataset}.
 *
 * @param entry - The entry.
 * @param options - Knobs, key or context for a generator.
 * @returns The data with its truth (empty for a vendored dataset without one).
 * @throws MdmlError with code `value` for a remote entry, or a knob outside the generator's space.
 * @concept datasets
 * @index function Make or load a registry entry's data synchronously.
 * @example
 * ```ts
 * import { lookup } from 'mdml-engine/base'
 * import { rowCount } from 'mdml-engine/data'
 * import { datasetRegistry, loadDataset } from 'mdml-engine/datasets'
 * import { key } from 'mdml-engine/random'
 * rowCount(loadDataset(lookup(datasetRegistry, 'moons'), { key: key(0), knobs: { n: 10 } }).data) // => 10
 * loadDataset(lookup(datasetRegistry, 'iris')).data.kind // => 'classification'
 * loadDataset(lookup(datasetRegistry, 'californiaHousing')) // => throws value
 * ```
 */
export function loadDataset(entry: DatasetEntry, options: GenerateOptions = {}): Generated {
  switch (entry.form) {
    case 'generator':
      return generate(entry.generator, { ...options, knobs: { ...entry.knobs, ...options.knobs } })
    case 'vendored':
      return entry.load()
    default:
      throw new MdmlError('value', `loadDataset: '${entry.name}' is a remote dataset; use fetchDataset`)
  }
}

/**
 * Make, load or fetch an entry's data: as {@link loadDataset}, and a remote entry is downloaded (through the cache
 * given, verified against its hash) and parsed.
 *
 * @param entry - The entry.
 * @param options - Knobs, key or context for a generator; cache and fetch for a remote asset.
 * @returns The data with its truth (empty for real data).
 * @throws MdmlError with code `value` when a download fails or its checksum differs.
 * @concept datasets
 * @example
 * ```ts
 * import { lookup } from 'mdml-engine/base'
 * import { datasetRegistry, fetchDataset } from 'mdml-engine/datasets'
 * (await fetchDataset(lookup(datasetRegistry, 'wine'))).data.kind // => 'classification'
 * ```
 */
export async function fetchDataset(
  entry: DatasetEntry,
  options: GenerateOptions & LoadOptions = {},
): Promise<Generated> {
  if (entry.form !== 'remote') return loadDataset(entry, options)
  const { cache, fetch: fetcher } = options
  return asGenerated(
    await load(entry.ref, {
      ...(cache === undefined ? {} : { cache }),
      ...(fetcher === undefined ? {} : { fetch: fetcher }),
    }),
  )
}
