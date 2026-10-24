import { WhiteboxError } from 'whitebox-engine/base'
import type { Dataset } from 'whitebox-engine/data'
import { datasetCollections } from 'whitebox-engine/datasets'
import type { Explained, ExpositionKind } from 'whitebox-engine/exposition'
import type { Key } from 'whitebox-engine/random'
import type { DictSpace, SpaceValue } from 'whitebox-engine/space'

/** Where a view sits on an entry's page. */
export type ViewArea = 'main' | 'side' | 'below'

/** One view on an entry's page: which exposition record it shows, with which render view, where. */
export interface ViewSlot {
  readonly record: ExpositionKind
  /** A render view name; omitted, the record kind's standard view. */
  readonly view?: string
  readonly area?: ViewArea
  readonly title?: string
  /** Which record of that kind, when the method returns several (default 0). */
  readonly index?: number
}

/** What an entry's `run` receives: the chosen dataset, the parameter values and a key. */
export interface HubRunInput {
  readonly data: Dataset
  readonly params: Readonly<Record<string, SpaceValue>>
  readonly key: Key
}

/** A method an entry runs: its source file below `packages/engine/src` and its exported name. */
export interface MethodRef {
  /** E.g. `methods/linear/regression.ts`. */
  readonly file: string
  /** E.g. `lasso`. */
  readonly name: string
}

/** A hub entry: a method, the datasets it suits, its controls and the views that explain it. Data, not drawing. */
export interface HubEntry {
  readonly kind: 'hubEntry'
  /** A path-like id, `<collection>/<name>`, e.g. `sklearn/linear-model/lasso`. */
  readonly id: string
  readonly title: string
  /** One line for cards and the catalogue. */
  readonly summary: string
  /** The hub collection the entry belongs to, e.g. `sklearn/linear-model`. */
  readonly collection: string
  readonly tags: readonly string[]
  /** Dataset collection ids (`whitebox-engine/datasets`' `datasetCollections`); the first is the default. */
  readonly datasets: readonly string[]
  /** The dataset to start from, by name in the first collection (default its first). */
  readonly dataset?: string
  /** The method's parameters as a space; its defaults are the demo defaults. */
  readonly params: DictSpace
  /** Run the method; the result carries the exposition records the views show. */
  readonly run: (input: HubRunInput) => Explained<unknown> | Promise<Explained<unknown>>
  readonly views: readonly ViewSlot[]
  /** The method symbols the entry runs, for links to their docs and source. */
  readonly methods?: readonly MethodRef[]
  readonly references?: readonly string[]
}

/**
 * Declare a hub entry, checking its id, collection and dataset collections.
 *
 * @param spec - The entry without its `kind`.
 * @returns The entry.
 * @throws WhiteboxError `value` when the id is not under its collection, or a dataset collection is unknown.
 * @example
 * ```ts
 * import { defineEntry } from 'whitebox-hub'
 * import { dict } from 'whitebox-engine/space'
 * const entry = defineEntry({
 *   id: 'demo/identity', title: 'Identity', summary: 'Returns nothing to explain.', collection: 'demo', tags: [],
 *   datasets: ['classification2d'], params: dict({}), run: () => ({ exposition: [] }), views: [],
 * })
 * entry.kind // => 'hubEntry'
 * ```
 */
export function defineEntry(spec: Omit<HubEntry, 'kind'>): HubEntry {
  if (!spec.id.startsWith(`${spec.collection}/`))
    throw new WhiteboxError('value', `defineEntry: id '${spec.id}' is not under its collection '${spec.collection}'`)
  if (spec.datasets.length === 0)
    throw new WhiteboxError('value', `defineEntry: ${spec.id} names no dataset collection`)
  for (const d of spec.datasets)
    if (datasetCollections[d] === undefined)
      throw new WhiteboxError('value', `defineEntry: ${spec.id} names an unknown dataset collection '${d}'`)
  const first = datasetCollections[spec.datasets[0] ?? '']
  if (spec.dataset !== undefined && first?.entries.every((e) => e.name !== spec.dataset) === true)
    throw new WhiteboxError('value', `defineEntry: ${spec.id} starts from an unknown dataset '${spec.dataset}'`)
  return { kind: 'hubEntry', ...spec }
}

/**
 * Whether a value is a hub entry.
 *
 * @param x - Any value.
 * @returns `true` for values made by {@link defineEntry}.
 * @example
 * ```ts
 * import { isHubEntry } from 'whitebox-hub'
 * isHubEntry({ kind: 'hubEntry' }) // => true
 * isHubEntry(3) // => false
 * ```
 */
export function isHubEntry(x: unknown): x is HubEntry {
  return typeof x === 'object' && x !== null && (x as { kind?: unknown }).kind === 'hubEntry'
}

/** The hub catalogue: every entry, by id and by collection. */
export interface Catalogue {
  readonly entries: readonly HubEntry[]
  readonly byId: ReadonlyMap<string, HubEntry>
  /** Collection to its entries, collections sorted. */
  readonly collections: ReadonlyMap<string, readonly HubEntry[]>
}

/**
 * Gather the hub entries exported by a list of modules into a catalogue (the generated `catalogue` calls this).
 *
 * @param modules - Module namespaces; every export that is a hub entry is collected.
 * @returns The catalogue, entries sorted by id.
 * @throws WhiteboxError `value` when two entries share an id.
 * @example
 * ```ts
 * import { buildCatalogue } from 'whitebox-hub'
 * buildCatalogue([]).entries.length // => 0
 * ```
 */
export function buildCatalogue(modules: readonly Readonly<Record<string, unknown>>[]): Catalogue {
  const byId = new Map<string, HubEntry>()
  for (const m of modules)
    for (const value of Object.values(m)) {
      if (!isHubEntry(value)) continue
      if (byId.has(value.id) && byId.get(value.id) !== value)
        throw new WhiteboxError('value', `buildCatalogue: two entries share the id '${value.id}'`)
      byId.set(value.id, value)
    }
  const entries = [...byId.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
  const collections = new Map<string, HubEntry[]>()
  for (const e of entries) collections.set(e.collection, [...(collections.get(e.collection) ?? []), e])
  return {
    entries,
    byId,
    collections: new Map([...collections].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))),
  }
}
