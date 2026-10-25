/**
 * The hub entry contract: a `HubEntry` names the engine method it demonstrates, the dataset collections it suits, its
 * controls as a parameter space, a `run` that produces exposition records, and view slots saying which render view
 * shows each record and where. `defineEntry` checks an entry as it is declared, and `buildCatalogue` gathers the
 * entries of many modules into the catalogue, by id and by collection.
 */

import { WhiteboxError } from 'whitebox-engine/base'
import type { Dataset, Knobs } from 'whitebox-engine/data'
import { datasetCollections, datasetGenerators } from 'whitebox-engine/datasets'
import type { Explained, Exposition, ExpositionKind } from 'whitebox-engine/exposition'
import type { Key } from 'whitebox-engine/random'
import { contains, type DictSpace, type SpaceValue } from 'whitebox-engine/space'

/** Where a view sits on an entry's page. */
export type ViewArea = 'main' | 'side' | 'below'

/** What a slot's `props` receives beside its record: the run's input rows, labels or targets, and parameters. */
export interface SlotContext {
  readonly points?: readonly (readonly number[])[]
  /** Class indices, for classification data. */
  readonly labels?: readonly number[]
  /** Targets, for regression data (a Swiss roll's position along the roll, say). */
  readonly targets?: readonly number[]
  readonly params: Readonly<Record<string, SpaceValue>>
}

/** One view on an entry's page: which exposition record it shows, with which render view, where. */
export interface ViewSlot {
  readonly record: ExpositionKind
  /**
   * A component of `whitebox-render/react` by name (`EpisodePlayer`, `EnvView`, a record view); omitted, the record
   * kind's standard view (`ExpositionView`).
   */
  readonly view?: string
  /**
   * Props for the named view beyond `record` and `figure`, made on the page from the record (an environment and a
   * policy for an `EpisodePlayer`, say: objects that cannot cross from a worker).
   */
  readonly props?: (record: Exposition, context: SlotContext) => Readonly<Record<string, unknown>>
  readonly area?: ViewArea
  readonly title?: string
  /** Which record of that kind, when the method returns several (default 0). */
  readonly index?: number
}

/** How far a long run has got, with the records so far (to draw a method learning live). */
export interface HubProgress {
  /** The fraction done, from 0 to 1. */
  readonly fraction: number
  /** A line saying where the run is (`episode 40 of 200`). */
  readonly label?: string
  /** The records so far (they replace the shown records until the run ends). */
  readonly exposition?: readonly Exposition[]
}

/** What an entry's `run` receives: the chosen dataset, the parameter values, a key, and where to report progress. */
export interface HubRunInput {
  /** The dataset (`undefined` for entries with no dataset collections, such as agents in their environments). */
  readonly data: Dataset | undefined
  readonly params: Readonly<Record<string, SpaceValue>>
  readonly key: Key
  /**
   * Report progress (long runs that learn live): the widget shows the fraction and draws the records so far. Reports
   * are throttled by the caller, so a run may call it every step.
   */
  readonly progress?: (p: HubProgress) => void
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
  /**
   * Dataset collection ids (`whitebox-engine/datasets`' `datasetCollections`); the first is the default. Empty for a
   * method that needs no dataset (an agent learning in an environment chosen by its parameters).
   */
  readonly datasets: readonly string[]
  /** The dataset to start from, by name in the first collection (default its first). */
  readonly dataset?: string
  /**
   * Knobs of that dataset to start from, over the collection's demo knobs (more rows of a Swiss roll, say, or circles
   * further apart).
   */
  readonly knobs?: Knobs
  /** Fit on at most this many rows (a keyed subsample), for methods too slow to run live on more. */
  readonly maxRows?: number
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
 * @throws WhiteboxError `value` when the id is not under its collection, a dataset collection is unknown, or a knob
 *   does not fit the starting dataset.
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
  for (const d of spec.datasets)
    if (datasetCollections[d] === undefined)
      throw new WhiteboxError('value', `defineEntry: ${spec.id} names an unknown dataset collection '${d}'`)
  const first = datasetCollections[spec.datasets[0] ?? '']
  if (spec.dataset !== undefined && first?.entries.every((e) => e.name !== spec.dataset) === true)
    throw new WhiteboxError('value', `defineEntry: ${spec.id} starts from an unknown dataset '${spec.dataset}'`)
  if (spec.knobs !== undefined) {
    const start = first?.entries.find((e) => e.name === (spec.dataset ?? first.entries[0]?.name))
    const space = start !== undefined && 'generator' in start ? datasetGenerators[start.generator]?.knobs : undefined
    for (const [k, v] of Object.entries(spec.knobs)) {
      const s = space?.spaces[k]
      if (s === undefined || !contains(s, v))
        throw new WhiteboxError('value', `defineEntry: ${spec.id} starts from a knob '${k}' its dataset does not take`)
    }
  }
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
