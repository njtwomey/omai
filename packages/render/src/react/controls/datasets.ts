/**
 * The pure logic behind the dataset picker and selector: the entries a source offers (a collection by id, a
 * collection, or a tagged dataset registry), the knob space of an entry's generator, the selection of an entry with
 * its demo knobs, and generating or loading the selected dataset with its generating truth.
 */

import type { Dataset, Generated, Knobs } from 'mdml-engine/data'
import {
  type CollectionEntry,
  type DatasetCollection,
  type DatasetEntry,
  type DatasetRegistry,
  datasetCollections,
  datasetGenerators,
  loadDataset,
  loadEntry,
} from 'mdml-engine/datasets'
import { key } from 'mdml-engine/random'
import { type DictSpace, defaults } from 'mdml-engine/space'

/** What a dataset picker has chosen: an entry of its source, the seed it is generated with, and its knobs. */
export interface DatasetSelection {
  readonly name: string
  readonly seed: number
  /** The generator's knob values (empty for a vendored dataset). */
  readonly knobs: Knobs
}

/** Where a picker's entries come from: a collection id, a collection, or a dataset registry (tagged entries). */
export type DatasetSource = string | DatasetCollection | DatasetRegistry

/** An entry a picker can offer: a collection's, or a registry's. */
export type PickableEntry = CollectionEntry | DatasetEntry

/**
 * A collection by id or as given.
 *
 * @param collection - A collection id in `datasetCollections`, or a collection.
 * @returns The collection.
 * @throws Error when the id names no collection.
 */
export function collectionOf(collection: string | DatasetCollection): DatasetCollection {
  if (typeof collection !== 'string') return collection
  const found = datasetCollections[collection]
  if (found === undefined) throw new Error(`no dataset collection '${collection}'`)
  return found
}

/**
 * The entries a source offers, in its order.
 *
 * @param source - A collection id, a collection or a registry.
 * @returns Its entries.
 * @throws Error when a collection id names no collection.
 */
export function entriesOf(source: DatasetSource): readonly PickableEntry[] {
  if (typeof source !== 'string' && !('id' in source)) return source.entries
  return collectionOf(source).entries
}

const sourceName = (source: DatasetSource): string =>
  typeof source === 'string' ? `collection '${source}'` : 'id' in source ? `collection '${source.id}'` : 'the registry'

/**
 * The knob space of an entry: its generator's knobs, or `undefined` for a vendored or remote dataset.
 *
 * @param entry - The entry.
 * @returns The knob space.
 */
export function entryKnobSpace(entry: PickableEntry): DictSpace | undefined {
  if ('form' in entry) return entry.form === 'generator' ? entry.generator.knobs : undefined
  return 'generator' in entry ? datasetGenerators[entry.generator]?.knobs : undefined
}

/** The knob values an entry starts from (a collection's demo knobs, a registry entry's starting knobs). */
function demoKnobs(entry: PickableEntry): Knobs {
  if ('form' in entry) return entry.form === 'generator' ? (entry.knobs ?? {}) : {}
  return 'generator' in entry ? (entry.knobs ?? {}) : {}
}

/**
 * The selection of one entry with its demo knobs (the rest at the generator's defaults).
 *
 * @param source - A collection id, a collection or a registry.
 * @param name - The entry's name (default the first entry).
 * @param seed - The seed (default 0).
 * @returns The selection.
 * @throws Error when the source has no such entry.
 */
export function selectEntry(source: DatasetSource, name?: string, seed = 0): DatasetSelection {
  const entries = entriesOf(source)
  const entry = name === undefined ? entries[0] : entries.find((e) => e.name === name)
  if (entry === undefined) throw new Error(`${sourceName(source)} has no entry '${name ?? ''}'`)
  const space = entryKnobSpace(entry)
  const knobs = space === undefined ? {} : { ...(defaults(space) as Knobs), ...demoKnobs(entry) }
  return { name: entry.name, seed, knobs }
}

/**
 * Generate (or load) the selected dataset, with the generating truth for generated entries.
 *
 * @param source - A collection id, a collection or a registry.
 * @param selection - The selection.
 * @returns The data and, for generated entries, its truth.
 * @throws Error when the source has no such entry, or the entry is a remote dataset (which loads asynchronously).
 */
export function loadSelection(
  source: DatasetSource,
  selection: DatasetSelection,
): { readonly data: Dataset } & Partial<Generated> {
  const entry = entriesOf(source).find((e) => e.name === selection.name)
  if (entry === undefined) throw new Error(`${sourceName(source)} has no entry '${selection.name}'`)
  if ('form' in entry) return loadDataset(entry, { key: key(selection.seed), knobs: selection.knobs })
  return loadEntry(entry, key(selection.seed), selection.knobs)
}
