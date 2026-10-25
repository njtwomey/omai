/**
 * The pure logic behind the dataset picker: finding a collection, the knob space of an entry's generator, the selection
 * of an entry with its demo knobs, and generating or loading the selected dataset with its generating truth.
 */

import type { Dataset, Generated, Knobs } from 'mdml-engine/data'
import {
  type CollectionEntry,
  type DatasetCollection,
  datasetCollections,
  datasetGenerators,
  loadEntry,
} from 'mdml-engine/datasets'
import { key } from 'mdml-engine/random'
import { type DictSpace, defaults } from 'mdml-engine/space'

/** What a dataset picker has chosen: an entry of a collection, the seed it is generated with, and its knobs. */
export interface DatasetSelection {
  readonly name: string
  readonly seed: number
  /** The generator's knob values (empty for a vendored dataset). */
  readonly knobs: Knobs
}

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
 * The knob space of a collection entry: its generator's knobs, or `undefined` for a vendored dataset.
 *
 * @param entry - The entry.
 * @returns The knob space.
 */
export function entryKnobSpace(entry: CollectionEntry): DictSpace | undefined {
  return 'generator' in entry ? datasetGenerators[entry.generator]?.knobs : undefined
}

/**
 * The selection of one entry with its demo knobs (the rest at the generator's defaults).
 *
 * @param collection - The collection.
 * @param name - The entry's name (default the first entry).
 * @param seed - The seed (default 0).
 * @returns The selection.
 * @throws Error when the collection has no such entry.
 */
export function selectEntry(collection: string | DatasetCollection, name?: string, seed = 0): DatasetSelection {
  const c = collectionOf(collection)
  const entry = name === undefined ? c.entries[0] : c.entries.find((e) => e.name === name)
  if (entry === undefined) throw new Error(`collection '${c.id}' has no entry '${name ?? ''}'`)
  const space = entryKnobSpace(entry)
  const knobs =
    space === undefined ? {} : { ...(defaults(space) as Knobs), ...('knobs' in entry ? (entry.knobs ?? {}) : {}) }
  return { name: entry.name, seed, knobs }
}

/**
 * Generate (or load) the selected dataset, with the generating truth for generated entries.
 *
 * @param collection - The collection.
 * @param selection - The selection.
 * @returns The data and, for generated entries, its truth.
 * @throws Error when the collection has no such entry.
 */
export function loadSelection(
  collection: string | DatasetCollection,
  selection: DatasetSelection,
): { readonly data: Dataset } & Partial<Generated> {
  const c = collectionOf(collection)
  const entry = c.entries.find((e) => e.name === selection.name)
  if (entry === undefined) throw new Error(`collection '${c.id}' has no entry '${selection.name}'`)
  return loadEntry(entry, key(selection.seed), selection.knobs)
}
