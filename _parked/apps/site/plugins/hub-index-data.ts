/**
 * The hub's index for the site: each catalogue entry's card and page header (id, title, summary, collection, tags,
 * method links) with the module that defines it (a key of `mdml-hub/lazy`'s `entryModules`), and the collection
 * descriptions. It is read by importing every entry module in Node once, so the hub's index pages show every entry
 * without loading any; an entry's page loads its own module. Run as a script
 * (`node --conditions=source apps/site/plugins/hub-index-data.ts`), it prints the index as JSON for hub-index.ts.
 */
import type { MethodRef } from 'mdml-hub'

/** What the hub's pages show of an entry before its module loads. */
export interface HubIndexEntry {
  readonly id: string
  readonly title: string
  readonly summary: string
  readonly collection: string
  readonly tags: readonly string[]
  readonly methods: readonly MethodRef[]
  /** The entry's module in `entryModules` (`sklearn/cluster/kmeans`). */
  readonly module: string
}

/** The hub's index: every entry, sorted by id, and the descriptions of the collections. */
export interface HubIndex {
  readonly entries: readonly HubIndexEntry[]
  readonly descriptions: Readonly<Record<string, string>>
}

/**
 * Build the index by importing each entry module (about 1.5 s).
 *
 * @returns The index.
 */
export async function hubIndex(): Promise<HubIndex> {
  const { entryModules } = await import('mdml-hub/lazy')
  const { collectionDescriptions, isHubEntry } = await import('mdml-hub/entry')
  const byId = new Map<string, HubIndexEntry>()
  for (const [module, load] of Object.entries(entryModules))
    for (const value of Object.values(await load())) {
      if (!isHubEntry(value) || byId.has(value.id)) continue
      const { id, title, summary, collection, tags, methods = [] } = value
      byId.set(id, { id, title, summary, collection, tags, methods, module })
    }
  const entries = [...byId.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
  return { entries, descriptions: collectionDescriptions }
}

if (import.meta.main) process.stdout.write(JSON.stringify(await hubIndex()))
