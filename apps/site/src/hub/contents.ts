import { catalogue, type HubEntry } from 'whitebox-hub'
import type { PageRef } from '../shell/PrevNext.tsx'
import type { TocNode } from '../shell/toc.ts'

/** The hub's front page, first in reading order. */
export const HUB_HOME: PageRef = { title: 'Hub', path: '/hub' }

/** The page of an entry. */
export const entryPath = (entry: HubEntry): string => `/hub/${entry.id}`

/** The page of a collection (`sklearn/linear-model`). */
export const collectionPath = (collection: string): string => `/hub/${collection}`

interface Folder {
  readonly path: string
  readonly folders: Map<string, Folder>
  readonly entries: HubEntry[]
}

const root: Folder = { path: '', folders: new Map(), entries: [] }
for (const e of catalogue.entries) {
  let at = root
  for (const part of e.collection.split('/')) {
    const path = at.path === '' ? part : `${at.path}/${part}`
    let next = at.folders.get(part)
    if (next === undefined) {
      next = { path, folders: new Map(), entries: [] }
      at.folders.set(part, next)
    }
    at = next
  }
  at.entries.push(e)
}

function toNode(name: string, f: Folder): TocNode {
  return {
    id: `hub:${f.path}`,
    title: name,
    children: [
      ...[...f.folders].sort(([a], [b]) => a.localeCompare(b)).map(([n, sub]) => toNode(n, sub)),
      ...f.entries.map((e) => ({ id: e.id, title: e.title, path: entryPath(e) })),
    ],
  }
}

/** The hub's contents as its collection folders (as on disk): each folder with its sub-collections, then its entries. */
export const HUB_TOC: readonly TocNode[] = [...root.folders]
  .sort(([a], [b]) => a.localeCompare(b))
  .map(([n, f]) => toNode(n, f))

/** Every collection with entries of its own, sorted (`sklearn/cluster`, `sklearn/decomposition`, ...). */
export const COLLECTIONS: readonly string[] = [...catalogue.collections.keys()]

/** Whether a collection path names a folder of the tree (a collection or a parent of collections). */
export function isCollection(path: string): boolean {
  return COLLECTIONS.some((c) => c === path || c.startsWith(`${path}/`))
}

/** The entries of a collection and every collection below it, grouped by collection. */
export function groupsUnder(path: string): { readonly collection: string; readonly entries: readonly HubEntry[] }[] {
  return COLLECTIONS.filter((c) => path === '' || c === path || c.startsWith(`${path}/`)).map((c) => ({
    collection: c,
    entries: catalogue.collections.get(c) ?? [],
  }))
}
