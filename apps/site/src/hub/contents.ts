import { catalogue, type HubEntry } from 'mdml-hub'
import type { PageRef } from '../shell/PrevNext.tsx'
import type { TocNode } from '../shell/toc.ts'
import { PAGES } from './standalone.ts'

/** The hub's front page, first in reading order. */
export const HUB_HOME: PageRef = { title: 'Hub', path: '/hub' }

/** The page of an entry. */
export const entryPath = (entry: HubEntry): string => `/hub/${entry.id}`

/** The page of a collection (`sklearn/linear-model`). */
export const collectionPath = (collection: string): string => `/hub/${collection}`

/** What a collection lists: a catalogue entry or a standalone page, with its card's text and its page. */
export interface HubItem {
  readonly id: string
  readonly title: string
  readonly text: string
  readonly path: string
}

const itemOf = (e: HubEntry): HubItem => ({ id: e.id, title: e.title, text: e.summary, path: entryPath(e) })

interface Folder {
  readonly path: string
  readonly folders: Map<string, Folder>
  readonly entries: HubItem[]
}

const root: Folder = { path: '', folders: new Map(), entries: [] }
const items: { readonly collection: string; readonly item: HubItem }[] = [
  ...catalogue.entries.map((e) => ({ collection: e.collection, item: itemOf(e) })),
  ...PAGES.map((p) => ({
    collection: p.collection,
    item: { id: p.id, title: p.title, text: p.question, path: p.path },
  })),
]
for (const { collection, item } of items) {
  let at = root
  for (const part of collection.split('/')) {
    const path = at.path === '' ? part : `${at.path}/${part}`
    let next = at.folders.get(part)
    if (next === undefined) {
      next = { path, folders: new Map(), entries: [] }
      at.folders.set(part, next)
    }
    at = next
  }
  at.entries.push(item)
}

function toNode(name: string, f: Folder): TocNode {
  return {
    id: `hub:${f.path}`,
    title: name,
    children: [
      ...[...f.folders].sort(([a], [b]) => a.localeCompare(b)).map(([n, sub]) => toNode(n, sub)),
      ...f.entries.map((e) => ({ id: e.id, title: e.title, path: e.path })),
    ],
  }
}

/** The hub's contents as its collection folders (as on disk): each folder with its sub-collections, then its entries. */
export const HUB_TOC: readonly TocNode[] = [...root.folders]
  .sort(([a], [b]) => a.localeCompare(b))
  .map(([n, f]) => toNode(n, f))

/** Every collection with entries or pages of its own, sorted (`sklearn/cluster`, `sklearn/decomposition`, ...). */
export const COLLECTIONS: readonly string[] = [...new Set(items.map((i) => i.collection))].sort()

/** Whether a collection path names a folder of the tree (a collection or a parent of collections). */
export function isCollection(path: string): boolean {
  return COLLECTIONS.some((c) => c === path || c.startsWith(`${path}/`))
}

/** The entries and pages of a collection and every collection below it, grouped by collection. */
export function groupsUnder(path: string): { readonly collection: string; readonly entries: readonly HubItem[] }[] {
  return COLLECTIONS.filter((c) => path === '' || c === path || c.startsWith(`${path}/`)).map((c) => ({
    collection: c,
    entries: items.filter((i) => i.collection === c).map((i) => i.item),
  }))
}

/** How many items the hub lists: catalogue entries and standalone pages. */
export const ITEM_COUNT: number = items.length
