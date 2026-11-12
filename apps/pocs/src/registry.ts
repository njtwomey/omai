import { type ComponentType, lazy } from 'react'
import { buildTree, dirOfKey, locate, type Poc, type PocDir, type PocEntry } from './poc.ts'

// A tree of folders under `pocs/`: a `dir.ts` per directory, and one folder per proof of concept holding `poc.ts` (metadata,
// loaded now) and `Page.tsx` (the page, loaded when opened). A folder directly under `pocs/` is a page not yet moved into
// its directory: it is listed in the directory its legacy `group` names, and in NOT_YET_MOVED.
const metas = import.meta.glob<{ readonly poc: Poc }>('./pocs/**/poc.ts', { eager: true })
const dirMetas = import.meta.glob<{ readonly dir: PocDir }>('./pocs/**/dir.ts', { eager: true })
const pages = import.meta.glob<{ readonly default: ComponentType }>('./pocs/**/Page.tsx')

const folderOf = (key: string): string => key.split('/').slice(0, -1).join('/')

const located = Object.entries(metas).map(([key, m]) => ({ key, poc: m.poc, at: locate(key, m.poc.group) }))

/** Every proof of concept, in the order of their folders. */
export const POCS: readonly PocEntry[] = located.map(({ poc, at }) => ({ ...poc, slug: at.slug, dir: at.dir }))

/** The slugs of the pages that still sit directly under `pocs/` (listed by the legacy `group` they name), to be moved. */
export const NOT_YET_MOVED: readonly string[] = located.filter(({ at }) => at.flat).map(({ at }) => at.slug)

/** The directory metadata found, keyed by the directory path joined with `/`. */
export const DIR_METAS: Readonly<Record<string, PocDir>> = Object.fromEntries(
  Object.entries(dirMetas).map(([key, m]) => [dirOfKey(key).join('/'), m.dir]),
)

/** The tree of directories and pocs: the root is the front page. */
export const TREE = buildTree(POCS, DIR_METAS)

const loaders = new Map(Object.entries(pages).map(([key, load]) => [locate(key).slug, load]))
const components = new Map<string, ComponentType>()

/** The page of a proof of concept (a lazy component, the same one every time), or undefined for an unknown slug. */
export function pageOf(slug: string): ComponentType | undefined {
  const load = loaders.get(slug)
  if (load === undefined) return undefined
  let c = components.get(slug)
  if (c === undefined) {
    c = lazy(load)
    components.set(slug, c)
  }
  return c
}

/** Folders that have a metadata file but no page, or the other way round: a mistake to be caught by the test. */
export function mismatched(): string[] {
  const m = new Set(Object.keys(metas).map(folderOf))
  const p = new Set(Object.keys(pages).map(folderOf))
  return [...m].filter((s) => !p.has(s)).concat([...p].filter((s) => !m.has(s)))
}
