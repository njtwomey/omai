/**
 * The docs catalogue in the browser, arranged as the engine's source tree: folders (`lib/numerics/linalg`), files
 * (`lib/numerics/linalg/cholesky.ts`) and the symbols each file exports. It comes from the precompiled files of
 * `make docs-catalogue` (wire.ts): the index (names and kinds only) once, lazily; a file's full docs when a page shows
 * it; every export's summary only when someone searches. On the dev server, an edit to the engine sends a new index.
 */

import type { PackageDoc, SymbolKind } from 'mdml-meta/schema'
import { useEffect, useState } from 'react'
import type { WireFileData, WireIndex, WireSearch, WireTypes } from './wire.ts'

/** An export as the index lists it. */
export interface IndexSymbol {
  readonly name: string
  readonly kind: SymbolKind
}

/** A source file: its path below `packages/engine/src`, its entry point, its opening sentence and its exports. */
export interface IndexFile {
  readonly path: string
  readonly module: string
  readonly summary: string
  /** The URL of its data (full comment and symbol docs). */
  readonly data: string
  readonly symbols: readonly IndexSymbol[]
}

/** A folder with its summary (the doc comment that opens its `index.ts`); an entry point's names its module. */
export interface IndexFolder {
  readonly path: string
  readonly module?: string
  readonly summary: string
  /** The URL of its types file (declarations for the hover on a type name), when it exports types. */
  readonly types?: string
}

/** The index as the pages use it. */
export interface DocsIndex {
  readonly version: string
  /** The URL of the search index. */
  readonly search: string
  readonly packages: readonly PackageDoc[]
  readonly folders: readonly IndexFolder[]
  readonly files: readonly IndexFile[]
}

/** The index as fetched, with URLs resolved against the index's own. */
export function fromWire(wire: WireIndex, url: string): DocsIndex {
  const dir = url.slice(0, url.lastIndexOf('/') + 1)
  return {
    version: wire.version,
    search: dir + wire.search,
    packages: wire.packages,
    folders: wire.folders.map((f) => (f.types === undefined ? f : { ...f, types: dir + f.types })),
    files: wire.files.map((f) => ({
      path: f.path,
      module: f.module,
      summary: f.summary,
      data: dir + f.data,
      symbols: f.symbols.map(([name, kind]) => ({ name, kind })),
    })),
  }
}

/** A folder of the source tree with what it holds. */
export interface Folder {
  /** `''` for the package root, else `lib`, `lib/numerics`, `methods/linear`, ... */
  readonly path: string
  readonly name: string
  readonly summary: string
  /** The entry point it is, when it is one (`mdml-engine/numerics/linalg`). */
  readonly module?: string
  readonly folders: readonly string[]
  readonly files: readonly IndexFile[]
}

/** The index with lookups. */
export interface Docs extends DocsIndex {
  readonly folder: (path: string) => Folder | undefined
  readonly file: (path: string) => IndexFile | undefined
  /** Where a symbol of this name is defined, preferring the given file's module, for `{@link name}`. */
  readonly find: (name: string, nearModule?: string) => { readonly file: IndexFile; readonly name: string } | undefined
  /** The export of a module with this name. */
  readonly symbol: (module: string, name: string) => { readonly file: IndexFile; readonly name: string } | undefined
  /** The URL of a module's types file. */
  readonly typesUrl: (module: string) => string | undefined
}

/** The URL of a folder or file path of the tree. */
export function docsPath(path: string): string {
  return path === '' ? '/docs' : `/docs/${path}`
}

/** The URL of a symbol: its file's URL and its name. */
export function symbolPath(file: string, name: string): string {
  return `/docs/${file}/${name}`
}

/** The last segment of a path. */
export function baseName(path: string): string {
  return path.slice(path.lastIndexOf('/') + 1)
}

function parentOf(path: string): string {
  const cut = path.lastIndexOf('/')
  return cut < 0 ? '' : path.slice(0, cut)
}

/** The index with its lookups (folders, files, symbols by name). */
export function withLookups(index: DocsIndex): Docs {
  const files = new Map(index.files.map((f) => [f.path, f]))
  const info = new Map(index.folders.map((f) => [f.path, f]))
  const subfolders = new Map<string, Set<string>>([['', new Set()]])
  const filesIn = new Map<string, IndexFile[]>()
  const addFolder = (path: string): void => {
    if (subfolders.has(path)) return
    subfolders.set(path, new Set())
    const parent = parentOf(path)
    addFolder(parent)
    subfolders.get(parent)?.add(path)
  }
  for (const f of index.folders) addFolder(f.path)
  for (const f of index.files) {
    const dir = parentOf(f.path)
    addFolder(dir)
    filesIn.set(dir, [...(filesIn.get(dir) ?? []), f])
  }
  const byName = new Map<string, { file: IndexFile; name: string }[]>()
  const inModule = new Map<string, { file: IndexFile; name: string }>()
  for (const f of index.files)
    for (const s of f.symbols) inModule.set(`${f.module}#${s.name}`, { file: f, name: s.name })
  const types = new Map(index.folders.flatMap((f) => (f.module === undefined ? [] : [[f.module, f.types] as const])))
  for (const f of index.files)
    for (const s of f.symbols) byName.set(s.name, [...(byName.get(s.name) ?? []), { file: f, name: s.name }])
  const byPath = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0)
  return {
    ...index,
    folder: (path) => {
      const subs = subfolders.get(path)
      if (subs === undefined) return undefined
      const meta = info.get(path)
      return {
        path,
        name: path === '' ? 'mdml-engine' : baseName(path),
        summary: meta?.summary ?? '',
        ...(meta?.module === undefined ? {} : { module: meta.module }),
        folders: [...subs].sort(byPath),
        files: [...(filesIn.get(path) ?? [])].sort((a, b) => byPath(a.path, b.path)),
      }
    },
    file: (path) => files.get(path),
    symbol: (module, name) => inModule.get(`${module}#${name}`),
    typesUrl: (module) => types.get(module),
    find: (name, nearModule) => {
      const all = byName.get(name) ?? []
      return all.find((s) => s.file.module === nearModule) ?? all.find((s) => s.file.path.startsWith('lib/')) ?? all[0]
    },
  }
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: ${res.status}`)
  return (await res.json()) as T
}

let docs: Docs | null = null
let started = false
const listeners = new Set<(d: Docs) => void>()

function loadIndex(url: string): Promise<void> {
  return fetchJson<WireIndex>(url).then((wire) => {
    docs = withLookups(fromWire(wire, url))
    for (const l of listeners) l(docs)
  })
}

// On the dev server, the plugin sends the new index's URL after regenerating the files for an edit to the engine.
if (import.meta.hot)
  import.meta.hot.on('mdml-docs:update', (data: { readonly index: string }) => {
    void loadIndex(data.index)
  })

/** The docs index, or `null` while it loads. */
export function useDocs(): Docs | null {
  const [value, setValue] = useState(docs)
  useEffect(() => {
    listeners.add(setValue)
    if (docs !== null) setValue(docs)
    else if (!started) {
      started = true
      void import('virtual:mdml-docs/index').then((m) =>
        loadIndex(m.default).catch(() => {
          started = false
        }),
      )
    }
    return () => {
      listeners.delete(setValue)
    }
  }, [])
  return value
}

// Content-addressed URLs: a file's data never changes under the same URL. Loaded data is also kept by value, so a
// page seen before draws at once.
const fileCache = new Map<string, Promise<WireFileData>>()
const fileData = new Map<string, WireFileData>()

/** One file's full comment and symbol docs, in source order, or `null` while they load. */
export function useFileData(file: IndexFile | undefined): WireFileData | null {
  const url = file?.data
  const [data, setData] = useState<{ readonly url: string; readonly value: WireFileData } | null>(null)
  useEffect(() => {
    if (url === undefined || fileData.has(url)) return
    let live = true
    let p = fileCache.get(url)
    if (p === undefined) {
      p = fetchJson<WireFileData>(url).then((value) => {
        fileData.set(url, value)
        return value
      })
      fileCache.set(url, p)
      p.catch(() => fileCache.delete(url))
    }
    void p.then((value) => {
      if (live) setData({ url, value })
    })
    return () => {
      live = false
    }
  }, [url])
  if (url === undefined) return null
  return fileData.get(url) ?? (data !== null && data.url === url ? data.value : null)
}

const searchCache = new Map<string, Promise<WireSearch>>()

/** Start fetching the search index (every export's summary) of an index. */
export function loadSearch(index: DocsIndex): Promise<WireSearch> {
  let p = searchCache.get(index.search)
  if (p === undefined) {
    p = fetchJson<WireSearch>(index.search)
    searchCache.set(index.search, p)
    p.catch(() => searchCache.delete(index.search))
  }
  return p
}

/**
 * The search index of an index once `wanted` (someone started searching), or after the page is idle; `null` until it
 * has loaded.
 */
export function useSearch(index: DocsIndex, wanted: boolean): WireSearch | null {
  const [search, setSearch] = useState<{ readonly url: string; readonly value: WireSearch } | null>(null)
  useEffect(() => {
    let live = true
    const start = (): void => {
      void loadSearch(index).then((value) => {
        if (live) setSearch({ url: index.search, value })
      })
    }
    if (wanted) start()
    else if (typeof requestIdleCallback === 'function') {
      const id = requestIdleCallback(start, { timeout: 5000 })
      return () => {
        live = false
        cancelIdleCallback(id)
      }
    }
    return () => {
      live = false
    }
  }, [index, wanted])
  return search !== null && search.url === index.search ? search.value : null
}

const typesCache = new Map<string, Promise<WireTypes>>()

/** A module's types file (fetched once, when first wanted), or `null` while it loads or when `url` is undefined. */
export function useTypes(url: string | undefined): WireTypes | null {
  const [types, setTypes] = useState<{ readonly url: string; readonly value: WireTypes } | null>(null)
  useEffect(() => {
    if (url === undefined) return
    let live = true
    let p = typesCache.get(url)
    if (p === undefined) {
      p = fetchJson<WireTypes>(url)
      typesCache.set(url, p)
      p.catch(() => typesCache.delete(url))
    }
    void p.then((value) => {
      if (live) setTypes({ url, value })
    })
    return () => {
      live = false
    }
  }, [url])
  return types !== null && types.url === url ? types.value : null
}
