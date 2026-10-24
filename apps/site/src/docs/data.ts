/**
 * The docs catalogue in the browser, arranged as the engine's source tree: folders (`lib/numerics/linalg`), files
 * (`lib/numerics/linalg/cholesky.ts`) and the symbols each file exports. The index loads once, lazily; a file's full
 * symbol docs load with its module's chunk.
 */
import { useEffect, useState } from 'react'
import type { PackageDoc, SymbolDoc, SymbolKind } from 'whitebox-meta/schema'

/** An export as the index lists it. */
export interface IndexSymbol {
  readonly name: string
  readonly kind: SymbolKind
  readonly summary: string
}

/** A source file: its path below `packages/engine/src`, the entry point it belongs to, its comment and exports. */
export interface IndexFile {
  readonly path: string
  readonly module: string
  readonly summary: string
  readonly description: string
  readonly symbols: readonly IndexSymbol[]
}

/** An entry-point folder with its summary (from `domain.json`, `area.json` or its index). */
export interface IndexFolder {
  readonly path: string
  readonly module: string
  readonly summary: string
}

/** What the plugin serves. */
export interface DocsIndex {
  readonly packages: readonly PackageDoc[]
  readonly folders: readonly IndexFolder[]
  readonly files: readonly IndexFile[]
}

/** A folder of the source tree with what it holds. */
export interface Folder {
  /** `''` for the package root, else `lib`, `lib/numerics`, `methods/linear`, ... */
  readonly path: string
  readonly name: string
  readonly summary: string
  /** The entry point it is, when it is one (`whitebox-engine/numerics/linalg`). */
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
}

/** The engine's source folder, as catalogue paths name it. */
export const SRC = 'packages/engine/src/'

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
        name: path === '' ? 'whitebox-engine' : baseName(path),
        summary: meta?.summary ?? '',
        ...(meta === undefined ? {} : { module: meta.module }),
        folders: [...subs].sort(byPath),
        files: [...(filesIn.get(path) ?? [])].sort((a, b) => byPath(a.path, b.path)),
      }
    },
    file: (path) => files.get(path),
    find: (name, nearModule) => {
      const all = byName.get(name) ?? []
      return all.find((s) => s.file.module === nearModule) ?? all.find((s) => s.file.path.startsWith('lib/')) ?? all[0]
    },
  }
}

let docs: Docs | null = null
let loading: Promise<Docs> | null = null

/** The docs index, or `null` while it loads. */
export function useDocs(): Docs | null {
  const [value, setValue] = useState(docs)
  useEffect(() => {
    if (value !== null) return
    let live = true
    loading ??= import('virtual:whitebox-docs/index').then((m) => {
      docs = withLookups(m.default)
      return docs
    })
    void loading.then((d) => {
      if (live) setValue(d)
    })
    return () => {
      live = false
    }
  }, [value])
  return value
}

const moduleCache = new Map<string, readonly SymbolDoc[]>()

/** The full symbol docs of one file (from its module's chunk), in source order, or `null` while they load. */
export function useFileSymbols(file: IndexFile | undefined): readonly SymbolDoc[] | null {
  const module = file?.module
  const [symbols, setSymbols] = useState<readonly SymbolDoc[] | null>(
    module === undefined ? null : (moduleCache.get(module) ?? null),
  )
  useEffect(() => {
    if (module === undefined) return
    const had = moduleCache.get(module)
    if (had !== undefined) {
      setSymbols(had)
      return
    }
    setSymbols(null)
    let live = true
    void import('virtual:whitebox-docs/modules').then(async (m) => {
      const load = m.default[module]
      const loaded = load === undefined ? [] : (await load()).default
      moduleCache.set(module, loaded)
      if (live) setSymbols(loaded)
    })
    return () => {
      live = false
    }
  }, [module])
  if (symbols === null || file === undefined) return null
  return symbols.filter((s) => s.source.file === `${SRC}${file.path}`)
}
