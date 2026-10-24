/** The docs catalogue in the browser: the index (loaded once, lazily) and each module's symbols (loaded per page). */
import { useEffect, useState } from 'react'
import type { ModuleDoc, PackageDoc, SymbolDoc, SymbolKind } from 'whitebox-meta/schema'

/** A symbol as the index lists it. */
export interface SymbolSummary {
  readonly id: string
  readonly module: string
  readonly name: string
  readonly kind: SymbolKind
  readonly summary: string
}

/** Everything the sidebar, home, module pages and search need. */
export interface DocsIndex {
  readonly packages: readonly PackageDoc[]
  readonly modules: readonly ModuleDoc[]
  readonly symbols: readonly SymbolSummary[]
}

/** The index with lookups. */
export interface Docs extends DocsIndex {
  readonly module: (id: string) => ModuleDoc | undefined
  readonly symbolsOf: (module: string) => readonly SymbolSummary[]
  /** A symbol by name, preferring the given module, for `{@link name}`. */
  readonly find: (name: string, near?: string) => SymbolSummary | undefined
}

const PACKAGE = 'whitebox-engine'

/** The URL path of a module (`/docs/numerics/linalg`). */
export function modulePath(id: string): string {
  return id === PACKAGE ? '/docs' : `/docs/${id.slice(PACKAGE.length + 1)}`
}

/** The URL path of a symbol (`/docs/numerics/linalg/cholesky`). */
export function symbolPath(s: Pick<SymbolSummary, 'module' | 'name'>): string {
  return `${modulePath(s.module)}/${s.name}`
}

function withLookups(index: DocsIndex): Docs {
  const modules = new Map(index.modules.map((m) => [m.id, m]))
  const byModule = new Map<string, SymbolSummary[]>()
  const byName = new Map<string, SymbolSummary[]>()
  for (const s of index.symbols) {
    byModule.set(s.module, [...(byModule.get(s.module) ?? []), s])
    byName.set(s.name, [...(byName.get(s.name) ?? []), s])
  }
  return {
    ...index,
    module: (id) => modules.get(id),
    symbolsOf: (m) => byModule.get(m) ?? [],
    find: (name, near) => {
      const all = byName.get(name) ?? []
      return all.find((s) => s.module === near) ?? all.find((s) => !s.module.includes('/methods/')) ?? all[0]
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

/** A module's full symbol docs, or `null` while they load. */
export function useModuleSymbols(module: string | undefined): readonly SymbolDoc[] | null {
  const [value, setValue] = useState<readonly SymbolDoc[] | null>(
    module === undefined ? null : (moduleCache.get(module) ?? null),
  )
  useEffect(() => {
    if (module === undefined) return
    const had = moduleCache.get(module)
    if (had !== undefined) {
      setValue(had)
      return
    }
    setValue(null)
    let live = true
    void import('virtual:whitebox-docs/modules').then(async (m) => {
      const load = m.default[module]
      const symbols = load === undefined ? [] : (await load()).default
      moduleCache.set(module, symbols)
      if (live) setValue(symbols)
    })
    return () => {
      live = false
    }
  }, [module])
  return value
}
