/**
 * The engine documentation as virtual modules, extracted from the source at build time by whitebox-meta/extract (a
 * few seconds for the whole engine), so the docs pages are generated and cannot drift from the code.
 *
 * - `virtual:whitebox-docs/index`: the source tree as data: every entry-point folder with its summary, every source
 *   file (path below `packages/engine/src`) with its opening comment and its exports' names, kinds and one-line
 *   summaries, in source order. The sidebar, folder pages and file indexes come from it.
 * - `virtual:whitebox-docs/modules`: a map from module id to a lazy import of that module's full `SymbolDoc`s, so a
 *   page loads only the module it shows.
 */
import { join } from 'node:path'
import type { Plugin } from 'vite'
import { extractCatalogue } from '../../../packages/meta/src/extract/index.ts'
import type { Catalogue } from '../../../packages/meta/src/schema/index.ts'

const INDEX = 'virtual:whitebox-docs/index'
const MODULES = 'virtual:whitebox-docs/modules'
const MODULE = 'virtual:whitebox-docs/module:'
const ROOT = join(import.meta.dirname, '..', '..', '..')
const SRC = 'packages/engine/src/'

function firstSentence(text: string): string {
  const end = text.search(/[.:](\s|$)/)
  const s = end < 0 ? text : text.slice(0, end + 1)
  return s.length > 200 ? `${s.slice(0, 197)}…` : s
}

/** Serves the docs catalogue; an edit to the engine's source rebuilds it. */
export function docsCatalogue(): Plugin {
  let built: Catalogue | null = null
  const get = (): Catalogue => {
    built ??= extractCatalogue({ root: ROOT, packages: ['engine'] })
    return built
  }
  return {
    name: 'whitebox-docs-catalogue',
    resolveId: (id) => (id === INDEX || id === MODULES || id.startsWith(MODULE) ? `\0${id}` : null),
    load(id) {
      if (!id.startsWith('\0virtual:whitebox-docs/')) return null
      const c = get()
      if (id === `\0${INDEX}`) {
        // Paths relative to packages/engine/src, so the tree mirrors the directories: lib/<domain>/..., methods/<area>/...
        const symbolsOf = new Map<string, { name: string; kind: string; summary: string }[]>()
        for (const s of c.symbols) {
          const list = symbolsOf.get(s.source.file) ?? []
          list.push({ name: s.name, kind: s.kind, summary: firstSentence(s.summary) })
          symbolsOf.set(s.source.file, list)
        }
        const files = c.files.map((f) => ({
          path: f.path.slice(SRC.length),
          module: f.module,
          summary: f.summary,
          description: f.description,
          symbols: symbolsOf.get(f.path) ?? [],
        }))
        const folders = c.modules.map((m) => ({
          path: m.layer === 'core' ? `lib/${m.path}` : m.path,
          module: m.id,
          summary: m.summary,
        }))
        return `export default ${JSON.stringify({ packages: c.packages, folders, files })}`
      }
      if (id === `\0${MODULES}`) {
        const lines = c.modules.map((m) => `  ${JSON.stringify(m.id)}: () => import(${JSON.stringify(MODULE + m.id)}),`)
        return `export default {\n${lines.join('\n')}\n}`
      }
      const module = id.slice(`\0${MODULE}`.length)
      return `export default ${JSON.stringify(c.symbols.filter((s) => s.module === module))}`
    },
    handleHotUpdate({ file, server }) {
      if (!file.startsWith(join(ROOT, 'packages', 'engine', 'src')) || !/\.tsx?$/.test(file)) return
      built = null
      for (const m of server.moduleGraph.idToModuleMap.values())
        if (m.id?.startsWith('\0virtual:whitebox-docs/')) server.moduleGraph.invalidateModule(m)
    },
  }
}
