/**
 * The engine documentation as virtual modules, extracted from the source at build time by whitebox-meta/extract (a
 * few seconds for the whole engine), so the docs pages are generated and cannot drift from the code.
 *
 * - `virtual:whitebox-docs/index`: packages, modules, and every symbol's id, kind and one-line summary (the sidebar,
 *   the home page, module pages and search).
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
        const symbols = c.symbols.map((s) => ({
          id: s.id,
          module: s.module,
          name: s.name,
          kind: s.kind,
          summary: firstSentence(s.summary),
        }))
        return `export default ${JSON.stringify({ packages: c.packages, modules: c.modules, symbols })}`
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
