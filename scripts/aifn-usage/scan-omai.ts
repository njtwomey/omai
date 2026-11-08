/**
 * Read the public surface of omai as the repository's own docs extractor sees it (`packages/meta`), the source of
 * `docs/api-surface.txt`: every export once at its canonical home, with the other modules that re-export it.
 *
 *   node scripts/aifn-usage/scan-omai.ts     print the counts
 *
 * Used by `match.ts`; nothing is written. The surface is not stored because it has about 18,000 entries.
 */
import { ROOT } from './lib.ts'

export interface OmaiSymbol {
  readonly module: string
  readonly name: string
  readonly kind: string
  readonly summary: string
  readonly indexed: boolean
  /** Other `<module>#<name>` places it is exported from. */
  readonly aliases: readonly string[]
}

export async function scanOmai(): Promise<OmaiSymbol[]> {
  const { readProject } = await import('../../packages/meta/src/extract/project.ts')
  const { readSymbols } = await import('../../packages/meta/src/extract/symbols.ts')
  const project = readProject(ROOT, ['engine', 'render', 'meta'])
  const { symbols } = readSymbols(project)
  return symbols.map((s) => ({
    module: s.module,
    name: s.name,
    kind: s.kind,
    summary: s.summary.replace(/\s+/g, ' ').slice(0, 110),
    indexed: s.index !== undefined,
    aliases: s.aliases,
  }))
}

if (import.meta.main) {
  const all = await scanOmai()
  const modules = new Set(all.map((s) => s.module))
  console.log(`omai surface: ${all.length} exports in ${modules.size} entry points`)
}
