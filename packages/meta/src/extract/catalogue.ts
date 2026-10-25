/**
 * The whole catalogue in one call: read the packages, their entry points, folders and files with their summaries, and
 * every exported symbol.
 */
import type { Catalogue } from '../schema/catalogue.ts'
import { readProject } from './project.ts'
import { readSymbols } from './symbols.ts'

/** Which packages to document. */
export interface CatalogueOptions {
  /** The repository root. */
  readonly root: string
  /** Package folders under `packages/` (default `['engine']`). */
  readonly packages?: readonly string[]
}

/**
 * Extract the documentation catalogue of a repository's packages from their source.
 *
 * @param options - The repository root and the packages.
 * @returns The catalogue.
 */
export function extractCatalogue(options: CatalogueOptions): Catalogue {
  const project = readProject(options.root, options.packages ?? ['engine'])
  const { modules, folders, files, symbols } = readSymbols(project)
  const packages = project.packages.map((p) => ({
    name: p.name,
    description: p.description,
    version: p.version,
    modules: modules.filter((m) => m.package === p.name && m.parent === undefined).map((m) => m.id),
  }))
  return { version: 1, packages, modules, folders, files, symbols }
}
