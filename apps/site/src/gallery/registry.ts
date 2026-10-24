import type { ComponentType } from 'react'
import { snippetOf } from '../shell/snippet.ts'
import { type Recipe, SECTIONS, type SectionId } from './recipe.ts'

interface RecipeModule {
  readonly recipe: Recipe
  readonly default: ComponentType
}

const modules = import.meta.glob<RecipeModule>('./recipes/*/*.tsx', { eager: true })
const sources = import.meta.glob<string>('./recipes/*/*.tsx', { eager: true, query: '?raw', import: 'default' })

/** A recipe as the gallery shows it. */
export interface Entry extends Recipe {
  readonly section: SectionId
  /** The file name without its order prefix, e.g. `number-field`. */
  readonly slug: string
  /** The page: `/gallery/<section>/<slug>`. */
  readonly path: string
  /** The file, relative to the site, e.g. `src/gallery/recipes/controls/01-number-field.tsx`. */
  readonly file: string
  readonly Example: ComponentType
  readonly snippet: string
}

const order = new Map<string, number>(SECTIONS.map((s, i) => [s.id, i]))

/** Every recipe, in section order and then file order. Adding a recipe is adding a file. */
export const ENTRIES: readonly Entry[] = Object.entries(modules)
  .map(([key, mod]): Entry => {
    const match = /\/recipes\/([^/]+)\/([^/]+)\.tsx$/.exec(key)
    const section = match?.[1] ?? ''
    const name = match?.[2] ?? ''
    if (!order.has(section)) throw new Error(`${key}: no section '${section}' in SECTIONS (gallery/recipe.ts)`)
    if (mod.recipe === undefined) throw new Error(`${key}: exports no \`recipe\``)
    const slug = name.replace(/^\d+-/, '')
    return {
      ...mod.recipe,
      section: section as SectionId,
      slug,
      path: `/gallery/${section}/${slug}`,
      file: `src/gallery/recipes/${section}/${name}.tsx`,
      Example: mod.default,
      snippet: snippetOf(sources[key] ?? ''),
    }
  })
  .sort((a, b) => (order.get(a.section) ?? 0) - (order.get(b.section) ?? 0) || a.file.localeCompare(b.file))

/** The recipes of one section. */
export function entriesOf(section: string): Entry[] {
  return ENTRIES.filter((e) => e.section === section)
}
