import type { ComponentType } from 'react'
import { snippetOf } from '../shell/snippet.ts'
import { SECTIONS as KNOWN, type Recipe, type Section } from './recipe.ts'

interface RecipeModule {
  readonly recipe: Recipe
  readonly default: ComponentType
}

const modules = import.meta.glob<RecipeModule>('./recipes/*/*.tsx', { eager: true })
const sources = import.meta.glob<string>('./recipes/*/*.tsx', { eager: true, query: '?raw', import: 'default' })

/** A recipe as the gallery shows it. */
export interface Entry extends Recipe {
  readonly section: string
  /** The file name without its order prefix, e.g. `number-field`. */
  readonly slug: string
  /** The page: `/gallery/<section>/<slug>`. */
  readonly path: string
  /** The file, relative to the site, e.g. `src/gallery/recipes/controls/01-number-field.tsx`. */
  readonly file: string
  readonly Example: ComponentType
  readonly snippet: string
}

const named = import.meta.glob<{ readonly section?: Partial<Section> }>('./recipes/*/_section.ts', { eager: true })

const folderOf = (key: string): string => /\/recipes\/([^/]+)\//.exec(key)?.[1] ?? ''
const titleOf = (id: string): string => {
  const words = id.replace(/[-_]+/g, ' ')
  return words.charAt(0).toUpperCase() + words.slice(1)
}

/**
 * Every section: one per recipe folder, in on-disk (alphabetical) order, titled and described by the known list, or by
 * the folder's `_section.ts`, or else from the folder's name.
 */
export const SECTIONS: readonly Section[] = [...new Set(Object.keys(modules).map(folderOf))].sort().map((id) => {
  const known = KNOWN.find((s) => s.id === id)
  const given = Object.entries(named).find(([k]) => folderOf(k) === id)?.[1].section
  return { id, title: titleOf(id), blurb: '', ...known, ...given }
})

const order = new Map<string, number>(SECTIONS.map((s, i) => [s.id, i]))

/** Every recipe, in section order and then file order. Adding a recipe is adding a file. */
export const ENTRIES: readonly Entry[] = Object.entries(modules)
  .map(([key, mod]): Entry => {
    const match = /\/recipes\/([^/]+)\/([^/]+)\.tsx$/.exec(key)
    const section = match?.[1] ?? ''
    const name = match?.[2] ?? ''
    if (mod.recipe === undefined) throw new Error(`${key}: exports no \`recipe\``)
    const slug = name.replace(/^\d+-/, '')
    return {
      ...mod.recipe,
      section,
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
