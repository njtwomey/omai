/**
 * The hub's standalone pages: one-off live demos written as React components, one file per page at
 * `pages/<collection>/<NN-slug>.tsx`, in the gallery's recipe format (a `recipe` export and the example as the default
 * export). They sit in the hub's collections beside the catalogue's entries, for demos that need code of their own
 * rather than a method and its views.
 */

import type { ComponentType } from 'react'
import type { Recipe } from '../gallery/recipe.ts'
import { snippetOf } from '../shell/snippet.ts'

interface PageModule {
  readonly recipe: Recipe
  readonly default: ComponentType
}

const modules = import.meta.glob<PageModule>('./pages/**/*.tsx', { eager: true })
const sources = import.meta.glob<string>('./pages/**/*.tsx', { eager: true, query: '?raw', import: 'default' })

/** A standalone page as the hub shows it. */
export interface HubPage extends Recipe {
  /** `<collection>/<slug>`, like a catalogue entry's id. */
  readonly id: string
  readonly collection: string
  /** The page: `/hub/<collection>/<slug>`. */
  readonly path: string
  /** The file, relative to the site. */
  readonly file: string
  readonly Example: ComponentType
  readonly snippet: string
}

/** Every standalone page, by collection and then file order. Adding a page is adding a file. */
export const PAGES: readonly HubPage[] = Object.entries(modules)
  .map(([key, mod]): HubPage => {
    const match = /\/pages\/(.+)\/([^/]+)\.tsx$/.exec(key)
    const collection = match?.[1] ?? ''
    const name = match?.[2] ?? ''
    if (mod.recipe === undefined) throw new Error(`${key}: exports no \`recipe\``)
    const id = `${collection}/${name.replace(/^\d+-/, '')}`
    return {
      ...mod.recipe,
      id,
      collection,
      path: `/hub/${id}`,
      file: `src/hub/pages/${collection}/${name}.tsx`,
      Example: mod.default,
      snippet: snippetOf(sources[key] ?? ''),
    }
  })
  .sort((a, b) => a.file.localeCompare(b.file))

/** The standalone page with this id, if any. */
export const pageById: ReadonlyMap<string, HubPage> = new Map(PAGES.map((p) => [p.id, p]))
