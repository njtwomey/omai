/**
 * The hub's standalone pages: one-off live demos written as React components, one file per page at
 * `pages/<collection>/<NN-slug>.tsx`, in the gallery's recipe format (a `recipe` export and the example as the default
 * export). They sit in the hub's collections beside the catalogue's entries, for demos that need code of their own
 * rather than a method and its views.
 */

import type { ComponentType } from 'react'
import type { Recipe } from '../gallery/recipe.ts'
import { type RecipeLoader, recipeLoader } from '../shell/LazyRecipe.tsx'

// Only the pages' metadata loads with the hub (`?recipe`, plugins/recipe-meta.ts); a page's code and source load when
// it is opened (some train a model as they load).
const metas = import.meta.glob<Recipe>('./pages/**/*.tsx', { eager: true, query: '?recipe', import: 'recipe' })
const modules = import.meta.glob<{ readonly default: ComponentType }>('./pages/**/*.tsx')
const sources = import.meta.glob<string>('./pages/**/*.tsx', { query: '?raw', import: 'default' })

/** A standalone page as the hub shows it. */
export interface HubPage extends Recipe {
  /** `<collection>/<slug>`, like a catalogue entry's id. */
  readonly id: string
  readonly collection: string
  /** The page: `/hub/<collection>/<slug>`. */
  readonly path: string
  /** The file, relative to the site. */
  readonly file: string
  /** Loads the live demo and the code shown beside it. */
  readonly load: RecipeLoader
}

/** Every standalone page, by collection and then file order. Adding a page is adding a file. */
export const PAGES: readonly HubPage[] = Object.entries(metas)
  .map(([key, recipe]): HubPage => {
    const match = /\/pages\/(.+)\/([^/]+)\.tsx$/.exec(key)
    const collection = match?.[1] ?? ''
    const name = match?.[2] ?? ''
    if (recipe === undefined) throw new Error(`${key}: exports no \`recipe\``)
    const id = `${collection}/${name.replace(/^\d+-/, '')}`
    return {
      ...recipe,
      id,
      collection,
      path: `/hub/${id}`,
      file: `src/hub/pages/${collection}/${name}.tsx`,
      load: recipeLoader(
        modules[key] as () => Promise<{ readonly default: ComponentType }>,
        sources[key] as () => Promise<string>,
      ),
    }
  })
  .sort((a, b) => a.file.localeCompare(b.file))

/** The standalone page with this id, if any. */
export const pageById: ReadonlyMap<string, HubPage> = new Map(PAGES.map((p) => [p.id, p]))
