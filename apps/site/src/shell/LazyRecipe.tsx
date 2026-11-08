/**
 * A recipe-format page's live example and code, loaded when the page is opened: the registry (the gallery's recipes)
 * lists every file from its `recipe` metadata alone (`?recipe`, plugins/recipe-meta.ts)
 * and keep a loader for the file's module and source, so an index page imports none of their code.
 */

import { type ComponentType, type LazyExoticComponent, lazy, type ReactNode, Suspense } from 'react'
import { ErrorBoundary } from './ErrorBoundary.tsx'
import { snippetOf } from './snippet.ts'

/** A recipe file's live example and the code shown beside it. */
export interface RecipeCode {
  readonly Example: ComponentType
  readonly snippet: string
}

/** Loads a recipe file's example and code (once; a failed load is tried again on the next call). */
export type RecipeLoader = () => Promise<RecipeCode>

/**
 * A loader for a recipe file from lazy globs of its module and its raw source.
 *
 * @param module - Imports the file (its default export is the example).
 * @param source - Imports the file's source (`?raw`).
 * @returns The loader.
 */
export function recipeLoader(
  module: () => Promise<{ readonly default: ComponentType }>,
  source: () => Promise<string>,
): RecipeLoader {
  let pending: Promise<RecipeCode> | undefined
  return () => {
    pending ??= Promise.all([module(), source()]).then(
      ([m, s]) => ({ Example: m.default, snippet: snippetOf(s) }),
      (e: unknown) => {
        pending = undefined
        throw e
      },
    )
    return pending
  }
}

const views = new Map<string, LazyExoticComponent<ComponentType>>()

function viewOf(id: string, file: string, load: RecipeLoader): LazyExoticComponent<ComponentType> {
  let View = views.get(id)
  if (View === undefined) {
    // The code view loads with the example too: it imports omai-render's React bindings (its icons).
    View = lazy(() =>
      Promise.all([load(), import('./RecipeView.tsx')]).then(
        ([{ Example, snippet }, { RecipeView }]) => ({
          default: (): ReactNode => (
            <RecipeView code={snippet} file={file}>
              <ErrorBoundary resetKey={id}>
                <Example />
              </ErrorBoundary>
            </RecipeView>
          ),
        }),
        (e: unknown) => {
          views.delete(id)
          throw e
        },
      ),
    )
    views.set(id, View)
  }
  return View
}

/**
 * A recipe's live example beside its own source, loaded on first render (a line says so meanwhile); a failure to load
 * shows in the page's error box.
 *
 * @param props - The recipe's page path (its identity), its file and its loader.
 * @returns The view.
 */
export function LazyRecipeView(props: {
  readonly id: string
  readonly file: string
  readonly load: RecipeLoader
}): ReactNode {
  const View = viewOf(props.id, props.file, props.load)
  return (
    <ErrorBoundary resetKey={props.id}>
      <Suspense fallback={<p className="text-sm text-muted">Loading the example…</p>}>
        <View />
      </Suspense>
    </ErrorBoundary>
  )
}
