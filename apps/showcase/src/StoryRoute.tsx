import { type ReactNode, Suspense } from 'react'
import { ErrorBoundary } from '../../site/src/shell/ErrorBoundary.tsx'
import { Link } from '../../site/src/shell/Link.tsx'
import { StoryContext } from './kit/StoryPage.tsx'
import { pageOf, STORIES, storyOf } from './registry.ts'
import { neighbours } from './story.ts'

/** `/story/<slug>`: the story's narrative inside the context of its entry and neighbours, or a short note for an unknown slug. */
export function StoryRoute(props: { readonly slug: string }): ReactNode {
  const entry = storyOf(props.slug)
  const Page = pageOf(props.slug)
  if (entry === undefined || Page === undefined)
    return (
      <div className="mx-auto max-w-[42.5rem] px-4 pt-16 font-sans sm:px-6">
        <h1 className="text-2xl font-bold">No such story</h1>
        <p className="mt-3 text-muted">
          There is no story called <code>{props.slug}</code>.{' '}
          <Link to="/" className="underline">
            Back to the stories
          </Link>
        </p>
      </div>
    )
  const { prev, next } = neighbours(STORIES, props.slug)
  return (
    <StoryContext.Provider value={{ entry, prev, next }}>
      <ErrorBoundary>
        <Suspense fallback={<p className="px-6 pt-16 text-center font-sans text-sm text-muted">Opening the story</p>}>
          <Page />
        </Suspense>
      </ErrorBoundary>
    </StoryContext.Provider>
  )
}
