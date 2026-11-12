import { createContext, type ReactNode, useContext } from 'react'
import { Link } from '../../../site/src/shell/Link.tsx'
import { type StoryEntry, storyHref } from '../story.ts'

/** What the app tells a story page about where it is: its own entry and its neighbours in reading order. */
export interface StoryContextValue {
  readonly entry: StoryEntry
  readonly prev: StoryEntry | undefined
  readonly next: StoryEntry | undefined
}

/** Provided by the story route around every story. */
export const StoryContext = createContext<StoryContextValue | undefined>(undefined)

function Neighbour(props: { readonly entry: StoryEntry | undefined; readonly direction: 'prev' | 'next' }): ReactNode {
  const { entry, direction } = props
  if (entry === undefined) return <span />
  return (
    <Link
      to={storyHref(entry.slug)}
      className={`flex flex-col gap-1 rounded-lg border border-line p-4 hover:border-muted hover:bg-surface ${direction === 'next' ? 'items-end text-right' : ''}`}
    >
      <span className="text-xs uppercase tracking-wider text-muted">{direction === 'prev' ? 'Previous' : 'Next'}</span>
      <span className="font-semibold leading-snug">{entry.title}</span>
      <span className="line-clamp-2 text-sm text-muted">{entry.standfirst}</span>
    </Link>
  )
}

/**
 * The layout of a story: the title, the one-line standfirst, the date and tags, then the story's own content in a single reading
 * column (plain paragraphs, headings and lists are styled by the column), and at the end links to the previous and next story.
 * The story's entry comes from the route, so a story's `Story.tsx` just renders `<StoryPage>...</StoryPage>`.
 *
 * @param props - The content of the story.
 * @returns The page.
 */
export function StoryPage(props: { readonly children: ReactNode }): ReactNode {
  const context = useContext(StoryContext)
  if (context === undefined) throw new Error('StoryPage is rendered by the story route, which provides its context')
  const { entry, prev, next } = context
  return (
    <article className="mx-auto w-full max-w-[42.5rem] px-4 pt-10 pb-24 sm:px-6 sm:pt-16">
      <header className="mb-12 font-sans">
        <p className="mb-4 text-sm text-muted">
          <time dateTime={entry.date}>{entry.date}</time>
          {entry.tags.length === 0 ? null : <span> · {entry.tags.join(', ')}</span>}
        </p>
        <h1 className="m-0 text-balance text-[2.35rem] font-bold leading-[1.1] tracking-tight sm:text-5xl">
          {entry.title}
        </h1>
        <p className="mt-5 text-balance text-xl leading-snug text-muted sm:text-[1.35rem]">{entry.standfirst}</p>
      </header>
      <div className="story-prose">{props.children}</div>
      <nav aria-label="More stories" className="mt-20 grid gap-3 font-sans sm:grid-cols-2">
        <Neighbour entry={prev} direction="prev" />
        <Neighbour entry={next} direction="next" />
      </nav>
    </article>
  )
}
