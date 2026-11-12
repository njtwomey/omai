import type { CSSProperties, ReactNode } from 'react'
import { Link } from '../../site/src/shell/Link.tsx'
import { STORIES } from './registry.ts'
import { bySection, type StoryEntry, storyHref } from './story.ts'

/** A hue from the slug, so a story keeps its own tint on the card without a thumbnail. */
export function hueOf(slug: string): number {
  let h = 0
  for (const c of slug) h = (h * 31 + c.charCodeAt(0)) % 360
  return h
}

function Thumb(props: { readonly entry: StoryEntry }): ReactNode {
  const { entry } = props
  if (entry.thumbnail !== undefined)
    return <img src={entry.thumbnail} alt="" loading="lazy" className="aspect-[16/9] w-full object-cover" />
  return (
    <div
      className="story-thumb aspect-[16/9] w-full"
      style={{ '--tint': hueOf(entry.slug) } as CSSProperties}
      aria-hidden="true"
    />
  )
}

function Card(props: { readonly entry: StoryEntry }): ReactNode {
  const { entry } = props
  return (
    <Link
      to={storyHref(entry.slug)}
      className="group flex flex-col overflow-hidden rounded-xl border border-line bg-bg transition-colors hover:border-muted"
    >
      <Thumb entry={entry} />
      <div className="flex flex-1 flex-col gap-1.5 p-4">
        <h3 className="m-0 text-lg font-semibold leading-snug tracking-tight">{entry.title}</h3>
        <p className="m-0 text-[0.95rem] leading-snug text-muted">{entry.hook ?? entry.standfirst}</p>
      </div>
    </Link>
  )
}

/**
 * The splash page at `/`: a line about what omai is, then the stories as a grid of cards, section by section (the order of
 * `sections.ts`; a section without a story is left out).
 *
 * @returns The page.
 */
export function Splash(): ReactNode {
  const groups = bySection(STORIES)
  return (
    <div className="mx-auto w-full max-w-5xl px-4 pb-24 font-sans sm:px-6">
      <section className="pt-14 pb-16 sm:pt-24 sm:pb-20">
        <h1 className="m-0 max-w-3xl text-balance text-[2.5rem] font-bold leading-[1.08] tracking-tight sm:text-6xl">
          Statistics and machine learning, written so you can read it.
        </h1>
        <p className="mt-6 max-w-2xl text-balance text-lg leading-relaxed text-muted sm:text-xl">
          omai is a TypeScript library for tensors, probability, inference and learning. Each story below is one small
          thing it does. Nothing runs until you press play; then the work happens in a background worker, in your
          browser, and the picture arrives when it is done.
        </p>
      </section>
      {groups.length === 0 ? <p className="text-muted">No stories yet.</p> : null}
      {groups.map((g) => (
        <section key={g.id} aria-labelledby={`section-${g.id}`} className="mb-14">
          <h2 id={`section-${g.id}`} className="m-0 text-2xl font-semibold tracking-tight">
            {g.title}
          </h2>
          <p className="mt-1 mb-5 text-muted">{g.blurb}</p>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {g.stories.map((e) => (
              <Card key={e.slug} entry={e} />
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}
