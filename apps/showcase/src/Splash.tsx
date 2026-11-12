import type { ReactNode } from 'react'
import { Link } from '../../site/src/shell/Link.tsx'
import { STORIES } from './registry.ts'
import { bySection, storyHref } from './story.ts'

/**
 * The splash page at `/`: a line about what omai is, then the stories as a plain list under their category (the order of
 * `sections.ts`; a category without a story is left out).
 *
 * @returns The page.
 */
export function Splash(): ReactNode {
  const groups = bySection(STORIES)
  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-24 font-sans sm:px-6">
      <section className="pt-14 pb-10 sm:pt-20 sm:pb-12">
        <h1 className="m-0 text-balance text-[2.25rem] font-bold leading-[1.1] tracking-tight sm:text-5xl">
          omai showcase
        </h1>
        <p className="mt-5 max-w-2xl text-balance text-lg leading-relaxed text-muted">
          omai is a TypeScript library for tensors, probability, inference and learning. Each page below shows one thing
          it does. Nothing runs until you press play. The work then runs in a background worker in your browser, and the
          figures appear when it is done.
        </p>
      </section>
      {groups.length === 0 ? <p className="text-muted">No stories yet.</p> : null}
      {groups.map((g) => (
        <section key={g.id} aria-labelledby={`section-${g.id}`} className="mb-10">
          <h2 id={`section-${g.id}`} className="m-0 text-xl font-semibold tracking-tight">
            {g.title}
          </h2>
          <ul className="mt-3 list-none space-y-2 p-0">
            {g.stories.map((e) => (
              <li key={e.slug} className="m-0">
                <Link to={storyHref(e.slug)} className="font-medium text-fg underline-offset-4 hover:underline">
                  {e.title}
                </Link>
                <span className="text-muted"> — {e.hook ?? e.standfirst}</span>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}
