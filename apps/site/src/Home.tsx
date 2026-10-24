import type { ReactNode } from 'react'
import { Link } from './shell/Link.tsx'
import { SECTIONS } from './shell/Shell.tsx'

/** The front page: the three sections. */
export function Home(): ReactNode {
  return (
    <div className="grid gap-4 sm:grid-cols-3">
      {SECTIONS.map((s) => (
        <Link
          key={s.path}
          to={s.path}
          className="rounded-lg border border-neutral-200 p-5 hover:border-neutral-400 dark:border-neutral-800"
        >
          <h2 className="font-semibold">{s.title}</h2>
          <p className="mt-1 text-sm text-neutral-500">{s.summary}</p>
        </Link>
      ))}
    </div>
  )
}
