import type { ReactNode } from 'react'
import { type SiteId, sites } from './sites.ts'

/** The shared page chrome: a top bar linking the three sites, the current one highlighted, and the page below. */
export function Shell({ site, children }: { readonly site: SiteId; readonly children: ReactNode }): ReactNode {
  return (
    <div className="min-h-screen bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
      <header className="flex items-center gap-6 border-b border-neutral-200 px-6 py-3 dark:border-neutral-800">
        <span className="font-semibold">whitebox</span>
        <nav className="flex gap-4 text-sm">
          {sites.map((s) => (
            <a
              key={s.id}
              href={s.url}
              className={s.id === site ? 'font-medium underline underline-offset-4' : 'text-neutral-500'}
            >
              {s.title}
            </a>
          ))}
        </nav>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
    </div>
  )
}
