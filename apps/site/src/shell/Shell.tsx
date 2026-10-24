import type { ReactNode } from 'react'
import { Link } from './Link.tsx'

/** The site's sections, in navigation order. */
export const SECTIONS = [
  { path: '/gallery', title: 'Gallery', summary: 'What whitebox-render can draw, lay out, control and animate.' },
  {
    path: '/hub',
    title: 'Hub',
    summary: 'Worked methods with their datasets, controls and the views that explain them.',
  },
  { path: '/docs', title: 'Docs', summary: 'The whitebox-engine reference, generated from the source.' },
] as const

/** The shared page chrome: the top bar with the three sections (the current one highlighted) and the page. */
export function Shell(props: { readonly path: string; readonly children: ReactNode }): ReactNode {
  return (
    <div className="min-h-screen bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
      <header className="flex items-center gap-6 border-b border-neutral-200 px-6 py-3 dark:border-neutral-800">
        <Link to="/" className="font-semibold">
          whitebox
        </Link>
        <nav className="flex gap-4 text-sm">
          {SECTIONS.map((s) => (
            <Link
              key={s.path}
              to={s.path}
              className={
                props.path.startsWith(s.path) ? 'font-medium underline underline-offset-4' : 'text-neutral-500'
              }
            >
              {s.title}
            </Link>
          ))}
        </nav>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">{props.children}</main>
    </div>
  )
}
