import type { ReactNode } from 'react'
import { cssText } from 'whitebox-render/theme'
import { ColourToggle } from './ColourToggle.tsx'
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

// whitebox-render's theme tokens as CSS variables, light on :root and dark under the `dark` class.
const THEME_CSS = `${cssText('light', ':root')}\n${cssText('dark', '.dark')}`

/**
 * The shared page chrome: the top bar with the three sections (the current one highlighted) and the colour toggle;
 * then the page, beside an optional `sidebar` (a section's {@link TocSidebar}) that stays in view and scrolls on its
 * own, the content on the right.
 */
export function Shell(props: {
  readonly path: string
  readonly children: ReactNode
  readonly sidebar?: ReactNode
}): ReactNode {
  const { path, children, sidebar } = props
  return (
    <div className="min-h-screen bg-bg text-fg">
      <style>{THEME_CSS}</style>
      <header className="sticky top-0 z-30 flex h-12 items-center gap-6 border-b border-line bg-bg/90 px-4 backdrop-blur md:px-6">
        <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <span className="size-3 rounded-sm bg-gradient-to-br from-[#1f77b4] via-[#2ca02c] to-[#ff7f0e]" />
          whitebox
        </Link>
        <nav className="flex gap-1 text-sm">
          {SECTIONS.map((s) => (
            <Link
              key={s.path}
              to={s.path}
              current={path.startsWith(s.path)}
              className={
                path.startsWith(s.path)
                  ? 'rounded-md bg-surface px-2.5 py-1 font-medium text-fg'
                  : 'rounded-md px-2.5 py-1 text-muted hover:bg-surface hover:text-fg'
              }
            >
              {s.title}
            </Link>
          ))}
        </nav>
        <ColourToggle />
      </header>
      {sidebar === undefined ? (
        <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
      ) : (
        <div className="flex">
          <aside className="sticky top-12 hidden h-[calc(100vh-3rem)] w-64 shrink-0 border-r border-line md:block lg:w-72">
            {sidebar}
          </aside>
          <main className="min-w-0 flex-1">
            <div className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-4 py-8 md:px-8">{children}</div>
          </main>
        </div>
      )}
    </div>
  )
}
