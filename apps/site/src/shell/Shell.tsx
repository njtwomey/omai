import { cssText } from 'omai-render/theme'
import { type ReactNode, useEffect, useState } from 'react'
import { ColourToggle } from './ColourToggle.tsx'
import { Link } from './Link.tsx'

/** The site's sections, in navigation order. */
export const SECTIONS = [
  { path: '/gallery', title: 'Gallery', summary: 'What omai-render can draw, lay out, control and animate.' },
  { path: '/docs', title: 'Docs', summary: 'The omai-engine reference, generated from the source.' },
] as const

// omai-render's theme tokens as CSS variables, light on :root and dark under the `dark` class.
const THEME_CSS = `${cssText('light', ':root')}\n${cssText('dark', '.dark')}`

/**
 * The shared page chrome (also the chrome of the proofs of concept app, apps/pocs): the top bar with the sections (the current one highlighted) and the colour toggle;
 * then the page, beside an optional `sidebar` (a section's {@link TocSidebar}) that stays in view and scrolls on its
 * own, the content on the right.
 */
export function Shell(props: {
  readonly path: string
  readonly children: ReactNode
  readonly sidebar?: ReactNode
  /** The sections of the top bar (default: the site's own); the proofs of concept app passes its own. */
  readonly sections?: readonly { readonly path: string; readonly title: string }[]
  /** The name beside the logo (default `omai`). */
  readonly brand?: string
}): ReactNode {
  const { path, children, sidebar, sections = SECTIONS, brand = 'omai' } = props
  // On narrow screens the sidebar is a drawer, closed again by navigating.
  const [drawer, setDrawer] = useState(false)
  // biome-ignore lint/correctness/useExhaustiveDependencies: close on every change of page
  useEffect(() => setDrawer(false), [path])
  return (
    <div className="min-h-screen bg-bg text-fg">
      <style>{THEME_CSS}</style>
      <header className="sticky top-0 z-30 flex h-12 items-center gap-3 border-b border-line bg-bg/90 px-3 backdrop-blur sm:gap-6 md:px-6">
        {sidebar === undefined ? null : (
          <button
            type="button"
            aria-label="Contents"
            onClick={() => setDrawer(true)}
            className="-mr-3 rounded-md p-1.5 text-muted hover:bg-surface hover:text-fg md:hidden"
          >
            <svg
              viewBox="0 0 24 24"
              width="16"
              height="16"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              aria-hidden="true"
            >
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        )}
        <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <span className="size-3 rounded-sm bg-gradient-to-br from-[#1f77b4] via-[#2ca02c] to-[#ff7f0e]" />
          {brand}
        </Link>
        <nav className="flex gap-1 text-sm">
          {sections.map((s) => (
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
          <aside
            className={`fixed inset-y-0 left-0 z-40 w-72 shrink-0 border-r border-line bg-bg md:sticky md:top-12 md:z-10 md:block md:h-[calc(100vh-3rem)] md:w-64 lg:w-72 ${drawer ? 'block shadow-xl' : 'hidden'}`}
          >
            {sidebar}
          </aside>
          {drawer ? (
            <button
              type="button"
              aria-label="Close the contents"
              className="fixed inset-0 z-30 bg-black/40 md:hidden"
              onClick={() => setDrawer(false)}
            />
          ) : null}
          <main className="min-w-0 flex-1">
            <div className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-4 py-8 md:px-8">{children}</div>
          </main>
        </div>
      )}
    </div>
  )
}
