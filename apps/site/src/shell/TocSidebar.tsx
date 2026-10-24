import { type ReactNode, useMemo, useState } from 'react'
import { Link } from './Link.tsx'
import { ancestorsOf, filterToc, innerIds, type TocNode } from './toc.ts'

/** Props of {@link TocSidebar}. */
export interface TocSidebarProps {
  /** The tree to browse (see {@link TocNode}), in the order it should show. */
  readonly nodes: readonly TocNode[]
  /** The current page's path: its node is highlighted and the folders on the way to it are open. */
  readonly current: string
  /** A link above the filter (the section's front page). */
  readonly home?: { readonly title: string; readonly path: string }
  /** The filter box's placeholder. */
  readonly placeholder?: string
}

/**
 * The left-hand contents of a section: a filter box, then the tree as given, every folder closed except those on the
 * way to the current page, which is highlighted. A folder's chevron opens or closes it; its name links its own page
 * when it has a `path` (and toggles it otherwise). Typing in the filter shows only the matching nodes and the folders
 * above them, opened.
 *
 * @param props - The tree, the current path, the front page link and the placeholder.
 * @returns The sidebar's contents.
 */
export function TocSidebar(props: TocSidebarProps): ReactNode {
  const { nodes, current, home, placeholder = 'Filter' } = props
  const [query, setQuery] = useState('')
  // Folders the reader has toggled on this page; navigating starts again from the path to the new page.
  const [toggled, setToggled] = useState<{ readonly page: string; readonly ids: ReadonlySet<string> }>({
    page: current,
    ids: new Set(),
  })
  const flipped = toggled.page === current ? toggled.ids : new Set<string>()
  const trail = useMemo(() => new Set(ancestorsOf(nodes, current)), [nodes, current])
  const shown = useMemo(() => filterToc(nodes, query), [nodes, query])
  const filtering = query.trim() !== ''
  const forced = useMemo(() => new Set(filtering ? innerIds(shown) : []), [filtering, shown])
  const isOpen = (id: string): boolean => forced.has(id) || trail.has(id) !== flipped.has(id)
  const toggle = (id: string): void => {
    const next = new Set(flipped)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    setToggled({ page: current, ids: next })
  }
  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-col gap-2 px-3 pt-3 pb-2">
        {home === undefined ? null : (
          <Link
            to={home.path}
            current={current === home.path}
            className={`rounded-md px-2 py-1 text-sm font-semibold ${current === home.path ? 'bg-surface text-accent' : 'hover:bg-surface'}`}
          >
            {home.title}
          </Link>
        )}
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={placeholder}
          aria-label={placeholder}
          className="h-8 w-full rounded-md border border-line bg-bg px-2.5 text-sm outline-none placeholder:text-muted focus:border-accent"
        />
      </div>
      <nav aria-label="Contents" className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-2 pb-6 text-sm">
        {shown.length === 0 ? <p className="px-2 py-1 text-xs text-muted">Nothing matches.</p> : null}
        <TocList nodes={shown} depth={0} current={current} isOpen={isOpen} toggle={toggle} />
      </nav>
    </div>
  )
}

function TocList(props: {
  readonly nodes: readonly TocNode[]
  readonly depth: number
  readonly current: string
  readonly isOpen: (id: string) => boolean
  readonly toggle: (id: string) => void
}): ReactNode {
  const { nodes, depth, current, isOpen, toggle } = props
  return (
    <ul className="m-0 list-none p-0">
      {nodes.map((n) => {
        const folder = n.children !== undefined && n.children.length > 0
        const here = n.path !== undefined && n.path === current
        const opened = folder && isOpen(n.id)
        const indent = { paddingLeft: `${0.5 + depth * 0.85}rem` }
        return (
          <li key={n.id}>
            {folder ? (
              <div
                style={indent}
                className={`flex items-center rounded-md ${here ? 'bg-surface' : 'hover:bg-surface'}`}
              >
                <button
                  type="button"
                  aria-expanded={opened}
                  aria-label={opened ? `Close ${n.title}` : `Open ${n.title}`}
                  onClick={() => toggle(n.id)}
                  className="flex size-5 shrink-0 items-center justify-center rounded text-muted hover:text-fg"
                >
                  <svg
                    viewBox="0 0 24 24"
                    width="12"
                    height="12"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    aria-hidden="true"
                    className={`transition-transform ${opened ? 'rotate-90' : ''}`}
                  >
                    <path d="m9 18 6-6-6-6" />
                  </svg>
                </button>
                {n.path === undefined ? (
                  <button
                    type="button"
                    onClick={() => toggle(n.id)}
                    className="min-w-0 flex-1 truncate py-1 pr-2 pl-0.5 text-left text-fg"
                  >
                    {n.title}
                  </button>
                ) : (
                  <Link
                    to={n.path}
                    current={here}
                    className={`min-w-0 flex-1 truncate py-1 pr-2 pl-0.5 ${here ? 'font-medium text-accent' : 'text-fg hover:text-fg'}`}
                  >
                    {n.title}
                  </Link>
                )}
              </div>
            ) : n.path === undefined ? (
              <span style={indent} className="block truncate py-1 pr-2 text-muted">
                {n.title}
              </span>
            ) : (
              <Link
                to={n.path}
                current={here}
                className={`block truncate rounded-md py-1 pr-2 ${here ? 'bg-surface font-medium text-accent' : 'text-fg-secondary hover:bg-surface hover:text-fg'}`}
              >
                <span style={{ paddingLeft: `${1.2 + depth * 0.85}rem` }} className="block truncate">
                  {n.title}
                </span>
              </Link>
            )}
            {opened ? (
              <TocList nodes={n.children ?? []} depth={depth + 1} current={current} isOpen={isOpen} toggle={toggle} />
            ) : null}
          </li>
        )
      })}
    </ul>
  )
}
