import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { Link } from './Link.tsx'
import { ancestorsOf, filterToc, innerIds, type TocNode } from './toc.ts'

/** Props of {@link TocSidebar}. */
export interface TocSidebarProps {
  /** The tree to browse (see {@link TocNode}). */
  readonly nodes: readonly TocNode[]
  /** The current page's path: its node is highlighted and its ancestors opened. */
  readonly current: string
  /** Names this tree's remembered open groups in `localStorage` (one key per section: `gallery`, `hub`, `docs`). */
  readonly storageKey: string
  /** A heading link above the search (the section's front page). */
  readonly home?: { readonly title: string; readonly path: string }
  /** The search box's placeholder. */
  readonly placeholder?: string
}

const STORE = (key: string): string => `whitebox-toc:${key}`

function remembered(key: string): Set<string> {
  try {
    const raw = localStorage.getItem(STORE(key))
    return new Set(raw === null ? [] : (JSON.parse(raw) as string[]))
  } catch {
    return new Set()
  }
}

/**
 * The left-hand table of contents of a section: a search box that filters the tree as you type (keeping the ancestors
 * of matches and opening them), then the tree as collapsible groups, the current page highlighted and its ancestors
 * open. Which groups are open is remembered per `storageKey`. Feed it any tree of {@link TocNode}s; place it with
 * the shell's `sidebar`, which makes it sticky and scroll on its own beside the content.
 *
 * @param props - The tree, the current path and the storage key.
 * @returns The sidebar's contents.
 */
export function TocSidebar(props: TocSidebarProps): ReactNode {
  const { nodes, current, storageKey, home, placeholder = 'Search' } = props
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState<Set<string>>(() => remembered(storageKey))
  // Navigating opens the way to the current page.
  useEffect(() => {
    const trail = ancestorsOf(nodes, current)
    if (trail.length === 0) return
    setOpen((o) => (trail.every((id) => o.has(id)) ? o : new Set([...o, ...trail])))
  }, [nodes, current])
  useEffect(() => {
    try {
      localStorage.setItem(STORE(storageKey), JSON.stringify([...open]))
    } catch {
      // Unavailable storage only means the groups are not remembered.
    }
  }, [open, storageKey])
  const searching = query.trim() !== ''
  const shown = useMemo(() => filterToc(nodes, query), [nodes, query])
  const forced = useMemo(() => new Set(searching ? innerIds(shown) : []), [searching, shown])
  const toggle = (id: string): void =>
    setOpen((o) => {
      const next = new Set(o)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-col gap-2 px-3 pt-3 pb-2">
        {home === undefined ? null : (
          <Link
            to={home.path}
            current={current === home.path}
            className={`rounded-md px-2 py-1 text-sm font-semibold ${current === home.path ? 'bg-surface text-fg' : 'hover:bg-surface'}`}
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
        <TocList
          nodes={shown}
          depth={0}
          current={current}
          isOpen={(id) => forced.has(id) || open.has(id)}
          toggle={toggle}
        />
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
    <ul className={depth === 0 ? 'm-0 list-none p-0' : 'm-0 ml-3 list-none border-l border-line p-0 pl-1'}>
      {nodes.map((n) => {
        const inner = n.children !== undefined && n.children.length > 0
        const here = n.path !== undefined && n.path === current
        const opened = inner && isOpen(n.id)
        const label = (
          <span className="flex min-w-0 items-center justify-between gap-2">
            <span className="truncate">{n.title}</span>
            {n.badge === undefined ? null : <span className="shrink-0 text-xs text-muted tabular-nums">{n.badge}</span>}
          </span>
        )
        const rowClass = `block min-w-0 flex-1 rounded-md px-2 py-1 ${
          here ? 'bg-surface font-medium text-accent' : 'text-fg-secondary hover:bg-surface hover:text-fg'
        }`
        return (
          <li key={n.id}>
            <div className="flex items-center">
              {inner ? (
                <button
                  type="button"
                  aria-label={opened ? `Close ${n.title}` : `Open ${n.title}`}
                  aria-expanded={opened}
                  onClick={() => toggle(n.id)}
                  className="flex size-6 shrink-0 items-center justify-center rounded text-muted hover:text-fg"
                >
                  <svg
                    viewBox="0 0 24 24"
                    width="12"
                    height="12"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    aria-hidden="true"
                    className={opened ? 'rotate-90 transition-transform' : 'transition-transform'}
                  >
                    <path d="m9 18 6-6-6-6" />
                  </svg>
                </button>
              ) : (
                <span className="size-6 shrink-0" />
              )}
              {n.path === undefined ? (
                <button
                  type="button"
                  onClick={() => toggle(n.id)}
                  className={`${rowClass} text-left ${depth === 0 ? 'font-medium text-fg' : ''}`}
                >
                  {label}
                </button>
              ) : (
                <Link to={n.path} current={here} className={`${rowClass} ${depth === 0 && inner ? 'font-medium' : ''}`}>
                  {label}
                </Link>
              )}
            </div>
            {opened ? (
              <TocList nodes={n.children ?? []} depth={depth + 1} current={current} isOpen={isOpen} toggle={toggle} />
            ) : null}
          </li>
        )
      })}
    </ul>
  )
}
