import { type ReactNode, useMemo, useState } from 'react'
import { Link } from '../shell/Link.tsx'
import type { TocNode } from './toc.ts'

const MAX_ROWS = 400

/** Whether a node is the current page or one of its ancestors (paths mirror the tree, so a prefix says it). */
function onPath(node: TocNode, path: string): boolean {
  return node.path !== undefined && (path === node.path || path.startsWith(`${node.path}/`))
}

/** The ids of the nodes to show for a filter: the matches and their ancestors. */
function shownFor(tree: readonly TocNode[], q: string): Set<string> {
  const out = new Set<string>()
  const visit = (n: TocNode): boolean => {
    let hit = n.title.toLowerCase().includes(q)
    for (const c of n.children ?? []) if (visit(c)) hit = true
    if (hit) out.add(n.id)
    return hit
  }
  for (const n of tree) visit(n)
  return out
}

/**
 * A temporary table of contents (until the shared one lands): everything collapsed except the path to the current
 * page, the current item highlighted, a chevron (or a second click on the current folder) toggles a group, and the
 * filter opens the groups with matches.
 */
export function Sidebar(props: { readonly tree: readonly TocNode[]; readonly path: string }): ReactNode {
  const { tree, path } = props
  const [query, setQuery] = useState('')
  const [toggled, setToggled] = useState<ReadonlyMap<string, boolean>>(new Map())
  // Navigating opens the new page's ancestors: forget any choice to close them (state adjusted during render).
  const [seen, setSeen] = useState(path)
  if (seen !== path) {
    setSeen(path)
    setToggled(new Map([...toggled].filter(([id]) => id !== 'whitebox-engine' && !path.startsWith(`/docs/${id}`))))
  }
  const q = query.trim().toLowerCase()
  const shown = useMemo(() => (q === '' ? null : shownFor(tree, q)), [tree, q])
  const toggle = (id: string, open: boolean): void => setToggled(new Map([...toggled, [id, !open]]))
  let rows = 0
  const render = (node: TocNode, depth: number): ReactNode => {
    if (shown !== null && !shown.has(node.id)) return null
    if (rows++ >= MAX_ROWS) return null
    const here = node.path === path
    const open = shown !== null ? true : (toggled.get(node.id) ?? onPath(node, path))
    const kids = node.children ?? []
    const label =
      node.path === undefined ? (
        <span>{node.title}</span>
      ) : (
        <Link
          to={node.path}
          className={`block truncate ${here ? 'font-medium text-sky-700 dark:text-sky-400' : 'hover:text-sky-700 dark:hover:text-sky-400'}`}
        >
          {node.title}
        </Link>
      )
    return (
      <li key={node.id}>
        <div
          className={`flex items-center gap-1 rounded py-0.5 pr-1 ${here ? 'bg-neutral-100 dark:bg-neutral-800' : ''}`}
          style={{ paddingLeft: `${depth * 0.75}rem` }}
        >
          {kids.length === 0 ? (
            <span className="inline-block w-4 shrink-0" />
          ) : (
            <button
              type="button"
              aria-label={open ? `Collapse ${node.title}` : `Expand ${node.title}`}
              aria-expanded={open}
              onClick={() => toggle(node.id, open)}
              className="w-4 shrink-0 text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100"
            >
              {open ? '▾' : '▸'}
            </button>
          )}
          {/* A click on the current folder's name toggles it; on any other, it navigates (which opens it). */}
          {/* biome-ignore lint/a11y/noStaticElementInteractions: the link inside is the interactive element */}
          {/* biome-ignore lint/a11y/useKeyWithClickEvents: the link inside handles the keyboard */}
          <span className="min-w-0 flex-1" onClick={() => here && kids.length > 0 && toggle(node.id, open)}>
            {label}
          </span>
        </div>
        {open && kids.length > 0 ? <ul>{kids.map((c) => render(c, depth + 1))}</ul> : null}
      </li>
    )
  }
  return (
    <nav className="text-sm">
      <input
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Filter"
        aria-label="Filter the table of contents"
        className="mb-3 w-full rounded border border-neutral-300 bg-transparent px-2 py-1 dark:border-neutral-700"
      />
      <ul>{tree.map((n) => render(n, 0))}</ul>
      {rows > MAX_ROWS ? <p className="mt-2 text-xs text-neutral-500">More matches: narrow the filter.</p> : null}
    </nav>
  )
}
