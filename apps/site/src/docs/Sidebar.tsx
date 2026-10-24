import { type ReactNode, useState } from 'react'
import { Link } from '../shell/Link.tsx'
import type { TocNode } from './toc.ts'

function matches(node: TocNode, q: string): boolean {
  return node.title.toLowerCase().includes(q) || (node.children ?? []).some((c) => matches(c, q))
}

function contains(node: TocNode, path: string): boolean {
  return node.path === path || (node.children ?? []).some((c) => contains(c, path))
}

function Node(props: { readonly node: TocNode; readonly path: string; readonly q: string }): ReactNode {
  const { node, path, q } = props
  if (q !== '' && !matches(node, q)) return null
  const here = node.path === path
  const label =
    node.path === undefined ? (
      <span className="font-medium text-neutral-500">{node.title}</span>
    ) : (
      <Link
        to={node.path}
        className={here ? 'font-medium text-sky-700 dark:text-sky-400' : 'hover:text-sky-700 dark:hover:text-sky-400'}
      >
        {node.title}
      </Link>
    )
  if (node.children === undefined) return <li className="truncate py-0.5">{label}</li>
  // Collapsed by default: open only the ancestors of the current page, and, while filtering, groups with matches.
  const open = q !== '' || contains(node, path)
  return (
    <li className="py-0.5">
      <details open={open}>
        <summary className="cursor-pointer truncate">{label}</summary>
        <ul className="ml-2 border-l border-neutral-200 pl-2 dark:border-neutral-800">
          {node.children.map((c) => (
            <Node key={c.id} node={c} path={path} q={q} />
          ))}
        </ul>
      </details>
    </li>
  )
}

/** A temporary table of contents (until the shared sidebar lands): collapsible groups with a filter. */
export function Sidebar(props: { readonly tree: readonly TocNode[]; readonly path: string }): ReactNode {
  const [q, setQ] = useState('')
  return (
    <nav className="text-sm">
      <input
        type="search"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Filter"
        className="mb-3 w-full rounded border border-neutral-300 bg-transparent px-2 py-1 dark:border-neutral-700"
      />
      <ul>
        {props.tree.map((n) => (
          <Node key={n.id} node={n} path={props.path} q={q.trim().toLowerCase()} />
        ))}
      </ul>
    </nav>
  )
}
