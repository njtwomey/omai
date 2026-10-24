import { type ReactNode, useEffect } from 'react'
import { Link } from './Link.tsx'
import { navigate } from './router.ts'
import type { TocNode } from './toc.ts'

/** A page in reading order. */
export interface PageRef {
  readonly path: string
  readonly title: string
}

/**
 * The pages of a contents tree in reading order: depth first, each node with a path before its children, so a section's
 * index comes before its pages and the last page of one group leads to the next group.
 *
 * @param nodes - The tree (the same one {@link TocSidebar} draws).
 * @param home - A page before the tree (the section's front page), if any.
 * @returns The pages in order.
 */
export function pageOrder(nodes: readonly TocNode[], home?: PageRef): PageRef[] {
  const out: PageRef[] = home === undefined ? [] : [home]
  const walk = (list: readonly TocNode[]): void => {
    for (const n of list) {
      if (n.path !== undefined && !out.some((p) => p.path === n.path)) out.push({ path: n.path, title: n.title })
      if (n.children !== undefined) walk(n.children)
    }
  }
  walk(nodes)
  return out
}

/** The pages before and after `current` in reading order. */
export function neighbours(
  order: readonly PageRef[],
  current: string,
): { readonly prev?: PageRef; readonly next?: PageRef } {
  const at = order.findIndex((p) => p.path === current)
  if (at < 0) return {}
  const prev = order[at - 1]
  const next = order[at + 1]
  return { ...(prev === undefined ? {} : { prev }), ...(next === undefined ? {} : { next }) }
}

const typing = (target: EventTarget | null): boolean =>
  target instanceof HTMLElement &&
  (target.isContentEditable ||
    ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) ||
    target.closest('[role="slider"], [role="radiogroup"], [role="button"]') !== null)

/**
 * Previous and next links over a contents tree (the order of {@link pageOrder}), with the Left and Right arrow keys
 * bound to them (ignored while typing or on a control that uses the arrows). `compact` draws only two arrow buttons,
 * for the top right of a page, and binds no keys (the full links at the foot of the page do).
 *
 * @param props - The tree, the section's front page, the current path and the style.
 * @returns The links.
 */
export function PrevNext(props: {
  readonly nodes: readonly TocNode[]
  readonly current: string
  readonly home?: PageRef
  readonly compact?: boolean
}): ReactNode {
  const { nodes, current, home, compact = false } = props
  const { prev, next } = neighbours(pageOrder(nodes, home), current)
  useEffect(() => {
    if (compact) return
    const onKey = (e: KeyboardEvent): void => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || e.shiftKey || typing(e.target)) return
      const to = e.key === 'ArrowLeft' ? prev : e.key === 'ArrowRight' ? next : undefined
      if (to === undefined) return
      e.preventDefault()
      navigate(to.path)
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [compact, prev, next])
  if (compact)
    return (
      <nav aria-label="Previous and next pages" className="flex items-center gap-1">
        {[
          { to: prev, label: 'Previous', d: 'm15 18-6-6 6-6' },
          { to: next, label: 'Next', d: 'm9 18 6-6-6-6' },
        ].map(({ to, label, d }) =>
          to === undefined ? (
            <span key={label} className="size-7" />
          ) : (
            <Link
              key={label}
              to={to.path}
              title={`${label}: ${to.title} (${label === 'Previous' ? '←' : '→'})`}
              className="flex size-7 items-center justify-center rounded-md border border-line text-muted hover:bg-surface hover:text-fg"
            >
              <svg
                viewBox="0 0 24 24"
                width="14"
                height="14"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                aria-hidden="true"
              >
                <path d={d} />
              </svg>
              <span className="sr-only">{label}</span>
            </Link>
          ),
        )}
      </nav>
    )
  const link = (to: PageRef, dir: 'prev' | 'next'): ReactNode => (
    <Link
      to={to.path}
      className={`flex min-w-0 max-w-[48%] flex-col rounded-lg border border-line px-4 py-2.5 hover:border-muted hover:bg-surface ${dir === 'next' ? 'ml-auto items-end text-right' : ''}`}
    >
      <span className="text-xs text-muted">{dir === 'prev' ? '← Previous' : 'Next →'}</span>
      <span className="max-w-full truncate text-sm font-medium text-fg">{to.title}</span>
    </Link>
  )
  return (
    <nav aria-label="Previous and next pages" className="flex gap-4 border-t border-line pt-6">
      {prev === undefined ? null : link(prev, 'prev')}
      {next === undefined ? null : link(next, 'next')}
    </nav>
  )
}
