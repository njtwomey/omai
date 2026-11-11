import type { ReactNode } from 'react'
import { Link } from '../../site/src/shell/Link.tsx'
import { navigate } from '../../site/src/shell/router.ts'
import { GROUPS } from './groups.ts'
import { groupPocs } from './poc.ts'
import { POCS } from './registry.ts'

const SELECT =
  'h-7 max-w-[16rem] truncate rounded-md border border-transparent bg-transparent px-1.5 text-sm text-fg hover:border-line focus-visible:outline-2 focus-visible:outline-(--wb-accent)/60'

/**
 * The trail in the top bar: home, then the group and the example with a drop-down each, so a move between groups or between
 * examples is one choice from anywhere.
 *
 * @param props - The current group and example slugs, when there are any.
 * @returns The trail.
 */
export function Crumbs(props: { readonly group?: string | undefined; readonly slug?: string | undefined }): ReactNode {
  const entry = props.slug === undefined ? undefined : POCS.find((p) => p.slug === props.slug)
  const group = entry?.group ?? props.group
  const sections = groupPocs(POCS, GROUPS)
  const siblings = POCS.filter((p) => p.group === group)
  const sep = <span className="text-muted">/</span>
  return (
    <nav aria-label="Where you are" className="flex min-w-0 items-center gap-1 text-sm">
      <Link to="/" className="rounded-md px-1.5 py-1 text-muted hover:bg-surface hover:text-fg">
        Examples
      </Link>
      {sep}
      <select
        aria-label="Group"
        className={SELECT}
        value={group ?? ''}
        onChange={(e) => navigate(e.target.value === '' ? '/' : `/group/${e.target.value}`)}
      >
        <option value="">All groups</option>
        {sections.map((s) => (
          <option key={s.group?.slug ?? 'more'} value={s.group?.slug ?? ''}>
            {s.group?.title ?? 'More'}
          </option>
        ))}
      </select>
      {entry === undefined ? null : (
        <>
          {sep}
          <select
            aria-label="Example"
            className={SELECT}
            value={entry.slug}
            onChange={(e) => navigate(`/poc/${e.target.value}`)}
          >
            {siblings.map((p) => (
              <option key={p.slug} value={p.slug}>
                {p.title}
              </option>
            ))}
          </select>
        </>
      )}
    </nav>
  )
}
