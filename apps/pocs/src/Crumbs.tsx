import type { ReactNode } from 'react'
import { Link } from '../../site/src/shell/Link.tsx'
import { navigate } from '../../site/src/shell/router.ts'
import { GROUPS } from './groups.ts'
import { groupPocs } from './poc.ts'
import { POCS } from './registry.ts'

const SWITCH =
  'absolute inset-0 h-full w-full cursor-pointer appearance-none bg-transparent text-transparent opacity-0 [&>option]:bg-bg [&>option]:text-fg'

/**
 * A crumb that is a link to its page with a chevron beside it that switches to a sibling: the label goes to the page (also
 * when it is the page you are on), the chevron offers the others.
 */
function Crumb(props: {
  readonly to: string
  readonly label: string
  readonly switchLabel: string
  readonly value: string
  readonly options: readonly { readonly value: string; readonly label: string }[]
  readonly onSwitch: (value: string) => void
  readonly current?: boolean
}): ReactNode {
  return (
    <span className="flex min-w-0 items-center">
      <Link
        to={props.to}
        {...(props.current === true ? { current: true } : {})}
        className="min-w-0 max-w-[16rem] truncate rounded-md px-1.5 py-1 text-fg hover:bg-surface"
      >
        {props.label}
      </Link>
      <span className="relative flex size-6 shrink-0 items-center justify-center rounded-md text-muted hover:bg-surface hover:text-fg">
        <svg
          viewBox="0 0 24 24"
          width="12"
          height="12"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          aria-hidden="true"
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
        <select
          aria-label={props.switchLabel}
          className={SWITCH}
          value={props.value}
          onChange={(e) => props.onSwitch(e.target.value)}
        >
          {props.options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </span>
    </span>
  )
}

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
      <Crumb
        to={group === undefined ? '/' : `/group/${group}`}
        label={sections.find((x) => (x.group?.slug ?? '') === (group ?? ''))?.group?.title ?? 'All groups'}
        switchLabel="Switch group"
        value={group ?? ''}
        options={[
          { value: '', label: 'All groups' },
          ...sections.map((x) => ({ value: x.group?.slug ?? '', label: x.group?.title ?? 'More' })),
        ]}
        onSwitch={(v) => navigate(v === '' ? '/' : `/group/${v}`)}
      />
      {entry === undefined ? null : (
        <>
          {sep}
          <Crumb
            to={`/poc/${entry.slug}`}
            label={entry.title}
            switchLabel="Switch example"
            value={entry.slug}
            options={siblings.map((p) => ({ value: p.slug, label: p.title }))}
            onSwitch={(v) => navigate(`/poc/${v}`)}
            current
          />
        </>
      )}
    </nav>
  )
}
