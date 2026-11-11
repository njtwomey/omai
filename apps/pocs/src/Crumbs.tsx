import { type ReactNode, useEffect, useRef, useState } from 'react'
import { Link } from '../../site/src/shell/Link.tsx'
import { currentPath } from '../../site/src/shell/router.ts'
import { GROUPS } from './groups.ts'
import { inDisplayOrder } from './order.ts'
import { groupPocs } from './poc.ts'
import { POCS } from './registry.ts'

/** One choice of a crumb's menu: where it goes, what it is called. */
interface Choice {
  readonly to: string
  readonly label: string
  readonly current: boolean
}

/**
 * A crumb is one button: its label is where you are, and a click opens the whole list at once. A choice is a link to its
 * page; choosing the page you are already on refreshes it. Escape or a click elsewhere closes the list.
 */
function CrumbMenu(props: {
  readonly label: string
  readonly title: string
  readonly choices: readonly Choice[]
}): ReactNode {
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    if (!open) return
    const away = (e: MouseEvent): void => {
      if (box.current !== null && !box.current.contains(e.target as Node)) setOpen(false)
    }
    const key = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', away)
    document.addEventListener('keydown', key)
    return () => {
      document.removeEventListener('mousedown', away)
      document.removeEventListener('keydown', key)
    }
  }, [open])
  return (
    <span ref={box} className="relative flex min-w-0 items-center">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        title={props.title}
        onClick={() => setOpen((o) => !o)}
        className="flex min-w-0 max-w-[18rem] items-center gap-1 rounded-md px-1.5 py-1 text-fg hover:bg-surface"
      >
        <span className="truncate">{props.label}</span>
        <svg
          viewBox="0 0 24 24"
          width="12"
          height="12"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          aria-hidden="true"
          className="shrink-0 text-muted"
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>
      {open ? (
        <div
          role="menu"
          className="absolute left-0 top-full z-50 mt-1 max-h-[70vh] w-72 overflow-y-auto rounded-md border border-line bg-bg p-1 shadow-lg"
        >
          {props.choices.map((c) => (
            <Link
              key={c.to}
              to={c.to}
              current={c.current}
              className={`block truncate rounded px-2 py-1.5 text-sm hover:bg-surface ${c.current ? 'font-medium text-fg' : 'text-muted hover:text-fg'}`}
              onChoose={() => {
                setOpen(false)
                // The page you are on: a plain refresh of it.
                if (c.to === currentPath()) window.location.reload()
              }}
            >
              {c.label}
            </Link>
          ))}
        </div>
      ) : null}
    </span>
  )
}

/** A previous or next arrow: a link to the neighbouring example, or a dimmed arrow at the end of the list. */
function Step(props: {
  readonly to: string | undefined
  readonly label: string
  readonly title: string | undefined
  readonly direction: 'previous' | 'next'
}): ReactNode {
  const arrow = (
    <svg
      viewBox="0 0 24 24"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      aria-hidden="true"
    >
      <path d={props.direction === 'previous' ? 'm15 6-6 6 6 6' : 'm9 6 6 6-6 6'} />
    </svg>
  )
  const base = 'flex size-7 shrink-0 items-center justify-center rounded-md'
  if (props.to === undefined)
    return (
      <span aria-hidden="true" className={`${base} text-muted opacity-30`}>
        {arrow}
      </span>
    )
  return (
    <Link
      to={props.to}
      title={`${props.label}: ${props.title ?? ''}`}
      className={`${base} text-muted hover:bg-surface hover:text-fg`}
    >
      {arrow}
      <span className="sr-only">{props.label}</span>
    </Link>
  )
}

/**
 * The trail in the top bar: home, then the group and the example, each one button that opens the whole list (one click, no
 * separate chevron to find), so a move between groups or between examples is one choice from anywhere; choosing where you
 * already are refreshes the page. Previous and next arrows go through a group one example after another.
 *
 * @param props - The current group and example slugs, when there are any.
 * @returns The trail.
 */
export function Crumbs(props: { readonly group?: string | undefined; readonly slug?: string | undefined }): ReactNode {
  const entry = props.slug === undefined ? undefined : POCS.find((p) => p.slug === props.slug)
  const group = entry?.group ?? props.group
  const sections = groupPocs(POCS, GROUPS)
  const siblings = inDisplayOrder(group)
  const at = entry === undefined ? -1 : siblings.findIndex((p) => p.slug === entry.slug)
  const previous = at > 0 ? siblings[at - 1] : undefined
  const next = at >= 0 ? siblings[at + 1] : undefined
  const sep = <span className="text-muted">/</span>
  const groupChoices: Choice[] = [
    { to: '/', label: 'All groups', current: group === undefined },
    ...sections.map((s) => ({
      to: s.group === undefined ? '/' : `/group/${s.group.slug}`,
      label: s.group?.title ?? 'More',
      current: (s.group?.slug ?? '') === (group ?? '-'),
    })),
  ]
  return (
    <nav aria-label="Where you are" className="flex min-w-0 items-center gap-1 text-sm">
      <Link to="/" className="rounded-md px-1.5 py-1 text-muted hover:bg-surface hover:text-fg">
        Examples
      </Link>
      {sep}
      <CrumbMenu
        label={sections.find((x) => (x.group?.slug ?? '') === (group ?? ''))?.group?.title ?? 'All groups'}
        title="Choose a group"
        choices={groupChoices}
      />
      {entry === undefined ? null : (
        <>
          {sep}
          <CrumbMenu
            label={entry.title}
            title="Choose an example"
            choices={siblings.map((p) => ({ to: `/poc/${p.slug}`, label: p.title, current: p.slug === entry.slug }))}
          />
          <Step
            to={previous === undefined ? undefined : `/poc/${previous.slug}`}
            label="Previous example"
            title={previous?.title}
            direction="previous"
          />
          <Step
            to={next === undefined ? undefined : `/poc/${next.slug}`}
            label="Next example"
            title={next?.title}
            direction="next"
          />
        </>
      )}
    </nav>
  )
}
