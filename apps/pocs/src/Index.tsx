import { type ReactNode, useDeferredValue, useState } from 'react'
import { Link } from '../../site/src/shell/Link.tsx'
import { GROUPS } from './groups.ts'
import { type PocEntry, searchPocs } from './poc.ts'
import { POCS } from './registry.ts'

/** One example as a row: its title and one line. */
function Row(props: { readonly entry: PocEntry; readonly group?: string | undefined }): ReactNode {
  const { entry } = props
  return (
    <Link
      to={`/poc/${entry.slug}`}
      className="flex items-baseline gap-3 rounded-md px-3 py-2 hover:bg-surface"
      title={entry.summary}
    >
      <span className="shrink-0 text-sm font-medium">{entry.title}</span>
      <span className="min-w-0 flex-1 truncate text-sm text-muted">{entry.summary}</span>
      {props.group === undefined ? null : <span className="shrink-0 text-xs text-muted">{props.group}</span>}
    </Link>
  )
}

/** A group as a row: its name, its one line and how many examples it holds. */
function GroupRow(props: { readonly slug: string; readonly title: string; readonly summary: string }): ReactNode {
  const count = POCS.filter((p) => p.group === props.slug).length
  return (
    <Link to={`/group/${props.slug}`} className="flex items-baseline gap-3 rounded-md px-3 py-2.5 hover:bg-surface">
      <span className="shrink-0 font-medium">{props.title}</span>
      <span className="min-w-0 flex-1 truncate text-sm text-muted">{props.summary}</span>
      <span className="shrink-0 text-sm tabular-nums text-muted">{count}</span>
    </Link>
  )
}

/**
 * The front page is the groups, one row each; a group's page (`/group/<slug>`) is its examples, one row each; a search
 * lists matches flat. The trail in the top bar moves between them.
 */
export function Index(props: { readonly group?: string } = {}): ReactNode {
  const [query, setQuery] = useState('')
  const q = useDeferredValue(query)
  const only = props.group === undefined ? undefined : GROUPS.find((g) => g.slug === props.group)
  const pool = props.group === undefined ? POCS : POCS.filter((p) => p.group === props.group)
  const shown = searchPocs(pool, q)
  const searching = q.trim() !== ''
  const ungrouped = POCS.filter((p) => p.group === undefined || !GROUPS.some((g) => g.slug === p.group))
  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-4">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h1 className="text-lg font-semibold tracking-tight">{only?.title ?? 'Examples'}</h1>
        <input
          type="search"
          aria-label="Search the examples"
          placeholder="Search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="ml-auto h-8 w-48 rounded-md border border-line bg-bg px-2.5 text-sm text-fg outline-none focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50"
        />
      </div>
      {only === undefined ? null : <p className="-mt-2 max-w-3xl text-sm text-muted">{only.summary}</p>}
      {searching || only !== undefined ? (
        shown.length === 0 ? (
          <p className="text-sm text-muted">{searching ? `Nothing matches “${query}”.` : 'Nothing here yet.'}</p>
        ) : (
          <div className="flex flex-col">
            {shown.map((e) => (
              <Row
                key={e.slug}
                entry={e}
                group={only === undefined ? GROUPS.find((g) => g.slug === e.group)?.title : undefined}
              />
            ))}
          </div>
        )
      ) : (
        <div className="flex flex-col divide-y divide-line">
          {GROUPS.filter((g) => POCS.some((p) => p.group === g.slug)).map((g) => (
            <GroupRow key={g.slug} slug={g.slug} title={g.title} summary={g.summary} />
          ))}
          {ungrouped.length === 0 ? null : (
            <div className="py-1">
              {ungrouped.map((e) => (
                <Row key={e.slug} entry={e} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
