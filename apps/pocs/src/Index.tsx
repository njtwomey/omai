import { type ReactNode, useDeferredValue, useState } from 'react'
import { Link } from '../../site/src/shell/Link.tsx'
import { GROUPS } from './groups.ts'
import { type PocEntry, searchPocs } from './poc.ts'
import { POCS } from './registry.ts'

const TILE = 'flex flex-col gap-1 rounded-lg border border-line p-4 hover:border-muted hover:bg-surface'

/** One example: its title and two lines of what it shows. */
function Tile(props: { readonly entry: PocEntry; readonly group?: string | undefined }): ReactNode {
  const { entry } = props
  return (
    <Link to={`/poc/${entry.slug}`} className={TILE}>
      <h2 className="font-semibold leading-snug">{entry.title}</h2>
      <p className="line-clamp-2 text-sm text-muted">{entry.summary}</p>
      {props.group === undefined ? null : <p className="mt-1 text-xs text-muted">{props.group}</p>}
    </Link>
  )
}

/** A group: its name, three lines of what it holds and how many examples it has. */
function GroupTile(props: { readonly slug: string; readonly title: string; readonly summary: string }): ReactNode {
  const count = POCS.filter((p) => p.group === props.slug).length
  return (
    <Link to={`/group/${props.slug}`} className={`${TILE} min-h-32`}>
      <h2 className="text-lg font-semibold leading-snug">{props.title}</h2>
      <p className="line-clamp-3 text-sm text-muted">{props.summary}</p>
      <p className="mt-auto pt-2 text-xs text-muted">
        {count} example{count === 1 ? '' : 's'}
      </p>
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
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h1 className="text-xl font-semibold tracking-tight">{only?.title ?? 'Examples'}</h1>
        <input
          type="search"
          aria-label="Search the examples"
          placeholder="Search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="ml-auto h-9 w-56 rounded-md border border-line bg-bg px-2.5 text-sm text-fg outline-none focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50"
        />
      </div>
      {only === undefined ? null : <p className="-mt-3 max-w-3xl text-muted">{only.summary}</p>}
      {searching || only !== undefined ? (
        shown.length === 0 ? (
          <p className="text-sm text-muted">{searching ? `Nothing matches “${query}”.` : 'Nothing here yet.'}</p>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {shown.map((e) => (
              <Tile
                key={e.slug}
                entry={e}
                group={only === undefined ? GROUPS.find((g) => g.slug === e.group)?.title : undefined}
              />
            ))}
          </div>
        )
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {GROUPS.filter((g) => POCS.some((p) => p.group === g.slug)).map((g) => (
            <GroupTile key={g.slug} slug={g.slug} title={g.title} summary={g.summary} />
          ))}
          {ungrouped.map((e) => (
            <Tile key={e.slug} entry={e} />
          ))}
        </div>
      )}
    </div>
  )
}
