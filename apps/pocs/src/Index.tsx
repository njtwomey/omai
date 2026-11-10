import { type ReactNode, useDeferredValue, useState } from 'react'
import { Link } from '../../site/src/shell/Link.tsx'
import { GROUPS } from './groups.ts'
import { groupPocs, type PocEntry, type PocGroup, searchPocs } from './poc.ts'
import { POCS } from './registry.ts'

function Card(props: { readonly entry: PocEntry; readonly showGroup: boolean }): ReactNode {
  const { entry } = props
  const group = props.showGroup ? GROUPS.find((g) => g.slug === entry.group) : undefined
  return (
    <Link to={`/poc/${entry.slug}`} className="rounded-lg border border-line p-5 hover:border-muted">
      <h2 className="font-semibold">{entry.title}</h2>
      <p className="mt-1 text-sm text-muted">{entry.summary}</p>
      <p className="mt-3 flex flex-wrap gap-1.5">
        {group === undefined ? null : (
          <span className="rounded-md bg-(--wb-accent)/15 px-1.5 py-0.5 text-xs text-fg">{group.title}</span>
        )}
        {entry.tags.map((t) => (
          <span key={t} className="rounded-md bg-surface px-1.5 py-0.5 text-xs text-muted">
            {t}
          </span>
        ))}
      </p>
    </Link>
  )
}

function Cards(props: { readonly entries: readonly PocEntry[]; readonly showGroup: boolean }): ReactNode {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {props.entries.map((e) => (
        <Card key={e.slug} entry={e} showGroup={props.showGroup} />
      ))}
    </div>
  )
}

/**
 * The front page: a search box over the proofs of concept, and their cards in one section per group (a flat list with a
 * group label on each card while a search is typed). With `group`, only that group's page (`/group/<slug>`).
 */
export function Index(props: { readonly group?: string } = {}): ReactNode {
  const [query, setQuery] = useState('')
  const searching = useDeferredValue(query).trim() !== ''
  const pool = props.group === undefined ? POCS : POCS.filter((p) => p.group === props.group)
  const shown = searchPocs(pool, useDeferredValue(query))
  const only: PocGroup | undefined = props.group === undefined ? undefined : GROUPS.find((g) => g.slug === props.group)
  const sections = groupPocs(shown, GROUPS)
  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        {only === undefined ? null : (
          <Link to="/" className="text-xs text-muted hover:text-fg">
            ← all proofs of concept
          </Link>
        )}
        <h1 className="text-xl font-semibold tracking-tight">{only?.title ?? 'Proofs of concept'}</h1>
        <p className="max-w-3xl text-sm text-muted">
          {only?.summary ??
            'Single live pages of what the library can do, on their own so the site stays small, grouped by area. Each is added when it is wanted: ask for one to be ported in.'}
        </p>
      </div>
      <input
        type="search"
        aria-label="Search the proofs of concept"
        placeholder={`Search ${pool.length} proof${pool.length === 1 ? '' : 's'} of concept`}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="h-9 w-full max-w-xl rounded-md border border-line bg-bg px-3 text-sm text-fg outline-none focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50"
      />
      {shown.length === 0 ? (
        <p className="text-sm text-muted">{query === '' ? 'Nothing here yet.' : `Nothing matches “${query}”.`}</p>
      ) : searching || only !== undefined ? (
        <Cards entries={shown} showGroup={only === undefined} />
      ) : (
        sections.map((s) => (
          <section key={s.group?.slug ?? 'more'} className="flex flex-col gap-3">
            <div className="flex flex-col gap-0.5">
              <h2 className="text-base font-semibold">
                {s.group === undefined ? (
                  'More'
                ) : (
                  <Link to={`/group/${s.group.slug}`} className="hover:underline">
                    {s.group.title}
                  </Link>
                )}{' '}
                <span className="text-sm font-normal text-muted">{s.entries.length}</span>
              </h2>
              {s.group === undefined ? null : <p className="max-w-3xl text-sm text-muted">{s.group.summary}</p>}
            </div>
            <Cards entries={s.entries} showGroup={false} />
          </section>
        ))
      )}
    </div>
  )
}
