import { type ReactNode, useDeferredValue, useState } from 'react'
import { Link } from '../../site/src/shell/Link.tsx'
import { type PocEntry, searchPocs } from './poc.ts'
import { POCS } from './registry.ts'

function Card(props: { readonly entry: PocEntry }): ReactNode {
  const { entry } = props
  return (
    <Link to={`/poc/${entry.slug}`} className="rounded-lg border border-line p-5 hover:border-muted">
      <h2 className="font-semibold">{entry.title}</h2>
      <p className="mt-1 text-sm text-muted">{entry.summary}</p>
      <p className="mt-3 flex flex-wrap gap-1.5">
        {entry.tags.map((t) => (
          <span key={t} className="rounded-md bg-surface px-1.5 py-0.5 text-xs text-muted">
            {t}
          </span>
        ))}
      </p>
    </Link>
  )
}

/** The front page: a search box over the proofs of concept, and a card for each match. */
export function Index(): ReactNode {
  const [query, setQuery] = useState('')
  const shown = searchPocs(POCS, useDeferredValue(query))
  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-xl font-semibold tracking-tight">Proofs of concept</h1>
        <p className="max-w-3xl text-sm text-muted">
          Single live pages of what the library can do, on their own so the site stays small. Each is added when it is
          wanted: ask for one to be ported in.
        </p>
      </div>
      <input
        type="search"
        aria-label="Search the proofs of concept"
        placeholder={`Search ${POCS.length} proof${POCS.length === 1 ? '' : 's'} of concept`}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="h-9 w-full max-w-xl rounded-md border border-line bg-bg px-3 text-sm text-fg outline-none focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50"
      />
      {shown.length === 0 ? (
        <p className="text-sm text-muted">Nothing matches “{query}”.</p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {shown.map((e) => (
            <Card key={e.slug} entry={e} />
          ))}
        </div>
      )}
    </div>
  )
}
