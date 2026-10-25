import { type ReactNode, useMemo, useState } from 'react'
import { Link } from '../shell/Link.tsx'
import { DocInline } from './DocText.tsx'
import { type Docs, docsPath, type IndexFile, type IndexSymbol, symbolPath, useSearch } from './data.ts'
import type { WireSearch } from './wire.ts'

interface Hit {
  readonly file: IndexFile
  readonly symbol?: IndexSymbol
  readonly summary: string
  readonly score: number
}

const LIMIT = 60

/**
 * Rank exports and files for a query: an exact name first, then names that start with it, then names that contain it,
 * then summaries that contain every word of it (once the search index of summaries has loaded).
 */
function searchDocs(docs: Docs, search: WireSearch | null, query: string): Hit[] {
  const q = query.trim().toLowerCase()
  if (q === '') return []
  const words = q.split(/\s+/)
  const hits: Hit[] = []
  const rank = (name: string, summary: string): number => {
    const n = name.toLowerCase()
    if (n === q) return 0
    if (n.startsWith(q)) return 1
    if (n.includes(q)) return 2
    const s = summary.toLowerCase()
    return words.every((w) => s.includes(w) || n.includes(w)) ? 3 : -1
  }
  docs.files.forEach((file, i) => {
    const name = file.path.slice(file.path.lastIndexOf('/') + 1)
    const f = rank(name, file.summary)
    if (f >= 0) hits.push({ file, summary: file.summary, score: f + 0.5 })
    const summaries = search?.symbols[i]
    file.symbols.forEach((symbol, j) => {
      const summary = summaries?.[j] ?? ''
      const r = rank(symbol.name, summary)
      if (r >= 0) hits.push({ file, symbol, summary, score: r })
    })
  })
  return hits.sort((a, b) => a.score - b.score || (a.symbol?.name ?? '').length - (b.symbol?.name ?? '').length)
}

/**
 * A search box over every export's name and summary and every file's name and comment, with ranked results. The
 * summaries load when the box gets focus (or once the page is idle); names match from the start.
 */
export function Search(props: { readonly docs: Docs }): ReactNode {
  const [query, setQuery] = useState('')
  const [wanted, setWanted] = useState(false)
  const search = useSearch(props.docs, wanted)
  const hits = useMemo(() => searchDocs(props.docs, search, query), [props.docs, search, query])
  return (
    <section className="space-y-3">
      <input
        type="search"
        value={query}
        onChange={(e) => {
          setWanted(true)
          setQuery(e.target.value)
        }}
        onFocus={() => setWanted(true)}
        placeholder="Search exports and files by name or description"
        aria-label="Search the docs"
        className="h-10 w-full rounded-lg border border-line bg-bg px-3 text-base outline-none placeholder:text-muted focus:border-accent"
      />
      {query.trim() === '' ? null : (
        <div>
          <p className="mb-2 text-sm text-muted">
            {hits.length === 0
              ? 'Nothing matches.'
              : `${hits.length} matches${hits.length > LIMIT ? `, the first ${LIMIT} shown` : ''}`}
          </p>
          <ul className="divide-y divide-line">
            {hits.slice(0, LIMIT).map((h) => (
              <li
                key={h.symbol === undefined ? h.file.path : `${h.file.path}#${h.symbol.name}`}
                className="py-1.5 text-sm"
              >
                <Link
                  to={h.symbol === undefined ? docsPath(h.file.path) : symbolPath(h.file.path, h.symbol.name)}
                  className="font-mono text-accent hover:underline"
                >
                  {h.symbol?.name ?? h.file.path.slice(h.file.path.lastIndexOf('/') + 1)}
                </Link>
                <span className="ml-2 font-mono text-xs text-muted">{h.file.path}</span>
                <span className="block text-fg-secondary">
                  <DocInline text={h.summary} docs={props.docs} near={h.file.module} />
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}
