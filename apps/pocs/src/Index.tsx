import { type ReactNode, useDeferredValue, useState } from 'react'
import { Link } from '../../site/src/shell/Link.tsx'
import { type DirNode, dirHref, dirTitles, findDir, type PocEntry, pocHref, searchPocs } from './poc.ts'
import { POCS, TREE } from './registry.ts'

const TILE = 'flex flex-col gap-1 rounded-lg border border-line p-4 hover:border-muted hover:bg-surface'

const plural = (n: number, word: string): string => `${n} ${word}${n === 1 ? '' : 's'}`

/** A small glyph at the start of a tile's title: a folder for a directory, a page for an example. */
function Glyph(props: { readonly kind: 'folder' | 'page' }): ReactNode {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="mt-[0.2em] shrink-0 text-muted"
    >
      {props.kind === 'folder' ? (
        <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      ) : (
        <>
          <path d="M6 3h8l4 4v14H6z" />
          <path d="M14 3v4h4" />
        </>
      )}
    </svg>
  )
}

/** One example (a file): its title and two lines of what it shows, and the directory it is in when it comes from a search. */
function Tile(props: { readonly entry: PocEntry; readonly where?: string | undefined }): ReactNode {
  const { entry } = props
  return (
    <Link to={pocHref(entry.slug)} className={TILE}>
      <h2 className="flex gap-2 font-semibold leading-snug">
        <Glyph kind="page" />
        <span>{entry.title}</span>
      </h2>
      <p className="line-clamp-2 text-sm text-muted">{entry.summary}</p>
      {props.where === undefined || props.where === '' ? null : (
        <p className="mt-1 text-xs text-muted">{props.where}</p>
      )}
    </Link>
  )
}

/** A directory: its name, three lines of what it holds, and how many examples (and folders) are in it. */
function DirTile(props: { readonly node: DirNode }): ReactNode {
  const { node } = props
  return (
    <Link to={dirHref(node.path)} className={`${TILE} min-h-32`}>
      <h2 className="flex gap-2 text-lg font-semibold leading-snug">
        <Glyph kind="folder" />
        <span>{node.meta.title}</span>
      </h2>
      <p className="line-clamp-3 text-sm text-muted">{node.meta.summary}</p>
      <p className="mt-auto pt-2 text-xs text-muted">
        {plural(node.count, 'example')}
        {node.dirs.length > 0 ? ` · ${plural(node.dirs.length, 'folder')}` : ''}
      </p>
    </Link>
  )
}

const GRID = 'grid gap-3 sm:grid-cols-2 lg:grid-cols-3'

/** The listing of a directory as a file browser shows it: the subdirectories first, then the examples in it. */
function Listing(props: { readonly node: DirNode }): ReactNode {
  const { node } = props
  if (node.dirs.length === 0 && node.pocs.length === 0)
    return (
      <p className="text-sm text-muted">
        Nothing here yet. Pages added to this directory (<code>src/pocs/{node.path.join('/')}/</code>) are listed here.
      </p>
    )
  const label = (text: string, n: number): ReactNode =>
    node.dirs.length > 0 && node.pocs.length > 0 ? (
      <h2 className="text-xs font-medium uppercase tracking-wide text-muted">
        {text} <span className="tabular-nums">{n}</span>
      </h2>
    ) : null
  return (
    <div className="flex flex-col gap-5">
      {node.dirs.length === 0 ? null : (
        <section className="flex flex-col gap-2">
          {label('Folders', node.dirs.length)}
          <div className={node.path.length === 0 ? 'grid gap-3 sm:grid-cols-2' : GRID}>
            {node.dirs.map((d) => (
              <DirTile key={d.name} node={d} />
            ))}
          </div>
        </section>
      )}
      {node.pocs.length === 0 ? null : (
        <section className="flex flex-col gap-2">
          {label('Examples', node.pocs.length)}
          <div className={GRID}>
            {node.pocs.map((e) => (
              <Tile key={e.slug} entry={e} />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

/**
 * A directory's page, like a file browser: the directory's title and summary, then its subdirectories and its examples
 * (the front page `/` is the root, `/dir/<a>/<b>` a subdirectory); a search lists the matching examples of the whole tree
 * flat, each with its directory path. The trail in the top bar moves up the tree.
 */
export function Index(props: { readonly path?: readonly string[] } = {}): ReactNode {
  const path = props.path ?? []
  const [query, setQuery] = useState('')
  const q = useDeferredValue(query)
  const node = findDir(TREE, path)
  const searching = q.trim() !== ''
  const shown = searching ? searchPocs(POCS, q) : []
  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h1 className="text-xl font-semibold tracking-tight">{node?.meta.title ?? 'Not found'}</h1>
        <input
          type="search"
          aria-label="Search the examples"
          placeholder="Search all examples"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="ml-auto h-9 w-56 rounded-md border border-line bg-bg px-2.5 text-sm text-fg outline-none focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50"
        />
      </div>
      {node === undefined || node.meta.summary === '' || searching ? null : (
        <p className="-mt-3 max-w-3xl text-muted">{node.meta.summary}</p>
      )}
      {searching ? (
        shown.length === 0 ? (
          <p className="text-sm text-muted">Nothing matches “{query}”.</p>
        ) : (
          <div className={GRID}>
            {shown.map((e) => (
              <Tile key={e.slug} entry={e} where={dirTitles(TREE, e).join(' / ')} />
            ))}
          </div>
        )
      ) : node === undefined ? (
        <p className="text-sm text-muted">
          There is no directory “{path.join('/')}”.{' '}
          <Link to="/" className="underline">
            All of them
          </Link>
          .
        </p>
      ) : (
        <Listing node={node} />
      )}
    </div>
  )
}
