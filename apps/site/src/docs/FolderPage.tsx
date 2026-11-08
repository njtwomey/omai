import type { ReactNode } from 'react'
import { Link } from '../shell/Link.tsx'
import { Crumbs } from './Crumbs.tsx'
import { DocInline } from './DocText.tsx'
import { baseName, type Docs, docsPath, type Folder, symbolPath } from './data.ts'
import { Search } from './Search.tsx'

function count(docs: Docs, path: string): number {
  const f = docs.folder(path)
  if (f === undefined) return 0
  return f.files.length + f.folders.reduce((n, p) => n + count(docs, p), 0)
}

/** A folder of the source tree: its summary, then its subfolders and its files, as on disk. */
export function FolderPage(props: {
  readonly docs: Docs
  readonly folder: Folder
  readonly nav?: ReactNode
}): ReactNode {
  const { docs, folder } = props
  const pkg = docs.packages[0]
  return (
    <div className="space-y-6">
      <header>
        <div className="flex items-start justify-between gap-4">
          <Crumbs path={folder.path} />
          {props.nav}
        </div>
        <h1 className="font-mono text-2xl font-semibold">{folder.path === '' ? 'omai-engine' : `${folder.path}/`}</h1>
        {folder.path === '' && pkg !== undefined ? <p className="mt-2 text-fg-secondary">{pkg.description}</p> : null}
        {folder.summary === '' ? null : <p className="mt-2 text-fg-secondary">{folder.summary}</p>}
        {folder.module === undefined ? null : (
          <p className="mt-1 font-mono text-xs text-muted">
            import {'{ … }'} from '{folder.module}'
          </p>
        )}
      </header>
      {folder.path === '' ? <Search docs={docs} /> : null}
      {folder.folders.length === 0 ? null : (
        <section>
          <h2 className="mb-2 text-lg font-semibold">Folders</h2>
          <ul className="grid gap-2 sm:grid-cols-2">
            {folder.folders.map((p) => {
              const sub = docs.folder(p)
              return (
                <li
                  key={p}
                  className="group relative rounded-lg border border-line bg-bg p-3 transition-colors hover:border-muted"
                >
                  <Link
                    to={docsPath(p)}
                    className="font-mono font-medium text-accent after:absolute after:inset-0 after:content-['']"
                  >
                    {baseName(p)}/
                  </Link>
                  <span className="ml-2 text-xs text-muted">{count(docs, p)} files</span>
                  {sub?.summary ? <p className="mt-1 line-clamp-3 text-sm text-fg-secondary">{sub.summary}</p> : null}
                </li>
              )
            })}
          </ul>
        </section>
      )}
      {folder.files.length === 0 ? null : (
        <section>
          <h2 className="mb-2 text-lg font-semibold">Files</h2>
          <ul className="grid gap-2 sm:grid-cols-2">
            {folder.files.map((f) => (
              <li
                key={f.path}
                className="group relative rounded-lg border border-line bg-bg p-3 transition-colors hover:border-muted"
              >
                <Link
                  to={docsPath(f.path)}
                  className="font-mono font-medium text-accent after:absolute after:inset-0 after:content-['']"
                >
                  {baseName(f.path)}
                </Link>
                <span className="ml-2 text-xs text-muted">{f.symbols.length} exports</span>
                {f.summary === '' ? null : (
                  <p className="mt-1 line-clamp-2 text-sm text-fg-secondary">
                    <DocInline text={f.summary} docs={docs} near={f.module} />
                  </p>
                )}
                <p className="relative z-10 mt-1 line-clamp-3 font-mono text-xs">
                  {f.symbols.map((s, i) => (
                    <span key={s.name}>
                      {i > 0 ? ' · ' : ''}
                      <Link to={symbolPath(f.path, s.name)} className="text-fg-secondary hover:underline">
                        {s.name}
                      </Link>
                    </span>
                  ))}
                </p>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}
