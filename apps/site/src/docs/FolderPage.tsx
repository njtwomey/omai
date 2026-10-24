import type { ReactNode } from 'react'
import { Link } from '../shell/Link.tsx'
import { Crumbs } from './Crumbs.tsx'
import { DocInline } from './DocText.tsx'
import { baseName, type Docs, docsPath, type Folder, symbolPath } from './data.ts'

function count(docs: Docs, path: string): number {
  const f = docs.folder(path)
  if (f === undefined) return 0
  return f.files.length + f.folders.reduce((n, p) => n + count(docs, p), 0)
}

/** A folder of the source tree: its summary, then its subfolders and its files, as on disk. */
export function FolderPage(props: { readonly docs: Docs; readonly folder: Folder }): ReactNode {
  const { docs, folder } = props
  const pkg = docs.packages[0]
  return (
    <div className="space-y-6">
      <header>
        <Crumbs path={folder.path} />
        <h1 className="font-mono text-2xl font-semibold">
          {folder.path === '' ? 'whitebox-engine' : `${folder.path}/`}
        </h1>
        {folder.path === '' && pkg !== undefined ? (
          <p className="mt-2 text-neutral-600 dark:text-neutral-400">{pkg.description}</p>
        ) : null}
        {folder.summary === '' ? null : <p className="mt-2 text-neutral-600 dark:text-neutral-400">{folder.summary}</p>}
        {folder.module === undefined ? null : (
          <p className="mt-1 font-mono text-xs text-neutral-500">
            import {'{ … }'} from '{folder.module}'
          </p>
        )}
      </header>
      {folder.folders.length === 0 ? null : (
        <section>
          <h2 className="mb-2 text-lg font-semibold">Folders</h2>
          <ul className="grid gap-2 sm:grid-cols-2">
            {folder.folders.map((p) => {
              const sub = docs.folder(p)
              return (
                <li key={p} className="rounded border border-neutral-200 p-3 dark:border-neutral-800">
                  <Link to={docsPath(p)} className="font-mono font-medium text-sky-700 dark:text-sky-400">
                    {baseName(p)}/
                  </Link>
                  <span className="ml-2 text-xs text-neutral-500">{count(docs, p)} files</span>
                  {sub?.summary ? (
                    <p className="mt-1 line-clamp-3 text-sm text-neutral-600 dark:text-neutral-400">{sub.summary}</p>
                  ) : null}
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
              <li key={f.path} className="rounded border border-neutral-200 p-3 dark:border-neutral-800">
                <Link to={docsPath(f.path)} className="font-mono font-medium text-sky-700 dark:text-sky-400">
                  {baseName(f.path)}
                </Link>
                <span className="ml-2 text-xs text-neutral-500">{f.symbols.length} exports</span>
                {f.summary === '' ? null : (
                  <p className="mt-1 line-clamp-2 text-sm text-neutral-600 dark:text-neutral-400">
                    <DocInline text={f.summary} docs={docs} near={f.module} />
                  </p>
                )}
                <p className="mt-1 line-clamp-3 font-mono text-xs">
                  {f.symbols.map((s, i) => (
                    <span key={s.name}>
                      {i > 0 ? ' · ' : ''}
                      <Link
                        to={symbolPath(f.path, s.name)}
                        className="text-neutral-600 hover:underline dark:text-neutral-400"
                      >
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
