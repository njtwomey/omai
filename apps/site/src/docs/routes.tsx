import type { ReactNode } from 'react'
import { type Docs, type Folder, type IndexFile, useDocs, useFileSymbols } from './data.ts'
import { FilePage } from './FilePage.tsx'
import { FolderPage } from './FolderPage.tsx'
import { Sidebar } from './Sidebar.tsx'
import { SymbolPage } from './SymbolPage.tsx'
import { docsToc } from './toc.ts'

/** What a docs path names, following the source tree: a folder, a file, or an export of a file. */
type Target =
  | { readonly kind: 'folder'; readonly folder: Folder }
  | { readonly kind: 'file'; readonly file: IndexFile }
  | { readonly kind: 'symbol'; readonly file: IndexFile; readonly name: string }
  | { readonly kind: 'missing' }

function resolve(docs: Docs, path: string): Target {
  const rest = decodeURIComponent(path.replace(/^\/docs\/?/, '').replace(/\/$/, ''))
  const folder = docs.folder(rest)
  if (folder !== undefined) return { kind: 'folder', folder }
  const file = docs.file(rest)
  if (file !== undefined) return { kind: 'file', file }
  const cut = rest.lastIndexOf('/')
  const owner = cut < 0 ? undefined : docs.file(rest.slice(0, cut))
  return owner === undefined ? { kind: 'missing' } : { kind: 'symbol', file: owner, name: rest.slice(cut + 1) }
}

function Loading(): ReactNode {
  return <p className="text-sm text-neutral-500">Loading…</p>
}

function SymbolTarget(props: { readonly docs: Docs; readonly file: IndexFile; readonly name: string }): ReactNode {
  const symbols = useFileSymbols(props.file)
  if (symbols === null) return <Loading />
  const symbol = symbols.find((s) => s.name === props.name)
  if (symbol === undefined)
    return (
      <p>
        No export {props.name} in {props.file.path}.
      </p>
    )
  return <SymbolPage docs={props.docs} file={props.file} symbol={symbol} />
}

/** The docs section: the engine's source tree under /docs, with the table of contents on the left. */
export function DocsSection(props: { readonly path: string }): ReactNode {
  const docs = useDocs()
  if (docs === null) return <Loading />
  const path = props.path.replace(/\/$/, '')
  const target = resolve(docs, path)
  const page =
    target.kind === 'folder' ? (
      <FolderPage docs={docs} folder={target.folder} />
    ) : target.kind === 'file' ? (
      <FilePage docs={docs} file={target.file} />
    ) : target.kind === 'symbol' ? (
      <SymbolTarget docs={docs} file={target.file} name={target.name} />
    ) : (
      <p>Nothing documented at {props.path}.</p>
    )
  return (
    <div className="flex gap-8">
      <aside className="sticky top-4 hidden max-h-[calc(100vh-6rem)] w-64 shrink-0 overflow-y-auto md:block">
        <Sidebar tree={docsToc(docs)} path={path} />
      </aside>
      <div className="min-w-0 flex-1">{page}</div>
    </div>
  )
}
