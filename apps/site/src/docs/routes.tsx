import type { ReactNode } from 'react'
import { PrevNext } from '../shell/PrevNext.tsx'
import { TocSidebar } from '../shell/TocSidebar.tsx'
import { type Docs, type Folder, type IndexFile, useDocs, useFileData } from './data.ts'
import { FilePage } from './FilePage.tsx'
import { FolderPage } from './FolderPage.tsx'
import { SymbolPage } from './SymbolPage.tsx'
import { RefsContext } from './TypeHover.tsx'
import { DOCS_HOME, docsToc } from './toc.ts'

/** What a docs path names, following the source tree: a folder, a file, or an export of a file. */
type Target =
  | { readonly kind: 'folder'; readonly folder: Folder }
  | { readonly kind: 'file'; readonly file: IndexFile }
  | { readonly kind: 'symbol'; readonly file: IndexFile; readonly name: string }
  | { readonly kind: 'missing' }

function resolve(docs: Docs, path: string): Target {
  const rest = decodeURIComponent(path.replace(/^\/docs\/?/, ''))
  const folder = docs.folder(rest)
  if (folder !== undefined) return { kind: 'folder', folder }
  const file = docs.file(rest)
  if (file !== undefined) return { kind: 'file', file }
  const cut = rest.lastIndexOf('/')
  const owner = cut < 0 ? undefined : docs.file(rest.slice(0, cut))
  return owner === undefined ? { kind: 'missing' } : { kind: 'symbol', file: owner, name: rest.slice(cut + 1) }
}

function Loading(): ReactNode {
  return <p className="text-sm text-muted">Loading…</p>
}

function SymbolTarget(props: {
  readonly docs: Docs
  readonly file: IndexFile
  readonly name: string
  readonly nav: ReactNode
}): ReactNode {
  const data = useFileData(props.file)
  if (data === null) return <Loading />
  const symbol = data.symbols.find((s) => s.name === props.name)
  if (symbol === undefined)
    return (
      <p>
        No export {props.name} in {props.file.path}.
      </p>
    )
  return (
    <RefsContext.Provider value={data.refs}>
      <SymbolPage docs={props.docs} file={props.file} symbol={symbol} nav={props.nav} />
    </RefsContext.Provider>
  )
}

const tocs = new WeakMap<Docs, ReturnType<typeof docsToc>>()
function tocOf(docs: Docs): ReturnType<typeof docsToc> {
  let t = tocs.get(docs)
  if (t === undefined) {
    t = docsToc(docs)
    tocs.set(docs, t)
  }
  return t
}

/** The docs section: the engine's source tree under /docs, with previous and next links in tree order. */
export function DocsSection(props: { readonly path: string }): ReactNode {
  const docs = useDocs()
  if (docs === null) return <Loading />
  const path = props.path.replace(/\/$/, '')
  const toc = tocOf(docs)
  const nav = <PrevNext nodes={toc} home={DOCS_HOME} current={path} compact />
  const target = resolve(docs, path)
  const page =
    target.kind === 'folder' ? (
      <FolderPage docs={docs} folder={target.folder} nav={nav} />
    ) : target.kind === 'file' ? (
      <FilePage docs={docs} file={target.file} nav={nav} />
    ) : target.kind === 'symbol' ? (
      <SymbolTarget docs={docs} file={target.file} name={target.name} nav={nav} />
    ) : (
      <p>Nothing documented at {props.path}.</p>
    )
  return (
    <>
      {page}
      <PrevNext nodes={toc} home={DOCS_HOME} current={path} />
    </>
  )
}

/** The docs sidebar: the source tree, collapsed but for the way to the current page. */
export function DocsToc(props: { readonly path: string }): ReactNode {
  const docs = useDocs()
  if (docs === null) return <div className="p-3 text-sm text-muted">Loading…</div>
  return (
    <TocSidebar
      nodes={tocOf(docs)}
      current={props.path.replace(/\/$/, '')}
      home={DOCS_HOME}
      placeholder="Filter files and exports"
    />
  )
}
