import type { ReactNode } from 'react'
import type { ModuleDoc } from 'whitebox-meta/schema'
import { type Docs, useDocs, useModuleSymbols } from './data.ts'
import { DocsHome } from './Home.tsx'
import { ModulePage } from './ModulePage.tsx'
import { Sidebar } from './Sidebar.tsx'
import { SymbolPage } from './SymbolPage.tsx'
import { docsToc } from './toc.ts'

/** What a docs path names: the home, a module, or a symbol of a module. */
type Target =
  | { readonly kind: 'home' }
  | { readonly kind: 'module'; readonly module: ModuleDoc }
  | { readonly kind: 'symbol'; readonly module: ModuleDoc; readonly name: string }
  | { readonly kind: 'missing' }

function resolve(docs: Docs, path: string): Target {
  const rest = path.replace(/^\/docs\/?/, '').replace(/\/$/, '')
  if (rest === '') return { kind: 'home' }
  const whole = docs.module(`whitebox-engine/${rest}`)
  if (whole !== undefined) return { kind: 'module', module: whole }
  const cut = rest.lastIndexOf('/')
  const module = cut < 0 ? undefined : docs.module(`whitebox-engine/${rest.slice(0, cut)}`)
  if (module === undefined) return { kind: 'missing' }
  return { kind: 'symbol', module, name: decodeURIComponent(rest.slice(cut + 1)) }
}

function Loading(): ReactNode {
  return <p className="text-sm text-neutral-500">Loading…</p>
}

function SymbolTarget(props: { readonly docs: Docs; readonly module: ModuleDoc; readonly name: string }): ReactNode {
  const symbols = useModuleSymbols(props.module.id)
  if (symbols === null) return <Loading />
  const symbol = symbols.find((s) => s.name === props.name)
  if (symbol === undefined)
    return (
      <p>
        No export {props.name} in {props.module.id}.
      </p>
    )
  return <SymbolPage docs={props.docs} module={props.module} symbol={symbol} />
}

/** The docs section: the engine reference under /docs, with the table of contents on the left. */
export function DocsSection(props: { readonly path: string }): ReactNode {
  const docs = useDocs()
  if (docs === null) return <Loading />
  const target = resolve(docs, props.path)
  const page =
    target.kind === 'home' ? (
      <DocsHome docs={docs} />
    ) : target.kind === 'module' ? (
      <ModulePage docs={docs} module={target.module} />
    ) : target.kind === 'symbol' ? (
      <SymbolTarget docs={docs} module={target.module} name={target.name} />
    ) : (
      <p>Nothing documented at {props.path}.</p>
    )
  return (
    <div className="flex gap-8">
      <aside className="sticky top-4 hidden max-h-[calc(100vh-6rem)] w-60 shrink-0 overflow-y-auto md:block">
        <Sidebar tree={docsToc(docs)} path={props.path.replace(/\/$/, '')} />
      </aside>
      <div className="min-w-0 flex-1">{page}</div>
    </div>
  )
}
