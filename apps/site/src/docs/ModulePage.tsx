import type { ReactNode } from 'react'
import type { ModuleDoc, SymbolKind } from 'whitebox-meta/schema'
import { Link } from '../shell/Link.tsx'
import { DocInline } from './DocText.tsx'
import { type Docs, modulePath, symbolPath } from './data.ts'

const GROUPS: readonly { readonly title: string; readonly kinds: readonly SymbolKind[] }[] = [
  { title: 'Functions', kinds: ['function'] },
  { title: 'Classes', kinds: ['class'] },
  { title: 'Constants', kinds: ['const', 'enum'] },
  { title: 'Types', kinds: ['interface', 'type'] },
]

/** The ancestors of a module, outermost first, as links. */
export function Breadcrumbs(props: {
  readonly docs: Docs
  readonly module: ModuleDoc
  readonly last?: string
}): ReactNode {
  const chain: ModuleDoc[] = []
  for (
    let m: ModuleDoc | undefined = props.module;
    m !== undefined;
    m = m.parent === undefined ? undefined : props.docs.module(m.parent)
  )
    chain.unshift(m)
  return (
    <nav className="mb-2 font-mono text-sm text-neutral-500">
      <Link to="/docs" className="hover:underline">
        whitebox-engine
      </Link>
      {chain.map((m) => (
        <span key={m.id}>
          {' / '}
          <Link to={modulePath(m.id)} className="hover:underline">
            {m.name}
          </Link>
        </span>
      ))}
      {props.last === undefined ? null : <span>{` / ${props.last}`}</span>}
    </nav>
  )
}

/** A module: its summary, submodules and exports grouped by kind, each with its one-line summary. */
export function ModulePage(props: { readonly docs: Docs; readonly module: ModuleDoc }): ReactNode {
  const { docs, module } = props
  const symbols = docs.symbolsOf(module.id)
  const reexports = Object.entries(module.reexports)
  return (
    <div className="space-y-6">
      <header>
        <Breadcrumbs docs={docs} module={module} />
        <h1 className="font-mono text-2xl font-semibold">{module.id}</h1>
        {module.summary === '' ? null : <p className="mt-2 text-neutral-600 dark:text-neutral-400">{module.summary}</p>}
        <p className="mt-1 text-sm text-neutral-500">
          {module.tier === undefined ? '' : `Tier ${module.tier} · `}
          {symbols.length} exports ·{' '}
          <code className="font-mono">
            import {'{ … }'} from '{module.id}'
          </code>
        </p>
      </header>
      {module.children.length === 0 ? null : (
        <section>
          <h2 className="mb-2 text-lg font-semibold">Modules</h2>
          <ul className="grid gap-2 sm:grid-cols-2">
            {module.children.map((c) => {
              const child = docs.module(c)
              return (
                <li key={c} className="rounded border border-neutral-200 p-2 dark:border-neutral-800">
                  <Link to={modulePath(c)} className="font-mono text-sky-700 dark:text-sky-400">
                    {child?.name ?? c}
                  </Link>
                  <span className="ml-2 text-sm text-neutral-500">
                    {child?.summary || `${docs.symbolsOf(c).length} exports`}
                  </span>
                </li>
              )
            })}
          </ul>
        </section>
      )}
      {GROUPS.map((g) => {
        const list = symbols.filter((s) => g.kinds.includes(s.kind))
        if (list.length === 0) return null
        return (
          <section key={g.title}>
            <h2 className="mb-2 text-lg font-semibold">
              {g.title} <span className="text-sm font-normal text-neutral-500">{list.length}</span>
            </h2>
            <ul className="divide-y divide-neutral-200 dark:divide-neutral-800">
              {list.map((s) => (
                <li key={s.id} className="flex gap-4 py-1.5 text-sm">
                  <Link to={symbolPath(s)} className="w-56 shrink-0 truncate font-mono text-sky-700 dark:text-sky-400">
                    {s.name}
                  </Link>
                  <span className="text-neutral-600 dark:text-neutral-400">
                    <DocInline text={s.summary} docs={docs} near={module.id} />
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )
      })}
      {reexports.length === 0 ? null : (
        <section>
          <h2 className="mb-2 text-lg font-semibold">Re-exports</h2>
          <ul className="text-sm">
            {reexports.map(([name, id]) => {
              const s = docs.symbols.find((x) => x.id === id)
              return (
                <li key={name} className="font-mono">
                  {name} ={' '}
                  {s === undefined ? (
                    id
                  ) : (
                    <Link to={symbolPath(s)} className="text-sky-700 dark:text-sky-400">
                      {id}
                    </Link>
                  )}
                </li>
              )
            })}
          </ul>
        </section>
      )}
    </div>
  )
}
