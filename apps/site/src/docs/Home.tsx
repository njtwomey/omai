import type { ReactNode } from 'react'
import { Link } from '../shell/Link.tsx'
import { type Docs, modulePath } from './data.ts'

/** The docs home: the core's domains by tier and the methods' areas, each with its summary. */
export function DocsHome(props: { readonly docs: Docs }): ReactNode {
  const { docs } = props
  const top = docs.modules.filter((m) => m.parent === undefined)
  const core = top.filter((m) => m.layer === 'core')
  const tiers = [...new Set(core.map((m) => m.tier ?? 99))].sort((a, b) => a - b)
  const methods = top.filter((m) => m.layer === 'methods')
  const pkg = docs.packages[0]
  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-semibold">{pkg?.name ?? 'whitebox-engine'}</h1>
        <p className="mt-2 text-neutral-600 dark:text-neutral-400">{pkg?.description}</p>
        <p className="mt-1 text-sm text-neutral-500">
          {docs.modules.length} modules · {docs.symbols.length} exports, generated from the source.
        </p>
      </header>
      <section>
        <h2 className="mb-3 text-lg font-semibold">Core, by tier</h2>
        <p className="mb-4 text-sm text-neutral-500">Each domain imports only domains of lower tiers.</p>
        <div className="space-y-5">
          {tiers.map((t) => (
            <div key={t}>
              <h3 className="mb-2 text-sm font-medium text-neutral-500">Tier {t}</h3>
              <ul className="grid gap-3 sm:grid-cols-2">
                {core
                  .filter((m) => (m.tier ?? 99) === t)
                  .map((m) => (
                    <li key={m.id} className="rounded border border-neutral-200 p-3 dark:border-neutral-800">
                      <Link to={modulePath(m.id)} className="font-mono font-medium text-sky-700 dark:text-sky-400">
                        {m.name}
                      </Link>
                      <p className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">{m.summary}</p>
                    </li>
                  ))}
              </ul>
            </div>
          ))}
        </div>
      </section>
      <section>
        <h2 className="mb-3 text-lg font-semibold">Methods</h2>
        <ul className="grid gap-3 sm:grid-cols-2">
          {methods.map((m) => (
            <li key={m.id} className="rounded border border-neutral-200 p-3 dark:border-neutral-800">
              <Link to={modulePath(m.id)} className="font-mono font-medium text-sky-700 dark:text-sky-400">
                {m.name}
              </Link>
              <p className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">{m.summary}</p>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
