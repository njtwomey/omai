import type { ReactNode } from 'react'

/** The docs section: renders the page for a path under /docs. */
export function DocsSection(props: { readonly path: string }): ReactNode {
  return (
    <section>
      <h1 className="text-2xl font-semibold">Docs</h1>
      <p className="mt-2 text-sm text-neutral-500">{props.path}</p>
    </section>
  )
}
