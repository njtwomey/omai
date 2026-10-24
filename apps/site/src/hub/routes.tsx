import type { ReactNode } from 'react'

/** The hub section: renders the page for a path under /hub. */
export function HubSection(props: { readonly path: string }): ReactNode {
  return (
    <section>
      <h1 className="text-2xl font-semibold">Hub</h1>
      <p className="mt-2 text-sm text-neutral-500">{props.path}</p>
    </section>
  )
}
