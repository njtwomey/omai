import type { ReactNode } from 'react'

/** The gallery section: renders the page for a path under /gallery. */
export function GallerySection(props: { readonly path: string }): ReactNode {
  return (
    <section>
      <h1 className="text-2xl font-semibold">Gallery</h1>
      <p className="mt-2 text-sm text-neutral-500">{props.path}</p>
    </section>
  )
}
