import type { ReactNode } from 'react'

/**
 * The one place every figure and block of this page passes through. For now it draws its children as they are; when the shared
 * collapsible panel of omai-render (title bar with a chevron, a panels strip, state remembered per page, no probes while
 * folded) lands, it is adopted here and nowhere else. `id` is the key the state will be remembered under and `heavy` marks a
 * panel whose probes the page should not ask for while it is folded.
 */
export function Panel(props: {
  readonly id: string
  readonly title: string
  readonly heavy?: boolean
  readonly children: ReactNode
}): ReactNode {
  return <section aria-label={props.title}>{props.children}</section>
}
