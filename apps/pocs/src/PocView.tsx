import { type ReactNode, Suspense } from 'react'
import { ErrorBoundary } from '../../site/src/shell/ErrorBoundary.tsx'
import { Link } from '../../site/src/shell/Link.tsx'
import { POCS, pageOf } from './registry.ts'

/** One proof of concept: its title and summary, then its page (loaded when first opened). */
export function PocView(props: { readonly slug: string }): ReactNode {
  const entry = POCS.find((p) => p.slug === props.slug)
  const Page = pageOf(props.slug)
  if (entry === undefined || Page === undefined) {
    return (
      <p className="text-sm text-muted">
        No proof of concept called “{props.slug}”.{' '}
        <Link to="/" className="underline">
          All of them
        </Link>
        .
      </p>
    )
  }
  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <Link to="/" className="text-xs text-muted hover:text-fg">
          ← all proofs of concept
        </Link>
        <h1 className="text-xl font-semibold tracking-tight">{entry.title}</h1>
        <p className="max-w-3xl text-sm text-muted">{entry.summary}</p>
      </div>
      <ErrorBoundary>
        <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
          <Page />
        </Suspense>
      </ErrorBoundary>
    </div>
  )
}
