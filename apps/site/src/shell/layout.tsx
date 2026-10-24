import type { ReactNode } from 'react'
import { Link } from './Link.tsx'

/** Text with `code` in backticks set as code (for one-line explanations held as data). */
export function Prose(props: { readonly text: string; readonly className?: string }): ReactNode {
  const parts = props.text.split(/(`[^`]+`)/g)
  return (
    <p className={props.className ?? 'text-sm leading-relaxed text-muted'}>
      {parts.map((p, i) =>
        p.startsWith('`') && p.endsWith('`') && p.length > 1 ? (
          // biome-ignore lint/suspicious/noArrayIndexKey: a fixed split of the text
          <code key={i} className="rounded bg-surface px-1 py-0.5 font-mono text-[0.85em] text-fg">
            {p.slice(1, -1)}
          </code>
        ) : (
          p
        ),
      )}
    </p>
  )
}

/** A page's heading: an optional eyebrow (a link up), the title and a lede. */
export function PageHeader(props: {
  readonly title: ReactNode
  readonly lede?: string
  readonly eyebrow?: { readonly label: string; readonly path: string }
  readonly actions?: ReactNode
}): ReactNode {
  const { title, lede, eyebrow, actions } = props
  return (
    <header className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-4">
        {eyebrow === undefined ? (
          <span />
        ) : (
          <Link to={eyebrow.path} className="text-xs font-medium tracking-wide text-muted uppercase hover:text-fg">
            {eyebrow.label}
          </Link>
        )}
        {actions}
      </div>
      <h1 className="m-0 text-2xl font-semibold tracking-tight">{title}</h1>
      {lede === undefined ? null : <Prose text={lede} className="m-0 max-w-3xl text-base leading-relaxed text-muted" />}
    </header>
  )
}

/** A titled block of a page, with an anchor id. */
export function PageSection(props: {
  readonly id?: string
  readonly title: ReactNode
  readonly blurb?: string
  readonly action?: ReactNode
  readonly children: ReactNode
}): ReactNode {
  const { id, title, blurb, action, children } = props
  return (
    <section id={id} className="flex scroll-mt-16 flex-col gap-3">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h2 className="m-0 text-lg font-semibold tracking-tight">{title}</h2>
          {blurb === undefined ? null : <Prose text={blurb} className="m-0 text-sm text-muted" />}
        </div>
        {action}
      </div>
      {children}
    </section>
  )
}

/** Cards in a responsive grid. */
export function CardGrid(props: { readonly children: ReactNode; readonly min?: number }): ReactNode {
  return (
    <div
      className="grid gap-3"
      style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${props.min ?? 220}px, 1fr))` }}
    >
      {props.children}
    </div>
  )
}

/** A card linking an in-app path: a title, a line or two, and an optional preview or badge above. */
export function Card(props: {
  readonly path: string
  readonly title: string
  readonly text?: string
  readonly preview?: ReactNode
  readonly badge?: string
}): ReactNode {
  const { path, title, text, preview, badge } = props
  return (
    <Link
      to={path}
      className="group flex flex-col overflow-hidden rounded-lg border border-line bg-bg transition-colors hover:border-muted"
    >
      {preview === undefined ? null : (
        <div className="flex aspect-[16/10] items-center justify-center overflow-hidden border-b border-line bg-surface">
          {preview}
        </div>
      )}
      <div className="flex flex-col gap-0.5 px-3 py-2.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-medium group-hover:text-accent">{title}</span>
          {badge === undefined ? null : <span className="text-xs text-muted tabular-nums">{badge}</span>}
        </div>
        {text === undefined ? null : <span className="line-clamp-2 text-xs leading-snug text-muted">{text}</span>}
      </div>
    </Link>
  )
}

/** Links to the previous and next pages of a sequence. */
export function PrevNext(props: {
  readonly prev?: { readonly path: string; readonly title: string }
  readonly next?: { readonly path: string; readonly title: string }
}): ReactNode {
  const { prev, next } = props
  const link = (to: { readonly path: string; readonly title: string }, dir: 'prev' | 'next'): ReactNode => (
    <Link
      to={to.path}
      className={`flex min-w-0 flex-col text-sm text-muted hover:text-fg ${dir === 'prev' ? 'text-left' : 'items-end text-right'}`}
    >
      <span className="text-xs">{dir === 'prev' ? 'Previous' : 'Next'}</span>
      <span className="truncate font-medium text-fg">{to.title}</span>
    </Link>
  )
  return (
    <nav aria-label="Neighbouring pages" className="flex justify-between gap-4 border-t border-line pt-4">
      {prev === undefined ? <span /> : link(prev, 'prev')}
      {next === undefined ? <span /> : link(next, 'next')}
    </nav>
  )
}
