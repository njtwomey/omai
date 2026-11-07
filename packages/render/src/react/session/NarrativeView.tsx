/**
 * A narrative in a page: one session told in several pages that share its tape and its step cursor. A switcher moves
 * between the pages without moving the cursor; each page shows its own text and draws its own layout of panels over the
 * same tape (the run is recorded fully, each page subscribes to the paths it is about); and the step-through, with
 * buttons and the arrow keys, moves the cursor to the page's marked steps and on to the next page's when a page has
 * none left.
 */

import { type KeyboardEvent, type ReactElement, type ReactNode, useId, useMemo, useState } from 'react'
import { cx, iconButtonClass } from '../controls/classes.ts'
import type { Narrative } from './layout.ts'
import { inlinePieces, momentIndex, paragraphs, resolveMoments, stepThrough, switchPage } from './narrative.ts'
import { SessionPanels } from './SessionPanels.tsx'
import { type LiveSession, type LiveSessions, variantsOf } from './useSession.ts'

/** Props of {@link NarrativeView}. */
export interface NarrativeViewProps {
  /** The name exported files start with. */
  readonly id: string
  readonly narrative: Narrative
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any> | LiveSessions<any, any>
  readonly at: number | null
  readonly onScrub: (t: number | null) => void
  /** The page shown, when the host holds it (e.g. in the address); else the view keeps its own. */
  readonly page?: number
  readonly onPage?: (page: number) => void
}

/** The text of a page: paragraphs with `code` and **bold**. */
function Prose(props: { readonly text: string }): ReactElement {
  return (
    <div className="flex max-w-3xl flex-col gap-2 text-sm leading-relaxed text-(--wb-fg)">
      {paragraphs({ text: props.text }).map((p) => (
        <p key={p}>
          {inlinePieces(p).map((piece, i): ReactNode => {
            const key = `${i}:${piece.text}`
            if (piece.kind === 'code')
              return (
                <code key={key} className="rounded bg-(--wb-surface) px-1 font-mono text-[0.85em]">
                  {piece.text}
                </code>
              )
            if (piece.kind === 'bold') return <strong key={key}>{piece.text}</strong>
            return <span key={key}>{piece.text}</span>
          })}
        </p>
      ))}
    </div>
  )
}

/**
 * The pages of a narrative over one session: the switcher, the page's text and panels, and the step-through.
 *
 * @param props - The narrative, the session and its cursor, and optionally the page and its setter.
 * @returns The narrative.
 * @index function React component: one session told in pages: `<NarrativeView id="n" narrative={n} live={live} ... />`
 */
export function NarrativeView(props: NarrativeViewProps): ReactElement {
  const { narrative, live, at, onScrub } = props
  const [own, setOwn] = useState(0)
  const page = Math.min(Math.max(0, props.page ?? own), Math.max(0, narrative.pages.length - 1))
  const head = live.head
  const viewed = at === null ? head : Math.min(at, head)
  const first = variantsOf(live)[0]?.live.session
  // The moments of every page against the tape so far (re-resolved as it grows).
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tape changes with the version
  const moments = useMemo(
    () => narrative.pages.map((p) => resolveMoments(p.moments, (e) => first?.stepsOf(e) ?? [], head, live.done)),
    [narrative, first, head, live.done, live.version],
  )
  const current = narrative.pages[page]
  const here = moments[page] ?? []
  const index = momentIndex(here, viewed)
  const go = (to: { readonly page: number; readonly step: number } | undefined): void => {
    if (to === undefined) return
    if (to.page !== page) (props.onPage ?? setOwn)(to.page)
    onScrub(to.step >= head && live.done ? null : to.step)
  }
  const through = (direction: 1 | -1): void => go(stepThrough(moments, { page, step: viewed }, direction))
  const select = (p: number): void => {
    const to = switchPage(narrative, { page, step: viewed }, p)
    ;(props.onPage ?? setOwn)(to.page)
  }
  const uid = useId()
  const tabId = (i: number): string => `${uid}-tab-${i}`
  // Arrow keys, Home and End move focus between the tabs; Enter or Space on a focused tab shows its page.
  const moveAmongTabs = (e: KeyboardEvent<HTMLElement>): void => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return
    const tabs = [...e.currentTarget.querySelectorAll<HTMLElement>('[role="tab"]')]
    const from = tabs.indexOf(e.target as HTMLElement)
    if (from < 0) return
    const step = e.key === 'ArrowRight' ? 1 : -1
    const to = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : (from + step + tabs.length) % tabs.length
    e.preventDefault()
    tabs[to]?.focus()
  }
  const hasPrev = stepThrough(moments, { page, step: viewed }, -1) !== undefined
  const hasNext = stepThrough(moments, { page, step: viewed }, 1) !== undefined
  const marked = here[index]
  if (current === undefined) return <div />
  return (
    <section
      className="flex flex-col gap-3 outline-none"
      aria-label={narrative.title ?? 'narrative'}
      data-testid="narrative"
      tabIndex={-1}
      onKeyDown={(e) => {
        if (e.target instanceof HTMLInputElement) return
        if (!e.altKey || (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft')) return
        // Alt+Arrow is also the browser's history; here it steps the moments, and the browser is told not to navigate.
        e.preventDefault()
        through(e.key === 'ArrowRight' ? 1 : -1)
      }}
    >
      <div className="flex flex-wrap items-center gap-2">
        {narrative.title === undefined ? null : <h3 className="mr-2 text-sm font-semibold">{narrative.title}</h3>}
        <div role="tablist" className="flex" onKeyDown={moveAmongTabs}>
          {narrative.pages.map((p, i) => (
            <button
              key={p.id}
              type="button"
              role="tab"
              id={tabId(i)}
              aria-selected={i === page}
              aria-controls={`${uid}-panel`}
              tabIndex={i === page ? 0 : -1}
              className={cx(
                'border border-(--wb-grid) px-3 py-1 text-xs',
                i === 0 ? 'rounded-l-md' : '-ml-px',
                i === narrative.pages.length - 1 ? 'rounded-r-md' : '',
                i === page ? 'bg-(--wb-accent) text-(--wb-bg)' : 'bg-(--wb-bg) text-(--wb-fg) hover:bg-(--wb-surface)',
              )}
              onClick={() => select(i)}
            >
              {i + 1}. {p.title}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-1 text-xs text-(--wb-muted)">
          <button
            type="button"
            aria-label="Previous moment"
            title="Previous moment (Alt+Left)"
            className={cx(iconButtonClass, 'rounded-l-md')}
            disabled={!hasPrev}
            onClick={() => through(-1)}
          >
            ‹
          </button>
          <button
            type="button"
            aria-label="Next moment"
            title="Next moment (Alt+Right)"
            className={cx(iconButtonClass, '-ml-px rounded-r-md')}
            disabled={!hasNext}
            onClick={() => through(1)}
          >
            ›
          </button>
          <span data-testid="moment" className="ml-1 tabular-nums">
            {here.length === 0
              ? 'no marked steps on this page'
              : `${index < 0 ? 'before the first' : `moment ${index + 1} of ${here.length}`}${marked?.label === undefined || index < 0 ? '' : `: ${marked.label}`}`}
          </span>
        </div>
      </div>
      <div role="tabpanel" id={`${uid}-panel`} aria-labelledby={tabId(page)} className="flex flex-col gap-3">
        <Prose text={current.text} />
        <SessionPanels
          key={current.id}
          id={`${props.id}-${current.id}`}
          live={live}
          at={at}
          onScrub={onScrub}
          layout={current.layout}
        />
      </div>
    </section>
  )
}
