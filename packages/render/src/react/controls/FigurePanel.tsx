/**
 * Panels a page can fold or hide: `usePanels` keeps which panels of a page are open, folded or hidden (remembered per
 * page in `localStorage`, silently without it), `FigurePanel` is one panel with a title bar (a chevron that folds the
 * body, a button that hides it) and `PanelsStrip` lists all the panels with a switch each, so a hidden one can be brought
 * back. A folded or hidden panel mounts none of its body, and `Panels.active(name)` says so, so a page asks the session
 * for a panel's probes only while the panel is on show and hiding a heavy figure makes the page faster.
 */

import { type ReactNode, useCallback, useId, useMemo, useState } from 'react'
import { cx } from './classes.ts'
import { ChevronDownIcon, CloseIcon } from './icons.tsx'

/** How a panel is shown: with its body, folded to its title bar, or not at all (only the strip lists it). */
export type PanelState = 'open' | 'folded' | 'hidden'

/** The panels of a page: what each is called and how each is shown. */
export interface Panels {
  /** The page's key for what is remembered. */
  readonly id: string
  /** The panels, by name, with the title each is listed under. */
  readonly labels: Readonly<Record<string, string>>
  /** How a panel is shown (an unknown name is open). */
  state(name: string): PanelState
  /**
   * Whether the panel's body is on show: true only when open. Ask the session for a panel's probes only when this is true,
   * so a folded or hidden panel costs nothing.
   */
  active(name: string): boolean
  set(name: string, state: PanelState): void
}

const STORAGE = 'omai-panels:'

function load(id: string, names: readonly string[]): Record<string, PanelState> {
  try {
    const raw = globalThis.localStorage?.getItem(STORAGE + id)
    const parsed: unknown = raw === null || raw === undefined ? {} : JSON.parse(raw)
    if (typeof parsed !== 'object' || parsed === null) return {}
    const out: Record<string, PanelState> = {}
    for (const name of names) {
      const v = (parsed as Record<string, unknown>)[name]
      if (v === 'open' || v === 'folded' || v === 'hidden') out[name] = v
    }
    return out
  } catch {
    return {}
  }
}

function save(id: string, states: Record<string, PanelState>): void {
  try {
    globalThis.localStorage?.setItem(STORAGE + id, JSON.stringify(states))
  } catch {
    // no storage (a private window, blocked data): the choice lasts for the visit
  }
}

/**
 * The state of a page's panels: all open at first, then as the viewer left them (remembered per `id` in `localStorage`
 * inside a try, so a page without storage works the same for the visit).
 *
 * @param id - The page's key (its slug).
 * @param labels - The panels by name, each with its title.
 * @returns The panels' state and the way to change it.
 * @index function React hook: which panels of a page are open, folded or hidden, remembered per page.
 */
export function usePanels(id: string, labels: Readonly<Record<string, string>>): Panels {
  const [states, setStates] = useState<Record<string, PanelState>>(() => load(id, Object.keys(labels)))
  const set = useCallback(
    (name: string, state: PanelState): void => {
      setStates((s) => {
        const next = { ...s, [name]: state }
        save(id, next)
        return next
      })
    },
    [id],
  )
  const key = JSON.stringify(labels)
  // biome-ignore lint/correctness/useExhaustiveDependencies: the labels are keyed by their content
  return useMemo(
    () => ({
      id,
      labels,
      state: (name: string): PanelState => states[name] ?? 'open',
      active: (name: string): boolean => (states[name] ?? 'open') === 'open',
      set,
    }),
    [id, key, states, set],
  )
}

/**
 * One panel of a page: a compact title bar with a chevron button that folds the body (`aria-expanded`) and a button that
 * hides the panel, then the body. A folded panel keeps only its bar and a hidden one is not drawn (the
 * {@link PanelsStrip} brings it back); the body is not mounted in either case.
 *
 * @param props - The page's panels, this panel's name, and its body.
 * @returns The panel, or nothing when hidden.
 * @index function React component: a figure with a title bar that folds or hides it: `<FigurePanel panels={p} name="surface">`
 */
export function FigurePanel(props: {
  readonly panels: Panels
  readonly name: string
  /** The title (default: the panel's label). */
  readonly title?: ReactNode
  /** Controls beside the title (shown while the panel is open or folded). */
  readonly actions?: ReactNode
  readonly children: ReactNode
  readonly className?: string
}): ReactNode {
  const { panels, name, children, actions, className } = props
  const bodyId = useId()
  const state = panels.state(name)
  if (state === 'hidden') return null
  const title = props.title ?? panels.labels[name] ?? name
  const open = state === 'open'
  return (
    <section
      aria-label={typeof title === 'string' ? title : (panels.labels[name] ?? name)}
      className={cx('min-w-0 rounded-lg border border-(--wb-grid) bg-(--wb-surface) text-(--wb-fg)', className)}
    >
      <div className="flex items-center gap-1 pr-1">
        <button
          type="button"
          aria-expanded={open}
          aria-controls={bodyId}
          onClick={() => panels.set(name, open ? 'folded' : 'open')}
          className="flex min-w-0 flex-1 items-center gap-1.5 rounded-lg px-3 py-1.5 text-left text-xs font-semibold tracking-wide text-(--wb-muted) uppercase outline-none hover:text-(--wb-fg) focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50"
        >
          <ChevronDownIcon
            size={14}
            className={cx('shrink-0 transition-transform motion-reduce:transition-none', !open && '-rotate-90')}
          />
          <span className="truncate">{title}</span>
        </button>
        {actions}
        <button
          type="button"
          aria-label={`Hide ${typeof title === 'string' ? title : (panels.labels[name] ?? name)}`}
          onClick={() => panels.set(name, 'hidden')}
          className="rounded p-1 text-(--wb-muted) outline-none hover:text-(--wb-fg) focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50"
        >
          <CloseIcon size={12} />
        </button>
      </div>
      {open ? (
        <div id={bodyId} className="px-3 pb-3">
          {children}
        </div>
      ) : null}
    </section>
  )
}

/**
 * The strip that lists a page's panels with a switch each (`aria-pressed`: on show or not): turning one off hides it, on
 * brings it back (open), so a viewer sees at a glance what the page has.
 *
 * @param props - The page's panels.
 * @returns The strip.
 * @index function React component: a strip of switches listing the page's panels, to hide or bring them back.
 */
export function PanelsStrip(props: { readonly panels: Panels; readonly className?: string }): ReactNode {
  const { panels, className } = props
  return (
    <fieldset aria-label="Panels" className={cx('flex min-w-0 flex-wrap items-center gap-1.5 text-xs', className)}>
      <span className="text-(--wb-muted)">panels</span>
      {Object.entries(panels.labels).map(([name, label]) => {
        const shown = panels.state(name) !== 'hidden'
        return (
          <button
            key={name}
            type="button"
            aria-pressed={shown}
            onClick={() => panels.set(name, shown ? 'hidden' : 'open')}
            className={cx(
              'rounded-full border px-2 py-0.5 outline-none focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50',
              shown
                ? 'border-(--wb-accent) text-(--wb-fg)'
                : 'border-(--wb-grid) text-(--wb-muted) line-through hover:text-(--wb-fg)',
            )}
          >
            {label}
          </button>
        )
      })}
    </fieldset>
  )
}
