/**
 * A popup menu drawn in a portal with fixed coordinates, below its anchor or above when there is no room, so no
 * scrolling or clipping container hides it.
 */

import {
  createContext,
  type ReactNode,
  type RefObject,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import { createPortal } from 'react-dom'
import { placeInContainer } from './_placement.ts'
import { cx, menuClass } from './classes.ts'

const PopoverContainerContext = createContext<HTMLElement | null>(null)

/**
 * Where the popups below it are drawn: into the given element instead of `document.body`. Use it when the controls sit in a
 * scaled or transformed container (a reveal.js slide, a zoomed island), where a popup on `document.body` would lose the
 * container's theme variables and its scale. The element must be positioned (`position: relative` or the like); the popup
 * is placed in its coordinates, corrected for the scale it is drawn at.
 *
 * @param props - The element (or `null` for the page) and the children.
 * @returns The provider.
 * @index function React component: draw popups of the controls inside a given element: `<PopoverContainer container={el}>`
 */
export function PopoverContainer(props: {
  readonly container: HTMLElement | null
  readonly children?: ReactNode
}): ReactNode {
  return <PopoverContainerContext.Provider value={props.container}>{props.children}</PopoverContainerContext.Provider>
}

/** Props of {@link Popover}. */
export interface PopoverProps {
  /** The element the popup hangs from (its trigger). */
  readonly anchor: RefObject<HTMLElement | null>
  readonly open: boolean
  readonly onClose: () => void
  /** Line the popup up with the anchor's left (`start`) or right (`end`) edge (default `start`). */
  readonly align?: 'start' | 'end'
  readonly children: ReactNode
  readonly role?: 'listbox' | 'menu' | 'dialog'
  readonly label?: string
  readonly className?: string
  /**
   * Draw the popup into this element rather than `document.body` (default the one given by an enclosing
   * {@link PopoverContainer}, else the page). It must be positioned. See {@link PopoverContainer}.
   */
  readonly container?: HTMLElement | null
}

// The items of a menu or listbox, which the arrow keys move between.
const ITEMS = '[role="menuitem"],[role="option"]'

/**
 * A popup menu drawn above the page (in a portal, with fixed coordinates), so no scrolling or clipping container hides
 * it: below its anchor, or above when there is no room below. It closes on a press outside it and its anchor, on
 * Escape; it follows its anchor as the page scrolls or resizes. Focus moves into it when it opens (to the selected
 * option or the first item of a menu or listbox, to the popup itself for a dialog) and returns to the anchor when it
 * closes, unless the reader has moved it elsewhere; the arrow keys, Home and End move among a menu's items.
 *
 * @param props - The anchor, whether it is open, what to call to close it, alignment and contents.
 * @returns The popup (nothing while closed or during server rendering).
 */
export function Popover(props: PopoverProps): ReactNode {
  const { anchor, open, onClose, align = 'start', children, role, label, className } = props
  const inherited = useContext(PopoverContainerContext)
  const container = props.container === undefined ? inherited : props.container
  const menu = useRef<HTMLDivElement>(null)
  const [at, setAt] = useState<{ readonly left: number; readonly top: number } | null>(null)
  const place = useCallback((): void => {
    const a = anchor.current?.getBoundingClientRect()
    const m = menu.current?.getBoundingClientRect()
    if (a === undefined) return
    if (container !== null) {
      // In a container: its own coordinates, corrected for the scale it is drawn at (a transform on it or above it).
      setAt(
        placeInContainer({
          anchor: a,
          container: container.getBoundingClientRect(),
          size: { width: container.offsetWidth, height: container.offsetHeight },
          menu: { width: m?.width ?? 0, height: m?.height ?? 0 },
          align,
        }),
      )
      return
    }
    const height = m?.height ?? 0
    const width = m?.width ?? 0
    const below = a.bottom + 4
    const top = below + height > innerHeight - 8 && a.top - height - 4 > 8 ? a.top - height - 4 : below
    const left = align === 'end' ? a.right - width : a.left
    setAt({ left: Math.max(8, Math.min(left, innerWidth - width - 8)), top })
  }, [anchor, align, container])
  useLayoutEffect(() => {
    if (open) place()
    else setAt(null)
  }, [open, place])
  // Focus in once the popup is placed (it is hidden until then, and a hidden element cannot take focus), and back to the
  // anchor on close, when nothing else has taken it (a press elsewhere moves it to what was pressed).
  const placed = at !== null
  useEffect(() => {
    const el = menu.current
    if (!open || !placed || el === null || role === undefined) return
    const items = [...el.querySelectorAll<HTMLElement>(ITEMS)]
    const first =
      role === 'dialog' ? undefined : (items.find((i) => i.getAttribute('aria-selected') === 'true') ?? items[0])
    ;(first ?? el).focus({ preventScroll: true })
    return () => {
      const held = document.activeElement
      if (held === null || held === document.body || el.contains(held)) anchor.current?.focus({ preventScroll: true })
    }
  }, [open, placed, role, anchor])
  useEffect(() => {
    if (!open) return
    const onPointer = (e: PointerEvent): void => {
      const t = e.target as Node
      if (menu.current?.contains(t) || anchor.current?.contains(t)) return
      onClose()
    }
    const onKey = (e: KeyboardEvent): void => {
      if (e.key !== 'Escape') return
      // Only the popup closes: a dialog or an expanded figure it was raised from stays.
      e.stopPropagation()
      onClose()
    }
    // The popup follows its anchor as the page scrolls or resizes.
    const onMove = (e: Event): void => {
      if (menu.current !== null && e.target instanceof Node && menu.current.contains(e.target)) return
      place()
    }
    document.addEventListener('pointerdown', onPointer)
    document.addEventListener('keydown', onKey)
    addEventListener('scroll', onMove, true)
    addEventListener('resize', onMove)
    return () => {
      document.removeEventListener('pointerdown', onPointer)
      document.removeEventListener('keydown', onKey)
      removeEventListener('scroll', onMove, true)
      removeEventListener('resize', onMove)
    }
  }, [open, onClose, anchor, place])
  if (!open || typeof document === 'undefined') return null
  return createPortal(
    // biome-ignore lint/a11y/useAriaPropsSupportedByRole: the role is the caller's (a listbox or a menu), both labelled
    // biome-ignore lint/a11y/noStaticElementInteractions: the role is the caller's; the keys move between its items
    <div
      ref={menu}
      role={role}
      aria-label={label}
      tabIndex={-1}
      onKeyDown={(e) => {
        if (role === 'dialog' || !['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) return
        const items = [...(menu.current?.querySelectorAll<HTMLElement>(ITEMS) ?? [])]
        const current = items.indexOf(document.activeElement as HTMLElement)
        const step = e.key === 'ArrowDown' ? 1 : -1
        const next =
          e.key === 'Home' ? 0 : e.key === 'End' ? items.length - 1 : (current + step + items.length) % items.length
        e.preventDefault()
        items[next]?.focus()
      }}
      className={cx(menuClass, container === null ? 'fixed' : 'absolute', className)}
      // Measured hidden on the first frame, then placed.
      style={at === null ? { left: 0, top: 0, visibility: 'hidden' } : { left: at.left, top: at.top }}
    >
      {children}
    </div>,
    container ?? document.body,
  )
}
