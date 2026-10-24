import { type ReactNode, type RefObject, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { cx, menuClass } from './classes.ts'

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
}

/**
 * A popup menu drawn above the page (in a portal, with fixed coordinates), so no scrolling or clipping container hides
 * it: below its anchor, or above when there is no room below. It closes on a press outside it and its anchor, on
 * Escape; it follows its anchor as the page scrolls or resizes.
 *
 * @param props - The anchor, whether it is open, what to call to close it, alignment and contents.
 * @returns The popup (nothing while closed or during server rendering).
 */
export function Popover(props: PopoverProps): ReactNode {
  const { anchor, open, onClose, align = 'start', children, role, label, className } = props
  const menu = useRef<HTMLDivElement>(null)
  const [at, setAt] = useState<{ readonly left: number; readonly top: number } | null>(null)
  const place = useCallback((): void => {
    const a = anchor.current?.getBoundingClientRect()
    const m = menu.current?.getBoundingClientRect()
    if (a === undefined) return
    const height = m?.height ?? 0
    const width = m?.width ?? 0
    const below = a.bottom + 4
    const top = below + height > innerHeight - 8 && a.top - height - 4 > 8 ? a.top - height - 4 : below
    const left = align === 'end' ? a.right - width : a.left
    setAt({ left: Math.max(8, Math.min(left, innerWidth - width - 8)), top })
  }, [anchor, align])
  useLayoutEffect(() => {
    if (open) place()
    else setAt(null)
  }, [open, place])
  useEffect(() => {
    if (!open) return
    const onPointer = (e: PointerEvent): void => {
      const t = e.target as Node
      if (menu.current?.contains(t) || anchor.current?.contains(t)) return
      onClose()
    }
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') onClose()
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
    <div
      ref={menu}
      role={role}
      aria-label={label}
      className={cx(menuClass, 'fixed', className)}
      // Measured hidden on the first frame, then placed.
      style={at === null ? { left: 0, top: 0, visibility: 'hidden' } : { left: at.left, top: at.top }}
    >
      {children}
    </div>,
    document.body,
  )
}
