// Internal: where keyboard focus goes for a modal (a dialog that holds it) and a popup (a menu that borrows it): in on
// open, kept inside by Tab for a modal, and back to where it was on close.

import { type RefObject, useEffect } from 'react'

/** The elements Tab can reach inside a container. */
export const FOCUSABLE =
  'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])'

/**
 * Where Tab (or Shift+Tab) must go to stay inside a container: the first item after the last, the last before the
 * first (and before the container itself, which holds focus when the dialog opens), or `undefined` when the browser's
 * own move stays inside.
 *
 * @param items - The focusable elements of the container, in order.
 * @param active - The element that has focus now.
 * @param shift - Whether it is Shift+Tab.
 * @param container - The container, which counts as before the first item.
 * @returns The element to focus instead of the browser's move, if any.
 */
export function wrapTarget<T>(items: readonly T[], active: unknown, shift: boolean, container: unknown): T | undefined {
  const first = items[0]
  const last = items[items.length - 1]
  if (first === undefined || last === undefined) return undefined
  if (shift) return active === first || active === container ? last : undefined
  return active === last ? first : undefined
}

// Whether something can take focus back: an element still in the page that is not the body.
function focusable(el: unknown): el is HTMLElement {
  const e = el as { focus?: unknown; isConnected?: boolean } | null
  return e !== null && typeof e.focus === 'function' && e.isConnected !== false && e !== document.body
}

/**
 * Hold keyboard focus in a modal while it is `active`: it moves into the container when it opens, Tab and Shift+Tab
 * wrap inside it, and when it closes focus returns to what had it before. The container needs `tabIndex={-1}` so it
 * can take focus itself.
 *
 * @param active - Whether the modal is open.
 * @param container - The modal's element.
 */
export function useFocusTrap(active: boolean, container: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    if (!active) return
    const previous = document.activeElement
    container.current?.focus()
    const onKey = (e: KeyboardEvent): void => {
      const root = container.current
      if (e.key !== 'Tab' || root === null) return
      const to = wrapTarget(
        [...root.querySelectorAll<HTMLElement>(FOCUSABLE)],
        document.activeElement,
        e.shiftKey,
        root,
      )
      if (to === undefined) return
      e.preventDefault()
      to.focus()
    }
    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('keydown', onKey, true)
      if (focusable(previous)) previous.focus()
    }
  }, [active, container])
}
