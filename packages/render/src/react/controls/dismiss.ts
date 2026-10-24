import { type RefObject, useEffect } from 'react'

/**
 * Close a popup on a press outside its container or on Escape (focus returns to the container's trigger when the
 * caller restores it).
 *
 * @param ref - The element holding the trigger and the popup.
 * @param open - Whether the popup is open.
 * @param close - Close it.
 */
export function useDismiss(ref: RefObject<HTMLElement | null>, open: boolean, close: () => void): void {
  useEffect(() => {
    if (!open) return
    const onPointer = (e: PointerEvent): void => {
      if (ref.current !== null && !ref.current.contains(e.target as Node)) close()
    }
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') close()
    }
    document.addEventListener('pointerdown', onPointer)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onPointer)
      document.removeEventListener('keydown', onKey)
    }
  }, [ref, open, close])
}
