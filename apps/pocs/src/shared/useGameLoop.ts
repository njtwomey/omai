import { type RefObject, useEffect, useRef } from 'react'

// Keys the games listen to (by `KeyboardEvent.code`) and the ones whose default action (scrolling) is suppressed.
const SCROLLING = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'])

const NOT_TEXT = new Set(['range', 'checkbox', 'radio', 'button', 'submit'])

/** Whether focus is in something the user types into (sliders, checkboxes and buttons do not count). */
function typing(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  if (el === null) return false
  if (el.tagName === 'INPUT') return !NOT_TEXT.has((el as HTMLInputElement).type)
  return el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable
}

/**
 * A frame loop for keyboard-driven pages: window key listeners keep the set of held keys (by `KeyboardEvent.code`,
 * ignored while focus is in a text field, with the arrow keys and space kept from scrolling the page), and
 * `requestAnimationFrame` calls `frame` with the seconds since the last frame (capped at 0.1), the held keys and the
 * frame's timestamp in milliseconds. Everything is removed on unmount.
 *
 * @param frame - Called once per animation frame; the latest function passed is the one called.
 * @returns The ref to the set of held keys.
 */
export function useGameLoop(
  frame: (dt: number, held: ReadonlySet<string>, now: number) => void,
): RefObject<ReadonlySet<string>> {
  const held = useRef<Set<string>>(new Set())
  const latest = useRef(frame)
  latest.current = frame
  useEffect(() => {
    const keys = held.current
    const down = (e: KeyboardEvent): void => {
      if (typing(e.target) || e.ctrlKey || e.metaKey || e.altKey) return
      if (SCROLLING.has(e.code)) e.preventDefault()
      keys.add(e.code)
    }
    const up = (e: KeyboardEvent): void => {
      keys.delete(e.code)
    }
    const blur = (): void => keys.clear()
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', blur)
    let last: number | null = null
    let id = 0
    const tick = (now: number): void => {
      const dt = last === null ? 0 : Math.min(0.1, (now - last) / 1000)
      last = now
      latest.current(dt, keys, now)
      id = requestAnimationFrame(tick)
    }
    id = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(id)
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blur)
      keys.clear()
    }
  }, [])
  return held
}
