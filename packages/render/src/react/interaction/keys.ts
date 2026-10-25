/**
 * Keyboard capture for figures that play like games: `useHeldKeys` makes a figure focusable and tracks the bound keys
 * held while it has focus, per press and per frame, without scrolling the page or catching keys meant for inputs.
 */

import { type FocusEvent, type KeyboardEvent, useEffect, useRef, useState } from 'react'

/** Props to spread on the element that captures keys: focusable, with the key handlers. */
export interface KeyCaptureProps {
  readonly tabIndex: 0
  readonly onKeyDown: (e: KeyboardEvent<HTMLElement>) => void
  readonly onKeyUp: (e: KeyboardEvent<HTMLElement>) => void
  readonly onBlur: (e: FocusEvent<HTMLElement>) => void
  readonly 'aria-keyshortcuts'?: string
}

/** Whether a key event comes from somewhere the reader is typing (an input, a text area, a select, editable text). */
export function isTyping(target: EventTarget | null): boolean {
  // Duck-typed, so it also runs where there is no DOM (server rendering, tests).
  const el = target as { readonly isContentEditable?: boolean; readonly closest?: (s: string) => unknown } | null
  if (el === null || typeof el.closest !== 'function') return false
  return el.isContentEditable === true || el.closest('input, textarea, select') !== null
}

/** A key's name for bindings: the letter in lower case (`w`), or the key's name (`ArrowUp`, ` ` for space). */
export function keyName(e: { readonly key: string }): string {
  return e.key.length === 1 ? e.key.toLowerCase() : e.key
}

/**
 * Keys pressed while a figure has focus, as a game reads them: the element made focusable (click it, or Tab to it)
 * captures the bound keys so they neither scroll the page nor reach other handlers, ignores keys while the reader types
 * in an input, and while any bound key is held calls `onFrame` once per display frame with the held keys and the
 * seconds since the last frame (for smooth movement). Each press also calls `onPress` once (for turn-based worlds).
 *
 * @param keys - The bound keys, as {@link keyName} names them (`w`, `a`, `ArrowUp`, ...).
 * @param handlers - `onPress(key)` per press, `onFrame(held, dt)` per frame while keys are held.
 * @returns Props for the capturing element, and the keys held now.
 */
export function useHeldKeys(
  keys: readonly string[],
  handlers: {
    readonly onPress?: (key: string) => void
    readonly onFrame?: (held: ReadonlySet<string>, dt: number) => void
  },
): { readonly props: KeyCaptureProps; readonly held: ReadonlySet<string> } {
  const [held, setHeld] = useState<ReadonlySet<string>>(() => new Set())
  const latest = useRef(handlers)
  latest.current = handlers
  const bound = new Set(keys)
  useEffect(() => {
    if (held.size === 0 || latest.current.onFrame === undefined || typeof requestAnimationFrame !== 'function') return
    let last: number | undefined
    let handle = requestAnimationFrame(function tick(now: number) {
      const dt = last === undefined ? 0 : Math.min((now - last) / 1000, 0.1)
      last = now
      if (dt > 0) latest.current.onFrame?.(held, dt)
      handle = requestAnimationFrame(tick)
    })
    return () => cancelAnimationFrame(handle)
  }, [held])
  return {
    held,
    props: {
      tabIndex: 0,
      'aria-keyshortcuts': keys.join(' '),
      onKeyDown: (e) => {
        const k = keyName(e)
        if (!bound.has(k) || isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return
        e.preventDefault()
        e.stopPropagation()
        if (!e.repeat) latest.current.onPress?.(k)
        if (!held.has(k)) setHeld((h) => new Set([...h, k]))
      },
      onKeyUp: (e) => {
        const k = keyName(e)
        if (!held.has(k)) return
        setHeld((h) => {
          const next = new Set(h)
          next.delete(k)
          return next
        })
      },
      onBlur: () => setHeld(new Set()),
    },
  }
}
