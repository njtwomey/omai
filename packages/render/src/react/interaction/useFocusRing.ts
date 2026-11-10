/**
 * The keyboard focus mark of the overlay tools: `useFocusRing` says which of a tool's focusable parts shows a ring, set
 * by React state because `:focus-visible` cannot be styled on an inline SVG element. A part takes the ring when focus
 * reaches it from the keyboard (Tab) or a key press on it, never when it was focused by a pointer press.
 */

import { useRef, useState } from 'react'

/** The colour of the focus ring: the accent token, with the default accent as its fallback. */
export const FOCUS_COLOUR = 'var(--wb-accent, #1f77b4)'

/** What {@link useFocusRing} returns. */
export interface FocusRing<K extends string> {
  /** The part that shows the ring now, or null. */
  readonly focused: K | null
  /** The `onFocus` and `onBlur` handlers of a part: spread them on its focusable element. */
  readonly bind: (key: K) => { readonly onFocus: () => void; readonly onBlur: () => void }
  /** Note a pointer press: the focus that follows it is not the keyboard's, so it shows no ring. */
  readonly press: () => void
  /** Note that the pointer press is over (a press that took no focus must not hide a later Tab). */
  readonly release: () => void
  /** Show the ring on a part (a key was pressed on it, so the reader is using the keyboard). */
  readonly reveal: (key: K) => void
}

/**
 * The focus ring state of an overlay tool with one or more focusable parts: a part gets the ring on focus unless a
 * pointer press just before it caused that focus, and loses it on blur. Call `press` from the pointer-down handler,
 * `release` from pointer-up and `reveal` from a handled key, and spread `bind(key)` on the element.
 *
 * @returns The part with the ring and the handlers.
 * @index function React hook: which part of an overlay tool shows its keyboard focus ring: `useFocusRing<'body'>()`
 * @example
 * ```tsx
 * const ring = useFocusRing<'line'>()
 * <g tabIndex={0} onPointerDown={ring.press} onPointerUp={ring.release} {...ring.bind('line')}>
 *   {ring.focused === 'line' && <rect fill="none" stroke={FOCUS_COLOUR} />}
 * </g>
 * ```
 */
export function useFocusRing<K extends string = string>(): FocusRing<K> {
  const [focused, setFocused] = useState<K | null>(null)
  const pointer = useRef(false)
  return {
    focused,
    bind: (key) => ({
      onFocus: () => {
        setFocused(pointer.current ? null : key)
        pointer.current = false
      },
      onBlur: () => {
        pointer.current = false
        setFocused((now) => (now === key ? null : now))
      },
    }),
    press: () => {
      pointer.current = true
    },
    release: () => {
      pointer.current = false
    },
    reveal: (key) => setFocused(key),
  }
}
