/**
 * Small parts the specialised panels share: a text button in the style of the player's buttons, a value that settles
 * after a pause (so a drag re-probes once it stops), and a static live session wrapped around a finished session (the
 * inner session's panels and the tests draw layouts over it).
 */

import type { Session } from 'mdml-engine/trace'
import { type ReactNode, useEffect, useState } from 'react'
import { cx, iconButtonClass } from '../controls/classes.ts'
import { type LiveSession, STILL_HANDLE } from './useSession.ts'

/**
 * A labelled text button, outlined like the player's buttons.
 *
 * @param props - The label, the action and whether it is pressed.
 * @returns The button.
 */
export function ToolButton(props: {
  readonly label: ReactNode
  readonly title: string
  readonly onClick: () => void
  readonly pressed?: boolean
  readonly disabled?: boolean
}): ReactNode {
  return (
    <button
      type="button"
      title={props.title}
      aria-label={props.title}
      aria-pressed={props.pressed}
      disabled={props.disabled}
      onClick={props.onClick}
      className={cx(
        iconButtonClass,
        'w-auto min-w-8 rounded-md px-2 text-xs',
        props.pressed === true && 'bg-(--wb-surface) font-semibold text-(--wb-accent)',
      )}
    >
      {props.label}
    </button>
  )
}

/**
 * A value that follows `value` after it has stood still for `ms` milliseconds.
 *
 * @param value - The value.
 * @param ms - The pause.
 * @returns The settled value.
 */
export function useSettled<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value)
  useEffect(() => {
    const handle = setTimeout(() => setSettled(value), ms)
    return () => clearTimeout(handle)
  }, [value, ms])
  return settled
}

/**
 * A finished session as a live one that never changes: the handle the panels read, with nothing to compute.
 *
 * @param session - The session.
 * @returns The live session.
 */
// biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
export function staticLive(session: Session<any, any>): LiveSession<any, any> {
  return {
    session,
    version: 0,
    head: session.t,
    done: session.status !== undefined,
    computing: false,
    compute: () => undefined,
    halt: () => undefined,
    advance: () => undefined,
    lastMs: 0,
    stale: false,
    held: false,
    ...STILL_HANDLE,
    apply: () => session,
  }
}
