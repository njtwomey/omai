/**
 * The step cursor: the one viewed step every panel of a session shares. Panels of a step read it; panels of the tape
 * move it on click. It is a context, so a custom panel inside `SessionPanels` needs no wiring to take part.
 */

import { createContext, type ReactNode, useContext } from 'react'

/** The viewed step, the head of the run, and the setter (`null` follows the head). */
export interface StepCursor {
  readonly viewed: number
  readonly head: number
  readonly scrub: (t: number | null) => void
}

const CursorContext = createContext<StepCursor | undefined>(undefined)

/**
 * Provides the step cursor to the panels inside.
 *
 * @param props - The cursor and the panels.
 * @returns The provider.
 */
export function StepCursorProvider(props: { readonly value: StepCursor; readonly children: ReactNode }): ReactNode {
  return <CursorContext.Provider value={props.value}>{props.children}</CursorContext.Provider>
}

/**
 * The enclosing step cursor, or `undefined` outside `SessionPanels`.
 *
 * @returns The cursor.
 */
export function useStepCursor(): StepCursor | undefined {
  return useContext(CursorContext)
}
