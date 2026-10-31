/**
 * The logic of a settings dialog, apart from the drawing: a draft copy of the settings that is edited freely and
 * either applied as a whole or thrown away, the structural equality that says whether the draft differs from what is
 * applied, and when a press on the backdrop may dismiss the dialog without losing edits.
 */

/** The state of a settings dialog: whether it is open, and the draft being edited while it is. */
export interface SettingsState<T> {
  readonly open: boolean
  readonly draft: T
}

/** What can happen to a settings dialog. */
export type SettingsAction<T> =
  | { readonly type: 'open'; readonly value: T }
  | { readonly type: 'edit'; readonly draft: T }
  | { readonly type: 'reset'; readonly defaults: T }
  | { readonly type: 'apply' }
  | { readonly type: 'cancel' }
  | { readonly type: 'backdrop'; readonly dirty: boolean }

/**
 * Structural equality of two settings values: numbers, strings, booleans, `null`, arrays and plain objects (key order
 * does not matter). NaN equals NaN, so a draft with a NaN is not forever "changed".
 *
 * @param a - One value.
 * @param b - The other.
 * @returns Whether they hold the same data.
 * @example
 * ```ts
 * import { sameSettings } from 'mdml-render/react'
 * sameSettings({ a: 1, b: [1, 2] }, { b: [1, 2], a: 1 }) // => true
 * ```
 */
export function sameSettings(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false
  if (Array.isArray(a) !== Array.isArray(b)) return false
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => sameSettings(x, b[i]))
  const ka = Object.keys(a)
  const kb = Object.keys(b)
  if (ka.length !== kb.length) return false
  return ka.every(
    (k) => Object.hasOwn(b, k) && sameSettings((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]),
  )
}

/**
 * The next state of a settings dialog. Opening takes a fresh draft from the applied value, so an earlier cancelled edit
 * never comes back. Applying and cancelling both close it; the caller hands the draft to its `onApply` before it
 * applies (the draft is in the state it was in), so cancelling is the only way an edit is dropped. A press on the
 * backdrop closes it only when nothing has been edited, so a stray click never costs the user their changes.
 *
 * @param state - The current state.
 * @param action - What happened.
 * @returns The next state.
 * @example
 * ```ts
 * import { settingsReducer } from 'mdml-render/react'
 * const open = settingsReducer({ open: false, draft: 1 }, { type: 'open', value: 5 })
 * const edited = settingsReducer(open, { type: 'edit', draft: 7 })
 * const stray = settingsReducer(edited, { type: 'backdrop', dirty: true })
 * [open.draft, edited.draft, stray.open] // => [5, 7, true]
 * ```
 */
export function settingsReducer<T>(state: SettingsState<T>, action: SettingsAction<T>): SettingsState<T> {
  switch (action.type) {
    case 'open':
      return { open: true, draft: action.value }
    case 'edit':
      return state.open ? { open: true, draft: action.draft } : state
    case 'reset':
      return state.open ? { open: true, draft: action.defaults } : state
    case 'apply':
    case 'cancel':
      return { ...state, open: false }
    case 'backdrop':
      return action.dirty ? state : { ...state, open: false }
  }
}
