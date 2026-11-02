/**
 * The reader's standing preferences for sessions, remembered in the browser from page to page: eager against live
 * (compute a run whole and show its end, or compute in the background while it plays) and whether a change to the
 * configuration waits for play. One store, so the player's eager button and the settings panel show the same state.
 */

import { useSyncExternalStore } from 'react'

/** The preferences. */
export interface SessionPreferences {
  /** Eager computes the whole run at once and shows its end; live computes in the background while it plays. */
  readonly mode: 'eager' | 'live'
  /** In live mode, hold a change to the configuration until play is pressed (eager applies it after a short pause). */
  readonly hold: boolean
}

/** The preferences before the reader has set any: live, holding changes. */
export const DEFAULT_PREFERENCES: SessionPreferences = { mode: 'live', hold: true }

const KEY = 'mdml-session-preferences'
const listeners = new Set<() => void>()
let current: SessionPreferences | undefined

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

/**
 * The preferences as they stand (the remembered ones on first use, else the defaults).
 *
 * @returns The preferences.
 */
export function sessionPreferences(): SessionPreferences {
  if (current !== undefined) return current
  current = DEFAULT_PREFERENCES
  try {
    const raw = typeof localStorage === 'undefined' ? null : localStorage.getItem(KEY)
    if (raw !== null) {
      const p = JSON.parse(raw) as Partial<SessionPreferences>
      current = { mode: p.mode === 'eager' ? 'eager' : 'live', hold: typeof p.hold === 'boolean' ? p.hold : true }
    }
  } catch {
    // Storage may be unavailable (private windows) or hold something else; the defaults stand for this page.
  }
  return current
}

/**
 * Change some preferences and remember them.
 *
 * @param changes - The preferences to change.
 */
export function setSessionPreferences(changes: Partial<SessionPreferences>): void {
  current = { ...sessionPreferences(), ...changes }
  try {
    localStorage.setItem(KEY, JSON.stringify(current))
  } catch {
    // Storage may be unavailable; the change still applies for this page.
  }
  for (const l of listeners) l()
}

/**
 * The preferences, re-rendering on change.
 *
 * @returns The preferences.
 * @index function React hook: the reader's live or eager mode and hold setting: `useSessionPreferences().mode`
 */
export function useSessionPreferences(): SessionPreferences {
  return useSyncExternalStore(subscribe, sessionPreferences, () => DEFAULT_PREFERENCES)
}
