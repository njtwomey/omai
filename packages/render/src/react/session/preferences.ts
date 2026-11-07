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
let stopStorage: (() => void) | undefined

// What the browser has remembered, or the defaults when it holds nothing, is unavailable or holds something else.
function stored(): SessionPreferences {
  try {
    const raw = typeof localStorage === 'undefined' ? null : localStorage.getItem(KEY)
    if (raw !== null) {
      const p = JSON.parse(raw) as Partial<SessionPreferences>
      return { mode: p.mode === 'eager' ? 'eager' : 'live', hold: typeof p.hold === 'boolean' ? p.hold : true }
    }
  } catch {
    // Storage may be unavailable (private windows) or hold something else; the defaults stand for this page.
  }
  return DEFAULT_PREFERENCES
}

// While anything listens, another tab's change to the remembered preferences (a `storage` event) is read in, so the
// tabs agree.
function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  if (listeners.size === 1 && typeof window !== 'undefined') {
    const onStorage = (e: StorageEvent): void => {
      if (e.key !== null && e.key !== KEY) return
      current = stored()
      for (const l of listeners) l()
    }
    window.addEventListener('storage', onStorage)
    stopStorage = () => window.removeEventListener('storage', onStorage)
  }
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0) {
      stopStorage?.()
      stopStorage = undefined
    }
  }
}

/**
 * The preferences as they stand (the remembered ones on first use, else the defaults; while a component listens, a
 * change made in another tab is picked up).
 *
 * @returns The preferences.
 */
export function sessionPreferences(): SessionPreferences {
  current ??= stored()
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
