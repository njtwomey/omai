import { useSyncExternalStore } from 'react'

/** The site's colour mode. */
export type ColourMode = 'dark' | 'light'

const KEY = 'mdml-colour-mode'
const listeners = new Set<() => void>()

function read(): ColourMode {
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

/** The current colour mode (dark unless the toggle chose light), re-rendering on change. */
export function useColourMode(): ColourMode {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l)
      return () => listeners.delete(l)
    },
    read,
    () => 'dark',
  )
}

/** Set the colour mode and remember it. */
export function setColourMode(mode: ColourMode): void {
  document.documentElement.classList.toggle('dark', mode === 'dark')
  try {
    localStorage.setItem(KEY, mode)
  } catch {
    // Storage may be unavailable (private windows); the mode still applies for this page.
  }
  for (const l of listeners) l()
}
