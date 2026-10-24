import { useCallback, useSyncExternalStore } from 'react'
import { COLOUR_STORAGE_KEY, DARK_QUERY } from './colour-script.ts'

/** The reader's choice: a fixed mode, or follow the system setting. */
export type ColourPreference = 'light' | 'dark' | 'system'

/** The mode pages are drawn in. */
export type ColourMode = 'light' | 'dark'

const EVENT = 'whitebox-colour-mode'

function storedPreference(): ColourPreference {
  try {
    const value = localStorage.getItem(COLOUR_STORAGE_KEY)
    return value === 'light' || value === 'dark' ? value : 'system'
  } catch {
    return 'system'
  }
}

function systemMode(): ColourMode {
  return typeof matchMedia === 'function' && matchMedia(DARK_QUERY).matches ? 'dark' : 'light'
}

/** The mode a preference resolves to now. */
export function resolveMode(preference: ColourPreference): ColourMode {
  return preference === 'system' ? systemMode() : preference
}

/** Mark the document with a mode: the `dark` class (Tailwind's dark variant), `data-theme` and `color-scheme`. */
export function applyMode(mode: ColourMode): void {
  const root = document.documentElement
  root.classList.toggle('dark', mode === 'dark')
  root.dataset['theme'] = mode
  root.style.colorScheme = mode
}

function subscribe(onChange: () => void): () => void {
  const media = typeof matchMedia === 'function' ? matchMedia(DARK_QUERY) : undefined
  const onStorage = (e: StorageEvent): void => {
    if (e.key === COLOUR_STORAGE_KEY) onChange()
  }
  media?.addEventListener('change', onChange)
  addEventListener('storage', onStorage)
  addEventListener(EVENT, onChange)
  return () => {
    media?.removeEventListener('change', onChange)
    removeEventListener('storage', onStorage)
    removeEventListener(EVENT, onChange)
  }
}

const snapshot = (): string => `${storedPreference()}:${systemMode()}`

/** What {@link useColourPreference} returns. */
export interface ColourPreferenceState {
  readonly preference: ColourPreference
  readonly mode: ColourMode
  readonly setPreference: (preference: ColourPreference) => void
}

/**
 * The stored colour preference and the mode it resolves to, kept in sync with other tabs and with the system
 * setting; setting it stores it and re-marks the document.
 */
export function useColourPreference(): ColourPreferenceState {
  const key = useSyncExternalStore(subscribe, snapshot, () => 'system:light')
  const [preference, system] = key.split(':') as [ColourPreference, ColourMode]
  const mode = preference === 'system' ? system : preference
  const setPreference = useCallback((next: ColourPreference) => {
    try {
      if (next === 'system') localStorage.removeItem(COLOUR_STORAGE_KEY)
      else localStorage.setItem(COLOUR_STORAGE_KEY, next)
    } catch {
      // Storage may be unavailable (private windows); the choice then lasts for this page only.
    }
    applyMode(resolveMode(next))
    dispatchEvent(new Event(EVENT))
  }, [])
  return { preference, mode, setPreference }
}
