/**
 * The reader's standing preferences for sessions, remembered in the browser from page to page: eager against live
 * (compute a run whole and show its end, or compute in the background while it plays), whether a change to the
 * configuration waits for play, and the player's rate. One store, so the player's eager button and the settings panel
 * show the same state. By default the store is the browser's `localStorage`; a `SessionPreferencesProvider` gives the
 * components below it a store of their own, with the defaults the host wants and the storage it chooses (none at all,
 * or its own), so a host with its own settings card is not at the mercy of the reader's stored choice.
 */

import {
  createContext,
  createElement,
  type ReactElement,
  type ReactNode,
  useContext,
  useMemo,
  useSyncExternalStore,
} from 'react'

/** The preferences. */
export interface SessionPreferences {
  /** Eager computes the whole run at once and shows its end; live computes in the background while it plays. */
  readonly mode: 'eager' | 'live'
  /** In live mode, hold a change to the configuration until play is pressed (eager applies it after a short pause). */
  readonly hold: boolean
  /** The player's rate in steps per second (default 60), until the reader picks another from the player's menu. */
  readonly fps: number
}

/** The preferences before the reader has set any: live, holding changes, 60 steps a second. */
export const DEFAULT_PREFERENCES: SessionPreferences = { mode: 'live', hold: true, fps: 60 }

const KEY = 'omai-session-preferences'

/** The parts of a `Storage` a preference store uses (so a host can pass a stand-in). */
export type PreferenceStorage = Pick<Storage, 'getItem' | 'setItem'>

/** A store of preferences: its current value, a change, and a subscription (for `useSyncExternalStore`). */
interface PreferenceStore {
  get(): SessionPreferences
  set(changes: Partial<SessionPreferences>): void
  subscribe(listener: () => void): () => void
}

// What `storage` has remembered over the defaults, or the defaults when it holds nothing, is unavailable or holds
// something else.
function read(storage: PreferenceStorage | undefined, defaults: SessionPreferences): SessionPreferences {
  try {
    const raw = storage?.getItem(KEY)
    if (raw !== null && raw !== undefined) {
      const p = JSON.parse(raw) as Partial<SessionPreferences>
      return {
        mode: p.mode === 'eager' ? 'eager' : p.mode === 'live' ? 'live' : defaults.mode,
        hold: typeof p.hold === 'boolean' ? p.hold : defaults.hold,
        fps: typeof p.fps === 'number' && p.fps > 0 ? p.fps : defaults.fps,
      }
    }
  } catch {
    // Storage may be unavailable (private windows) or hold something else; the defaults stand for this page.
  }
  return defaults
}

function browserStorage(): PreferenceStorage | undefined {
  try {
    return typeof localStorage === 'undefined' ? undefined : localStorage
  } catch {
    return undefined
  }
}

// A store over a storage, which is looked up again when it is read, `undefined` meaning the browser's (found only when
// used, so a page that stubs `localStorage` is read as it stands then). While anything listens, another tab's change to
// the remembered preferences (a `storage` event) is read in, so the tabs agree.
function makeStore(defaults: SessionPreferences, storage: () => PreferenceStorage | undefined): PreferenceStore {
  const listeners = new Set<() => void>()
  let current: SessionPreferences | undefined
  let stopStorage: (() => void) | undefined
  const get = (): SessionPreferences => {
    current ??= read(storage(), defaults)
    return current
  }
  return {
    get,
    set: (changes) => {
      current = { ...get(), ...changes }
      try {
        storage()?.setItem(KEY, JSON.stringify(current))
      } catch {
        // Storage may be unavailable; the change still applies for this page.
      }
      for (const l of listeners) l()
    },
    subscribe: (listener) => {
      listeners.add(listener)
      if (listeners.size === 1 && typeof window !== 'undefined' && storage() !== undefined) {
        const onStorage = (e: StorageEvent): void => {
          if (e.key !== null && e.key !== KEY) return
          current = read(storage(), defaults)
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
    },
  }
}

// The store of a page with no provider: the browser's `localStorage`, with the library's defaults.
const browser = makeStore(DEFAULT_PREFERENCES, browserStorage)
const StoreContext = createContext<PreferenceStore>(browser)

/**
 * The preferences as they stand for a page with no provider (the remembered ones on first use, else the defaults; while
 * a component listens, a change made in another tab is picked up). Inside a `SessionPreferencesProvider` use
 * `useSessionPreferences`.
 *
 * @returns The preferences.
 */
export function sessionPreferences(): SessionPreferences {
  return browser.get()
}

/**
 * Change some preferences and remember them, for a page with no provider. Inside a `SessionPreferencesProvider` use
 * `useSetSessionPreferences`.
 *
 * @param changes - The preferences to change.
 */
export function setSessionPreferences(changes: Partial<SessionPreferences>): void {
  browser.set(changes)
}

/**
 * The preferences, re-rendering on change: those of the nearest `SessionPreferencesProvider`, else the browser's.
 *
 * @returns The preferences.
 * @index function React hook: the reader's live or eager mode, hold and rate: `useSessionPreferences().mode`
 */
export function useSessionPreferences(): SessionPreferences {
  const store = useContext(StoreContext)
  return useSyncExternalStore(store.subscribe, store.get, () => (store === browser ? DEFAULT_PREFERENCES : store.get()))
}

/**
 * The setter of the preferences of the nearest `SessionPreferencesProvider`, else the browser's.
 *
 * @returns A function that changes some preferences (and remembers them in the store's storage, if it has one).
 */
export function useSetSessionPreferences(): (changes: Partial<SessionPreferences>) => void {
  return useContext(StoreContext).set
}

/**
 * Give the components below it a store of session preferences of their own. `defaults` are what holds until the reader
 * (or the host) changes something, and a value the storage remembers wins over them; `storage` is where changes are
 * remembered: omitted, the browser's `localStorage` (the key `omai-session-preferences`, as without a provider);
 * `false`, nowhere, so the defaults hold on every load and a change lasts for the page; or a `Storage` of the host's.
 * Read the store with `useSessionPreferences`, change it with `useSetSessionPreferences`; the player reads its starting
 * mode and rate from it. A provider replaces the page's store for the components below it; it does not read or write
 * the browser's.
 *
 * @param props - The defaults, the storage and the components.
 * @returns The components, with the store in context.
 * @index function React component: a store of session preferences for a subtree: `<SessionPreferencesProvider defaults={{ fps: 30 }} storage={false}>`
 * @example
 * ```tsx
 * import { SessionPreferencesProvider } from 'omai-render/react'
 * const app = (
 *   <SessionPreferencesProvider defaults={{ mode: 'eager', fps: 30 }} storage={false}>
 *     <Page />
 *   </SessionPreferencesProvider>
 * )
 * ```
 */
export function SessionPreferencesProvider(props: {
  readonly defaults?: Partial<SessionPreferences>
  readonly storage?: false | PreferenceStorage
  readonly children?: ReactNode
}): ReactElement {
  const { mode, hold, fps } = { ...DEFAULT_PREFERENCES, ...props.defaults }
  const { storage } = props
  const store = useMemo(
    () =>
      makeStore({ mode, hold, fps }, () => {
        if (storage === false) return undefined
        return storage ?? browserStorage()
      }),
    [mode, hold, fps, storage],
  )
  return createElement(StoreContext.Provider, { value: store }, props.children)
}
