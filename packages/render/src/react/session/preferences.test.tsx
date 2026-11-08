import { afterEach, describe, expect, it, vi } from 'vitest'
import { inAct, mountElement, mountHook, resetGlobals } from '../_harness.tsx'
import {
  SessionPreferencesProvider,
  sessionPreferences,
  useSessionPreferences,
  useSetSessionPreferences,
} from './preferences.ts'

afterEach(() => {
  resetGlobals()
  vi.unstubAllGlobals()
})

describe('session preferences', () => {
  it('reads another tab’s change from the storage event while a component listens, and stops listening after', async () => {
    const KEY = 'omai-session-preferences'
    const store = new Map<string, string>()
    vi.stubGlobal('localStorage', {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
    })
    const m = await mountHook(useSessionPreferences)
    expect(m.current).toEqual({ mode: 'live', hold: true, fps: 60 })
    expect(m.window.count('storage')).toBe(1)
    // Another tab remembers eager mode, and the event tells this one.
    store.set(KEY, JSON.stringify({ mode: 'eager', hold: false, fps: 60 }))
    await m.window.fire('storage', { key: KEY })
    expect(m.current).toEqual({ mode: 'eager', hold: false, fps: 60 })
    expect(sessionPreferences()).toEqual({ mode: 'eager', hold: false, fps: 60 })
    // A change to some other key of the page's storage is none of its business.
    store.set(KEY, JSON.stringify({ mode: 'live', hold: true, fps: 60 }))
    await m.window.fire('storage', { key: 'theme' })
    expect(m.current).toEqual({ mode: 'eager', hold: false, fps: 60 })
    // Clearing the storage (key `null`) sends every tab back to the defaults it then holds.
    store.clear()
    await m.window.fire('storage', { key: null })
    expect(m.current).toEqual({ mode: 'live', hold: true, fps: 60 })
    await m.unmount()
    expect(m.window.count('storage')).toBe(0)
  })
})

const seen: { prefs?: ReturnType<typeof useSessionPreferences>; set?: ReturnType<typeof useSetSessionPreferences> } = {}
function Probe(): null {
  seen.prefs = useSessionPreferences()
  seen.set = useSetSessionPreferences()
  return null
}

describe('SessionPreferencesProvider', () => {
  const KEY = 'omai-session-preferences'
  const memory = (initial?: string): Storage & { readonly values: Map<string, string> } => {
    const values = new Map<string, string>(initial === undefined ? [] : [[KEY, initial]])
    return {
      values,
      getItem: (k: string) => values.get(k) ?? null,
      setItem: (k: string, v: string) => void values.set(k, v),
    } as unknown as Storage & { readonly values: Map<string, string> }
  }
  it('gives its subtree the defaults it was given, over the library’s, with no storage', async () => {
    const store = new Map<string, string>()
    vi.stubGlobal('localStorage', {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
    })
    await mountElement(
      <SessionPreferencesProvider defaults={{ mode: 'eager', fps: 30 }} storage={false}>
        <Probe />
      </SessionPreferencesProvider>,
    )
    expect(seen.prefs).toEqual({ mode: 'eager', hold: true, fps: 30 })
    // A change lasts for the subtree and is remembered nowhere: not in the browser's storage either.
    await inAct(() => seen.set?.({ mode: 'live', fps: 10 }))
    expect(seen.prefs).toEqual({ mode: 'live', hold: true, fps: 10 })
    expect(store.size).toBe(0)
    expect(sessionPreferences()).toEqual({ mode: 'live', hold: true, fps: 60 })
  })

  it('remembers changes in the storage it is given, and what that holds wins over the defaults', async () => {
    const storage = memory(JSON.stringify({ mode: 'live', hold: false, fps: 5 }))
    await mountElement(
      <SessionPreferencesProvider defaults={{ mode: 'eager', fps: 30 }} storage={storage}>
        <Probe />
      </SessionPreferencesProvider>,
    )
    expect(seen.prefs).toEqual({ mode: 'live', hold: false, fps: 5 })
    await inAct(() => seen.set?.({ fps: 120 }))
    expect(JSON.parse(storage.values.get(KEY) ?? '{}')).toEqual({ mode: 'live', hold: false, fps: 120 })
  })

  it('defaults to the browser’s storage, as without a provider, and outside any provider the browser’s store is read', async () => {
    const store = new Map<string, string>([[KEY, JSON.stringify({ mode: 'eager', hold: true, fps: 12 })]])
    vi.stubGlobal('localStorage', {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
    })
    await mountElement(
      <SessionPreferencesProvider>
        <Probe />
      </SessionPreferencesProvider>,
    )
    expect(seen.prefs).toEqual({ mode: 'eager', hold: true, fps: 12 })
    await inAct(() => seen.set?.({ hold: false }))
    expect(JSON.parse(store.get(KEY) ?? '{}')).toEqual({ mode: 'eager', hold: false, fps: 12 })
  })
})
