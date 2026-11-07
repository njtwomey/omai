import { afterEach, describe, expect, it, vi } from 'vitest'
import { mountHook, resetGlobals } from '../_harness.tsx'
import { sessionPreferences, useSessionPreferences } from './preferences.ts'

afterEach(() => {
  resetGlobals()
  vi.unstubAllGlobals()
})

describe('session preferences', () => {
  it('reads another tab’s change from the storage event while a component listens, and stops listening after', async () => {
    const KEY = 'mdml-session-preferences'
    const store = new Map<string, string>()
    vi.stubGlobal('localStorage', {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
    })
    const m = await mountHook(useSessionPreferences)
    expect(m.current).toEqual({ mode: 'live', hold: true })
    expect(m.window.count('storage')).toBe(1)
    // Another tab remembers eager mode, and the event tells this one.
    store.set(KEY, JSON.stringify({ mode: 'eager', hold: false }))
    await m.window.fire('storage', { key: KEY })
    expect(m.current).toEqual({ mode: 'eager', hold: false })
    expect(sessionPreferences()).toEqual({ mode: 'eager', hold: false })
    // A change to some other key of the page's storage is none of its business.
    store.set(KEY, JSON.stringify({ mode: 'live', hold: true }))
    await m.window.fire('storage', { key: 'theme' })
    expect(m.current).toEqual({ mode: 'eager', hold: false })
    // Clearing the storage (key `null`) sends every tab back to the defaults it then holds.
    store.clear()
    await m.window.fire('storage', { key: null })
    expect(m.current).toEqual({ mode: 'live', hold: true })
    await m.unmount()
    expect(m.window.count('storage')).toBe(0)
  })
})
