import { afterEach, describe, expect, it, vi } from 'vitest'
import { methodRegistry } from '../../producer/registry.ts'
import type { TapeFile } from '../../producer/tape.ts'
import { mountHook, resetGlobals } from '../_harness.tsx'
import { useTape } from './useTape.ts'

afterEach(resetGlobals)

describe('useTape', () => {
  it('gives no handle and reports why for a tape whose method the page does not register, instead of throwing', async () => {
    const onError = vi.fn()
    const file = { method: 'nowhere', variants: [] } as unknown as TapeFile
    const registry = methodRegistry([])
    const m = await mountHook(() => useTape(file, registry, onError))
    expect(m.current).toBeUndefined()
    expect(onError).toHaveBeenCalledTimes(1)
    expect(onError.mock.calls[0]?.[0]).toMatch(/no method called 'nowhere'/)
    await m.rerender()
    expect(onError).toHaveBeenCalledTimes(1)
    await m.unmount()
  })
})
