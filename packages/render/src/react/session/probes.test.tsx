import { afterEach, describe, expect, it, vi } from 'vitest'
import { mountHook, resetGlobals } from '../_harness.tsx'
import { useProbe } from './RowsPanel.tsx'
import type { LiveSession } from './useSession.ts'

afterEach(resetGlobals)

describe('useProbe', () => {
  it('computes a step once however often it is viewed again, and does not keep a step the tape has not reached', async () => {
    const probe = vi.fn((t: number, path: string) => ({ t, path }))
    const live = { session: { t: 5, probe }, version: 0 } as unknown as LiveSession<never, never>
    const view = { t: 1 }
    const m = await mountHook(() => useProbe(live, view.t, 'plane'))
    const visit = async (t: number): Promise<unknown> => {
      view.t = t
      await m.rerender()
      return m.current.value
    }
    expect(m.current.value).toEqual({ t: 1, path: 'plane' })
    // Out and back: step 1 again is the answer already held (the same object, not a new computation).
    const first = m.current.value
    await visit(2)
    expect(await visit(1)).toBe(first)
    expect(probe).toHaveBeenCalledTimes(2)
    // A step past the tape's head may change when the tape grows, so each look computes it.
    await visit(9)
    await visit(8)
    await visit(9)
    expect(probe.mock.calls.map((c) => c[0])).toEqual([1, 2, 9, 8, 9])
    await m.unmount()
  })
})
