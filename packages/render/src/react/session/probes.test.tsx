import { afterEach, describe, expect, it, vi } from 'vitest'
import { mountHook, resetGlobals } from '../_harness.tsx'
import { useProbe } from './RowsPanel.tsx'

afterEach(resetGlobals)

describe('useProbe', () => {
  it('computes a step once however often it is viewed again, and does not keep a step the tape has not reached', async () => {
    const probe = vi.fn((t: number, path: string) => ({ t, path }))
    const live = { session: { t: 5, probe }, version: 0 } as never
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

  it('asks again when a worker has answered, and shows the answer and not the stand-in kept for the pending one', async () => {
    let answer: unknown
    const probe = vi.fn(() => {
      if (answer === undefined) throw new Error('computing plane at step 3 in the worker')
      return answer
    })
    const view = { probed: 0 }
    const live = (): never =>
      ({ session: { t: 5, probe, probeAsync: () => Promise.resolve(null) }, version: 0, probed: view.probed }) as never
    const m = await mountHook(() => useProbe(live(), 3, 'plane'))
    expect(m.current.error).toMatch(/in the worker/)
    answer = { kind: 'plane' }
    view.probed = 1
    await m.rerender()
    expect(m.current.value).toEqual({ kind: 'plane' })
    await m.unmount()
  })
})
