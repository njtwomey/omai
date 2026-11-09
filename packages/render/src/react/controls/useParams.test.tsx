import { afterEach, describe, expect, it } from 'vitest'
import { inAct, mountHook, resetGlobals } from '../_harness.tsx'
import { useParams } from './useParams.ts'

afterEach(resetGlobals)

const start = { noise: 0.1, kind: 'sine', smooth: true, knots: [1, 2, 3] }

describe('useParams', () => {
  it('starts at the initial values, all default', async () => {
    const m = await mountHook(() => useParams(start))
    expect(m.current.values).toEqual(start)
    expect(m.current.initial).toBe(start)
    expect(m.current.isDefault).toBe(true)
  })

  it('merges a patch, and isDefault follows the values', async () => {
    const m = await mountHook(() => useParams(start))
    await inAct(() => m.current.set({ noise: 0.5 }))
    expect(m.current.values).toEqual({ ...start, noise: 0.5 })
    expect(m.current.isDefault).toBe(false)
    await inAct(() => m.current.set({ kind: 'square', smooth: false }))
    expect(m.current.values).toEqual({ noise: 0.5, kind: 'square', smooth: false, knots: [1, 2, 3] })
    // Setting a value back to its initial one, an array by its elements, makes it default again.
    await inAct(() => m.current.set({ noise: 0.1, kind: 'sine', smooth: true, knots: [1, 2, 3] }))
    expect(m.current.isDefault).toBe(true)
  })

  it('keeps the values object and the callbacks when nothing changes', async () => {
    const m = await mountHook(() => useParams(start))
    const first = m.current
    const rendered = m.renders
    await inAct(() => first.set({ noise: 0.1, knots: [1, 2, 3] }))
    await inAct(() => first.set({}))
    await inAct(() => first.reset())
    expect(m.current.values).toBe(first.values)
    expect(m.current).toBe(first)
    await inAct(() => first.set({ noise: 0.2 }))
    expect(m.current.values).not.toBe(first.values)
    expect(m.current.set).toBe(first.set)
    expect(m.current.reset).toBe(first.reset)
    expect(m.renders).toBeGreaterThan(rendered)
  })

  it('treats numbers by ===, so 0 and -0 and NaN with NaN agree', async () => {
    const m = await mountHook(() => useParams({ a: 0, b: Number.NaN }))
    await inAct(() => m.current.set({ a: -0, b: Number.NaN }))
    expect(m.current.isDefault).toBe(true)
    await inAct(() => m.current.set({ b: 1 }))
    expect(m.current.isDefault).toBe(false)
  })

  it('resets the named parameters, or all of them', async () => {
    const m = await mountHook(() => useParams(start))
    await inAct(() => m.current.set({ noise: 0.9, kind: 'saw', knots: [4] }))
    await inAct(() => m.current.reset(['noise', 'knots']))
    expect(m.current.values).toEqual({ ...start, kind: 'saw' })
    expect(m.current.isDefault).toBe(false)
    await inAct(() => m.current.reset())
    expect(m.current.values).toEqual(start)
    expect(m.current.isDefault).toBe(true)
  })

  it('reads the initial values once', async () => {
    let seed = { a: 1 }
    const m = await mountHook(() => useParams(seed))
    seed = { a: 2 }
    await m.rerender()
    expect(m.current.initial).toEqual({ a: 1 })
    expect(m.current.values).toEqual({ a: 1 })
  })

  it('holds a list of points, takes a read-only tuple where the initial value was a mutable array, and compares nested arrays by value', async () => {
    const waypoints: readonly (readonly [number, number])[] = [
      [0, 0],
      [1, 2],
    ]
    const m = await mountHook(() => useParams({ waypoints, microphone: [0.5, 0.5] }))
    const first = m.current
    const tuple = [0.5, 0.5] as const
    await inAct(() => first.set({ microphone: tuple }))
    expect(m.current).toBe(first)
    await inAct(() =>
      first.set({
        waypoints: [
          [0, 0],
          [1, 2],
        ],
      }),
    )
    expect(m.current).toBe(first)
    await inAct(() =>
      first.set({
        waypoints: [
          [0, 0],
          [1, 3],
        ],
        microphone: [0.1, 0.9] as const,
      }),
    )
    expect(m.current.values.waypoints[1]).toEqual([1, 3])
    expect(m.current.isDefault).toBe(false)
  })

  it('resets all when reset is called as an event handler, and the named ones when given a list', async () => {
    const m = await mountHook(() => useParams({ a: 1, b: 2 }))
    await inAct(() => m.current.set({ a: 5, b: 6 }))
    await inAct(() => m.current.reset({ type: 'click' }))
    expect(m.current.values).toEqual({ a: 1, b: 2 })
    await inAct(() => m.current.set({ a: 5, b: 6 }))
    await inAct(() => m.current.reset(['a']))
    expect(m.current.values).toEqual({ a: 1, b: 6 })
  })
})
