import { circle, scene, structure } from 'omai-engine/scene'
import type { Algorithm } from 'omai-engine/trace'
import type { ReactElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { type FakeFrames, fakeFrames, inAct, mountElement, mountHook, resetGlobals } from './_harness.tsx'
import { SceneView } from './SceneView.tsx'
import { StructureView } from './StructureView.tsx'
import { settledStateOf, waitForSettled } from './settled.ts'
import { useLiveRun } from './usePlayback.ts'

let frames: FakeFrames | undefined
beforeEach(() => {
  frames = fakeFrames()
})
afterEach(() => {
  frames?.restore()
  vi.useRealTimers()
  delete (globalThis as Record<string, unknown>)['MutationObserver']
  resetGlobals()
})

describe('settledStateOf', () => {
  it('is failed on an error or a disposed incomplete tape, else ready once whole, else preparing', () => {
    const base = { done: false, error: undefined, disposed: false }
    expect(settledStateOf(base)).toBe('preparing')
    expect(settledStateOf({ ...base, done: true })).toBe('ready')
    expect(settledStateOf({ ...base, error: 'x' })).toBe('failed')
    expect(settledStateOf({ ...base, error: 'x', done: true })).toBe('failed')
    expect(settledStateOf({ ...base, disposed: true })).toBe('failed')
    expect(settledStateOf({ ...base, disposed: true, done: true })).toBe('ready')
  })
})

// The figures with a state, as the stand-in DOM and the observer see them: elements that answer `getAttribute`, and an
// observer that a test fires by hand.
class Figure {
  constructor(public state: string | null) {}
  getAttribute(name: string): string | null {
    return name === 'data-omai-state' ? this.state : null
  }
}
class Root {
  constructor(public figures: Figure[]) {}
  querySelectorAll(): Figure[] {
    return this.figures
  }
}
class FakeObserver {
  static live: FakeObserver[] = []
  disconnected = false
  constructor(readonly callback: () => void) {
    FakeObserver.live.push(this)
  }
  observe = vi.fn()
  disconnect(): void {
    this.disconnected = true
  }
}
const fire = (): void => {
  for (const o of FakeObserver.live) if (!o.disconnected) o.callback()
}

describe('waitForSettled', () => {
  const install = (): void => {
    FakeObserver.live = []
    ;(globalThis as Record<string, unknown>)['MutationObserver'] = FakeObserver
  }

  it('resolves at once, with a count, when every figure is ready and when there are none', async () => {
    install()
    expect(await waitForSettled(new Root([]) as never)).toEqual({ ok: true, ready: 0 })
    expect(await waitForSettled(new Root([new Figure('ready'), new Figure('ready')]) as never)).toEqual({
      ok: true,
      ready: 2,
    })
    expect(FakeObserver.live).toEqual([])
  })

  it('waits while any figure is preparing, watching the subtree for state changes and new figures', async () => {
    install()
    const a = new Figure('preparing')
    const root = new Root([a])
    let result: unknown
    void waitForSettled(root as never).then((r) => {
      result = r
    })
    await Promise.resolve()
    expect(result).toBeUndefined()
    const watching = FakeObserver.live[0]
    expect(watching?.observe).toHaveBeenCalledWith(
      root,
      expect.objectContaining({ subtree: true, attributeFilter: ['data-omai-state'] }),
    )
    // A figure that appears later counts too.
    root.figures.push(new Figure('preparing'))
    a.state = 'ready'
    fire()
    await Promise.resolve()
    expect(result).toBeUndefined()
    for (const f of root.figures) f.state = 'ready'
    fire()
    await Promise.resolve()
    expect(result).toEqual({ ok: true, ready: 2 })
    expect(watching?.disconnected).toBe(true)
  })

  it('reports failed figures once none is preparing, and the figures still preparing at the deadline', async () => {
    install()
    const failed = new Figure('failed')
    const slow = new Figure('preparing')
    const root = new Root([failed, slow])
    vi.useFakeTimers()
    const outcome = waitForSettled(root as never, { timeoutMs: 500 })
    vi.advanceTimersByTime(499)
    fire()
    slow.state = 'ready'
    fire()
    expect(await outcome).toEqual({ ok: false, reason: 'failed', failed: [failed], pending: [] })

    const stuck = new Figure('preparing')
    const late = waitForSettled(new Root([stuck]) as never, { timeoutMs: 500 })
    vi.advanceTimersByTime(500)
    expect(await late).toEqual({ ok: false, reason: 'timeout', failed: [], pending: [stuck] })
  })

  it('counts the root itself when it is a figure, and stops at the deadline without a MutationObserver', async () => {
    const self = { querySelectorAll: () => [], getAttribute: () => 'preparing' }
    vi.useFakeTimers()
    const outcome = waitForSettled(self as never, { timeoutMs: 100 })
    vi.advanceTimersByTime(100)
    expect(await outcome).toMatchObject({ ok: false, reason: 'timeout', pending: [self] })
  })
})

describe('SceneView and StructureView settled signal', () => {
  const stateOf = (m: Awaited<ReturnType<typeof mountElement>>, mark: string): unknown =>
    m.elements().find((e) => e.props[mark] !== undefined)?.props['data-omai-state']

  it('a scene is ready from its first commit on', async () => {
    const m = await mountElement(<SceneView scene={scene(100, 50, [circle(20, 20, 5)])} mode="light" />)
    expect(stateOf(m, 'data-omai-scene')).toBe('ready')
    await m.unmount()
  })

  it('a diagram is ready from its first commit on', async () => {
    const s = structure(
      't',
      [
        { id: 'a', label: 'a', role: 'input' },
        { id: 'b', label: 'b', role: 'output' },
      ],
      [{ from: 'a', to: 'b' }],
    )
    const element: ReactElement = <StructureView structure={s} frame={false} chrome={false} />
    const m = await mountElement(element)
    const root = m.elements().find((e) => e.props['data-omai-state'] !== undefined && e.tag === 'div')
    expect(root?.props['data-omai-state']).toBe('ready')
    await m.unmount()
  })
})

describe('useLiveRun settled', () => {
  const counter: Algorithm<number> = { name: 'counter', config: {}, init: () => 0, step: (s) => s + 1 }

  it('resolves ok once the whole run exists, a fresh promise for a new seed, disposed for a run let go of', async () => {
    const seed = { n: 0 }
    const m = await mountHook(() => useLiveRun(counter, { maxSteps: 30, start: 'first', seed: seed.n }))
    const first = m.current.settled
    for (let i = 0; i < 100 && m.current.phase !== 'ready'; i++) await (frames as FakeFrames).frame()
    expect(m.current.phase).toBe('ready')
    expect(await first).toEqual({ ok: true })
    // A new seed is a new run: a fresh promise, pending while it fills; the old result is kept.
    seed.n = 1
    await m.rerender()
    const second = m.current.settled
    expect(second).not.toBe(first)
    expect(await first).toEqual({ ok: true })
    await inAct(() => m.current.dispose())
    expect(m.current.disposed).toBe(true)
    expect(await second).toEqual({ ok: false, error: 'disposed' })
  })
})
