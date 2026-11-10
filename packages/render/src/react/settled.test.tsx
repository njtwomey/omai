import { circle, scene, structure } from 'omai-engine/scene'
import type { Algorithm } from 'omai-engine/trace'
import type { ReactElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { type FakeFrames, fakeFrames, inAct, mountElement, mountHook, resetGlobals } from './_harness.tsx'
import { Player } from './Player.tsx'
import { SceneView } from './SceneView.tsx'
import { StructureView } from './StructureView.tsx'
import { settledStateOf, waitForSettled } from './settled.ts'
import { useLiveRun, usePlayback } from './usePlayback.ts'

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

  it('is idle for an incomplete tape with nothing computing, and idle never outranks ready, failed or disposed', () => {
    const base = { done: false, error: undefined, disposed: false }
    expect(settledStateOf({ ...base, idle: true })).toBe('idle')
    expect(settledStateOf({ ...base, idle: false })).toBe('preparing')
    expect(settledStateOf({ ...base, idle: true, done: true })).toBe('ready')
    expect(settledStateOf({ ...base, idle: true, error: 'x' })).toBe('failed')
    expect(settledStateOf({ ...base, idle: true, disposed: true })).toBe('failed')
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

  it('settles on idle figures like on ready ones, and says which were idle', async () => {
    install()
    const idle = new Figure('idle')
    expect(await waitForSettled(new Root([new Figure('ready'), idle]) as never)).toEqual({
      ok: true,
      ready: 2,
      idle: [idle],
    })
    // A figure still preparing keeps it waiting, and the idle one is reported with the failure or the timeout.
    const failed = new Figure('failed')
    expect(await waitForSettled(new Root([idle, failed]) as never)).toEqual({
      ok: false,
      reason: 'failed',
      failed: [failed],
      pending: [],
      idle: [idle],
    })
    vi.useFakeTimers()
    const slow = new Figure('preparing')
    const late = waitForSettled(new Root([idle, slow]) as never, { timeoutMs: 200 })
    vi.advanceTimersByTime(200)
    expect(await late).toEqual({ ok: false, reason: 'timeout', failed: [], pending: [slow], idle: [idle] })
    // It resolves when the last preparing figure goes idle.
    const root = new Root([new Figure('preparing')])
    const outcome = waitForSettled(root as never)
    ;(root.figures[0] as Figure).state = 'idle'
    fire()
    expect(await outcome).toEqual({ ok: true, ready: 1, idle: root.figures })
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

  it('is idle while unstarted, preparing while played, idle when paused, ready once whole; inactive is idle', async () => {
    const live = { active: true }
    const m = await mountHook(() => useLiveRun(counter, { maxSteps: 30, active: live.active }))
    // Never started: nothing computes, the state is idle (the promise still waits for the end).
    expect(m.current.controller.state).toBe('idle')
    expect(await Promise.race([m.current.settled, Promise.resolve('pending')])).toBe('pending')
    await inAct(() => m.current.controller.play())
    expect(m.current.controller.state).toBe('preparing')
    await inAct(() => m.current.controller.pause())
    expect(m.current.controller.state).toBe('idle')
    await inAct(() => m.current.controller.play())
    // Playing but not active: the ticker is off, so nothing computes.
    live.active = false
    await m.rerender()
    expect(m.current.controller.state).toBe('idle')
    live.active = true
    await m.rerender()
    for (let i = 0; i < 200 && m.current.phase !== 'ready'; i++) await (frames as FakeFrames).frame()
    expect(m.current.controller.state).toBe('ready')
    expect(await m.current.settled).toEqual({ ok: true })
  })

  it('is preparing while a primed run fills, from the first render, and idle when it is primed but inactive', async () => {
    const m = await mountHook(() => useLiveRun(counter, { maxSteps: 30, start: 'first' }))
    expect(m.current.controller.state).toBe('preparing')
    const off = await mountHook(() => useLiveRun(counter, { maxSteps: 30, start: 'first', active: false }))
    expect(off.current.controller.state).toBe('idle')
  })

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

const ticking: Algorithm<number> = { name: 'counter', config: {}, init: () => 0, step: (s) => s + 1 }
function Live(props: { readonly maxSteps: number }): ReactElement {
  const live = useLiveRun(ticking, { maxSteps: props.maxSteps, start: 'first' })
  return <Player controller={live.controller} />
}
function Fixed(): ReactElement {
  return <Player controller={usePlayback(5)} />
}
function Bare(): ReactElement {
  const { state: _state, ...rest } = usePlayback(5)
  return <Player controller={rest} />
}

describe('Player settled signal', () => {
  type Mount = Awaited<ReturnType<typeof mountElement>>
  const stateOfPlayer = (m: Mount): unknown =>
    m.elements().find((e) => e.props['aria-label'] === undefined)?.props['data-omai-state']
  // The elements that carry a state, as a root for `waitForSettled` (the stand-in DOM has no document-level query).
  const rootOf = (m: Mount, chart: Figure): { querySelectorAll: () => Figure[] } => ({
    querySelectorAll: () => [
      chart,
      ...m
        .elements()
        .filter((e) => e.props['data-omai-state'] !== undefined)
        .map((e) => ({ getAttribute: (n: string) => (n === 'data-omai-state' ? String(e.props[n]) : null) }) as Figure),
    ],
  })

  it('follows a live run from preparing to ready, and a fixed number of frames is ready at once', async () => {
    const m = await mountElement(<Live maxSteps={60} />)
    expect(stateOfPlayer(m)).toBe('preparing')
    let ready = false
    for (let i = 0; i < 200 && !ready; i++) {
      await (frames as FakeFrames).frame()
      ready = stateOfPlayer(m) === 'ready'
    }
    expect(ready).toBe(true)
    await m.unmount()
    const fixed = await mountElement(<Fixed />)
    expect(stateOfPlayer(fixed)).toBe('ready')
    await fixed.unmount()
  })

  it('carries no attribute when its controller has no state', async () => {
    const m = await mountElement(<Bare />)
    expect(m.elements().some((e) => e.props['data-omai-state'] !== undefined)).toBe(false)
    await m.unmount()
  })

  it('makes waitForSettled on a root holding a Player and a ready Chart wait for the run', async () => {
    ;(globalThis as Record<string, unknown>)['MutationObserver'] = class {
      constructor(readonly cb: () => void) {
        observers.push(this)
      }
      observe(): void {}
      disconnect(): void {}
    }
    const observers: { cb: () => void }[] = []
    const m = await mountElement(<Live maxSteps={60} />)
    const root = rootOf(m, new Figure('ready'))
    let result: unknown
    void waitForSettled(root as never).then((r) => {
      result = r
    })
    await Promise.resolve()
    expect(result).toBeUndefined()
    for (let i = 0; i < 200 && stateOfPlayer(m) !== 'ready'; i++) await (frames as FakeFrames).frame()
    for (const o of observers) o.cb()
    await Promise.resolve()
    expect(result).toEqual({ ok: true, ready: 2 })
    await m.unmount()
  })
})
