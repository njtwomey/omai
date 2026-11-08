import type { Algorithm } from 'omai-engine/trace'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { type FakeFrames, fakeFrames, inAct, mountHook, resetGlobals } from './_harness.tsx'
import { type LiveRun, type LiveRunOptions, useLiveRun, useTicker } from './usePlayback.ts'

const counter: Algorithm<number> = { name: 'counter', config: {}, init: () => 0, step: (s) => s + 1 }

let frames: FakeFrames
beforeEach(() => {
  frames = fakeFrames()
})
afterEach(() => {
  frames.restore()
  resetGlobals()
})

const mountRun = (options: LiveRunOptions, strict = false) => mountHook(() => useLiveRun(counter, options), { strict })

describe('useLiveRun history', () => {
  it('keeps the latest states in a ring buffer, and stepping back stops at the oldest kept', async () => {
    const m = await mountRun({ maxSteps: 100, history: 10, fps: 60, autoplay: true })
    // Play for a while: the run is far past the history, which holds ten states, the newest at the cursor.
    await frames.frame()
    for (let n = 0; n < 40; n++) await frames.frame(100)
    const run = m.current
    expect(run.controller.count).toBe(10)
    expect(run.t).toBeGreaterThan(20)
    expect(run.oldest).toBe(run.t - 9)
    // Stepping back one at a time reaches the oldest state kept and stays there.
    for (let i = 0; i < 20; i++) await inAct(() => m.current.controller.step(-1))
    expect(m.current.controller.frame).toBe(0)
    expect(m.current.t).toBe(run.oldest)
    // Seeking inside the buffer shows that state.
    await inAct(() => m.current.controller.seek(4))
    expect(m.current.t).toBe(run.oldest + 4)
  })

  it('keeps everything up to the history length', async () => {
    const m = await mountRun({ maxSteps: 30, history: 2000, fps: 60, autoplay: true })
    await frames.frame()
    for (let n = 0; n < 30; n++) await frames.frame(100)
    expect(m.current.status).toBe('completed')
    expect(m.current.controller.count).toBe(31)
    expect(m.current.oldest).toBe(0)
  })
})

describe('useLiveRun start policies', () => {
  it('first: computes the whole run in slices, parked at step 0 and paused, then ready', async () => {
    const ready = vi.fn()
    const m = await mountRun({ maxSteps: 300, history: 5, start: 'first', onReady: ready })
    expect(m.current.phase).toBe('preparing')
    expect(m.current.controller.playing).toBe(false)
    let n = 0
    const progress: number[] = []
    while (m.current.status === undefined && n++ < 400) {
      await frames.frame()
      progress.push(m.current.progress)
      expect(m.current.t).toBe(0)
      expect(m.current.controller.playing).toBe(false)
    }
    expect(n).toBeGreaterThan(2)
    expect(progress.some((p) => p > 0 && p < 1)).toBe(true)
    expect(m.current.phase).toBe('ready')
    expect(m.current.progress).toBe(1)
    // The whole run is kept, whatever the history said, so step 0 is still the first frame.
    expect(m.current.controller.count).toBe(301)
    expect(m.current.oldest).toBe(0)
    expect(ready).toHaveBeenCalledTimes(1)
    await inAct(() => m.current.controller.play())
    expect(m.current.phase).toBe('running')
    await m.unmount()
  })

  it('end: shows the end when the whole run exists', async () => {
    const m = await mountRun({ maxSteps: 120, start: 'end' })
    expect(m.current.t).toBe(0)
    for (let n = 0; n < 400 && m.current.status === undefined; n++) await frames.frame()
    await frames.frame()
    expect(m.current.t).toBe(120)
    expect(m.current.phase).toBe('ready')
  })

  it('play (the default): nothing is computed until play', async () => {
    const m = await mountRun({ maxSteps: 100 })
    await frames.frames(5)
    expect(m.current.controller.count).toBe(1)
    expect(m.current.phase).toBe('preparing')
    expect(frames.pending).toBe(0)
  })
})

describe('useLiveRun dispose', () => {
  it('stops priming and releases the states; played again it starts from step 0', async () => {
    const m = await mountRun({ maxSteps: 300, start: 'first' })
    await frames.frames(3)
    const partway = m.current.controller.count
    expect(partway).toBeGreaterThan(1)
    await inAct(() => m.current.dispose())
    expect(m.current.controller.count).toBe(1)
    expect(m.current.t).toBe(0)
    expect(m.current.progress).toBe(0)
    expect(frames.pending).toBe(0)
    await frames.frames(3)
    expect(m.current.controller.count).toBe(1)
    await inAct(() => m.current.controller.play())
    await frames.frames(3)
    expect(m.current.controller.count).toBeGreaterThan(1)
    await m.unmount()
  })

  it('unmounting stops the priming, and a strict-mode double effect does not release the run', async () => {
    const m = await mountRun({ maxSteps: 300, start: 'first' }, true)
    await frames.frames(2)
    expect(m.current.controller.count).toBeGreaterThan(1)
    await m.unmount()
    expect(frames.pending).toBe(0)
  })
})

describe('useLiveRun dispose races', () => {
  it('a priming frame queued before dispose runs after it without throwing, and nothing more is computed', async () => {
    let live: LiveRun<number> | undefined
    let queued = false
    const m = await mountHook(() => {
      // Asked for before the hook's own effects, so it is the first frame callback of the next frame.
      if (!queued) {
        queued = true
        requestAnimationFrame(() => live?.dispose())
      }
      live = useLiveRun(counter, { maxSteps: 300, start: 'first' })
      return live
    })
    expect(frames.pending).toBe(2)
    await frames.frame()
    expect(m.current.disposed).toBe(true)
    expect(m.current.controller.count).toBe(1)
    expect(m.current.t).toBe(0)
    await frames.frames(3)
    expect(m.current.controller.count).toBe(1)
    expect(frames.pending).toBe(0)
    await m.unmount()
  })

  it('a clock tick queued before dispose runs after it without throwing', async () => {
    let live: LiveRun<number> | undefined
    let armed = false
    const m = await mountHook(() => {
      // Its tick is queued ahead of the run's own, every frame.
      useTicker(true, 60, () => {
        if (armed) live?.dispose()
      })
      live = useLiveRun(counter, { maxSteps: 300, autoplay: true, fps: 60 })
      return live
    })
    await frames.frame()
    await frames.frame(100)
    expect(m.current.controller.count).toBeGreaterThan(1)
    armed = true
    await frames.frame(100)
    expect(m.current.disposed).toBe(true)
    expect(m.current.controller.count).toBe(1)
    expect(m.current.controller.playing).toBe(false)
    await m.unmount()
  })

  it('a step by hand and a play after dispose use the run again', async () => {
    const m = await mountRun({ maxSteps: 300, start: 'first' })
    await frames.frames(2)
    await inAct(() => m.current.dispose())
    expect(m.current.disposed).toBe(true)
    expect(m.current.phase).toBe('preparing')
    await inAct(() => m.current.controller.step(1))
    expect(m.current.disposed).toBe(false)
    expect(m.current.controller.count).toBe(2)
    await inAct(() => m.current.dispose())
    await inAct(() => m.current.controller.play())
    expect(m.current.disposed).toBe(false)
    await m.unmount()
  })
})

describe('useLiveRun progress', () => {
  // A run that stops by itself after 8 of its 1000 steps.
  const early: Algorithm<number> = { ...counter, name: 'early', done: (s) => (s >= 8 ? 'converged' : undefined) }

  it('is against the budget while the run goes on and 1 on the step that ends it, whatever its own length', async () => {
    const m = await mountHook(() => useLiveRun(early, { maxSteps: 1000, start: 'first' }))
    expect(m.current.progress).toBe(0)
    for (let n = 0; n < 50 && m.current.status === undefined; n++) await frames.frame()
    expect(m.current.status).toBe('converged')
    expect(m.current.controller.count).toBe(9)
    expect(m.current.progress).toBe(1)
    expect(m.current.phase).toBe('ready')
    await m.unmount()
  })
})

describe('useLiveRun active', () => {
  it('computes nothing while inactive (no priming, the clock off), keeps the states, and carries on when active', async () => {
    let active = false
    const m = await mountHook(() => useLiveRun(counter, { maxSteps: 300, start: 'first', active }))
    expect(frames.pending).toBe(0)
    await frames.frames(3)
    expect(m.current.controller.count).toBe(1)
    active = true
    await m.rerender()
    await frames.frames(2)
    const partway = m.current.controller.count
    expect(partway).toBeGreaterThan(1)
    // Hidden again: the states stay, nothing more is computed, no frame is asked for.
    active = false
    await m.rerender()
    await frames.frames(2)
    expect(m.current.controller.count).toBe(partway)
    expect(frames.pending).toBe(0)
    expect(m.current.disposed).toBe(false)
    active = true
    await m.rerender()
    for (let n = 0; n < 400 && m.current.status === undefined; n++) await frames.frame()
    expect(m.current.status).toBe('completed')
    expect(m.current.controller.count).toBe(301)
    await m.unmount()
  })

  it('stops the clock of a playing run, which carries on from where it stood', async () => {
    let active = true
    const m = await mountHook(() => useLiveRun(counter, { maxSteps: 300, autoplay: true, fps: 60, active }))
    await frames.frame()
    for (let n = 0; n < 3; n++) await frames.frame(100)
    const at = m.current.t
    expect(at).toBeGreaterThan(0)
    active = false
    await m.rerender()
    await frames.frames(3)
    expect(m.current.t).toBe(at)
    expect(frames.pending).toBe(0)
    active = true
    await m.rerender()
    await frames.frame()
    for (let n = 0; n < 3; n++) await frames.frame(100)
    expect(m.current.t).toBeGreaterThan(at)
    await m.unmount()
  })
})

describe('useLiveRun snapshot', () => {
  // The frontier is changed in place: without a snapshot every state in the history is the same object.
  interface Frontier {
    readonly open: number[]
  }
  const search = (snapshot: boolean): Algorithm<Frontier> => ({
    name: 'search',
    config: {},
    init: () => ({ open: [] }),
    step: (s, ctx) => {
      s.open.push(ctx.t)
      return s
    },
    ...(snapshot ? { snapshot: (s: Frontier): Frontier => ({ open: [...s.open] }) } : {}),
  })

  it('keeps each frame as it was when the algorithm gives a snapshot', async () => {
    const algorithm = search(true)
    const m = await mountHook(() => useLiveRun(algorithm, { maxSteps: 5, start: 'end' }))
    for (let n = 0; n < 50 && m.current.status === undefined; n++) await frames.frame()
    await frames.frame()
    const shown: number[][] = []
    for (let f = 0; f <= 5; f++) {
      await inAct(() => m.current.controller.seek(f))
      shown.push([...m.current.state.open])
    }
    expect(shown).toEqual([[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]])
    await m.unmount()
  })

  it('shows the latest state at every frame without one', async () => {
    const algorithm = search(false)
    const m = await mountHook(() => useLiveRun(algorithm, { maxSteps: 5, start: 'end' }))
    for (let n = 0; n < 50 && m.current.status === undefined; n++) await frames.frame()
    await frames.frame()
    await inAct(() => m.current.controller.seek(0))
    expect(m.current.state.open).toEqual([0, 1, 2, 3, 4])
    await m.unmount()
  })
})
