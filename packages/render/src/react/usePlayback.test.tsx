import type { Algorithm } from 'mdml-engine/trace'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { type FakeFrames, fakeFrames, inAct, mountHook } from './_harness.tsx'
import { type LiveRunOptions, useLiveRun } from './usePlayback.ts'

const counter: Algorithm<number> = { name: 'counter', config: {}, init: () => 0, step: (s) => s + 1 }

let frames: FakeFrames
beforeEach(() => {
  frames = fakeFrames()
})
afterEach(() => {
  frames.restore()
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
