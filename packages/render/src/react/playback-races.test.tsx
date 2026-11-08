import type { Algorithm } from 'omai-engine/trace'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { type FakeFrames, fakeFrames, inAct, mountHook, resetGlobals } from './_harness.tsx'
import {
  type LiveRunOptions,
  type PlaybackController,
  type PlaybackOptions,
  useLiveRun,
  usePlayback,
  useTicker,
} from './usePlayback.ts'

// The races between a controller call and a clock tick that was already queued: a frame callback that comes after
// `pause(); seek(n)` (or `reset()`, `step()`) was called but before the render that call causes. The harness runs the
// callbacks of a frame in order inside one `act`, so a ticker mounted ahead of the hook's own (`before`) runs first, makes
// the calls, and the hook's tick runs next against the render it had.

const counter: Algorithm<number> = { name: 'counter', config: {}, init: () => 0, step: (s) => s + 1 }

let frames: FakeFrames
let before: (() => void) | undefined
beforeEach(() => {
  frames = fakeFrames()
  before = undefined
})
afterEach(() => {
  frames.restore()
  resetGlobals()
})

/** A ticker that runs ahead of the hook's own in every frame, and calls what a test left in `before`, once. */
function useBefore(): void {
  useTicker(true, 60, () => {
    const call = before
    before = undefined
    call?.()
  })
}

/** Run a frame that is long enough for several steps to be due (6 at 60 fps). */
const tick = (): Promise<void> => frames.frame(100)

/** Warm a mounted player up: the first frame only starts the clock, the next ones move it. */
async function playFor(n: number): Promise<void> {
  await frames.frame()
  for (let i = 0; i < n; i++) await tick()
}

const BIG = Number.MAX_SAFE_INTEGER

describe('useLiveRun: the controller wins over a tick queued before it', () => {
  type Mount = Awaited<ReturnType<typeof mountLive>>
  const mountLive = (options: LiveRunOptions, getOptions?: () => LiveRunOptions) =>
    mountHook(() => {
      useBefore()
      return useLiveRun(counter, getOptions === undefined ? options : getOptions())
    })

  /** Mount a playing run and let it get a few steps in. */
  async function playing(options: Partial<LiveRunOptions> = {}): Promise<Mount> {
    const m = await mountLive({ maxSteps: 450, autoplay: true, fps: 60, ...options })
    await playFor(3)
    expect(m.current.t).toBeGreaterThan(0)
    expect(m.current.controller.playing).toBe(true)
    return m
  }

  it('pause then seek to the end (a late tick used to continue from the step before the seek)', async () => {
    const m = await mountLive({ maxSteps: 36, start: 'first', autoplay: true, fps: 60 })
    await playFor(1)
    // Computed in the background while it plays; once the whole run exists, the island parks at the end.
    for (let n = 0; n < 200 && m.current.status === undefined; n++) await frames.frame()
    expect(m.current.status).toBe('completed')
    await m.rerender()
    // Play from the start again, so the cursor is mid-run, then the island pauses and seeks to the end.
    await inAct(() => m.current.controller.seek(2))
    await inAct(() => m.current.controller.play())
    before = () => {
      m.current.controller.pause()
      m.current.controller.seek(BIG)
    }
    await tick()
    expect(m.current.controller.playing).toBe(false)
    expect(m.current.t).toBe(36)
    expect(m.current.controller.frame).toBe(36)
    await frames.frames(3)
    expect(m.current.t).toBe(36)
    await m.unmount()
  })

  it('a run remade while it plays, then parked at its end once ready (repro 1 of the report)', async () => {
    let seed = 0
    const m = await mountLive({ maxSteps: 36, start: 'first', autoplay: true, fps: 60 }, () => ({
      maxSteps: 36,
      start: 'first',
      autoplay: true,
      fps: 60,
      seed,
    }))
    await playFor(2)
    seed = 1
    await m.rerender()
    expect(m.current.madeWith.seed).toBe(1)
    for (let n = 0; n < 200 && m.current.status === undefined; n++) await frames.frame()
    expect(m.current.status).toBe('completed')
    await inAct(() => m.current.controller.seek(1))
    await inAct(() => m.current.controller.play())
    before = () => {
      m.current.controller.pause()
      m.current.controller.seek(BIG)
    }
    await tick()
    expect(m.current.t).toBe(36)
    expect(m.current.controller.playing).toBe(false)
    await frames.frames(3)
    expect(m.current.t).toBe(36)
    await m.unmount()
  })

  it('pause then seek to 0 after the page was left and returned to (repro 2 of the report)', async () => {
    let active = true
    const m = await mountLive({ maxSteps: 450, autoplay: true, fps: 60 }, () => ({
      maxSteps: 450,
      autoplay: true,
      fps: 60,
      active,
    }))
    await playFor(15)
    const at = m.current.t
    expect(at).toBeGreaterThan(30)
    active = false
    await m.rerender()
    active = true
    await m.rerender()
    await frames.frame()
    before = () => {
      m.current.controller.pause()
      m.current.controller.seek(0)
    }
    await tick()
    expect(m.current.t).toBe(0)
    expect(m.current.controller.playing).toBe(false)
    await frames.frames(3)
    expect(m.current.t).toBe(0)
    await m.unmount()
  })

  it('seek then pause', async () => {
    const m = await playing()
    before = () => {
      m.current.controller.seek(0)
      m.current.controller.pause()
    }
    await tick()
    expect(m.current.t).toBe(0)
    expect(m.current.controller.playing).toBe(false)
    await frames.frames(2)
    expect(m.current.t).toBe(0)
    await m.unmount()
  })

  it('pause alone stops a tick that was queued before it', async () => {
    const m = await playing()
    const at = m.current.t
    before = () => m.current.controller.pause()
    await tick()
    expect(m.current.t).toBe(at)
    expect(m.current.controller.playing).toBe(false)
    await m.unmount()
  })

  it('seek then play carries on from the step sought, not from the one before it', async () => {
    const m = await playing()
    expect(m.current.t).toBeGreaterThan(12)
    before = () => {
      m.current.controller.pause()
      m.current.controller.seek(5)
      m.current.controller.play()
    }
    await tick()
    // 5, plus the frames due in this tick (6) at most: never the old position plus a tick.
    expect(m.current.controller.playing).toBe(true)
    expect(m.current.t).toBeGreaterThanOrEqual(5)
    expect(m.current.t).toBeLessThanOrEqual(11)
    await m.unmount()
  })

  it('reset wins over a tick queued before it', async () => {
    const m = await playing()
    before = () => m.current.controller.reset()
    await tick()
    expect(m.current.t).toBe(0)
    expect(m.current.controller.playing).toBe(false)
    expect(m.current.controller.count).toBe(1)
    await frames.frames(2)
    expect(m.current.t).toBe(0)
    await m.unmount()
  })

  it('step back wins over a tick queued before it, and steps from the latest position', async () => {
    const m = await playing()
    before = () => {
      m.current.controller.seek(10)
      m.current.controller.step(-1)
    }
    await tick()
    expect(m.current.t).toBe(9)
    expect(m.current.controller.playing).toBe(false)
    await m.unmount()
  })

  it('a controller kept from an earlier render acts on the latest state', async () => {
    const m = await playing()
    const old: PlaybackController = m.current.controller
    await tick()
    await tick()
    await inAct(() => {
      old.pause()
      old.seek(3)
    })
    expect(m.current.t).toBe(3)
    expect(m.current.controller.playing).toBe(false)
    await inAct(() => old.toggle())
    expect(m.current.controller.playing).toBe(true)
    await inAct(() => old.toggle())
    expect(m.current.controller.playing).toBe(false)
    await m.unmount()
  })

  it('seek during priming lands on the states that exist, stays there, and a tick does not move it', async () => {
    const m = await mountLive({ maxSteps: 400, start: 'first' })
    await frames.frames(2)
    const partway = m.current.controller.count
    expect(partway).toBeGreaterThan(1)
    expect(m.current.status).toBeUndefined()
    await inAct(() => {
      m.current.controller.pause()
      m.current.controller.seek(BIG)
    })
    expect(m.current.controller.frame).toBeLessThan(m.current.controller.count)
    const where = m.current.t
    expect(where).toBeGreaterThan(0)
    for (let n = 0; n < 400 && m.current.status === undefined; n++) await frames.frame()
    expect(m.current.status).toBe('completed')
    expect(m.current.t).toBe(where)
    expect(m.current.controller.playing).toBe(false)
    // And a seek in the middle of the priming is not overridden by the slice of the same frame.
    await inAct(() => m.current.controller.seek(7))
    await frames.frames(2)
    expect(m.current.t).toBe(7)
    await m.unmount()
  })

  it('seek after a remade run is ready, with a tick queued before it', async () => {
    let seed = 0
    const m = await mountLive({ maxSteps: 40, start: 'first' }, () => ({ maxSteps: 40, start: 'first', seed }))
    for (let n = 0; n < 100 && m.current.status === undefined; n++) await frames.frame()
    await inAct(() => m.current.controller.play())
    await playFor(1)
    seed = 7
    await m.rerender()
    expect(m.current.t).toBe(0)
    for (let n = 0; n < 100 && m.current.status === undefined; n++) await frames.frame()
    expect(m.current.status).toBe('completed')
    before = () => {
      m.current.controller.pause()
      m.current.controller.seek(BIG)
    }
    await tick()
    expect(m.current.t).toBe(40)
    expect(m.current.controller.playing).toBe(false)
    await m.unmount()
  })

  it('seek right after the run is remade (before the effect that restarts the cursor) is kept', async () => {
    let seed = 0
    const m = await mountLive({ maxSteps: 40, start: 'first' }, () => ({ maxSteps: 40, start: 'first', seed }))
    for (let n = 0; n < 100 && m.current.status === undefined; n++) await frames.frame()
    await inAct(() => m.current.controller.seek(20))
    expect(m.current.t).toBe(20)
    seed = 3
    await m.rerender()
    // A fresh run starts from its first state.
    expect(m.current.t).toBe(0)
    await m.unmount()
  })

  it('eager (start end): a seek made before the run is done keeps the cursor off the end', async () => {
    const m = await mountLive({ maxSteps: 100, start: 'end' })
    await frames.frames(1)
    await inAct(() => m.current.controller.seek(0))
    for (let n = 0; n < 200 && m.current.status === undefined; n++) await frames.frame()
    await frames.frame()
    expect(m.current.t).toBe(0)
    await m.unmount()
  })

  it('a tick queued before a change of the rate still moves from the latest position', async () => {
    const m = await playing()
    before = () => {
      m.current.controller.seek(2)
      m.current.controller.setFps(30)
    }
    await tick()
    // The seek is kept; the new rate's clock starts afresh, and the old tick (60 fps, 6 frames) did move from step 2.
    expect(m.current.t).toBeGreaterThanOrEqual(2)
    expect(m.current.t).toBeLessThanOrEqual(8)
    expect(m.current.controller.fps).toBe(30)
    await m.unmount()
  })

  it('a tick queued before the run went inactive does nothing', async () => {
    let active = true
    const m = await mountLive({ maxSteps: 450, autoplay: true, fps: 60 }, () => ({
      maxSteps: 450,
      autoplay: true,
      fps: 60,
      active,
    }))
    await playFor(3)
    const at = m.current.t
    // The page is hidden between the render that went inactive and the cleanup of its clock: simulated by the flag
    // the tick reads, which a render sets.
    before = () => {
      active = false
    }
    await tick()
    await m.rerender()
    await frames.frames(2)
    expect(m.current.t).toBeGreaterThanOrEqual(at)
    const frozen = m.current.t
    await frames.frames(3)
    expect(m.current.t).toBe(frozen)
    await m.unmount()
  })
})

describe('useLiveRun: madeWith', () => {
  it('is what the run on the tape was made with, and switches when the run is remade', async () => {
    let seed = 1
    let steps = 20
    const m = await mountHook(() => useLiveRun(counter, { maxSteps: steps, seed }))
    expect(m.current.madeWith).toEqual({
      algorithm: counter,
      config: {},
      seed: 1,
      maxSteps: 20,
      capacity: 2000,
    })
    seed = 2
    await m.rerender()
    expect(m.current.madeWith.seed).toBe(2)
    steps = 50
    await m.rerender()
    expect(m.current.madeWith.maxSteps).toBe(50)
    await m.unmount()
  })

  it('keeps its values when something that does not remake the run changes, and when the run is reset', async () => {
    let perFrame = 1
    const m = await mountHook(() => useLiveRun(counter, { maxSteps: 20, seed: 4, stepsPerFrame: perFrame }))
    const first = m.current.madeWith
    perFrame = 3
    await m.rerender()
    expect(m.current.madeWith).toBe(first)
    await inAct(() => m.current.controller.reset())
    expect(m.current.madeWith).toEqual(first)
    await m.unmount()
  })

  it('reads the algorithm config of the run, not of the options', async () => {
    const make = (k: number): Algorithm<number> => ({ ...counter, name: 'k', config: { k } })
    let algorithm = make(1)
    const m = await mountHook(() => useLiveRun(algorithm, { maxSteps: 20 }))
    expect(m.current.madeWith.config).toEqual({ k: 1 })
    algorithm = make(2)
    await m.rerender()
    expect(m.current.madeWith.config).toEqual({ k: 2 })
    await m.unmount()
  })
})

describe('usePlayback: the controller wins over a tick queued before it', () => {
  const mountPlayback = (count: number, options: () => PlaybackOptions) =>
    mountHook(() => {
      useBefore()
      return usePlayback(count, options())
    })
  const playingFor = async (options: () => PlaybackOptions = () => ({ fps: 60, autoplay: true })) => {
    const m = await mountPlayback(500, options)
    await playFor(3)
    expect(m.current.frame).toBeGreaterThan(6)
    expect(m.current.playing).toBe(true)
    return m
  }

  it('pause then seek to the end', async () => {
    const m = await playingFor()
    before = () => {
      m.current.pause()
      m.current.seek(BIG)
    }
    await tick()
    expect(m.current.frame).toBe(499)
    expect(m.current.playing).toBe(false)
    await frames.frames(3)
    expect(m.current.frame).toBe(499)
    await m.unmount()
  })

  it('pause then seek to 0, after leaving and returning (the player was inactive meanwhile)', async () => {
    let active = true
    const m = await playingFor(() => ({ fps: 60, autoplay: true, active }))
    active = false
    await m.rerender()
    active = true
    await m.rerender()
    await frames.frame()
    before = () => {
      m.current.pause()
      m.current.seek(0)
    }
    await tick()
    expect(m.current.frame).toBe(0)
    expect(m.current.playing).toBe(false)
    await frames.frames(3)
    expect(m.current.frame).toBe(0)
    await m.unmount()
  })

  it('seek then pause, and pause alone', async () => {
    const m = await playingFor()
    before = () => {
      m.current.seek(0)
      m.current.pause()
    }
    await tick()
    expect(m.current.frame).toBe(0)
    expect(m.current.playing).toBe(false)
    await m.unmount()
    const n = await playingFor()
    const at = n.current.frame
    before = () => n.current.pause()
    await tick()
    expect(n.current.frame).toBe(at)
    await n.unmount()
  })

  it('seek then play carries on from the step sought', async () => {
    const m = await playingFor()
    before = () => {
      m.current.pause()
      m.current.seek(5)
      m.current.play()
    }
    await tick()
    expect(m.current.playing).toBe(true)
    expect(m.current.frame).toBeGreaterThanOrEqual(5)
    expect(m.current.frame).toBeLessThanOrEqual(11)
    await m.unmount()
  })

  it('reset and step win over a tick queued before them', async () => {
    const m = await playingFor()
    before = () => m.current.reset()
    await tick()
    expect(m.current.frame).toBe(0)
    expect(m.current.playing).toBe(false)
    await inAct(() => m.current.play())
    await tick()
    before = () => {
      m.current.seek(10)
      m.current.step(-1)
    }
    await tick()
    expect(m.current.frame).toBe(9)
    expect(m.current.playing).toBe(false)
    await m.unmount()
  })

  it('stops by itself at the last frame, and a seek after that stays', async () => {
    const m = await mountPlayback(20, () => ({ fps: 60, autoplay: true }))
    await playFor(6)
    expect(m.current.frame).toBe(19)
    expect(m.current.playing).toBe(false)
    await inAct(() => m.current.seek(4))
    await frames.frames(3)
    expect(m.current.frame).toBe(4)
    await m.unmount()
  })

  it('a controller kept from an earlier render acts on the latest state', async () => {
    const m = await playingFor()
    const old = m.current
    await tick()
    await inAct(() => {
      old.pause()
      old.seek(3)
    })
    expect(m.current.frame).toBe(3)
    expect(m.current.playing).toBe(false)
    await inAct(() => old.toggle())
    expect(m.current.playing).toBe(true)
    await m.unmount()
  })

  it('a seek is clamped to the frames there are, also when the count changes', async () => {
    let count = 10
    const m = await mountHook(() => usePlayback(count))
    await inAct(() => m.current.seek(BIG))
    expect(m.current.frame).toBe(9)
    count = 4
    await m.rerender()
    expect(m.current.frame).toBe(3)
    await m.unmount()
  })
})

describe('usePlayback: active', () => {
  it('turns the clock off, keeps the frame and playing, and carries on when active again', async () => {
    let active = true
    const m = await mountHook(() => usePlayback(500, { fps: 60, autoplay: true, active }))
    await playFor(3)
    const at = m.current.frame
    expect(at).toBeGreaterThan(0)
    active = false
    await m.rerender()
    expect(frames.pending).toBe(0)
    await frames.frames(3)
    expect(m.current.frame).toBe(at)
    expect(m.current.playing).toBe(true)
    active = true
    await m.rerender()
    await playFor(2)
    expect(m.current.frame).toBeGreaterThan(at)
    await m.unmount()
  })

  it('does not resume a player the caller paused while it was inactive', async () => {
    let active = true
    const m = await mountHook(() => usePlayback(500, { fps: 60, autoplay: true, active }))
    await playFor(2)
    active = false
    await m.rerender()
    await inAct(() => m.current.pause())
    const at = m.current.frame
    active = true
    await m.rerender()
    await playFor(3)
    expect(m.current.frame).toBe(at)
    expect(m.current.playing).toBe(false)
    await m.unmount()
  })

  it('is on by default, and a step or a seek by hand still moves while inactive', async () => {
    const m = await mountHook(() => usePlayback(50, { fps: 60, autoplay: true }))
    await playFor(2)
    expect(m.current.frame).toBeGreaterThan(0)
    await m.unmount()
    const off = await mountHook(() => usePlayback(50, { active: false }))
    await inAct(() => off.current.step(1))
    expect(off.current.frame).toBe(1)
    await inAct(() => off.current.seek(9))
    expect(off.current.frame).toBe(9)
    await off.unmount()
  })
})
