import { dict, int } from 'omai-engine/space'
import { type Config, defineMethod, Log, type Method, scalar, stateFromJson, stateToJson } from 'omai-engine/trace'
import { type ComponentProps, type ReactElement, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { type FakeFrames, fakeFrames, inAct, mountElement, resetGlobals } from '../_harness.tsx'
import type { StartPolicy } from '../start.ts'
import { type PlaybackController, useTicker } from '../usePlayback.ts'
import { SessionPlayer } from './SessionPlayer.tsx'
import { type LiveSession, useSession } from './useSession.ts'

// The player's buttons are not under test: stand in for it and keep the controller it is given.
const seen: { controller?: PlaybackController } = {}
vi.mock('../Player.tsx', () => ({
  Player: (props: { controller: PlaybackController }): null => {
    seen.controller = props.controller
    return null
  },
}))

interface S {
  readonly k: number
}
const count: Method<S, Config> = defineMethod<S, Config>({
  name: 'count',
  config: dict({ len: int(1, 1000, { default: 200 }) }),
  make: (c) => ({
    name: 'count',
    config: c,
    init: () => ({ k: 0 }),
    step: (s, ctx) => {
      ctx.log('n', s.k + 1)
      return { k: s.k + 1 }
    },
    done: (s) => (s.k >= (c['len'] as number) ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as S,
  }),
  contract: { n: scalar() },
  defaults: { n: Log() },
})

const page: {
  at: number | null
  active: boolean
  poke?: () => void
  setAt?: (at: number | null) => void
  /** Mount the clock that runs ahead of the player's own (it keeps a frame pending, which other tests count). */
  ahead?: boolean
  /** Run once, by a ticker mounted ahead of the player's own, so before its tick in the same frame. */
  before?: () => void
  live?: LiveSession<S, Config>
  playhead: { current: number }
} = {
  at: null,
  active: true,
  playhead: { current: -1 },
}

/** A clock ahead of the player's own in every frame, which calls what a test left in `page.before`, once. */
function Before(): null {
  useTicker(true, 60, () => {
    const call = page.before
    delete page.before
    call?.()
  })
  return null
}

function Page(props: {
  readonly start: StartPolicy
  readonly steps?: number
  readonly player?: Partial<ComponentProps<typeof SessionPlayer>>
}): ReactElement {
  const [at, setAt] = useState<number | null>(null)
  const [, poke] = useState(0)
  page.poke = () => poke((n) => n + 1)
  page.setAt = setAt
  const live = useSession(count, { steps: props.steps ?? 200 }, [], {
    start: props.start,
    active: page.active,
  })
  page.at = at
  page.live = live
  return (
    <>
      {page.ahead === true ? <Before /> : null}
      <SessionPlayer {...props.player} live={live} at={at} onScrub={setAt} playhead={page.playhead} />
    </>
  )
}

const controller = (): PlaybackController => seen.controller as PlaybackController
const live = (): LiveSession<S, Config> => page.live as LiveSession<S, Config>

let frames: FakeFrames
beforeEach(() => {
  frames = fakeFrames()
  page.at = null
  page.active = true
  page.ahead = false
  delete page.before
  page.playhead.current = -1
  delete page.before
  delete seen.controller
})
afterEach(() => {
  frames.restore()
  resetGlobals()
})

async function untilDone(limit = 400): Promise<void> {
  for (let n = 0; n <= limit && !live().done; n++) await frames.frame()
  expect(live().done).toBe(true)
}

describe('the player over the start policies', () => {
  it('primed: parked at step 0 and paused while the whole run fills, then ready and still parked', async () => {
    await mountElement(<Page start="first" />)
    expect(page.at).toBe(0)
    expect(controller().playing).toBe(false)
    expect(live().status).toBe('preparing')
    for (let n = 0; n < 400 && !live().done; n++) {
      await frames.frame()
      expect(page.at).toBe(0)
      expect(controller().playing).toBe(false)
      expect(controller().frame).toBe(0)
    }
    expect(live().done).toBe(true)
    expect(live().status).toBe('ready')
    expect(page.at).toBe(0)
    expect(controller().count).toBe(201)
    await frames.frames(5)
    expect(page.at).toBe(0)
    // Play moves it on from step 0.
    await inAct(() => controller().play())
    await frames.frames(5)
    expect(page.at).toBeGreaterThan(0)
    expect(live().status).toBe('running')
  })

  it('primed: play pressed before the tape is ready plays at the head, behind the computing, and goes on', async () => {
    await mountElement(<Page start="first" />)
    await frames.frames(2)
    expect(live().done).toBe(false)
    await inAct(() => controller().play())
    let last = -1
    for (let n = 0; n < 400 && !live().done; n++) {
      await frames.frame()
      const at = page.at ?? live().head
      // Never past what is computed, and never backwards.
      expect(at).toBeLessThanOrEqual(live().head)
      expect(at).toBeGreaterThanOrEqual(last)
      last = at
    }
    expect(last).toBeGreaterThan(0)
    expect(live().done).toBe(true)
    expect(live().status).toBe('running')
  })

  it('eager: waits at step 0 while it computes, then jumps to the end', async () => {
    await mountElement(<Page start="end" />)
    expect(page.at).toBe(0)
    expect(controller().playing).toBe(false)
    for (let n = 0; n < 400 && !live().done; n++) {
      await frames.frame()
      if (!live().done) expect(page.at).toBe(0)
    }
    await frames.frame()
    expect(live().done).toBe(true)
    expect(page.at).toBeNull()
    expect(controller().frame).toBe(200)
  })

  it('live: nothing computed and the cursor at the start until play', async () => {
    await mountElement(<Page start="play" />)
    expect(page.at).toBeNull()
    await frames.frames(5)
    expect(live().head).toBe(0)
    expect(controller().playing).toBe(false)
    await inAct(() => controller().play())
    await frames.frames(5)
    expect(controller().playing).toBe(true)
    expect(live().head).toBeGreaterThan(0)
  })

  it('the host can start and stop the clock through the handle', async () => {
    await mountElement(<Page start="first" />)
    await untilDone()
    expect(controller().playing).toBe(false)
    await inAct(() => live().play())
    await frames.frames(5)
    expect(controller().playing).toBe(true)
    expect(page.at).toBeGreaterThan(0)
    await inAct(() => live().pause())
    await frames.frame()
    expect(controller().playing).toBe(false)
    const stopped = page.at
    await frames.frames(5)
    expect(page.at).toBe(stopped)
  })
})

describe('the playhead', () => {
  it('keeps its time when frames are slow: it skips steps rather than slowing down', async () => {
    await mountElement(<Page start="first" />)
    await untilDone()
    await inAct(() => controller().setFps(60))
    await inAct(() => controller().play())
    // Frames of 100 ms are 6 steps each at 60 steps a second; a clock that capped them at 4 would fall behind.
    await frames.frame()
    const from = page.at as number
    for (let n = 0; n < 8; n++) await frames.frame(100)
    const slow = (page.at as number) - from
    expect(slow).toBeGreaterThanOrEqual(40)
    expect(page.playhead.current).toBe(page.at)
  })

  it('tells the page once per frame, whatever the rate', async () => {
    await mountElement(<Page start="first" />)
    await untilDone()
    await inAct(() => controller().setFps(1000))
    await inAct(() => controller().play())
    await frames.frame()
    let told = 0
    const before = page.at
    for (let n = 0; n < 4; n++) {
      await frames.frame()
      if (page.at !== before) told++
    }
    expect(told).toBeGreaterThan(0)
    expect(page.at).toBeLessThanOrEqual(200)
  })
})

describe('the rate, the eager toggle and the statistics line', () => {
  it('starts at the stored preference rate, 60 by default, and keeps the reader’s pick', async () => {
    await mountElement(<Page start="first" />)
    expect(controller().fps).toBe(60)
    await inAct(() => controller().setFps(10))
    expect(controller().fps).toBe(10)
  })

  it('takes the rate from `fps` when given, which rules, and tells `onFpsChange` of the reader’s pick', async () => {
    const picked: number[] = []
    const m = await mountElement(<Page start="first" player={{ fps: 24, onFpsChange: (f) => picked.push(f) }} />)
    expect(controller().fps).toBe(24)
    await inAct(() => controller().setFps(5))
    expect(picked).toEqual([5])
    // The host owns the rate: it stays 24 until the host passes another.
    expect(controller().fps).toBe(24)
    await m.rerender()
    expect(controller().fps).toBe(24)
  })

  it('the player ticks at the rate it is given', async () => {
    await mountElement(<Page start="first" player={{ fps: 2 }} />)
    await untilDone()
    await inAct(() => controller().play())
    await frames.frame()
    const from = page.at as number
    // Two steps a second: a second of 100 ms frames is two steps, nowhere near the 60 of the default.
    for (let n = 0; n < 10; n++) await frames.frame(100)
    expect((page.at as number) - from).toBeLessThanOrEqual(3)
  })

  it('offers the eager toggle unless `eagerToggle` is false or the start is given', async () => {
    await mountElement(<Page start="first" />)
    expect(controller().setEager).toBeDefined()
    await mountElement(<Page start="first" player={{ eagerToggle: false }} />)
    expect(controller().setEager).toBeUndefined()
    await mountElement(<Page start="first" player={{ eagerToggle: true, start: 'first' }} />)
    expect(controller().setEager).toBeUndefined()
    await mountElement(<Page start="first" player={{ eagerToggle: true }} />)
    expect(controller().setEager).toBeDefined()
  })

  it('shows the statistics line, and hides it with `statistics` false', async () => {
    const shown = await mountElement(<Page start="first" />)
    await untilDone()
    expect(shown.text()).toContain('last slice')
    const hidden = await mountElement(<Page start="first" player={{ statistics: false }} />)
    await untilDone()
    expect(hidden.text()).not.toContain('last slice')
    expect(hidden.text()).not.toContain('step')
  })

  it('does not compute or tick while the session is inactive, and goes on when it is active again', async () => {
    page.active = false
    await mountElement(<Page start="first" />)
    await inAct(() => controller().play())
    await frames.frames(5)
    expect(live().head).toBe(0)
    expect(live().computing).toBe(false)
    expect(frames.pending).toBe(0)
    page.active = true
    await inAct(() => page.poke?.())
    await frames.frames(5)
    expect(live().head).toBeGreaterThan(0)
    expect(live().computing).toBe(true)
  })
})

describe('the player: a pause or a seek wins over a tick queued before it', () => {
  async function playingAt(): Promise<number> {
    page.ahead = true
    await mountElement(<Page start="first" />)
    await untilDone()
    await inAct(() => controller().setFps(60))
    await inAct(() => controller().play())
    await frames.frame()
    for (let n = 0; n < 4; n++) await frames.frame(100)
    expect(controller().playing).toBe(true)
    const at = page.at as number
    expect(at).toBeGreaterThan(10)
    return at
  }

  it("the player's own pause then seek to 0", async () => {
    await playingAt()
    page.before = () => {
      controller().pause()
      controller().seek(0)
    }
    await frames.frame(100)
    expect(page.at).toBe(0)
    expect(controller().playing).toBe(false)
    await frames.frames(3)
    expect(page.at).toBe(0)
    expect(page.playhead.current).toBe(0)
  })

  it('the host pauses the handle and scrubs the page to 0 (parking a slide)', async () => {
    await playingAt()
    page.before = () => {
      live().pause()
      page.setAt?.(0)
    }
    await frames.frame(100)
    expect(page.at).toBe(0)
    expect(controller().playing).toBe(false)
    await frames.frames(3)
    expect(page.at).toBe(0)
  })

  it('the player step and reset win too, and a seek then play carries on from the step sought', async () => {
    await playingAt()
    page.before = () => {
      controller().reset()
    }
    await frames.frame(100)
    expect(page.at).toBe(0)
    expect(controller().playing).toBe(false)
    await inAct(() => controller().play())
    await frames.frame(100)
    page.before = () => {
      controller().pause()
      controller().seek(10)
      controller().step(1)
    }
    await frames.frame(100)
    expect(page.at).toBe(11)
    expect(controller().playing).toBe(false)
    await frames.frames(2)
    expect(page.at).toBe(11)
  })

  it('a seek while priming is not undone by a tick', async () => {
    page.ahead = true
    await mountElement(<Page start="first" />)
    await frames.frames(2)
    await inAct(() => controller().play())
    await frames.frame(100)
    page.before = () => {
      controller().pause()
      controller().seek(1)
    }
    await frames.frame(100)
    expect(controller().playing).toBe(false)
    expect(page.at).toBe(1)
    await frames.frames(3)
    expect(page.at).toBe(1)
  })
})
