import { dict, int } from 'mdml-engine/space'
import { type Config, defineMethod, Log, type Method, scalar, stateFromJson, stateToJson } from 'mdml-engine/trace'
import { type ReactElement, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { type FakeFrames, fakeFrames, inAct, mountElement, resetGlobals } from '../_harness.tsx'
import type { StartPolicy } from '../start.ts'
import type { PlaybackController } from '../usePlayback.ts'
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

const page: { at: number | null; live?: LiveSession<S, Config>; playhead: { current: number } } = {
  at: null,
  playhead: { current: -1 },
}

function Page(props: { readonly start: StartPolicy; readonly steps?: number }): ReactElement {
  const [at, setAt] = useState<number | null>(null)
  const live = useSession(count, { steps: props.steps ?? 200 }, [], { start: props.start })
  page.at = at
  page.live = live
  return <SessionPlayer live={live} at={at} onScrub={setAt} playhead={page.playhead} />
}

const controller = (): PlaybackController => seen.controller as PlaybackController
const live = (): LiveSession<S, Config> => page.live as LiveSession<S, Config>

let frames: FakeFrames
beforeEach(() => {
  frames = fakeFrames()
  page.at = null
  page.playhead.current = -1
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
