import { dict, int } from 'omai-engine/space'
import { type Config, defineMethod, Log, type Method, scalar, stateFromJson, stateToJson } from 'omai-engine/trace'
import type { ReactElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { type FakeFrames, fakeFrames, inAct, mountElement, mountHook, resetGlobals } from '../_harness.tsx'
import { waitForSettled } from '../settled.ts'
import { SessionPlayer } from './SessionPlayer.tsx'
import { type LiveSession, type SessionHookOptions, STILL_HANDLE, useSession } from './useSession.ts'

// The player's buttons are not under test.
vi.mock('../Player.tsx', () => ({ Player: (): null => null }))

interface S {
  readonly k: number
}
// A counting method of `len` steps; with `failAt` its step throws on that step.
const count: Method<S, Config> = defineMethod<S, Config>({
  name: 'count',
  config: dict({ len: int(1, 1000, { default: 20 }), failAt: int(0, 1000, { default: 0 }) }),
  make: (c) => ({
    name: 'count',
    config: c,
    init: () => ({ k: 0 }),
    step: (s, ctx) => {
      if (c['failAt'] !== 0 && s.k + 1 === c['failAt']) throw new Error('boom')
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

let frames: FakeFrames
beforeEach(() => {
  frames = fakeFrames()
})
afterEach(() => {
  frames.restore()
  resetGlobals()
})

const hookOf = (hook: SessionHookOptions, config: Partial<Config> = {}, strict = false) =>
  mountHook(() => useSession(count, { steps: 200, config }, [JSON.stringify(config)], hook), { strict })

// What a promise holds right now, without waiting for it: undefined while pending.
async function peek<T>(p: Promise<T>): Promise<T | undefined> {
  const pending = Symbol('pending')
  const got = await Promise.race([p, Promise.resolve(pending)])
  return got === pending ? undefined : (got as T)
}
async function until(done: () => boolean, limit = 200): Promise<void> {
  for (let n = 0; n <= limit && !done(); n++) await frames.frame()
  expect(done()).toBe(true)
}

describe('live.settled', () => {
  it('primed: pending while the tape fills, then resolves ok', async () => {
    const m = await hookOf({ start: 'first' })
    const first = m.current.settled
    expect(await peek(first)).toBeUndefined()
    await until(() => m.current.done)
    expect(await first).toEqual({ ok: true })
    // The same promise on a later render: one per run.
    expect(m.current.settled).toBe(first)
  })

  it('eager: resolves ok once the whole run is computed in the background', async () => {
    const m = await hookOf({ start: 'end' })
    expect(await peek(m.current.settled)).toBeUndefined()
    await until(() => m.current.done)
    expect(await m.current.settled).toEqual({ ok: true })
  })

  it('live: pending until play computes the whole tape, and a playing run with an incomplete tape is still preparing', async () => {
    const m = await hookOf({ start: 'play' })
    await frames.frames(5)
    expect(m.current.head).toBe(0)
    expect(await peek(m.current.settled)).toBeUndefined()
    await inAct(() => m.current.play())
    expect(m.current.status).toBe('running')
    expect(m.current.done).toBe(false)
    expect(await peek(m.current.settled)).toBeUndefined()
    await until(() => m.current.done)
    expect(await m.current.settled).toEqual({ ok: true })
  })

  it('is already resolved on a handle made by hand over a finished tape, and stays resolved once the tape is whole', async () => {
    const m = await hookOf({ start: 'end' })
    await until(() => m.current.done)
    const whole = m.current.settled
    // A handle made by hand over a finished tape carries a resolved promise.
    expect(await peek(STILL_HANDLE.settled)).toEqual({ ok: true })
    expect(await peek(whole)).toEqual({ ok: true })
  })

  it('failed: a step that throws ends the run as failed, and settled resolves with the error', async () => {
    const m = await hookOf({ start: 'first' }, { failAt: 3 })
    await until(() => m.current.error !== undefined)
    expect(m.current.error).toBe('boom')
    expect(m.current.computing).toBe(false)
    expect(await m.current.settled).toEqual({ ok: false, error: 'boom' })
  })

  it('dispose: an incomplete run settles as disposed, and using the session again gives a fresh promise', async () => {
    const m = await hookOf({ start: 'first' })
    await frames.frames(1)
    const first = m.current.settled
    await inAct(() => m.current.dispose())
    expect(await first).toEqual({ ok: false, error: 'disposed' })
    expect(m.current.disposed).toBe(true)
    // Still the same settled one while it stays disposed; after use, a new run's promise.
    await inAct(() => m.current.play())
    const again = m.current.settled
    expect(again).not.toBe(first)
    expect(await peek(again)).toBeUndefined()
    await until(() => m.current.done)
    expect(await again).toEqual({ ok: true })
  })

  it('a remake gives a fresh promise, and the old one settles as disposed if its run had not ended', async () => {
    const m = await hookOf({ start: 'first' })
    await frames.frames(1)
    const old = m.current.settled
    await inAct(() => void m.current.apply())
    expect(m.current.settled).not.toBe(old)
    expect(await old).toEqual({ ok: false, error: 'disposed' })
    await until(() => m.current.done)
    expect(await m.current.settled).toEqual({ ok: true })
  })

  it('a run that ended keeps its result through a remake', async () => {
    const m = await hookOf({ start: 'end' })
    await until(() => m.current.done)
    const old = m.current.settled
    await inAct(() => void m.current.apply())
    expect(await old).toEqual({ ok: true })
    expect(m.current.settled).not.toBe(old)
  })

  it('unmounting settles a pending promise as disposed, but a strict-mode double effect does not', async () => {
    const strict = await hookOf({ start: 'first' }, {}, true)
    const kept = strict.current.settled
    await frames.frames(1)
    await Promise.resolve()
    expect(await peek(kept)).toBeUndefined()
    await strict.unmount()
    await Promise.resolve()
    expect(await kept).toEqual({ ok: false, error: 'disposed' })
  })
})

function Page(props: {
  readonly start: 'first' | 'end' | 'play'
  readonly failAt?: number
  readonly active?: boolean
}): ReactElement {
  const config = props.failAt === undefined ? {} : { failAt: props.failAt }
  const live: LiveSession<S, Config> = useSession(count, { steps: 200, config }, [props.failAt], {
    start: props.start,
    ...(props.active === undefined ? {} : { active: props.active }),
  })
  page.live = live
  return <SessionPlayer live={live} at={null} onScrub={() => undefined} />
}
const page: { live?: LiveSession<S, Config> } = {}

describe('idle sessions', () => {
  it('a live session is idle before Play, preparing while it computes, ready when whole', async () => {
    const m = await hookOf({ start: 'play' })
    expect(m.current.idle).toBe(true)
    await frames.frames(3)
    expect(m.current.idle).toBe(true)
    expect(await peek(m.current.settled)).toBeUndefined()
    await inAct(() => m.current.play())
    expect(m.current.idle).toBe(false)
    // Pausing the player does not stop the producer: the tape keeps filling, so it is still working towards the end.
    await inAct(() => m.current.pause())
    expect(m.current.idle).toBe(false)
    await until(() => m.current.done)
    expect(await m.current.settled).toEqual({ ok: true })
  })

  it('a session that is not active, or whose computing was halted, is idle; a primed one is not, from its first render', async () => {
    const primed = await hookOf({ start: 'first' })
    expect(primed.current.idle).toBe(false)
    await inAct(() => primed.current.halt())
    expect(primed.current.idle).toBe(true)
    await inAct(() => primed.current.compute())
    expect(primed.current.idle).toBe(false)

    const off = await hookOf({ start: 'first', active: false })
    expect(off.current.idle).toBe(true)
    await inAct(() => off.current.play())
    expect(off.current.idle).toBe(true)
    await frames.frames(3)
    expect(off.current.head).toBe(0)
  })
})

describe('data-omai-state of the session figures', () => {
  const state = (m: Awaited<ReturnType<typeof mountElement>>): unknown =>
    m.elements().find((e) => e.props['data-omai-state'] !== undefined)?.props['data-omai-state']

  it('is preparing while the tape fills (also while it plays), ready when whole, failed when the run failed', async () => {
    const m = await mountElement(<Page start="first" />)
    expect(state(m)).toBe('preparing')
    await until(() => page.live?.done === true)
    expect(state(m)).toBe('ready')
    await m.unmount()

    const live = await mountElement(<Page start="play" />)
    await inAct(() => page.live?.play())
    expect(page.live?.status).toBe('running')
    expect(state(live)).toBe('preparing')
    await until(() => page.live?.done === true)
    expect(state(live)).toBe('ready')
    await live.unmount()

    const failing = await mountElement(<Page start="first" failAt={3} />)
    await until(() => page.live?.error !== undefined)
    expect(state(failing)).toBe('failed')
    await failing.unmount()
  })

  it('is idle for a live session before Play and for an inactive one, preparing once played', async () => {
    const m = await mountElement(<Page start="play" />)
    expect(state(m)).toBe('idle')
    await frames.frames(3)
    expect(state(m)).toBe('idle')
    await inAct(() => page.live?.play())
    expect(state(m)).toBe('preparing')
    await m.unmount()

    const off = await mountElement(<Page start="first" active={false} />)
    expect(state(off)).toBe('idle')
    await frames.frames(3)
    expect(state(off)).toBe('idle')
    await off.unmount()
  })

  it('lets waitForSettled resolve on an idle session figure and report it', async () => {
    const m = await mountElement(<Page start="play" />)
    const figures = m
      .elements()
      .filter((e) => e.props['data-omai-state'] !== undefined)
      .map((e) => ({ getAttribute: (n: string) => (n === 'data-omai-state' ? String(e.props[n]) : null) }))
    const result = await waitForSettled({ querySelectorAll: () => figures } as never)
    expect(result).toEqual({ ok: true, ready: 1, idle: figures })
    await m.unmount()
  })
})
