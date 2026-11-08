import { dict, int } from 'omai-engine/space'
import { type Config, defineMethod, Log, type Method, scalar, stateFromJson, stateToJson } from 'omai-engine/trace'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fakeWire, tickingClock } from '../../producer/_fake.ts'
import { workerProducer } from '../../producer/client.ts'
import { inlineProducer } from '../../producer/inline.ts'
import type { Producer, ProducerFactory } from '../../producer/producer.ts'
import { methodRegistry } from '../../producer/registry.ts'
import { serveProducer } from '../../producer/serve.ts'
import { type FakeFrames, fakeFrames, inAct, mountHook, resetGlobals } from '../_harness.tsx'
import { type LiveSession, type SessionHookOptions, useSession } from './useSession.ts'

interface S {
  readonly k: number
}

// A counting method of 200 steps: one scalar a step.
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

let frames: FakeFrames
beforeEach(() => {
  frames = fakeFrames()
})
afterEach(() => {
  frames.restore()
  resetGlobals()
})

const mountSession = (hook: SessionHookOptions, options: { mode?: 'eager' | 'live'; strict?: boolean } = {}) =>
  mountHook(
    () => useSession(count, { steps: 200, ...(options.mode === undefined ? {} : { mode: options.mode }) }, [], hook),
    options.strict === undefined ? {} : { strict: options.strict },
  )

/** Run frames until the session is whole; returns how many it took (and fails if it never is). */
async function untilDone(m: { readonly current: LiveSession<S, Config> }, limit = 400): Promise<number> {
  for (let n = 0; n <= limit; n++) {
    if (m.current.done) return n
    await frames.frame()
  }
  throw new Error('the session never finished')
}

describe('start: first (primed)', () => {
  it('computes the whole run in the background, in time slices, and plays nothing', async () => {
    const ready = vi.fn()
    const m = await mountSession({ start: 'first', onReady: ready })
    const live = (): LiveSession<S, Config> => m.current
    // Mounting computed nothing on the spot: step 0, preparing, the producer going.
    expect(live().start).toBe('first')
    expect(live().head).toBe(0)
    expect(live().status).toBe('preparing')
    expect(live().progress).toBe(0)
    expect(live().playing).toBe(false)
    expect(live().computing).toBe(true)
    // It fills over several frames, never in one: progress rises through the middle.
    const seen: number[] = []
    let n = 0
    while (!live().done && n++ < 400) {
      await frames.frame()
      seen.push(live().progress)
      expect(live().playing).toBe(false)
    }
    expect(n).toBeGreaterThan(3)
    expect(seen.some((p) => p > 0 && p < 1)).toBe(true)
    expect([...seen].sort((a, b) => a - b)).toEqual(seen)
    // Once the whole tape exists: ready, whole, 1, and `onReady` has fired once, with nothing played.
    expect(live().status).toBe('ready')
    expect(live().progress).toBe(1)
    expect(live().head).toBe(200)
    expect(live().computing).toBe(false)
    expect(live().playing).toBe(false)
    expect(ready).toHaveBeenCalledTimes(1)
    await frames.frames(3)
    expect(ready).toHaveBeenCalledTimes(1)
    expect(frames.pending).toBe(0)
    await m.unmount()
  })

  it('plays at the head when play is pressed before the tape is ready, and keeps filling', async () => {
    const m = await mountSession({ start: 'first' })
    await frames.frames(2)
    expect(m.current.done).toBe(false)
    const before = m.current.head
    await frames.frame()
    m.current.play()
    await frames.frame()
    expect(m.current.playing).toBe(true)
    expect(m.current.status).toBe('running')
    expect(m.current.done).toBe(false)
    expect(m.current.computing).toBe(true)
    await untilDone(m)
    expect(m.current.head).toBeGreaterThan(before)
    // Done, but still playing until it is stopped; then the tape is ready.
    expect(m.current.status).toBe('running')
    m.current.pause()
    await frames.frame()
    expect(m.current.status).toBe('ready')
    await m.unmount()
  })
})

describe('start: end (eager)', () => {
  it('computes a 200-step run in chunks, so the main thread is not blocked', async () => {
    const m = await mountSession({}, { mode: 'eager' })
    // The engine mode 'eager' is the 'end' policy, and the session was not computed on the spot.
    expect(m.current.start).toBe('end')
    expect(m.current.session.t).toBe(0)
    expect(m.current.done).toBe(false)
    expect(m.current.status).toBe('preparing')
    const heads: number[] = []
    let n = 0
    while (!m.current.done && n++ < 400) {
      await frames.frame()
      heads.push(m.current.session.t)
    }
    expect(n).toBeGreaterThan(3)
    expect(m.current.session.t).toBe(200)
    expect(m.current.status).toBe('ready')
    // No frame took more than a slice of the run.
    const steps = heads.map((h, i) => h - (heads[i - 1] ?? 0))
    expect(Math.max(...steps)).toBeLessThan(200)
    expect(Math.max(...steps)).toBeGreaterThan(0)
    await m.unmount()
  })

  it('is the same with the policy named', async () => {
    const m = await mountSession({ start: 'end' })
    expect(m.current.start).toBe('end')
    expect(m.current.session.t).toBe(0)
    await untilDone(m)
    expect(m.current.session.t).toBe(200)
    await m.unmount()
  })

  it('lets an explicit policy win over the mode', async () => {
    const m = await mountSession({ start: 'play' }, { mode: 'eager' })
    expect(m.current.start).toBe('play')
    await frames.frames(5)
    expect(m.current.head).toBe(0)
    await m.unmount()
  })
})

describe('start: play (live, the default)', () => {
  it('computes nothing until play, then computes ahead while it plays', async () => {
    const m = await mountSession({})
    expect(m.current.start).toBe('play')
    await frames.frames(5)
    expect(m.current.head).toBe(0)
    expect(m.current.computing).toBe(false)
    expect(m.current.status).toBe('preparing')
    m.current.play()
    await frames.frames(3)
    expect(m.current.head).toBeGreaterThan(0)
    expect(m.current.status).toBe('running')
    await m.unmount()
  })
})

// A factory that hands back real inline producers and keeps them, to watch what happens to them.
function watched(): { factory: ProducerFactory; made: Producer<never, never>[]; disposed: () => number } {
  const made: Producer<never, never>[] = []
  let disposals = 0
  const factory: ProducerFactory = (method, options, variants) => {
    const p = inlineProducer(method, options, variants)
    const dispose = p.dispose.bind(p)
    p.dispose = () => {
      disposals += 1
      dispose()
    }
    made.push(p as never)
    return p
  }
  return { factory, made, disposed: () => disposals }
}

describe('dispose', () => {
  it('lets go of the producer, the tape and the clock, and the handle starts from nothing if used again', async () => {
    const w = watched()
    const m = await mountSession({ start: 'first', producer: w.factory })
    await untilDone(m)
    const old = m.current.session
    expect(old.t).toBe(200)
    expect(old.run().events.length).toBeGreaterThan(100)
    m.current.dispose()
    await frames.frame()
    expect(w.disposed()).toBe(1)
    // Released: a new session at step 0 with an empty tape, nothing computing, nothing playing, no frame asked for.
    expect(m.current.session).not.toBe(old)
    expect(m.current.session.t).toBe(0)
    expect(m.current.session.run().events.length).toBeLessThan(5)
    expect(m.current.head).toBe(0)
    expect(m.current.computing).toBe(false)
    expect(m.current.playing).toBe(false)
    expect(m.current.status).toBe('preparing')
    expect(m.current.progress).toBe(0)
    expect(frames.pending).toBe(0)
    // Used again, it computes from step 0.
    m.current.play()
    await frames.frames(3)
    expect(m.current.head).toBeGreaterThan(0)
    await m.unmount()
  })

  it('is done by unmounting, and not by a strict-mode double effect', async () => {
    const w = watched()
    const m = await mountSession({ start: 'first', producer: w.factory }, { strict: true })
    // Strict mode set the effects up, cleaned them up and set them up again; the producer was kept and still works.
    expect(w.disposed()).toBe(0)
    await untilDone(m)
    expect(m.current.session.t).toBe(200)
    expect(w.disposed()).toBe(0)
    await m.unmount()
    await Promise.resolve()
    expect(w.disposed()).toBeGreaterThanOrEqual(1)
    expect(frames.pending).toBe(0)
    // Whatever was made is released.
    for (const p of w.made) expect(p.sessions[0]?.t).toBe(0)
  })
})

describe('in a worker', () => {
  const registry = methodRegistry([count])

  it('primes the whole run, and dispose and unmount terminate the worker', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
    try {
      const wires: ReturnType<typeof fakeWire>[] = []
      const producer = workerProducer({
        available: true,
        createWorker: () => {
          const w = fakeWire()
          serveProducer(w.scope, registry, { clock: tickingClock(5), schedule: w.schedule, sliceMs: 20, postMs: 0 })
          wires.push(w)
          return w.worker
        },
      })
      const m = await mountSession({ start: 'first', producer })
      expect(m.current.producer?.kind).toBe('worker')
      // The worker was started by priming, without play.
      expect(wires).toHaveLength(1)
      expect(m.current.status).toBe('preparing')
      const wire = wires[0] as ReturnType<typeof fakeWire>
      for (let i = 0; i < 20 && !m.current.done; i++) {
        wire.drain()
        await vi.advanceTimersByTimeAsync(150)
        await frames.frame()
      }
      expect(m.current.done).toBe(true)
      expect(m.current.status).toBe('ready')
      expect(m.current.head).toBe(200)
      expect(m.current.playing).toBe(false)
      // Disposing terminates the worker and drops the tape.
      m.current.dispose()
      await frames.frame()
      expect(wire.worker.terminated).toBe(true)
      expect(m.current.session.t).toBe(0)
      expect(m.current.done).toBe(false)
      expect(m.current.status).toBe('preparing')
      // Played again, a new worker starts, from nothing; leaving terminates that one too.
      m.current.play()
      await frames.frame()
      expect(wires).toHaveLength(2)
      await m.unmount()
      await Promise.resolve()
      expect((wires[1] as ReturnType<typeof fakeWire>).worker.terminated).toBe(true)
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('disposed', () => {
  it('is set by dispose, with the status still `preparing`, and cleared by play, compute, advance and apply', async () => {
    const m = await mountSession({ start: 'first' })
    expect(m.current.disposed).toBe(false)
    await untilDone(m)
    expect(m.current.status).toBe('ready')
    await inAct(() => m.current.dispose())
    expect(m.current.disposed).toBe(true)
    // The status is the one of a fresh session; `disposed` is what tells them apart.
    expect(m.current.status).toBe('preparing')
    expect(m.current.computing).toBe(false)
    // Nothing restarts it by itself: it stays disposed and still.
    await frames.frames(3)
    expect(m.current.disposed).toBe(true)
    expect(m.current.head).toBe(0)
    expect(frames.pending).toBe(0)
    await inAct(() => m.current.play())
    expect(m.current.disposed).toBe(false)
    await frames.frames(2)
    expect(m.current.head).toBeGreaterThan(0)
    for (const use of [
      () => m.current.compute(),
      () => m.current.advance({ steps: 1 }),
      () => void m.current.apply(),
    ]) {
      await inAct(() => m.current.dispose())
      expect(m.current.disposed).toBe(true)
      await inAct(use)
      expect(m.current.disposed).toBe(false)
    }
    await m.unmount()
  })

  it('computes nothing in a frame queued before it', async () => {
    let live: LiveSession<S, Config> | undefined
    let queued = false
    const m = await mountHook(() => {
      if (!queued) {
        queued = true
        requestAnimationFrame(() => live?.dispose())
      }
      live = useSession(count, { steps: 200 }, [], { start: 'first' })
      return live
    })
    await frames.frame()
    expect(m.current.disposed).toBe(true)
    await frames.frames(2)
    expect(m.current.session.t).toBe(0)
    expect(m.current.head).toBe(0)
    await m.unmount()
  })
})

describe('madeWith', () => {
  it('is the configuration and seed of the run on the tape, which stay until a held change is applied', async () => {
    let len = 200
    const m = await mountHook(() => useSession(count, { steps: 300, seed: 7, config: { len } }, [len], {}))
    expect(m.current.madeWith).toEqual({ config: { len: 200 }, seed: 7 })
    len = 50
    await m.rerender()
    // The page's options are newer than the run: the change is held until play.
    expect(m.current.stale).toBe(true)
    expect(m.current.held).toBe(true)
    expect(m.current.madeWith).toEqual({ config: { len: 200 }, seed: 7 })
    expect(m.current.madeWith.config).toBe(m.current.session.config)
    await inAct(() => void m.current.apply())
    expect(m.current.stale).toBe(false)
    expect(m.current.madeWith).toEqual({ config: { len: 50 }, seed: 7 })
    await m.unmount()
  })

  it('stays the old one for the pause before an unheld change is applied, then is the new one', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
    try {
      let len = 200
      const m = await mountHook(() => useSession(count, { steps: 300, config: { len } }, [len], { hold: false }))
      len = 50
      await m.rerender()
      expect(m.current.madeWith.config).toEqual({ len: 200 })
      await vi.advanceTimersByTimeAsync(300)
      await frames.frame()
      expect(m.current.madeWith.config).toEqual({ len: 50 })
      await m.unmount()
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('progress of a run that ends early', () => {
  it('is against the budget while it goes on, and exactly 1 once it has ended, whatever its length', async () => {
    const m = await mountHook(() => useSession(count, { steps: 1000, config: { len: 8 } }, [], { start: 'first' }))
    expect(m.current.progress).toBe(0)
    const seen: number[] = []
    for (let n = 0; n < 50 && !m.current.done; n++) {
      await frames.frame()
      seen.push(m.current.progress)
    }
    expect(m.current.done).toBe(true)
    expect(m.current.session.status).toBe('completed')
    expect(m.current.head).toBe(8)
    expect(m.current.progress).toBe(1)
    // Never above its budget fraction until it ended: a run of 8 steps of 1000 cannot read more than 0.008 before.
    for (const p of seen.slice(0, -1)) expect(p).toBeLessThanOrEqual(0.008)
    await m.unmount()
  })
})

describe('active', () => {
  it('computes nothing while false (no priming, no frames), keeps the tape, and carries on when true', async () => {
    let active = false
    const m = await mountHook(() => useSession(count, { steps: 200 }, [], { start: 'first', active }))
    expect(m.current.active).toBe(false)
    expect(m.current.computing).toBe(false)
    await frames.frames(3)
    expect(m.current.head).toBe(0)
    expect(frames.pending).toBe(0)
    active = true
    await m.rerender()
    expect(m.current.active).toBe(true)
    expect(m.current.computing).toBe(true)
    await frames.frames(2)
    const partway = m.current.session.t
    expect(partway).toBeGreaterThan(0)
    expect(m.current.done).toBe(false)
    // Hidden: the tape stays, nothing more is computed, no frame is asked for.
    active = false
    await m.rerender()
    expect(m.current.computing).toBe(false)
    await frames.frames(3)
    expect(m.current.session.t).toBe(partway)
    expect(m.current.disposed).toBe(false)
    expect(frames.pending).toBe(0)
    active = true
    await m.rerender()
    await untilDone(m)
    expect(m.current.head).toBe(200)
    expect(m.current.status).toBe('ready')
    await m.unmount()
  })

  it('pauses a session that was playing, and play resumes it', async () => {
    let active = true
    const m = await mountHook(() => useSession(count, { steps: 200 }, [], { active }))
    await inAct(() => m.current.play())
    await frames.frames(2)
    const partway = m.current.session.t
    expect(partway).toBeGreaterThan(0)
    expect(m.current.playing).toBe(true)
    active = false
    await m.rerender()
    await frames.frames(3)
    expect(m.current.session.t).toBe(partway)
    expect(m.current.playing).toBe(true)
    active = true
    await m.rerender()
    await frames.frames(2)
    expect(m.current.session.t).toBeGreaterThan(partway)
    await m.unmount()
  })

  it('does not prime a session disposed while inactive', async () => {
    let active = true
    const m = await mountHook(() => useSession(count, { steps: 200 }, [], { start: 'first', active }))
    await frames.frames(1)
    await inAct(() => m.current.dispose())
    active = false
    await m.rerender()
    active = true
    await m.rerender()
    await frames.frames(3)
    expect(m.current.head).toBe(0)
    expect(m.current.disposed).toBe(true)
    await m.unmount()
  })
})
