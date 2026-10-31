import { dict, int } from 'mdml-engine/space'
import {
  type Config,
  defineMethod,
  derived,
  event,
  Log,
  type Method,
  scalar,
  session,
  stateFromJson,
  stateToJson,
  tensor,
} from 'mdml-engine/trace'
import { describe, expect, it } from 'vitest'
import { type FakeWire, fakeWire, tickingClock } from './_fake.ts'
import { workerProducer } from './client.ts'
import { methodRegistry } from './registry.ts'
import { serveProducer } from './serve.ts'
import { parseTape, probeSteps, type ReplaySession, replaySessions, tapeOf } from './tape.ts'

interface S {
  readonly k: number
  readonly v: Float64Array
}

// A method whose probe returns a typed array inside a record, to see that the file keeps what JSON cannot.
const orbit: Method<S, Config> = defineMethod<S, Config>({
  name: 'orbit',
  config: dict({ len: int(1, 100, { default: 30 }) }),
  make: (c) => ({
    name: 'orbit',
    config: c,
    init: () => ({ k: 0, v: Float64Array.of(0, 0) }),
    step: (s, ctx) => {
      const k = s.k + 1
      ctx.log('r', Math.sqrt(k))
      ctx.log('pos', [Math.cos(k), Math.sin(k)])
      if (k % 10 === 0) ctx.emit('lap')
      return { k, v: Float64Array.of(Math.cos(k), Math.sin(k)) }
    },
    done: (s) => (s.k >= (c['len'] as number) ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as S,
  }),
  contract: {
    r: scalar(),
    pos: tensor({ role: 'state', shape: [2] }),
    lap: event(),
    'field/at': derived<S>({
      compute: (s, args) => ({ kind: 'vectorField', k: s.k, v: s.v, args: args ?? null }),
    }),
  },
  defaults: { r: Log(), pos: Log({ detail: 'full' }) },
})
const registry = methodRegistry([orbit])

describe('probeSteps', () => {
  it('takes every step when few and a spread with both ends and the viewed step when many', () => {
    expect(probeSteps(3, 10)).toEqual([0, 1, 2, 3])
    const spread = probeSteps(1000, 20, 333)
    expect(spread.length).toBeLessThanOrEqual(21)
    expect(spread[0]).toBe(0)
    expect(spread.at(-1)).toBe(1000)
    expect(spread).toContain(333)
    expect([...spread].sort((a, b) => a - b)).toEqual(spread)
  })
})

describe('the tape as a file', () => {
  const local = session(orbit, { mode: 'eager', seed: 5 })

  it('round-trips: the replayed session has the same tape and the same probes', async () => {
    const file = await tapeOf('orbit', [{ label: '', session: local }], {
      requests: [{ path: 'field/at', args: { n: 2 } }, { path: 'r' }],
      at: 12,
      title: 'orbit',
    })
    const text = JSON.stringify(file)
    const back = parseTape(text)
    expect(back).toEqual(JSON.parse(text))
    const [replayed] = replaySessions(back, registry)
    const r = replayed?.session as ReplaySession<unknown>
    expect(r.t).toBe(local.t)
    expect(r.status).toBe(local.status)
    expect(r.run()).toEqual(JSON.parse(JSON.stringify(local.run())))
    expect(r.stepsOf('lap')).toEqual(local.stepsOf('lap'))
    for (const t of [0, 1, 17, 30]) {
      const want = local.probe(t, 'field/at', { n: 2 }) as unknown as { v: Float64Array }
      const got = r.probe(t, 'field/at', { n: 2 }) as unknown as { v: Float64Array }
      expect(got.v).toBeInstanceOf(Float64Array)
      expect(got).toEqual(want)
    }
    // Only derived paths are stored; the tape path was asked for and dropped.
    expect(back.variants[0]?.probes.every((p) => p.path === 'field/at')).toBe(true)
    expect(back.at).toBe(12)
    expect(back.title).toBe('orbit')
  })

  it('answers with the nearest stored step when a step was not stored', async () => {
    const file = await tapeOf('orbit', [{ label: 'a', session: local }], {
      requests: [{ path: 'field/at' }],
      maxSteps: 4,
    })
    const [replayed] = replaySessions(parseTape(JSON.stringify(file)), registry)
    const r = replayed?.session as ReplaySession<unknown>
    const stored = file.variants[0]?.probes.map((p) => p.t) ?? []
    expect(stored.length).toBeLessThanOrEqual(5)
    const near = (r.probe(17, 'field/at') as { k: number }).k
    expect(stored).toContain(near)
    expect(() => r.probe(3, 'other' as never)).toThrow(/holds no probe/)
  })

  it('keeps comparisons: one entry per variant', async () => {
    const a = session(orbit, { mode: 'eager', config: { len: 10 } })
    const b = session(orbit, { mode: 'eager', config: { len: 20 } })
    const file = await tapeOf(
      'orbit',
      [
        { label: 'short', session: a },
        { label: 'long', session: b },
      ],
      { requests: [] },
    )
    const sessions = replaySessions(parseTape(JSON.stringify(file)), registry)
    expect(sessions.map((s) => [s.label, s.session.t])).toEqual([
      ['short', 10],
      ['long', 20],
    ])
  })

  it('rejects what is not a tape file', () => {
    expect(() => parseTape('not json')).toThrow(/JSON/)
    expect(() => parseTape({ schema: 'other' })).toThrow(/not a tape file/)
    expect(() => parseTape({ schema: 'mdml.tape/1', method: 'orbit', seed: 0, variants: [{}] })).toThrow(/malformed/)
    expect(() =>
      replaySessions(parseTape({ schema: 'mdml.tape/1', method: 'nope', seed: 0, variants: [] }), registry),
    ).toThrow(/no method called/)
  })

  it('writes the tape of a session computed in a worker, with its probes asked of the worker', async () => {
    const wires: FakeWire[] = []
    const make = workerProducer({
      available: true,
      createWorker: () => {
        const w = fakeWire()
        serveProducer(w.scope, registry, { clock: tickingClock(5), schedule: w.schedule, sliceMs: 20, postMs: 0 })
        wires.push(w)
        return w.worker
      },
    })
    const producer = make(orbit, { seed: 5 }, [{ label: '', config: {} }])
    producer.start()
    wires[0]?.drain()
    const remote = producer.sessions[0]
    if (remote === undefined) throw new Error('no session')
    const written = tapeOf('orbit', [{ label: '', session: remote }], { requests: [{ path: 'field/at' }], maxSteps: 8 })
    // The worker answers the probes as the conversation is stepped.
    for (let i = 0; i < 200; i++) {
      wires[0]?.drain()
      await Promise.resolve()
    }
    const file = await written
    const [replayed] = replaySessions(parseTape(JSON.stringify(file)), registry)
    const r = replayed?.session as ReplaySession<unknown>
    expect(r.run().events.filter((e) => e.type !== 'checkpoint')).toEqual(
      JSON.parse(JSON.stringify(local.run())).events.filter((e: { type: string }) => e.type !== 'checkpoint'),
    )
    expect(r.probe(30, 'field/at')).toEqual(local.probe(30, 'field/at'))
  })
})
