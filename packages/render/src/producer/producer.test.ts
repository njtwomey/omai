import { dict, int, real } from 'omai-engine/space'
import {
  type Config,
  defineMethod,
  derived,
  event,
  Log,
  type Method,
  On,
  type Run,
  type SessionOptions,
  scalar,
  session,
  stateFromJson,
  stateToJson,
  type TraceEvent,
} from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import { type FakeWire, fakeWire, tickingClock } from './_fake.ts'
import { workerProducer } from './client.ts'
import { inlineProducer } from './inline.ts'
import type { Producer } from './producer.ts'
import { isProducerEvent, isProducerRequest, type ProducerEvent, wireOptions } from './protocol.ts'
import { methodRegistry } from './registry.ts'
import { ProbePending } from './remote.ts'
import { serveProducer } from './serve.ts'

interface S {
  readonly k: number
}

// A counting method: a scalar every step, a scalar kept as a window of the last three, an event every ten steps, a
// derived path, and an option to fail at step 5.
const counter: Method<S, Config> = defineMethod<S, Config>({
  name: 'counter',
  config: dict({
    rate: real(0, 10, { default: 1 }),
    len: int(1, 1000, { default: 60 }),
    boom: int(0, 1, { default: 0 }),
  }),
  make: (c) => ({
    name: 'counter',
    config: c,
    init: () => ({ k: 0 }),
    step: (s, ctx) => {
      const k = s.k + 1
      if (c['boom'] === 1 && k === 5) throw new Error('boom at five')
      ctx.log('n', k * (c['rate'] as number))
      ctx.log('window', k)
      if (k % 10 === 0) ctx.emit('tick', { k })
      return { k }
    },
    done: (s) => (s.k >= (c['len'] as number) ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as S,
  }),
  contract: {
    n: scalar(),
    window: scalar(),
    tick: event(),
    square: derived<S>({ compute: (s, args) => ({ kind: 'square', value: s.k * s.k, args: args ?? null }) }),
  },
  defaults: { n: Log(), window: Log({ keep: 3 }) },
  hooks: { halve: () => () => undefined },
})

const registry = methodRegistry([counter])

interface Rig {
  readonly wires: FakeWire[]
  readonly producer: Producer<S, Config>
  readonly wire: () => FakeWire
}

function rig(
  options: SessionOptions<S, Config> = {},
  settings: { lookahead?: number; config?: Partial<Config> } = {},
  method: Method<S, Config> = counter,
): Rig {
  const wires: FakeWire[] = []
  const make = workerProducer({
    available: true,
    ...(settings.lookahead === undefined ? {} : { lookahead: settings.lookahead }),
    createWorker: () => {
      const w = fakeWire()
      serveProducer(w.scope, registry, { clock: tickingClock(5), schedule: w.schedule, sliceMs: 20, postMs: 0 })
      wires.push(w)
      return w.worker
    },
  })
  const producer = make(method, options, [{ label: '', config: settings.config ?? {} }])
  return { wires, producer, wire: () => wires.at(-1) as FakeWire }
}

const stripped = (run: Run): TraceEvent[] => run.events.filter((e) => e.type !== 'checkpoint')

describe('the wire', () => {
  it('knows its messages', () => {
    expect(isProducerRequest({ type: 'run', id: 'p1', eager: false })).toBe(true)
    expect(isProducerRequest({ type: 'ready', id: 'p1' })).toBe(false)
    expect(isProducerRequest(null)).toBe(false)
    expect(isProducerEvent({ type: 'idle', id: 'p1', seq: 3, reason: 'ended' })).toBe(true)
    expect(isProducerEvent({ type: 'run', id: 'p1' })).toBe(false)
  })

  it('lets JSON options cross, hooks by name, and refuses what cannot', () => {
    expect(wireOptions({ seed: 3, steps: 50, config: { rate: 2 } })).toEqual({
      config: { rate: 2 },
      seed: 3,
      steps: 50,
    })
    expect(
      wireOptions({ logs: { n: Log({ every: 2 }), tick: On('halve', { patience: 3 }) } as never }).logs,
    ).toBeDefined()
    expect(() => wireOptions({ logs: { tick: On(() => undefined) } as never })).toThrow(/function and cannot cross/)
    expect(() => wireOptions({ key: { kind: 'key', words: [1, 2] } })).toThrow(/seed/)
  })

  it('finds methods by name', () => {
    expect(registry.get('counter')).toBe(counter)
    expect(() => registry.need('nope')).toThrow(/known: counter/)
    // The same method listed twice is kept once; two different methods under one name are an error.
    expect(methodRegistry([counter, counter]).names()).toEqual(['counter'])
    expect(() => methodRegistry([counter, { ...counter }])).toThrow(/two methods/)
  })
})

describe('the worker producer', () => {
  it('starts no worker until something is needed', () => {
    const { wires, producer } = rig()
    expect(wires).toHaveLength(0)
    expect(producer.kind).toBe('worker')
    expect(producer.sessions[0]?.t).toBe(0)
    producer.stop()
    expect(wires).toHaveLength(0)
  })

  it('streams a tape equal to the one computed here, in order', () => {
    const { producer, wire } = rig({ seed: 4 })
    producer.start()
    wire().drain()
    const remote = producer.sessions[0]
    expect(remote?.status).toBe('completed')
    const local = session(counter, { seed: 4, mode: 'eager' })
    expect(remote?.t).toBe(local.t)
    expect(stripped(remote?.run() as Run)).toEqual(stripped(local.run()))
    // The worker numbers its events one by one, and the page saw them in that order.
    const seqs = wire().events.map((e) => e.seq)
    expect(seqs).toEqual(seqs.map((_, i) => i + 1))
    expect(wire().events.some((e) => e.type === 'idle' && e.reason === 'ended')).toBe(true)
  })

  it('tells the page what the worker resolved: configuration, seed and the kind of each path', () => {
    const { producer, wire } = rig({ seed: 9 }, { config: { rate: 3 } })
    producer.start()
    wire().drain()
    const s = producer.sessions[0]
    expect(s?.seed).toBe(9)
    expect(s?.config['rate']).toBe(3)
    expect(s?.contract['square']?.kind).toBe('derived')
    expect(s?.contract['tick']?.kind).toBe('event')
  })

  it('follows a window that forgets: the page tape equals the worker tape', () => {
    const { producer, wire } = rig()
    producer.start()
    // A few slices at a time, so deltas arrive while the window is forgetting events.
    for (let i = 0; i < 40 && wire().drain(3) > 0; i++);
    wire().drain()
    const local = session(counter, { mode: 'eager' })
    const kept = stripped(producer.sessions[0]?.run() as Run).filter((e) => e.type === 'metric' && e.name === 'window')
    expect(kept).toEqual(stripped(local.run()).filter((e) => e.type === 'metric' && e.name === 'window'))
    expect(kept).toHaveLength(3)
  })

  it('reports where an event fired', () => {
    const { producer, wire } = rig()
    producer.start()
    wire().drain()
    const local = session(counter, { mode: 'eager' })
    expect(local.stepsOf('tick')).toHaveLength(6)
    expect(producer.sessions[0]?.stepsOf('tick')).toEqual(local.stepsOf('tick'))
  })

  it('stays within the lookahead of the cursor and goes on when the cursor moves', () => {
    const { producer, wire } = rig({}, { lookahead: 20 })
    producer.follow(0)
    producer.start()
    wire().drain()
    const head = producer.sessions[0]?.t ?? 0
    expect(head).toBeGreaterThanOrEqual(20)
    expect(head).toBeLessThan(40)
    expect(wire().events.at(-1)).toMatchObject({ type: 'idle', reason: 'lookahead' })
    producer.follow(30)
    wire().drain()
    expect(producer.sessions[0]?.t).toBeGreaterThanOrEqual(50)
    expect(producer.sessions[0]?.t).toBeLessThan(60)
    // Nothing more is computed while the cursor stands still, and no further idle message is sent for it.
    const idles = wire().events.filter((e) => e.type === 'idle').length
    producer.follow(30)
    wire().drain()
    expect(wire().events.filter((e) => e.type === 'idle').length).toBe(idles)
  })

  it('runs to the end regardless of the cursor when eager', () => {
    const { producer, wire } = rig({}, { lookahead: 10 })
    producer.start({ eager: true })
    wire().drain()
    expect(producer.sessions[0]?.status).toBe('completed')
  })

  it('computes exactly the steps asked for with advance, then idles', () => {
    const { producer, wire } = rig()
    producer.advance({ steps: 7 })
    wire().drain()
    expect(producer.sessions[0]?.t).toBe(7)
    expect(wire().events.at(-1)).toMatchObject({ type: 'idle', reason: 'advanced' })
    producer.advance({ steps: 3 })
    wire().drain()
    expect(producer.sessions[0]?.t).toBe(10)
  })

  it('pauses between slices and picks up again', () => {
    const { producer, wire } = rig()
    producer.start()
    wire().step() // the request
    wire().step() // the first slice is scheduled and run
    for (let i = 0; i < 4; i++) wire().step()
    producer.stop()
    wire().drain()
    const paused = producer.sessions[0]?.t ?? 0
    expect(paused).toBeGreaterThan(0)
    expect(paused).toBeLessThan(60)
    expect(wire().events.some((e) => e.type === 'idle' && e.reason === 'paused')).toBe(true)
    producer.start()
    wire().drain()
    expect(producer.sessions[0]?.status).toBe('completed')
  })

  it('answers probes on request: pending first, then the value of the worker', async () => {
    const { producer, wire } = rig()
    producer.start()
    wire().drain()
    const s = producer.sessions[0]
    let threw: unknown
    try {
      s?.probe(12, 'square', { a: 1 })
    } catch (e) {
      threw = e
    }
    expect(threw).toBeInstanceOf(ProbePending)
    wire().drain()
    const expected = session(counter, { mode: 'eager' }).probe(12, 'square', { a: 1 })
    expect(s?.probe(12, 'square', { a: 1 })).toEqual(expected)
    expect(producer.probed).toBe(1)
    // A different step or arguments is a different probe, and a repeated one is not asked twice.
    const before = wire().requests.length
    s?.probe(12, 'square', { a: 1 })
    expect(wire().requests.length).toBe(before)
    const later = producer.probeAsync(0, 20, 'square')
    wire().drain()
    await expect(later).resolves.toEqual(session(counter, { mode: 'eager' }).probe(20, 'square'))
  })

  it('answers only the newest of the probes a view asks for one path while the run is computing', () => {
    const { producer, wire } = rig()
    producer.advance({ steps: 12 })
    wire().drain()
    producer.start()
    const s = producer.sessions[0]
    for (const t of [3, 6, 9]) {
      try {
        s?.probe(t, 'square')
      } catch {
        // pending
      }
    }
    wire().drain()
    const probed = wire().events.filter((e) => e.type === 'probed')
    expect(probed.map((e) => (e.type === 'probed' ? [e.ok, e.superseded === true] : []))).toEqual([
      [false, true],
      [false, true],
      [true, false],
    ])
    const local = session(counter, { mode: 'eager' })
    expect(s?.probe(9, 'square')).toEqual(local.probe(9, 'square'))
    // A step that was skipped is asked again when a view comes back to it.
    const before = wire().requests.length
    s?.probe(3, 'square')
    expect(wire().requests.length).toBe(before + 1)
  })

  it('keeps showing the last answer for a path while the next is on its way', () => {
    const { producer, wire } = rig()
    producer.start()
    wire().drain()
    const s = producer.sessions[0]
    try {
      s?.probe(12, 'square')
    } catch {
      // pending
    }
    wire().drain()
    const local = session(counter, { mode: 'eager' })
    expect(s?.probe(12, 'square')).toEqual(local.probe(12, 'square'))
    // Step 14 is not answered yet: the view has the answer for 12 until it is, then the exact one.
    expect(s?.probe(14, 'square')).toEqual(local.probe(12, 'square'))
    wire().drain()
    expect(s?.probe(14, 'square')).toEqual(local.probe(14, 'square'))
  })

  it('shows the error of a probe that fails', async () => {
    const { producer, wire } = rig()
    const later = producer.probeAsync(0, 5, 'n')
    wire().drain()
    await expect(later).rejects.toThrow(/not a derived path/)
  })

  it('surfaces a failure of the run, keeping the tape up to it', () => {
    const { producer, wire } = rig({}, { config: { boom: 1 } })
    producer.start()
    wire().drain()
    expect(producer.error).toMatch(/boom at five/)
    expect(producer.sessions[0]?.status).toBe('failed')
    expect(producer.sessions[0]?.t).toBeLessThan(5)
  })

  it('surfaces a method the worker does not know', () => {
    const ghost = defineMethod<S, Config>({ ...counter, name: 'ghost' })
    const { producer, wire } = rig({}, {}, ghost)
    producer.start()
    wire().drain()
    expect(producer.error).toMatch(/no method called 'ghost'/)
    expect(producer.sessions[0]?.status).toBe('failed')
  })

  it('surfaces a worker that breaks', () => {
    const { producer, wire } = rig()
    producer.start()
    wire().worker.onerror?.({ message: 'script failed to load' })
    expect(producer.error).toBe('script failed to load')
  })

  it('is told when things happen', () => {
    const { producer, wire } = rig()
    let calls = 0
    const off = producer.subscribe(() => calls++)
    producer.start()
    wire().drain()
    expect(calls).toBeGreaterThan(2)
    off()
    const settled = calls
    producer.start()
    wire().drain()
    expect(calls).toBe(settled)
  })

  it('terminates cleanly on dispose, forgets the tape, and starts afresh when used again', () => {
    const { producer, wires } = rig()
    producer.start()
    wires[0]?.drain(8)
    expect(producer.sessions[0]?.t).toBeGreaterThan(0)
    producer.dispose()
    expect(wires[0]?.worker.terminated).toBe(true)
    expect(wires[0]?.requests.at(-1)).toMatchObject({ type: 'dispose' })
    expect(producer.sessions[0]?.t).toBe(0)
    expect(producer.sessions[0]?.run().events).toHaveLength(0)
    // Messages that were still in flight from the old worker do nothing.
    wires[0]?.drain()
    expect(producer.sessions[0]?.t).toBe(0)
    producer.start()
    expect(wires).toHaveLength(2)
    wires[1]?.drain()
    expect(producer.sessions[0]?.status).toBe('completed')
  })

  it('rejects the probes still waiting when disposed', async () => {
    const { producer } = rig()
    const waiting = producer.probeAsync(0, 3, 'square')
    producer.dispose()
    await expect(waiting).rejects.toThrow(/disposed/)
  })

  it('tells variants apart: each has its own tape, kept in step', () => {
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
    const producer = make(counter, {}, [
      { label: 'slow', config: { rate: 1 } },
      { label: 'fast', config: { rate: 5 } },
    ])
    producer.start()
    wires[0]?.drain()
    expect(producer.sessions).toHaveLength(2)
    const values = (i: number): number[] =>
      (producer.sessions[i]?.run().events ?? []).flatMap((e) =>
        e.type === 'metric' && e.name === 'n' ? [e.value] : [],
      )
    expect(values(1)).toEqual(values(0).map((v) => v * 5))
    expect(producer.labels).toEqual(['slow', 'fast'])
  })
})

describe('the fallback to the main thread', () => {
  it('makes the main-thread producer when Workers are unavailable, and says why', () => {
    const make = workerProducer({ available: false, createWorker: () => fakeWire().worker })
    const p = make(counter, {}, [{ label: '', config: {} }])
    expect(p.kind).toBe('inline')
    expect(p.fallback).toMatch(/not available/)
    p.advance({ steps: 5 })
    expect(p.sessions[0]?.t).toBe(5)
  })

  it('makes it too when the options cannot cross', () => {
    const make = workerProducer({ available: true, createWorker: () => fakeWire().worker })
    const p = make(counter, { logs: { tick: On(() => undefined) } as never }, [{ label: '', config: {} }])
    expect(p.kind).toBe('inline')
    expect(p.fallback).toMatch(/cannot cross/)
  })

  it('computes a comparison in step on the main thread', () => {
    const p = inlineProducer(counter, {}, [
      { label: 'a', config: {} },
      { label: 'b', config: { rate: 2 } },
    ])
    p.advance({ steps: 4 })
    expect(p.sessions.map((s) => s.t)).toEqual([4, 4])
  })
})

describe('the worker is told what to make', () => {
  it('sends the method by name with JSON options', () => {
    const { producer, wire } = rig({ seed: 2, steps: 30, config: { rate: 2 } })
    producer.start()
    const init = wire().requests[0]
    expect(init).toMatchObject({
      type: 'init',
      method: 'counter',
      options: { seed: 2, steps: 30, config: { rate: 2 } },
      checkpoints: false,
      lookahead: null,
    })
    expect(JSON.parse(JSON.stringify(init))).toEqual(init)
  })

  it('keeps checkpoint events off the wire unless asked', () => {
    const { producer, wire } = rig()
    producer.start()
    wire().drain()
    const sent = wire().events.flatMap((e: ProducerEvent) =>
      e.type === 'tape' ? e.deltas.flatMap((d) => d.events) : [],
    )
    expect(sent.some((e) => e.type === 'checkpoint')).toBe(false)
  })
})
