import { dict, int, real } from 'omai-engine/space'
import { type Config, defineMethod, type Method, scalar } from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import { fakeWire, tickingClock } from './_fake.ts'
import { workerProducer } from './client.ts'
import { inlineProducer } from './inline.ts'
import { methodRegistry } from './registry.ts'
import { serveProducer } from './serve.ts'
import { guardBudget, stepsLeft } from './step.ts'

// A method that never ends by itself and has the budget knobs every inference method has.
const endless: Method<{ readonly k: number }, Config> = defineMethod({
  name: 'endless',
  config: dict({ budgetMs: real(0, 1e6, { default: 0 }), budgetSteps: int(0, 1e9, { default: 0 }) }),
  make: (c) => ({
    name: 'endless',
    config: c,
    init: () => ({ k: 0 }),
    step: (s, ctx) => {
      ctx.log('n', s.k + 1)
      return { k: s.k + 1 }
    },
  }),
  contract: { n: scalar() },
  defaults: {},
})

const options = (config: Config) => ({ config, steps: 100_000, mode: 'live' as const })

describe('the computational guard', () => {
  it('stops a run on the step budget (main thread)', () => {
    const p = inlineProducer(endless, options({ budgetSteps: 25 }), [{ label: '', config: {} }])
    for (let i = 0; i < 10 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 1000 })
    expect(p.sessions[0]?.t).toBe(25)
    expect(p.sessions[0]?.status).toBe('stopped')
  })

  it('leaves a run with no budget alone', () => {
    const p = inlineProducer(endless, options({}), [{ label: '', config: {} }])
    p.advance({ steps: 40 })
    expect(p.sessions[0]?.t).toBe(40)
    expect(p.sessions[0]?.status).toBeUndefined()
  })

  it('stops a run in a worker when its time is spent, and the tape ends there', () => {
    const wire = fakeWire()
    serveProducer(wire.scope, methodRegistry([endless]), {
      clock: tickingClock(5),
      schedule: wire.schedule,
      sliceMs: 20,
      postMs: 0,
    })
    const producer = workerProducer({ available: true, createWorker: () => wire.worker })(
      endless,
      options({ budgetMs: 100 }),
      [{ label: '', config: {} }],
    )
    producer.start({ eager: true })
    wire.drain(2000)
    const s = producer.sessions[0]
    expect(s?.status).toBe('stopped')
    expect(s?.t).toBeGreaterThan(0)
    expect(wire.events.some((e) => e.type === 'idle' && e.reason === 'ended')).toBe(true)
  })

  it('stops a run in a worker exactly at its step budget', () => {
    const wire = fakeWire()
    serveProducer(wire.scope, methodRegistry([endless]), { clock: tickingClock(1), schedule: wire.schedule })
    const producer = workerProducer({ available: true, createWorker: () => wire.worker })(
      endless,
      options({ budgetSteps: 37 }),
      [{ label: '', config: {} }],
    )
    producer.start({ eager: true })
    wire.drain(2000)
    expect(producer.sessions[0]?.t).toBe(37)
    expect(producer.sessions[0]?.status).toBe('stopped')
  })

  it('reads the knobs of the sessions', () => {
    const fake = (config: Config, t: number) =>
      ({ config, t, status: undefined, advance: () => undefined, stop: () => undefined }) as never
    expect(stepsLeft([fake({ budgetSteps: 10 }, 4), fake({}, 0)])).toBe(6)
    expect(stepsLeft([fake({}, 0)])).toBe(Number.POSITIVE_INFINITY)
    expect(guardBudget([fake({ budgetMs: 50 }, 0)], 49)).toBe(false)
    expect(guardBudget([fake({ budgetMs: 50 }, 0)], 50)).toBe(true)
  })
})
