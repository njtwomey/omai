import { describe, expect, it } from 'vitest'
import { dict, int } from '../space/index.ts'
import type { Algorithm } from './algorithm.ts'
import { derived, event, Log, On, scalar } from './contract.ts'
import { messages, metricSeries } from './query.ts'
import { defineMethod, session } from './session.ts'

/** Counts up by `by`, logging the count, firing `tick/end` every third step, with a native `twice` metric. */
const counter = defineMethod<number, { readonly by: number }>({
  name: 'counter',
  config: dict({ by: int(1, 5, { default: 1 }) }),
  make: (c) => {
    const alg: Algorithm<number> = {
      name: 'count',
      config: { by: c.by },
      init: () => 0,
      step: (s, ctx) => {
        ctx.log('count', s)
        if (ctx.wants('expensive')) ctx.log('expensive', -s)
        const next = s + c.by
        if (next % 3 === 0) ctx.emit('tick/end', { at: next })
        return next
      },
      done: (s) => (s >= 100 ? 'converged' : undefined),
      observe: (s) => ({ metrics: { twice: 2 * s } }),
      serialise: (s) => s,
      deserialise: (j) => j as number,
    }
    return alg
  },
  contract: {
    count: scalar(),
    expensive: scalar(),
    doubled: scalar({ from: 'twice' }),
    square: derived({ compute: (s: number) => s * s }),
    'tick/end': event({ context: { at: 'the count' } }),
  },
  defaults: { count: Log() },
  hooks: {
    stopAt: (o) => (h) => {
      if ((h.context as { at: number }).at >= (o as { at: number }).at) h.stop('enough')
    },
  },
})

describe('sessions', () => {
  it('eager and live modes record the same tape and reach the same state', () => {
    const eager = session(counter, { mode: 'eager', steps: 10, logs: 'all' })
    const live = session(counter, { mode: 'live', steps: 10, logs: 'all' })
    expect(live.t).toBe(0)
    live.advance({ steps: 3 })
    live.advance({ steps: 3 })
    live.advance()
    expect(live.t).toBe(10)
    expect(live.status).toBe('completed')
    expect(live.run()).toEqual(eager.run())
    expect(metricSeries(eager.run(), 'count').values).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    expect(metricSeries(eager.run(), 'doubled').values).toEqual([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    expect(metricSeries(eager.run(), 'square').values).toEqual([0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    expect(messages(eager.run(), 'tick/end').map((m) => m.step)).toEqual([2, 5, 8])
  })

  it("'off' keeps nothing and costs nothing, and the method's defaults are a small choice", () => {
    const off = session(counter, { mode: 'eager', steps: 10, logs: 'off', past: false })
    expect(off.state).toBe(10)
    expect(off.run().events).toEqual([])
    expect(() => off.stateAt(3)).toThrow(/keeps no checkpoints/)
    const some = session(counter, { mode: 'eager', steps: 10 })
    expect(
      some
        .run()
        .events.filter((e) => e.type === 'metric')
        .every((e) => e.type === 'metric' && e.name === 'count'),
    ).toBe(true)
  })

  it("an 'off' session still keeps its past: it can seek a middle step, and only the run, checkpoints and end are kept", () => {
    const off = session(counter, { mode: 'eager', steps: 10, logs: 'off', checkpointEvery: 3 })
    expect(off.stateAt(5)).toBe(5)
    expect(new Set(off.run().events.map((e) => e.type))).toEqual(new Set(['run', 'checkpoint', 'end']))
  })

  it('schedules by steps and by events, and skips what nobody asked for', () => {
    const s = session(counter, {
      mode: 'eager',
      steps: 12,
      logs: { count: Log({ every: 4 }), doubled: Log({ every: ['tick/end', 2] }), square: Log({ every: 'tick/end' }) },
    })
    expect(metricSeries(s.run(), 'count').steps).toEqual([0, 4, 8])
    expect(metricSeries(s.run(), 'square').steps).toEqual([3, 6, 9, 12])
    expect(metricSeries(s.run(), 'doubled').steps).toEqual([6, 12])
    expect(metricSeries(s.run(), 'expensive').values).toEqual([])
    expect(s.stepsOf('tick/end')).toEqual([2, 5, 8, 11])
  })

  it('seeks any step from checkpoints, probes derived paths there, and forks exactly or on a new seed', () => {
    const s = session(counter, { mode: 'eager', steps: 20, checkpointEvery: 8, config: { by: 2 } })
    expect(s.stateAt(5)).toBe(10)
    expect(s.stateAt(20)).toBe(40)
    expect(s.probe(7, 'square')).toBe(196)
    expect(() => s.probe(7, 'count')).toThrow(/not a derived path/)
    expect(() => s.stateAt(21)).toThrow(/not been reached/)
    const same = s.fork(5)
    expect(same.t).toBe(5)
    same.advance()
    expect(same.state).toBe(s.state)
    expect(metricSeries(same.run(), 'count').steps[0]).toBe(5)
    const slower = s.fork(5, { config: { by: 1 } })
    slower.advance()
    expect(slower.state).toBe(10 + 15)
    expect(slower.stateAt(5)).toBe(10)
  })

  it('runs named hooks and stops the run when asked', () => {
    const s = session(counter, { mode: 'eager', steps: 50, logs: { 'tick/end': On('stopAt', { at: 9 }) } })
    expect(s.status).toBe('stopped')
    expect(s.state).toBe(9)
    const inline = session(counter, { mode: 'live', steps: 50, logs: { 'tick/end': On((h) => h.stop('now')) } })
    inline.advance()
    expect(inline.state).toBe(3)
  })

  it('eager to a point, then live', () => {
    const s = session(counter, { mode: { eager: ['tick/end', 2] }, steps: 100 })
    expect(s.stepsOf('tick/end').length).toBeGreaterThanOrEqual(2)
    expect(s.status).toBeUndefined()
    s.advance()
    expect(s.status).toBe('converged')
    const steps = session(counter, { mode: { eager: 4 }, steps: 100 })
    expect(steps.t).toBe(4)
  })

  it('rejects bad methods and configurations', () => {
    expect(() => defineMethod({ ...counter, contract: { 'a//b': scalar() } })).toThrow(/malformed/)
    expect(() => defineMethod({ ...counter, defaults: { nope: Log() } })).toThrow(/no path/)
    expect(() => session(counter, { config: { by: 99 } })).toThrow(/outside/)
    expect(() => session(counter, { logs: { nope: Log() } })).toThrow(/no path/)
  })
})
