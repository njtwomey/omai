import { describe, expect, it } from 'vitest'
import { tensor } from '../tensor/index.ts'
import type { TensorEvent } from './events.ts'
import { instrument, traceCall } from './instrument.ts'
import { policies } from './policy.ts'
import { metricSeries, summary } from './query.ts'
import { validateAgainstSchema } from './schema.ts'
import { tapContext, tapsFor } from './taps.ts'
import { recorder, startRun } from './tracer.ts'

describe('instrument', () => {
  it('records numeric and tensor inputs and outputs, one run per call', () => {
    const rec = recorder()
    const scaled = instrument((x: ReturnType<typeof tensor>, _k: number, label: string) => (label === '' ? x : x), {
      name: 'scale',
      tracer: rec,
    })
    scaled(tensor([1, 2]), 3, 'a')
    scaled(tensor([3, 4]), 5, 'b')
    const run = rec.run()
    expect(Object.keys(summary(run))).toEqual(['scale#1', 'scale#2'])
    expect(metricSeries(run, 'arg1').values).toEqual([3, 5])
    const tensors = run.events.filter((e): e is TensorEvent => e.type === 'tensor')
    expect(tensors.map((e) => [e.name, e.role])).toEqual([
      ['arg0', 'state'],
      ['output', 'output'],
      ['arg0', 'state'],
      ['output', 'output'],
    ])
    const declared = run.events.find((e) => e.type === 'run')
    expect(declared?.type === 'run' && declared.channels !== undefined).toBe(true)
  })

  it('handles typed arrays, number arrays and booleans, and stays quiet untraced', () => {
    const rec = recorder()
    const f = instrument((a: Float64Array, b: number[], c: boolean) => (c ? a.length + b.length : 0), {
      tracer: rec,
      policy: policies.debug,
    })
    expect(f(new Float64Array([1, 2]), [3], true)).toBe(3)
    expect(rec.events.filter((e) => e.type === 'tensor')).toHaveLength(2)
    expect(metricSeries(rec.run(), 'arg2').values).toEqual([1])
    expect(rec.events[0]?.runId.startsWith('call#') || rec.events[0]?.runId === 'f#1').toBe(true)
    expect(instrument((x: number) => x * 2)(4)).toBe(8)
  })

  it('logs durations only from an injected clock, and records failures', () => {
    const rec = recorder()
    let now = 10
    const timed = instrument((x: number) => x, { tracer: rec, name: 'id', clock: () => (now += 2.5) })
    timed(1)
    expect(rec.events.filter((e) => e.type === 'log').map((e) => (e as { text: string }).text)).toEqual([
      'duration 2.5 ms',
    ])
    const failing = instrument(
      (): number => {
        throw new Error('boom')
      },
      { tracer: rec, name: 'bad' },
    )
    expect(() => failing()).toThrow('boom')
    expect(summary(rec.run())['bad#1']?.status).toBe('failed')
    expect(() =>
      traceCall(
        'bad',
        () => {
          throw new Error('x')
        },
        { tracer: rec },
      ),
    ).toThrow()
  })

  it('records only declared channels', () => {
    const rec = recorder()
    instrument((a: number, b: number) => a + b, { tracer: rec, name: 'add' })(1, 2)
    const run = rec.run()
    const declared = run.events[0]
    if (declared?.type !== 'run') throw new Error('no run event')
    const schema = declared.channels as unknown as Parameters<typeof validateAgainstSchema>[1]
    expect(validateAgainstSchema(run.events, schema)).toEqual([])
  })
})

describe('taps', () => {
  it('tapsFor records activations only when the policy and schedule ask', () => {
    const rec = recorder()
    const run = startRun(rec, {
      name: 'net',
      seed: 0,
      config: {},
      policy: { ...policies.debug, schedule: { every: 2 } },
    })
    const on = tapsFor(run, 2)
    expect(on.tap('h', tensor([1]), 'output')).toEqual(tensor([1]))
    expect([...on.entries.keys()]).toEqual(['h'])
    const off = tapsFor(run, 3)
    off.tap('h', tensor([1]))
    expect(off.enabled).toBe(false)
    expect(off.entries.size).toBe(0)
    expect(rec.events.filter((e) => e.type === 'tensor').map((e) => (e as TensorEvent).role)).toEqual(['output'])
    expect(tapContext(false).tap('x', tensor([1]))).toEqual(tensor([1]))
  })
})
