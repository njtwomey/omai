import { describe, expect, it } from 'vitest'
import { Log, message, resolve, scalar, tensor } from './contract.ts'
import type { TraceEvent } from './events.ts'
import { gated, logContext, scheduler } from './logging.ts'
import { recorder, startRun, tee } from './tracer.ts'

const contract = {
  'a/t': tensor({ role: 'state', description: 'full values wanted' }),
  'b/t': tensor({ role: 'state', description: 'summaries wanted' }),
  'c/n': scalar({ description: 'a counter with a byte budget' }),
  'd/m': message({ description: 'a note per step' }),
  'model/activations/*': tensor({ role: 'activation' }),
}

const stepsOf = (events: readonly TraceEvent[], type: TraceEvent['type']): number[] =>
  events.flatMap((e) => (e.type === type && 'step' in e ? [e.step] : []))

describe('per-path detail and budgets', () => {
  it('keeps a summary on a path that asked for one even when another path turned full values on', () => {
    const r = resolve(contract, { 'a/t': Log({ detail: 'full' }), 'b/t': Log() })
    const s = scheduler(r)
    const rec = recorder()
    const run = startRun(gated(rec, r, s), { name: 'x', seed: 0, config: {}, policy: r.policy })
    const at = logContext(run, r, s, 1, { stop: () => {} })
    at.log('a/t', [1, 2, 3, 4])
    at.log('b/t', [1, 2, 3, 4])
    const tensors = rec.events.filter((e) => e.type === 'tensor')
    expect(
      tensors.map((e) => (e.type === 'tensor' ? [e.name, e.values !== undefined, e.summary !== undefined] : [])),
    ).toEqual([
      ['a/t', true, false],
      ['b/t', false, true],
    ])
  })

  it('stops a path once its events pass their byte budget, and leaves the others alone', () => {
    const r = resolve(contract, { 'c/n': Log({ budget: 150 }), 'a/t': Log() })
    const s = scheduler(r)
    const rec = recorder()
    const run = startRun(gated(rec, r, s), { name: 'x', seed: 0, config: {}, policy: r.policy })
    for (let t = 0; t < 10; t++) {
      const at = logContext(run, r, s, t, { stop: () => {} })
      at.log('c/n', t)
      at.log('a/t', [t])
    }
    const counts = rec.events.filter((e) => e.type === 'metric').length
    expect(counts).toBeGreaterThan(0)
    expect(counts).toBeLessThan(10)
    expect(rec.events.filter((e) => e.type === 'tensor').length).toBe(10)
  })

  it('keeps a window of a path in every channel, forgetting its oldest events as new ones arrive, in order', () => {
    const r = resolve(contract, { 'c/n': Log({ keep: 3 }), 'a/t': Log({ keep: 'last' }), 'd/m': Log({ keep: 2 }) })
    const s = scheduler(r)
    const rec = recorder()
    const run = startRun(gated(rec, r, s), { name: 'x', seed: 0, config: {}, policy: r.policy })
    for (let t = 0; t < 10; t++) {
      const at = logContext(run, r, s, t, { stop: () => {} })
      at.log('c/n', t)
      at.log('a/t', [t])
      at.log('d/m', { t })
    }
    expect(stepsOf(rec.events, 'metric')).toEqual([7, 8, 9])
    expect(stepsOf(rec.events, 'tensor')).toEqual([9])
    expect(stepsOf(rec.events, 'message')).toEqual([8, 9])
    const order = rec.events.flatMap((e) => ('step' in e ? [e.step] : []))
    expect(order).toEqual([...order].sort((a, b) => a - b))
  })

  it('the byte budget counts what is held, so a kept window lets a budgeted path go on', () => {
    const r = resolve(contract, { 'c/n': Log({ keep: 'last', budget: 150 }) })
    const s = scheduler(r)
    const rec = recorder()
    const run = startRun(gated(rec, r, s), { name: 'x', seed: 0, config: {}, policy: r.policy })
    for (let t = 0; t < 10; t++) logContext(run, r, s, t, { stop: () => {} }).log('c/n', t)
    expect(stepsOf(rec.events, 'metric')).toEqual([9])
  })

  it('a destination that cannot forget holds everything it is sent', () => {
    const r = resolve(contract, { 'c/n': Log({ keep: 'last' }) })
    const s = scheduler(r)
    const rec = recorder()
    const run = startRun(gated(tee(rec), r, s), { name: 'x', seed: 0, config: {}, policy: r.policy })
    for (let t = 0; t < 4; t++) logContext(run, r, s, t, { stop: () => {} }).log('c/n', t)
    expect(stepsOf(rec.events, 'metric')).toEqual([0, 1, 2, 3])
  })

  it('remembers pattern matches for tapped names', () => {
    const r = resolve(contract, { 'model/activations/*': Log({ every: 2 }) })
    const s = scheduler(r)
    expect(s.entry('model/activations/h1')?.path).toBe('model/activations/*')
    expect(s.entry('model/activations/h1')?.path).toBe('model/activations/*')
    expect(s.due('model/activations/h1', 2)).toBe(true)
    expect(s.due('model/activations/h1', 3)).toBe(false)
    expect(s.entry('nothing/here')).toBeUndefined()
  })
})
