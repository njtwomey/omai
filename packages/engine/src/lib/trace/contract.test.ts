import { describe, expect, it } from 'vitest'
import {
  derived,
  event,
  frame,
  Log,
  matchesPath,
  message,
  On,
  resolve,
  type Subscription,
  scalar,
  tensor,
} from './contract.ts'
import type { TraceEvent } from './events.ts'
import { defaultLogContext, gated, logContext, scheduler } from './logging.ts'
import { recorder, startRun } from './tracer.ts'

const contract = {
  'training/loss': scalar({ direction: 'lower' }),
  'training/lr': scalar(),
  'model/params/w': tensor({ role: 'param', shape: [2, 2] }),
  'model/activations/*': tensor({ role: 'activation' }),
  'smo/working-set': message({ from: 'workingSet' }),
  'smo/objective': scalar({ from: 'objective' }),
  'world/frame': frame(),
  'model/surface': derived({ cost: 'dear', compute: (s: number) => s * 10 }),
  'training/epoch/end': event({ context: { epoch: 'the epoch' } }),
}

describe('path patterns', () => {
  it('match one segment with * and any number with **', () => {
    expect(matchesPath('a/b', 'a/b')).toBe(true)
    expect(matchesPath('a/*', 'a/b')).toBe(true)
    expect(matchesPath('a/*', 'a/b/c')).toBe(false)
    expect(matchesPath('a/**', 'a')).toBe(true)
    expect(matchesPath('a/**', 'a/b/c')).toBe(true)
    expect(matchesPath('**/loss', 'training/val/loss')).toBe(true)
    expect(matchesPath('*/loss', 'training/val/loss')).toBe(false)
  })
})

describe('resolve', () => {
  it('matches exact paths and patterns, keeping each path once with the first spec', () => {
    const r = resolve(contract, { 'training/loss': Log({ every: 3 }), 'training/**': Log() })
    expect(r.logs.map((l) => [l.path, l.spec.every])).toEqual([
      ['training/loss', 3],
      ['training/lr', undefined],
      ['training/epoch/end', undefined],
    ])
    expect(r.policy.metrics).toEqual(['training/loss', 'training/lr'])
    expect(r.policy.messages).toEqual(['training/epoch/end'])
    expect(r.policy.frames).toBe(false)
    expect(r.policy.checkpoints).toBe(false)
  })

  it('selects native names for renamed channels and widens to true for pattern sources', () => {
    const r = resolve(contract, { 'smo/**': Log(), 'model/activations/*': Log({ detail: 'full' }) })
    expect(r.policy.metrics).toEqual(['objective'])
    expect(r.policy.messages).toEqual(['workingSet'])
    expect(r.policy.tensors).toBe(true)
    expect(r.policy.roles).toEqual(['activation'])
    expect(r.policy.detail).toBe('full')
  })

  it('rejects unknown exact paths, hooks on non-events and unknown named hooks', () => {
    expect(() => resolve(contract, { 'training/nope': Log() } as Subscription)).toThrow(/no path/)
    expect(() => resolve(contract, { 'training/loss': On(() => {}) })).toThrow(/not an event/)
    expect(() => resolve(contract, { 'training/epoch/end': On('missing') })).toThrow(/no hook/)
    expect(resolve(contract, { 'nothing/**': Log() }).logs).toEqual([])
  })

  it('keeps all, the last or the last n of a path, and rejects anything else', () => {
    expect(resolve(contract, { 'training/loss': Log({ keep: 'last' }) }).logs[0]?.spec.keep).toBe('last')
    expect(resolve(contract, { 'training/**': Log({ keep: 4 }) }).logs.map((l) => l.spec.keep)).toEqual([4, 4, 4])
    expect(() => resolve(contract, { 'training/loss': Log({ keep: 0 }) })).toThrow(/keep/)
    expect(() => resolve(contract, { 'training/**': Log({ keep: 1.5 }) })).toThrow(/keep/)
  })

  it('binds hooks by function and by name', () => {
    let n = 0
    const r = resolve(
      contract,
      { 'training/epoch/end': On('count', { by: 2 }) },
      {
        hooks: {
          count: (o) => () => {
            n += (o as { by: number }).by
          },
        },
      },
    )
    r.hooks.get('training/epoch/end')?.[0]?.({
      t: 0,
      path: '',
      context: {},
      run: undefined as never,
      log: () => {},
      stop: () => {},
    })
    expect(n).toBe(2)
  })

  it("'all' takes every exact path, dear derived ones log-spaced; 'off' resolves to nothing", () => {
    const all = resolve(contract, 'all')
    expect(all.logs.find((l) => l.path === 'model/surface')?.spec.every).toBe('logSpaced')
    expect(all.logs.some((l) => l.path === 'model/activations/*')).toBe(false)
    expect(all.policy.frames).toBe(true)
    expect(resolve(contract, 'off').off).toBe(true)
    expect(resolve(contract, 'defaults', { defaults: { 'training/lr': Log() } }).logs.map((l) => l.path)).toEqual([
      'training/lr',
    ])
  })
})

describe('scheduler', () => {
  it('schedules by steps, log-spaced steps and the nth firing of an event', () => {
    const r = resolve(contract, {
      'training/loss': Log({ every: 4 }),
      'training/lr': Log({ every: 'logSpaced' }),
      'model/params/w': Log({ every: ['training/epoch/end', 2] }),
      'world/frame': Log({ every: 'training/epoch/end' }),
    })
    const s = scheduler(r)
    expect([0, 1, 4, 6, 8].map((t) => s.due('training/loss', t))).toEqual([true, false, true, false, true])
    expect([0, 1, 2, 3, 4].map((t) => s.due('training/lr', t))).toEqual([true, true, true, false, true])
    expect(s.due('model/params/w', 5)).toBe(false)
    s.fired('training/epoch/end', 5)
    expect(s.due('model/params/w', 5)).toBe(false)
    expect(s.due('world/frame', 5)).toBe(true)
    s.consume('world/frame')
    expect(s.due('world/frame', 6)).toBe(false)
    s.fired('training/epoch/end', 9)
    expect(s.due('model/params/w', 9)).toBe(true)
    expect(s.stepsOf('training/epoch/end')).toEqual([5, 9])
    expect(s.due('nothing', 0)).toBe(false)
  })
})

describe('log context and gating tracer', () => {
  it('logs subscribed paths by kind when due, renames native channels, and drops the rest', () => {
    const r = resolve(contract, {
      'training/loss': Log({ every: 2 }),
      'smo/objective': Log(),
      'smo/working-set': Log({ every: 'training/epoch/end' }),
      'model/activations/*': Log(),
      'training/epoch/end': Log(),
    })
    const s = scheduler(r)
    const rec = recorder()
    const run = startRun(gated(rec, r, s), { name: 'x', seed: 0, config: {}, policy: r.policy })
    const at = (t: number) => logContext(run, r, s, t, { stop: () => {} })
    at(1).log('training/loss', 0.5) // not due
    at(2).log('training/loss', 0.4)
    at(2).log('training/lr', 1) // not subscribed
    run.metric(2, 'objective', 7) // a native channel, renamed
    run.metric(2, 'violation', 1) // native and unsubscribed
    run.message(2, 'workingSet', [1, 2]) // waits for the event
    at(3).emit('training/epoch/end', { epoch: 1 })
    run.message(3, 'workingSet', [3, 4])
    run.message(4, 'workingSet', [5, 6]) // consumed: not due again
    const taps = at(5).taps('model/activations')
    expect(taps.enabled).toBe(true)
    const kinds = rec.events
      .filter((e) => e.type !== 'run')
      .map((e) => `${e.type}:${'name' in e ? e.name : 'kind' in e ? e.kind : ''}@${'step' in e ? e.step : ''}`)
    expect(kinds).toEqual([
      'metric:training/loss@2',
      'metric:smo/objective@2',
      'message:training/epoch/end@3',
      'message:smo/working-set@3',
    ])
    expect(at(0).wants('training/loss')).toBe(true)
    expect(at(1).wants('training/loss')).toBe(false)
  })

  it('runs hooks on events and lets them stop the run', () => {
    let stopped = ''
    const r = resolve(contract, {
      'training/epoch/end': On((h) => h.stop(`epoch ${(h.context as { epoch: number }).epoch}`)),
    })
    const run = startRun(recorder(), { name: 'x', seed: 0, config: {}, policy: r.policy })
    logContext(run, r, scheduler(r), 4, {
      stop: (why) => {
        stopped = why
      },
    }).emit('training/epoch/end', { epoch: 2 })
    expect(stopped).toBe('epoch 2')
  })

  it('outside a session, the default context logs by value type under the run policy', () => {
    const rec = recorder()
    const run = startRun(rec, {
      name: 'x',
      seed: 0,
      config: {},
      policy: { ...resolve({}, 'all').policy, metrics: true, messages: true },
    })
    const ctx = defaultLogContext(run, 1)
    ctx.log('a/n', 3)
    ctx.log('a/flag', true)
    ctx.emit('a/done', { ok: 1 })
    const types = rec.events.map((e: TraceEvent) => e.type)
    expect(types).toEqual(['run', 'metric', 'metric', 'message'])
  })
})
