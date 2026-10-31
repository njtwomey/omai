import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { dict } from '../space/index.ts'
import { stepContext } from './algorithm.ts'
import { type Contract, defineContract, event, Log, prefixContract, scalar, tensor } from './contract.ts'
import { mountContext } from './nesting.ts'
import { metricSeries } from './query.ts'
import { defineMethod, session } from './session.ts'
import { recorder, startRun } from './tracer.ts'

type CountPath = 'count' | 'count/doubled' | 'tick/end' | 'taps/*'

/** A counter whose step logs where `where(t)` says, so a test can aim a log at a path the contract lacks. */
const counter = (where: (t: number) => { readonly log?: string; readonly wants?: string; readonly emit?: string }) =>
  defineMethod<number, Record<string, never>, CountPath>({
    name: 'counter',
    config: dict({}),
    make: () => ({
      name: 'count',
      config: {},
      init: () => 0,
      step: (s, ctx) => {
        const at = where(ctx.t)
        // The typed context would refuse these literals; the point is what the run does with a path it was handed.
        const loose = ctx as unknown as {
          log(p: string, v: number): void
          wants(p: string): boolean
          emit(p: string): void
        }
        ctx.log('count', s)
        if (at.log !== undefined) loose.log(at.log, s)
        if (at.wants !== undefined) loose.wants(at.wants)
        if (at.emit !== undefined) loose.emit(at.emit)
        return s + 1
      },
    }),
    contract: {
      count: scalar(),
      'count/doubled': scalar(),
      'tick/end': event(),
      'taps/*': tensor({ role: 'activation' }),
    },
    defaults: { count: Log() },
  })

const run = (m: ReturnType<typeof counter>, logs?: 'off' | 'all'): number =>
  session(m, { mode: 'eager', steps: 3, ...(logs === undefined ? {} : { logs }) }).t

describe('the backstop for paths the contract does not have', () => {
  it('throws, naming the path, when an algorithm logs to a path that is not in the contract', () => {
    const m = counter(() => ({ log: 'count/doubld' }))
    expect(() => run(m)).toThrow(/log: the contract has no path 'count\/doubld'/)
    expect(() => run(m, 'all')).toThrow(/no path 'count\/doubld'/)
    expect(() => run(m, 'off')).toThrow(/no path 'count\/doubld'/)
  })

  it('throws for an emit of a path the contract does not have, and answers false to a wants of one', () => {
    expect(() => run(counter(() => ({ wants: 'cuont' })))).not.toThrow()
    expect(() => run(counter(() => ({ emit: 'tick/start' })))).toThrow(/emit: the contract has no path 'tick\/start'/)
  })

  it('does not throw for a path the contract has and nobody subscribed to', () => {
    const m = counter(() => ({ log: 'count/doubled', wants: 'count/doubled', emit: 'tick/end' }))
    const s = session(m, { mode: 'eager', steps: 3 })
    expect(metricSeries(s.run(), 'count').values).toEqual([0, 1, 2])
    expect(session(m, { mode: 'eager', steps: 3, logs: { 'count/doubled': Log() } }).t).toBe(3)
  })

  it('lets a path under a pattern key of the contract through', () => {
    const m = counter(() => ({ log: 'taps/fc1' }))
    expect(run(m)).toBe(3)
  })

  it('checks nothing when the run keeps no tape at all, and outside a session', () => {
    const m = counter(() => ({ log: 'count/doubld' }))
    expect(() => session(m, { mode: 'eager', steps: 3, logs: 'off', past: false })).not.toThrow()
    const rec = recorder()
    const ctx = stepContext(0, key(0), startRun(rec, { name: 'x', seed: 0, config: {} }))
    expect(() => ctx.log('anything/at/all', 1)).not.toThrow()
  })
})

describe('contract helpers', () => {
  it('defineContract returns the contract and refuses a malformed path', () => {
    const c = defineContract({ 'a/b': scalar(), c: scalar() })
    expect(Object.keys(c)).toEqual(['a/b', 'c'])
    expect(() => defineContract({ 'a//b': scalar() })).toThrow(/malformed path/)
    expect(() => defineContract({ '': scalar() })).toThrow(/malformed path/)
  })

  it('prefixContract mounts every path under the prefix', () => {
    const inner: Contract<unknown, 'loss' | 'x/y'> = defineContract({ loss: scalar(), 'x/y': scalar() })
    expect(Object.keys(prefixContract('inner', inner))).toEqual(['inner/loss', 'inner/x/y'])
  })

  it("mountContext sends an inner method's logs, queries and events under the prefix", () => {
    const rec = recorder()
    const outer = stepContext(1, key(0), startRun(rec, { name: 'x', seed: 0, config: {} }))
    const inner = mountContext<'in', 'loss' | 'end'>(outer, 'in')
    inner.log('loss', 0.25)
    inner.emit('end')
    expect(inner.wants('loss')).toBe(true)
    expect(rec.events.flatMap((e) => (e.type === 'metric' ? [e.name] : []))).toEqual(['in/loss'])
  })
})
