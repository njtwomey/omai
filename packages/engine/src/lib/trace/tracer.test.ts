import { describe, expect, it } from 'vitest'
import { tensor } from '../tensor/index.ts'
import { type Algorithm, runAlgorithm } from './algorithm.ts'
import type { TensorEvent } from './events.ts'
import { subRun } from './nesting.ts'
import { childPolicy, policies, tracePolicy } from './policy.ts'
import { runTree } from './query.ts'
import { channels, validateAgainstSchema } from './schema.ts'
import { subsample, summarise } from './summaries.ts'
import { recorder, startRun } from './tracer.ts'

const info = { name: 'r', seed: 0, config: {} }

describe('run handles apply the policy', () => {
  it('drops unchanged values under an on-change schedule', () => {
    const rec = recorder()
    const run = startRun(rec, {
      ...info,
      policy: tracePolicy({ schedule: { onChange: true, tolerance: 0.1 }, tensors: true, roles: ['state'] }),
    })
    for (const [t, v] of [1, 1.05, 2, 2, 1].entries()) run.metric(t, 'x', v)
    for (const [t, v] of [
      [1, 2],
      [1, 2],
      [1, 3],
    ].entries())
      run.tensor(t, 'w', 'state', v)
    expect(rec.events.filter((e) => e.type === 'metric').map((e) => e.step)).toEqual([0, 2, 4])
    expect(rec.events.filter((e) => e.type === 'tensor').map((e) => e.step)).toEqual([0, 2])
    expect(run.due(17)).toBe(true)
  })

  it('stops at the byte and event budgets but always ends', () => {
    const rec = recorder()
    const run = startRun(rec, { ...info, policy: tracePolicy({ maxBytes: 400 }) })
    for (let t = 0; t < 100; t++) run.metric(t, 'loss', t)
    run.end('completed', 100)
    const metrics = rec.events.filter((e) => e.type === 'metric')
    expect(metrics.length).toBeGreaterThan(0)
    expect(metrics.length).toBeLessThan(10)
    expect(rec.events[rec.events.length - 1]?.type).toBe('end')
    const few = recorder()
    const capped = startRun(few, { ...info, policy: tracePolicy({ maxEvents: 3 }) })
    for (let t = 0; t < 10; t++) capped.log(t, 'info', 'x')
    capped.end('stopped', 10)
    expect(few.events.map((e) => e.type)).toEqual(['run', 'log', 'log', 'end'])
  })

  it('records full values, summaries, or a subsample beside the summary', () => {
    const rec = recorder()
    const big = Array.from({ length: 100 }, (_, i) => i)
    const run = startRun(rec, {
      ...info,
      policy: tracePolicy({ tensors: true, roles: ['param'], detail: 'full', maxTensorSize: 10, subsample: 5 }),
    })
    run.tensor(0, 'small', 'param', tensor([1, 2, 3]))
    run.tensor(0, 'big', 'param', big, [10, 10])
    run.tensor(0, 'grad', 'grad', [1])
    const [small, large] = rec.events.filter((e): e is TensorEvent => e.type === 'tensor')
    expect(small?.values).toEqual([1, 2, 3])
    expect(large?.shape).toEqual([10, 10])
    expect(large?.indices).toEqual([0, 25, 50, 74, 99])
    expect(large?.values).toEqual([0, 25, 50, 74, 99])
    expect(large?.summary?.n).toBe(100)
    expect(rec.events.filter((e) => e.type === 'tensor')).toHaveLength(2)
  })

  it('gives child runs the inner policy, or none', () => {
    const inner: Algorithm<number> = {
      name: 'inner',
      config: {},
      init: () => 0,
      step: (s) => s + 1,
      observe: (s) => ({ metrics: { s } }),
    }
    const outer: Algorithm<number> = {
      name: 'outer',
      config: {},
      init: () => 0,
      step: (s, ctx) => s + subRun(ctx, inner, { maxSteps: 4 }).state,
      observe: (s) => ({ metrics: { total: s } }),
    }
    const light = recorder()
    runAlgorithm(outer, {
      maxSteps: 2,
      tracer: light,
      policy: tracePolicy({ inner: tracePolicy({ schedule: { firstAndLast: true } }) }),
    })
    const inners = light.events.filter((e) => e.type === 'metric' && e.name === 's')
    expect(inners.map((e) => (e as { step: number }).step)).toEqual([0, 4, 0, 4])
    const none = recorder()
    expect(runAlgorithm(outer, { maxSteps: 2, tracer: none, policy: tracePolicy({ inner: 'off' }) }).state).toBe(8)
    expect(runTree(none.run())[0]?.children).toEqual([])
    expect(childPolicy(policies.standard)).toBe(policies.standard)
  })
})

describe('schemas and summaries', () => {
  it('validates roles and shapes with named sizes', () => {
    const schema = channels({
      tensors: { w: { role: 'param', description: 'weights', shape: ['n', 'n'] } },
      messages: { split: { description: 'a split', payload: { feature: 'index' } } },
      episodes: true,
    })
    const tensorEvent = (shape: number[], role: 'param' | 'grad' = 'param') =>
      ({ type: 'tensor', runId: 'r', step: 0, name: 'w', role, shape }) as const
    expect(validateAgainstSchema([tensorEvent([3, 3])], schema)).toEqual([])
    expect(validateAgainstSchema([tensorEvent([3, 2])], schema)).toEqual(['tensor w has shape [3,2], declared [n,n]'])
    expect(validateAgainstSchema([tensorEvent([3])], schema)).toHaveLength(1)
    expect(validateAgainstSchema([tensorEvent([3, 3], 'grad')], schema)).toEqual([
      'tensor w has role grad, declared param',
    ])
    expect(
      validateAgainstSchema(
        [
          { type: 'message', runId: 'r', step: 0, kind: 'split', payload: {} },
          { type: 'episode', runId: 'r', episode: 0, return: 1, length: 2, terminated: true },
          { type: 'frame', runId: 'r', step: 0, frame: {} },
        ],
        schema,
      ),
    ).toEqual(['undeclared frames'])
  })

  it('summarises and subsamples', () => {
    const s = summarise([0, 0, 1, Number.NaN])
    expect(s).toMatchObject({ n: 4, min: 0, max: 1, zeros: 0.5 })
    expect(Number.isNaN(summarise([]).mean)).toBe(true)
    expect(subsample([1, 2, 3], 10)).toEqual({ values: [1, 2, 3], indices: [0, 1, 2] })
    expect(subsample([1, 2, 3], 1)).toEqual({ values: [1], indices: [0] })
    expect(subsample([], 4)).toEqual({ values: [], indices: [] })
  })
})
