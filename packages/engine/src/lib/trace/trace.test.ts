import { describe, expect, it } from 'vitest'
import { normal } from '../random/index.ts'
import { tensor } from '../tensor/index.ts'
import {
  type Algorithm,
  channels,
  diffRuns,
  exportRun,
  iterate,
  metricSeries,
  onSchedule,
  parseRun,
  policies,
  RunSchema,
  recorder,
  resume,
  runAlgorithm,
  runTree,
  seek,
  subRun,
  summary,
  tapContext,
  tensorSeries,
  traceCall,
  tracePolicy,
  validateAgainstSchema,
} from './index.ts'

// A noisy halving process: state is a number, each step draws from its step key.
const noisyHalve: Algorithm<number> = {
  name: 'noisy-halve',
  config: { factor: 0.5 },
  schema: channels({
    metrics: { x: { description: 'value' } },
    tensors: { x: { role: 'state', description: 'value as a tensor' } },
  }),
  init: () => 16,
  step: (x, ctx) => x * 0.5 + 0.01 * (normal(ctx.key, [1]).data[0] ?? 0),
  done: (x) => (Math.abs(x) < 0.1 ? 'converged' : undefined),
  observe: (x) => ({ metrics: { x }, tensors: { x: { role: 'state', value: tensor([x]) } } }),
  serialise: (x) => x,
  deserialise: (j) => j as number,
}

describe('runAlgorithm', () => {
  it('is deterministic for a seed and different across seeds', () => {
    const a = recorder()
    const b = recorder()
    const c = recorder()
    runAlgorithm(noisyHalve, { seed: 1, maxSteps: 50, tracer: a })
    runAlgorithm(noisyHalve, { seed: 1, maxSteps: 50, tracer: b })
    runAlgorithm(noisyHalve, { seed: 2, maxSteps: 50, tracer: c })
    expect(exportRun(a.run())).toEqual(exportRun(b.run()))
    expect(exportRun(a.run())).not.toEqual(exportRun(c.run()))
  })

  it('records run, metrics and end events that validate', () => {
    const rec = recorder()
    const result = runAlgorithm(noisyHalve, { seed: 0, maxSteps: 50, tracer: rec })
    expect(result.status).toBe('converged')
    const run = rec.run()
    expect(RunSchema.safeParse(run).success).toBe(true)
    expect(run.events[0]?.type).toBe('run')
    expect(run.events.at(-1)?.type).toBe('end')
    expect(metricSeries(run, 'x').steps).toEqual(Array.from({ length: result.steps + 1 }, (_, i) => i))
    expect(validateAgainstSchema(run.events, noisyHalve.schema ?? channels({}))).toEqual([])
  })

  it('follows the policy schedule and always observes the last step', () => {
    const rec = recorder()
    const result = runAlgorithm(noisyHalve, {
      seed: 0,
      maxSteps: 50,
      tracer: rec,
      policy: tracePolicy({ schedule: { every: 3 } }),
    })
    const steps = metricSeries(rec.run(), 'x').steps
    expect(steps.slice(0, -1).every((s) => s % 3 === 0)).toBe(true)
    expect(steps.at(-1)).toBe(result.steps)
  })

  it('records tensors as summaries or full values by policy', () => {
    const summaryRec = recorder()
    runAlgorithm(noisyHalve, {
      seed: 0,
      maxSteps: 3,
      tracer: summaryRec,
      policy: tracePolicy({ tensors: true, roles: ['state'] }),
    })
    const s = tensorSeries(summaryRec.run(), 'x')
    expect(s[0]?.summary?.mean).toBe(16)
    expect(s[0]?.values).toBeUndefined()
    const fullRec = recorder()
    runAlgorithm(noisyHalve, { seed: 0, maxSteps: 3, tracer: fullRec, policy: policies.debug })
    expect(tensorSeries(fullRec.run(), 'x')[0]?.values).toEqual([16])
  })

  it('nests child runs under their parent', () => {
    const inner: Algorithm<number> = { name: 'inner', config: {}, init: () => 0, step: (s) => s + 1 }
    const outer: Algorithm<number> = {
      name: 'outer',
      config: {},
      init: () => 0,
      step: (s, ctx) => s + subRun(ctx, inner, { maxSteps: 2 }).state,
    }
    const rec = recorder()
    expect(runAlgorithm(outer, { seed: 0, maxSteps: 3, tracer: rec }).state).toBe(6)
    const tree = runTree(rec.run())
    expect(tree).toHaveLength(1)
    expect(tree[0]?.children).toHaveLength(3)
    expect(tree[0]?.children[0]?.run.runId).toBe('outer:0/inner#1')
  })

  it('resumes and seeks exactly from checkpoints', () => {
    const rec = recorder()
    const full = runAlgorithm(noisyHalve, {
      seed: 5,
      maxSteps: 6,
      tracer: rec,
      policy: tracePolicy({ checkpoints: true, checkpointInterval: 2 }),
    })
    const run = rec.run()
    const checkpoint = run.events.find((e) => e.type === 'checkpoint' && e.step === 2)
    if (checkpoint === undefined || checkpoint.type !== 'checkpoint') throw new Error('missing checkpoint')
    expect(resume(noisyHalve, checkpoint, { seed: 5, maxSteps: 4 }).state).toBe(full.state)
    expect(seek(noisyHalve, run, 6, { seed: 5 })).toBe(full.state)
  })

  it('iterates states lazily', () => {
    const states = Array.from(iterate(noisyHalve, { seed: 0, maxSteps: 2 }), (x) => x.state)
    expect(states[0]).toBe(16)
    expect(states).toHaveLength(3)
  })
})

describe('runs as data', () => {
  it('round-trip through JSON and compare', () => {
    const a = recorder()
    runAlgorithm(noisyHalve, { seed: 0, maxSteps: 10, tracer: a })
    const parsed = parseRun(exportRun(a.run()))
    expect(parsed).toEqual(a.run())
    expect(diffRuns(a.run(), parsed).config).toEqual({})
    expect(summary(parsed)['noisy-halve:0']?.status).toBe('converged')
    expect(() => parseRun('{"schema":"nope"}')).toThrow(/invalid run/)
  })

  it('traces single calls, including failures', () => {
    const rec = recorder()
    expect(() =>
      traceCall(
        'boom',
        () => {
          throw new Error('x')
        },
        { tracer: rec },
      ),
    ).toThrow('x')
    expect(rec.events.at(-1)).toMatchObject({ type: 'end', status: 'failed' })
  })

  it('schedules and taps', () => {
    expect([0, 1, 2, 3, 4].filter((s) => onSchedule({ firstAndLast: true }, s))).toEqual([0])
    const taps = tapContext(false)
    taps.tap('h', tensor([1]))
    expect(taps.entries.size).toBe(0)
  })
})
