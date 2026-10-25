import { describe, expect, it } from 'vitest'
import { type Algorithm, runAlgorithm, runFor } from './algorithm.ts'
import { checkpointEvery, resume, seek } from './checkpoint.ts'
import type { CheckpointEvent } from './events.ts'
import { policies } from './policy.ts'
import { recorder, startRun } from './tracer.ts'

const count: Algorithm<number> = {
  name: 'count',
  config: {},
  init: () => 0,
  step: (s) => s + 1,
  done: (s) => (s >= 50 ? 'converged' : undefined),
  serialise: (s) => s,
  deserialise: (j) => j as number,
}

describe('runFor budgets', () => {
  it('stops on an injected clock, always taking at least one step', () => {
    let now = 0
    const clock = (): number => {
      now += 1
      return now
    }
    const slice = runFor(count, { ms: 3, clock }, { maxSteps: 100 })
    expect(slice.t).toBe(3)
    const one = runFor(count, { ms: 1, clock: () => 0 }, { maxSteps: 100, from: slice })
    expect(one.t).toBeGreaterThan(3)
  })

  it('ends with the algorithm status and returns a finished slice unchanged', () => {
    const done = runFor(count, {}, { maxSteps: 100 })
    expect(done).toMatchObject({ state: 50, t: 50, status: 'converged' })
    expect(runFor(count, { steps: 1 }, { maxSteps: 100, from: done })).toBe(done)
  })

  it('rejects empty budgets', () => {
    expect(() => runFor(count, { steps: 0 }, { maxSteps: 10 })).toThrow()
    expect(() => runFor(count, { ms: 0, clock: () => 0 }, { maxSteps: 10 })).toThrow()
  })

  it('records checkpoints into a run handle across slices', () => {
    const rec = recorder()
    const policy = checkpointEvery(4)
    const whole = recorder()
    runAlgorithm(count, { maxSteps: 10, tracer: whole, policy })
    const steps = (r: ReturnType<typeof recorder>): number[] =>
      r.events.filter((e): e is CheckpointEvent => e.type === 'checkpoint').map((e) => e.step)
    expect(steps(whole)).toEqual([0, 4, 8])
    const handle = startRun(rec, { name: 'count', seed: null, config: {}, policy })
    const first = runFor(count, { steps: 5 }, { maxSteps: 10, run: handle })
    runFor(count, { steps: 5 }, { maxSteps: 10, run: handle, from: first })
    expect(steps(rec)).toEqual([0, 4, 8])
  })
})

describe('checkpoints', () => {
  it('checkpointEvery validates and drives resume and seek', () => {
    expect(() => checkpointEvery(0)).toThrow()
    expect(() => checkpointEvery(1.5)).toThrow()
    expect(checkpointEvery(5, policies.standard)).toMatchObject({
      checkpoints: true,
      checkpointInterval: 5,
      roles: ['param', 'grad'],
    })
    const rec = recorder()
    runAlgorithm(count, { maxSteps: 20, tracer: rec, policy: checkpointEvery(5) })
    expect(seek(count, rec.run(), 13, {})).toBe(13)
    expect(() => seek(count, { schema: 'mdml.run/1', events: [] }, 3, {})).toThrow()
    const { deserialise: _, ...plain } = count
    expect(() => resume(plain, { type: 'checkpoint', runId: 'r', step: 1, state: 1 }, { maxSteps: 1 })).toThrow()
  })

  it('debug traces carry checkpoints; resumed runs do not repeat their starting checkpoint', () => {
    const rec = recorder()
    runAlgorithm(count, { maxSteps: 12, tracer: rec, policy: policies.debug })
    expect(rec.events.filter((e) => e.type === 'checkpoint').map((e) => (e as CheckpointEvent).step)).toEqual([0, 10])
    const again = recorder()
    resume(
      count,
      { type: 'checkpoint', runId: 'r', step: 10, state: 10 },
      { maxSteps: 12, tracer: again, policy: policies.debug },
    )
    expect(again.events.filter((e) => e.type === 'checkpoint').map((e) => (e as CheckpointEvent).step)).toEqual([20])
  })
})
