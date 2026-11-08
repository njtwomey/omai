import { describe, expect, it } from 'vitest'
import { type Algorithm, iterate, runAlgorithm, runFor, snapshotOf } from './algorithm.ts'
import { checkpointEvery, resume, seek } from './checkpoint.ts'
import type { CheckpointEvent, RunStatus } from './events.ts'
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

describe('the three drivers end every run the same way', () => {
  // `done` is asked at every step, the limit included; a run that reaches the limit with no verdict is `completed`.
  const stopAt = (t: number, status: RunStatus): Algorithm<number> => ({
    ...count,
    done: (_s, at) => (at === t ? status : undefined),
  })
  const cases: readonly [string, Algorithm<number>, number, number, RunStatus][] = [
    ['verdict before the limit', stopAt(3, 'diverged'), 10, 3, 'diverged'],
    ['verdict at the limit', stopAt(3, 'converged'), 3, 3, 'converged'],
    ['no verdict at the limit', stopAt(99, 'converged'), 3, 3, 'completed'],
    ['verdict after the limit', stopAt(4, 'failed'), 3, 3, 'completed'],
    ['stopped at the start', stopAt(0, 'stopped'), 5, 0, 'stopped'],
    ['an empty budget', stopAt(99, 'converged'), 0, 0, 'completed'],
  ]
  it.each(cases)('%s', (_name, algorithm, maxSteps, steps, status) => {
    const states = Array.from(iterate(algorithm, { seed: 0, maxSteps }))
    expect(runAlgorithm(algorithm, { seed: 0, maxSteps })).toMatchObject({ steps, status, state: steps })
    expect(states.at(-1)).toEqual({ t: steps, state: steps, status })
    expect(states.length).toBe(steps + 1)
    expect(states.slice(0, -1).every((x) => x.status === undefined)).toBe(true)
    expect(runFor(algorithm, {}, { seed: 0, maxSteps })).toMatchObject({ t: steps, state: steps, status })
    // resumed from a stored state, the budget counts from there
    const rec = recorder()
    const resumed = runAlgorithm(algorithm, {
      seed: 0,
      maxSteps,
      tracer: rec,
      start: { state: 0, step: 0 },
    })
    expect(resumed).toMatchObject({ steps, status })
    expect(rec.events.at(-1)).toMatchObject({ type: 'end', status, steps })
  })
})

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
    expect(() => seek(count, { schema: 'omai.run/1', events: [] }, 3, {})).toThrow()
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

describe('state and mutation', () => {
  interface Frontier {
    readonly open: number[]
  }
  const search: Algorithm<Frontier> = {
    name: 'search',
    config: {},
    init: () => ({ open: [] }),
    step: (s, ctx) => {
      s.open.push(ctx.t)
      return s
    },
    snapshot: (s) => ({ open: [...s.open] }),
    serialise: (s) => ({ open: [...s.open] }),
    deserialise: (j) => ({ open: [...(j as { open: number[] }).open] }),
  }

  it('iterate yields the live object: an algorithm that mutates in place shows every kept state as the latest', () => {
    const states = Array.from(iterate(search, { maxSteps: 3 }), (x) => x.state)
    expect(states.every((s) => s === states[0])).toBe(true)
    expect(states[0]?.open).toEqual([0, 1, 2])
  })

  it('snapshotOf keeps each state as it was, and is the identity without a snapshot', () => {
    const kept = Array.from(iterate(search, { maxSteps: 3 }), (x) => snapshotOf(search, x.state).open)
    expect(kept).toEqual([[], [0], [0, 1], [0, 1, 2]])
    const plain = { ...search }
    delete plain.snapshot
    const live = { open: [1] }
    expect(snapshotOf(plain, live)).toBe(live)
  })
})
