// The shared method convention: knobs with the budget and snapshot knobs, a typed contract, a capability verdict, and
// runAnytime driving a session within a budget with periodic snapshots and cancellation (a fake clock stands in for the
// page's, since the library reads none).
import { describe, expect, it } from 'vitest'
import { contains, defaults, dict, int } from '../../space/index.ts'
import { scalar, session } from '../../trace/index.ts'
import * as d from '../model/dists/index.ts'
import { analyseSpec, defineModel } from '../model/index.ts'
import {
  anytimeKnobs,
  defineInferenceMethod,
  runAnytime,
  runAnytimeAsync,
  type Snapshot,
  withAnytime,
} from './index.ts'

const counter = (ends?: number) =>
  defineInferenceMethod({
    name: 'count',
    family: 'exact',
    description: 'counts steps',
    config: dict({ extra: int(0, 10, { default: 3 }) }),
    make: () => ({
      name: 'count',
      config: {},
      init: () => 0,
      step: (s: number, ctx) => {
        ctx.log('n', s)
        return s + 1
      },
      ...(ends === undefined ? {} : { done: (s: number) => (s >= ends ? ('converged' as const) : undefined) }),
      serialise: (s: number) => s,
      deserialise: (j) => j as number,
    }),
    contract: { n: scalar() },
    check: (a) =>
      a.hasLikelihood
        ? { applies: true, reason: 'there is a likelihood', missing: [] }
        : { applies: false, reason: 'no observed nodes', missing: ['likelihood'], hint: 'add an observation' },
  })

describe('the convention', () => {
  it('adds the budget and snapshot knobs to a method, with their defaults', () => {
    const m = counter()
    expect(Object.keys(m.config.spaces)).toEqual(['extra', ...Object.keys(anytimeKnobs.spaces)])
    expect(defaults(m.config)).toMatchObject({
      extra: 3,
      budgetMs: 0,
      budgetSteps: 0,
      snapshotEveryMs: 1000,
      snapshotEverySteps: 0,
    })
    expect(contains(m.config, { ...(defaults(m.config) as object), budgetMs: 5000 })).toBe(true)
    expect(() => withAnytime(dict({ budgetMs: int(0, 1) }))).toThrow(/reserved/)
  })

  it('says whether a method applies to a model, and why not', () => {
    const none = analyseSpec(defineModel('prior', (b) => void b.latent('x', d.normal(0, 1))))
    const some = analyseSpec(
      defineModel('lik', (b) =>
        b.observe('y', d.normal(b.latent('x', d.normal(0, 1)), 1), b.data('y', { dims: ['n'] })),
      ),
    )
    expect(counter().check?.(none)).toEqual({
      applies: false,
      reason: 'no observed nodes',
      missing: ['likelihood'],
      hint: 'add an observation',
    })
    expect(counter().check?.(some).applies).toBe(true)
  })

  it('is a steppable, resumable method: sessions seek and fork', () => {
    const s = session(counter(), { steps: 20, mode: 'eager' })
    expect(s.t).toBe(20)
    expect(s.stateAt(7)).toBe(7)
    expect(s.fork(5).state).toBe(5)
  })
})

describe('runAnytime', () => {
  /** A clock that advances one millisecond per reading. */
  const ticking = (): (() => number) => {
    let now = 0
    return () => now++
  }

  it('stops at a step budget and hands out snapshots every few steps, the last flagged', () => {
    const seen: Snapshot<number>[] = []
    const s = session(counter(), { steps: 1000 })
    const r = runAnytime(s, { budget: { steps: 10 }, snapshot: { everySteps: 4 }, onSnapshot: (x) => seen.push(x) })
    expect(r).toMatchObject({ reason: 'budget', steps: 10, t: 10, snapshots: 3 })
    expect(seen.map((x) => [x.t, x.final])).toEqual([
      [4, false],
      [8, false],
      [10, true],
    ])
    expect(seen.map((x) => x.state)).toEqual([4, 8, 10])
  })

  it('reads its budget from the knobs of the session', () => {
    const s = session(counter(), { steps: 1000, config: { budgetSteps: 6, snapshotEverySteps: 3 } })
    const seen: number[] = []
    const r = runAnytime(s, { onSnapshot: (x) => seen.push(x.t) })
    expect([r.reason, r.t, seen]).toEqual(['budget', 6, [3, 6]])
  })

  it('stops at a time budget with a clock, snapshotting by time', () => {
    const seen: Snapshot<number>[] = []
    const s = session(counter(), { steps: 100_000 })
    const r = runAnytime(s, {
      budget: { ms: 40 },
      snapshot: { everyMs: 10 },
      clock: ticking(),
      onSnapshot: (x) => seen.push(x),
    })
    expect(r.reason).toBe('budget')
    expect(r.elapsedMs).toBeGreaterThanOrEqual(40)
    expect(seen.length).toBeGreaterThanOrEqual(3)
    expect(seen.at(-1)?.final).toBe(true)
    // Snapshots come in order, and strictly later in steps.
    const ts = seen.map((x) => x.t)
    expect(ts).toEqual([...ts].sort((a, b) => a - b))
  })

  it('needs a clock for a time budget', () => {
    expect(() => runAnytime(session(counter()), { budget: { ms: 5 } })).toThrow(/needs a clock/)
  })

  it('ends when the method finishes before the budget does', () => {
    const s = session(counter(5), { steps: 100 })
    const r = runAnytime(s, { budget: { steps: 50 } })
    expect(r).toMatchObject({ reason: 'ended', status: 'converged', t: 5 })
  })

  it('can be cancelled between slices, and continued', () => {
    const s = session(counter(), { steps: 1000 })
    let slices = 0
    const r = runAnytime(s, { snapshot: { everySteps: 5 }, shouldStop: () => ++slices > 3 })
    expect(r).toMatchObject({ reason: 'cancelled', t: 15 })
    const more = runAnytime(s, { budget: { steps: 5 } })
    expect([more.t, more.steps]).toEqual([20, 5])
  })
})

describe('runAnytimeAsync', () => {
  const ticking = (): (() => number) => {
    let now = 0
    return () => now++
  }

  it('hands the page the thread between slices and gives a snapshot per interval, the last flagged', async () => {
    let handovers = 0
    const seen: Snapshot<number>[] = []
    const s = session(counter(), { steps: 100_000 })
    const r = await runAnytimeAsync(s, {
      budget: { ms: 100 },
      snapshot: { everyMs: 30 },
      sliceMs: 10,
      clock: ticking(),
      yieldToPage: async () => {
        handovers += 1
      },
      onSnapshot: (x) => seen.push(x),
    })
    expect(r.reason).toBe('budget')
    expect(handovers).toBeGreaterThanOrEqual(5)
    expect(seen.length).toBeGreaterThanOrEqual(3)
    expect(seen.at(-1)?.final).toBe(true)
    expect(seen.slice(0, -1).every((x) => !x.final)).toBe(true)
  })

  it('reaches the state of runAnytime and of an uninterrupted session', async () => {
    const whole = session(counter(), { steps: 37, mode: 'eager' })
    const s = session(counter(), { steps: 37 })
    const r = await runAnytimeAsync(s, {
      snapshot: { everySteps: 5 },
      yieldToPage: () => Promise.resolve(),
    })
    expect([r.reason, r.t, s.state]).toEqual(['ended', whole.t, whole.state])
  })

  it('stops at a step budget with snapshots every few steps, as runAnytime does', async () => {
    const seen: number[] = []
    const r = await runAnytimeAsync(session(counter(), { steps: 1000 }), {
      budget: { steps: 10 },
      snapshot: { everySteps: 4 },
      yieldToPage: () => Promise.resolve(),
      onSnapshot: (x) => seen.push(x.t),
    })
    expect([r.reason, r.t, seen]).toEqual(['budget', 10, [4, 8, 10]])
  })

  it('asks shouldStop after every slice, and can be continued', async () => {
    const s = session(counter(), { steps: 1000 })
    let asked = 0
    const r = await runAnytimeAsync(s, {
      snapshot: { everySteps: 5 },
      yieldToPage: () => Promise.resolve(),
      shouldStop: () => ++asked > 3,
    })
    expect(r).toMatchObject({ reason: 'cancelled', t: 15 })
    const more = await runAnytimeAsync(s, { budget: { steps: 5 }, yieldToPage: () => Promise.resolve() })
    expect([more.t, more.steps]).toEqual([20, 5])
  })

  it('needs a clock for a time budget or a slice length', async () => {
    const yieldToPage = (): Promise<void> => Promise.resolve()
    await expect(runAnytimeAsync(session(counter()), { budget: { ms: 5 }, yieldToPage })).rejects.toThrow(
      /needs a clock/,
    )
    await expect(runAnytimeAsync(session(counter()), { sliceMs: 5, yieldToPage })).rejects.toThrow(/needs a clock/)
  })
})
