import { describe, expect, it } from 'vitest'
import { dict } from '../space/index.ts'
import { type Algorithm, iterate, runAlgorithm, runFor } from './algorithm.ts'
import { checkpointEvery } from './checkpoint.ts'
import { scalar } from './contract.ts'
import type { TraceEvent } from './events.ts'
import { filtered, isRun, mergeRuns, recorder, replay, startRun } from './index.ts'
import { defineMethod, session } from './session.ts'

// A stochastic algorithm: each step adds a number derived from its step key, so a step run under another key, in another
// order or twice gives another state. It finishes at step 9 with the status `diverged`, and observes the state.
const seenKeys: string[] = []
const drift = (log: string[] | undefined): Algorithm<number> => ({
  name: 'drift',
  config: {},
  init: (k) => (k.words[0] % 7) + (k.words[1] % 5) * 0.25,
  step: (s, ctx) => {
    log?.push(`${ctx.t}:${ctx.key.words.join(',')}`)
    return s + ((ctx.key.words[0] % 1000) + (ctx.key.words[1] % 1000)) / 1000
  },
  done: (_s, t) => (t === 9 ? 'diverged' : undefined),
  observe: (s) => ({ metrics: { x: s } }),
  serialise: (s) => s,
  deserialise: (j) => j as number,
})
const method = defineMethod<number, Record<string, never>>({
  name: 'drift',
  config: dict({}),
  make: () => drift(seenKeys),
  contract: { x: scalar() },
})

const OPTIONS = { seed: 3, maxSteps: 12 }
const metricsOf = (events: readonly TraceEvent[]): string[] =>
  events.filter((e) => e.type === 'metric').map((e) => `${e.step}:${e.name}=${e.value}`)

describe('the four drivers compute one run', () => {
  it('draw the same step keys and reach the same states and the same status', () => {
    const logs = { run: [] as string[], iterate: [] as string[], slices: [] as string[] }
    const whole = runAlgorithm(drift(logs.run), OPTIONS)
    const steps = Array.from(iterate(drift(logs.iterate), OPTIONS))
    let slice = runFor(drift(logs.slices), { steps: 4 }, OPTIONS)
    const stops: number[] = [slice.t]
    while (slice.status === undefined) {
      slice = runFor(drift(logs.slices), { steps: 4 }, { ...OPTIONS, from: slice })
      stops.push(slice.t)
    }
    seenKeys.length = 0
    const live = session(method, { seed: 3, steps: 12, mode: 'live' })
    live.advance({ steps: 5 })
    live.advance()
    const eager = session(method, { seed: 3, steps: 12, mode: 'eager' })

    // the status and the step at which `done` fired (step 9, before the budget of 12)
    expect([whole.steps, whole.status]).toEqual([9, 'diverged'])
    expect(steps.at(-1)).toMatchObject({ t: 9, status: 'diverged' })
    expect([slice.t, slice.status, stops]).toEqual([9, 'diverged', [4, 8, 9]])
    expect([live.t, live.status, eager.t, eager.status]).toEqual([9, 'diverged', 9, 'diverged'])
    // the states agree, bit for bit
    expect(steps.map((x) => x.state).at(-1)).toBe(whole.state)
    expect(slice.state).toBe(whole.state)
    expect([live.state, eager.state]).toEqual([whole.state, whole.state])
    // every state along the way: the lazy iterator, and the session's checkpoints re-run forward
    for (const [t, x] of steps.entries()) {
      expect(live.stateAt(t), `step ${t}`).toBe(x.state)
    }
    // the keys: one per step, `foldIn(stepsKey, t)`, the same whichever driver took the step
    expect(logs.run).toHaveLength(9)
    expect(logs.iterate).toEqual(logs.run)
    expect(logs.slices).toEqual(logs.run)
    expect(seenKeys).toEqual([...logs.run, ...logs.run])
    expect(new Set(logs.run.map((s) => s.split(':')[1])).size).toBe(9)
  })

  it('record the same tape (metrics, checkpoints and end), and the trace helpers replay, mergeRuns, isRun and filtered work on it', () => {
    const policy = checkpointEvery(4)
    const rec = recorder()
    runAlgorithm(drift(undefined), { ...OPTIONS, tracer: rec, policy })
    const sliced = recorder()
    const run = startRun(sliced, { name: 'drift', seed: 3, config: {}, policy })
    let slice = runFor(drift(undefined), { steps: 4 }, { ...OPTIONS, run })
    while (slice.status === undefined) slice = runFor(drift(undefined), { steps: 4 }, { ...OPTIONS, from: slice, run })
    run.end(slice.status, slice.t)

    const kinds = (events: readonly TraceEvent[]): string[] =>
      events.filter((e) => e.type !== 'run').map((e) => `${e.type}@${e.step ?? ''}`)
    const checkpoints = (events: readonly TraceEvent[]): number[] =>
      events.filter((e) => e.type === 'checkpoint').map((e) => e.step)
    // one metric per observed step (every step up to the verdict), checkpoints at the interval and the last step
    expect(metricsOf(rec.events)).toHaveLength(10)
    expect(checkpoints(rec.events)).toEqual([0, 4, 8])
    expect(rec.events.at(-1)).toMatchObject({ type: 'end', status: 'diverged', steps: 9 })
    expect(metricsOf(sliced.events)).toEqual(metricsOf(rec.events))
    expect(checkpoints(sliced.events)).toEqual(checkpoints(rec.events))
    expect(kinds(sliced.events)).toEqual(kinds(rec.events))
    // a session records the same observed values on its tape
    const tape = session(method, { seed: 3, steps: 12, mode: 'eager', logs: 'all' }).run()
    expect(metricsOf(tape.events)).toEqual(metricsOf(rec.events))

    const a = rec.run()
    const seen: string[] = []
    replay(a, { metric: (e) => seen.push(`${e.step}`), end: (e) => seen.push(`end ${e.steps}`) })
    expect(seen).toEqual([...Array.from({ length: 10 }, (_, i) => `${i}`), 'end 9'])
    const merged = mergeRuns([a, a])
    expect(merged.events).toHaveLength(2 * a.events.length)
    expect(isRun(merged)).toBe(true)
    expect([isRun({ schema: 'mdml.run/1' }), isRun(null), isRun({ schema: 'other', events: [] })]).toEqual([
      false,
      false,
      false,
    ])
    const inner = recorder()
    runAlgorithm(drift(undefined), { ...OPTIONS, tracer: filtered(inner, (e) => e.type === 'metric') })
    expect(inner.events.every((e) => e.type === 'metric')).toBe(true)
    expect(inner.events).toHaveLength(10)
  })
})
