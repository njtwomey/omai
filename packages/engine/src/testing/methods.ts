// Helpers the inference-method tests share: a fake clock, a resume check (a run restored from a serialised state equals an
// uninterrupted one), a cadence check for anytime snapshots and a cancellation check. Node only; not published.
import { expect } from 'vitest'
import type { Json } from '../lib/base/index.ts'
import { type InferenceMethod, runAnytime, type Snapshot } from '../lib/probprog/inference/index.ts'
import { type Config, session } from '../lib/trace/index.ts'

/** A clock that advances `by` milliseconds every time it is read (as the page's `performance.now` would, between steps). */
export function tickingClock(by: number): () => number {
  let now = 0
  return () => {
    now += by
    return now
  }
}

/** The numbers of a posterior's summary table of one method state, for comparing two runs. */
export function summaryOf<S, K extends string, A>(
  method: InferenceMethod<S, Config, K, A>,
  state: S,
  config: Config,
): Json {
  const p = method.posterior?.(state, config)
  if (p === undefined) throw new Error(`${method.name} has no posterior`)
  return JSON.parse(JSON.stringify(p.summary(), (_, v) => (typeof v === 'number' && !Number.isFinite(v) ? null : v)))
}

/**
 * A run restored from a serialised state equals an uninterrupted one: run to the end; run to `at`, serialise the state through
 * JSON, restore it into a session made afresh (its own algorithm, never initialised), continue to the same end; and fork the
 * uninterrupted run at a recorded step. All three end at the same step with the same posterior summary.
 */
export function expectResumeEqual<S, K extends string, A>(
  method: InferenceMethod<S, Config, K, A>,
  options: { readonly config?: Partial<Config>; readonly seed?: number; readonly steps: number; readonly at: number },
): void {
  const { config: partial, seed = 5, steps, at } = options
  const open = { ...(partial === undefined ? {} : { config: partial }), seed, steps, checkpointEvery: 1 }
  const whole = session(method, { ...open, mode: 'eager' })
  const cfg = whole.config
  const first = session(method, { ...open, mode: 'live' })
  first.advance({ steps: at })
  expect(first.t).toBe(at)
  const json = JSON.parse(JSON.stringify(first.algorithm.serialise?.(first.state) ?? null)) as Json
  const probe = session(method, { ...open, mode: 'live' })
  const restored = probe.algorithm.deserialise?.(json) as S
  const resumed = session(method, { ...open, mode: 'live', start: { state: restored, t: at } })
  resumed.advance()
  expect(resumed.t).toBe(whole.t)
  expect(resumed.status).toBe(whole.status)
  expect(summaryOf(method, resumed.state, cfg)).toEqual(summaryOf(method, whole.state, cfg))
  const forked = whole.fork(at)
  forked.advance()
  expect(forked.t).toBe(whole.t)
  expect(summaryOf(method, forked.state, cfg)).toEqual(summaryOf(method, whole.state, cfg))
}

/**
 * A run driven by `runAnytime` with a fake clock reports a snapshot per `everyMs` of clock time, each one a posterior that
 * reads (a summary table), the last flagged final, and stops at a time budget.
 */
export function expectSnapshotCadence<S, K extends string, A>(
  method: InferenceMethod<S, Config, K, A>,
  options: {
    readonly config?: Partial<Config>
    readonly steps: number
    readonly budgetMs: number
    readonly everyMs: number
    readonly tick: number
  },
): Snapshot<S>[] {
  const s = session(method, {
    ...(options.config === undefined ? {} : { config: options.config }),
    seed: 2,
    steps: options.steps,
  })
  const seen: Snapshot<S>[] = []
  const r = runAnytime(s, {
    budget: { ms: options.budgetMs },
    snapshot: { everyMs: options.everyMs },
    clock: tickingClock(options.tick),
    onSnapshot: (x) => seen.push({ ...x }),
  })
  expect(['budget', 'ended']).toContain(r.reason)
  expect(seen.at(-1)?.final).toBe(true)
  const times = seen.filter((x) => !x.final).map((x) => x.elapsedMs)
  // One snapshot per `everyMs` of clock time (a slice may overshoot by a step's worth of ticks).
  times.forEach((t, i) => {
    expect(t).toBeGreaterThanOrEqual(options.everyMs * (i + 1) - options.tick * 2)
    expect(t).toBeLessThan(options.everyMs * (i + 1) + options.tick * 50)
  })
  for (const x of seen) {
    const rows = summaryOf(method, x.state, s.config) as unknown[]
    expect(Array.isArray(rows)).toBe(true)
  }
  return seen
}

/** A run cancelled between slices stops at the step the caller said, and continues from there. */
export function expectCancelAndContinue<S, K extends string, A>(
  method: InferenceMethod<S, Config, K, A>,
  options: { readonly config?: Partial<Config>; readonly steps: number; readonly cancelAfterSlices: number },
): void {
  const s = session(method, {
    ...(options.config === undefined ? {} : { config: options.config }),
    seed: 4,
    steps: options.steps,
  })
  let slices = 0
  const r = runAnytime(s, { snapshot: { everySteps: 1 }, shouldStop: () => ++slices > options.cancelAfterSlices })
  expect(r.reason).toBe('cancelled')
  // One step per slice: the run stopped exactly where it was told to.
  expect(r.t).toBe(options.cancelAfterSlices)
  const more = runAnytime(s, { budget: { steps: 2 } })
  expect(more.t).toBe(options.cancelAfterSlices + 2)
}
