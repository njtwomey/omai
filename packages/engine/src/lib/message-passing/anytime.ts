/**
 * Anytime message passing: running a graph under a time budget the caller chooses, with a snapshot of the current
 * marginals about once a second, and cancellation between any two steps. The library never reads a clock, so the
 * clock is passed in (the browser's `performance.now`); the budget and the snapshot interval only decide where the run
 * stops and when it reports, never what it computes, so any slicing of a run reaches the same states.
 */

import { OmaiError } from '../base/index.ts'
import { type Algorithm, initialSlice, type RunHandle, type RunStatus, runFor, type Slice } from '../trace/index.ts'
import { type MpSnapshot, type MpState, mpSnapshot } from './engine.ts'
import type { CompiledGraph } from './graph.ts'

/** Options of {@link runAnytime}; `S` is the state of the algorithm and `P` the snapshot it is reported as. */
export interface AnytimeOptions<S = MpState, P = MpSnapshot> {
  /** The browser budget in milliseconds: the run stops, resumable, when it has used this much of the clock. */
  readonly budgetMs: number
  /** The clock, in milliseconds (injected by the app: the browser's high-resolution timer). */
  readonly clock: () => number
  /** The step cap of the algorithm's own budget (default 1,000,000). */
  readonly maxSteps?: number
  /** How often to report a snapshot (default 1000 ms); the first and the last are always reported. */
  readonly snapshotEveryMs?: number
  /** Called with each snapshot. */
  readonly onSnapshot?: (snapshot: P, info: { readonly t: number; readonly elapsedMs: number }) => void
  /** Asked between steps: true stops the run (the state stays resumable). */
  readonly isCancelled?: () => boolean
  readonly seed?: number
  /** A run to record into (snapshots are recorded as `mp/snapshot` messages, and the run is ended when it finishes). */
  readonly run?: RunHandle
  /** Continue from where an earlier call stopped. */
  readonly from?: Slice<S>
}

/** How an anytime run ended: the algorithm's own status, or why it was stopped. */
export type AnytimeStatus = RunStatus | 'budget' | 'cancelled'

/** The result of {@link runAnytime}. */
export interface AnytimeResult<S = MpState, P = MpSnapshot> {
  readonly state: S
  /** Steps taken in total. */
  readonly t: number
  readonly status: AnytimeStatus
  /** The final snapshot. */
  readonly snapshot: P
  /** How many snapshots were reported. */
  readonly snapshots: number
  /** Where to resume from (a stopped run continues exactly where it left off). */
  readonly slice: Slice<S>
}

/**
 * Run a message-passing algorithm one step at a time under a time budget: after every step it asks whether the run is
 * cancelled, reports a snapshot of the marginals when `snapshotEveryMs` has passed since the last, and stops when the
 * budget is used, the algorithm is done, or the step cap is reached. A step is one message, one factor or one sweep by
 * the algorithm's granularity, so the cancellation and the report lag by at most one step.
 *
 * @param g - The compiled graph the algorithm runs.
 * @param algorithm - A message-passing algorithm of that graph.
 * @param options - The budget, the clock and the callbacks.
 * @returns The state, the status (`converged`, `completed`, `budget` or `cancelled`...), the last snapshot, and a slice to resume from.
 * @throws OmaiError with code `value` for a budget or interval that is not positive.
 * @concept message-passing
 * @index function Run message passing under a time budget with periodic marginals snapshots.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, epAlgorithm, gaussianPrior, runAnytime } from 'omai-engine/message-passing'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [gaussianPrior('x', 0, 1)] }, defaultOperators())
 * let now = 0
 * const seen: number[] = []
 * const r = runAnytime(g, epAlgorithm(g), { budgetMs: 1000, clock: () => (now += 1), onSnapshot: (s) => seen.push(s.sweeps) })
 * [r.status, seen.length >= 1] // => ['converged', true]
 * ```
 */
export function runAnytime(g: CompiledGraph, algorithm: Algorithm<MpState>, options: AnytimeOptions): AnytimeResult {
  return runAnytimeWith((state) => mpSnapshot(g, state), algorithm, options)
}

/**
 * The anytime loop of {@link runAnytime} for any algorithm whose state has a snapshot: the budget, the clock, the
 * cancellation, the periodic report and the slice to resume from are the same, and `snapshotOf` says what the state is
 * reported as. The variants of message passing with a state of their own (stochastic EP, region graphs, AMP) use it.
 *
 * @param snapshotOf - The snapshot of a state: a function of the state alone.
 * @param algorithm - The algorithm.
 * @param options - The budget, the clock and the callbacks.
 * @returns The state, the status, the last snapshot, and a slice to resume from.
 * @throws OmaiError with code `value` for a budget or interval that is not positive.
 * @concept message-passing
 * @index function The anytime loop for any state with a snapshot: budget, clock, cancellation, resume.
 * @example
 * ```ts
 * import { runAnytimeWith } from 'omai-engine/message-passing'
 * import type { Algorithm } from 'omai-engine/trace'
 * const count: Algorithm<number> = { name: 'count', config: {}, init: () => 0, step: (n) => n + 1, done: (n) => (n >= 3 ? 'completed' : undefined) }
 * let now = 0
 * runAnytimeWith((n: number) => ({ n }), count, { budgetMs: 100, clock: () => (now += 1) }).snapshot // => { n: 3 }
 * ```
 */
export function runAnytimeWith<S, P>(
  snapshotOf: (state: S) => P,
  algorithm: Algorithm<S>,
  options: AnytimeOptions<S, P>,
): AnytimeResult<S, P> {
  const every = options.snapshotEveryMs ?? 1000
  if (!(options.budgetMs > 0) || !(every > 0))
    throw new OmaiError('value', 'runAnytime: the budget and the snapshot interval must be positive')
  const maxSteps = options.maxSteps ?? 1_000_000
  const started = options.clock()
  let lastReport = Number.NEGATIVE_INFINITY
  let reports = 0
  const report = (slice: Slice<S>, now: number): void => {
    const snap = snapshotOf(slice.state)
    reports++
    lastReport = now
    options.onSnapshot?.(snap, { t: slice.t, elapsedMs: now - started })
    options.run?.message(slice.t, 'mp/snapshot', snap as never)
  }
  const base = {
    maxSteps,
    ...(options.seed === undefined ? {} : { seed: options.seed }),
    ...(options.run === undefined ? {} : { run: options.run }),
  }
  let slice: Slice<S> = options.from ?? initialSlice(algorithm, base)
  let status: AnytimeStatus | undefined = slice.status
  if (status === undefined) report(slice, started)
  while (status === undefined) {
    if (options.isCancelled?.() === true) {
      status = 'cancelled'
      break
    }
    slice = runFor(algorithm, { steps: 1 }, { ...base, from: slice })
    const now = options.clock()
    if (slice.status !== undefined) {
      status = slice.status
      break
    }
    if (now - lastReport >= every) report(slice, now)
    if (now - started >= options.budgetMs) status = 'budget'
  }
  report(slice, options.clock())
  if (options.run !== undefined && slice.status !== undefined) options.run.end(slice.status, slice.t)
  return { state: slice.state, t: slice.t, status, snapshot: snapshotOf(slice.state), snapshots: reports, slice }
}
