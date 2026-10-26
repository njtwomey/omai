/**
 * The `Algorithm` protocol, the one shape for iterative methods (optimisers, training loops, samplers, EM, message
 * passing, solvers, searches, agents): an `init` and a `step` over an explicit state. `runAlgorithm` runs one to
 * completion while recording a traced run, `iterate` steps through it lazily, and `runFor` advances it in time-budgeted
 * slices for interactive pages.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { Value } from '../tensor/index.ts'
import type { RunStatus, TensorRole } from './events.ts'
import { defaultLogContext, type LogContext } from './logging.ts'
import { shouldRecord, type TracePolicy } from './policy.ts'
import type { TraceSchema } from './schema.ts'
import { nullTracer, type RunHandle, startRun, type Tracer } from './tracer.ts'

/** What an algorithm sees at each step. */
export interface StepContext extends Pick<LogContext, 'log' | 'wants' | 'emit' | 'taps'> {
  readonly t: number
  /** A key derived from the run's key and `t`: the same at step `t` however the run was paused or resumed. */
  readonly key: Key
  /** The run, for emitting extra events or starting child runs of inner algorithms. */
  readonly run: RunHandle
}

/**
 * The context of a step: the step, its key, the run, and the log context (the run's own by default, a session's
 * when it drives the algorithm).
 *
 * @param t - The step.
 * @param key - The step's key.
 * @param run - The run.
 * @param logs - The log context (default the run's).
 * @returns The context.
 * @example
 * ```ts
 * import { nullTracer, startRun, stepContext } from 'mdml-engine/trace'
 * import { key } from 'mdml-engine/random'
 * stepContext(3, key(0), startRun(nullTracer, { name: 'x', seed: 0, config: {} })).t // => 3
 * ```
 */
export function stepContext(t: number, key: Key, run: RunHandle, logs?: LogContext): StepContext {
  const l = logs ?? defaultLogContext(run, t)
  return { t, key, run, log: l.log, wants: l.wants, emit: l.emit, taps: l.taps }
}

/** What an algorithm reports about a state when it is observed. */
export interface Observation {
  readonly metrics?: Readonly<Record<string, number>>
  readonly tensors?: Readonly<
    Record<
      string,
      { readonly role: TensorRole; readonly value: Value | ArrayLike<number>; readonly shape?: readonly number[] }
    >
  >
  readonly messages?: readonly {
    readonly kind: string
    readonly payload: Json
    readonly from?: string
    readonly to?: string
  }[]
  readonly frame?: Json
}

/**
 * The one protocol for iterative methods: optimisers, training loops, samplers, EM, message passing, solvers, searches,
 * agents. An algorithm is plain data plus pure functions over an explicit state; {@link runAlgorithm} drives it and
 * records what the caller's policy asks for. `observe` is only called when someone is recording, so tracing never
 * slows an untraced run.
 *
 * @concept algorithm-protocol
 */
export interface Algorithm<S> {
  readonly name: string
  readonly config: Json
  readonly schema?: TraceSchema
  init(key: Key): S
  step(state: S, ctx: StepContext): S
  /** A status to stop with, or `undefined` to continue. */
  done?(state: S, t: number): RunStatus | undefined
  observe?(state: S, t: number): Observation
  /** Serialise a state for checkpoints (and restore it), enabling resume and seek. */
  serialise?(state: S): Json
  deserialise?(json: Json): S
}

/** Options for {@link runAlgorithm}. */
export interface RunOptions {
  readonly maxSteps: number
  readonly seed?: number
  readonly key?: Key
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
  readonly runId?: string
  /** Run as a child of another run (an inner algorithm); the tracer and policy come from the parent. */
  readonly parent?: RunHandle
  /** Start from a stored state at a step instead of `init` (used by resume and seek). */
  readonly start?: { readonly state: unknown; readonly step: number; readonly serialised?: boolean }
}

/** The outcome of {@link runAlgorithm}. */
export interface RunResult<S> {
  readonly state: S
  readonly steps: number
  readonly status: RunStatus
  readonly runId: string
}

function rootKey(options: Pick<RunOptions, 'key' | 'seed'>): Key {
  return options.key ?? makeKey(options.seed ?? 0)
}

function record<S>(algorithm: Algorithm<S>, run: RunHandle, state: S, t: number): Readonly<Record<string, number>> {
  if (algorithm.observe === undefined) return {}
  const o = algorithm.observe(state, t)
  for (const [name, value] of Object.entries(o.metrics ?? {})) run.metric(t, name, value)
  for (const [name, x] of Object.entries(o.tensors ?? {})) run.tensor(t, name, x.role, x.value, x.shape)
  for (const m of o.messages ?? []) {
    run.message(t, m.kind, m.payload, {
      ...(m.from === undefined ? {} : { from: m.from }),
      ...(m.to === undefined ? {} : { to: m.to }),
    })
  }
  if (o.frame !== undefined) run.frame(t, o.frame)
  return o.metrics ?? {}
}

function checkpoint<S>(algorithm: Algorithm<S>, run: RunHandle, state: S, t: number): void {
  if (run.enabled && algorithm.serialise !== undefined && shouldRecord(run.policy, { kind: 'checkpoint' }, t)) {
    run.checkpoint(t, algorithm.serialise(state))
  }
}

/**
 * Run an {@link Algorithm}, recording a traced run.
 *
 * Step keys are `foldIn(stepsKey, t)`, so step `t` draws the same numbers however the run was paused, resumed or
 * replayed. The state is observed on the policy's schedule and always at the last step; when the policy records
 * checkpoints with an interval (see `checkpointEvery`) and the algorithm serialises, a checkpoint is stored at every
 * multiple of the interval.
 *
 * @param algorithm - The algorithm.
 * @param options - Step budget, seed or key, tracer and policy (or a parent run), and a start state.
 * @returns The final state, steps taken, status and run id.
 * @concept algorithm-protocol
 * @example
 * ```ts
 * import { runAlgorithm, recorder } from 'mdml-engine/trace'
 * const halve = {
 *   name: 'halve',
 *   config: {},
 *   init: () => 8,
 *   step: (x: number) => x / 2,
 *   done: (x: number) => (x < 1 ? ('converged' as const) : undefined),
 *   observe: (x: number) => ({ metrics: { x } }),
 * }
 * const rec = recorder()
 * runAlgorithm(halve, { seed: 0, maxSteps: 10, tracer: rec }).steps // => 4
 * rec.run() // => trace
 * ```
 */
export function runAlgorithm<S>(algorithm: Algorithm<S>, options: RunOptions): RunResult<S> {
  const info = {
    name: algorithm.name,
    seed: options.seed ?? null,
    config: algorithm.config,
    ...(options.runId === undefined ? {} : { runId: options.runId }),
    ...(algorithm.schema === undefined ? {} : { schema: algorithm.schema }),
  }
  const run =
    options.parent !== undefined
      ? options.parent.child({ ...info, ...(options.policy === undefined ? {} : { policy: options.policy }) })
      : startRun(options.tracer ?? nullTracer, {
          ...info,
          ...(options.policy === undefined ? {} : { policy: options.policy }),
        })
  const root = rootKey(options)
  const stepsKey = foldIn(root, 1)
  const restore = options.start
  let state: S =
    restore === undefined
      ? algorithm.init(foldIn(root, 0))
      : restore.serialised === true && algorithm.deserialise !== undefined
        ? algorithm.deserialise(restore.state as Json)
        : (restore.state as S)
  let t = restore?.step ?? 0
  let last: Readonly<Record<string, number>> = {}
  let observedAt = -1
  const observe = (): void => {
    if (!run.enabled) return
    last = record(algorithm, run, state, t)
    observedAt = t
  }
  if (run.due(t)) observe()
  if (restore === undefined) checkpoint(algorithm, run, state, t)
  let status: RunStatus = 'completed'
  const limit = (restore?.step ?? 0) + options.maxSteps
  while (t < limit) {
    const stop = algorithm.done?.(state, t)
    if (stop !== undefined) {
      status = stop
      break
    }
    state = algorithm.step(state, stepContext(t, foldIn(stepsKey, t), run))
    t += 1
    if (run.due(t)) observe()
    checkpoint(algorithm, run, state, t)
  }
  if (status === 'completed') status = algorithm.done?.(state, t) ?? 'completed'
  if (observedAt !== t) observe()
  run.end(status, t, last)
  return { state, steps: t, status, runId: run.runId }
}

/**
 * Step through an algorithm lazily, yielding every state (for live figures and for stepping by hand). Nothing is
 * recorded.
 *
 * @param algorithm - The algorithm.
 * @param options - Seed or key and a step budget.
 * @returns A generator of `{ t, state, status }`; `status` is set on the last item.
 * @example
 * ```ts
 * import { iterate } from 'mdml-engine/trace'
 * const count = { name: 'count', config: {}, init: () => 0, step: (s: number) => s + 1 }
 * Array.from(iterate(count, { maxSteps: 3 }), (x) => x.state) // => [0, 1, 2, 3]
 * ```
 */
export function* iterate<S>(
  algorithm: Algorithm<S>,
  options: Pick<RunOptions, 'seed' | 'key' | 'maxSteps'>,
): Generator<{ readonly t: number; readonly state: S; readonly status?: RunStatus }> {
  const root = rootKey(options)
  const stepsKey = foldIn(root, 1)
  const run = startRun(nullTracer, { name: algorithm.name, seed: options.seed ?? null, config: algorithm.config })
  let state = algorithm.init(foldIn(root, 0))
  let t = 0
  for (;;) {
    const stop = t >= options.maxSteps ? 'completed' : algorithm.done?.(state, t)
    if (stop !== undefined) {
      yield { t, state, status: stop }
      return
    }
    yield { t, state }
    state = algorithm.step(state, stepContext(t, foldIn(stepsKey, t), run))
    t += 1
  }
}

/** Where a sliced run stands: the state at step `t`, and its status once it has finished. */
export interface Slice<S> {
  readonly state: S
  readonly t: number
  readonly status?: RunStatus
  /** The last step whose state was recorded (so the next slice does not record it twice). */
  readonly observedAt?: number
}

/**
 * How much one slice may do: at most `steps` steps, and, with an injected `clock` in milliseconds (the browser timer
 * passed in by the app, say), at most `ms` milliseconds. The library never reads a clock itself, because traces must be
 * deterministic; a time budget only changes where slices end, never what the run computes.
 */
export interface SliceBudget {
  readonly steps?: number
  readonly ms?: number
  readonly clock?: () => number
}

/**
 * The slice a run starts from: the initial state, observed and checkpointed as `runAlgorithm` would, without taking a
 * step (so a session can stand at step 0 until asked to advance).
 *
 * @param algorithm - The algorithm.
 * @param options - Seed or key, and a run handle to record into.
 * @returns The slice at step 0.
 * @example
 * ```ts
 * import { initialSlice, runFor, type Algorithm } from 'mdml-engine/trace'
 * const count: Algorithm<number> = { name: 'count', config: {}, init: () => 0, step: (s) => s + 1 }
 * const first = initialSlice(count, {})
 * [first.t, runFor(count, { steps: 2 }, { maxSteps: 10, from: first }).state] // => [0, 2]
 * ```
 */
export function initialSlice<S>(
  algorithm: Algorithm<S>,
  options: Pick<RunOptions, 'seed' | 'key'> & { readonly run?: RunHandle },
): Slice<S> {
  const root = rootKey(options)
  const run = options.run ?? startRun(nullTracer, { name: algorithm.name, seed: null, config: algorithm.config })
  const state = algorithm.init(foldIn(root, 0))
  let observedAt = -1
  if (run.enabled && run.due(0)) {
    record(algorithm, run, state, 0)
    observedAt = 0
  }
  checkpoint(algorithm, run, state, 0)
  return { state, t: 0, observedAt }
}

/**
 * Advance an algorithm by one budgeted slice, for long runs on interactive pages: run a slice, yield to the event loop,
 * run the next. Slices continue exactly where the last stopped (step keys are `foldIn(stepsKey, t)`, as in
 * {@link runAlgorithm}), so any sequence of slices computes the same states as one uninterrupted run. With a `run`
 * handle, states are recorded on the policy's schedule and at the end, exactly as `runAlgorithm` records them; the
 * caller ends the run when a slice comes back with a status.
 *
 * @param algorithm - The algorithm.
 * @param budget - Steps and, with an injected clock, milliseconds for this slice (at least one step is taken).
 * @param options - Seed or key and the total step budget as for `runAlgorithm`, the slice to continue from, and a run
 * handle to record into.
 * @returns The slice reached; `status` is set once the algorithm is done or the total budget is used up.
 * @throws MdmlError with code `value` when the budget is not positive.
 * @concept algorithm-protocol
 * @example
 * ```ts
 * import { runFor, type Algorithm } from 'mdml-engine/trace'
 * const count: Algorithm<number> = { name: 'count', config: {}, init: () => 0, step: (s) => s + 1 }
 * const first = runFor(count, { steps: 4 }, { maxSteps: 10 })
 * first.state // => 4
 * const second = runFor(count, { steps: 100 }, { maxSteps: 10, from: first })
 * [second.state, second.status] // => [10, 'completed']
 * ```
 */
export function runFor<S>(
  algorithm: Algorithm<S>,
  budget: SliceBudget,
  options: Pick<RunOptions, 'seed' | 'key' | 'maxSteps'> & { readonly from?: Slice<S>; readonly run?: RunHandle },
): Slice<S> {
  const steps = budget.steps ?? Number.POSITIVE_INFINITY
  if (!(steps > 0) || (budget.ms !== undefined && !(budget.ms > 0))) {
    throw new MdmlError('value', 'runFor: the step and time budgets must be positive')
  }
  const root = rootKey(options)
  const stepsKey = foldIn(root, 1)
  const run = options.run ?? startRun(nullTracer, { name: algorithm.name, seed: null, config: algorithm.config })
  const from = options.from
  if (from?.status !== undefined) return from
  let state = from === undefined ? algorithm.init(foldIn(root, 0)) : from.state
  let t = from?.t ?? 0
  let observedAt = from?.observedAt ?? -1
  const observe = (): void => {
    if (!run.enabled) return
    record(algorithm, run, state, t)
    observedAt = t
  }
  if (from === undefined) {
    if (run.due(t)) observe()
    checkpoint(algorithm, run, state, t)
  }
  const started = budget.clock?.()
  const outOfTime = (): boolean =>
    budget.clock !== undefined && budget.ms !== undefined && budget.clock() - (started ?? 0) >= budget.ms
  let taken = 0
  let status: RunStatus | undefined
  for (;;) {
    status = t >= options.maxSteps ? (algorithm.done?.(state, t) ?? 'completed') : algorithm.done?.(state, t)
    if (status !== undefined || taken >= steps || (taken > 0 && outOfTime())) break
    state = algorithm.step(state, stepContext(t, foldIn(stepsKey, t), run))
    t += 1
    taken += 1
    if (run.due(t)) observe()
    checkpoint(algorithm, run, state, t)
  }
  if (status !== undefined && observedAt !== t) observe()
  return { state, t, observedAt, ...(status === undefined ? {} : { status }) }
}
