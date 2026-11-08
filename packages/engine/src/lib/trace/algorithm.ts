/**
 * The `Algorithm` protocol, the one shape for iterative methods (optimisers, training loops, samplers, EM, message
 * passing, solvers, searches, agents): an `init` and a `step` over an explicit state. `runAlgorithm` runs one to
 * completion while recording a traced run, `iterate` steps through it lazily, and `runFor` advances it in time-budgeted
 * slices for interactive pages. The three are loops over one stepper, which owns the keys of a run (init, steps), the
 * order of observing and checkpointing, and the rule that ends it, so a run ends the same way whichever way it is
 * driven. `done` is asked at every step, the last included, and its verdict ends the run with that `RunStatus`
 * (`converged`, `diverged`, `stopped`, `failed`); a run that reaches its step budget with no verdict ends `completed`.
 * So `completed` means the budget ran out, and an algorithm whose `done` fires at the budget's last step ends with its
 * own status, not `completed`, in all three.
 *
 * @index structure The Algorithm protocol (init and step) every iterative method implements.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { Value } from '../tensor/index.ts'
import { keyTags } from './_keys.ts'
import type { RunStatus, TensorRole } from './events.ts'
import { defaultLogContext, type LogContext } from './logging.ts'
import { shouldRecord, type TracePolicy } from './policy.ts'
import type { TraceSchema } from './schema.ts'
import { nullTracer, type RunHandle, startRun, type Tracer } from './tracer.ts'

/**
 * What an algorithm sees at each step. The parameter is the type of the paths it may log (`string` by default: any
 * path; a union of literals when the algorithm is typed by its method's contract, see {@link defineContract}).
 */
export interface StepContext<K extends string = string> extends Pick<LogContext<K>, 'log' | 'wants' | 'emit' | 'taps'> {
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
 * import { nullTracer, startRun, stepContext } from 'omai-engine/trace'
 * import { key } from 'omai-engine/random'
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
 * The second parameter is the type of the paths the algorithm logs (`ctx.log`, `ctx.emit`, `ctx.wants`): `string`
 * unless its method's contract declares them, when only those compile. An `Algorithm<S, 'a' | 'b'>` can be given
 * wherever an `Algorithm<S>` is wanted, and a runner hands it a context that accepts any path.
 *
 * ## State and mutation
 *
 * A state does not have to be immutable: the library's own searches (`search/_best-first.ts`) update their frontier in
 * place, and that is fine. The rule is about where a view reads from. What a view draws at a step comes from what was
 * recorded when the step happened: `observe` (metrics, tensors, frame, messages) or `ctx.log` and `ctx.emit` at emit
 * time, which are copied onto the tape then. `session.state` (and the `state` of `iterate` and `runFor`) is the live
 * object, the one the next step goes on to change, so a drawing that holds on to it shows the latest state at every
 * frame. `session.stateAt(t)` is exact through checkpoints, for an algorithm that `serialise`s. A consumer that keeps
 * states of its own across steps (a history for stepping back, as `useLiveRun` does) calls `snapshot` on each, and an
 * algorithm that mutates its state in place gives it, so each kept frame is the state of its own step.
 *
 * @concept algorithm-protocol
 * @example
 * ```ts
 * import { iterate, snapshotOf, type Algorithm } from 'omai-engine/trace'
 * // The frontier is changed in place, so the states an iteration yields are one object; a snapshot copies it.
 * const frontier: Algorithm<{ open: number[] }> = {
 *   name: 'frontier',
 *   config: {},
 *   init: () => ({ open: [] }),
 *   step: (s, ctx) => {
 *     s.open.push(ctx.t)
 *     return s
 *   },
 *   snapshot: (s) => ({ open: [...s.open] }),
 * }
 * const kept = Array.from(iterate(frontier, { maxSteps: 3 }), (x) => snapshotOf(frontier, x.state).open)
 * kept // => [[], [0], [0, 1], [0, 1, 2]]
 * ```
 */
export interface Algorithm<S, K extends string = string> {
  readonly name: string
  readonly config: Json
  readonly schema?: TraceSchema
  init(key: Key): S
  step(state: S, ctx: StepContext<K>): S
  /** A status to stop with, or `undefined` to continue. */
  done?(state: S, t: number): RunStatus | undefined
  observe?(state: S, t: number): Observation
  /** Serialise a state for checkpoints (and restore it), enabling resume and seek. */
  serialise?(state: S): Json
  deserialise?(json: Json): S
  /**
   * A copy of a state that later steps cannot change, for an algorithm whose `step` mutates its state in place. A
   * consumer that keeps states across steps (a ring buffer of frames, a history) takes them through this (see
   * {@link snapshotOf}); the stepping itself never calls it. Omitted, a state is kept as it is.
   */
  snapshot?(state: S): S
}

/**
 * A state as a consumer that keeps it should hold it: the algorithm's `snapshot` of it when it has one, else the state
 * itself.
 *
 * @param algorithm - The algorithm.
 * @param state - A state it produced.
 * @returns A state that later steps will not change (the state itself when the algorithm gives no `snapshot`).
 * @example
 * ```ts
 * import { snapshotOf, type Algorithm } from 'omai-engine/trace'
 * const set: Algorithm<Set<number>> = {
 *   name: 'set', config: {}, init: () => new Set<number>(), step: (s) => s,
 *   snapshot: (s) => new Set(s),
 * }
 * const live = new Set([1])
 * const kept = snapshotOf(set, live)
 * live.add(2)
 * kept.size // => 1
 * ```
 */
export function snapshotOf<S>(algorithm: Algorithm<S>, state: S): S {
  return algorithm.snapshot === undefined ? state : algorithm.snapshot(state)
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

/** A run that records nothing, for the drivers that are given no run to record into. */
function quietRun<S>(algorithm: Algorithm<S>, options: Pick<RunOptions, 'seed'>): RunHandle {
  return startRun(nullTracer, { name: algorithm.name, seed: options.seed ?? null, config: algorithm.config })
}

/**
 * The one stepper behind the three drivers: it owns the keys of a run, the rule that ends it and the order of observing
 * and checkpointing, so that however a run is driven (to the end, lazily, in slices) it computes, ends and records the
 * same. The drivers differ only in where they stop to look.
 */
interface Stepper<S> {
  /** The initial state, from the run's init key. */
  init(): S
  /**
   * The status of the run at `(state, t)`, or `undefined` to continue: the algorithm's `done` is asked at every step,
   * the limit included, and a run that reaches the limit with no verdict is `completed`.
   */
  verdict(state: S, t: number): RunStatus | undefined
  /** Take step `t` with its derived key; the state at `t + 1` is observed if due and checkpointed. */
  advance(state: S, t: number): S
  /** Observe the state at `t` when the policy's schedule says so. */
  open(state: S, t: number): void
  /** Store a checkpoint of the state at `t` when the policy records them. */
  mark(state: S, t: number): void
  /** Observe the state at `t` unless it was just observed (the last state is always recorded); the last metrics. */
  close(state: S, t: number): Readonly<Record<string, number>>
  /** The last step whose state was observed, `-1` before any. */
  readonly observedAt: number
}

function stepper<S>(algorithm: Algorithm<S>, root: Key, run: RunHandle, limit: number, observedAt = -1): Stepper<S> {
  const stepsKey = foldIn(root, keyTags.steps)
  let seen = observedAt
  let last: Readonly<Record<string, number>> = {}
  const observe = (state: S, t: number): void => {
    if (!run.enabled) return
    const o = algorithm.observe?.(state, t) ?? {}
    for (const [name, value] of Object.entries(o.metrics ?? {})) run.metric(t, name, value)
    for (const [name, x] of Object.entries(o.tensors ?? {})) run.tensor(t, name, x.role, x.value, x.shape)
    for (const m of o.messages ?? []) {
      run.message(t, m.kind, m.payload, {
        ...(m.from === undefined ? {} : { from: m.from }),
        ...(m.to === undefined ? {} : { to: m.to }),
      })
    }
    if (o.frame !== undefined) run.frame(t, o.frame)
    last = o.metrics ?? {}
    seen = t
  }
  const mark = (state: S, t: number): void => {
    if (run.enabled && algorithm.serialise !== undefined && shouldRecord(run.policy, { kind: 'checkpoint' }, t)) {
      run.checkpoint(t, algorithm.serialise(state))
    }
  }
  return {
    init: () => algorithm.init(foldIn(root, keyTags.init)),
    verdict: (state, t) => algorithm.done?.(state, t) ?? (t >= limit ? 'completed' : undefined),
    advance: (state, t) => {
      const next = algorithm.step(state, stepContext(t, foldIn(stepsKey, t), run))
      if (run.due(t + 1)) observe(next, t + 1)
      mark(next, t + 1)
      return next
    },
    open: (state, t) => {
      if (run.due(t)) observe(state, t)
    },
    mark,
    close: (state, t) => {
      if (seen !== t) observe(state, t)
      return last
    },
    get observedAt() {
      return seen
    },
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
 * @index function Run an Algorithm and record a traced run.
 * @example
 * ```ts
 * import { runAlgorithm, recorder } from 'omai-engine/trace'
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
  const restore = options.start
  const t0 = restore?.step ?? 0
  const steps = stepper(algorithm, rootKey(options), run, t0 + options.maxSteps)
  let state: S =
    restore === undefined
      ? steps.init()
      : restore.serialised === true && algorithm.deserialise !== undefined
        ? algorithm.deserialise(restore.state as Json)
        : (restore.state as S)
  let t = t0
  steps.open(state, t)
  if (restore === undefined) steps.mark(state, t)
  let status = steps.verdict(state, t)
  while (status === undefined) {
    state = steps.advance(state, t)
    t += 1
    status = steps.verdict(state, t)
  }
  run.end(status, t, steps.close(state, t))
  return { state, steps: t, status, runId: run.runId }
}

/**
 * Step through an algorithm lazily, yielding every state (for live figures and for stepping by hand). Nothing is
 * recorded.
 *
 * @param algorithm - The algorithm.
 * @param options - Seed or key and a step budget.
 * @returns A generator of `{ t, state, status }`; `status` is set on the last item and is what `runAlgorithm` ends with
 *   (the algorithm's own verdict, `completed` when the budget ran out first).
 * @example
 * ```ts
 * import { iterate } from 'omai-engine/trace'
 * const count = { name: 'count', config: {}, init: () => 0, step: (s: number) => s + 1 }
 * Array.from(iterate(count, { maxSteps: 3 }), (x) => x.state) // => [0, 1, 2, 3]
 * ```
 */
export function* iterate<S>(
  algorithm: Algorithm<S>,
  options: Pick<RunOptions, 'seed' | 'key' | 'maxSteps'>,
): Generator<{ readonly t: number; readonly state: S; readonly status?: RunStatus }> {
  const run = quietRun(algorithm, options)
  const steps = stepper(algorithm, rootKey(options), run, options.maxSteps)
  let state = steps.init()
  let t = 0
  for (;;) {
    const status = steps.verdict(state, t)
    if (status !== undefined) {
      yield { t, state, status }
      return
    }
    yield { t, state }
    state = steps.advance(state, t)
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
 * import { initialSlice, runFor, type Algorithm } from 'omai-engine/trace'
 * const count: Algorithm<number> = { name: 'count', config: {}, init: () => 0, step: (s) => s + 1 }
 * const first = initialSlice(count, {})
 * [first.t, runFor(count, { steps: 2 }, { maxSteps: 10, from: first }).state] // => [0, 2]
 * ```
 */
export function initialSlice<S>(
  algorithm: Algorithm<S>,
  options: Pick<RunOptions, 'seed' | 'key'> & { readonly run?: RunHandle },
): Slice<S> {
  const run = options.run ?? quietRun(algorithm, options)
  const steps = stepper(algorithm, rootKey(options), run, Number.POSITIVE_INFINITY)
  const state = steps.init()
  steps.open(state, 0)
  steps.mark(state, 0)
  return { state, t: 0, observedAt: steps.observedAt }
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
 * @throws OmaiError with code `value` when the budget is not positive.
 * @concept algorithm-protocol
 * @example
 * ```ts
 * import { runFor, type Algorithm } from 'omai-engine/trace'
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
  const limit = budget.steps ?? Number.POSITIVE_INFINITY
  if (!(limit > 0) || (budget.ms !== undefined && !(budget.ms > 0))) {
    throw new OmaiError('value', 'runFor: the step and time budgets must be positive')
  }
  const run = options.run ?? quietRun(algorithm, options)
  if (options.from?.status !== undefined) return options.from
  const from = options.from ?? initialSlice(algorithm, { ...options, run })
  const steps = stepper(algorithm, rootKey(options), run, options.maxSteps, from.observedAt)
  const started = budget.clock?.()
  const outOfTime = (): boolean =>
    budget.clock !== undefined && budget.ms !== undefined && budget.clock() - (started ?? 0) >= budget.ms
  let state = from.state
  let t = from.t
  let taken = 0
  let status = steps.verdict(state, t)
  while (status === undefined && taken < limit && !(taken > 0 && outOfTime())) {
    state = steps.advance(state, t)
    t += 1
    taken += 1
    status = steps.verdict(state, t)
  }
  if (status !== undefined) steps.close(state, t)
  return { state, t, observedAt: steps.observedAt, ...(status === undefined ? {} : { status }) }
}
