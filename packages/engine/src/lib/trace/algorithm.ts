import type { Json } from '../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { Value } from '../tensor/index.ts'
import type { RunStatus, TensorRole } from './events.ts'
import type { TracePolicy } from './policy.ts'
import type { TraceSchema } from './schema.ts'
import { nullTracer, type RunHandle, startRun, type Tracer } from './tracer.ts'

/** What an algorithm sees at each step. */
export interface StepContext {
  readonly t: number
  /** A key derived from the run's key and `t`: the same at step `t` however the run was paused or resumed. */
  readonly key: Key
  /** The run, for emitting extra events or starting child runs of inner algorithms. */
  readonly run: RunHandle
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
  /** Emit a checkpoint every this many steps (when the policy records checkpoints and the algorithm serialises). */
  readonly checkpointEvery?: number
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

function rootKey(options: RunOptions): Key {
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

/**
 * Run an {@link Algorithm}, recording a traced run.
 *
 * Step keys are `foldIn(stepsKey, t)`, so step `t` draws the same numbers however the run was paused, resumed or
 * replayed. The state is observed on the policy's schedule and always at the last step.
 *
 * @param algorithm - The algorithm.
 * @param options - Step budget, seed or key, tracer and policy (or a parent run), checkpointing and a start state.
 * @returns The final state, steps taken, status and run id.
 * @concept algorithm-protocol
 * @example
 * ```ts
 * import { runAlgorithm, recorder } from 'whitebox-engine/trace'
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
  const checkpointEvery = options.checkpointEvery ?? 0
  let last: Readonly<Record<string, number>> = {}
  let observedAt = -1
  const observe = (): void => {
    if (!run.enabled) return
    last = record(algorithm, run, state, t)
    observedAt = t
    if (checkpointEvery > 0 && t % checkpointEvery === 0 && algorithm.serialise !== undefined) {
      run.checkpoint(t, algorithm.serialise(state))
    }
  }
  if (run.due(t)) observe()
  let status: RunStatus = 'completed'
  const limit = (restore?.step ?? 0) + options.maxSteps
  while (t < limit) {
    const stop = algorithm.done?.(state, t)
    if (stop !== undefined) {
      status = stop
      break
    }
    state = algorithm.step(state, { t, key: foldIn(stepsKey, t), run })
    t += 1
    if (run.due(t)) observe()
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
 * import { iterate } from 'whitebox-engine/trace'
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
    state = algorithm.step(state, { t, key: foldIn(stepsKey, t), run })
    t += 1
  }
}
