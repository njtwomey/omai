/**
 * Methods and sessions: a `Method` declares how to build a run from its configuration and what can be known about it
 * (its contract), and a `Session` is a method running or run: the tape plus the verbs to advance it eagerly or live,
 * seek any recorded step, fork from one, probe a derived path at any step and export the tape. Eager against live is
 * the session's business: the same tape and the same numbers either way.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import { contains, type DictSpace, defaults, type SpaceValue } from '../space/index.ts'
import { type Algorithm, initialSlice, runFor, type Slice, type SliceBudget } from './algorithm.ts'
import { seek } from './checkpoint.ts'
import {
  type Contract,
  contractSchema,
  type Derived,
  type EventSchedule,
  type Hook,
  type Resolved,
  resolve,
  type Subscription,
} from './contract.ts'
import type { Run, RunStatus } from './events.ts'
import { gated, logContext, type Scheduler, scheduler } from './logging.ts'
import { nullTracer, type Recorder, type RunHandle, recorder, startRun } from './tracer.ts'

/** A method's configuration: its knobs' values. */
export type Config = Readonly<Record<string, SpaceValue>>

/**
 * A method: a declaration of how to build a run and what can be known about it. It writes no loop and no hook of its
 * own; everything it has to say goes through `ctx.log` against its contract.
 *
 * @concept log-contract
 */
export interface Method<S, C extends Config = Config> {
  readonly name: string
  /** Knobs with defaults: the application's parameters, never functions. */
  readonly config: DictSpace
  /** Build the run from constructables (a dataset, a model, an environment, a problem). */
  make(config: C, key: Key): Algorithm<S>
  /**
   * What can be logged and hooked, by path. A function of the configuration and key when derived paths need the
   * constructables `make` builds (the same key builds the same ones).
   */
  readonly contract: Contract<S> | ((config: C, key: Key) => Contract<S>)
  /** The subscription used when the caller gives none (default: nothing). */
  readonly defaults?: Subscription
  /** Hooks exported by name, so `On('name', options)` can cross to a worker. */
  readonly hooks?: Readonly<Record<string, (options?: Json) => Hook>>
  /** Bespoke records from the tape, for views the generic ones cannot give. */
  readonly projections?: Readonly<Record<string, (run: Run, options?: Json) => Derived>>
}

/**
 * Declare a method, checking its contract's paths and that its defaults resolve against it.
 *
 * @param method - The method.
 * @returns The method.
 * @throws MdmlError with code `value` for an empty or malformed path, or defaults the contract cannot satisfy.
 * @example
 * ```ts
 * import { defineMethod, scalar } from 'mdml-engine/trace'
 * import { dict } from 'mdml-engine/space'
 * const m = defineMethod({
 *   name: 'count', config: dict({}),
 *   make: () => ({ name: 'count', config: {}, init: () => 0, step: (s: number, ctx) => { ctx.log('n', s); return s + 1 } }),
 *   contract: { n: scalar() },
 * })
 * m.name // => 'count'
 * ```
 */
export function defineMethod<S, C extends Config>(method: Method<S, C>): Method<S, C> {
  const contract = contractOf(method, defaults(method.config) as C, makeKey(0))
  for (const path of Object.keys(contract)) {
    if (path === '' || path.split('/').some((seg) => seg === ''))
      throw new MdmlError('value', `defineMethod: ${method.name} has a malformed path '${path}'`)
  }
  resolve(contract, method.defaults ?? {}, { ...(method.hooks === undefined ? {} : { hooks: method.hooks }) })
  return method
}

/**
 * A method's contract for a configuration and key.
 *
 * @param method - The method.
 * @param config - The configuration.
 * @param key - The key `make` is given.
 * @returns The contract.
 */
export function contractOf<S, C extends Config>(method: Method<S, C>, config: C, key: Key): Contract<S> {
  return typeof method.contract === 'function' ? method.contract(config, key) : method.contract
}

/** When a session computes: up front, when asked, or up front to a point and then when asked. */
export type Mode = 'eager' | 'live' | { readonly eager: number | EventSchedule }

/** Options of {@link session}. */
export interface SessionOptions<S, C extends Config> {
  /** Knob values over the method's defaults. */
  readonly config?: Partial<C>
  readonly seed?: number
  /** What to keep (default the method's `defaults`). */
  readonly logs?: Subscription
  /** Default `live`. */
  readonly mode?: Mode
  /** The whole run's budget in steps (default 1000); eager mode runs to it. */
  readonly steps?: number
  /** Store a checkpoint every this many steps (default about a hundred over the run), for `stateAt` and `fork`. */
  readonly checkpointEvery?: number
  /** Start from a state at a step instead of `init` (forks). */
  readonly start?: { readonly state: S; readonly t: number }
}

/**
 * A method running or run: the tape plus the verbs to advance it, seek it, fork it, probe it and export it.
 *
 * @concept log-contract
 */
export interface Session<S, C extends Config = Config> {
  readonly method: Method<S, C>
  readonly config: C
  readonly seed: number
  readonly algorithm: Algorithm<S>
  /** The contract this session was made with. */
  readonly contract: Contract<S>
  /** The step reached, the state there, and the status once the run has ended. */
  readonly t: number
  readonly state: S
  readonly status: RunStatus | undefined
  /** Compute more: by a budget of steps or milliseconds (default: the rest of the run). */
  advance(budget?: SliceBudget): Slice<S>
  /** The state at any recorded step: the nearest earlier checkpoint re-run forward with the original keys. */
  stateAt(t: number): S
  /** A new session starting from `stateAt(t)`: the same seed to continue exactly, another to branch. */
  fork(t: number, options?: { readonly seed?: number; readonly config?: Partial<C> }): Session<S, C>
  /** A derived path of the contract, computed on the state at a step. */
  probe(t: number, path: string, args?: Json): Derived
  /** A bespoke record of the method, from the tape so far. */
  project(name: string, options?: Json): Derived
  /** The tape so far. */
  run(): Run
  /** The steps at which an event has fired so far. */
  stepsOf(event: string): readonly number[]
}

function configOf<C extends Config>(space: DictSpace, given: Partial<C> | undefined, who: string): C {
  const config = { ...(defaults(space) as Config), ...given } as C
  if (!contains(space, config)) throw new MdmlError('value', `${who}: the configuration is outside the method's space`)
  return config
}

/**
 * Start a session of a method: build its run, resolve what to keep, and, in eager mode, compute up front.
 *
 * @param method - The method.
 * @param options - Configuration, seed, subscription, mode and budgets.
 * @returns The session.
 * @throws MdmlError with code `value` for a configuration outside the method's space or a subscription the contract
 *   cannot satisfy.
 * @concept log-contract
 * @example
 * ```ts
 * import { defineMethod, Log, metricSeries, scalar, session } from 'mdml-engine/trace'
 * import { dict } from 'mdml-engine/space'
 * const count = defineMethod({
 *   name: 'count', config: dict({}),
 *   make: () => ({
 *     name: 'count', config: {}, init: () => 0, step: (s: number, ctx) => { ctx.log('n', s); return s + 1 },
 *     serialise: (s: number) => s, deserialise: (j) => j as number,   // checkpoints make stateAt and fork possible
 *   }),
 *   contract: { n: scalar() },
 * })
 * const s = session(count, { mode: 'eager', steps: 6, logs: { n: Log({ every: 2 }) } })
 * metricSeries(s.run(), 'n').values // => [0, 2, 4]
 * s.stateAt(3) // => 3
 * ```
 */
export function session<S, C extends Config>(method: Method<S, C>, options: SessionOptions<S, C> = {}): Session<S, C> {
  const config = configOf<C>(method.config, options.config, method.name)
  const seed = options.seed ?? 0
  const total = options.steps ?? 1000
  const checkpointEvery = options.checkpointEvery ?? Math.max(1, Math.floor(total / 100))
  const makeKeyOf = foldIn(makeKey(seed), 7)
  const contract = contractOf(method, config, makeKeyOf)
  const resolved: Resolved<S> = resolve(contract, options.logs ?? 'defaults', {
    ...(method.defaults === undefined ? {} : { defaults: method.defaults }),
    ...(method.hooks === undefined ? {} : { hooks: method.hooks }),
    checkpointInterval: checkpointEvery,
  })
  const sched: Scheduler<S> = scheduler(resolved)
  const rec: Recorder | undefined = resolved.off ? undefined : recorder()
  const startAt = options.start?.t ?? 0
  const run: RunHandle = startRun(rec === undefined ? nullTracer : gated(rec, resolved, sched), {
    name: method.name,
    seed,
    config: config as Json,
    runId: startAt === 0 ? `${method.name}:${seed}` : `${method.name}:${seed}@${startAt}`,
    schema: contractSchema(contract),
    policy: resolved.policy,
  })
  const raw = method.make(config, makeKeyOf)
  let stopped: RunStatus | undefined
  const control = {
    stop: (): void => {
      stopped = 'stopped'
    },
  }
  const derive = (state: S, t: number): void => {
    if (!run.enabled) return
    const logs = logContext(run, resolved, sched, t, control)
    for (const l of resolved.logs) {
      if (l.source.kind !== 'derived' || !logs.wants(l.path)) continue
      logs.log(l.path, l.source.compute(state, l.spec.args))
    }
  }
  const algorithm: Algorithm<S> = {
    ...raw,
    init: (k) => {
      const s = raw.init(k)
      derive(s, 0)
      return s
    },
    step: (s, ctx) => {
      const logs = logContext(run, resolved, sched, ctx.t, control)
      const next = raw.step(s, { ...ctx, log: logs.log, wants: logs.wants, emit: logs.emit, taps: logs.taps })
      derive(next, ctx.t + 1)
      return next
    },
    done: (s, t) => stopped ?? raw.done?.(s, t),
  }
  let slice: Slice<S> | undefined = options.start === undefined ? undefined : { ...options.start }
  if (slice !== undefined && raw.serialise !== undefined && run.wants('checkpoint'))
    run.checkpoint(slice.t, raw.serialise(slice.state))
  const current = (): Slice<S> => {
    if (slice === undefined) slice = initialSlice(algorithm, { seed, run })
    return slice
  }
  const advance = (budget: SliceBudget = { steps: Number.POSITIVE_INFINITY }): Slice<S> => {
    const from = current()
    if (from.status !== undefined) return from
    const next = runFor(algorithm, budget, { seed, maxSteps: total, from, run })
    slice = next
    if (next.status !== undefined) run.end(next.status, next.t)
    return next
  }
  const stateAt = (t: number): S => {
    const now = current()
    if (t === now.t) return now.state
    if (t > now.t || t < 0) throw new MdmlError('value', `${method.name}: step ${t} has not been reached`)
    if (rec === undefined) throw new MdmlError('value', `${method.name}: an 'off' session keeps no past`)
    return seek(raw, rec.run(), t, { seed })
  }
  const self: Session<S, C> = {
    method,
    config,
    seed,
    algorithm: raw,
    contract,
    get t() {
      return current().t
    },
    get state() {
      return current().state
    },
    get status() {
      return current().status
    },
    advance,
    stateAt,
    fork: (t, forkOptions = {}) =>
      // A fork stands at its start until asked to advance, whatever mode its parent ran in.
      session(method, {
        ...options,
        mode: 'live',
        config: { ...config, ...forkOptions.config } as Partial<C>,
        seed: forkOptions.seed ?? seed,
        start: { state: stateAt(t), t },
      }),
    probe: (t, path, args) => {
      const source = contract[path]
      if (source === undefined || source.kind !== 'derived')
        throw new MdmlError('value', `${method.name}: '${path}' is not a derived path`)
      return source.compute(stateAt(t), args)
    },
    project: (name, projectOptions) => {
      const p = method.projections?.[name]
      if (p === undefined) throw new MdmlError('value', `${method.name}: no projection '${name}'`)
      return p(self.run(), projectOptions)
    },
    run: () => (rec === undefined ? { schema: 'mdml.run/1', events: [] } : rec.run()),
    stepsOf: (ev) => sched.stepsOf(ev),
  }
  const mode = options.mode ?? 'live'
  if (mode === 'eager') advance()
  else if (mode !== 'live') {
    const until = mode.eager
    if (typeof until === 'number') advance({ steps: until })
    else {
      const [ev, n] = typeof until === 'string' ? [until, 1] : until
      // Slices of a few dozen steps overshoot the nth firing by at most a slice; the states stay exact.
      while (sched.stepsOf(ev).length < n && current().status === undefined) advance({ steps: 32 })
    }
  }
  return self
}
