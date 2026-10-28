/**
 * The log contract: what a method can log and hook, by path (`training/loss`, `agent/q`, `rl/episode/end`), and the
 * subscription that says which of those paths to keep and how often (`Log`) or what to run when an event fires (`On`).
 * `resolve` reconciles a contract with a subscription once, into the trace policy, the per-path schedules and the hooks
 * a session installs; after that nothing is clever. Paths give the hierarchy and patterns (`model/params/**`,
 * `**\/loss`) select over it.
 */

import { type Direction, type Json, MdmlError } from '../base/index.ts'
import type { Value } from '../tensor/index.ts'
import type { TensorRole } from './events.ts'
import { policies, type TracePolicy } from './policy.ts'
import { channels, type TraceSchema } from './schema.ts'
import type { RunHandle } from './tracer.ts'

/** A path in the contract's hierarchy: segments joined by `/`. */
export type Path = string

interface SourceBase {
  readonly description: string
  /**
   * The native channel name this path renames, for algorithms written before the contract (an SMO's `objective`
   * becomes `smo/dual-objective`). Without it the path is the channel name, and the algorithm logs to it by path.
   */
  readonly from?: string
}

/** A scalar per step (a loss, a return, a duality gap). */
export interface ScalarSource extends SourceBase {
  readonly kind: 'scalar'
  readonly unit?: string
  readonly direction?: Direction
}

/** A tensor per step, with its role (parameters, gradients, a state, activations, an output). */
export interface TensorSource extends SourceBase {
  readonly kind: 'tensor'
  readonly role: TensorRole
  readonly shape?: readonly (number | string)[]
}

/** A structured message per step: a decision and its alternatives, a batch's rows, a working pair. */
export interface MessageSource extends SourceBase {
  readonly kind: 'message'
  readonly payload?: Readonly<Record<string, string>>
}

/** A picture of the state per step (a scene as JSON). */
export interface FrameSource extends SourceBase {
  readonly kind: 'frame'
}

/** What a derived path yields: JSON, or an exposition record (any object with a `kind`). */
export type Derived = Json | { readonly kind: string }

/**
 * A path computed from the state when it is due or when it is probed, never emitted inline: a decision surface on a
 * grid, a greedy rollout, a policy table. `dear` ones default to rare schedules.
 */
export interface DerivedSource<S> {
  readonly kind: 'derived'
  readonly description: string
  readonly cost?: 'cheap' | 'dear'
  readonly compute: (state: S, args?: Json) => Derived
}

/** A lifecycle point hooks attach to and schedules count (`training/epoch/end`, `rl/episode/end`). */
export interface EventSource {
  readonly kind: 'event'
  readonly description: string
  readonly context?: Readonly<Record<string, string>>
}

/** Anything a path can be. */
export type Source<S> = ScalarSource | TensorSource | MessageSource | FrameSource | DerivedSource<S>

/**
 * What a method can log and hook, by path. A key may be a pattern (`model/activations/*`) when the names are only
 * known as the model runs (taps).
 *
 * @concept log-contract
 */
export type Contract<S> = Readonly<Record<Path, Source<S> | EventSource>>

type Described<T> = Omit<T, 'kind' | 'description'> & { readonly description?: string }

/**
 * A scalar source.
 *
 * @param options - Description, unit, direction and the native name it renames.
 * @returns The source.
 * @example
 * ```ts
 * import { scalar } from 'mdml-engine/trace'
 * scalar({ direction: 'lower' }).kind // => 'scalar'
 * ```
 */
export function scalar(options: Described<ScalarSource> = {}): ScalarSource {
  return { kind: 'scalar', description: '', ...options }
}

/**
 * A tensor source.
 *
 * @param options - Role, shape, description and the native name it renames.
 * @returns The source.
 * @example
 * ```ts
 * import { tensor } from 'mdml-engine/trace'
 * tensor({ role: 'state', shape: ['n'] }).role // => 'state'
 * ```
 */
export function tensor(options: Described<TensorSource>): TensorSource {
  return { kind: 'tensor', description: '', ...options }
}

/**
 * A message source.
 *
 * @param options - Payload fields, description and the native kind it renames.
 * @returns The source.
 * @example
 * ```ts
 * import { message } from 'mdml-engine/trace'
 * message({ payload: { i: 'first index' } }).kind // => 'message'
 * ```
 */
export function message(options: Described<MessageSource> = {}): MessageSource {
  return { kind: 'message', description: '', ...options }
}

/**
 * A frame source.
 *
 * @param options - Description.
 * @returns The source.
 * @example
 * ```ts
 * import { frame } from 'mdml-engine/trace'
 * frame().kind // => 'frame'
 * ```
 */
export function frame(options: Described<FrameSource> = {}): FrameSource {
  return { kind: 'frame', description: '', ...options }
}

/**
 * A derived source.
 *
 * @param options - The function of the state, its cost and description.
 * @returns The source.
 * @example
 * ```ts
 * import { derived } from 'mdml-engine/trace'
 * derived({ compute: (s: number) => s * 2 }).compute(4) // => 8
 * ```
 */
export function derived<S>(options: Described<DerivedSource<S>>): DerivedSource<S> {
  return { kind: 'derived', description: '', ...options }
}

/**
 * An event source.
 *
 * @param options - The context a hook receives, and a description.
 * @returns The source.
 * @example
 * ```ts
 * import { event } from 'mdml-engine/trace'
 * event({ context: { epoch: 'the epoch just finished' } }).kind // => 'event'
 * ```
 */
export function event(options: Described<EventSource> = {}): EventSource {
  return { kind: 'event', description: '', ...options }
}

/** A schedule in events: whenever an event fires, or every nth time it fires. */
export type EventSchedule = Path | readonly [Path, number]

/** How often a path is logged: every n steps, at log-spaced steps, or by an event. */
export type LogEvery = number | 'logSpaced' | EventSchedule

/** How much of a path the tape holds: every event, the last one, or the most recent n. */
export type LogKeep = 'all' | 'last' | number

/** What a subscription asks for one path. */
export interface LogSpec {
  readonly kind: 'log'
  /** Default: every step (and every firing, for an event path). */
  readonly every?: LogEvery
  /** Tensors: norms and histograms, or the values (default summary). */
  readonly detail?: 'summary' | 'full'
  /** Arguments of a derived path (a grid, a sample count). */
  readonly args?: Json
  /** Stop recording this path once the events it holds pass this many bytes of JSON. */
  readonly budget?: number
  /**
   * How much of this path the tape holds (default `all`): with `last` only its most recent event, with a number its
   * most recent n; older ones are forgotten as new ones arrive. Checkpoints are never subject to it, so seeking is
   * unaffected, and the budget counts only what is held.
   */
  readonly keep?: LogKeep
}

/** What a value logged against a contract can be: a number, a tensor or array, JSON, or an exposition record. */
export type LogValue = number | boolean | Value | ArrayLike<number> | Json | { readonly kind: string }

/** What a hook receives when its event fires. */
export interface HookContext {
  readonly t: number
  readonly path: Path
  readonly context: Json
  readonly run: RunHandle
  log(path: Path, value: LogValue): void
  /** End the run after this step, with status `stopped`. */
  stop(reason: string): void
}

/** A hook on an event path. */
export type Hook = (ctx: HookContext) => void

/** A subscription to an event path: a hook, or the name of a hook the method exports (so it can cross to a worker). */
export interface HookSpec {
  readonly kind: 'hook'
  readonly hook?: Hook
  readonly name?: string
  readonly options?: Json
}

/**
 * What a caller asks of a method: nothing (`off`: the raw algorithm at full speed), the method's own small choice
 * (`defaults`), everything at summary detail (`all`), or paths and patterns with how to log or hook them.
 *
 * @concept log-contract
 */
export type Subscription = 'off' | 'defaults' | 'all' | Readonly<Record<string, LogSpec | HookSpec>>

/**
 * Ask for a path: how often, at what detail, with what arguments, and how much of it the tape holds.
 *
 * @param options - Schedule, detail, arguments, budget and window.
 * @returns The spec.
 * @example
 * ```ts
 * import { Log } from 'mdml-engine/trace'
 * Log({ every: ['rl/episode/end', 10] }).every // => ['rl/episode/end', 10]
 * Log({ keep: 'last' }).keep // => 'last'
 * ```
 */
export function Log(options: Omit<LogSpec, 'kind'> = {}): LogSpec {
  return { kind: 'log', ...options }
}

/**
 * Attach a hook to an event path: a function, or the name of a hook the method exports with its options.
 *
 * @param hook - The hook, or its name.
 * @param options - The named hook's options.
 * @returns The spec.
 * @example
 * ```ts
 * import { On } from 'mdml-engine/trace'
 * On('earlyStopping', { patience: 5 }).name // => 'earlyStopping'
 * ```
 */
export function On(hook: Hook | string, options?: Json): HookSpec {
  if (typeof hook === 'string') return { kind: 'hook', name: hook, ...(options === undefined ? {} : { options }) }
  return { kind: 'hook', hook }
}

/**
 * Whether a path is a pattern (`*` matches one segment, `**` any number).
 *
 * @param path - A path or pattern.
 * @returns `true` for patterns.
 * @example
 * ```ts
 * import { isPattern } from 'mdml-engine/trace'
 * [isPattern('a/b'), isPattern('a/*')] // => [false, true]
 * ```
 */
export function isPattern(path: Path): boolean {
  return path.includes('*')
}

/**
 * Whether a pattern matches a path: `*` matches one segment, `**` any number (including none).
 *
 * @param pattern - The pattern.
 * @param path - The path.
 * @returns `true` when it matches.
 * @example
 * ```ts
 * import { matchesPath } from 'mdml-engine/trace'
 * [matchesPath('model/**', 'model/params/w'), matchesPath('*\/loss', 'training/val/loss')] // => [true, false]
 * ```
 */
export function matchesPath(pattern: Path, path: Path): boolean {
  const p = pattern.split('/')
  const s = path.split('/')
  const go = (i: number, j: number): boolean => {
    if (i === p.length) return j === s.length
    const seg = p[i]
    if (seg === '**') return go(i + 1, j) || (j < s.length && go(i, j + 1))
    if (j === s.length) return false
    return (seg === '*' || seg === s[j]) && go(i + 1, j + 1)
  }
  return go(0, 0)
}

/** One subscribed path, with its source and spec. */
export interface Subscribed<S> {
  readonly path: Path
  readonly source: Source<S> | EventSource
  readonly spec: LogSpec
}

/** A contract and a subscription reconciled: the policy, the paths to log, and the hooks by event. */
export interface Resolved<S> {
  /** `true` for the `off` subscription: no tracer, no derived work, no hooks. */
  readonly off: boolean
  readonly policy: TracePolicy
  readonly logs: readonly Subscribed<S>[]
  readonly hooks: ReadonlyMap<Path, readonly Hook[]>
}

/** Options of {@link resolve}. */
export interface ResolveOptions {
  /** What `defaults` means for this method. */
  readonly defaults?: Subscription
  /** Hooks the method exports by name, for `On('name', options)`. */
  readonly hooks?: Readonly<Record<string, (options?: Json) => Hook>>
  /** Store checkpoints every this many steps (default none). */
  readonly checkpointInterval?: number
}

const EVERY_PATH = <S>(source: Source<S> | EventSource): LogEvery | undefined =>
  source.kind === 'derived' && source.cost === 'dear' ? 'logSpaced' : undefined

function everything<S>(contract: Contract<S>): Readonly<Record<string, LogSpec>> {
  const out: Record<string, LogSpec> = {}
  for (const [path, source] of Object.entries(contract)) {
    if (isPattern(path)) continue
    const every = EVERY_PATH(source)
    out[path] = Log(every === undefined ? {} : { every })
  }
  return out
}

/**
 * Reconcile a contract with a subscription: match patterns to paths, build the trace policy that selects the matched
 * channels (with checkpoints when asked), keep each path's schedule and detail, and bind the hooks.
 *
 * @param contract - The method's contract.
 * @param subscription - What the caller asks for.
 * @param options - The method's defaults and named hooks, and the checkpoint interval.
 * @returns The resolved subscription.
 * @throws MdmlError with code `value` for an exact path the contract lacks, a hook on a path that is not an event, a
 *   named hook the method does not export, or a `keep` that is not `all`, `last` or a positive integer.
 * @concept log-contract
 * @example
 * ```ts
 * import { Log, resolve, scalar } from 'mdml-engine/trace'
 * const r = resolve({ 'training/loss': scalar(), 'training/lr': scalar() }, { 'training/**': Log({ every: 5 }) })
 * r.logs.map((l) => l.path) // => ['training/loss', 'training/lr']
 * r.policy.metrics // => ['training/loss', 'training/lr']
 * ```
 */
export function resolve<S>(
  contract: Contract<S>,
  subscription: Subscription,
  options: ResolveOptions = {},
): Resolved<S> {
  if (subscription === 'off') return { off: true, policy: policies.minimal, logs: [], hooks: new Map() }
  const asked: Subscription =
    subscription === 'all'
      ? everything(contract)
      : subscription === 'defaults'
        ? (options.defaults ?? {})
        : subscription
  if (asked === 'off' || asked === 'all' || asked === 'defaults')
    return resolve(contract, asked, { ...options, defaults: {} })
  const logs = new Map<Path, Subscribed<S>>()
  const hooks = new Map<Path, Hook[]>()
  for (const [key, spec] of Object.entries(asked)) {
    const exact = contract[key]
    if (spec.kind === 'hook') {
      if (exact === undefined || exact.kind !== 'event')
        throw new MdmlError('value', `resolve: '${key}' is not an event of the contract`)
      const hook = spec.hook ?? (spec.name === undefined ? undefined : options.hooks?.[spec.name]?.(spec.options))
      if (hook === undefined) throw new MdmlError('value', `resolve: no hook '${spec.name ?? ''}' for '${key}'`)
      hooks.set(key, [...(hooks.get(key) ?? []), hook])
      continue
    }
    const keep = spec.keep
    if (keep !== undefined && keep !== 'all' && keep !== 'last' && !(Number.isInteger(keep) && keep > 0))
      throw new MdmlError(
        'value',
        `resolve: '${key}' keeps ${String(keep)}; keep is 'all', 'last' or a positive integer`,
      )
    if (exact !== undefined) {
      logs.set(key, { path: key, source: exact, spec })
      continue
    }
    if (!isPattern(key)) throw new MdmlError('value', `resolve: the contract has no path '${key}'`)
    for (const [path, source] of Object.entries(contract)) {
      if (!matchesPath(key, path) || logs.has(path)) continue
      const every = spec.every ?? EVERY_PATH(source)
      logs.set(path, { path, source, spec: every === undefined ? spec : { ...spec, every } })
    }
  }
  const subscribed = [...logs.values()]
  const names = (kind: 'scalar' | 'tensor' | 'message'): boolean | string[] => {
    const exact = subscribed.filter((l) => l.source.kind === kind)
    if (exact.some((l) => isPattern(l.path))) return true
    const list = exact.map((l) => (l.source as SourceBase).from ?? l.path)
    // Events logged as messages, and derived paths (which become any channel), need their own names selected.
    if (kind === 'message') list.push(...subscribed.filter((l) => l.source.kind === 'event').map((l) => l.path))
    const anyDerived = subscribed.some((l) => l.source.kind === 'derived')
    return anyDerived ? true : list
  }
  const roles = new Set<TensorRole>()
  for (const l of subscribed) {
    if (l.source.kind === 'tensor') roles.add(l.source.role)
    if (l.source.kind === 'derived') roles.add('output')
  }
  const full = subscribed.some((l) => l.spec.detail === 'full')
  const frames = subscribed.some((l) => l.source.kind === 'frame' || l.source.kind === 'derived')
  const interval = options.checkpointInterval
  const policy: TracePolicy = {
    ...policies.minimal,
    metrics: names('scalar'),
    tensors: names('tensor'),
    roles: [...roles],
    messages: names('message'),
    frames,
    checkpoints: interval !== undefined,
    checkpointInterval: interval ?? 0,
    schedule: { every: 1 },
    detail: full ? 'full' : 'summary',
    ...(full ? { maxTensorSize: 1_000_000 } : {}),
  }
  return { off: false, policy, logs: subscribed, hooks }
}

/**
 * The trace schema of a contract, for the run event: its scalars, tensors and messages by path.
 *
 * @param contract - The contract.
 * @returns The schema.
 * @example
 * ```ts
 * import { contractSchema, scalar } from 'mdml-engine/trace'
 * Object.keys(contractSchema({ 'a/b': scalar() }).metrics) // => ['a/b']
 * ```
 */
export function contractSchema<S>(contract: Contract<S>): TraceSchema {
  const metrics: Record<
    string,
    { readonly description: string; readonly unit?: string; readonly direction?: Direction }
  > = {}
  const tensors: Record<
    string,
    { readonly role: TensorRole; readonly description: string; readonly shape?: readonly (number | string)[] }
  > = {}
  const messages: Record<
    string,
    { readonly description: string; readonly payload?: Readonly<Record<string, string>> }
  > = {}
  let frames = false
  for (const [path, source] of Object.entries(contract)) {
    switch (source.kind) {
      case 'scalar':
        metrics[path] = {
          description: source.description,
          ...(source.unit === undefined ? {} : { unit: source.unit }),
          ...(source.direction === undefined ? {} : { direction: source.direction }),
        }
        break
      case 'tensor':
        tensors[path] = {
          role: source.role,
          description: source.description,
          ...(source.shape === undefined ? {} : { shape: source.shape }),
        }
        break
      case 'message':
        messages[path] = {
          description: source.description,
          ...(source.payload === undefined ? {} : { payload: source.payload }),
        }
        break
      case 'event':
        messages[path] = {
          description: source.description,
          ...(source.context === undefined ? {} : { payload: source.context }),
        }
        break
      case 'frame':
        frames = true
        break
      default:
        break
    }
  }
  return channels({ metrics, tensors, messages, frames })
}
