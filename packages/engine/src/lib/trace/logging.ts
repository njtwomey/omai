/**
 * Logging against a contract: the scheduler that says whether a path is due at a step (every n steps, at log-spaced
 * steps, or on the nth firing of an event), the log context algorithms write to (`log`, `wants`, `emit`, `taps`), the
 * default context runners give algorithms outside a session, and the gating tracer that renames native channels to
 * contract paths, lets through only what is subscribed and due, and holds no more of a path than its `keep` allows.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { isTensor, type Value } from '../tensor/index.ts'
import {
  type Concrete,
  type Hook,
  isPattern,
  type LogValue,
  matchesPath,
  type Path,
  type Resolved,
  type Subscribed,
} from './contract.ts'
import type { TensorRole, TraceEvent } from './events.ts'
import { summarise } from './summaries.ts'
import { type TapContext, tapContext } from './taps.ts'
import type { Recorder, RunHandle, Tracer } from './tracer.ts'

/** Whether a path is due at a step, by its subscription's schedule; counts events and the paths that wait on them. */
export interface Scheduler<S = unknown> {
  /** Whether a subscribed path is due at a step (unsubscribed paths never are). */
  due(path: Path, t: number): boolean
  /** An event fired at a step: count it, and make the paths scheduled on its nth firing due. */
  fired(event: Path, t: number): void
  /** A path scheduled by an event was logged: not due again until the event fires. */
  consume(path: Path): void
  /** The step of every firing of an event so far. */
  stepsOf(event: Path): readonly number[]
  /** The subscribed entry a path falls under: its own, or the first pattern it matches. */
  entry(path: Path): Subscribed<S> | undefined
  /**
   * Whether the contract has the path (its own key, or a pattern key it falls under), subscribed or not; `true` for
   * every path when the resolution carries no contract. Remembered per path, so asking again costs a lookup.
   */
  known(path: Path): boolean
}

/**
 * The scheduler of a resolved subscription.
 *
 * @param resolved - The resolved subscription.
 * @returns The scheduler.
 * @example
 * ```ts
 * import { Log, resolve, scalar, scheduler } from 'mdml-engine/trace'
 * const s = scheduler(resolve({ 'a/x': scalar() }, { 'a/x': Log({ every: 4 }) }))
 * [s.due('a/x', 0), s.due('a/x', 3), s.due('a/x', 8), s.due('a/y', 8)] // => [true, false, true, false]
 * ```
 */
export function scheduler<S>(resolved: Resolved<S>): Scheduler<S> {
  const exact = new Map<Path, Subscribed<S>>()
  const patterns: Subscribed<S>[] = []
  for (const l of resolved.logs) if (isPattern(l.path)) patterns.push(l)
  for (const l of resolved.logs) if (!isPattern(l.path)) exact.set(l.path, l)
  const pending = new Set<Path>()
  const counts = new Map<Path, number>()
  const steps = new Map<Path, number[]>()
  // Pattern matches are remembered: taps log the same names every step.
  const found = new Map<Path, Subscribed<S> | undefined>()
  const entry = (path: Path): Subscribed<S> | undefined => {
    if (found.has(path)) return found.get(path)
    const e = exact.get(path) ?? patterns.find((p) => matchesPath(p.path, path))
    found.set(path, e)
    return e
  }
  // Only logs scheduled on an event wait for firings: found once here, not by scanning every log per firing.
  const onEvent: { readonly path: Path; readonly on: Path; readonly n: number }[] = []
  for (const l of resolved.logs) {
    const every = l.spec.every
    if (every === undefined || typeof every === 'number' || every === 'logSpaced') continue
    const [on, n] = typeof every === 'string' ? [every, 1] : every
    onEvent.push({ path: l.path, on, n: Math.max(1, n) })
  }
  const names = resolved.paths === undefined ? undefined : new Set(resolved.paths)
  const contractPatterns = resolved.paths?.filter(isPattern) ?? []
  const known = new Map<Path, boolean>()
  return {
    entry,
    known: (path) => {
      if (names === undefined) return true
      let k = known.get(path)
      if (k === undefined) {
        k = names.has(path) || contractPatterns.some((p) => matchesPath(p, path))
        known.set(path, k)
      }
      return k
    },
    due: (path, t) => {
      const e = entry(path)
      if (e === undefined) return false
      const every = e.spec.every
      if (every === undefined) return true
      if (typeof every === 'number') return every <= 1 || t % every === 0
      if (every === 'logSpaced') return t === 0 || (t & (t - 1)) === 0
      return pending.has(e.path)
    },
    fired: (ev, t) => {
      const c = (counts.get(ev) ?? 0) + 1
      counts.set(ev, c)
      const fired = steps.get(ev)
      if (fired === undefined) steps.set(ev, [t])
      else fired.push(t)
      for (const l of onEvent) if (l.on === ev && c % l.n === 0) pending.add(l.path)
    },
    consume: (path) => {
      const e = entry(path)
      if (e !== undefined) pending.delete(e.path)
    },
    // A live view: the array grows as the event fires (Session.stepsOf copies it).
    stepsOf: (ev) => steps.get(ev) ?? [],
  }
}

/**
 * What an algorithm logs through: values by path, events by path, and taps under a prefix. Every call is a no-op for
 * a path nobody subscribed to, so algorithms log freely. A path the contract does not have is another matter: in a
 * session (not in a bare run, which has no contract) `log` and `emit` throw `MdmlError('value')` for it, since it could
 * only be a typo that would otherwise drop the value silently. `wants` is a question, not a statement, and answers
 * `false` (an algorithm shared by methods may ask about a path only some of them have); a typo in a `wants` guard is
 * caught by the path type, not at run time.
 *
 * The parameter is the type of the contract's paths (`string` unless the method declares them, see
 * {@link defineContract}); then a path that is not one of them does not compile. A contract key that is a pattern
 * (`model/activations/*`) is logged by any path under it.
 *
 * @concept log-contract
 */
export interface LogContext<K extends string = string> {
  readonly enabled: boolean
  /** Whether a path is subscribed and due now (so a computation nobody asked for can be skipped). */
  wants(path: Concrete<K>): boolean
  /** Log a value against a path: a scalar, a tensor, a message, a frame, by the contract's kind or the value's type. */
  log(path: Concrete<K>, value: LogValue): void
  /** Fire an event: hooks run, schedules on it advance, and it is recorded as a message when subscribed. */
  emit(path: Concrete<K>, context?: Json): void
  /** A tap context whose names go under `prefix` (`model/activations`), enabled only when something there is due. */
  taps(prefix: Path): TapContext
}

type Kind = 'scalar' | 'tensor' | 'message' | 'frame' | 'derived' | 'event' | undefined

function emitValue(run: RunHandle, t: number, path: Path, kind: Kind, role: TensorRole, value: LogValue): void {
  const asKind: Kind =
    kind === undefined || kind === 'derived'
      ? typeof value === 'number' || typeof value === 'boolean'
        ? 'scalar'
        : isTensor(value) ||
            ArrayBuffer.isView(value) ||
            (Array.isArray(value) && value.every((v) => typeof v === 'number' || Array.isArray(v)))
          ? 'tensor'
          : typeof value === 'object' && value !== null && (value as { readonly kind?: unknown }).kind === 'scene'
            ? 'frame'
            : 'message'
      : kind
  switch (asKind) {
    case 'scalar':
      run.metric(t, path, typeof value === 'boolean' ? (value ? 1 : 0) : (value as number))
      return
    case 'tensor':
      run.tensor(t, path, role, value as Value | ArrayLike<number>)
      return
    case 'frame':
      run.frame(t, value as Json)
      return
    default:
      run.message(t, path, value as Json)
  }
}

/**
 * The log context of a session at a step: `wants` and `log` follow the resolved subscription's schedules, `emit`
 * counts events, records them when subscribed and runs their hooks, and `taps` logs activations under a prefix.
 *
 * @param run - The run to log into.
 * @param resolved - The resolved subscription.
 * @param sched - Its scheduler.
 * @param t - The step.
 * @param control - Where a hook's `stop` goes.
 * @returns The context.
 */
export function logContext<S>(
  run: RunHandle,
  resolved: Resolved<S>,
  sched: Scheduler<S>,
  t: number,
  control: { stop(reason: string): void },
): LogContext {
  const enabled = run.enabled
  // `wants` is a question, and an algorithm shared by methods may ask about a path only some of them have (CHAID's
  // `tree/merge`), so it answers `false` for a path the contract lacks. `log` and `emit` are statements: the backstop
  // there is that a path which is not due is either unsubscribed (fine) or not in the contract (a typo that would
  // drop the value silently), and only the second is an error. The check is a lookup remembered per path, made only
  // where the answer is already `false`.
  const wants = (path: Path): boolean => enabled && sched.due(path, t)
  const put = (path: Path, value: LogValue): void => {
    const e = sched.entry(path)
    const role: TensorRole = e?.source.kind === 'tensor' ? e.source.role : 'output'
    emitValue(run, t, path, e?.source.kind, role, value)
  }
  const log = (path: Path, value: LogValue): void => {
    if (!enabled) return
    if (sched.due(path, t)) put(path, value)
    else if (!sched.known(path)) throw new MdmlError('value', `log: the contract has no path '${path}'`)
  }
  const emit = (path: Path, context: Json = {}): void => {
    if (enabled && !sched.known(path)) throw new MdmlError('value', `emit: the contract has no path '${path}'`)
    sched.fired(path, t)
    if (wants(path)) run.message(t, path, context)
    const hooks: readonly Hook[] = resolved.hooks.get(path) ?? []
    for (const hook of hooks) hook({ t, path, context, run, log, stop: control.stop })
  }
  return {
    enabled,
    wants,
    log,
    emit,
    taps: (prefix) => {
      const under = `${prefix}/`
      const live =
        enabled &&
        resolved.logs.some((l) => (isPattern(l.path) ? matchesPath(l.path, `${under}_`) : l.path.startsWith(under)))
      const inner = tapContext(live)
      return {
        enabled: live,
        entries: inner.entries,
        tap: (name, value, role = 'activation') => {
          // Tap names are only known as the model runs: unsubscribed ones are dropped, never an error.
          if (live && sched.due(`${under}${name}`, t)) put(`${under}${name}`, value)
          return inner.tap(name, value, role)
        },
      }
    },
  }
}

/**
 * The log context runners give an algorithm outside a session: paths are channel names under the run's own policy,
 * values are typed by what they are, events are recorded as messages when the policy selects them, and taps record
 * activations when the policy asks for that role.
 *
 * @param run - The run.
 * @param t - The step.
 * @returns The context.
 * @example
 * ```ts
 * import { defaultLogContext, recorder, startRun } from 'mdml-engine/trace'
 * const rec = recorder()
 * defaultLogContext(startRun(rec, { name: 'x', seed: 0, config: {} }), 3).log('a/loss', 0.5)
 * rec.events.at(-1) // => { type: 'metric', runId: 'x:0', step: 3, name: 'a/loss', value: 0.5 }
 * ```
 */
export function defaultLogContext(run: RunHandle, t: number): LogContext {
  const wants = (path: Path): boolean =>
    run.enabled && (run.wants('metric', path) || run.wants('message', path) || run.wantsTensor(path, 'output'))
  return {
    enabled: run.enabled,
    wants,
    log: (path, value) => {
      if (run.enabled) emitValue(run, t, path, undefined, 'output', value)
    },
    emit: (path, context = {}) => {
      if (run.wants('message', path)) run.message(t, path, context)
    },
    taps: (prefix) => {
      const live = run.enabled && run.policy.roles.includes('activation') && run.due(t)
      const inner = tapContext(live)
      return {
        enabled: live,
        entries: inner.entries,
        tap: (name, value, role = 'activation') => {
          if (live && isTensor(value)) run.tensor(t, `${prefix}/${name}`, role, value)
          return inner.tap(name, value, role)
        },
      }
    },
  }
}

/**
 * A tracer that lets through only the events of subscribed paths when they are due, renaming native channels
 * (`objective`) to their contract paths (`smo/dual-objective`) on the way. Run, end, checkpoint, episode and log
 * events always pass. A path that keeps a window (`Log({ keep })`) has its oldest events forgotten by the destination
 * as new ones arrive, when the destination is a recorder; a destination that cannot forget holds all it is sent.
 *
 * @param tracer - The destination.
 * @param resolved - The resolved subscription.
 * @param sched - Its scheduler (shared with the log context, so a path is consumed once).
 * @returns The gating tracer.
 * @example
 * ```ts
 * import { gated, Log, recorder, resolve, scalar, scheduler, startRun } from 'mdml-engine/trace'
 * const r = resolve({ 'a/x': scalar() }, { 'a/x': Log({ keep: 2 }) })
 * const rec = recorder()
 * const run = startRun(gated(rec, r, scheduler(r)), { name: 'x', seed: 0, config: {}, policy: r.policy })
 * for (let t = 0; t < 5; t++) run.metric(t, 'a/x', t)
 * rec.events.flatMap((e) => (e.type === 'metric' ? [e.step] : [])) // => [3, 4]
 * ```
 */
export function gated<S>(tracer: Tracer, resolved: Resolved<S>, sched: Scheduler<S>): Tracer {
  const forget: ((event: TraceEvent) => void) | undefined = 'forget' in tracer ? (tracer as Recorder).forget : undefined
  const native = { metric: new Map<string, Path>(), tensor: new Map<string, Path>(), message: new Map<string, Path>() }
  let framePath: Path | undefined
  for (const l of resolved.logs) {
    const s = l.source
    if (s.kind === 'frame' && framePath === undefined) framePath = l.path
    if ((s.kind === 'scalar' || s.kind === 'tensor' || s.kind === 'message') && s.from !== undefined) {
      native[s.kind === 'scalar' ? 'metric' : s.kind].set(s.from, l.path)
    }
  }
  // Derived paths may become frames too; without a frame source, a derived scene still needs a name.
  const derivedFrame = resolved.logs.find((l) => l.source.kind === 'derived')?.path
  const pathOf = (table: Map<string, Path>, name: string): Path | undefined => {
    const renamed = table.get(name)
    if (renamed !== undefined) return renamed
    return sched.entry(name) === undefined ? undefined : name
  }
  const pass = (path: Path | undefined, step: number): path is Path => {
    if (path === undefined || !sched.due(path, step)) return false
    sched.consume(path)
    return true
  }
  // A path's byte budget: its events stop once the JSON of those held passes it.
  const spent = new Map<Path, number>()
  // A path's window: the events held under `keep`, oldest first, so the oldest can be forgotten as a new one arrives.
  const windows = new Map<Path, { readonly event: TraceEvent; readonly size: number }[]>()
  const emitPath = (path: Path, event: TraceEvent): void => {
    const spec = sched.entry(path)?.spec
    const budget = spec?.budget
    const size = budget === undefined ? 0 : JSON.stringify(event).length
    if (budget !== undefined) {
      const held = (spent.get(path) ?? 0) + size
      if (held > budget) return
      spent.set(path, held)
    }
    tracer.emit(event)
    const keep = spec?.keep
    if (forget === undefined || keep === undefined || keep === 'all') return
    const n = keep === 'last' ? 1 : keep
    const window = windows.get(path) ?? []
    windows.set(path, window)
    window.push({ event, size })
    while (window.length > n) {
      const oldest = window.shift()
      if (oldest === undefined) break
      forget(oldest.event)
      if (budget !== undefined) spent.set(path, (spent.get(path) ?? 0) - oldest.size)
    }
  }
  return {
    enabled: tracer.enabled,
    emit: (e: TraceEvent) => {
      switch (e.type) {
        case 'metric': {
          const path = pathOf(native.metric, e.name)
          if (pass(path, e.step)) emitPath(path, { ...e, name: path })
          return
        }
        case 'tensor': {
          const path = pathOf(native.tensor, e.name)
          if (!pass(path, e.step)) return
          // A path that asked for summaries keeps them, even when another path turned full values on for the policy.
          const full = sched.entry(path)?.spec.detail === 'full'
          if (full || e.values === undefined) {
            emitPath(path, { ...e, name: path })
            return
          }
          const { values, indices: _indices, ...rest } = e
          emitPath(path, { ...rest, name: path, summary: e.summary ?? summarise(values) })
          return
        }
        case 'message': {
          const path = pathOf(native.message, e.kind)
          if (pass(path, e.step)) emitPath(path, { ...e, kind: path })
          return
        }
        case 'frame': {
          const path = framePath ?? derivedFrame
          if (pass(path, e.step)) emitPath(path, e)
          return
        }
        default:
          tracer.emit(e)
      }
    },
  }
}
