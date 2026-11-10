/**
 * The small convention every inference method of the probabilistic layer follows, and nothing more: there is no
 * universal `infer` and no automatic choice here. Each method (NUTS, EP, SVI, Gibbs, ...) is its own bespoke thing with its
 * own knobs, its own tape and its own state; what they share is a handful of conventions so that a page, a worker and a
 * test can drive any of them the same way:
 * - a `Method` of the trace library (`defineMethod`): a `Space` of knobs with defaults, a typed contract of tape paths
 *   (`defineContract`), and an `Algorithm` whose step is small enough to show and to cancel, whose state serialises so a
 *   run resumes, seeks and forks;
 * - a `family` label, a `description`, and an optional `check` that says whether the method applies to a model and, if
 *   not, why and what is missing (the method decides; nothing selects it for the user);
 * - an optional `posterior(state)`, so that any state (the last one, or a snapshot taken while running) reads as a
 *   {@link Posterior};
 * - **a budget the user selects and anytime results.** `budgetMs`, `budgetSteps`, `snapshotEveryMs` and
 *   `snapshotEverySteps` are knobs every inference method has (`withAnytime` adds them to a method's own knobs);
 *   `runAnytime` drives a session within the budget and hands a snapshot of the state to a callback about once a second
 *   (the default), so a page shows a trace or a posterior filling in rather than waiting for the end. The library never
 *   reads a clock (traces are deterministic): a time budget or a time-based snapshot needs a `clock` injected by the
 *   caller (the page's `performance.now`), and only moves where slices end, never what a run computes.
 *
 * @index structure InferenceMethod: knobs, typed tape, check, budget and anytime snapshots, the shared convention.
 */

import { OmaiError } from '../../base/index.ts'
import { type DictSpace, dict, int, real } from '../../space/index.ts'
import { type Config, defineMethod, type Method, type RunStatus, type Session } from '../../trace/index.ts'
import type { SpecAnalysis } from '../model/analysis.ts'
import type { Posterior } from '../posterior.ts'

/** What kind of answer a method gives; a label for listings and explanations, not a switch. */
export type MethodFamily =
  | 'exact'
  | 'message-passing'
  | 'variational'
  | 'mcmc'
  | 'smc'
  | 'nested'
  | 'likelihood-free'
  | 'composite'

/**
 * Whether a method applies to a model, and why not when it does not: the structured reason a page shows beside a
 * disabled method. `missing` lists the features the model lacks that the method needs (`finite discrete latents`,
 * `exponential-family factors`, `differentiable density`, ...) and `hint` what would help (marginalise the discrete
 * sites, use another method).
 */
export interface Verdict {
  readonly applies: boolean
  readonly reason: string
  readonly missing: readonly string[]
  readonly hint?: string
}

/**
 * An inference method: a `Method` of the trace library plus the label, an optional capability check and an optional
 * reading of a state as a posterior. The model a method runs on is a constructable its factory closes over (as every
 * method of the library does), so a method is made by a function of the bound model, e.g. `nutsMethod(bound)`; the
 * knobs are only what a page's controls may change and are never functions.
 *
 * @concept inference-engines
 */
export interface InferenceMethod<S, C extends Config = Config, K extends string = string, A = SpecAnalysis>
  extends Method<S, C, K> {
  readonly family: MethodFamily
  readonly description: string
  /** Whether the method applies to a model, given its analysis; absent when the method applies to anything. */
  check?(analysis: A): Verdict
  /** The posterior a state stands for (partial while running); the thing an anytime snapshot shows. */
  posterior?(state: S, config: C): Posterior
}

/**
 * The knobs of the budget and the snapshots: every inference method has them. A limit of `0` means none.
 *
 * - `budgetMs`: stop after this many milliseconds (needs a clock);
 * - `budgetSteps`: stop after this many steps of the method;
 * - `snapshotEveryMs`: hand out a snapshot this often (default 1000; needs a clock);
 * - `snapshotEverySteps`: hand out a snapshot this often in steps (`0`: only by time).
 */
export const anytimeKnobs: DictSpace = dict({
  budgetMs: real(0, 3_600_000, {
    default: 0,
    unit: 'ms',
    label: 'time budget',
    description: 'stop after this long (0: no limit)',
  }),
  budgetSteps: int(0, 1_000_000_000, {
    default: 0,
    label: 'step budget',
    description: 'stop after this many steps (0: no limit)',
  }),
  snapshotEveryMs: real(0, 3_600_000, {
    default: 1000,
    unit: 'ms',
    label: 'snapshot interval',
    description: 'hand out the current state this often while running (0: not by time)',
  }),
  snapshotEverySteps: int(0, 1_000_000_000, {
    default: 0,
    label: 'snapshot every steps',
    description: 'hand out the current state every this many steps (0: not by steps)',
  }),
})

/**
 * A method's own knobs with the budget and snapshot knobs added.
 *
 * @param own - The method's knobs.
 * @returns A dict with the method's knobs and the {@link anytimeKnobs}.
 * @throws OmaiError with code `value` when a knob of the method has the name of an anytime knob.
 * @concept inference-engines
 * @example
 * ```ts
 * import { withAnytime } from 'omai-engine/probprog/inference'
 * import { dict, int } from 'omai-engine/space'
 * Object.keys(withAnytime(dict({ numDraws: int(1, 10000, { default: 500 }) })).spaces) // => ['numDraws', 'budgetMs', 'budgetSteps', 'snapshotEveryMs', 'snapshotEverySteps']
 * ```
 */
export function withAnytime(own: DictSpace): DictSpace {
  for (const k of Object.keys(anytimeKnobs.spaces))
    if (k in own.spaces) throw new OmaiError('value', `withAnytime: the knob '${k}' is reserved for the budget`)
  return { ...own, spaces: { ...own.spaces, ...anytimeKnobs.spaces } }
}

/**
 * Declare an inference method: checks the family and name, gives the method's knobs the budget and snapshot knobs, and
 * declares it as a trace `Method` (its contract and defaults are checked there).
 *
 * @param method - The method, whose `config` holds only its own knobs.
 * @returns The method, with the anytime knobs added to its `config`.
 * @throws OmaiError with code `value` for an unnamed method or a knob that clashes with the budget's.
 * @concept inference-engines
 * @example
 * ```ts
 * import { defineInferenceMethod } from 'omai-engine/probprog/inference'
 * import { scalar } from 'omai-engine/trace'
 * import { dict } from 'omai-engine/space'
 * const m = defineInferenceMethod({
 *   name: 'count', family: 'exact', description: 'counts', config: dict({}),
 *   make: () => ({ name: 'count', config: {}, init: () => 0, step: (s: number, ctx) => { ctx.log('n', s); return s + 1 } }),
 *   contract: { n: scalar() },
 * })
 * Object.keys(m.config.spaces).includes('budgetMs') // => true
 * ```
 */
export function defineInferenceMethod<S, C extends Config, K extends string = string, A = SpecAnalysis>(
  method: InferenceMethod<S, C, K, A>,
): InferenceMethod<S, C, K, A> {
  if (method.name === '') throw new OmaiError('value', 'defineInferenceMethod: a method needs a name')
  const declared = defineMethod({ ...method, config: withAnytime(method.config) })
  return { ...method, ...declared }
}

/** What a snapshot hands to its callback: the state at a step, how long the run has taken, and whether it is the last. */
export interface Snapshot<S> {
  readonly t: number
  /** Milliseconds since the run was started by `runAnytime` (0 without a clock). */
  readonly elapsedMs: number
  /** The live state (see `Algorithm` on state and mutation: take a copy of what is kept). */
  readonly state: S
  readonly status: RunStatus | undefined
  /** True for the snapshot taken when the run ends (finished, budget spent or stopped). */
  readonly final: boolean
}

/** Options of {@link runAnytime}. Knobs of the session's config are the defaults for `budget` and `snapshot`. */
export interface AnytimeOptions<S> {
  /** Limits for this call: `steps` and `ms` (default the config's `budgetSteps` and `budgetMs`; 0 means none). */
  readonly budget?: { readonly steps?: number; readonly ms?: number }
  /** How often to hand out a snapshot (default the config's `snapshotEveryMs`, 1000, and `snapshotEverySteps`). */
  readonly snapshot?: { readonly everyMs?: number; readonly everySteps?: number }
  /** A millisecond clock (the page's `performance.now`); required by any time budget and by time-based snapshots. */
  readonly clock?: () => number
  readonly onSnapshot?: (snapshot: Snapshot<S>) => void
  /** Asked between slices; true cancels the run (it ends `stopped` through the session's own stop). */
  readonly shouldStop?: () => boolean
}

/** The outcome of {@link runAnytime}: where the session stands and how it ended. */
export interface AnytimeResult {
  readonly steps: number
  readonly t: number
  readonly elapsedMs: number
  readonly snapshots: number
  readonly status: RunStatus | undefined
  /** Why this call returned: the run ended, the budget was spent, or `shouldStop` said so. */
  readonly reason: 'ended' | 'budget' | 'cancelled'
}

const knob = (config: Config, name: string): number => {
  const v = config[name]
  return typeof v === 'number' ? v : 0
}

/**
 * Drive a session within a user-selected budget, handing out a snapshot of its state about once a second (the default
 * `snapshotEveryMs`) so a page can show the current trace or posterior while the run goes on, and stopping early when
 * asked. It advances the session in slices of at most one snapshot interval, so a cancellation or a budget takes
 * effect within a slice, and a method's own step (one iteration of every chain, one sweep) is the finest grain. Calling
 * it again continues the session: budgets count from the call.
 *
 * @param s - The session of an inference method (or of any method with the anytime knobs).
 * @param options - Budget, snapshot interval, clock, callback and cancellation.
 * @returns How far the run got and why it returned.
 * @throws OmaiError with code `value` for a time budget or time-based snapshot without a clock.
 * @concept inference-engines
 * @example
 * ```ts
 * import { runAnytime, defineInferenceMethod } from 'omai-engine/probprog/inference'
 * import { scalar, session } from 'omai-engine/trace'
 * import { dict } from 'omai-engine/space'
 * const m = defineInferenceMethod({
 *   name: 'count', family: 'exact', description: 'counts', config: dict({}),
 *   make: () => ({ name: 'count', config: {}, init: () => 0, step: (s: number, ctx) => { ctx.log('n', s); return s + 1 } }),
 *   contract: { n: scalar() },
 * })
 * const seen: number[] = []
 * const r = runAnytime(session(m, { steps: 100 }), { budget: { steps: 10 }, snapshot: { everySteps: 4 }, onSnapshot: (x) => seen.push(x.t) })
 * [r.reason, r.t, seen] // => ['budget', 10, [4, 8, 10]]
 * ```
 */
export function runAnytime<S, C extends Config, K extends string = string>(
  s: Session<S, C, K>,
  options: AnytimeOptions<S> = {},
): AnytimeResult {
  const config: Config = s.config
  const limitSteps = options.budget?.steps ?? knob(config, 'budgetSteps')
  const limitMs = options.budget?.ms ?? knob(config, 'budgetMs')
  const everyMs =
    options.snapshot?.everyMs ?? (config['snapshotEveryMs'] === undefined ? 1000 : knob(config, 'snapshotEveryMs'))
  const everySteps = options.snapshot?.everySteps ?? knob(config, 'snapshotEverySteps')
  const clock = options.clock
  if (limitMs > 0 && clock === undefined)
    throw new OmaiError('value', 'runAnytime: a time budget needs a clock (the library reads none)')
  const byTime = everyMs > 0 && clock !== undefined
  const t0 = s.t
  const started = clock?.() ?? 0
  const elapsed = (): number => (clock === undefined ? 0 : clock() - started)
  const spent = (): boolean =>
    (limitSteps > 0 && s.t - t0 >= limitSteps) || (limitMs > 0 && clock !== undefined && elapsed() >= limitMs)
  let snapshots = 0
  const emit = (final: boolean): void => {
    snapshots += 1
    options.onSnapshot?.({ t: s.t, elapsedMs: elapsed(), state: s.state, status: s.status, final })
  }
  let cancelled = false
  while (s.status === undefined && !spent()) {
    if (options.shouldStop?.() === true) {
      cancelled = true
      break
    }
    let steps = Number.POSITIVE_INFINITY
    if (limitSteps > 0) steps = Math.min(steps, limitSteps - (s.t - t0))
    if (everySteps > 0) steps = Math.min(steps, everySteps)
    let ms = Number.POSITIVE_INFINITY
    if (byTime) ms = Math.min(ms, everyMs)
    if (limitMs > 0) ms = Math.min(ms, limitMs - elapsed())
    s.advance({
      ...(Number.isFinite(steps) ? { steps } : {}),
      ...(Number.isFinite(ms) && clock !== undefined ? { ms, clock } : {}),
    })
    if (s.status === undefined && !spent() && (byTime || everySteps > 0)) emit(false)
  }
  emit(true)
  const reason: AnytimeResult['reason'] = cancelled ? 'cancelled' : s.status !== undefined ? 'ended' : 'budget'
  return { steps: s.t - t0, t: s.t, elapsedMs: elapsed(), snapshots, status: s.status, reason }
}

/** Options of {@link runAnytimeAsync}: those of {@link runAnytime}, and how the run shares the page. */
export interface AnytimeAsyncOptions<S> extends AnytimeOptions<S> {
  /** The most milliseconds computed before the page is given the thread back (default 40); needs the `clock`. */
  readonly sliceMs?: number
  /**
   * Resolves when the page may run again (a timer or a frame): the library has no timers, so the caller
   * says how a slice hands over, and what it awaits is where the page paints, handles a click and answers `shouldStop`.
   */
  readonly yieldToPage: () => Promise<void>
}

/**
 * Drive a session within a budget as {@link runAnytime} does, in slices that hand the thread back to the page between
 * them, so a page keeps painting and answering clicks while it runs: the snapshot callback fires about once a second
 * (the default), the page draws what it was given, and `shouldStop` is asked after every slice rather than once a
 * second. A slice is at most `sliceMs` (a method's own step is the finest grain); the run computes exactly what
 * `runAnytime` and an uninterrupted session do, only the places where it pauses differ.
 *
 * @param s - The session of an inference method.
 * @param options - Budget, snapshot interval, clock, callback, cancellation, the slice length and the hand-over.
 * @returns How far the run got and why it returned.
 * @throws OmaiError with code `value` for a time budget, time-based snapshot or slice without a clock.
 * @concept inference-engines
 * @index function runAnytime in slices that hand the page the thread: budget, snapshots, cancellable.
 * @example
 * ```ts
 * import { defineInferenceMethod, runAnytimeAsync } from 'omai-engine/probprog/inference'
 * import { scalar, session } from 'omai-engine/trace'
 * import { dict } from 'omai-engine/space'
 * const m = defineInferenceMethod({
 *   name: 'count', family: 'exact', description: 'counts', config: dict({}),
 *   make: () => ({ name: 'count', config: {}, init: () => 0, step: (s: number, ctx) => { ctx.log('n', s); return s + 1 } }),
 *   contract: { n: scalar() },
 * })
 * // A page awaits it (and reads the snapshots as they come); here the promise is only made.
 * const r = runAnytimeAsync(session(m, { steps: 100 }), {
 *   budget: { steps: 10 }, snapshot: { everySteps: 4 }, yieldToPage: () => Promise.resolve(),
 * })
 * typeof r.then // => 'function'
 * ```
 */
export async function runAnytimeAsync<S, C extends Config, K extends string = string>(
  s: Session<S, C, K>,
  options: AnytimeAsyncOptions<S>,
): Promise<AnytimeResult> {
  const config: Config = s.config
  const limitSteps = options.budget?.steps ?? knob(config, 'budgetSteps')
  const limitMs = options.budget?.ms ?? knob(config, 'budgetMs')
  const everyMs =
    options.snapshot?.everyMs ?? (config['snapshotEveryMs'] === undefined ? 1000 : knob(config, 'snapshotEveryMs'))
  const everySteps = options.snapshot?.everySteps ?? knob(config, 'snapshotEverySteps')
  const clock = options.clock
  const sliceMs = options.sliceMs ?? 40
  if (limitMs > 0 && clock === undefined)
    throw new OmaiError('value', 'runAnytimeAsync: a time budget needs a clock (the library reads none)')
  if (clock === undefined && options.sliceMs !== undefined)
    throw new OmaiError('value', 'runAnytimeAsync: a slice length needs a clock (the library reads none)')
  const byTime = everyMs > 0 && clock !== undefined
  const t0 = s.t
  const started = clock?.() ?? 0
  const elapsed = (): number => (clock === undefined ? 0 : clock() - started)
  const spent = (): boolean =>
    (limitSteps > 0 && s.t - t0 >= limitSteps) || (limitMs > 0 && clock !== undefined && elapsed() >= limitMs)
  let snapshots = 0
  let lastMs = 0
  let lastT = s.t
  const emit = (final: boolean): void => {
    snapshots += 1
    lastMs = elapsed()
    lastT = s.t
    options.onSnapshot?.({ t: s.t, elapsedMs: lastMs, state: s.state, status: s.status, final })
  }
  let cancelled = false
  while (s.status === undefined && !spent()) {
    if (options.shouldStop?.() === true) {
      cancelled = true
      break
    }
    let steps = Number.POSITIVE_INFINITY
    if (limitSteps > 0) steps = Math.min(steps, limitSteps - (s.t - t0))
    if (everySteps > 0) steps = Math.min(steps, everySteps - (s.t - lastT))
    let ms = Number.POSITIVE_INFINITY
    if (clock !== undefined) ms = sliceMs
    if (limitMs > 0) ms = Math.min(ms, limitMs - elapsed())
    s.advance({
      ...(Number.isFinite(steps) ? { steps: Math.max(1, steps) } : {}),
      ...(Number.isFinite(ms) && clock !== undefined ? { ms: Math.max(1e-6, ms), clock } : {}),
    })
    const due = (byTime && elapsed() - lastMs >= everyMs) || (everySteps > 0 && s.t - lastT >= everySteps)
    if (s.status === undefined && !spent() && due) emit(false)
    if (s.status === undefined && !spent()) await options.yieldToPage()
  }
  emit(true)
  const reason: AnytimeResult['reason'] = cancelled ? 'cancelled' : s.status !== undefined ? 'ended' : 'budget'
  return { steps: s.t - t0, t: s.t, elapsedMs: elapsed(), snapshots, status: s.status, reason }
}
