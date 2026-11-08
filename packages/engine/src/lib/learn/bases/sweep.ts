/**
 * The sweep and repeated-fit base: a study that runs an inner `Method` as a `session` over a grid or a sample of some of
 * its knobs, one trial per step, as an `Algorithm` that serialises (so a study can be paused, sought and forked like
 * any run). A trial is one configuration at one repeat; repeats are fresh keys on the same configuration, so the study
 * of one estimator's variance is a sweep over no knobs with many repeats. The study logs each trial's knob values and
 * final metrics (`trial/<knob>`, `trial/<metric>`), the running mean and spread of the configuration's repeats
 * (`config/<metric>/mean`, `config/<metric>/std`) and the best objective so far (`trial/best`); the inner tapes are
 * kept on request (`keep`) and otherwise summarised in the trial's record. The race is the sweep over a slot done in
 * lockstep: one method whose single run drives several variants of a configuration with the same key, the same data
 * and the same step keys, logging each under `race/<variant>/`.
 *
 * @index structure Sweep and race: run an inner method over a grid of knobs.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import { choice, type DictSpace, dict, grid, int, type Space, type SpaceValue, sample } from '../../space/index.ts'
import {
  type Algorithm,
  type Concrete,
  type Config,
  type Contract,
  contractOf,
  defineMethod,
  derived,
  type EventSource,
  Log,
  type LogSpec,
  type Method,
  message,
  metricSeries,
  mountContext,
  type Pattern,
  type Prefixed,
  type Run,
  type RunStatus,
  type SessionOptions,
  type Source,
  scalar,
  session,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'

/** What of the inner tapes a sweep keeps: none (summaries only), the last trial's, or every trial's. */
export type SweepKeep = 'none' | 'last' | 'all'

/** Which way is better for the objective of a sweep. */
export type SweepDirection = 'lower' | 'higher'

/**
 * The paths of a sweep: each trial's knob values and metrics (`trial/<knob>`, `trial/<metric>`), the configuration's
 * spread (`config/<metric>/mean`, `config/<metric>/std`), the best objective so far, and the derived summaries.
 */
export type SweepPath =
  | 'trial/config'
  | 'trial/repeat'
  | 'trial/best'
  | `trial/${string}`
  | `config/${string}/mean`
  | `config/${string}/std`
  | 'sweep/configurations'
  | 'sweep/best'

/** The paths of a race of variants of a method with paths `K`: each variant's own, under `race/<variant>/`. */
export type RacePath<K extends string = string> = Prefixed<`race/${string}`, K>

/** Subscriptions of `LogSpec`s over the paths `K` (any string key for `string`; a plain key must be one of `K`). */
type LogMap<K extends string> = string extends K
  ? Readonly<Record<string, LogSpec>>
  : Readonly<Partial<Record<K, LogSpec>>> & Readonly<Record<Pattern, LogSpec>>

/**
 * What a sweep is told: the inner method, the knobs to vary and how, what to read from each trial, what to keep. The
 * last parameter is the type of the inner method's paths (`string`: any); with literal paths, `metrics`, `final` and
 * `logs` are checked against them.
 */
export interface SweepSpec<S, C extends Config, K extends string = string> {
  /** The inner method, run as a session for every trial. */
  readonly method: Method<S, C, K>
  /** The knobs to vary, as a dict over some of the method's knob names (see {@link knobsOf}); none for a repeated fit. */
  readonly vary?: DictSpace
  /** The configurations to try, in place of a grid or sample of `vary`. */
  readonly configs?: readonly Partial<C>[]
  /** Knobs held at these values in every trial, over the method's defaults. */
  readonly fixed?: Partial<C>
  /** `grid` (default) or `sample` of `vary`. */
  readonly design?: 'grid' | 'sample'
  /** Points per continuous knob of the grid (default 5). */
  readonly points?: number
  /** Configurations drawn by `sample` (default 10). */
  readonly trials?: number
  /** Fresh keys per configuration (default 1); the spread of the metrics over them is logged. */
  readonly repeats?: number
  /** Steps of each inner session (default 1000). */
  readonly steps?: number
  /** Metric paths of the inner method read at the end of each trial (the last value logged). */
  readonly metrics: readonly NoInfer<Concrete<K>>[]
  /** Derived paths of the inner method computed on each trial's final state; they must give numbers. */
  readonly final?: readonly NoInfer<K>[]
  /** What `trial/best` and the best configuration are judged by (default the first metric, lower). */
  readonly objective?: { readonly metric?: string; readonly direction?: SweepDirection }
  /** What of the inner tapes to keep (default none: the trial's record summarises them). */
  readonly keep?: SweepKeep
  /** Extra subscriptions for the inner sessions, so the kept tapes hold more than the metrics. */
  readonly logs?: LogMap<NoInfer<K>>
  /**
   * Whether repeat `r` uses the same key in every configuration (default `true`: the same data and start, so the only
   * difference between configurations is the configuration), or every trial a key of its own.
   */
  readonly paired?: boolean
}

/** A series of one logged path, summarised: how many values, the first and last, the extremes. */
export interface SeriesSummary {
  readonly n: number
  readonly first: number
  readonly last: number
  readonly min: number
  readonly max: number
}

/** One trial: a configuration at a repeat, what the inner session ended with, and a summary of its tape. */
export interface TrialRecord {
  readonly index: number
  /** The configuration's index in the design. */
  readonly config: number
  readonly repeat: number
  /** The varied knobs' values. */
  readonly values: SpaceValue
  /** The final value of every metric and derived path asked for. */
  readonly metrics: Readonly<Record<string, number>>
  /** The inner session's steps and status. */
  readonly steps: number
  readonly status: RunStatus | null
  /** The metric paths' series, summarised (the tape itself is in `tapes` when kept). */
  readonly series: Readonly<Record<string, SeriesSummary>>
}

/** The state of a sweep: the design, the trials so far, the best objective, and the inner tapes kept. */
export interface SweepState {
  /** The study's key; repeat `r` of a paired study runs under `foldIn(key, r)`. */
  readonly key: Key
  readonly configs: readonly SpaceValue[]
  readonly trials: readonly TrialRecord[]
  /** The inner tapes kept (`keep`): the last trial's or every trial's, in order. */
  readonly tapes: readonly Run[]
  /** The best objective value of any trial so far (null before one has a finite value). */
  readonly best: number | null
  readonly objective: { readonly metric: string; readonly direction: SweepDirection }
}

/** A configuration with the metrics of its repeats run so far, summarised. */
export interface ConfigurationSummary {
  readonly index: number
  readonly values: SpaceValue
  /** The repeats run so far. */
  readonly n: number
  readonly mean: Readonly<Record<string, number>>
  /** The sample standard deviation over the repeats (0 for one). */
  readonly std: Readonly<Record<string, number>>
}

/**
 * The knobs of a method as a dict space to vary: its own spaces for the names given (a condition on a sibling that is
 * not among them is dropped), optionally narrowed to other ranges.
 *
 * @param method - The method.
 * @param names - The knobs to vary.
 * @param ranges - Spaces to use in place of the method's own for some of them.
 * @returns The dict space.
 * @throws OmaiError with code `value` for a name the method has no knob of.
 * @example
 * ```ts
 * import { knobsOf } from 'omai-engine/learn'
 * import { dict, real } from 'omai-engine/space'
 * const method = { name: 'm', config: dict({ alpha: real(0, 1), beta: real(0, 1) }), make: () => { throw new Error() }, contract: {} }
 * Object.keys(knobsOf(method, ['alpha']).spaces) // => ['alpha']
 * ```
 */
export function knobsOf<S, C extends Config, K extends string = string>(
  method: Method<S, C, K>,
  names: readonly string[],
  ranges: Readonly<Record<string, Space>> = {},
): DictSpace {
  const out: Record<string, Space> = {}
  for (const name of names) {
    const own = ranges[name] ?? method.config.spaces[name]
    if (own === undefined) throw new OmaiError('value', `knobsOf: ${method.name} has no knob '${name}'`)
    const { when, ...rest } = own
    out[name] = when !== undefined && !names.includes(when.key) ? (rest as Space) : own
  }
  return dict(out)
}

const isRecord = (v: unknown): v is Readonly<Record<string, SpaceValue>> =>
  typeof v === 'object' && v !== null && !Array.isArray(v)

const knobNames = <S, C extends Config, K extends string>(spec: SweepSpec<S, C, K>): string[] =>
  spec.vary !== undefined
    ? Object.keys(spec.vary.spaces)
    : [...new Set((spec.configs ?? []).flatMap((c) => Object.keys(c)))]

/** The configurations a spec tries, given the study's key (only `sample` uses it). */
function designOf<S, C extends Config, K extends string>(spec: SweepSpec<S, C, K>, k: Key): SpaceValue[] {
  if (spec.configs !== undefined) return spec.configs.map((c) => ({ ...c }) as SpaceValue)
  const vary = spec.vary ?? dict({})
  if ((spec.design ?? 'grid') === 'sample') {
    const n = spec.trials ?? 10
    return Array.from({ length: n }, (_, i) => sample(vary, foldIn(k, 100 + i)))
  }
  return grid(vary, spec.points ?? 5)
}

/**
 * How many trials a sweep runs: configurations times repeats.
 *
 * @param spec - The sweep.
 * @returns The number of steps of the algorithm.
 * @example
 * ```ts
 * import { sweepTrials } from 'omai-engine/learn'
 * import { dict, real } from 'omai-engine/space'
 * const method = { name: 'm', config: dict({ a: real(0, 1) }), make: () => { throw new Error() }, contract: {} }
 * sweepTrials({ method, vary: dict({ a: real(0, 1) }), points: 4, repeats: 3, metrics: [] }) // => 12
 * ```
 */
export function sweepTrials<S, C extends Config, K extends string = string>(spec: SweepSpec<S, C, K>): number {
  const repeats = spec.repeats ?? 1
  if (spec.configs !== undefined) return spec.configs.length * repeats
  if ((spec.design ?? 'grid') === 'sample') return (spec.trials ?? 10) * repeats
  return grid(spec.vary ?? dict({}), spec.points ?? 5).length * repeats
}

const finite = (x: number): boolean => Number.isFinite(x)

function summariseSeries(values: readonly number[]): SeriesSummary {
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (const v of values) {
    if (v < min) min = v
    if (v > max) max = v
  }
  return { n: values.length, first: values[0] ?? Number.NaN, last: values.at(-1) ?? Number.NaN, min, max }
}

const better = (a: number, b: number, direction: SweepDirection): boolean =>
  finite(a) && (!finite(b) || (direction === 'lower' ? a < b : a > b))

/**
 * Each configuration of a sweep with the mean and spread of its repeats so far.
 *
 * @param state - The sweep's state.
 * @returns One summary per configuration that has a trial, in design order.
 * @example
 * ```ts
 * import { sweepConfigurations } from 'omai-engine/learn'
 * const state = {
 *   key: { kind: 'key' as const, words: [0, 0] as [number, number] },
 *   configs: [{ a: 1 }], tapes: [], best: 2, objective: { metric: 'm', direction: 'lower' as const },
 *   trials: [2, 4].map((m, i) => ({ index: i, config: 0, repeat: i, values: { a: 1 }, metrics: { m }, steps: 1, status: null, series: {} })),
 * }
 * sweepConfigurations(state)[0]?.mean // => { m: 3 }
 * ```
 */
export function sweepConfigurations(state: SweepState): ConfigurationSummary[] {
  const out: ConfigurationSummary[] = []
  for (const [index, values] of state.configs.entries()) {
    const trials = state.trials.filter((t) => t.config === index)
    if (trials.length === 0) continue
    const names = Object.keys(trials[0]?.metrics ?? {})
    const mean: Record<string, number> = {}
    const std: Record<string, number> = {}
    for (const name of names) {
      const xs = trials.map((t) => t.metrics[name] ?? Number.NaN)
      const m = xs.reduce((a, b) => a + b, 0) / xs.length
      mean[name] = m
      std[name] = xs.length > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)) : 0
    }
    out.push({ index, values, n: trials.length, mean, std })
  }
  return out
}

/**
 * The best configuration of a sweep by the mean of its objective over the repeats run so far.
 *
 * @param state - The sweep's state.
 * @returns The configuration's summary, or undefined before any trial has a finite objective.
 * @example
 * ```ts
 * import { sweepBest } from 'omai-engine/learn'
 * const trial = (index: number, a: number, m: number) =>
 *   ({ index, config: index, repeat: 0, values: { a }, metrics: { m }, steps: 1, status: null, series: {} })
 * const state = {
 *   key: { kind: 'key' as const, words: [0, 0] as [number, number] },
 *   configs: [{ a: 1 }, { a: 2 }], tapes: [], best: 1, objective: { metric: 'm', direction: 'lower' as const },
 *   trials: [trial(0, 1, 5), trial(1, 2, 1)],
 * }
 * sweepBest(state)?.values // => { a: 2 }
 * ```
 */
export function sweepBest(state: SweepState): ConfigurationSummary | undefined {
  let best: ConfigurationSummary | undefined
  for (const c of sweepConfigurations(state)) {
    const value = c.mean[state.objective.metric] ?? Number.NaN
    if (best === undefined || better(value, best.mean[state.objective.metric] ?? Number.NaN, state.objective.direction))
      best = c
  }
  return best !== undefined && finite(best.mean[state.objective.metric] ?? Number.NaN) ? best : undefined
}

/** The path a knob's value is logged under, and what kind of path it is. */
function knobSource(space: Space | undefined): Source<SweepState> {
  if (space?.kind === 'real' || space?.kind === 'int' || space?.kind === 'bool' || space?.kind === 'discrete')
    return scalar({ description: 'the knob’s value in this trial' })
  if (space?.kind === 'choice' && space.options.every((o) => typeof o !== 'string'))
    return scalar({ description: 'the knob’s value in this trial' })
  return message({ description: 'the knob’s value in this trial' })
}

/** The contract of a sweep: the trial's knob values and metrics, the configuration's spread, the best so far. */
function sweepContract<S, C extends Config, K extends string>(
  spec: SweepSpec<S, C, K>,
): Contract<SweepState, SweepPath> {
  const metrics: string[] = [...spec.metrics, ...(spec.final ?? [])]
  const out: Record<string, Source<SweepState> | EventSource> = {
    'trial/config': scalar({ description: 'the index of the trial’s configuration in the design' }),
    'trial/repeat': scalar({ description: 'which repeat of the configuration this trial is' }),
    'trial/best': scalar({ description: 'the best objective value of any trial so far' }),
    'sweep/configurations': derived({
      description: 'each configuration with the mean and spread of its repeats so far',
      compute: (s) => sweepConfigurations(s) as unknown as Json,
    }),
    'sweep/best': derived({
      description: 'the best configuration by its mean objective',
      compute: (s) => (sweepBest(s) ?? null) as unknown as Json,
    }),
  }
  for (const name of knobNames(spec)) out[`trial/${name}`] = knobSource(spec.vary?.spaces[name])
  for (const m of metrics) {
    out[`trial/${m}`] = scalar({ description: `the final value of ${m} in this trial` })
    out[`config/${m}/mean`] = scalar({ description: `the mean of ${m} over the configuration’s repeats so far` })
    out[`config/${m}/std`] = scalar({ description: `the standard deviation of ${m} over the repeats so far` })
  }
  // The knobs and metrics are named by the spec, so `out` is a plain record of `SweepPath` keys.
  return out as Contract<SweepState, SweepPath>
}

/** Run one trial: the inner method as an eager session under a key, read at the end. */
function runTrial<S, C extends Config, K extends string>(
  spec: SweepSpec<S, C, K>,
  values: SpaceValue,
  k: Key,
  seed: number,
  index: number,
  config: number,
  repeat: number,
  keep: boolean,
): { readonly record: TrialRecord; readonly tape: Run | undefined } {
  const logs: Record<string, LogSpec> = {}
  for (const m of spec.metrics) logs[m as string] = Log()
  Object.assign(logs, spec.logs)
  const s = session(spec.method, {
    config: { ...spec.fixed, ...(isRecord(values) ? values : {}) } as Partial<C>,
    seed,
    key: k,
    mode: 'eager',
    steps: spec.steps ?? 1000,
    logs: (Object.keys(logs).length === 0 ? 'off' : logs) as NonNullable<SessionOptions<S, C, K>['logs']>,
    past: false,
  })
  const run = s.run()
  const metrics: Record<string, number> = {}
  const series: Record<string, SeriesSummary> = {}
  for (const m of spec.metrics as readonly string[]) {
    const xs = metricSeries(run, m).values
    if (xs.length === 0) throw new OmaiError('value', `sweep: ${spec.method.name} logged no '${m}' in trial ${index}`)
    series[m] = summariseSeries(xs)
    metrics[m] = xs.at(-1) as number
  }
  for (const p of spec.final ?? []) {
    const v = s.probe(s.t, p) as unknown
    if (typeof v !== 'number')
      throw new OmaiError('value', `sweep: the derived path '${p}' gave ${typeof v}, not a number`)
    metrics[p] = v
  }
  return {
    record: { index, config, repeat, values, metrics, steps: s.t, status: s.status ?? null, series },
    tape: keep ? run : undefined,
  }
}

/**
 * A sweep as an algorithm: one trial per step, over the grid or sample of the knobs, each configuration repeated under
 * fresh keys. Each step runs the inner method as a session, reads the final metrics off its tape, and logs them with
 * the trial's knob values (`trial/<knob>`, `trial/<metric>`), the running mean and spread of the configuration's
 * repeats (`config/<metric>/mean`, `config/<metric>/std`) and the best objective so far (`trial/best`). The inner
 * tapes are kept as `keep` says and otherwise summarised in the trial records. Everything in the state serialises.
 *
 * @param spec - The inner method, the knobs and how to vary them, the repeats, the metrics and what to keep.
 * @returns The algorithm; it ends after `sweepTrials(spec)` steps.
 * @throws OmaiError with code `value` for no metrics, a non-positive count, or an objective that is not a metric.
 * @concept log-contract
 * @index algorithm Run a method over a grid of knobs, one trial per step.
 * @example
 * ```ts
 * import { sweep } from 'omai-engine/learn'
 * import { dict, real } from 'omai-engine/space'
 * import { runAlgorithm, scalar } from 'omai-engine/trace'
 * // The inner method logs (a - 0.3)^2 once; the sweep finds the best of five values of a.
 * const method = {
 *   name: 'bowl', config: dict({ a: real(0, 1, { default: 0.5 }) }),
 *   make: (c: { a: number }) => ({ name: 'bowl', config: {}, init: () => 0,
 *     step: (s: number, ctx: { log(p: string, v: number): void }) => { ctx.log('loss', (c.a - 0.3) ** 2); return s + 1 },
 *     done: (s: number) => (s >= 1 ? 'completed' : undefined) }),
 *   contract: { loss: scalar() },
 * }
 * const study = sweep({ method, vary: dict({ a: real(0, 1) }), points: 5, metrics: ['loss'] })
 * const { state } = runAlgorithm(study, { seed: 0, maxSteps: 5 })
 * state.trials.map((t) => t.values.a) // => [0, 0.25, 0.5, 0.75, 1]
 * ```
 */
export function sweep<S, C extends Config, K extends string = string>(
  spec: SweepSpec<S, C, K>,
): Algorithm<SweepState, SweepPath> {
  const repeats = spec.repeats ?? 1
  const keep = spec.keep ?? 'none'
  const metrics: string[] = [...spec.metrics, ...(spec.final ?? [])]
  if (metrics.length === 0) throw new OmaiError('value', 'sweep: name at least one metric to read from each trial')
  if (!Number.isInteger(repeats) || repeats < 1)
    throw new OmaiError('value', 'sweep: repeats must be a positive integer')
  const objective = {
    metric: spec.objective?.metric ?? (metrics[0] as string),
    direction: spec.objective?.direction ?? 'lower',
  } as const
  if (!metrics.includes(objective.metric))
    throw new OmaiError('value', `sweep: the objective '${objective.metric}' is not one of the metrics`)
  const total = sweepTrials(spec)
  const names = knobNames(spec)
  return {
    name: `sweep(${spec.method.name})`,
    config: {
      method: spec.method.name,
      knobs: names,
      design: spec.configs !== undefined ? 'explicit' : (spec.design ?? 'grid'),
      repeats,
      keep,
      trials: total,
      objective,
    },
    init: (k) => ({ key: k, configs: designOf(spec, k), trials: [], tapes: [], best: null, objective }),
    step: (s, ctx) => {
      const t = s.trials.length
      const c = Math.floor(t / repeats)
      const r = t % repeats
      const values = s.configs[c] as SpaceValue
      const k = spec.paired === false ? foldIn(s.key, t) : foldIn(s.key, r)
      const { record, tape } = runTrial(spec, values, k, r, t, c, r, keep !== 'none')
      const trials = [...s.trials, record]
      const value = record.metrics[objective.metric] as number
      const best =
        s.best === null || better(value, s.best, objective.direction) ? (finite(value) ? value : s.best) : s.best
      const next: SweepState = {
        ...s,
        trials,
        tapes: tape === undefined ? s.tapes : keep === 'all' ? [...s.tapes, tape] : [tape],
        best,
      }
      if (isRecord(values)) {
        for (const name of names) {
          const v = values[name]
          if (v !== undefined) ctx.log(`trial/${name}`, v as Json)
        }
      }
      ctx.log('trial/config', c)
      ctx.log('trial/repeat', r)
      for (const m of metrics) ctx.log(`trial/${m}`, record.metrics[m] as number)
      if (best !== null) ctx.log('trial/best', best)
      if (metrics.some((m) => ctx.wants(`config/${m}/mean`) || ctx.wants(`config/${m}/std`))) {
        const summary = sweepConfigurations(next).find((x) => x.index === c)
        for (const m of metrics) {
          ctx.log(`config/${m}/mean`, summary?.mean[m] ?? Number.NaN)
          ctx.log(`config/${m}/std`, summary?.std[m] ?? Number.NaN)
        }
      }
      return next
    },
    done: (s) => (s.trials.length >= total ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SweepState,
  }
}

/**
 * A sweep as a method: the study with its design as knobs (repeats, steps, what to keep, and the grid's points or the
 * sample's size), so it runs as a session with the contract of {@link sweep} (and the derived `sweep/configurations`
 * and `sweep/best`) and a page can drive it live, seek it and probe it.
 *
 * @param spec - The sweep; its `repeats`, `steps`, `keep`, `design`, `points` and `trials` are the knobs' defaults.
 * @returns The method.
 * @concept log-contract
 * @example
 * ```ts
 * import { sweepMethod } from 'omai-engine/learn'
 * import { dict, real } from 'omai-engine/space'
 * import { scalar } from 'omai-engine/trace'
 * const inner = { name: 'noop', config: dict({ a: real(0, 1) }), make: () => { throw new Error('unused') }, contract: { loss: scalar() } }
 * sweepMethod({ method: inner, vary: dict({ a: real(0, 1) }), metrics: ['loss'], repeats: 4 }).name // => 'sweep(noop)'
 * ```
 */
export function sweepMethod<S, C extends Config, K extends string = string>(
  spec: SweepSpec<S, C, K>,
): Method<SweepState, Config, SweepPath> {
  const explicit = spec.configs !== undefined
  const knobs: Record<string, Space> = {
    repeats: int(1, 100_000, { default: spec.repeats ?? 1, label: 'repeats per configuration' }),
    steps: int(1, 10_000_000, { default: spec.steps ?? 1000, label: 'steps of each inner run' }),
    keep: choice(['none', 'last', 'all'], { default: spec.keep ?? 'none', label: 'inner tapes kept' }),
    ...(explicit
      ? {}
      : {
          design: choice(['grid', 'sample'], { default: spec.design ?? 'grid', label: 'design' }),
          points: int(1, 1000, {
            default: spec.points ?? 5,
            label: 'points per knob',
            when: { key: 'design', in: ['grid'] },
          }),
          trials: int(1, 100_000, {
            default: spec.trials ?? 10,
            label: 'configurations drawn',
            when: { key: 'design', in: ['sample'] },
          }),
        }),
  }
  return defineMethod<SweepState, Config, SweepPath>({
    name: `sweep(${spec.method.name})`,
    config: dict(knobs),
    make: (config) =>
      sweep({
        ...spec,
        repeats: config['repeats'] as number,
        steps: config['steps'] as number,
        keep: config['keep'] as SweepKeep,
        ...(explicit
          ? {}
          : {
              design: config['design'] as 'grid' | 'sample',
              points: (config['points'] as number | undefined) ?? spec.points ?? 5,
              trials: (config['trials'] as number | undefined) ?? spec.trials ?? 10,
            }),
      }),
    contract: sweepContract(spec),
  })
}

/** The state of a race: every variant's own state, and the status of those that have finished. */
export interface RaceState {
  readonly states: Readonly<Record<string, unknown>>
  readonly finished: Readonly<Record<string, RunStatus | null>>
}

/**
 * A race: one method whose single run drives several variants of the inner method's configuration in lockstep, all
 * from the same key (the same data and start) and with the same key at every step (the same batches), each logging
 * its own paths under `race/<variant>/`. It is the sweep over a slot done in one run, so a page can play the variants
 * side by side and every probe of the inner method works per variant. The race ends when every variant has.
 *
 * @param method - The inner method.
 * @param variants - The variants by name, each the knobs it sets over the race's configuration.
 * @returns The method; its configuration is the inner method's, under the variants' changes.
 * @throws OmaiError with code `value` for no variants, or a variant whose configuration is outside the method's space.
 * @concept log-contract
 * @example
 * ```ts
 * import { race } from 'omai-engine/learn'
 * import { choice, dict } from 'omai-engine/space'
 * import { scalar, session } from 'omai-engine/trace'
 * const count = {
 *   name: 'count', config: dict({ by: choice([1, 2]) }),
 *   make: (c: { by: number }) => ({ name: 'count', config: {}, init: () => 0,
 *     step: (s: number, ctx: { log(p: string, v: number): void }) => { ctx.log('n', s + c.by); return s + c.by },
 *     done: (s: number) => (s >= 6 ? 'completed' : undefined) }),
 *   contract: { n: scalar() },
 * }
 * const r = session(race(count, { slow: { by: 1 }, fast: { by: 2 } }), { mode: 'eager', logs: 'all' })
 * r.state.states // => { slow: 6, fast: 6 }
 * ```
 */
export function race<S, C extends Config, K extends string = string>(
  method: Method<S, C, K>,
  variants: Readonly<Record<string, Partial<C>>>,
): Method<RaceState, C, RacePath<K>> {
  const names = Object.keys(variants)
  if (names.length === 0) throw new OmaiError('value', 'race: need at least one variant')
  const configOf = (config: C, v: string): C => ({ ...config, ...variants[v] }) as C
  return defineMethod<RaceState, C, RacePath<K>>({
    name: `race(${method.name})`,
    config: method.config,
    make: (config, k) => {
      const algorithms: Record<string, Algorithm<S, K>> = {}
      for (const v of names) algorithms[v] = method.make(configOf(config, v), k)
      const each = <T>(f: (v: string, a: Algorithm<S, K>) => T): Record<string, T> =>
        Object.fromEntries(names.map((v) => [v, f(v, algorithms[v] as Algorithm<S, K>)]))
      const serialisable = names.every(
        (v) => algorithms[v]?.serialise !== undefined && algorithms[v]?.deserialise !== undefined,
      )
      const status = (a: Algorithm<S, K>, s: S, t: number): RunStatus | null => a.done?.(s, t) ?? null
      return {
        name: `race(${method.name})`,
        config: { variants: names } as Json,
        init: (key) => {
          const states = each((_, a) => a.init(key))
          return { states, finished: each((v, a) => status(a, states[v] as S, 0)) }
        },
        step: (s, ctx) => {
          const states: Record<string, unknown> = { ...s.states }
          const finished: Record<string, RunStatus | null> = { ...s.finished }
          for (const v of names) {
            if (finished[v] !== null) continue
            const a = algorithms[v] as Algorithm<S, K>
            const next = a.step(s.states[v] as S, mountContext<`race/${string}`, K>(ctx, `race/${v}`))
            states[v] = next
            finished[v] = status(a, next, ctx.t + 1)
          }
          return { states, finished }
        },
        done: (s) => (names.every((v) => s.finished[v] !== null) ? 'completed' : undefined),
        ...(serialisable
          ? {
              serialise: (s: RaceState): Json => ({
                states: each((v, a) => (a.serialise as (x: S) => Json)(s.states[v] as S)),
                finished: s.finished as Json,
              }),
              deserialise: (j: Json): RaceState => {
                const o = j as { states: Record<string, Json>; finished: Record<string, RunStatus | null> }
                return {
                  states: each((v, a) => (a.deserialise as (x: Json) => S)(o.states[v] as Json)),
                  finished: o.finished,
                }
              },
            }
          : {}),
      }
    },
    contract: (config, k) => {
      const out: Record<string, Source<RaceState> | EventSource> = {}
      for (const v of names) {
        for (const [path, source] of Object.entries(contractOf(method, configOf(config, v), k) as Contract<S>)) {
          out[`race/${v}/${path}`] =
            source.kind === 'derived'
              ? derived<RaceState>({
                  description: source.description,
                  ...(source.cost === undefined ? {} : { cost: source.cost }),
                  compute: (s, args) => source.compute(s.states[v] as S, args),
                })
              : (source as Source<RaceState> | EventSource)
        }
      }
      // The variants are named by the caller, so `out` is a plain record of `race/<variant>/<path>` keys.
      return out as Contract<RaceState, RacePath<K>>
    },
  })
}
