/**
 * Non-parametric estimation of the survival function and the cumulative hazard from right-censored data: the
 * Kaplan-Meier product-limit estimator with Greenwood's variance and its log-log, plain and log intervals, the
 * Nelson-Aalen cumulative hazard, quantiles and restricted mean survival of a curve, and `kaplanMeierAlgorithm`,
 * which adds one subject (an event or a censoring) per step so the curve can be watched forming.
 */

import { type Json, OmaiError, resolveConfidenceLevel } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { type Algorithm, type Observation, type RunStatus, stateFromJson, stateToJson } from '../trace/index.ts'
import { normPpf } from './_util.ts'
import { eventTable, type SurvivalSample, survivalSample } from './data.ts'

/** How the interval of a survival probability is transformed: `'log-log'` (the default), `'plain'` or `'log'`. */
export type SurvivalInterval = 'log-log' | 'plain' | 'log'

/** A Kaplan-Meier survival curve: a step function that drops at each distinct event time. */
export interface SurvivalCurve {
  /** Subjects in the sample. */
  readonly n: number
  /** Distinct event times, ascending. */
  readonly time: readonly number[]
  /** Subjects at risk just before each time. */
  readonly atRisk: readonly number[]
  /** Events at each time. */
  readonly events: readonly number[]
  /** Subjects censored from each time up to the next event time. */
  readonly censored: readonly number[]
  /** The estimate of S(t) from each time on. */
  readonly survival: readonly number[]
  /** Greenwood's standard error of S(t). */
  readonly standardError: readonly number[]
  /** The lower limit of the interval at each time. */
  readonly lower: readonly number[]
  /** The upper limit of the interval at each time. */
  readonly upper: readonly number[]
  readonly level: number
  readonly interval: SurvivalInterval
  /** The median survival time: the first time S(t) <= 1/2 (Infinity when the curve stays above it). */
  readonly median: number
}

/** Options of {@link kaplanMeier}. */
export interface KaplanMeierOptions {
  /** The transformation of the interval (default `'log-log'`). */
  readonly interval?: SurvivalInterval
  /** The confidence level (default 0.95). */
  readonly level?: number
}

function intervalBounds(
  s: number,
  greenwood: number,
  interval: SurvivalInterval,
  z: number,
): readonly [number, number] {
  if (s <= 0) return [0, 0]
  if (!Number.isFinite(greenwood)) return [Number.NaN, Number.NaN]
  const seLog = Math.sqrt(greenwood)
  if (interval === 'plain') {
    const se = s * seLog
    return [Math.max(0, s - z * se), Math.min(1, s + z * se)]
  }
  if (interval === 'log') return [Math.exp(Math.log(s) - z * seLog), Math.min(1, Math.exp(Math.log(s) + z * seLog))]
  if (s >= 1) return [1, 1]
  const seLogLog = seLog / Math.abs(Math.log(s))
  return [s ** Math.exp(z * seLogLog), s ** Math.exp(-z * seLogLog)]
}

/**
 * The Kaplan-Meier product-limit estimate of the survival function, `S(t) = prod (1 - d_k / n_k)` over the event times
 * up to t, with Greenwood's variance `S^2 sum d_k / (n_k (n_k - d_k))` and an interval on the log-log (default), plain
 * or log scale (as `survfit` and lifelines report them).
 *
 * @param time - Observed times.
 * @param event - Event indicators (omitted: all events).
 * @param options - `interval` and `level`.
 * @returns The curve.
 * @index function Kaplan-Meier survival curve for right-censored data.
 * @example
 * ```ts
 * import { kaplanMeier } from 'omai-engine/survival'
 * const km = kaplanMeier([1, 2, 2, 3, 4, 5], [1, 1, 0, 1, 0, 1])
 * km.survival.map((s) => Number(s.toFixed(4))) // => [0.8333, 0.6667, 0.4444, 0]
 * km.median // => 3
 * ```
 */
export function kaplanMeier(time: TensorLike, event?: TensorLike, options: KaplanMeierOptions = {}): SurvivalCurve {
  return kaplanMeierOf(survivalSample(time, event, 'kaplanMeier'), options)
}

/**
 * The Kaplan-Meier curve of a validated sample.
 *
 * @param sample - The sample.
 * @param options - `interval` and `level`.
 * @returns The curve.
 * @example
 * ```ts
 * import { kaplanMeierOf, survivalSample } from 'omai-engine/survival'
 * kaplanMeierOf(survivalSample([1, 2, 3])).survival.map((s) => Number(s.toFixed(4))) // => [0.6667, 0.3333, 0]
 * ```
 */
export function kaplanMeierOf(sample: SurvivalSample, options: KaplanMeierOptions = {}): SurvivalCurve {
  const level = resolveConfidenceLevel(options.level)
  const interval = options.interval ?? 'log-log'
  const z = normPpf(0.5 + level / 2)
  const t = eventTable(sample)
  const m = t.time.length
  const survival: number[] = []
  const standardError: number[] = []
  const lower: number[] = []
  const upper: number[] = []
  let s = 1
  let g = 0
  let median = Number.POSITIVE_INFINITY
  for (let k = 0; k < m; k++) {
    const n = t.atRisk[k] ?? 0
    const d = t.events[k] ?? 0
    s *= 1 - d / n
    g += n > d ? d / (n * (n - d)) : Number.POSITIVE_INFINITY
    survival.push(s)
    standardError.push(s <= 0 ? 0 : Number.isFinite(g) ? s * Math.sqrt(g) : Number.NaN)
    const [lo, hi] = intervalBounds(s, g, interval, z)
    lower.push(lo)
    upper.push(hi)
    if (median === Number.POSITIVE_INFINITY && s <= 0.5) median = t.time[k] ?? Number.NaN
  }
  return {
    n: sample.n,
    time: Array.from(t.time),
    atRisk: Array.from(t.atRisk),
    events: Array.from(t.events),
    censored: Array.from(t.censored),
    survival,
    standardError,
    lower,
    upper,
    level,
    interval,
    median,
  }
}

/**
 * The value of a step survival curve at a time: 1 before the first event, then the level of the last step at or
 * before it.
 *
 * @param curve - The curve.
 * @param t - The time.
 * @returns S(t).
 * @example
 * ```ts
 * import { kaplanMeier, survivalAt } from 'omai-engine/survival'
 * survivalAt(kaplanMeier([1, 2, 3, 4], [1, 1, 0, 1]), 2.5) // => 0.5
 * ```
 */
export function survivalAt(curve: Pick<SurvivalCurve, 'time' | 'survival'>, t: number): number {
  let value = 1
  for (let k = 0; k < curve.time.length && (curve.time[k] ?? Number.POSITIVE_INFINITY) <= t; k++)
    value = curve.survival[k] ?? value
  return value
}

/**
 * The p-quantile of a survival curve: the first time S(t) <= 1 - p (Infinity when the curve never gets there).
 *
 * @param curve - The curve.
 * @param p - The probability in (0, 1) (0.5 is the median).
 * @returns The time.
 * @example
 * ```ts
 * import { kaplanMeier, survivalQuantile } from 'omai-engine/survival'
 * survivalQuantile(kaplanMeier([1, 2, 3, 4, 5, 6, 7, 8]), 0.25) // => 2
 * ```
 */
export function survivalQuantile(curve: Pick<SurvivalCurve, 'time' | 'survival'>, p: number): number {
  if (!(p > 0 && p < 1)) throw new OmaiError('value', `survivalQuantile: p must be in (0, 1), got ${p}`)
  for (let k = 0; k < curve.time.length; k++)
    if ((curve.survival[k] ?? 1) <= 1 - p + 1e-12) return curve.time[k] ?? Number.NaN
  return Number.POSITIVE_INFINITY
}

/**
 * The restricted mean survival time up to a horizon: the area under the curve on `[0, tau]`.
 *
 * @param curve - The curve.
 * @param tau - The horizon (default the last event time).
 * @returns The area.
 * @example
 * ```ts
 * import { kaplanMeier, restrictedMeanSurvival } from 'omai-engine/survival'
 * restrictedMeanSurvival(kaplanMeier([1, 2, 3, 4]), 4) // => 2.5
 * ```
 */
export function restrictedMeanSurvival(curve: Pick<SurvivalCurve, 'time' | 'survival'>, tau?: number): number {
  const horizon = tau ?? curve.time[curve.time.length - 1] ?? 0
  let area = 0
  let previous = 0
  let level = 1
  for (let k = 0; k < curve.time.length; k++) {
    const t = Math.min(curve.time[k] ?? 0, horizon)
    area += level * (t - previous)
    previous = t
    level = curve.survival[k] ?? level
    if ((curve.time[k] ?? 0) >= horizon) return area
  }
  return area + level * (horizon - previous)
}

/** The Nelson-Aalen estimate of the cumulative hazard `H(t) = sum d_k / n_k`. */
export interface CumulativeHazardCurve {
  readonly n: number
  readonly time: readonly number[]
  readonly atRisk: readonly number[]
  readonly events: readonly number[]
  /** The estimate of H(t) from each time on. */
  readonly cumulativeHazard: readonly number[]
  /** The standard error `sqrt(sum d_k / n_k^2)`. */
  readonly standardError: readonly number[]
  /** The lower limit of the log-transformed interval `H exp(+-z se / H)`. */
  readonly lower: readonly number[]
  /** The upper limit of the log-transformed interval. */
  readonly upper: readonly number[]
  readonly level: number
  /** Fleming-Harrington's survival estimate `exp(-H(t))`. */
  readonly survival: readonly number[]
}

/**
 * The Nelson-Aalen estimate of the cumulative hazard with its standard error and log-transformed interval, and the
 * Fleming-Harrington survival estimate `exp(-H)` that comes with it.
 *
 * @param time - Observed times.
 * @param event - Event indicators (omitted: all events).
 * @param options - The confidence `level`.
 * @returns The curve.
 * @example
 * ```ts
 * import { nelsonAalen } from 'omai-engine/survival'
 * nelsonAalen([1, 2, 3, 4], [1, 1, 0, 1]).cumulativeHazard.map((h) => Number(h.toFixed(4))) // => [0.25, 0.5833, 1.5833]
 * ```
 */
export function nelsonAalen(
  time: TensorLike,
  event?: TensorLike,
  options: { readonly level?: number } = {},
): CumulativeHazardCurve {
  const sample = survivalSample(time, event, 'nelsonAalen')
  const level = resolveConfidenceLevel(options.level)
  const z = normPpf(0.5 + level / 2)
  const t = eventTable(sample)
  const h: number[] = []
  const se: number[] = []
  const lower: number[] = []
  const upper: number[] = []
  let cum = 0
  let variance = 0
  for (let k = 0; k < t.time.length; k++) {
    const n = t.atRisk[k] ?? 0
    const d = t.events[k] ?? 0
    cum += d / n
    variance += d / (n * n)
    h.push(cum)
    se.push(Math.sqrt(variance))
    const factor = Math.exp((z * Math.sqrt(variance)) / cum)
    lower.push(cum / factor)
    upper.push(cum * factor)
  }
  return {
    n: sample.n,
    time: Array.from(t.time),
    atRisk: Array.from(t.atRisk),
    events: Array.from(t.events),
    cumulativeHazard: h,
    standardError: se,
    lower,
    upper,
    level,
    survival: h.map((v) => Math.exp(-v)),
  }
}

/** The state of {@link kaplanMeierAlgorithm}: the subjects in the order they leave the risk set, and the curve so far. */
export type KaplanMeierState = {
  /** The subjects processed. */
  readonly i: number
  readonly n: number
  /** Times of the subjects in order (events before censorings at a tied time). */
  readonly time: Float64Array
  readonly event: Uint8Array
  readonly survival: number
  /** Greenwood's sum `sum d / (n (n - d))` so far. */
  readonly greenwood: number
  /** The time of the last subject processed. */
  readonly now: number
  readonly nEvents: number
  readonly nCensored: number
}

/**
 * Kaplan-Meier as an algorithm that takes one subject per step, in time order (an event before a censoring at a tied
 * time): an event multiplies the curve by `1 - 1 / n` (the factors of tied events telescope to `1 - d / n`), a
 * censoring only shrinks the risk set. After `n` steps the state holds the same curve {@link kaplanMeier} returns.
 *
 * @param time - Observed times.
 * @param event - Event indicators (omitted: all events).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { kaplanMeierAlgorithm } from 'omai-engine/survival'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(kaplanMeierAlgorithm([1, 2, 3, 4], [1, 0, 1, 1]), { seed: 0, maxSteps: 4 })
 * run.state.survival // => 0
 * run.status // => 'completed'
 * ```
 */
export function kaplanMeierAlgorithm(time: TensorLike, event?: TensorLike): Algorithm<KaplanMeierState> {
  const sample = survivalSample(time, event, 'kaplanMeierAlgorithm')
  const order = Array.from({ length: sample.n }, (_, i) => i).sort(
    (a, b) => (sample.time[a] ?? 0) - (sample.time[b] ?? 0) || (sample.event[b] ?? 0) - (sample.event[a] ?? 0),
  )
  const t = Float64Array.from(order, (i) => sample.time[i] ?? 0)
  const e = Uint8Array.from(order, (i) => sample.event[i] ?? 0)
  return {
    name: 'kaplan-meier',
    config: { n: sample.n },
    init: () => ({ i: 0, n: sample.n, time: t, event: e, survival: 1, greenwood: 0, now: 0, nEvents: 0, nCensored: 0 }),
    step: (s) => {
      if (s.i >= s.n) return s
      const atRisk = s.n - s.i
      const isEvent = s.event[s.i] === 1
      const survival = isEvent ? s.survival * (1 - 1 / atRisk) : s.survival
      const greenwood = isEvent
        ? s.greenwood + (atRisk > 1 ? 1 / (atRisk * (atRisk - 1)) : Number.POSITIVE_INFINITY)
        : s.greenwood
      return {
        ...s,
        i: s.i + 1,
        survival,
        greenwood,
        now: s.time[s.i] ?? s.now,
        nEvents: s.nEvents + (isEvent ? 1 : 0),
        nCensored: s.nCensored + (isEvent ? 0 : 1),
      }
    },
    done: (s): RunStatus | undefined => (s.i >= s.n ? 'completed' : undefined),
    observe: (s): Observation => ({
      metrics: {
        time: s.now,
        survival: s.survival,
        atRisk: s.n - s.i,
        standardError: Number.isFinite(s.greenwood) ? s.survival * Math.sqrt(s.greenwood) : Number.NaN,
        events: s.nEvents,
        censored: s.nCensored,
      },
    }),
    serialise: (s): Json => stateToJson(s),
    deserialise: (j): KaplanMeierState => stateFromJson(j) as KaplanMeierState,
  }
}
