/**
 * Sequential change detectors for a stream, one observation per step: the two-sided tabular CUSUM of Page, which
 * accumulates standardised deviations beyond an allowance until they cross a decision limit, and the Page-Hinkley test
 * (Gama et al.), which watches the cumulative deviation from the running mean for a rise above its own minimum. Each is
 * an `Algorithm` whose state serialises and which records its statistic and alarms. Bayesian online changepoint
 * detection lives in `dynamics/filtering/changepoint.ts`.
 */

import { OmaiError } from '../base/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { asSeries } from './_series.ts'

/** Options of {@link cusumAlgorithm}. */
export interface CusumOptions {
  /** The in-control mean; with `sigma` omitted or this omitted both are estimated from the first `warmup` observations. */
  readonly target?: number
  /** The in-control standard deviation. */
  readonly sigma?: number
  /** The allowance k in standard deviations: half the shift to detect quickly (default 0.5). */
  readonly allowance?: number
  /** The decision limit h in standard deviations (default 5). */
  readonly threshold?: number
  /** Observations used to estimate the target and spread when they are not given (default 20). */
  readonly warmup?: number
  /** Restart both sums at zero after an alarm (default true). */
  readonly resetOnAlarm?: boolean
}

/** An alarm of a detector: when it fired and, for a two-sided test, the direction of the change. */
export interface Alarm {
  readonly index: number
  readonly direction: 'up' | 'down'
}

/** The state of {@link cusumAlgorithm}. */
export interface CusumState {
  /** Observations consumed. */
  readonly t: number
  /** The upper and lower cumulative sums (in standard deviations). */
  readonly high: number
  readonly low: number
  /** The Welford accumulators of the warm-up: its mean and sum of squared deviations. */
  readonly mean: number
  readonly m2: number
  readonly alarms: readonly Alarm[]
}

/**
 * Two-sided tabular CUSUM as an algorithm, one observation per step: with z_t = (x_t - mean) / sigma, the sums are
 * high_t = max(0, high_(t-1) + z_t - k) and low_t = max(0, low_(t-1) - z_t - k), and an alarm is raised when either
 * exceeds h. Without a given target and spread, the first `warmup` observations estimate them and raise no alarm.
 *
 * @param x - The stream.
 * @param options - Target, spread, allowance k, limit h, warm-up and reset.
 * @returns The algorithm; it is done after the last observation.
 * @example
 * ```ts
 * import { cusumAlgorithm } from 'omai-engine/timeseries'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [...new Array(10).fill(0), ...new Array(10).fill(3)]
 * const s = runAlgorithm(cusumAlgorithm(x, { target: 0, sigma: 1 }), { maxSteps: 100 }).state
 * s.alarms[0] // => { index: 12, direction: 'up' }
 * ```
 */
export function cusumAlgorithm(x: ArrayLike<number>, options: CusumOptions = {}): Algorithm<CusumState> {
  const data = asSeries(Float64Array.from(x), 'cusum')
  const k = options.allowance ?? 0.5
  const h = options.threshold ?? 5
  const warmup = options.warmup ?? 20
  const known = options.target !== undefined && options.sigma !== undefined
  if (!(h > 0 && k >= 0)) throw new OmaiError('value', 'cusum: need a positive threshold and a non-negative allowance')
  if (options.sigma !== undefined && !(options.sigma > 0)) throw new OmaiError('value', 'cusum: sigma must be positive')
  const reset = options.resetOnAlarm ?? true
  return {
    name: 'cusum',
    config: { allowance: k, threshold: h, warmup, known },
    schema: channels({
      metrics: {
        high: { description: 'the upper cumulative sum, in standard deviations' },
        low: { description: 'the lower cumulative sum, in standard deviations' },
        alarm: { description: '1 when this observation raised an alarm' },
      },
    }),
    init: () => ({ t: 0, high: 0, low: 0, mean: 0, m2: 0, alarms: [] }),
    step: (s) => {
      const v = data[s.t] ?? Number.NaN
      const n = s.t + 1
      if (!known && s.t < warmup) {
        // Welford's update of the warm-up mean and sum of squared deviations.
        const delta = v - s.mean
        const mean = s.mean + delta / n
        return { ...s, t: n, mean, m2: s.m2 + delta * (v - mean) }
      }
      const centre = options.target ?? s.mean
      const seen = Math.min(s.t, warmup)
      const scale = options.sigma ?? (seen > 1 ? Math.sqrt(s.m2 / (seen - 1)) : 1) ?? 1
      const z = (v - centre) / (scale || 1)
      let high = Math.max(0, s.high + z - k)
      let low = Math.max(0, s.low - z - k)
      const alarms = [...s.alarms]
      if (high > h || low > h) {
        alarms.push({ index: s.t, direction: high > h ? 'up' : 'down' })
        if (reset) {
          high = 0
          low = 0
        }
      }
      return { ...s, t: n, high, low, alarms }
    },
    done: (s) => (s.t >= data.length ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { high: s.high, low: s.low, alarm: s.alarms.at(-1)?.index === s.t - 1 && s.t > 0 ? 1 : 0 },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CusumState,
  }
}

/** The result of running a detector over a whole series. */
export interface DetectionRun {
  /** The alarms, in order. */
  readonly alarms: readonly Alarm[]
}

/**
 * Run the two-sided CUSUM over a series.
 *
 * @param x - The series.
 * @param options - See {@link cusumAlgorithm}.
 * @returns The alarms.
 * @example
 * ```ts
 * import { cusum } from 'omai-engine/timeseries'
 * const x = [...new Array(30).fill(0).map((_, t) => (t % 2 ? 0.2 : -0.2)), ...new Array(10).fill(2)]
 * cusum(x, { warmup: 10 }).alarms[0]?.index // => 30
 * ```
 */
export function cusum(x: ArrayLike<number>, options: CusumOptions = {}): DetectionRun {
  const run = runAlgorithm(cusumAlgorithm(x, options), { maxSteps: Number.MAX_SAFE_INTEGER })
  return { alarms: run.state.alarms }
}

/** Options of {@link pageHinkleyAlgorithm}. */
export interface PageHinkleyOptions {
  /** The magnitude of changes tolerated, subtracted from every deviation (default 0.005). */
  readonly delta?: number
  /** The alarm threshold lambda on the test statistic (default 50). */
  readonly threshold?: number
  /** The forgetting factor of the cumulative sum, 1 for none (default 1 - 1e-4). */
  readonly alpha?: number
  /** Observations before an alarm is possible (default 30). */
  readonly minInstances?: number
  /** Detect increases of the mean, decreases, or both (default `both`). */
  readonly direction?: 'up' | 'down' | 'both'
  /** Restart the statistics after an alarm (default true). */
  readonly resetOnAlarm?: boolean
}

/** The state of {@link pageHinkleyAlgorithm}. */
export interface PageHinkleyState {
  /** Observations consumed (since the last reset). */
  readonly n: number
  /** Observations consumed in all. */
  readonly t: number
  /** The running mean. */
  readonly mean: number
  /** The cumulative sums for increases and decreases, and their running extrema. */
  readonly up: number
  readonly upMin: number
  readonly down: number
  readonly downMin: number
  readonly alarms: readonly Alarm[]
}

/**
 * The Page-Hinkley test as an algorithm, one observation per step: with the running mean x_t-bar, the sums
 * m_t = alpha m_(t-1) + (x_t - x_t-bar - delta) (for increases; with the signs flipped for decreases) are compared with
 * their running minima M_t, and an alarm is raised when m_t - M_t exceeds lambda. It records the statistics and alarms.
 *
 * @param x - The stream.
 * @param options - Tolerance delta, threshold lambda, forgetting factor, warm-up, direction and reset.
 * @returns The algorithm; it is done after the last observation.
 * @example
 * ```ts
 * import { pageHinkleyAlgorithm } from 'omai-engine/timeseries'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [...new Array(40).fill(0), ...new Array(20).fill(5)]
 * runAlgorithm(pageHinkleyAlgorithm(x, { threshold: 20, direction: 'up' }), { maxSteps: 100 }).state.alarms[0]?.index // => 44
 * ```
 */
export function pageHinkleyAlgorithm(
  x: ArrayLike<number>,
  options: PageHinkleyOptions = {},
): Algorithm<PageHinkleyState> {
  const data = asSeries(Float64Array.from(x), 'pageHinkley')
  const delta = options.delta ?? 0.005
  const lambda = options.threshold ?? 50
  const alpha = options.alpha ?? 1 - 1e-4
  const minInstances = options.minInstances ?? 30
  const direction = options.direction ?? 'both'
  const reset = options.resetOnAlarm ?? true
  if (!(lambda > 0 && alpha > 0 && alpha <= 1))
    throw new OmaiError('value', 'pageHinkley: need a positive threshold and a forgetting factor in (0, 1]')
  const fresh = (t: number, alarms: readonly Alarm[]): PageHinkleyState => ({
    n: 0,
    t,
    mean: 0,
    up: 0,
    upMin: 0,
    down: 0,
    downMin: 0,
    alarms,
  })
  return {
    name: 'page-hinkley',
    config: { delta, threshold: lambda, alpha, minInstances, direction },
    schema: channels({
      metrics: {
        statisticUp: { description: 'the Page-Hinkley statistic for an increase of the mean' },
        statisticDown: { description: 'the Page-Hinkley statistic for a decrease of the mean' },
        alarm: { description: '1 when this observation raised an alarm' },
      },
    }),
    init: () => fresh(0, []),
    step: (s) => {
      const v = data[s.t] ?? Number.NaN
      const n = s.n + 1
      const mean = s.mean + (v - s.mean) / n
      const up = alpha * s.up + (v - mean - delta)
      const down = alpha * s.down + (mean - v - delta)
      const upMin = Math.min(s.upMin, up)
      const downMin = Math.min(s.downMin, down)
      const next: PageHinkleyState = { ...s, n, t: s.t + 1, mean, up, upMin, down, downMin }
      if (n < minInstances) return next
      const rise = direction !== 'down' && up - upMin > lambda
      const fall = direction !== 'up' && down - downMin > lambda
      if (!rise && !fall) return next
      const alarms = [...s.alarms, { index: s.t, direction: rise ? ('up' as const) : ('down' as const) }]
      return reset ? fresh(s.t + 1, alarms) : { ...next, alarms }
    },
    done: (s) => (s.t >= data.length ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        statisticUp: s.up - s.upMin,
        statisticDown: s.down - s.downMin,
        alarm: s.alarms.at(-1)?.index === s.t - 1 && s.t > 0 ? 1 : 0,
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PageHinkleyState,
  }
}

/**
 * Run the Page-Hinkley test over a series.
 *
 * @param x - The series.
 * @param options - See {@link pageHinkleyAlgorithm}.
 * @returns The alarms.
 * @example
 * ```ts
 * import { pageHinkley } from 'omai-engine/timeseries'
 * const x = [...new Array(40).fill(0), ...new Array(20).fill(-5)]
 * pageHinkley(x, { threshold: 20 }).alarms[0] // => { index: 44, direction: 'down' }
 * ```
 */
export function pageHinkley(x: ArrayLike<number>, options: PageHinkleyOptions = {}): DetectionRun {
  const run = runAlgorithm(pageHinkleyAlgorithm(x, options), { maxSteps: Number.MAX_SAFE_INTEGER })
  return { alarms: run.state.alarms }
}
