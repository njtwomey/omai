/**
 * Prediction intervals for ETS models of every type, after Hyndman, Koehler, Ord and Snyder (2008, chapter 6). The models
 * fall in four classes: class 1 has additive error with additive (or no) trend and season, whose forecast variance is
 * sigma2 (1 + sum c_j^2) in closed form; class 2 has multiplicative error with the same components, whose variance is
 * (1 + sigma2) theta_h - mu_h^2 with theta_h = mu_h^2 + sigma2 sum c_j^2 theta_(h-j); classes 3 (multiplicative error and
 * season) and 4 (the rest, such as a multiplicative trend) have no tractable closed form here and are simulated. The
 * simulation works for every type, so the closed forms can be checked against it.
 */

import { OmaiError } from '../base/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import type { Key } from '../random/index.ts'
import { item } from '../tensor/index.ts'
import { type EtsFit, type EtsSpec, etsForecast, etsSimulate } from './smoothing.ts'

/** The class of an ETS model by the form of its forecast variance (Hyndman et al. 2008, table 6.x). */
export function etsClass(spec: EtsSpec): 1 | 2 | 3 | 4 {
  const additiveParts = spec.trend !== 'mul' && spec.seasonal !== 'mul'
  if (spec.error === 'add') return additiveParts ? 1 : 4
  if (additiveParts) return 2
  return spec.trend !== 'mul' ? 3 : 4
}

/**
 * The closed-form forecast variance of an ETS model of class 1 or 2.
 *
 * @param fit - The fitted model.
 * @param steps - The horizon.
 * @returns The variance at each horizon, or `undefined` for classes 3 and 4 (which are simulated).
 * @example
 * ```ts
 * import { etsPredictionVariance } from 'omai-engine/timeseries'
 * const fit = { spec: { error: 'mul', trend: 'none', seasonal: 'none' }, params: { alpha: 0.5, level0: 10 }, sigma2: 0.01,
 *   final: { level: 10, trend: 0, seasonal: [] } } as never
 * etsPredictionVariance(fit, 1)?.[0] // => 1
 * ```
 */
export function etsPredictionVariance(
  fit: Pick<EtsFit, 'spec' | 'params' | 'sigma2' | 'final'>,
  steps: number,
): number[] | undefined {
  const cls = etsClass(fit.spec)
  if (cls > 2) return undefined
  const { spec, params } = fit
  const m = spec.seasonal === 'none' ? 0 : (spec.period ?? 0)
  const phi = spec.trend !== 'none' && spec.damped === true ? (params.phi ?? 1) : 1
  const mu = etsForecast(fit as EtsFit, steps).mean
  const c: number[] = []
  let cumPhi = 0
  let ph = 1
  for (let j = 1; j < steps; j++) {
    ph *= phi
    cumPhi += ph
    let cj = params.alpha + (spec.trend === 'add' ? (params.beta ?? 0) * cumPhi : 0)
    if (m > 0 && j % m === 0) cj += params.gamma ?? 0
    c.push(cj)
  }
  const out: number[] = []
  if (cls === 1) {
    let sumSq = 0
    for (let h = 1; h <= steps; h++) {
      if (h >= 2) sumSq += (c[h - 2] ?? 0) ** 2
      out.push(fit.sigma2 * (1 + sumSq))
    }
    return out
  }
  const theta: number[] = []
  for (let h = 1; h <= steps; h++) {
    const mh = mu[h - 1] ?? 0
    let t = mh * mh
    for (let j = 1; j < h; j++) t += fit.sigma2 * (c[j - 1] ?? 0) ** 2 * (theta[h - j - 1] ?? 0)
    theta.push(t)
    out.push((1 + fit.sigma2) * t - mh * mh)
  }
  return out
}

/** Options of {@link etsPredictionIntervals}. */
export interface EtsIntervalOptions {
  /** The interval levels (default `[0.95]`). */
  readonly levels?: readonly number[]
  /** `'auto'` (default): the closed form where there is one, else simulation; or force one. */
  readonly method?: 'auto' | 'analytic' | 'simulation'
  /** The key for simulation. */
  readonly key?: Key
  /** The number of simulated paths (default 5000). */
  readonly paths?: number
}

/** The intervals of an ETS forecast. */
export interface EtsIntervals {
  readonly mean: readonly number[]
  readonly variance: readonly number[]
  readonly method: 'analytic' | 'simulation'
  readonly intervals: readonly {
    readonly level: number
    readonly lower: readonly number[]
    readonly upper: readonly number[]
  }[]
}

/**
 * Prediction intervals of an ETS model, closed-form (normal) for classes 1 and 2 and from the quantiles of simulated
 * paths otherwise.
 *
 * @param fit - The fitted model.
 * @param steps - The horizon.
 * @param options - Levels, method, key and number of paths.
 * @returns The point forecast, the forecast variance and the intervals.
 * @example
 * ```ts
 * import { etsPredictionIntervals } from 'omai-engine/timeseries'
 * const fit = { spec: { error: 'mul', trend: 'none', seasonal: 'none' }, params: { alpha: 0.5, level0: 10 }, sigma2: 0.01,
 *   final: { level: 10, trend: 0, seasonal: [] } } as never
 * etsPredictionIntervals(fit, 2).method // => 'analytic'
 * ```
 */
export function etsPredictionIntervals(
  fit: Pick<EtsFit, 'spec' | 'params' | 'sigma2' | 'final'>,
  steps: number,
  options: EtsIntervalOptions = {},
): EtsIntervals {
  const levels = options.levels ?? [0.95]
  const requested = options.method ?? 'auto'
  const closed = etsPredictionVariance(fit, steps)
  if (requested === 'analytic' && closed === undefined)
    throw new OmaiError('value', 'etsPredictionIntervals: no closed form for this class of model; use simulation')
  if (closed !== undefined && requested !== 'simulation') {
    const mean = etsForecast(fit as EtsFit, steps).mean
    return {
      mean,
      variance: closed,
      method: 'analytic',
      intervals: levels.map((level) => {
        const q = item(ndtri((1 + level) / 2))
        return {
          level,
          lower: mean.map((v, h) => v - q * Math.sqrt(closed[h] ?? 0)),
          upper: mean.map((v, h) => v + q * Math.sqrt(closed[h] ?? 0)),
        }
      }),
    }
  }
  if (options.key === undefined) throw new OmaiError('value', 'etsPredictionIntervals: simulation needs a key')
  const paths = etsSimulate(fit, steps, options.paths ?? 5000, options.key)
  const column = (h: number): number[] => paths.map((p) => p[h] ?? 0).sort((a, b) => a - b)
  const quantile = (sorted: readonly number[], q: number): number => {
    const pos = q * (sorted.length - 1)
    const lo = Math.floor(pos)
    const hi = Math.ceil(pos)
    return (sorted[lo] ?? 0) + ((sorted[hi] ?? 0) - (sorted[lo] ?? 0)) * (pos - lo)
  }
  const cols = Array.from({ length: steps }, (_, h) => column(h))
  const mean = cols.map((c) => c.reduce((a, b) => a + b, 0) / c.length)
  const variance = cols.map((c, h) => c.reduce((a, b) => a + (b - (mean[h] ?? 0)) ** 2, 0) / (c.length - 1))
  return {
    mean,
    variance,
    method: 'simulation',
    intervals: levels.map((level) => ({
      level,
      lower: cols.map((c) => quantile(c, (1 - level) / 2)),
      upper: cols.map((c) => quantile(c, (1 + level) / 2)),
    })),
  }
}
