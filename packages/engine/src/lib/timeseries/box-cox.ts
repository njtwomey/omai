/**
 * Box-Cox transformed decomposition: the exponent lambda is chosen from the series (Guerrero's method, which makes the
 * spread of the seasonal blocks independent of their level, or the profile likelihood of the Box-Cox family), the MSTL
 * decomposition is done on the transformed scale where trend and seasonals add, and the fitted components are carried
 * back to the original scale. With lambda = 0 the original series is the product of its components, which is what a
 * seasonal amplitude growing with the level calls for.
 */

import { meanOf, minOf, OmaiError, varianceOf } from '../base/index.ts'
import { fminbound } from '../optim/index.ts'
import { boxcoxNormmax } from '../stats/index.ts'
import { asSeries } from './_series.ts'
import { type MstlOptions, type MstlResult, mstl } from './stl.ts'

/** Its inverse. */
const inverse = (w: number, lambda: number): number => (lambda === 0 ? Math.exp(w) : (lambda * w + 1) ** (1 / lambda))

/**
 * Guerrero's choice of the Box-Cox exponent (as R's `forecast::BoxCox.lambda`): the lambda in [lower, upper] that makes
 * the coefficient of variation of sd / mean^(1 - lambda) smallest over the consecutive blocks of one seasonal period,
 * so that the within-block spread no longer grows with the block's level.
 *
 * @param y - A positive series.
 * @param period - The seasonal period (the block length).
 * @param bounds - The interval searched (default [-1, 2]).
 * @returns The exponent.
 * @example
 * ```ts
 * import { guerreroLambda } from 'omai-engine/timeseries'
 * // Blocks whose spread is proportional to their level call for the logarithm.
 * const y = [1, 2, 3, 2, 10, 20, 30, 20, 100, 200, 300, 200]
 * Math.abs(guerreroLambda(y, 4)) < 0.01 // => true
 * ```
 */
export function guerreroLambda(
  y: ArrayLike<number>,
  period: number,
  bounds: readonly [number, number] = [-1, 2],
): number {
  const x = asSeries(Float64Array.from(y), 'guerreroLambda')
  if (x.some((v) => !(v > 0))) throw new OmaiError('value', 'guerreroLambda: the series must be positive')
  const blocks = Math.floor(x.length / period)
  if (blocks < 2) throw new OmaiError('value', 'guerreroLambda: need at least two blocks of one period')
  const tail = x.slice(x.length - blocks * period)
  const means: number[] = []
  const sds: number[] = []
  for (let b = 0; b < blocks; b++) {
    const block = tail.slice(b * period, (b + 1) * period)
    means.push(meanOf(block))
    sds.push(Math.sqrt(varianceOf(block, 1)))
  }
  const cv = (lambda: number): number => {
    const ratio = sds.map((s, b) => s / (means[b] ?? 1) ** (1 - lambda))
    return Math.sqrt(varianceOf(ratio, 1)) / meanOf(ratio)
  }
  return fminbound(cv, { bounds: [bounds[0], bounds[1]] }).x
}

/** How the exponent is chosen: a number, Guerrero's method, or the profile likelihood. */
export type BoxCoxChoice = number | 'guerrero' | 'mle'

/** The options of {@link boxCoxMstl}. */
export interface BoxCoxMstlOptions extends Omit<MstlOptions, 'lambda'> {
  /** The exponent, or the rule that picks it (default `'guerrero'` with the shortest period). */
  readonly lambda?: BoxCoxChoice
}

/** The result of {@link boxCoxMstl}. */
export interface BoxCoxMstlResult {
  readonly lambda: number
  /** The decomposition on the transformed scale, where the components add. */
  readonly decomposition: MstlResult
  /** The trend plus every seasonal component, carried back to the original scale. */
  readonly fitted: readonly number[]
  /** The trend carried back to the original scale. */
  readonly trend: readonly number[]
  /** For each period, the seasonal factor at the original scale: the fitted value with the seasonal over without it. */
  readonly seasonalFactors: readonly (readonly number[])[]
}

/**
 * MSTL on the Box-Cox transformed series with the exponent chosen from the data, and the components back-transformed.
 *
 * @param y - A positive series.
 * @param options - The periods, STL settings and the exponent or the rule that picks it.
 * @returns The exponent, the decomposition on the transformed scale and the original-scale fit, trend and seasonal factors.
 * @example
 * ```ts
 * import { boxCoxMstl } from 'omai-engine/timeseries'
 * const y = Array.from({ length: 96 }, (_, t) => Math.exp(0.02 * t) * (2 + Math.sin((2 * Math.PI * t) / 12)))
 * boxCoxMstl(y, { periods: [12], lambda: 0 }).lambda // => 0
 * ```
 */
export function boxCoxMstl(y: ArrayLike<number>, options: BoxCoxMstlOptions): BoxCoxMstlResult {
  const x = Array.from(asSeries(Float64Array.from(y), 'boxCoxMstl'))
  const choice = options.lambda ?? 'guerrero'
  const shortest = minOf(options.periods)
  const lambda = typeof choice === 'number' ? choice : choice === 'mle' ? boxcoxNormmax(x) : guerreroLambda(x, shortest)
  const { lambda: _drop, ...rest } = options
  const decomposition = mstl(x, { ...rest, lambda })
  const n = x.length
  const seasonalSum = (t: number): number => decomposition.seasonal.reduce((a, s) => a + (s[t] ?? 0), 0)
  const fitted = Array.from({ length: n }, (_, t) => inverse((decomposition.trend[t] ?? 0) + seasonalSum(t), lambda))
  const trend = Array.from({ length: n }, (_, t) => inverse(decomposition.trend[t] ?? 0, lambda))
  const seasonalFactors = decomposition.seasonal.map((s) =>
    Array.from({ length: n }, (_, t) => inverse((decomposition.trend[t] ?? 0) + (s[t] ?? 0), lambda) / (trend[t] ?? 1)),
  )
  return { lambda, decomposition, fitted, trend, seasonalFactors }
}
