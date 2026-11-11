/**
 * The moments of a Gaussian truncated to an interval, the closed form behind every EP message of a comparison, a
 * probit link or a step: the log of the mass left, the mean and the variance of what remains, accurate in the tails
 * (the mass and the density are taken as ratios in the log scale, and an interval in the upper tail is reflected into
 * the lower one).
 */

import { OmaiError } from '../../base/index.ts'
import { logIntervalProbability } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'
import { logPhiDensity } from '../numerics/_scalar.ts'

/** The mass and the first two moments of a truncated Gaussian. */
export interface TruncatedMoments {
  /** log P(lower < x < upper) under the untruncated Gaussian. */
  readonly logMass: number
  readonly mean: number
  readonly variance: number
}

/**
 * The moments of N(mean, variance) restricted to (lower, upper).
 *
 * @param mean - The mean of the Gaussian.
 * @param variance - Its variance.
 * @param lower - The lower limit (`-Infinity` for none).
 * @param upper - The upper limit (`Infinity` for none).
 * @returns The log mass of the interval, and the mean and variance of the truncated distribution.
 * @throws OmaiError with code `value` for a non-positive variance or an empty interval.
 * @concept truncated-gaussian
 * @index function Mass, mean and variance of a Gaussian truncated to an interval.
 * @example
 * ```ts
 * import { truncatedNormalMoments } from 'omai-engine/message-passing'
 * // The half-normal: mean sqrt(2 / pi), variance 1 - 2 / pi.
 * const t = truncatedNormalMoments(0, 1, 0, Infinity)
 * [t.mean.toFixed(6), t.variance.toFixed(6)] // => ['0.797885', '0.363380']
 * ```
 */
export function truncatedNormalMoments(mean: number, variance: number, lower: number, upper: number): TruncatedMoments {
  if (!(variance > 0))
    throw new OmaiError('value', `truncatedNormalMoments: the variance must be positive, got ${variance}`)
  if (!(lower < upper))
    throw new OmaiError('value', `truncatedNormalMoments: the interval (${lower}, ${upper}) is empty`)
  const sd = Math.sqrt(variance)
  let a = (lower - mean) / sd
  let b = (upper - mean) / sd
  // An interval in the upper tail is the reflection of one in the lower tail: the tail ratios stay well conditioned.
  const flip = a > 0
  if (flip) [a, b] = [-b, -a]
  const logMass = item(logIntervalProbability(a, b))
  const ratio = (z: number): number => (Number.isFinite(z) ? Math.exp(logPhiDensity(z) - logMass) : 0)
  const ra = ratio(a)
  const rb = ratio(b)
  const first = ra - rb
  const second = 1 + (Number.isFinite(a) ? a * ra : 0) - (Number.isFinite(b) ? b * rb : 0) - first * first
  const width = b - a
  const floor = Number.isFinite(width) ? (width * width) / 12 : 0
  const standardVariance = Math.max(second, floor * 1e-3, 1e-300)
  return { logMass, mean: mean + (flip ? -1 : 1) * sd * first, variance: variance * standardVariance }
}
