/**
 * Hazard functions for Bayesian online changepoint detection: the probability that a segment which has run for r
 * observations ends there, h(r), from a prior on the segment length T (the number of observations in a segment, at
 * least one). A geometric prior gives the constant hazard (`constantHazard`); a minimum length forbids short segments;
 * Weibull, gamma and negative-binomial lengths make the hazard rise or fall with the run length. Each is a function of
 * the run length, which is what `bocpd` takes.
 */

import { OmaiError } from '../../base/index.ts'
import { betainc, gammaincc } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'
import { positive } from './_check.ts'
import { survivalHazard } from './changepoint.ts'

/**
 * A geometric prior on the length beyond a minimum: no segment ends before it has `minSegment` observations, then each
 * step ends it with probability `rate` (so the length is `minSegment` plus a geometric number, as the generator
 * `changepointSeries` draws it).
 *
 * @param rate - The probability of a changepoint at each step once the minimum is reached, in (0, 1).
 * @param minSegment - The shortest segment (at least 1).
 * @returns The hazard function of the run length.
 * @throws OmaiError with code `value` for a rate outside (0, 1) or a minimum below 1.
 * @concept hazard-function
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * const h = filtering.minimumSegmentHazard(0.05, 4)
 * [h(3), h(4)] // => [0, 0.05]
 * ```
 */
export function minimumSegmentHazard(rate: number, minSegment: number): (runLength: number) => number {
  if (!(rate > 0 && rate < 1)) throw new OmaiError('value', 'minimumSegmentHazard: the rate must be in (0, 1)')
  if (!(Number.isInteger(minSegment) && minSegment >= 1))
    throw new OmaiError('value', 'minimumSegmentHazard: the minimum segment must be an integer of at least 1')
  return (r) => (r >= minSegment ? rate : 0)
}

/**
 * Weibull segment lengths, T = ceil(W) for W Weibull with the given shape and scale: P(T >= r) = exp(-((r - 1) /
 * scale)^shape). A shape below one makes a segment likelier to end the younger it is, above one the older it is, and
 * one is the memoryless (constant) hazard.
 *
 * @param shape - The Weibull shape (> 0).
 * @param scale - The Weibull scale, about the typical segment length (> 0).
 * @returns The hazard function of the run length.
 * @throws OmaiError with code `value` for a shape or scale that is not positive.
 * @concept hazard-function
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * const h = filtering.weibullHazard(3, 20)
 * [h(5) < h(15), Math.round(filtering.weibullHazard(1, 10)(7) * 1e6) / 1e6] // => [true, 0.095163]
 * ```
 */
export function weibullHazard(shape: number, scale: number): (runLength: number) => number {
  positive('weibullHazard', 'shape', shape)
  positive('weibullHazard', 'scale', scale)
  return survivalHazard((r) => (r <= 1 ? 1 : Math.exp(-(((r - 1) / scale) ** shape))))
}

/**
 * Gamma segment lengths, T = ceil(G) for G gamma with the given shape and rate: P(T >= r) is the upper regularised
 * incomplete gamma function of the shape at rate * (r - 1). A shape above one concentrates the lengths around
 * shape / rate, so the hazard rises as the segment reaches that age.
 *
 * @param shape - The gamma shape (> 0).
 * @param rate - The gamma rate, one over the scale (> 0).
 * @returns The hazard function of the run length.
 * @throws OmaiError with code `value` for a shape or rate that is not positive.
 * @concept hazard-function
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * const h = filtering.gammaHazard(8, 0.4)
 * [h(5) < h(25), Math.round(filtering.gammaHazard(1, 0.1)(7) * 1e6) / 1e6] // => [true, 0.095163]
 * ```
 */
export function gammaHazard(shape: number, rate: number): (runLength: number) => number {
  positive('gammaHazard', 'shape', shape)
  positive('gammaHazard', 'rate', rate)
  return survivalHazard((r) => (r <= 1 ? 1 : item(gammaincc(shape, rate * (r - 1)))))
}

/**
 * Negative-binomial segment lengths, T = 1 + X for X the number of failures before the `successes`-th success of
 * trials with success probability `p`: P(T >= r) = 1 - I_p(successes, r - 1) with I the regularised incomplete beta
 * function. The mean length is 1 + successes (1 - p) / p; one success is the geometric (constant) hazard.
 *
 * @param successes - The number of successes that ends the count (> 0, not necessarily an integer).
 * @param p - The success probability, in (0, 1).
 * @returns The hazard function of the run length.
 * @throws OmaiError with code `value` for successes that are not positive or a probability outside (0, 1).
 * @concept hazard-function
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * const h = filtering.negativeBinomialHazard(6, 0.2)
 * [h(3) < h(30), Math.round(filtering.negativeBinomialHazard(1, 0.1)(7) * 1e6) / 1e6] // => [true, 0.1]
 * ```
 */
export function negativeBinomialHazard(successes: number, p: number): (runLength: number) => number {
  positive('negativeBinomialHazard', 'successes', successes)
  if (!(p > 0 && p < 1)) throw new OmaiError('value', 'negativeBinomialHazard: the probability must be in (0, 1)')
  return survivalHazard((r) => (r <= 1 ? 1 : 1 - item(betainc(successes, r - 1, p))))
}
