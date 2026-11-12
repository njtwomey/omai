// The sums every numeric rule of the message algebra starts from: log weights shifted by their maximum and exponentiated, so a
// grid or a mixture whose weights are tiny or huge never underflows, with the total of the shifted weights and the log mass
// (`top + log total`) the rules report. One place for the loop that used to be written in every operator that integrates; the
// sum runs in the order of the array, so a rule that moved here gave the same number it gave before. Also the two things every
// variance of a tilted distribution found from moments goes through: `centredMoments` (the variance as the weighted squares of the
// deviations, not a difference of second moments) and `resolvedVariance` (a variance that is not positive is a collapse onto a
// point, reported as an improper message and never floored to a tiny number).

import { ImproperMessageError } from '../core/errors.ts'

/** The log weights shifted by their maximum: what `shiftedSum` returns. */
export interface ShiftedSum {
  /** The largest log weight (`-Infinity` when every weight is zero, `NaN` for a NaN: the caller decides what that means). */
  readonly top: number
  /** `exp(logw[i] - top)`: every weight is at most one, and the largest is exactly one. */
  readonly weights: Float64Array
  /** The sum of `weights`, added in order from the first. */
  readonly total: number
  /** The log of the sum of `exp(logw)`: `top + log(total)`. */
  readonly logTotal: number
}

/**
 * The log weights shifted by their maximum and exponentiated, with their total and the log of the sum of the unshifted weights.
 * The total is added in the order of the array, from zero; a rule that adds the same weights in the same order in its own loop
 * gets the same last digits from this.
 *
 * @param logw - The log weights.
 * @returns The shift `top`, the shifted weights, their `total` and `logTotal = top + log(total)`.
 */
export function shiftedSum(logw: ArrayLike<number>): ShiftedSum {
  let top = Number.NEGATIVE_INFINITY
  for (let i = 0; i < logw.length; i++) if ((logw[i] as number) > top) top = logw[i] as number
  const weights = new Float64Array(logw.length)
  let total = 0
  for (let i = 0; i < logw.length; i++) {
    const w = Math.exp((logw[i] as number) - top)
    weights[i] = w
    total += w
  }
  return { top, weights, total, logTotal: top + Math.log(total) }
}

/**
 * The variance of a tilted distribution found from moments of a grid or a mixture, which has to be positive and finite. One that is
 * not (every weight on one node, the arithmetic unable to resolve a spread) says the distribution collapsed onto a point, and that is
 * reported as an improper message, for the engine's improper policy to count and repair, instead of being floored to a tiny number:
 * a floor hands the next update a point mass with a precision of 1e300 as if it were a message.
 *
 * @param variance - The variance from the moments.
 * @param what - The operator, for the message.
 * @returns The variance.
 * @throws ImproperMessageError when the variance is not positive and finite.
 */
export function resolvedVariance(variance: number, what: string): number {
  if (variance > 0 && Number.isFinite(variance)) return variance
  throw new ImproperMessageError(
    `${what}: the tilted distribution collapsed onto a point (its variance is ${variance})`,
  )
}

/**
 * The mean and the variance of `values` under weights that sum to `total`, in two passes: the variance is the weighted mean of the
 * squares of the deviations from the mean, never the difference of two large numbers `E v^2 - (E v)^2` (which loses every digit
 * when the weights sit on a spread far narrower than the values are large, the case a grid zoomed onto its mass produces).
 *
 * @param weights - The weights.
 * @param total - Their sum (the caller has it).
 * @param values - The value at each weight.
 * @returns The mean and the centred variance (zero when all the weight is on one value).
 */
export function centredMoments(
  weights: ArrayLike<number>,
  total: number,
  values: ArrayLike<number>,
): { readonly mean: number; readonly variance: number } {
  let m = 0
  for (let i = 0; i < weights.length; i++) m += (weights[i] as number) * (values[i] as number)
  const mean = m / total
  let s = 0
  for (let i = 0; i < weights.length; i++) s += (weights[i] as number) * ((values[i] as number) - mean) ** 2
  return { mean, variance: s / total }
}
