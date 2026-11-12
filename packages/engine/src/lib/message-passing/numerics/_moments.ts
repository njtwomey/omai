// The sums every numeric rule of the message algebra starts from: log weights shifted by their maximum and exponentiated, so a
// grid or a mixture whose weights are tiny or huge never underflows, with the total of the shifted weights and the log mass
// (`top + log total`) the rules report. One place for the loop that used to be written in every operator that integrates; the
// sum runs in the order of the array, so a rule that moved here gives the same number it gave before.

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
