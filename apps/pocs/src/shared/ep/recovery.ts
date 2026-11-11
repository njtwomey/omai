/** The recovery check of an EP program: the truth against EP's credible intervals, and what the posterior spread says. */

/** One coordinate EP inferred: its posterior mean and spread and its equal-tailed interval. */
export interface Estimate {
  readonly mean: number
  readonly sd: number
  readonly lower: number
  readonly upper: number
}

/** How well a set of intervals recovered the truth. */
export interface Recovery {
  /** The fraction of true values inside their interval. */
  readonly coverage: number
  /** Root mean square error of the posterior means. */
  readonly rmse: number
  /** The posterior sd averaged (root mean square), to compare with `rmse`: an honest posterior has them near equal. */
  readonly meanSd: number
  /** The root mean square of (error / sd): near 1 for a calibrated posterior, above 1 for an overconfident one. */
  readonly zrms: number
}

/**
 * The truth against EP's estimates, coordinate by coordinate.
 *
 * @param estimates - EP's posterior for each coordinate.
 * @param truth - The true value of each coordinate, in the same order.
 * @returns Coverage of the intervals, the error and the calibration of the spread.
 */
export function recovery(estimates: readonly Estimate[], truth: readonly number[]): Recovery {
  const n = Math.min(estimates.length, truth.length)
  let hit = 0
  let se = 0
  let sv = 0
  let sz = 0
  for (let i = 0; i < n; i++) {
    const e = estimates[i] as Estimate
    const t = truth[i] as number
    if (t >= e.lower && t <= e.upper) hit++
    se += (e.mean - t) ** 2
    sv += e.sd ** 2
    sz += ((e.mean - t) / e.sd) ** 2
  }
  return { coverage: hit / n, rmse: Math.sqrt(se / n), meanSd: Math.sqrt(sv / n), zrms: Math.sqrt(sz / n) }
}

/** What a program promises: the intervals are 90%, so at least `coverage` of the truth must be inside; the error is bounded. */
export interface Tolerance {
  readonly coverage: number
  readonly zrms: number
}

/** The default check: at least 80% of the truth inside the 90% intervals, and errors no more than 1.6 posterior sds. */
export const TOLERANCE: Tolerance = { coverage: 0.8, zrms: 1.6 }

/**
 * Whether a recovery passes.
 *
 * @param r - The recovery.
 * @param t - The tolerance.
 * @returns True when coverage is at least the tolerance's and the calibration ratio at most its.
 */
export function passes(r: Recovery, t: Tolerance = TOLERANCE): boolean {
  return r.coverage >= t.coverage && r.zrms <= t.zrms
}

/** EP's posterior against a reference's (a long NUTS run), coordinate by coordinate. */
export interface Agreement {
  /** The largest |EP mean - reference mean| in reference sds. */
  readonly meanZ: number
  /** The mean of EP sd / reference sd (below 1: EP is more confident than the reference). */
  readonly sdRatio: number
  /** The smallest and the largest sd ratio. */
  readonly sdRatioRange: readonly [number, number]
}

/**
 * How closely EP's means and spreads match a reference's.
 *
 * @param ep - EP's estimates.
 * @param reference - The reference's posterior means and sds, in the same order.
 * @returns The worst mean gap, the average spread ratio and its range.
 */
export function agreement(
  ep: readonly Estimate[],
  reference: { readonly mean: readonly number[]; readonly sd: readonly number[] },
): Agreement {
  let meanZ = 0
  let sum = 0
  let lo = Number.POSITIVE_INFINITY
  let hi = 0
  const n = Math.min(ep.length, reference.mean.length)
  for (let i = 0; i < n; i++) {
    const e = ep[i] as Estimate
    const sd = reference.sd[i] as number
    meanZ = Math.max(meanZ, Math.abs(e.mean - (reference.mean[i] as number)) / sd)
    const r = e.sd / sd
    sum += r
    lo = Math.min(lo, r)
    hi = Math.max(hi, r)
  }
  return { meanZ, sdRatio: sum / n, sdRatioRange: [lo, hi] }
}
