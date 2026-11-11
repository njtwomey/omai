/**
 * What it takes to say that an approximate posterior recovered the truth a program was simulated from: the 90% interval of a
 * quantity that is a mixture of Gaussians (a gated coefficient), the coverage of the true values by those intervals, the error
 * in units of the posterior sd, the agreement of the spreads with a reference, and the score of an inferred active set (which
 * coefficients are non-zero) against the true one. The EP programs of the pocs group use these for their check and their test.
 */

import type { MixtureComponent } from './ep-atoms-quad.ts'
import { normalCdf } from './ep-atoms-quad.ts'

/**
 * The equal-tailed interval of a finite mixture of normals, by bisection on its distribution function.
 *
 * @param components - The weighted components (the weights are normalised).
 * @param level - The mass inside (default 0.9).
 * @returns The lower and upper ends.
 * @index function The equal-tailed credible interval of a mixture of Gaussians (a gated coefficient).
 * @example
 * ```ts
 * import { mixtureInterval } from 'omai-engine/probprog/ep'
 * const [lo, hi] = mixtureInterval([{ weight: 1, mean: 0, sd: 1 }], 0.9)
 * [lo.toFixed(3), hi.toFixed(3)] // => ['-1.645', '1.645']
 * ```
 */
export function mixtureInterval(components: readonly MixtureComponent[], level = 0.9): readonly [number, number] {
  const total = components.reduce((a, c) => a + c.weight, 0)
  const cdf = (x: number): number =>
    components.reduce((a, c) => a + (c.weight / total) * normalCdf((x - c.mean) / c.sd), 0)
  const lo0 = Math.min(...components.map((c) => c.mean - 12 * c.sd))
  const hi0 = Math.max(...components.map((c) => c.mean + 12 * c.sd))
  const quantile = (p: number): number => {
    let lo = lo0
    let hi = hi0
    for (let i = 0; i < 80; i++) {
      const mid = 0.5 * (lo + hi)
      if (cdf(mid) < p) lo = mid
      else hi = mid
    }
    return 0.5 * (lo + hi)
  }
  return [quantile((1 - level) / 2), quantile(1 - (1 - level) / 2)]
}

/** The values a program recovered, beside the truth it was simulated from (one entry per quantity). */
export interface Recovered {
  readonly truth: readonly number[]
  readonly mean: readonly number[]
  readonly sd: readonly number[]
  /** The ends of each quantity's credible interval. */
  readonly lower: readonly number[]
  readonly upper: readonly number[]
  /** A reference posterior (exact, or a long run) of the same quantities, when there is one. */
  readonly reference?: { readonly mean: readonly number[]; readonly sd: readonly number[] }
}

/** How well a posterior recovered the truth. */
export interface RecoverySummary {
  /** The share of the true values inside their interval. */
  readonly coverage: number
  /** The root mean square error of the means. */
  readonly rmse: number
  /** The largest error of a mean in posterior sds. */
  readonly maxZ: number
  /** Against the reference: the largest error of a mean in reference sds, and the range of the ratio of the sds (EP's over the reference's). */
  readonly reference?: { readonly maxMeanZ: number; readonly sdRatio: readonly [number, number] }
}

/**
 * The recovery of a posterior: the coverage of the true values by its intervals, the error of its means, and, with a
 * reference, how far its means and spreads are from the reference's.
 *
 * @param r - The truth, the posterior's means, sds and intervals, and the reference.
 * @returns The summary.
 * @index function How well a posterior recovered the truth: interval coverage, error, agreement with a reference.
 * @example
 * ```ts
 * import { recoverySummary } from 'omai-engine/probprog/ep'
 * const s = recoverySummary({ truth: [0, 1], mean: [0.1, 0.9], sd: [0.1, 0.1], lower: [-0.1, 0.7], upper: [0.3, 1.1] })
 * [s.coverage, s.rmse.toFixed(2)] // => [1, '0.10']
 * ```
 */
export function recoverySummary(r: Recovered): RecoverySummary {
  const n = r.truth.length
  let covered = 0
  let sq = 0
  let maxZ = 0
  for (let i = 0; i < n; i++) {
    const t = r.truth[i] as number
    const m = r.mean[i] as number
    if (t >= (r.lower[i] as number) && t <= (r.upper[i] as number)) covered++
    sq += (m - t) ** 2
    maxZ = Math.max(maxZ, Math.abs(m - t) / Math.max(r.sd[i] as number, 1e-12))
  }
  const base = { coverage: covered / n, rmse: Math.sqrt(sq / n), maxZ }
  const ref = r.reference
  if (ref === undefined) return base
  let maxMeanZ = 0
  let lo = Number.POSITIVE_INFINITY
  let hi = 0
  for (let i = 0; i < n; i++) {
    const s = Math.max(ref.sd[i] as number, 1e-12)
    maxMeanZ = Math.max(maxMeanZ, Math.abs((r.mean[i] as number) - (ref.mean[i] as number)) / s)
    const ratio = (r.sd[i] as number) / s
    lo = Math.min(lo, ratio)
    hi = Math.max(hi, ratio)
  }
  return { ...base, reference: { maxMeanZ, sdRatio: [lo, hi] } }
}

/** How well inclusion probabilities found the true active set. */
export interface ActiveSetScore {
  /** The probability that an active quantity is ranked above an inactive one (1 is a perfect ranking). */
  readonly auc: number
  /** At the threshold: the active ones found, the active ones missed and the inactive ones included. */
  readonly found: number
  readonly missed: number
  readonly falsePositives: number
}

/**
 * The score of inclusion probabilities against the true active set: the ranking quality (AUC) and the counts at a threshold.
 *
 * @param probability - The posterior probability that each quantity is active.
 * @param active - Which quantities are truly active.
 * @param threshold - The probability above which one is called active (default 0.5).
 * @returns The score.
 * @index function The score of inclusion probabilities against the true active set: AUC and counts at a threshold.
 * @example
 * ```ts
 * import { activeSetScore } from 'omai-engine/probprog/ep'
 * activeSetScore([0.99, 0.2, 0.7, 0.01], [true, false, false, false]) // => { auc: 1, found: 1, missed: 0, falsePositives: 1 }
 * ```
 */
export function activeSetScore(
  probability: readonly number[],
  active: readonly boolean[],
  threshold = 0.5,
): ActiveSetScore {
  let wins = 0
  let pairs = 0
  let found = 0
  let missed = 0
  let falsePositives = 0
  probability.forEach((p, i) => {
    if (active[i] === true) {
      if (p > threshold) found++
      else missed++
    } else if (p > threshold) falsePositives++
    if (active[i] !== true) return
    probability.forEach((q, j) => {
      if (active[j] === true) return
      pairs++
      wins += p > q ? 1 : p === q ? 0.5 : 0
    })
  })
  return { auc: pairs === 0 ? 1 : wins / pairs, found, missed, falsePositives }
}
