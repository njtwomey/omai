/**
 * The named tolerance tiers of the atoms. How close EP's beliefs must be to the reference is a ladder of a dozen rungs, from
 * `exact` (a conjugate model: rounding only) through the Gaussian projections of a skewed or truncated posterior (`tight`, `close`,
 * `skewed`, `wide`) to the lumpy posteriors of the gates (`lumpy`, `lumpiest`), and an atom says its rung and, in a sentence, why
 * EP is that far off. A rung is the tightest that holds the atom's measured statistics with room to spare; an atom that sits
 * near the edge of every rung (or whose reference has its own resolution) says `measured` and gives its numbers. Recoveries of
 * the learned atoms have their own three rungs.
 */

import { OmaiError } from '../../base/index.ts'
import type { AtomRecoverySpec, AtomTolerance } from './atom-types.ts'

/** The name of a rung of the tolerance ladder, or `measured` (the atom gives its own numbers). */
export type ToleranceTier =
  | 'exact'
  | 'exact-curve'
  | 'exact-heavy-tail'
  | 'exact-moments'
  | 'plated'
  | 'tight'
  | 'gamma-projected'
  | 'close'
  | 'truncated'
  | 'skew-normal'
  | 'skewed'
  | 'wide'
  | 'coupled'
  | 'lumpy'
  | 'lumpiest'
  | 'truth'
  | 'measured'

type MeasuredFreeTier = Exclude<ToleranceTier, 'measured'>

/** The rungs: the mean's error in reference sds, the range of the sd ratio, the Kolmogorov-Smirnov distance. */
const RUNGS: Readonly<Record<MeasuredFreeTier, Omit<AtomTolerance, 'reason'>>> = {
  /** A conjugate model: EP carries the exact posterior, so the belief equals the closed form up to rounding. */
  exact: { meanZ: 1e-6, sdRatio: [0.999999, 1.000001], ks: 1e-4 },
  /** Exact moments, a reference curve with its own grid resolution in the Kolmogorov distance. */
  'exact-curve': { meanZ: 1e-6, sdRatio: [0.999999, 1.000001], ks: 0.005 },
  /** Exact moments of a heavy-tailed posterior (a Student t), whose reference curve is coarse in the tails. */
  'exact-heavy-tail': { meanZ: 1e-6, sdRatio: [0.999999, 1.000001], ks: 0.1 },
  /** Exact moments, and a posterior that is a mixture (a gate) of which EP draws one Gaussian. */
  'exact-moments': { meanZ: 1e-3, sdRatio: [0.999, 1.001], ks: 0.25 },
  /** One plated factor: the marginals are the exact ones to a few per cent. */
  plated: { meanZ: 0.1, sdRatio: [0.9, 1.1], ks: 0.06 },
  /** A Gaussian projection of each tilted distribution, evaluated by quadrature. */
  tight: { meanZ: 0.1, sdRatio: [0.8, 1.15], ks: 0.06 },
  /** A gamma projection of a skewed precision. */
  'gamma-projected': { meanZ: 0.1, sdRatio: [0.7, 1.15], ks: 0.1 },
  /** Gaussian projections of tilted distributions that are not Gaussian. */
  close: { meanZ: 0.15, sdRatio: [0.8, 1.2], ks: 0.08 },
  /** A truncated Gaussian projected onto a Gaussian. */
  truncated: { meanZ: 0.15, sdRatio: [0.75, 1.15], ks: 0.1 },
  /** A skew-normal posterior matched by its mean and variance. */
  'skew-normal': { meanZ: 0.15, sdRatio: [0.7, 1.2], ks: 0.12 },
  /** A skewed posterior of a hyperparameter carried by one gamma or Gaussian. */
  skewed: { meanZ: 0.15, sdRatio: [0.75, 1.25], ks: 0.1 },
  /** As `skewed`, with a little more room on the mean. */
  wide: { meanZ: 0.2, sdRatio: [0.75, 1.25], ks: 0.1 },
  /** Coupled unknowns, each given one Gaussian. */
  coupled: { meanZ: 0.25, sdRatio: [0.7, 1.3], ks: 0.15 },
  /** A lumpy or curved posterior of which EP matches the mean and variance. */
  lumpy: { meanZ: 0.3, sdRatio: [0.6, 1.5], ks: 0.2 },
  /** A posterior with several humps, of which one Gaussian is a coarse summary. */
  lumpiest: { meanZ: 0.5, sdRatio: [0.5, 1.6], ks: 0.3 },
  /** There is no reference posterior: the check is the recovery of the truth alone (a pass is never refused on these numbers). */
  truth: { meanZ: 1, sdRatio: [0, 10], ks: 1 },
}

/**
 * The tolerance of an atom: a named rung of the ladder with the atom's own reason, or `measured` with the atom's numbers where
 * no rung fits (an atom whose statistics sit near every rung's edge, and whose numbers are then load-bearing).
 *
 * @param tier - The rung, or `measured`.
 * @param reason - Why EP is that far from the reference.
 * @param own - The numbers, for `measured` (required) and as an override of the rung's (optional).
 * @returns The tolerance.
 * @throws Error when `measured` gives no numbers.
 * @example
 * ```ts
 * import { tolerance } from 'omai-engine/probprog/ep'
 * tolerance('close', 'a Gaussian projection').meanZ // => 0.15
 * ```
 */
export function tolerance(
  tier: ToleranceTier,
  reason: string,
  own?: Partial<Omit<AtomTolerance, 'reason'>>,
): AtomTolerance {
  if (tier === 'measured') {
    if (own?.meanZ === undefined || own.sdRatio === undefined || own.ks === undefined)
      throw new OmaiError('value', 'tolerance: a measured tolerance gives meanZ, sdRatio and ks')
    return { meanZ: own.meanZ, sdRatio: own.sdRatio, ks: own.ks, reason }
  }
  return { ...RUNGS[tier], ...own, reason }
}

/** The name of a rung of the recovery ladder, or `measured`. */
export type RecoveryTier = 'standard' | 'quick' | 'calibrated' | 'measured'

/** The rungs of the recovery ladder: the number of seeds, the least share of true values inside the 90% intervals, the most rms error in sds. */
const RECOVERY_RUNGS: Readonly<Record<Exclude<RecoveryTier, 'measured'>, Omit<AtomRecoverySpec, 'reason'>>> = {
  /** Forty seeds, the intervals hold the truth at least three times in four, errors under 1.6 sds. */
  standard: { seeds: 40, coverage: 0.75, zrms: 1.6 },
  /** Twenty seeds (a slower atom), a little more room on the coverage. */
  quick: { seeds: 20, coverage: 0.7, zrms: 1.6 },
  /** Forty seeds of a posterior that is exact or nearly so: errors under 1.5 sds. */
  calibrated: { seeds: 40, coverage: 0.75, zrms: 1.5 },
}

/**
 * What an atom promises about recovering its ground truth: a named rung with the atom's reason, or `measured` with its own
 * numbers.
 *
 * @param tier - The rung, or `measured`.
 * @param reason - Why these floors (the number of seeds, what EP approximates, what the prior brings).
 * @param own - The numbers, for `measured` (required).
 * @returns The recovery promise.
 * @throws Error when `measured` gives no numbers.
 * @example
 * ```ts
 * import { recovery } from 'omai-engine/probprog/ep'
 * recovery('standard', 'an exact posterior').seeds // => 40
 * ```
 */
export function recovery(
  tier: RecoveryTier,
  reason: string,
  own?: Partial<Omit<AtomRecoverySpec, 'reason'>>,
): AtomRecoverySpec {
  if (tier === 'measured') {
    if (own?.seeds === undefined || own.coverage === undefined || own.zrms === undefined)
      throw new OmaiError('value', 'recovery: a measured recovery gives seeds, coverage and zrms')
    return { seeds: own.seeds, coverage: own.coverage, zrms: own.zrms, reason }
  }
  return { ...RECOVERY_RUNGS[tier], ...own, reason }
}
