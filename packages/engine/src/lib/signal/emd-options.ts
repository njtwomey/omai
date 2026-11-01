/**
 * The stopping rule of empirical mode decomposition as one slot: an option table over Huang's standard-deviation
 * criterion, the S-number criterion and a fixed number of sifts, each with the knobs of its own. Each table entry is one
 * `union` space for controls, grids and tuners and builds the `SiftStopRule` an `emdAlgorithm` takes.
 */

import { dict, int, type OptionTable, optionTable, real } from '../space/index.ts'
import { fixedSiftStop, type SiftStopRule, sdStop, sNumberStop } from './decompositions.ts'

/** What a stopping rule is like, for validity rules and selectors. */
export interface SiftStopProperties {
  /** Whether the rule looks at how much a sift changed the candidate (rather than at its shape or a count). */
  readonly looksAtChange: boolean
  /** Whether the number of sifts is fixed in advance. */
  readonly fixedCount: boolean
}

/**
 * The sifting stopping rules as one slot: `sd` (stop when a sift changes the candidate by less than a fraction of its
 * energy: Huang 1998), `s-number` (stop when the number of extrema and zero crossings agree to within one for S sifts in a
 * row: Huang, Wu et al. 2003) and `fixed` (a set number of sifts). Looser rules give fewer sifts and modes that are not
 * quite intrinsic; tighter ones sift the amplitude modulation away.
 *
 * @concept empirical-mode-decomposition
 * @example
 * ```ts
 * import { siftStops } from 'mdml-engine/signal'
 * siftStops.names // => ['sd', 's-number', 'fixed']
 * ```
 */
export const siftStops: OptionTable<SiftStopRule, SiftStopProperties> = optionTable(
  {
    sd: {
      description: 'stop when sum (h_old - h)^2 / sum h_old^2 falls below a threshold',
      properties: { looksAtChange: true, fixedCount: false },
      space: dict({ threshold: real(1e-4, 1, { log: true, default: 0.2, label: 'standard-deviation threshold' }) }),
      make: (config) => sdStop((config['threshold'] as number | undefined) ?? 0.2),
    },
    's-number': {
      description: 'stop when extrema and zero crossings differ by at most one for S sifts in a row',
      properties: { looksAtChange: false, fixedCount: false },
      space: dict({ s: int(1, 20, { default: 4, label: 'consecutive sifts S' }) }),
      make: (config) => sNumberStop((config['s'] as number | undefined) ?? 4),
    },
    fixed: {
      description: 'a set number of sifts for every mode',
      properties: { looksAtChange: false, fixedCount: true },
      space: dict({ count: int(1, 100, { default: 10, label: 'number of sifts' }) }),
      make: (config) => fixedSiftStop((config['count'] as number | undefined) ?? 10),
    },
  },
  {
    default: 'sd',
    label: 'sifting stopping rule',
    description: 'When the sifting of an intrinsic mode function ends.',
  },
)
