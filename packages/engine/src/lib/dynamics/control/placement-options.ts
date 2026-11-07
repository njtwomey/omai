/**
 * The slot of pole placement that varies as an option table: the scheme that improves the conditioning of the closed-loop
 * eigenvectors (Tits and Yang's pairwise updates, or Kautsky-Nichols-Van Dooren's method 0 one eigenvector at a time),
 * with the tolerance and sweep cap as knobs. The table is one `union` space for controls and grids, and builds the options
 * of `place` and `placeAlgorithm`.
 */

import { dict, fillKnobs, int, type Option, type OptionTable, optionTable, real } from '../../space/index.ts'
import type { PlaceOptions } from './placement.ts'

function scheme(description: string, method: 'YT' | 'KNV0'): Option<PlaceOptions, Record<string, never>> {
  const space = dict({
    rtol: real(1e-8, 0.5, { default: 1e-3, log: true, label: 'relative tolerance on |det X|' }),
    maxIterations: int(1, 200, { default: 30, label: 'most sweeps' }),
  })
  return {
    description,
    properties: {},
    space,
    make: (config) => {
      const full = fillKnobs(space, config)
      return { method, rtol: full['rtol'] as number, maxIterations: full['maxIterations'] as number }
    },
  }
}

/**
 * The pole-placement schemes as one slot: `yt` (Tits and Yang: pairs of eigenvectors updated together, the default) and
 * `knv0` (Kautsky, Nichols and Van Dooren's method 0: one eigenvector at a time), each with the tolerance and sweep cap.
 *
 * @concept pole-placement
 * @example
 * ```ts
 * import { control } from 'mdml-engine/dynamics'
 * control.placementSchemes.make({ kind: 'knv0' }).method // => 'KNV0'
 * ```
 */
export const placementSchemes: OptionTable<PlaceOptions, Record<string, never>> = optionTable(
  {
    yt: scheme('pairs of eigenvectors updated together (Tits and Yang)', 'YT'),
    knv0: scheme('one eigenvector at a time (KNV method 0)', 'KNV0'),
  },
  { default: 'yt', label: 'placement scheme', description: 'How the closed-loop eigenvectors are made robust.' },
)
