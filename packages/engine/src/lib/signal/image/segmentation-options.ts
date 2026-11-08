/**
 * The slot of graph-based segmentation that varies as an option table: how coarse the components are, from fine to
 * coarse, each a scale of observation and minimum size with the smoothing sigma as a knob. The table is one `union`
 * space for controls and grids, and builds the options of `felzenszwalb` and `felzenszwalbAlgorithm`.
 */

import { dict, fillKnobs, int, type Option, type OptionTable, optionTable, real } from '../../space/index.ts'
import type { FelzenszwalbOptions } from './segmentation.ts'

function coarseness(
  description: string,
  scale: number,
  minSize: number,
): Option<FelzenszwalbOptions, Record<string, never>> {
  const space = dict({
    scale: real(0.01, 1000, { default: scale, log: true, label: 'scale of observation' }),
    sigma: real(0, 5, { default: 0.8, label: 'smoothing sigma' }),
    minSize: int(1, 1000, { default: minSize, label: 'minimum component size' }),
  })
  return {
    description,
    properties: {},
    space,
    make: (config) => {
      const full = fillKnobs(space, config)
      return { scale: full['scale'] as number, sigma: full['sigma'] as number, minSize: full['minSize'] as number }
    },
  }
}

/**
 * How coarse the segmentation is, as one slot: `fine` (scale 1, components of at least 10 pixels), `medium` (scale 100)
 * and `coarse` (scale 500, at least 100 pixels), each with the scale, smoothing sigma and minimum size as knobs.
 *
 * @concept graph-segmentation
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * image.felzenszwalbCoarseness.make({ kind: 'coarse' }).scale // => 500
 * ```
 */
export const felzenszwalbCoarseness: OptionTable<FelzenszwalbOptions, Record<string, never>> = optionTable(
  {
    fine: coarseness('many small components', 1, 10),
    medium: coarseness('components of a few dozen pixels', 100, 30),
    coarse: coarseness('few large components', 500, 100),
  },
  { default: 'fine', label: 'coarseness', description: 'How large the components of the segmentation are.' },
)
