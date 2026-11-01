/**
 * The slot of dynamic time warping that varies as an option table: the step pattern (dtw-python's symmetric1,
 * symmetric2 and asymmetric) with the Sakoe-Chiba band as a knob, and what each pattern normalises its distance by. The
 * table is one `union` space for controls and grids, and builds the options of `dtw` and `dtwAlgorithm`.
 */

import { defaults, dict, int, type Option, type OptionTable, optionTable, type SpaceValue } from '../space/index.ts'
import type { DtwOptions } from './similarity.ts'

/** What a step pattern means, for validity rules and selectors. */
export interface StepPatternProperties {
  /** What the cumulative distance is divided by to normalise it. */
  readonly normalisedBy: 'n + m' | 'n' | 'none'
  /** Whether the pattern allows the (1, 2) step, so the second series may be skipped through. */
  readonly skips: boolean
}

function pattern(
  description: string,
  stepPattern: NonNullable<DtwOptions['stepPattern']>,
  properties: StepPatternProperties,
): Option<DtwOptions, StepPatternProperties> {
  const space = dict({ window: int(0, 10000, { default: 10000, label: 'Sakoe-Chiba band (10000 for none)' }) })
  return {
    description,
    properties,
    space,
    make: (config) => {
      const full = { ...(defaults(space) as Readonly<Record<string, SpaceValue>>), ...config }
      const window = full['window'] as number
      return { stepPattern, ...(window >= 10000 ? {} : { window }) }
    },
  }
}

/**
 * The DTW step patterns as one slot: `symmetric2` (diagonal steps weigh double, normalised by n + m, the default),
 * `symmetric1` (every cell counts once, no normalisation) and `asymmetric` (steps through the first series with a (1, 2)
 * skip, normalised by n), each with a Sakoe-Chiba band as a knob.
 *
 * @concept dynamic-time-warping
 * @example
 * ```ts
 * import { dtwStepPatterns } from 'mdml-engine/signal'
 * dtwStepPatterns.make({ kind: 'asymmetric', window: 3 }) // => { stepPattern: 'asymmetric', window: 3 }
 * ```
 */
export const dtwStepPatterns: OptionTable<DtwOptions, StepPatternProperties> = optionTable(
  {
    symmetric2: pattern('diagonal steps weigh double; distance normalised by n + m', 'symmetric2', {
      normalisedBy: 'n + m',
      skips: false,
    }),
    symmetric1: pattern('every cell on the path counts once; no normalisation', 'symmetric1', {
      normalisedBy: 'none',
      skips: false,
    }),
    asymmetric: pattern('one step through the first series per cell, skipping up to one of the second', 'asymmetric', {
      normalisedBy: 'n',
      skips: true,
    }),
  },
  {
    default: 'symmetric2',
    label: 'step pattern',
    description: 'Which moves a warping path may make, and how they are weighed.',
  },
)
