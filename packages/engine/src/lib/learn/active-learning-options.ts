/**
 * The slot of stream-based active learning that varies as an option table: the uncertainty measure that decides
 * whether an arriving row is worth a label (least confidence, margin or normalised entropy of the predicted class
 * probabilities), each scaled to `[0, 1]` with larger meaning less certain so one threshold fits all. The table is one
 * `union` space for controls, grids and tuners, and builds the measure from a configuration.
 */

import { dict, type Option, type OptionTable, optionTable } from '../space/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { classMargin, leastConfidence, predictiveEntropy } from './active-learning.ts'
import { matrixView } from './validation.ts'

/** Uncertainty of each row of class probabilities `[n, k]`, in `[0, 1]`, larger meaning less certain. */
export type UncertaintyMeasure = (proba: TensorLike) => number[]

/** What a measure is sensitive to, for validity rules and selectors. */
export interface UncertaintyProperties {
  /** Whether it looks at the whole distribution (entropy) rather than the top one or two classes. */
  readonly wholeDistribution: boolean
}

function measure(
  description: string,
  properties: UncertaintyProperties,
  build: () => UncertaintyMeasure,
): Option<UncertaintyMeasure, UncertaintyProperties> {
  return { description, properties, space: dict({}), make: build }
}

/**
 * The uncertainty measures of stream active learning as one slot: `leastConfidence` (1 minus the top probability, the
 * default), `margin` (1 minus the gap between the two most probable classes) and `entropy` (the entropy of the
 * distribution over its maximum, log k).
 *
 * @concept active-learning
 * @example
 * ```ts
 * import { streamUncertainties } from 'mdml-engine/learn'
 * streamUncertainties.make({ kind: 'entropy' })([[0.5, 0.5], [1, 0]]) // => [1, 0]
 * ```
 */
export const streamUncertainties: OptionTable<UncertaintyMeasure, UncertaintyProperties> = optionTable(
  {
    leastConfidence: measure(
      'one minus the probability of the most likely class',
      { wholeDistribution: false },
      () => leastConfidence,
    ),
    margin: measure(
      'one minus the gap between the two most likely classes',
      { wholeDistribution: false },
      () => (p) => classMargin(p).map((m) => 1 - m),
    ),
    entropy: measure(
      'the entropy of the class distribution over its largest possible value',
      { wholeDistribution: true },
      () => (p) => {
        const k = matrixView(p, 'streamUncertainties').d
        return predictiveEntropy(p).map((h) => (k > 1 ? h / Math.log(k) : 0))
      },
    ),
  },
  {
    default: 'leastConfidence',
    label: 'uncertainty measure',
    description: 'How unsure the model is of an arriving row, from 0 (certain) to 1.',
  },
)
