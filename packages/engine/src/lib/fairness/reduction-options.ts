/**
 * The slot of the fairness reductions that varies as an option table: the moment constrained, either demographic
 * parity, equalised odds, or one half of it (true-positive or false-positive rate parity). The table is one `union`
 * space for controls, grids and tuners, and builds the constraint name from a configuration.
 */

import { dict, type Option, type OptionTable, optionTable } from '../space/index.ts'
import type { ReductionConstraint } from './in-processing.ts'

/** What a constraint equalises, for validity rules and selectors. */
export interface ReductionConstraintProperties {
  /** Whether it conditions on the true label (so it needs labels) or only on the decision. */
  readonly conditionsOnLabel: boolean
}

function constraint(
  description: string,
  name: ReductionConstraint,
  conditionsOnLabel: boolean,
): Option<ReductionConstraint, ReductionConstraintProperties> {
  return { description, properties: { conditionsOnLabel }, space: dict({}), make: () => name }
}

/**
 * The constraints of the reductions as one slot: `demographicParity` (equal rate of positive decisions across groups),
 * `equalisedOdds` (equal true- and false-positive rates), `truePositiveRateParity` (equal opportunity) and
 * `falsePositiveRateParity`.
 *
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { reductionConstraints } from 'omai-engine/fairness'
 * reductionConstraints.make({ kind: 'equalisedOdds' }) // => 'equalisedOdds'
 * ```
 */
export const reductionConstraints: OptionTable<ReductionConstraint, ReductionConstraintProperties> = optionTable(
  {
    demographicParity: constraint('equal rate of positive decisions across groups', 'demographicParity', false),
    equalisedOdds: constraint('equal true-positive and false-positive rates across groups', 'equalisedOdds', true),
    truePositiveRateParity: constraint(
      'equal true-positive rate across groups (equal opportunity)',
      'truePositiveRateParity',
      true,
    ),
    falsePositiveRateParity: constraint('equal false-positive rate across groups', 'falsePositiveRateParity', true),
  },
  {
    default: 'demographicParity',
    label: 'constraint',
    description: 'The moment the classifier is constrained to equalise.',
  },
)
