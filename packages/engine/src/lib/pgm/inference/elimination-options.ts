/**
 * The semiring of variable elimination as an option table: how a variable is removed from the product of the factors
 * that mention it. Summing it out (`sum`) gives marginals and the partition function; maximising it out (`max`) gives
 * the most probable explanation, with a traceback over the products formed on the way.
 */

import { dict, type Option, type OptionTable, optionTable } from '../../space/index.ts'
import { type Factor, marginalise, maxMarginalise } from '../factor.ts'

/** How a variable is eliminated from a factor. */
export interface EliminationSemiring {
  /** The option's name. */
  readonly name: 'sum' | 'max'
  /** Remove the variables from a factor. */
  readonly eliminate: (f: Factor, variables: readonly string[]) => Factor
  /** Whether the products formed must be kept for a traceback of the maximising assignment. */
  readonly traceback: boolean
}

/** What a semiring answers, for validity rules and selectors. */
export interface EliminationSemiringProperties {
  /** What the elimination yields: marginals (and the partition function) or the most probable assignment. */
  readonly answers: 'marginals' | 'mode'
}

const semiring = (
  semiringValue: EliminationSemiring,
  description: string,
  properties: EliminationSemiringProperties,
): Option<EliminationSemiring, EliminationSemiringProperties> => ({
  description,
  properties,
  space: dict({}),
  make: () => semiringValue,
})

/**
 * The semirings of variable elimination as one slot: `sum` (sum-product: marginals and the partition function) and
 * `max` (max-product: the most probable explanation, recovered by traceback).
 *
 * @concept variable-elimination
 * @example
 * ```ts
 * import { eliminationSemirings } from 'omai-engine/pgm'
 * eliminationSemirings.make({ kind: 'max' }).traceback // => true
 * ```
 */
export const eliminationSemirings: OptionTable<EliminationSemiring, EliminationSemiringProperties> = optionTable(
  {
    sum: semiring(
      { name: 'sum', eliminate: marginalise, traceback: false },
      'sum the variable out (marginals, log Z)',
      {
        answers: 'marginals',
      },
    ),
    max: semiring(
      { name: 'max', eliminate: maxMarginalise, traceback: true },
      'maximise the variable out (the most probable explanation, by traceback)',
      { answers: 'mode' },
    ),
  },
  { default: 'sum', label: 'semiring', description: 'How a variable is removed from the factors that mention it.' },
)
