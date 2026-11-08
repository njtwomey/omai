/**
 * The greedy elimination criteria as an option table: the slot of an elimination ordering (and of variable
 * elimination and the junction tree built on it) that says which vertex goes next. Each option is a cost function over
 * the current graph, with the properties a selector reads: whether it weighs vertices by their number of states, and
 * whether it looks at the fill-in an elimination adds or only at the vertex's degree.
 */

import { dict, type Option, type OptionTable, optionTable } from '../space/index.ts'
import { type EliminationCostFn, eliminationCost } from './triangulation.ts'

/** What a criterion looks at, for validity rules and selectors. */
export interface EliminationHeuristicProperties {
  /** Whether it uses the cardinalities of the variables. */
  readonly weighted: boolean
  /** Whether it counts the fill-in edges elimination would add (rather than the neighbours only). */
  readonly looksAtFill: boolean
}

const criterion = (
  heuristic: 'minDegree' | 'minFill' | 'minWeight' | 'weightedMinFill',
  description: string,
  properties: EliminationHeuristicProperties,
): Option<EliminationCostFn, EliminationHeuristicProperties> => ({
  description,
  properties,
  space: dict({}),
  make: () => (graph, v, cardinality) => eliminationCost(graph, v, heuristic, cardinality),
})

/**
 * The greedy criteria of variable elimination as one slot: `minFill` (the default; fewest fill-in edges),
 * `minDegree` (fewest neighbours), `minWeight` (smallest product of the neighbours' cardinalities, the size of the
 * factor formed) and `weightedMinFill` (fill-in edges weighted by the cardinalities at their ends).
 *
 * @concept elimination-orderings
 * @example
 * ```ts
 * import { eliminationHeuristics, undirectedGraph } from 'omai-engine/pgm'
 * const cost = eliminationHeuristics.make({ kind: 'minDegree' })
 * cost(undirectedGraph([['a', 'b'], ['a', 'c']]), 'a', {}) // => 2
 * ```
 */
export const eliminationHeuristics: OptionTable<EliminationCostFn, EliminationHeuristicProperties> = optionTable(
  {
    minFill: criterion('minFill', 'the vertex whose elimination adds the fewest fill-in edges', {
      weighted: false,
      looksAtFill: true,
    }),
    minDegree: criterion('minDegree', 'the vertex with the fewest neighbours', { weighted: false, looksAtFill: false }),
    minWeight: criterion(
      'minWeight',
      'the vertex whose neighbours have the fewest joint states (the smallest factor)',
      {
        weighted: true,
        looksAtFill: false,
      },
    ),
    weightedMinFill: criterion('weightedMinFill', 'fill-in edges weighted by the states at their two ends', {
      weighted: true,
      looksAtFill: true,
    }),
  },
  {
    default: 'minFill',
    label: 'elimination heuristic',
    description: 'Which variable to eliminate next, by a greedy cost over the current graph.',
  },
)
