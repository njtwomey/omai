/**
 * A grid world as a search problem: moves to open neighbours costing the terrain of the cell entered, the distance to
 * the nearest goal as an admissible heuristic, and the reverse moves for bidirectional search. The problem the path
 * searches of `lib/search` run on the maps and mazes of `lib/worlds`.
 */

import { MdmlError } from '../base/index.ts'
import type { BidirectionalProblem } from '../search/index.ts'
import { type CellMetric, cellCoords, exits, heuristicTo, terrainOf, type World } from '../worlds/index.ts'

/** Options of {@link gridProblem}. */
export interface GridProblemOptions {
  /** 4 (default) or 8 neighbours. */
  readonly connectivity?: 4 | 8
  /** The heuristic's cell distance (default manhattan, octile under 8-connectivity). */
  readonly metric?: CellMetric
}

/**
 * A grid world as a search problem over cells: moves to open neighbours, costing the terrain of the cell entered, with
 * the distance to the nearest goal as heuristic (admissible: no terrain costs less than 1) and the reverse moves for
 * bidirectional search.
 *
 * @param w - The world (its first start and its goals).
 * @param options - Connectivity and the heuristic's metric.
 * @returns The problem.
 * @throws MdmlError `value` when the world has no start or no goal.
 * @concept search-problems
 * @example
 * ```ts
 * import { gridProblem, searchMap } from 'mdml-engine/problems'
 * const p = gridProblem(searchMap('open'))
 * p.isGoal(p.initial) // => false
 * ```
 */
export function gridProblem(w: World, options: GridProblemOptions = {}): BidirectionalProblem<number, number> {
  const start = w.starts[0]
  const goal = w.goals[0]
  if (start === undefined || goal === undefined) throw new MdmlError('value', 'gridProblem: needs a start and a goal')
  const connectivity = options.connectivity ?? 4
  const move = { connectivity }
  const metric = options.metric ?? (connectivity === 8 ? 'octile' : 'manhattan')
  const h = heuristicTo(w, { metric })
  const goals = new Set(w.goals)
  const cost = (cell: number): number => terrainOf(w, cell).cost
  return {
    initial: start,
    goal,
    actions: (cell) => exits(w, cell, move).map(([d]) => d),
    result: (cell, d) => exits(w, cell, move).find(([e]) => e === d)?.[1] ?? cell,
    isGoal: (cell) => goals.has(cell),
    stepCost: (_cell, _d, next) => cost(next),
    heuristic: h,
    key: (cell) => cell,
    label: (cell) => (w.topology.kind === 'graph' ? String(cell) : `(${cellCoords(w, cell).join(', ')})`),
    predecessors: (cell) =>
      exits(w, cell, move).flatMap(([, from]) =>
        exits(w, from, move)
          .filter(([, to]) => to === cell)
          .map(([d]) => ({ state: from, action: d, cost: cost(cell) })),
      ),
  }
}
