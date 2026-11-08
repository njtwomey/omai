/**
 * Paths read off a world's values: the greedy path that climbs a value per cell (a value function, a distance to the
 * goal negated) from a start cell, for drawing the policy a set of values implies.
 */

import { type MoveOptions, neighbours, terrainOf, type World } from './world.ts'

/**
 * The greedy path over a value per cell: from `start`, step to the neighbouring cell with the highest value, and stop at
 * a goal, when no neighbour has a higher value than the cell it is on, or after `maxSteps`. Ties go to the first
 * neighbour in the world's own order. Cells of a hazardous terrain (holes, lava, cliffs) are never stepped onto, because
 * a value table usually holds a placeholder there (often 0, the highest of a table of negative values): give `skip` to
 * choose other cells, or `skip: () => false` to walk anywhere. Draw it with `toScene(w, { path })`.
 *
 * @param w - The world.
 * @param values - A value per cell (a state-value function, or the negated distance to the goal).
 * @param start - The cell to start from.
 * @param options - Connectivity and corner rule of the moves, `maxSteps` (default one move per cell) and `skip`, the
 *   cells that are never entered (default: cells of a hazardous terrain).
 * @returns The cells visited, from `start` on.
 * @concept structures
 * @index function The greedy path up a value per cell, for drawing a policy.
 * @example
 * ```ts
 * import { fromAscii, greedyPath } from 'omai-engine/worlds'
 * const w = fromAscii(['S..G'])
 * greedyPath(w, [0, 1, 2, 3], 0) // => [0, 1, 2, 3]
 * greedyPath(w, [0, 1, 0, 3], 0) // => [0, 1]
 * ```
 */
export function greedyPath(
  w: World,
  values: ArrayLike<number>,
  start: number,
  options: MoveOptions & { readonly maxSteps?: number; readonly skip?: (cell: number) => boolean } = {},
): number[] {
  const goals = new Set(w.goals)
  const skip = options.skip ?? ((cell: number): boolean => terrainOf(w, cell).hazard)
  const out = [start]
  let cell = start
  for (let step = 0; step < (options.maxSteps ?? w.n) && !goals.has(cell); step++) {
    let best = cell
    for (const next of neighbours(w, cell, options))
      if (!skip(next) && (values[next] ?? -Infinity) > (values[best] ?? -Infinity)) best = next
    if (best === cell) break
    out.push(best)
    cell = best
  }
  return out
}
