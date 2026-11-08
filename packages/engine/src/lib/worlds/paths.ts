/**
 * Paths read off a world: the greedy path that climbs a value per cell (a value function, a distance to the goal
 * negated) from a start cell, and the path a policy takes from a start cell, for drawing what a set of values or a
 * policy implies.
 */

import { destination, type MoveOptions, neighbours, terrainOf, type World } from './world.ts'

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

/**
 * The path a policy takes from a start cell: at each cell, the policy's action (the direction to move in, as `DIRECTIONS`),
 * and the cell it leads to. The walk stops at a goal, when the move is blocked (a wall, the edge of the map), on entering
 * a hazardous cell (which is the last cell of the path), when it comes back to a cell it has been on (the policy goes
 * round in a loop, as an untrained one does), or after `maxSteps`. This is a rollout of the policy's own choices, so unlike
 * {@link greedyPath} it shows a policy that has not learnt yet as it is: stuck where it is stuck.
 *
 * @param w - The world.
 * @param policy - An action per cell, or a probability per action for each cell (the most probable action is taken, the
 *   first on a tie).
 * @param start - The cell to start from.
 * @param options - Connectivity and corner rule of the moves, and `maxSteps` (default one move per cell).
 * @returns The cells visited, from `start` on.
 * @concept structures
 * @index function The path a policy takes from a cell, for drawing a rollout.
 * @example
 * ```ts
 * import { fromAscii, policyPath } from 'omai-engine/worlds'
 * const w = fromAscii(['S..G'])
 * policyPath(w, [1, 1, 1, 1], 0) // => [0, 1, 2, 3]
 * policyPath(w, [1, 3, 1, 1], 0) // => [0, 1]
 * ```
 */
export function policyPath(
  w: World,
  policy: ArrayLike<number> | readonly ArrayLike<number>[],
  start: number,
  options: MoveOptions & { readonly maxSteps?: number } = {},
): number[] {
  const goals = new Set(w.goals)
  const actionAt = (cell: number): number => {
    const row = (policy as readonly (number | ArrayLike<number>)[])[cell]
    if (typeof row !== 'object' || row === null) return typeof row === 'number' ? row : -1
    let best = -1
    let top = Number.NEGATIVE_INFINITY
    for (let a = 0; a < row.length; a++) {
      const p = row[a] ?? Number.NEGATIVE_INFINITY
      if (p > top) {
        top = p
        best = a
      }
    }
    return best
  }
  const out = [start]
  const seen = new Set<number>([start])
  let cell = start
  for (let step = 0; step < (options.maxSteps ?? w.n) && !goals.has(cell); step++) {
    const action = actionAt(cell)
    const next = action < 0 ? -1 : destination(w, cell, action, options)
    if (next < 0 || seen.has(next)) break
    out.push(next)
    if (terrainOf(w, next).hazard) break
    seen.add(next)
    cell = next
  }
  return out
}
