/**
 * The data of 'Breaking a stick': 80 numbers on a line, drawn from four groups that the page does not name to the model. The
 * groups have centres at -5, -1.5, 2 and 5.5 and hold 40%, 30%, 20% and 10% of the numbers; each is a bell curve of width 0.8.
 * The numbers are drawn once from a fixed key, so every run, whatever its seed, fits the same data.
 */

import { foldIn, key, normal, uniform } from 'omai-engine/random'
import { add, cast, cumsum, greater, mul, reshape, sum, take, toFlat } from 'omai-engine/tensor'

/** The centres of the groups the numbers were drawn from. */
export const CENTRES = [-5, -1.5, 2, 5.5] as const

/** The share of the numbers each group holds. */
export const SHARES = [0.4, 0.3, 0.2, 0.1] as const

/** The width (standard deviation) of every group: the data are drawn with it and the model is told it. */
export const WIDTH = 0.8

/** How many numbers there are, and the seed they are drawn from. */
export const ROWS = 80
export const DATA_SEED = 3

/** The part of the line the figures show. */
export const RANGE = [-9, 9] as const

/**
 * The numbers everyone sees.
 *
 * @param rows - How many numbers.
 * @returns The numbers.
 */
export function drawNumbers(rows: number = ROWS): number[] {
  const seed = key(DATA_SEED)
  // the group of each number: how many of the running shares its uniform draw has passed
  const passed = greater(
    reshape(uniform(foldIn(seed, 0), [rows]), [rows, 1]),
    reshape(cumsum(SHARES), [1, SHARES.length]),
  )
  const group = cast(sum(passed, 1), 'int32')
  const values = add(take(CENTRES, group, 0), mul(normal(foldIn(seed, 1), [rows]), WIDTH))
  return toFlat(values)
}

/** The number of pieces the stick is cut off at. */
export const PIECES = 8
