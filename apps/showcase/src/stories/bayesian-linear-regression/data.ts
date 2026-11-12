/**
 * The twelve points of the story, drawn once with known truth (a line with intercept 1.2 and slope 0.7, noise 0.9) from a
 * fixed key, so the page can show them before anything runs and every run, whatever its seed, fits the same data. Simulating
 * twelve numbers is not the computation the reader asks for by pressing play; the inference is.
 */

import { foldIn, key, normal, uniform } from 'omai-engine/random'
import { add, mul, toFlat } from 'omai-engine/tensor'

/** The line the data were drawn from. */
export const TRUTH = { intercept: 1.2, slope: 0.7, noise: 0.9 } as const

/** How many points there are. */
export const ROWS = 12

/** The range of x the points fall in, and the range the figures draw (past the data on the right, to show the doubt grow). */
export const DATA_RANGE = [0.3, 4.3] as const
export const DRAW_RANGE = [-1, 8] as const

/** The points. */
export type Points = {
  readonly x: readonly number[]
  readonly y: readonly number[]
}

/**
 * The story's data.
 *
 * @returns The twelve points.
 */
export function simulate(): Points {
  const seed = key(2026)
  const x = add(mul(uniform(foldIn(seed, 0), [ROWS]), DATA_RANGE[1] - DATA_RANGE[0]), DATA_RANGE[0])
  const noise = mul(normal(foldIn(seed, 1), [ROWS]), TRUTH.noise)
  const y = add(add(mul(x, TRUTH.slope), TRUTH.intercept), noise)
  return { x: toFlat(x), y: toFlat(y) }
}
