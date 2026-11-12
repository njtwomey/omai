/**
 * The sixty points of the story, drawn once from a fixed key so that the page can show them before anything runs and every
 * run, whatever its seed, uses the same data. The true curve is a staircase with two steps, at x = 1.4 (from 0 to 1.1) and
 * x = 2.8 (from 1.1 to 2), and the noise is 0.5. The data are not the computation the reader asks for by pressing play; the
 * trees are.
 */

import { foldIn, key, normal, uniform } from 'omai-engine/random'
import { add, mul, toFlat } from 'omai-engine/tensor'

/** What the data were drawn from. */
export const TRUTH = { cuts: [1.4, 2.8], levels: [0, 1.1, 2], noise: 0.5 } as const

/** How close a cut has to be to a cut of the greedy tree to count as the same cut. */
export const NEAR = 0.3

/** How many points there are, and the range of x they fall in. */
export const ROWS = 60
export const X_RANGE = [0, 4] as const

/** The points. */
export type Points = {
  readonly x: readonly number[]
  readonly y: readonly number[]
}

/** The true value of the curve at `x`. */
export const truthAt = (x: number): number =>
  x < TRUTH.cuts[0] ? TRUTH.levels[0] : x < TRUTH.cuts[1] ? TRUTH.levels[1] : TRUTH.levels[2]

/**
 * The story's data.
 *
 * @returns The sixty points.
 */
export function simulate(): Points {
  const seed = key(2026)
  const x = mul(uniform(foldIn(seed, 0), [ROWS]), X_RANGE[1])
  const clean = toFlat(x).map(truthAt)
  const y = add(clean, mul(normal(foldIn(seed, 1), [ROWS]), TRUTH.noise))
  return { x: toFlat(x), y: toFlat(y) }
}
