/**
 * The data of 'Logistic regression': forty points, each with two measurements and a mark, yes or no. The measurements are drawn
 * evenly over a square, and the mark is drawn from a chance that rises smoothly across a line (an offset of 0.4, a weight of 1.3
 * on the first measurement and a weight of -0.9 on the second). The points are drawn once from a fixed key, so the page shows
 * them before anything runs and every run, whatever its seed, fits the same data.
 */

import { foldIn, key, uniform } from 'omai-engine/random'
import { add, cast, less, mul, sigmoid, stack, sub, toArray, toFlat } from 'omai-engine/tensor'

/** The offset and the two weights the marks were drawn from. */
export const TRUTH = [0.4, 1.3, -0.9] as const

/** How many points there are, and the seed they are drawn from. */
export const ROWS = 40
export const DATA_SEED = 4

/** The measurements lie in a square of this half-width, and the figures show a little more. */
export const SPREAD = 3
export const LIMIT = 3.5

/** The points. */
export interface Points {
  /** The two measurements of each point. */
  readonly x: readonly (readonly [number, number])[]
  /** The mark of each point: 1 for yes, 0 for no. */
  readonly y: readonly number[]
}

/**
 * The story's points.
 *
 * @param rows - How many points.
 * @returns The measurements and the marks.
 */
export function drawPoints(rows: number = ROWS): Points {
  const seed = key(DATA_SEED)
  const first = sub(mul(uniform(foldIn(seed, 0), [rows]), 2 * SPREAD), SPREAD)
  const second = sub(mul(uniform(foldIn(seed, 1), [rows]), 2 * SPREAD), SPREAD)
  const score = add(add(TRUTH[0], mul(first, TRUTH[1])), mul(second, TRUTH[2]))
  const mark = cast(less(uniform(foldIn(seed, 2), [rows]), sigmoid(score)), 'float64')
  const x = toArray(stack([first, second], 1)) as [number, number][]
  return { x, y: toFlat(mark) }
}

/**
 * The points as the model reads them: a row of `[1, first, second]` for each point, so the offset is the weight of the 1.
 *
 * @param points - The points.
 * @returns The design matrix.
 */
export const designOf = (points: Points): number[][] => points.x.map((p) => [1, p[0], p[1]])

/** The levels of the chance whose lines are drawn on the map. */
export const LEVELS = [0.2, 0.4, 0.5, 0.6, 0.8] as const

/** The rows of the map (values of measurement 2) along which the gap between the 0.2 and 0.8 lines is measured: the middle and near the edge. */
export const GAP_ROWS = [0, 3] as const

/**
 * How the bell curve of expectation propagation compared with a sampler (NUTS) on `dataSets` data sets like the story's: the spreads
 * were narrower by up to `narrower` (a share of the sampler's), and the expected chance of yes at `places` places was within
 * `chanceWithin` of an average over the sampler's draws. A measurement, not computed here or covered by a test: made by the story's
 * author on 2026-11-12 (commit 19267ce0, 'Logistic regression') in a one-off script that was not kept. Measure it again before
 * changing the data, the prior or the model.
 */
export const NUTS_CHECK = { dataSets: 8, narrower: 0.13, places: 6, chanceWithin: 0.003 } as const
