/**
 * The sixteen points of the story: the library's Bayesian-neural-network benchmark data (a noisy sin(1.5 x) sampled in two clumps,
 * on [-2, -0.5] and [0.5, 2], with nothing in between), written out here so the page can draw them before anything runs, to three
 * decimals. The computation uses these same numbers, and the test checks that they are the library's.
 */

/** The points. */
export type Points = {
  readonly x: readonly number[]
  readonly y: readonly number[]
}

/** The two clumps in x, and the range the figures draw (well past the data on both sides, to show the doubt grow). */
export const CLUMPS = [
  [-2, -0.5],
  [0.5, 2],
] as const
export const DRAW_RANGE = [-3.5, 3.5] as const

/** The sixteen points. */
export const POINTS: Points = {
  x: [
    -1.845, -1.572, -1.149, -1.127, -1.108, -1.041, -0.948, -0.531, 0.569, 0.597, 0.834, 1.017, 1.178, 1.639, 1.769,
    1.963,
  ],
  y: [
    -0.481, -0.776, -0.979, -0.823, -1.083, -1.218, -0.841, -0.407, 0.853, 0.73, 0.657, 1.168, 1.176, 0.561, 0.669,
    0.334,
  ],
}
