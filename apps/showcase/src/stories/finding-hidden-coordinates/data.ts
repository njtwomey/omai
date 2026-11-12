/**
 * The data of 'Finding hidden coordinates': twenty points, each with five measurements that are smooth functions of one hidden
 * coordinate per point, plus noise. The coordinates and the functions are made up here, so the page can say how close the
 * recovered coordinates come to the true ones. The simulation is the library's (`simulateGplvm`), a pure function of the seed;
 * the page draws the data of the first seed before anything runs, and the run of that seed makes the same data.
 */

import { simulateGplvm } from 'omai-engine/probprog/models'
import { key } from 'omai-engine/random'
import { toArray } from 'omai-engine/tensor'

/** The numbers that describe the problem: points, measurements per point, features of the model, the noise and the method's settings. */
export const POINTS = 20
export const OUTPUTS = 5
export const FEATURES = 6
export const NOISE = 0.1
export const SWEEPS = 40
export const DAMPING = 0.5

/** The seed of the first run. Its data are drawn before play and it is a run in which the method finds the coordinate. */
export const FIRST_SEED = 2

/** The measurements of every point for a seed, one row per point. */
export function measurements(seed: number, points = POINTS, outputs = OUTPUTS): readonly (readonly number[])[] {
  const data = simulateGplvm(key(seed), { points, coords: 1, outputs, noise: NOISE })
  return toArray(data.y) as number[][]
}

/**
 * What a check of the same computation found when this page was written: the number of runs, on different points and different
 * random waves (seeds 1 to `runs`), and how many of them found a poor solution (a correlation with the true coordinate below 0.9).
 * The page quotes these numbers; they are a measurement made once, not something the run reads.
 */
export const CHECKED = { runs: 24, poor: 2 } as const
