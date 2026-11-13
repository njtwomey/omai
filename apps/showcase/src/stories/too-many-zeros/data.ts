/**
 * The counts of 'Counts with too many zeros': the library's `zeroInflatedCounts` generator, which draws a covariate for each
 * row, makes a count from a Poisson rate that grows with the covariate, and sets some counts to zero through a hidden switch. The
 * generator keeps what the counts do not show: which rows had the switch on, and the rate of every row. Those are the truth, which
 * the inference is never given. The page shows the counts before anything runs (the first run uses {@link FIRST_SEED}, so they are
 * the counts the run will read); it does not show the switches until the run has happened. Drawing one hundred and twenty counts is
 * not the computation the reader asks for by pressing play: the inference is.
 */

import { generate } from 'omai-engine/data'
import { zeroInflatedCounts } from 'omai-engine/datasets'
import { key } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'

/** The seed of the first run, which the page also uses to show the counts before play. */
export const FIRST_SEED = 1

/** How many counts there are. */
export const ROWS = 100

/** The numbers of counts the model is run on: the first few, then more, ending with all of them. */
export const CHECKPOINTS: readonly number[] = [10, 25, 50, ROWS]

/** What the data were made from. The model is not told any of it. */
export const TRUTH = { inflation: 0.25, intercept: 0.3, slope: 0.9 } as const

/** The range of the covariate the generator draws from. */
export const X_RANGE = [-1.5, 1.5] as const

/** The counts of one seed, with the truth that goes with them. */
export interface Series {
  /** The covariate of each count. */
  readonly x: readonly number[]
  /** The counts. */
  readonly y: readonly number[]
  /** For each count, 1 if its switch was on (a structural zero) and 0 if it was not. The inference never sees this. */
  readonly structural: readonly number[]
}

/**
 * The counts of a seed.
 *
 * @param seed - Draws the covariates, the switches and the counts.
 * @param rows - How many counts (the story's {@link ROWS}; fewer for a quick test).
 * @returns The covariates, the counts and the true switches.
 */
export function series(seed: number, rows: number = ROWS): Series {
  const { data, truth } = generate(zeroInflatedCounts, { key: key(seed), knobs: { n: rows, ...TRUTH } })
  return {
    x: toFlat(data.x),
    y: toFlat(data.y),
    structural: toFlat(truth.latents?.['structural'] as never),
  }
}
