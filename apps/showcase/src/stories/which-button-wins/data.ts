/**
 * The experiment of 'Which button wins?': two versions of a page, each shown to the same number of visitors, each visitor
 * clicking or not. The visitors of a seed are drawn here from a seeded key, with click rates the seed also picks and the
 * inference is never told, so the page can show the raw counts before anything runs (the first run uses {@link FIRST_SEED}, so
 * the counts it shows are the counts the run will read) and the story can reveal the true rates at the end. Simulating two
 * thousand coin tosses is not the computation the reader asks for by pressing play; the inference is.
 */

import { bernoulli, foldIn, key, uniform } from 'omai-engine/random'
import { add, item, mul, sum, type Tensor } from 'omai-engine/tensor'

/** The seed of the first run, which the page also uses to show the raw counts before play. */
export const FIRST_SEED = 1

/** Visitors shown each version. */
export const VISITORS = 1000

/** The numbers of visitors per version after which the chance that B is better is read off, growing to all of them. */
export const CHECKPOINTS: readonly number[] = [20, 40, 60, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

/** The probability at which most teams would stop and call a test. */
export const CALL = 0.95

/** What the experiment really was, which the inference never sees. */
export interface Truth {
  readonly rateA: number
  readonly rateB: number
}

/** The visitors of one experiment: a 1 for each who clicked, a 0 for each who did not, in the order they arrived. */
export interface Experiment {
  readonly a: Tensor
  readonly b: Tensor
  readonly truth: Truth
}

/**
 * The experiment of a seed: version A converts between 8% and 14% of visitors, and version B converts it times a factor
 * between 0.9 and 1.5 (a little worse to a lot better), each visitor an independent coin toss.
 *
 * @param seed - The seed.
 * @param visitors - How many visitors each version is shown.
 * @returns The visitors and the true rates.
 */
export function simulate(seed: number, visitors: number = VISITORS): Experiment {
  const base = key(seed)
  const rateA = item(add(mul(uniform(foldIn(base, 0), []), 0.06), 0.08))
  const rateB = rateA * item(add(mul(uniform(foldIn(base, 1), []), 0.6), 0.9))
  return {
    a: bernoulli(foldIn(base, 2), rateA, [visitors]),
    b: bernoulli(foldIn(base, 3), rateB, [visitors]),
    truth: { rateA, rateB },
  }
}

/** The raw tally of an experiment. */
export interface Tally {
  readonly visitors: number
  readonly clicksA: number
  readonly clicksB: number
}

/**
 * How many visitors clicked on each version.
 *
 * @param experiment - The experiment.
 * @returns The tally.
 */
export function tally(experiment: Experiment): Tally {
  return {
    visitors: experiment.a.shape[0] ?? 0,
    clicksA: item(sum(experiment.a)),
    clicksB: item(sum(experiment.b)),
  }
}
