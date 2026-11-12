/**
 * What the prose of the coin story says about a result: the belief at each moment, how wide it is, what it says about
 * fairness and whether the true bias fell inside it, so the text always describes the picture beside it whatever coin the seed
 * made. Pure functions of the result, light enough for the page's bundle.
 */

import type { Result, Stage } from './compute.ts'

/** The width of a stage's 90% interval. */
export const widthOf = (s: Stage): number => s.hi - s.lo

/** What the last stage says about fairness: the data rule it out, leave it open, or make it a fair bet. */
export type Verdict = 'unlikely' | 'open' | 'plausible'

/** The numbers the prose quotes. */
export interface Reading {
  readonly start: Stage
  readonly one: Stage
  readonly three: Stage
  readonly ten: Stage
  readonly thirty: Stage
  readonly last: Stage
  /** How many times narrower the interval is after the last stage than after ten flips. */
  readonly narrowing: number
  /** Whether the last flips leaned to heads or to tails (the posterior mean against one half). */
  readonly leans: 'heads' | 'tails'
  readonly verdict: Verdict
  /** Whether the bias the coin really had lies inside the final 90% interval. */
  readonly inside: boolean
  /** Whether the single first flip was heads. */
  readonly firstWasHeads: boolean
}

function at(result: Result, flips: number): Stage {
  const stage = result.stages.find((s) => s.flips === flips)
  if (stage === undefined) throw new Error(`the result has no stage after ${flips} flips`)
  return stage
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readCoin(result: Result): Reading {
  const last = result.stages.at(-1) as Stage
  const ten = at(result, 10)
  return {
    start: at(result, 0),
    one: at(result, 1),
    three: at(result, 3),
    ten,
    thirty: at(result, 30),
    last,
    narrowing: widthOf(ten) / widthOf(last),
    leans: last.mean >= 0.5 ? 'heads' : 'tails',
    verdict: last.fair < 0.05 ? 'unlikely' : last.fair < 0.5 ? 'open' : 'plausible',
    inside: last.lo <= result.truth && result.truth <= last.hi,
    firstWasHeads: result.flips[0] === 1,
  }
}
