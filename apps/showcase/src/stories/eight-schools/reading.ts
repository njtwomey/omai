/**
 * What the prose of the story says about a result: a few numbers read off it (how far each school moved, how hard each was
 * pulled, how much the eight estimates drew together, what the data say about the spread between schools), so the text always
 * describes the picture beside it, whatever the seed was. Pure functions of the result, light enough for the page's bundle.
 */

import type { Result, School } from './compute.ts'

const sd = (values: readonly number[]): number => {
  const m = values.reduce((a, b) => a + b, 0) / values.length
  return Math.sqrt(values.reduce((a, b) => a + (b - m) ** 2, 0) / (values.length - 1))
}

/** Where the histogram of tau stops: the draws beyond it are not drawn. */
export const SPREAD_LIMIT = 30

/** The numbers the prose quotes. */
export interface Reading {
  /** The school whose estimate moved most (in points) when pooled. */
  readonly mover: School
  readonly moved: number
  /** The school pulled hardest (the largest share of the way to the mean) and the one pulled least. */
  readonly pulledMost: School
  readonly pulledLeast: School
  /** The standard deviation of the eight raw estimates, and of the eight pooled ones. */
  readonly rawSpread: number
  readonly pooledSpread: number
  /** How many of the eight pooled intervals contain zero. */
  readonly containZero: number
  /** The share of tau's draws below 5, and how many are above where the histogram stops. */
  readonly smallTau: number
  readonly beyond: number
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readResult(result: Result): Reading {
  const { schools, tauDraws } = result
  const byMove = [...schools].sort(
    (a, b) => Math.abs(b.estimate - b.pooled.mean) - Math.abs(a.estimate - a.pooled.mean),
  )
  const byPull = [...schools].sort((a, b) => b.pull - a.pull)
  const mover = byMove[0] as School
  return {
    mover,
    moved: Math.abs(mover.estimate - mover.pooled.mean),
    pulledMost: byPull[0] as School,
    pulledLeast: byPull.at(-1) as School,
    rawSpread: sd(schools.map((s) => s.estimate)),
    pooledSpread: sd(schools.map((s) => s.pooled.mean)),
    containZero: schools.filter((s) => s.pooled.lo <= 0 && 0 <= s.pooled.hi).length,
    smallTau: tauDraws.filter((t) => t < 5).length / tauDraws.length,
    beyond: tauDraws.filter((t) => t > SPREAD_LIMIT).length,
  }
}
