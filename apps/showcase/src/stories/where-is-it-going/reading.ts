/**
 * What the prose of 'Where is it going?' says: from the readings alone (how many, their range), and from the result (how close
 * the smoothed path is to the true positions compared with the readings, how wide the 90 percent band is in the middle, at the
 * ends and at the end of the forecast, and how often the true positions lie inside the band). Pure functions, light enough for
 * the page's bundle.
 */

import type { Belief, Result } from './compute.ts'
import { DT } from './data.ts'

/** The time of a step. */
export const timeOf = (step: number): number => step * DT

/** The numbers the prose quotes before anything has run. */
export interface DataReading {
  readonly count: number
  readonly first: number
  readonly last: number
  readonly lowest: number
  readonly highest: number
}

/**
 * The numbers about the readings alone.
 *
 * @param readings - The readings.
 * @returns The reading.
 */
export const readData = (readings: readonly number[]): DataReading => ({
  count: readings.length,
  first: readings[0] as number,
  last: readings.at(-1) as number,
  lowest: Math.min(...readings),
  highest: Math.max(...readings),
})

/** The half-width of a belief's 90 percent interval. */
export const halfWidth = (b: Belief): number => (b.hi - b.lo) / 2

/** The numbers the prose quotes after the run. */
export interface Reading {
  /** The average distance from the true position of a reading, and of the smoothed path, over the steps with a reading. */
  readonly readingError: number
  readonly pathError: number
  /** How many steps with a reading have their true position inside the 90 percent band, and how many steps that is out of. */
  readonly inside: number
  readonly steps: number
  /** The half-width of the band at the first step, in the middle of the readings, and at the last reading. */
  readonly firstHalf: number
  readonly middleHalf: number
  readonly lastHalf: number
  /** The belief about the position at the end of the forecast, the true position there and the number of steps ahead. */
  readonly end: Belief
  readonly endTruth: number
  readonly ahead: number
  /** How many of the steps ahead have their true position inside the band. */
  readonly aheadInside: number
  /** The half-width of the band at the end of the forecast. */
  readonly endHalf: number
  /** The time at the last reading and at the end of the forecast. */
  readonly lastTime: number
  readonly endTime: number
  /** Whether the true velocity at the last reading lies inside the velocity's 90 percent interval. */
  readonly velocityInside: boolean
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readResult(result: Result): Reading {
  const n = result.readings.length
  const total = result.beliefs.length
  const mean = (values: readonly number[]): number => values.reduce((a, b) => a + b, 0) / values.length
  const within = (i: number): boolean => {
    const b = result.beliefs[i] as Belief
    const t = result.truth[i] as number
    return t >= b.lo && t <= b.hi
  }
  const range = (from: number, to: number): number[] => result.beliefs.slice(from, to).map((_, k) => from + k)
  const end = result.beliefs[total - 1] as Belief
  return {
    readingError: mean(result.readings.map((y, i) => Math.abs(y - (result.truth[i] as number)))),
    pathError: mean(range(0, n).map((i) => Math.abs((result.beliefs[i] as Belief).mean - (result.truth[i] as number)))),
    inside: range(0, n).filter(within).length,
    steps: n,
    firstHalf: halfWidth(result.beliefs[0] as Belief),
    middleHalf: halfWidth(result.beliefs[Math.floor(n / 2)] as Belief),
    lastHalf: halfWidth(result.beliefs[n - 1] as Belief),
    end,
    endTruth: result.truth[total - 1] as number,
    ahead: total - n,
    aheadInside: range(n, total).filter(within).length,
    endHalf: halfWidth(end),
    lastTime: timeOf(n - 1),
    endTime: timeOf(total - 1),
    velocityInside:
      result.velocity.truthLast >= result.velocity.last.lo && result.velocity.truthLast <= result.velocity.last.hi,
  }
}
