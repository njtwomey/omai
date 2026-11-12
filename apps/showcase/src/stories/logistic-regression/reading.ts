/**
 * What the prose of 'Logistic regression' says: from the points alone (how many are yes and how many no), and from the result
 * (how unsure the score is, and what each weight does to the odds of yes). Pure functions, light enough for the page's bundle.
 */

import type { Interval, Result } from './compute.ts'
import type { Points } from './data.ts'

/** The numbers the prose quotes before anything has run. */
export interface DataReading {
  readonly count: number
  readonly yes: number
  readonly no: number
}

/**
 * The numbers about the points alone.
 *
 * @param points - The points.
 * @returns The count, and how many are yes and how many no.
 */
export const readData = (points: Points): DataReading => ({
  count: points.y.length,
  yes: points.y.filter((v) => v === 1).length,
  no: points.y.filter((v) => v === 0).length,
})

/** What one more unit of a measurement does to the odds of yes: the factor at the weight's mean and at the ends of its interval. */
export interface OddsReading {
  readonly weight: Interval
  readonly factor: number
  readonly low: number
  readonly high: number
  /** Whether the weight is positive (the measurement makes yes more likely) at its mean. */
  readonly raises: boolean
  /** Whether the whole interval lies on one side of zero. */
  readonly sure: boolean
}

/** The odds factor of a weight: the exponential of the weight, at its mean and at the ends of its 90% interval. */
export const oddsOf = (weight: Interval): OddsReading => ({
  weight,
  factor: Math.exp(weight.mean),
  low: Math.exp(weight.lo),
  high: Math.exp(weight.hi),
  raises: weight.mean > 0,
  sure: weight.lo > 0 || weight.hi < 0,
})

const within = (value: number, interval: Interval): boolean => interval.lo <= value && value <= interval.hi

/** The numbers the prose quotes after the run. */
export interface Reading {
  readonly first: OddsReading
  readonly second: OddsReading
  /** The standard deviation of the score in the middle of the points and the average of it at the four corners of the map. */
  readonly doubtMiddle: number
  readonly doubtCorners: number
  /** Whether the true offset and each true weight lie inside the 90% interval of the belief. */
  readonly inside: { readonly offset: boolean; readonly first: boolean; readonly second: boolean }
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The odds of each weight and how unsure the score is in the middle and at the corners of the map.
 */
export function readResult(result: Result): Reading {
  const last = result.xs.length - 1
  const mid = Math.floor(last / 2)
  const at = (row: number, col: number): number => result.doubt[row]?.[col] as number
  return {
    first: oddsOf(result.weights.first),
    second: oddsOf(result.weights.second),
    doubtMiddle: at(mid, mid),
    inside: {
      offset: within(result.truth.offset, result.weights.offset),
      first: within(result.truth.first, result.weights.first),
      second: within(result.truth.second, result.weights.second),
    },
    doubtCorners: (at(0, 0) + at(0, last) + at(last, 0) + at(last, last)) / 4,
  }
}
