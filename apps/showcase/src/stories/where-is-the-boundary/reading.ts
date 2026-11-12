/**
 * What the prose of 'Where is the boundary?' says: from the points alone (how many in each class), and from the result (how many
 * points land on their own side, how much of the plane the model is unsure about, what the probability is far from every
 * point, and how the doubt about the score compares near the points and far from them). Pure functions, light enough for the
 * page's bundle.
 */

import type { Result } from './compute.ts'
import type { Points } from './data.ts'

/** A cell is called unsure when its probability is between these. */
export const UNSURE = [0.25, 0.75] as const

/** A cell is near the data when a point lies within this distance, and far from it when none lies within the second. */
export const NEAR = 0.5
export const FAR = 1.5

/** The numbers the prose quotes before anything has run. */
export interface DataReading {
  readonly count: number
  readonly first: number
  readonly second: number
}

/**
 * The numbers about the points alone.
 *
 * @param points - The points.
 * @returns The count and the size of each class.
 */
export const readData = (points: Points): DataReading => ({
  count: points.y.length,
  first: points.y.filter((c) => c === 0).length,
  second: points.y.filter((c) => c === 1).length,
})

/** The numbers the prose quotes after the run. */
export interface Reading {
  /** How many points land on the side of their own class. */
  readonly right: number
  readonly count: number
  /** The share of the map's cells that are unsure (a probability between {@link UNSURE}). */
  readonly unsureShare: number
  /** How many of the unsure cells are near the data and how many are far from it. */
  readonly unsureNear: number
  readonly unsureFar: number
  /** The probability at the four corners of the map, which are far from every point. */
  readonly corners: readonly number[]
  /** The average doubt about the score in cells near the data and in cells far from it. */
  readonly doubtNear: number
  readonly doubtFar: number
  /** The length scale that was fitted. */
  readonly lengthScale: number
  /** The lowest and the highest probability on the map. */
  readonly lowest: number
  readonly highest: number
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readResult(result: Result): Reading {
  const cells = result.ys.flatMap((y, row) =>
    result.xs.map((x, col) => {
      const distance = Math.min(...result.points.x.map((p) => Math.hypot(p[0] - x, p[1] - y)))
      return {
        p: result.probability[row]?.[col] as number,
        doubt: result.doubt[row]?.[col] as number,
        distance,
      }
    }),
  )
  const near = cells.filter((c) => c.distance < NEAR)
  const far = cells.filter((c) => c.distance > FAR)
  const mean = (v: readonly number[]): number => (v.length === 0 ? Number.NaN : v.reduce((a, b) => a + b, 0) / v.length)
  const unsure = cells.filter((c) => c.p > UNSURE[0] && c.p < UNSURE[1])
  const lastRow = result.probability.length - 1
  const lastCol = (result.probability[0]?.length ?? 1) - 1
  return {
    right: Math.round(result.accuracy * result.points.y.length),
    count: result.points.y.length,
    unsureShare: unsure.length / cells.length,
    unsureNear: unsure.filter((c) => c.distance < NEAR).length,
    unsureFar: unsure.filter((c) => c.distance > FAR).length,
    corners: [
      result.probability[0]?.[0] as number,
      result.probability[0]?.[lastCol] as number,
      result.probability[lastRow]?.[0] as number,
      result.probability[lastRow]?.[lastCol] as number,
    ],
    doubtNear: mean(near.map((c) => c.doubt)),
    doubtFar: mean(far.map((c) => c.doubt)),
    lengthScale: result.kernel.lengthScale,
    lowest: Math.min(...cells.map((c) => c.p)),
    highest: Math.max(...cells.map((c) => c.p)),
  }
}
