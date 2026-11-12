/**
 * What the prose of the story says about a result: how wide the band of networks is where the points are, in the gap between the
 * clumps and beyond the two ends, read off the result so the text always describes the picture beside it, whatever the seed was.
 * Pure functions of the result, light enough for the page's bundle.
 */

import type { Result } from './compute.ts'
import { CLUMPS } from './data.ts'

/** The width of the band at the grid point nearest `x`. */
export function bandAt(result: Result, x: number): number {
  let best = 0
  result.grid.forEach((g, i) => {
    if (Math.abs(g - x) < Math.abs((result.grid[best] as number) - x)) best = i
  })
  return (result.upper[best] as number) - (result.lower[best] as number)
}

/** The numbers the prose quotes. */
export interface Reading {
  /** The average width of the band over the stretches where the points are (a little inside each clump). */
  readonly dataWidth: number
  /** The width at the middle of the gap, x = 0. */
  readonly gapWidth: number
  /** The widths at x = -3 and x = 3, beyond the last point on each side. */
  readonly leftWidth: number
  readonly rightWidth: number
  /** How many times wider those are than over the points. */
  readonly gapRatio: number
  readonly leftRatio: number
  readonly rightRatio: number
  /** Whether the band is wider at the gap and at both ends than over the points (the story's claim). */
  readonly widerAwayFromData: boolean
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readResult(result: Result): Reading {
  const inClumps = result.grid
    .map((x, i) => ({ x, i }))
    .filter(({ x }) => CLUMPS.some(([a, b]) => x >= a + 0.1 && x <= b - 0.1))
  const widths = inClumps.map(({ i }) => (result.upper[i] as number) - (result.lower[i] as number))
  const dataWidth = widths.reduce((a, b) => a + b, 0) / Math.max(1, widths.length)
  const gapWidth = bandAt(result, 0)
  const leftWidth = bandAt(result, -3)
  const rightWidth = bandAt(result, 3)
  return {
    dataWidth,
    gapWidth,
    leftWidth,
    rightWidth,
    gapRatio: gapWidth / dataWidth,
    leftRatio: leftWidth / dataWidth,
    rightRatio: rightWidth / dataWidth,
    widerAwayFromData: gapWidth > dataWidth && leftWidth > dataWidth && rightWidth > dataWidth,
  }
}
