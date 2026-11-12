/**
 * What the prose of the story says about a result: a few numbers read off it (how wide the band is where the data are and where
 * they are not, whether the true values fall inside their intervals), so the text always describes the picture beside it, whatever
 * the seed was. Pure functions of the result and the data, light enough for the page's bundle.
 */

import type { Result } from './compute.ts'
import { type Points, TRUTH } from './data.ts'

/** The width of the band at the grid point nearest `x`, and the grid point it is at. */
export function bandAt(result: Result, x: number): { readonly at: number; readonly width: number } {
  let best = 0
  result.grid.forEach((g, i) => {
    if (Math.abs(g - x) < Math.abs((result.grid[best] as number) - x)) best = i
  })
  return { at: result.grid[best] as number, width: (result.upper[best] as number) - (result.lower[best] as number) }
}

/** The numbers the prose quotes. */
export interface Reading {
  /** The middle of the data in x, and the width of the band there. */
  readonly centre: number
  readonly centreWidth: number
  /** The far right of the figure, where there are no data, and the width of the band there. */
  readonly edge: number
  readonly edgeWidth: number
  /** How many times wider the band is at the edge than in the middle. */
  readonly ratio: number
  readonly slope: Result['summary']['slope']
  readonly intercept: Result['summary']['intercept']
  readonly noise: Result['summary']['noise']
  readonly slopeInside: boolean
  readonly interceptInside: boolean
  readonly noiseInside: boolean
}

const inside = (v: number, i: { readonly lo: number; readonly hi: number }): boolean => i.lo <= v && v <= i.hi

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @param data - The points it was fitted to.
 * @returns The reading.
 */
export function readResult(result: Result, data: Points): Reading {
  const centre = data.x.reduce((a, b) => a + b, 0) / data.x.length
  const middle = bandAt(result, centre)
  const edge = bandAt(result, result.grid.at(-1) as number)
  return {
    centre: middle.at,
    centreWidth: middle.width,
    edge: edge.at,
    edgeWidth: edge.width,
    ratio: edge.width / middle.width,
    slope: result.summary.slope,
    intercept: result.summary.intercept,
    noise: result.summary.noise,
    slopeInside: inside(TRUTH.slope, result.summary.slope),
    interceptInside: inside(TRUTH.intercept, result.summary.intercept),
    noiseInside: inside(TRUTH.noise, result.summary.noise),
  }
}
