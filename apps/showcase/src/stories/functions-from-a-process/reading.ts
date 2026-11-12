/**
 * What the prose of the story says about a result: a few numbers read off it (how wide the band is before the points, at the
 * points and between them, how far the sampled functions are from the points), so the text always describes the picture beside
 * it, whatever the seed was. Pure functions of the result, light enough for the page's bundle.
 */

import type { Result } from './compute.ts'

/** The index of the grid point nearest `x`. */
export function nearest(grid: readonly number[], x: number): number {
  return grid.reduce((best, g, i) => (Math.abs(g - x) < Math.abs((grid[best] as number) - x) ? i : best), 0)
}

/** The numbers the prose quotes. */
export interface Reading {
  /** The width of the prior's 90% band (the same everywhere, up to rounding: the one at the middle of the grid). */
  readonly priorWidth: number
  /** The width of the posterior band, averaged over the observed points, and the widest of those widths. */
  readonly atPoints: number
  /** The widest the posterior band is between the first and the last observed point, and where. */
  readonly between: number
  readonly betweenAt: number
  /** The lowest and highest value of the sampled posterior functions at that place. */
  readonly betweenLow: number
  readonly betweenHigh: number
  /** The widest the posterior band is anywhere on the grid, and where (it may be beyond the outer points). */
  readonly widest: number
  readonly widestAt: number
  /** The largest distance, over the sampled posterior functions and the points, between a function and a point's reading. */
  readonly largestMiss: number
  /** The same, for the prior's functions: how far, at the largest, they are from the readings. */
  readonly priorMiss: number
}

const widthAt = (c: Result['prior'], i: number): number => (c.upper[i] as number) - (c.lower[i] as number)

const largestMiss = (result: Result, draws: readonly (readonly number[])[]): number =>
  Math.max(
    ...draws.flatMap((f) =>
      result.data.x.map((x, k) => Math.abs((f[nearest(result.grid, x)] as number) - (result.data.y[k] as number))),
    ),
  )

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readResult(result: Result): Reading {
  const { grid, data, posterior, prior } = result
  const at = data.x.map((x) => widthAt(posterior, nearest(grid, x)))
  const first = nearest(grid, Math.min(...data.x))
  const last = nearest(grid, Math.max(...data.x))
  const inside = grid.map((_, i) => i).filter((i) => i >= first && i <= last)
  const peak = inside.reduce((best, i) => (widthAt(posterior, i) > widthAt(posterior, best) ? i : best), first)
  const widest = grid.reduce((best, _, i) => (widthAt(posterior, i) > widthAt(posterior, best) ? i : best), 0)
  const spread = posterior.draws.map((f) => f[peak] as number)
  return {
    priorWidth: widthAt(prior, Math.floor(grid.length / 2)),
    atPoints: at.reduce((a, b) => a + b, 0) / at.length,
    between: widthAt(posterior, peak),
    betweenAt: grid[peak] as number,
    betweenLow: Math.min(...spread),
    betweenHigh: Math.max(...spread),
    widest: widthAt(posterior, widest),
    widestAt: grid[widest] as number,
    largestMiss: largestMiss(result, posterior.draws),
    priorMiss: largestMiss(result, prior.draws),
  }
}
