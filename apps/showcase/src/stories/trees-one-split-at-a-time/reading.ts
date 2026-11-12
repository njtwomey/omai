/**
 * What the prose of the story says about a result: a few numbers read off it (where the first cut is, how sure the posterior
 * is of a second one, how far the cuts could move), so the text always describes the picture beside it, whatever the seed
 * was. Pure functions of the result, light enough for the page's bundle.
 */

import type { Result } from './compute.ts'
import { NEAR } from './data.ts'

/** The range of candidate cuts with at least `floor` chance of being in the tree, between `lo` and `hi`. */
export function spanOf(
  result: Result,
  lo: number,
  hi: number,
  floor: number,
): { from: number; to: number } | undefined {
  const inside = result.posterior.thresholds.filter(
    (t, i) => t >= lo && t <= hi && (result.posterior.cutChance[i] as number) >= floor,
  )
  return inside.length === 0 ? undefined : { from: Math.min(...inside), to: Math.max(...inside) }
}

/** The numbers the prose quotes. */
export interface Reading {
  /** The first and the second cut of the greedy tree (the second is 0 when the growth stopped after one). */
  readonly firstCut: number
  readonly secondCut: number
  readonly cuts: number
  /** The leaves of the final greedy tree: the fewest points in one, and its step size. */
  readonly smallLeaf: { readonly n: number; readonly stepSize: number }
  readonly largeLeaf: { readonly n: number; readonly stepSize: number }
  /** The chance that the posterior puts on no cut, one, two and three cuts: they add to one. */
  readonly none: number
  readonly one: number
  readonly two: number
  readonly three: number
  /** The chance that some cut of the tree lies within `NEAR` of the greedy tree's first and second cut. */
  readonly nearFirst: number
  readonly nearSecond: number
  /** The range of positions where the second cut has at least 5% chance, from the first cut onward. */
  readonly secondSpan: { readonly from: number; readonly to: number } | undefined
  /** The posterior's most probable tree: its cuts and probability. */
  readonly top: { readonly cuts: readonly number[]; readonly probability: number }
  /** The widest the average over trees is from the greedy curve, and the widest band of the average. */
  readonly widestDifference: number
  /** Where the average over trees differs most from the greedy tree. */
  readonly widestAt: number
  readonly widestBand: number
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readResult(result: Result): Reading {
  const last = result.greedy.stages.at(-1) as Result['greedy']['stages'][number]
  const byPosition = [...result.greedy.cuts].sort((a, b) => a.threshold - b.threshold)
  const cuts = byPosition.map((c) => c.threshold)
  const leaves = [...last.leaves].sort((a, b) => a.n - b.n)
  const gaps = result.posterior.mean.map((m, i) => Math.abs(m - (last.mean[i] as number)))
  const widest = { index: gaps.indexOf(Math.max(...gaps)) }
  const first = cuts[0] ?? 0
  const second = cuts[1] ?? 0
  const small = leaves[0] as Result['greedy']['stages'][number]['leaves'][number]
  const large = leaves[leaves.length - 1] as typeof small
  return {
    firstCut: first,
    secondCut: second,
    cuts: cuts.length,
    smallLeaf: { n: small.n, stepSize: small.stepSize },
    largeLeaf: { n: large.n, stepSize: large.stepSize },
    none: result.posterior.splits[0] as number,
    one: result.posterior.splits[1] as number,
    two: result.posterior.splits[2] as number,
    three: result.posterior.splits[3] as number,
    nearFirst: byPosition[0]?.chanceNear ?? 0,
    nearSecond: byPosition[1]?.chanceNear ?? 0,
    secondSpan: spanOf(result, first + NEAR, Number.POSITIVE_INFINITY, 0.05),
    top: result.posterior.top[0] as Result['posterior']['top'][number],
    widestAt: result.grid[widest.index] as number,
    widestDifference: Math.max(...result.posterior.mean.map((m, i) => Math.abs(m - (last.mean[i] as number)))),
    widestBand: Math.max(...result.posterior.upper.map((u, i) => u - (result.posterior.lower[i] as number))),
  }
}
