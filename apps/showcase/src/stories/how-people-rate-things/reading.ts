/**
 * What the prose of 'How people rate things' says: from the ratings alone (before anything runs) how often each score was given and
 * how much the raters and the items differ in their average score, and, from the result, how close the fitted cut points,
 * qualities and generosities are to the ones the ratings were made from, and how the spread of the cut points fell as ratings
 * arrived. Pure functions, light enough for the page's bundle.
 */

import type { Belief, Result } from './compute.ts'
import { CATEGORIES, counts, type Ratings } from './data.ts'
import { REFERENCE } from './reference.ts'

/** The z value of a 90 percent interval of a bell curve. */
export const Z90 = 1.645

const sum = (v: readonly number[]): number => v.reduce((s, x) => s + x, 0)
const average = (v: readonly number[]): number => sum(v) / v.length

/** The correlation of two lists of the same length. */
export function correlation(a: readonly number[], b: readonly number[]): number {
  const [ma, mb] = [average(a), average(b)]
  const cross = sum(a.map((x, i) => (x - ma) * ((b[i] as number) - mb)))
  return cross / Math.sqrt(sum(a.map((x) => (x - ma) ** 2)) * sum(b.map((x) => (x - mb) ** 2)))
}

/** The average rating, on the scale of 1 to 5, of each group (a rater or an item) that has at least one rating. */
function averageBy(group: readonly number[], score: readonly number[], size: number): number[] {
  return Array.from({ length: size }, (_, g) => score.filter((_, i) => group[i] === g).map((s) => s + 1))
    .filter((s) => s.length > 0)
    .map(average)
}

/** What the page says about the ratings before the model is fitted. */
export interface DataReading {
  readonly total: number
  /** How many ratings of each score, from 1 to 5. */
  readonly counts: readonly number[]
  /** The average rating of the most generous and of the least generous rater. */
  readonly raterHigh: number
  readonly raterLow: number
  /** The average rating of the best-rated and of the worst-rated item. */
  readonly itemHigh: number
  readonly itemLow: number
  /** The fewest and the most ratings an item received. */
  readonly itemMin: number
  readonly itemMax: number
}

/**
 * The numbers the prose quotes before the run.
 *
 * @param ratings - The ratings.
 * @param raters - The number of raters.
 * @param items - The number of items.
 * @returns The reading.
 */
export function readData(ratings: Ratings, raters: number, items: number): DataReading {
  const byRater = averageBy(ratings.rater, ratings.score, raters)
  const byItem = averageBy(ratings.item, ratings.score, items)
  const perItem = Array.from({ length: items }, (_, i) => ratings.item.filter((x) => x === i).length)
  return {
    total: ratings.score.length,
    counts: counts(ratings.score),
    raterHigh: Math.max(...byRater),
    raterLow: Math.min(...byRater),
    itemHigh: Math.max(...byItem),
    itemLow: Math.min(...byItem),
    itemMin: Math.min(...perItem),
    itemMax: Math.max(...perItem),
  }
}

/** One cut point: what the fit believes and what the ratings were made with. */
export interface CutRow {
  /** The two scores the cut point lies between, as the page names it ('1|2'). */
  readonly name: string
  readonly mean: number
  readonly sd: number
  readonly lo: number
  readonly hi: number
  readonly truth: number
  /** Whether the true cut point lies inside the 90 percent interval. */
  readonly inside: boolean
  /** The spread the sampler found for it, and the fit's spread as a fraction of that. */
  readonly referenceSd: number
  readonly ratio: number
}

/** The numbers the prose quotes after the run. */
export interface Reading {
  readonly cuts: readonly CutRow[]
  /** How many of the four true cut points lie inside their 90 percent interval. */
  readonly cutsInside: number
  /** The average spread of the four cut points after `k` ratings, for the stops the run made. */
  readonly spreadAfter: (k: number) => number | undefined
  /** The number of ratings in the run. */
  readonly total: number
  /** The correlation of the fitted qualities with the true ones, and the root mean square gap. */
  readonly qualityCorrelation: number
  readonly qualityError: number
  /** How many of the true qualities lie inside their 90 percent interval, out of how many. */
  readonly qualityInside: number
  readonly items: number
  /** The correlation of the fitted generosities with the true ones. */
  readonly generosityCorrelation: number
  /** The counts the ratings have and the counts the fitted model expects, and the largest gap between them. */
  readonly observed: readonly number[]
  readonly expected: readonly number[]
  readonly largestGap: number
  /** The average of the cut points' ratios to the sampler's spreads. */
  readonly meanRatio: number
  /** The largest gap between a cut point's mean and the sampler's mean for it. */
  readonly maxMeanGap: number
  /** The average ratio of the items' spreads to the sampler's spreads for them. */
  readonly qualityRatio: number
}

const intervalOf = (mean: number, sd: number): readonly [number, number] => [mean - Z90 * sd, mean + Z90 * sd]
const spreadOf = (b: Belief): number => average(b.sd)

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readResult(result: Result): Reading {
  const { ratings, final } = result
  const cuts = final.cuts.mean.map((mean, k): CutRow => {
    const sd = final.cuts.sd[k] as number
    const [lo, hi] = intervalOf(mean, sd)
    const truth = ratings.cuts[k] as number
    const referenceSd = REFERENCE.cutSd[k] as number
    return {
      name: `${k + 1}|${k + 2}`,
      mean,
      sd,
      lo,
      hi,
      truth,
      inside: truth >= lo && truth <= hi,
      referenceSd,
      ratio: sd / referenceSd,
    }
  })
  const quality = final.quality
  const qualityInside = quality.mean.filter((m, i) => {
    const [lo, hi] = intervalOf(m, quality.sd[i] as number)
    const truth = ratings.quality[i] as number
    return truth >= lo && truth <= hi
  }).length
  const observed = counts(ratings.score)
  return {
    cuts,
    cutsInside: cuts.filter((c) => c.inside).length,
    spreadAfter: (k) => {
      const at = result.history.find((h) => h.after === k)
      return at === undefined ? undefined : spreadOf(at.cuts)
    },
    total: ratings.score.length,
    qualityCorrelation: correlation(quality.mean, ratings.quality),
    qualityError: Math.sqrt(average(quality.mean.map((m, i) => (m - (ratings.quality[i] as number)) ** 2))),
    qualityInside,
    items: quality.mean.length,
    generosityCorrelation: correlation(final.generosity.mean, ratings.generosity),
    observed,
    expected: final.expected,
    largestGap: Math.max(...observed.map((o, k) => Math.abs(o - (final.expected[k] as number)))),
    meanRatio: average(cuts.map((c) => c.ratio)),
    maxMeanGap: Math.max(...cuts.map((c, k) => Math.abs(c.mean - (REFERENCE.cutMean[k] as number)))),
    qualityRatio: average(quality.sd.map((sd, i) => sd / (REFERENCE.qualitySd[i] as number))),
  }
}

/** The number of rating values, for the figure's category names. */
export const SCORE_NAMES: readonly string[] = Array.from({ length: CATEGORIES }, (_, k) => String(k + 1))
