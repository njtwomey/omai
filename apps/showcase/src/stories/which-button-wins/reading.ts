/**
 * What the prose of the button story says about a result: how the two raw rates compare, what the probability that B is better
 * is in plain words, whether the difference is clearly above zero, and what the line of 'how sure are we so far' did on its way
 * (how wildly it moved early, when it first crossed the line most teams stop at, and whether it stayed there), so the text
 * always describes the picture beside it whatever visitors the seed brought. Pure functions of the result, light enough for the
 * page's bundle.
 */

import type { Result } from './compute.ts'
import { CALL } from './data.ts'

/** A probability as a percentage for a sentence: whole numbers, with 'over 99%' and 'under 1%' at the ends. */
export function pct(p: number): string {
  if (p >= 0.995) return 'over 99%'
  if (p < 0.005) return 'under 1%'
  return `${Math.round(p * 100)}%`
}

/** A rate (a fraction) as a percentage with one decimal: 0.1034 is '10.3%'. */
export const rate = (r: number): string => `${(r * 100).toFixed(1)}%`

/** A difference of rates in percentage points: 0.0214 is '2.1', -0.004 is '-0.4'. */
export const points = (r: number): string => (r * 100).toFixed(1)

/** When the line first crossed one of the two calling lines, and which side it fell on. */
export interface FirstCall {
  readonly visitors: number
  readonly side: 'A' | 'B'
  /** Whether every later look stayed beyond the same line. */
  readonly stayed: boolean
}

/** A promising early reading that the later data took back: where it peaked and where it fell to. */
export interface Dip {
  readonly peakVisitors: number
  readonly peak: number
  readonly lowVisitors: number
  readonly low: number
}

/** The numbers the prose quotes. */
export interface Reading {
  readonly visitors: number
  readonly clicksA: number
  readonly clicksB: number
  /** The raw click rates. */
  readonly rawA: number
  readonly rawB: number
  /** The probability that B's rate is the higher. */
  readonly better: number
  /** The probability of the version that is more likely better (whichever it is), and which. */
  readonly leader: 'A' | 'B'
  readonly leaderChance: number
  /** How firm the call is: beyond the line, leaning, or open. */
  readonly call: 'clear' | 'leaning' | 'open'
  /** Whether the 90% intervals of the two rates overlap. */
  readonly overlap: boolean
  /** Whether the 90% interval of the difference excludes zero. */
  readonly excludesZero: boolean
  /** The lowest and highest probability that B is better in the first 100 visitors of each version. */
  readonly earlyLow: number
  readonly earlyHigh: number
  readonly firstCall: FirstCall | undefined
  /** The first time the leader looked at least 80% likely, if it later fell back by at least ten points. */
  readonly dip: Dip | undefined
  /** Whether B's rate really is the higher one (which nothing in the inference saw). */
  readonly bReallyHigher: boolean
  /** Whether the true difference lies in the 90% interval of the difference. */
  readonly truthInside: boolean
  /** The true difference of the rates (B minus A). */
  readonly trueGap: number
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readButtons(result: Result): Reading {
  const { tally, strip, difference, better, truth } = result
  const early = strip.better.filter((_, i) => (strip.visitors[i] as number) <= 100)
  const leader = better >= 0.5 ? 'B' : 'A'
  const leaderChance = Math.max(better, 1 - better)
  const side = (p: number): 'A' | 'B' | undefined => (p >= CALL ? 'B' : p <= 1 - CALL ? 'A' : undefined)
  const at = strip.better.findIndex((p) => side(p) !== undefined)
  const first = at < 0 ? undefined : side(strip.better[at] as number)
  const trueGap = truth.rateB - truth.rateA
  // the chance that the eventual leader is the better button, at each look
  const lead = strip.better.map((p) => (leader === 'B' ? p : 1 - p))
  const peakAt = lead.findIndex((p) => p >= 0.8)
  const later = peakAt < 0 ? [] : lead.slice(peakAt + 1)
  const low = later.length === 0 ? 1 : Math.min(...later)
  const dip: Dip | undefined =
    peakAt >= 0 && (lead[peakAt] as number) - low >= 0.1
      ? {
          peakVisitors: strip.visitors[peakAt] as number,
          peak: lead[peakAt] as number,
          lowVisitors: strip.visitors[peakAt + 1 + later.indexOf(low)] as number,
          low,
        }
      : undefined
  return {
    visitors: tally.visitors,
    clicksA: tally.clicksA,
    clicksB: tally.clicksB,
    rawA: tally.clicksA / tally.visitors,
    rawB: tally.clicksB / tally.visitors,
    better,
    leader,
    leaderChance,
    call: leaderChance >= CALL ? 'clear' : leaderChance >= 0.8 ? 'leaning' : 'open',
    overlap: result.rateA.hi >= result.rateB.lo && result.rateB.hi >= result.rateA.lo,
    excludesZero: difference.lo > 0 || difference.hi < 0,
    earlyLow: Math.min(...early),
    earlyHigh: Math.max(...early),
    firstCall:
      first === undefined
        ? undefined
        : {
            visitors: strip.visitors[at] as number,
            side: first,
            stayed: strip.better.slice(at).every((p) => side(p) === first),
          },
    dip,
    bReallyHigher: trueGap > 0,
    truthInside: difference.lo <= trueGap && trueGap <= difference.hi,
    trueGap,
  }
}
