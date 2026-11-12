/**
 * What the prose of 'When did the rate change?' says: a few numbers read off the real counts (their total and how the bars
 * fall) and off the result (the most likely year of the switch, the shortest run of years that holds nine tenths of the chance,
 * the two rates and how much smaller one is than the other, and how the same window compares for the made-up series), so the
 * text always describes the picture beside it. Pure functions, light enough for the page's bundle.
 */

import type { Detection, Result } from './compute.ts'
import { FIRST_YEAR, yearOf } from './data.ts'

/** The share of the chance a window of years has to hold to be reported. */
export const MASS = 0.9

/** A run of consecutive years and the chance it holds. */
export interface Window {
  readonly from: number
  readonly to: number
  /** How many years it spans. */
  readonly years: number
  readonly mass: number
}

/**
 * The shortest run of consecutive years that holds at least `mass` of the chance.
 *
 * @param when - The chance of the switch in each year.
 * @param mass - The share of the chance to hold.
 * @returns The window, as indices into `when`.
 */
export function shortestWindow(when: readonly number[], mass: number = MASS): Window {
  let best: Window = { from: 0, to: when.length - 1, years: when.length, mass: 1 }
  let to = 0
  let held = 0
  when.forEach((_, from) => {
    while (to < when.length && held < mass) held += when[to++] as number
    if (held >= mass && to - from < best.years) best = { from, to: to - 1, years: to - from, mass: held }
    held -= when[from] as number
  })
  return best
}

/** The numbers about the real counts alone. */
export interface DataReading {
  readonly years: number
  readonly total: number
  readonly firstYear: number
  readonly lastYear: number
}

/**
 * The numbers the prose quotes before anything has run.
 *
 * @param counts - The count in each year.
 * @returns The reading.
 */
export function readData(counts: readonly number[]): DataReading {
  return {
    years: counts.length,
    total: counts.reduce((a, b) => a + b, 0),
    firstYear: FIRST_YEAR,
    lastYear: yearOf(counts.length - 1),
  }
}

/** The numbers the prose quotes about one series. */
export interface DetectionReading {
  /** The most likely year of the switch and its chance. */
  readonly likeliest: number
  readonly likeliestChance: number
  /** The shortest run of years holding nine tenths of the chance, as calendar years where the series is the real one. */
  readonly window: Window
  /** How many times larger the old rate is than the new one. */
  readonly drop: number
}

/** Read one detection. */
export function readDetection(d: Detection): DetectionReading {
  const top = d.when.reduce((best, p, i) => (p > (d.when[best] as number) ? i : best), 0)
  return {
    likeliest: top,
    likeliestChance: d.when[top] as number,
    window: shortestWindow(d.when),
    drop: d.before.mean / d.after.mean,
  }
}

/** The numbers the prose quotes after the run. */
export interface Reading {
  readonly coal: DetectionReading
  readonly abrupt: DetectionReading
  readonly gentle: DetectionReading
  /** Whether the abrupt series' window is shorter than the gentle series' (it nearly always is; the prose reads either way). */
  readonly abruptSharper: boolean
  /** Whether the real change index falls inside the 90% window of each made-up series. */
  readonly abruptCovers: boolean
  readonly gentleCovers: boolean
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readResult(result: Result): Reading {
  const abrupt = readDetection(result.abrupt)
  const gentle = readDetection(result.gentle)
  const covers = (r: DetectionReading): boolean => r.window.from <= result.truth && result.truth <= r.window.to
  return {
    coal: readDetection(result.coal),
    abrupt,
    gentle,
    abruptSharper: abrupt.window.years < gentle.window.years,
    abruptCovers: covers(abrupt),
    gentleCovers: covers(gentle),
  }
}
