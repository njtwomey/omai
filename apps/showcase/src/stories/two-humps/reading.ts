/**
 * What the prose of 'Two humps' says: a few numbers read off the data (how many waits fall where the average sits) and off the
 * result (how the eruptions split between the two groups, where the centres sit, when the notes stopped changing anything), so the
 * text always describes the picture beside it, whatever the order of the eruptions was. Pure functions, light enough for the
 * page's bundle.
 */

import type { Result } from './compute.ts'

/** How near the average a wait counts as 'at the average', in minutes. */
export const NEAR = 3

/** The chance above which an eruption is called certain of its group (and below one minus which it is certain of the other). */
export const SURE = 0.9

/** The numbers about the data alone. */
export interface DataReading {
  readonly rows: number
  readonly average: number
  /** How many waits are within {@link NEAR} minutes of the average. */
  readonly nearAverage: number
  readonly shortest: number
  readonly longest: number
}

/**
 * The numbers the prose quotes before anything has run.
 *
 * @param waits - The waiting times in minutes.
 * @returns The reading.
 */
export function readData(waits: readonly number[]): DataReading {
  const average = waits.reduce((a, b) => a + b, 0) / waits.length
  return {
    rows: waits.length,
    average,
    nearAverage: waits.filter((w) => Math.abs(w - average) <= NEAR).length,
    shortest: Math.min(...waits),
    longest: Math.max(...waits),
  }
}

/** The numbers the prose quotes after the run. */
export interface Reading {
  /** How many eruptions are surely in the lower group, surely in the upper, and in between. */
  readonly lowerSure: number
  readonly upperSure: number
  readonly unsure: number
  /** The shortest and the longest wait among the unsure ones (undefined when there are none). */
  readonly unsureFrom: number | undefined
  readonly unsureTo: number | undefined
  /** The distance between the two centres, in minutes. */
  readonly gap: number
  /** The centres after the first sweep of notes, in minutes (undefined if the first sweep was not kept). */
  readonly afterFirst: readonly [number, number] | undefined
  /** The first kept sweep after which both centres are within a twentieth of a minute of where they end. */
  readonly settledBy: number | undefined
  /** The largest move of a centre over the last kept sweeps' comparison, in minutes (how far the first sweep still was from the end). */
  readonly firstSweepError: number | undefined
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @param waits - The waiting times it was fitted to, in the order of the data.
 * @returns The reading.
 */
export function readResult(result: Result, waits: readonly number[]): Reading {
  const used = waits.slice(0, result.chance.length)
  const upperSure = result.chance.filter((c) => c >= SURE).length
  const lowerSure = result.chance.filter((c) => c <= 1 - SURE).length
  const unsure = used.filter((_, i) => (result.chance[i] as number) > 1 - SURE && (result.chance[i] as number) < SURE)
  const near = (i: number): boolean =>
    Math.abs((result.trace.lower[i] as number) - result.means[0]) < 0.05 &&
    Math.abs((result.trace.upper[i] as number) - result.means[1]) < 0.05
  const settled = result.trace.sweep.findIndex((_, i) => near(i))
  const first = result.trace.sweep.indexOf(1)
  return {
    lowerSure,
    upperSure,
    unsure: unsure.length,
    unsureFrom: unsure.length === 0 ? undefined : Math.min(...unsure),
    unsureTo: unsure.length === 0 ? undefined : Math.max(...unsure),
    gap: result.means[1] - result.means[0],
    afterFirst: first < 0 ? undefined : [result.trace.lower[first] as number, result.trace.upper[first] as number],
    settledBy: settled < 0 ? undefined : (result.trace.sweep[settled] as number),
    firstSweepError:
      first < 0
        ? undefined
        : Math.max(
            Math.abs((result.trace.lower[first] as number) - result.means[0]),
            Math.abs((result.trace.upper[first] as number) - result.means[1]),
          ),
  }
}
