/**
 * What the prose of 'Gaussian mixtures' says: a few numbers read off the data (before anything runs) and off the result (how many
 * groups hold real weight, how the biggest groups sit, how many eruptions are unclear, how much the eruptions inside a group lean),
 * so the text always describes the picture beside it, whatever the choice or the seed. Pure functions, light enough for the
 * page's bundle.
 */

import { CLEAR, type GroupFit, REAL, type Result } from './compute.ts'

/** The correlation of two columns of numbers (0 when either has no spread). */
export function correlation(a: readonly number[], b: readonly number[]): number {
  const n = a.length
  if (n < 3) return 0
  const [ma, mb] = [a.reduce((s, v) => s + v, 0) / n, b.reduce((s, v) => s + v, 0) / n]
  const cov = a.reduce((s, v, i) => s + (v - ma) * ((b[i] as number) - mb), 0)
  const va = a.reduce((s, v) => s + (v - ma) ** 2, 0)
  const vb = b.reduce((s, v) => s + (v - mb) ** 2, 0)
  return va === 0 || vb === 0 ? 0 : cov / Math.sqrt(va * vb)
}

/** What the page says about one of the groups that hold real weight. */
export interface GroupReading {
  /** The index of the group (0 is the heaviest). */
  readonly index: number
  readonly fit: GroupFit
  /** How many eruptions are most likely in it. */
  readonly count: number
  /** The correlation of the length and the wait of those eruptions (how much they lean). */
  readonly lean: number
}

/** The numbers the prose quotes after the run. */
export interface Reading {
  readonly rows: number
  /** The groups that hold at least {@link REAL} of the eruptions, heaviest first. */
  readonly real: readonly GroupReading[]
  /** The pieces of the model that hold less, and the share of the eruptions they hold together. */
  readonly small: number
  readonly smallShare: number
  /** How many eruptions are not clearly in one group (the grey dots). */
  readonly unclear: number
  /** The weight of the heaviest group that holds no real weight, when the model was asked for a fixed number of groups. */
  readonly lightest: number
}

/**
 * Read the result against the data.
 *
 * @param result - The result.
 * @param length - The eruption lengths, in the order of the data.
 * @param wait - The waits, in the order of the data.
 * @returns The reading.
 */
export function readResult(result: Result, length: readonly number[], wait: readonly number[]): Reading {
  const real = result.groups.flatMap((fit, index): GroupReading[] => {
    if (fit.weight < REAL) return []
    const members = result.group.flatMap((g, i) => (g === index ? [i] : []))
    const lean = correlation(
      members.map((i) => length[i] as number),
      members.map((i) => wait[i] as number),
    )
    return [{ index, fit, count: members.length, lean }]
  })
  const rest = result.groups.filter((g) => g.weight < REAL)
  return {
    rows: result.group.length,
    real,
    small: rest.length,
    smallShare: rest.reduce((s, g) => s + g.weight, 0),
    unclear: result.sure.filter((s) => s < CLEAR).length,
    lightest: rest.length === 0 ? 0 : Math.max(...rest.map((g) => g.weight)),
  }
}

/** The numbers about the data alone. */
export interface DataReading {
  readonly rows: number
  /** The correlation of the length and the wait over all the eruptions. */
  readonly together: number
}

/**
 * The numbers the prose quotes before anything has run.
 *
 * @param length - The eruption lengths.
 * @param wait - The waits.
 * @returns The reading.
 */
export const readData = (length: readonly number[], wait: readonly number[]): DataReading => ({
  rows: length.length,
  together: correlation(length, wait),
})
